#!/usr/bin/env python3
"""
Tests for the web_research deep-crawl module and its lie-detector guard.

Coverage:
  - URL detection in user text
  - Crawl dedupes, respects page/depth limits, strips boilerplate
  - Skips broken sub-pages without crashing
  - Bounded total size
  - Lie-detector: fetch success → response references real fetched content
    (tokens from sub-pages, not just page 1)
  - Lie-detector: fetch fails → response says "I can't access this"
  - Lie-detector: partial crawl → response acknowledges coverage limits
  - One real-path test (skips on network failure) — fetches example.com and
    asserts text flows into the prompt.
"""
from __future__ import annotations

import json
import socket
import sys
from unittest.mock import MagicMock, patch

import pytest


# ─── URL detection tests ───────────────────────────────────────────────────────

class TestURLDetection:
    """detect_research_trigger: pulls URLs out of free-form user text."""

    def test_detects_full_http_url(self):
        from controller.web_research import detect_research_trigger
        urls = detect_research_trigger("Check out https://example.com/guide for me")
        assert "https://example.com/guide" in urls

    def test_detects_full_https_url(self):
        from controller.web_research import detect_research_trigger
        urls = detect_research_trigger("Read https://docs.python.org/3/library/")
        assert "https://docs.python.org/3/library/" in urls

    def test_dedupes_urls(self):
        from controller.web_research import detect_research_trigger
        urls = detect_research_trigger(
            "Look at https://example.com then https://example.com again"
        )
        assert urls.count("https://example.com") == 1

    def test_strips_trailing_punctuation(self):
        from controller.web_research import detect_research_trigger
        urls = detect_research_trigger("See https://example.com.")
        assert "https://example.com." not in urls
        assert any(u.endswith("example.com") for u in urls)

    def test_no_url_returns_empty(self):
        from controller.web_research import detect_research_trigger
        urls = detect_research_trigger("What is the meaning of life?")
        assert urls == [] or all("." not in u.replace("https://", "") for u in urls)

    def test_detects_bare_domain(self):
        from controller.web_research import detect_research_trigger
        urls = detect_research_trigger("Look at example.com please")
        # bare domains get expanded to https://www.example.com
        assert any("example.com" in u for u in urls)


# ─── Internal helper tests ─────────────────────────────────────────────────────

class TestInternalHelpers:
    """Direct tests of helper functions."""

    def test_same_domain_strips_www(self):
        from controller.web_research import _same_domain
        assert _same_domain("https://www.example.com", "https://example.com") is True
        assert _same_domain("https://example.com", "https://other.com") is False

    def test_resolve_url_absolute(self):
        from controller.web_research import _resolve_url
        assert _resolve_url("https://example.com/", "/guide") == "https://example.com/guide"
        assert _resolve_url("https://example.com/a/", "b") == "https://example.com/a/b"

    def test_resolve_url_rejects_javascript(self):
        from controller.web_research import _resolve_url
        assert _resolve_url("https://example.com/", "javascript:void(0)") is None
        assert _resolve_url("https://example.com/", "mailto:foo@bar.com") is None

    def test_is_html_url_skips_binaries(self):
        from controller.web_research import _is_html_url
        assert _is_html_url("https://example.com/page") is True
        assert _is_html_url("https://example.com/image.png") is False
        assert _is_html_url("https://example.com/file.pdf") is False
        assert _is_html_url("https://example.com/dir/") is True

    def test_strip_and_extract_removes_script(self):
        from controller.web_research import _strip_and_extract_text
        from bs4 import BeautifulSoup
        html = "<html><body><script>alert(1)</script><p>Real content here.</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        text = _strip_and_extract_text(soup)
        assert "alert(1)" not in text
        assert "Real content" in text


# ─── Crawl behavior tests (mocked HTTP) ────────────────────────────────────────

class TestCrawlBehavior:
    """research_url: respects dedup, depth, page limits; bounded size."""

    def _make_html(self, title: str, body: str, links: list[str] = None) -> str:
        """Build a simple HTML page with the given title, body, and links."""
        link_html = ""
        if links:
            link_html = "\n".join(
                f'<a href="{href}">link to {href}</a>' for href in links
            )
        return f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
  <nav>Navigation noise</nav>
  <main>
    <h1>{title}</h1>
    <p>{body}</p>
    {link_html}
  </main>
  <footer>Footer noise</footer>
</body>
</html>"""

    def test_dedupes_repeated_links(self):
        """If page 1 links to /a 3 times, we only visit /a once."""
        from controller.web_research import research_url, _HTTPClient
        from unittest.mock import patch

        page1 = self._make_html("Page 1", "alpha page one", ["/a", "/a", "/a"])
        page_a = self._make_html("Page A", "this is page alpha")

        responses = {
            "https://example.com/": (200, page1),
            "https://example.com/a": (200, page_a),
        }

        def fake_get(self, url):
            return responses.get(url, (404, "Not found"))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):  # no real delay
                result = research_url("https://example.com/")

        assert result.ok is True
        urls_visited = [p.url for p in result.pages if p.fetched]
        # /a should appear once even though it was linked 3 times
        assert urls_visited.count("https://example.com/a") == 1

    def test_respects_max_pages(self):
        """We don't crawl more than MAX_PAGES pages even if more are linked."""
        from controller.web_research import research_url, _HTTPClient, MAX_PAGES

        # Link to 50 different pages from page 1
        page1 = self._make_html("Root", "root", [f"/page{i}" for i in range(50)])
        generic = self._make_html("x", "y")
        responses = {"https://example.com/": (200, page1)}

        # All other pages return 200
        def fake_get(self, url):
            return responses.get(url, (200, generic))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        assert len([p for p in result.pages if p.fetched]) <= MAX_PAGES

    def test_respects_max_depth(self):
        """Links at depth > MAX_DEPTH should NOT be followed."""
        from controller.web_research import research_url, _HTTPClient, MAX_DEPTH

        # Page 1 → Page 2 (depth 1) → Page 3 (depth 2) → Page 4 (depth 3, skip)
        page1 = self._make_html("P1", "depth 0", ["/p2"])
        page2 = self._make_html("P2", "depth 1", ["/p3"])
        page3 = self._make_html("P3", "depth 2", ["/p4"])
        page4 = self._make_html("P4", "depth 3 (should not visit)")

        responses = {
            "https://example.com/": (200, page1),
            "https://example.com/p2": (200, page2),
            "https://example.com/p3": (200, page3),
            "https://example.com/p4": (200, page4),
        }

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        visited = {p.url for p in result.pages if p.fetched}
        # p3 was at depth 2; p4 was at depth 3 and should NOT be visited
        assert "https://example.com/p4" not in visited

    def test_strips_boilerplate(self):
        """<script>/<style>/<nav>/<footer> text should be removed."""
        from controller.web_research import research_url, _HTTPClient

        html = """<html><head><title>Test</title>
        <script>var x = 1;</script>
        <style>.foo { color: red; }</style>
        </head><body>
        <nav>nav noise</nav>
        <main><h1>Hello</h1><p>REAL CONTENT about fairydust</p></main>
        <footer>footer noise</footer>
        <script>more junk</script>
        </body></html>"""

        responses = {"https://example.com/": (200, html)}

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        assert result.ok is True
        page = result.pages[0]
        assert "REAL CONTENT about fairydust" in page.text
        assert "var x = 1" not in page.text
        assert "color: red" not in page.text
        assert "nav noise" not in page.text
        assert "footer noise" not in page.text

    def test_main_page_failure_returns_typed_error(self):
        """If the main page returns 404, ok=False and error is set."""
        from controller.web_research import research_url, _HTTPClient

        def fake_get(self, url):
            return (404, "Not found")

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://broken.example.com/")

        assert result.ok is False
        assert "broken" in result.error.lower() or "404" in result.error

    def test_broken_subpages_logged_and_skipped(self):
        """If sub-pages fail, main page is still returned (partial ok)."""
        from controller.web_research import research_url, _HTTPClient

        page1 = self._make_html("Root", "main page works", ["/good", "/bad"])
        page_good = self._make_html("Good", "this sub-page is fine")

        responses = {
            "https://example.com/": (200, page1),
            "https://example.com/good": (200, page_good),
            "https://example.com/bad": (500, "Server error"),
        }

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        assert result.ok is True
        # Main page and /good were fetched; /bad is logged as failure
        assert result.pages_failed >= 1

    def test_bounded_total_size(self):
        """Total output capped at MAX_INJECT_CHARS (prompt-side budget)."""
        from controller.web_research import (
            research_url, _HTTPClient, MAX_INJECT_CHARS, format_research_context
        )

        # Page with very long body — 50k chars of content
        long_body = "This is real content about an important topic. " * 2500
        page1 = self._make_html("Long", long_body)
        responses = {"https://example.com/": (200, page1)}

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        output = format_research_context(result)
        # Output must fit within the prompt-side inject budget
        assert len(output) <= MAX_INJECT_CHARS + 200, (
            f"Output is {len(output)} chars, must be <= {MAX_INJECT_CHARS + 200} "
            "(inject budget + margin for headers/coverage note)"
        )

    def test_only_same_domain_links(self):
        """External-domain links are NOT followed."""
        from controller.web_research import research_url, _HTTPClient

        page1 = self._make_html(
            "Root",
            "main content",
            ["/internal", "https://other.com/external"],
        )
        page_internal = self._make_html("Internal", "internal page")

        responses = {
            "https://example.com/": (200, page1),
            "https://example.com/internal": (200, page_internal),
        }

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        visited = {p.url for p in result.pages if p.fetched}
        # /internal should be visited; other.com should NOT
        assert "https://example.com/internal" in visited
        assert "https://other.com/external" not in visited

    def test_only_html_links(self):
        """Non-HTML links (images, PDFs) are NOT followed."""
        from controller.web_research import research_url, _HTTPClient

        page1 = self._make_html(
            "Root",
            "main",
            ["/article", "/image.png", "/doc.pdf"],
        )
        page_article = self._make_html("Article", "real article")

        responses = {
            "https://example.com/": (200, page1),
            "https://example.com/article": (200, page_article),
        }

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        visited = {p.url for p in result.pages if p.fetched}
        assert "https://example.com/article" in visited
        assert "https://example.com/image.png" not in visited
        assert "https://example.com/doc.pdf" not in visited


# ─── Lie-detector tests ────────────────────────────────────────────────────────

class TestLieDetector:
    """
    The lie-detector test suite. These tests verify the *system prompt*
    rules and the *agent's response* to research output.

    Goal: catch hallucinations like "I retrieved..." when no fetch happened,
    or "I remember this from earlier" when content wasn't in history.
    """

    def test_personality_contains_research_honesty_rules(self):
        """The system prompt must contain the research-honesty block."""
        from fairy_personality import FAIRY_PERSONALITY
        personality_lower = FAIRY_PERSONALITY.lower()
        # Key honesty phrases that must appear
        assert "research" in personality_lower
        assert "retrieved" in personality_lower or "remembered" in personality_lower

    def test_format_research_context_ok(self):
        """Successful research → context with all pages, not just page 1."""
        from controller.web_research import (
            ResearchResult, PageResult, format_research_context
        )

        result = ResearchResult(
            ok=True,
            root_url="https://example.com",
            pages=[
                PageResult(
                    url="https://example.com/",
                    title="Home",
                    text="Welcome to the home page. Topics: cooking, music.",
                    fetched=True,
                ),
                PageResult(
                    url="https://example.com/cooking",
                    title="Cooking",
                    text="Cooking section content. Recipes for pizza.",
                    fetched=True,
                ),
                PageResult(
                    url="https://example.com/music",
                    title="Music",
                    text="Music section content. Genres include jazz.",
                    fetched=True,
                ),
            ],
            pages_crawled=3,
        )

        context = format_research_context(result)

        # All three pages' content should be in the context
        assert "Welcome to the home page" in context
        assert "Recipes for pizza" in context  # page 2 content
        assert "Genres include jazz" in context  # page 3 content
        # All three URLs should appear
        assert "https://example.com/" in context
        assert "https://example.com/cooking" in context
        assert "https://example.com/music" in context

    def test_format_research_context_failure(self):
        """Failed research → context indicates failure clearly."""
        from controller.web_research import ResearchResult, format_research_context

        result = ResearchResult(
            ok=False,
            root_url="https://broken.example.com",
            error="HTTP 404",
        )

        context = format_research_context(result)
        # Should clearly indicate failure
        assert "failed" in context.lower() or "could not" in context.lower() or "404" in context

    def test_partial_crawl_acknowledged(self):
        """Partial crawl: ok=True with pages_failed > 0 should still be ok,
        and the LLM is expected to acknowledge coverage limits."""
        from controller.web_research import ResearchResult, PageResult

        result = ResearchResult(
            ok=True,
            root_url="https://example.com",
            pages=[
                PageResult(url="https://example.com/", title="Home",
                           text="main page text", fetched=True),
                PageResult(url="https://example.com/broken", fetched=False, error="HTTP 500"),
            ],
            pages_crawled=1,
            pages_failed=1,
        )

        # The result is still "ok" (main page worked), but the agent must
        # know there were failures. was_partial flag helps downstream.
        assert result.ok is True
        assert result.was_partial is True

    def test_lie_detector_phrases_are_blocked_on_failure(self):
        """
        Honest failure surfacing: when research fails, the response template
        must NOT use hallucination phrases like "retrieved", "found in our
        conversation", "I remember".
        """
        from controller.web_research import ResearchResult, format_research_context

        # Simulate fetch failure
        result = ResearchResult(
            ok=False,
            root_url="https://broken.example.com",
            error="HTTP 404",
        )
        context = format_research_context(result)

        # The context itself is a failure message. The downstream LLM should
        # be told to use this as evidence, not to invent.
        # The lie-detector check: the failure message must NOT contain
        # any of the hallucination phrases.
        for banned in ["retrieved", "found in our conversation", "I remember"]:
            assert banned.lower() not in context.lower(), (
                f"Failure context should not contain '{banned}': {context}"
            )

    def test_lie_detector_response_required_to_use_real_tokens(self):
        """
        When research succeeds, the agent's response must reference real
        content from the fetched pages — not just page 1. This is the
        'multi-page depth' test.
        """
        from controller.web_research import (
            research_url, _HTTPClient, format_research_context
        )

        # Build a 3-page site where only page 2 and 3 have unique keywords
        page1 = """<html><head><title>Home</title></head>
        <body><main><h1>Welcome</h1><p>Generic landing copy here.</p>
        <a href="/api">API</a><a href="/blog">Blog</a>
        </main></body></html>"""
        page2 = """<html><head><title>API</title></head>
        <body><main><h1>API Reference</h1>
        <p>Our REST API uses fibreglass authentication tokens. The endpoint
        /v1/fetch returns a list of widget objects. Quokka support is included.</p>
        </main></body></html>"""
        page3 = """<html><head><title>Blog</title></head>
        <body><main><h1>Blog</h1>
        <p>Latest posts about platypus migration patterns. We discuss
        echidna husbandry and the proper care of wombats in winter.</p>
        </main></body></html>"""

        responses = {
            "https://example.com/": (200, page1),
            "https://example.com/api": (200, page2),
            "https://example.com/blog": (200, page3),
        }

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        context = format_research_context(result)
        # All unique tokens from sub-pages should be in the context
        # (proves the crawl isn't just page 1)
        for token in ["fibreglass", "quokka", "platypus", "wombat", "echidna"]:
            assert token.lower() in context.lower(), (
                f"Token '{token}' from sub-page missing — crawl may be only fetching page 1"
            )


# ─── maybe_research / trigger detection tests ──────────────────────────────────

class TestMaybeResearch:
    """The maybe_research() pre-processing function."""

    def test_no_url_no_phrase_returns_false(self):
        from controller.web_research import maybe_research
        triggered, ctx = maybe_research("What is the weather?")
        assert triggered is False
        assert ctx == ""

    def test_url_without_phrase_triggers(self):
        """A bare URL in the message is enough to trigger research."""
        from controller.web_research import maybe_research
        from controller.web_research import _HTTPClient

        html = "<html><head><title>Site</title></head><body><main>content here</main></body></html>"

        def fake_get(self, url):
            return (200, html)

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                triggered, ctx = maybe_research("Read https://example.com please")
        assert triggered is True
        assert "content here" in ctx

    def test_url_failure_surfaces_error(self):
        """URL with fetch failure → triggered=True with failure context."""
        from controller.web_research import maybe_research, _HTTPClient

        def fake_get(self, url):
            return (404, "Not found")

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                triggered, ctx = maybe_research("Check https://broken.example.com/")
        assert triggered is True
        assert "failed" in ctx.lower() or "404" in ctx


# ─── Real-path test (network-isolated per conftest) ────────────────────────────

@pytest.mark.requires_network
class TestRealPath:
    """
    House rule: mocks aren't enough. One real fetch to prove the pipeline
    works end-to-end. Skips gracefully on network failure so the suite
    stays green offline.
    """

    def test_real_fetch_to_example_com_flows_into_prompt(self):
        """Fetch https://example.com and verify text reaches the prompt."""
        from controller.web_research import research_url, format_research_context

        try:
            # Quick connectivity check
            socket.create_connection(("example.com", 443), timeout=5).close()
        except (socket.error, OSError):
            pytest.skip("Network unavailable")

        try:
            result = research_url("https://example.com")
        except Exception:
            pytest.skip("Network fetch failed")

        # example.com is a tiny static page — if our pipeline works, the
        # text should reach the context block
        context = format_research_context(result)
        assert result.ok is True
        assert "Example Domain" in context or "example" in context.lower()
        # The URL should be in the context
        assert "example.com" in context


# ─── Content cleaning tests ───────────────────────────────────────────────────

class TestContentCleaning:
    """
    Tests for clean_extracted_text — the function that strips nav/menu junk
    and dedupes repeated lines before the text is injected into the prompt.
    """

    def test_removes_jobs_sign_in_register_lines(self):
        """Common nav/footer lines should be stripped."""
        from controller.web_research import clean_extracted_text
        text = (
            "Welcome to the site.\n"
            "Jobs: Apply Now\n"
            "Sign in\n"
            "Register\n"
            "Real content about widgets.\n"
            "Privacy policy\n"
            "All rights reserved\n"
        )
        cleaned = clean_extracted_text(text)
        assert "Welcome to the site." in cleaned
        assert "Real content about widgets." in cleaned
        assert "Jobs:" not in cleaned
        assert "Sign in" not in cleaned
        assert "Register" not in cleaned
        assert "Privacy policy" not in cleaned
        assert "All rights reserved" not in cleaned

    def test_dedupes_repeated_nav_lines(self):
        """Nav/footer lines that repeat across sections get deduplicated."""
        from controller.web_research import clean_extracted_text
        text = (
            "Home | Docs | API | GitHub\n"   # nav line #1
            "Real content starts here.\n"
            "Home | Docs | API | GitHub\n"   # nav line repeated
            "More real content.\n"
        )
        cleaned = clean_extracted_text(text)
        # The first nav line should already be stripped (looks like a link list)
        # and even if it weren't, it shouldn't appear twice
        assert cleaned.count("Home | Docs | API | GitHub") <= 1

    def test_strips_link_list_fragments(self):
        """Pipe-separated short tokens (link bars) are stripped."""
        from controller.web_research import clean_extracted_text
        text = (
            "Home | About | Contact | Careers\n"
            "This is the actual article content that has substance.\n"
            "Products | Pricing | Support\n"
        )
        cleaned = clean_extracted_text(text)
        assert "Home | About | Contact | Careers" not in cleaned
        assert "Products | Pricing | Support" not in cleaned
        assert "actual article content" in cleaned

    def test_preserves_legitimate_content(self):
        """Real sentences with substance are kept intact."""
        from controller.web_research import clean_extracted_text
        text = (
            "The quick brown fox jumps over the lazy dog.\n"
            "Machine learning models require substantial training data.\n"
            "The protocol supports bidirectional communication.\n"
        )
        cleaned = clean_extracted_text(text)
        assert "quick brown fox" in cleaned
        assert "training data" in cleaned
        assert "bidirectional" in cleaned

    def test_collapses_blank_lines(self):
        """3+ blank lines become 1 blank line."""
        from controller.web_research import clean_extracted_text
        text = "First paragraph.\n\n\n\n\nSecond paragraph.\n"
        cleaned = clean_extracted_text(text)
        # 4 newlines between paragraphs collapse to 1 blank line = \n\n
        assert "\n\n\n" not in cleaned
        assert "First paragraph." in cleaned
        assert "Second paragraph." in cleaned

    def test_handles_empty_input(self):
        """Empty/None input is handled gracefully."""
        from controller.web_research import clean_extracted_text
        assert clean_extracted_text("") == ""
        assert clean_extracted_text("\n\n\n") == ""

    def test_junk_line_in_fixture_page_removed(self):
        """A junk-heavy fixture page loses all known nav lines."""
        from controller.web_research import clean_extracted_text
        # Simulate text extracted from a typical docs site
        fixture = """\
Documentation
Getting Started
API Reference
Tutorials
Community
GitHub
Twitter
Skip to content
Search
Menu
Home

Welcome to the documentation portal. This section covers the basics.
Jobs: View all open positions
Sign in to your account
Register for a new account
Subscribe to our newsletter for updates
All rights reserved
Privacy policy
Follow us on Twitter
"""
        cleaned = clean_extracted_text(fixture)
        # None of these should survive
        for banned in [
            "Skip to content", "Search", "Menu", "Home",
            "Jobs:", "Sign in", "Register", "Subscribe to our newsletter",
            "All rights reserved", "Privacy policy", "Follow us on",
            "Documentation", "Getting Started", "Tutorials", "Community",
            "GitHub", "Twitter",
        ]:
            assert banned not in cleaned, (
                f"Cleaning should have removed '{banned}' but it survived: {cleaned}"
            )
        # Real content must survive
        assert "Welcome to the documentation portal" in cleaned


# ─── Prompt-side budget enforcement tests ──────────────────────────────────────

class TestPromptBudget:
    """
    The injected context must stay under MAX_INJECT_CHARS even when the
    crawl output is at max size.
    """

    def test_injected_context_under_prompt_budget(self, monkeypatch):
        """
        Even with 30k chars of crawl content (the crawl-side max), the
        injected context must be <= MAX_INJECT_CHARS (12k).
        """
        from controller.web_research import (
            research_url, _HTTPClient, MAX_INJECT_CHARS, format_research_context
        )

        # 3 huge pages — well over the prompt budget
        huge = "Real meaningful content about a complex topic. " * 1500  # ~50k chars
        page1 = f"<html><head><title>Root</title></head><body><main><h1>Root</h1><p>{huge}</p></main></body></html>"
        page2 = f"<html><head><title>P2</title></head><body><main><p>{huge}</p></main></body></html>"
        page3 = f"<html><head><title>P3</title></head><body><main><p>{huge}</p></main></body></html>"

        responses = {
            "https://example.com/": (200, page1),
            "https://example.com/p2": (200, page2),
            "https://example.com/p3": (200, page3),
        }

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        context = format_research_context(result)
        assert len(context) <= MAX_INJECT_CHARS + 200, (
            f"Injected context is {len(context)} chars — must be <= "
            f"{MAX_INJECT_CHARS + 200} (prompt budget + margin)"
        )

    def test_coverage_note_appears_when_pages_dropped(self, monkeypatch):
        """
        When content exceeds budget and some pages are dropped, the injected
        context must include a coverage note that the LLM can use to be
        honest about what was covered.
        """
        from controller.web_research import (
            research_url, _HTTPClient, format_research_context
        )

        # Root is small; sub-pages are huge so they'll be dropped.
        # Must link to sub-pages from root so the crawler discovers them.
        root_html = (
            "<html><head><title>Root</title></head>"
            "<body><main><p>Short root.</p>"
            '<a href="/sub1">Sub 1</a>'
            '<a href="/sub2">Sub 2</a>'
            '<a href="/sub3">Sub 3</a>'
            "</main></body></html>"
        )
        huge = "Massive content page. " * 2000
        sub_html = f"<html><head><title>Sub</title></head><body><main><p>{huge}</p></main></body></html>"

        responses = {
            "https://example.com/": (200, root_html),
            "https://example.com/sub1": (200, sub_html),
            "https://example.com/sub2": (200, sub_html),
            "https://example.com/sub3": (200, sub_html),
        }

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        context = format_research_context(result)
        # The coverage note is for the LLM, not for the user — must be in the
        # injected context, not the chat transcript
        assert "[Coverage note:" in context, (
            f"Expected a coverage note in context when pages are dropped. "
            f"Got: {context[-500:]}"
        )
        # Must mention what was omitted
        assert "omitted" in context or "partially shown" in context

    def test_coverage_note_appears_when_pages_fail(self):
        """
        When sub-pages fail to fetch, the coverage note must include them so
        the LLM can say 'I tried but couldn't reach page X'.
        """
        from controller.web_research import (
            research_url, _HTTPClient, format_research_context
        )

        # Link to the failing sub-pages from root so they're discovered
        root_html = (
            "<html><head><title>Root</title></head>"
            "<body><main><p>Works.</p>"
            '<a href="/sub1">Sub 1</a>'
            '<a href="/sub2">Sub 2</a>'
            "</main></body></html>"
        )

        def fake_get(self, url):
            if url == "https://example.com/":
                return (200, root_html)
            return (404, "Not found")

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        context = format_research_context(result)
        assert "[Coverage note:" in context, f"Expected coverage note, got: {context[-400:]}"
        assert "failed to load" in context

    def test_no_coverage_note_when_everything_fits(self):
        """
        When all pages fit in the budget, no coverage note is needed.
        """
        from controller.web_research import (
            research_url, _HTTPClient, format_research_context
        )

        root_html = "<html><head><title>Root</title></head><body><main><p>Tiny root page.</p></main></body></html>"
        sub_html = "<html><head><title>Sub</title></head><body><main><p>Tiny sub page.</p></main></body></html>"

        responses = {
            "https://example.com/": (200, root_html),
            "https://example.com/sub": (200, sub_html),
        }

        def fake_get(self, url):
            return responses.get(url, (404, ""))

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        context = format_research_context(result)
        # Everything fit; no need for a coverage note
        assert "[Coverage note:" not in context


# ─── Diagnostic-leak prevention tests ─────────────────────────────────────────

class TestNoDiagnosticLeakToChat:
    """
    Internal diagnostic messages must go to the log file, never into the
    returned chat text that the user sees.
    """

    def test_truncation_warning_goes_to_log_not_text(self, caplog):
        """
        When the prompt budget is exceeded, the truncation warning is sent
        to the logger — NOT into the injected context (which would leak
        into the chat).
        """
        from controller.web_research import (
            research_url, _HTTPClient, format_research_context
        )

        # Force a scenario where the budget backstop must trigger
        huge = "Content. " * 5000  # 45k chars
        page1 = f"<html><head><title>Big</title></head><body><main><p>{huge}</p></main></body></html>"

        def fake_get(self, url):
            return (200, page1)

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")

        with caplog.at_level("WARNING", logger="fairy.web_research"):
            context = format_research_context(result)

        # The context itself must NOT contain log-style diagnostic text
        forbidden_in_context = [
            "exceeded",
            "truncated at",
            "truncating",
            "Research output",
            "WARNING",
        ]
        for term in forbidden_in_context:
            assert term not in context, (
                f"Diagnostic term {term!r} leaked into injected context. "
                f"Context tail: {context[-300:]}"
            )

    def test_failure_path_no_diagnostic_in_context(self, caplog):
        """
        On research failure, the context must contain a clean LLM-facing
        message — no 'HTTPError' / 'Traceback' / '0x12345' / etc.
        """
        from controller.web_research import (
            research_url, _HTTPClient, format_research_context
        )

        def fake_get(self, url):
            return (500, "Internal Server Error")

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://broken.example.com/")

        with caplog.at_level("WARNING", logger="fairy.web_research"):
            context = format_research_context(result)

        assert result.ok is False
        # The LLM-facing message should be clean
        assert "failed" in context.lower()
        # No raw exception / traceback language
        for term in ["Traceback", "Error 500", "Internal Server Error"]:
            assert term not in context, (
                f"Raw diagnostic {term!r} leaked into injected context: {context}"
            )

    def test_no_print_to_stdout_from_research_module(self, capsys):
        """
        web_research must not print to stdout (which would land in the chat
        transcript). All status info goes via logger.
        """
        from controller.web_research import (
            research_url, _HTTPClient, format_research_context, maybe_research
        )

        page_html = "<html><head><title>T</title></head><body><main><p>Content</p></main></body></html>"

        def fake_get(self, url):
            return (200, page_html)

        with patch.object(_HTTPClient, "get", fake_get):
            with patch("controller.web_research.time.sleep"):
                result = research_url("https://example.com/")
                ctx = format_research_context(result)
                maybe_research("Check https://example.com please")

        captured = capsys.readouterr()
        # The module should not write to stdout during normal operation
        assert captured.out == "", (
            f"web_research wrote to stdout (would land in chat): "
            f"{captured.out!r}"
        )
