#!/usr/bin/env python3
"""
Deep web research crawler for Fairy.

Fetches the target page, strips boilerplate (script/style/nav/footer),
extracts text. Then discovers and crawls internal links on the same domain
(up to ~8-10 pages, max depth 2, ~0.5s delay between requests).
Deduplicates links. Skips binaries, images, non-HTML.

Per page: title + main text.
Total crawl budget: ~30k chars (crawl-side cap).
Total inject budget: 12k chars (prompt-side cap — what gets injected into LLM).
Content is cleaned (nav junk removed, duplicates collapsed) before injection.
When pages are dropped due to budget, a coverage note is included so the LLM
knows what was omitted.

Failed sub-pages are logged and skipped; if the main page fails, returns
a typed error — never crashes the chat loop.

Usage:
    from controller.web_research import research_url, format_research_context

    result = research_url("https://docs.example.com/guide")
    if result.ok:
        context = format_research_context(result)
        # inject context into LLM messages
    else:
        logger.error("Research failed: %s", result.error)
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

# Use httpx for async-capable sync HTTP (with connection pooling).
# Fall back to requests if httpx is not available.
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    import requests as _requests_placeholder
    _requests_placeholder = None  # type: ignore[assignment]

from bs4 import BeautifulSoup

logger = logging.getLogger("fairy.web_research")

# ─── Configuration ───────────────────────────────────────────────────────────────

MAX_PAGES = 10          # total pages to crawl (including the root)
MAX_DEPTH = 2           # link-following depth
MAX_CHARS_PER_PAGE = 4000  # truncate each page to this many chars
MAX_TOTAL_CHARS = 30000  # hard cap on combined crawl output
# Prompt-side budget: what gets injected into the LLM. Keep well under the
# model's effective context window so there's always room for a full digest.
MAX_INJECT_CHARS = 12000
REQUEST_DELAY = 0.5      # seconds between requests (polite crawling)
REQUEST_TIMEOUT = 15.0  # seconds per HTTP request

# ─── Result types ───────────────────────────────────────────────────────────────

@dataclass
class PageResult:
    """Result from fetching a single page."""
    url: str
    title: str = ""
    text: str = ""
    fetched: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.fetched and bool(self.text)

    def truncate(self, max_chars: int = MAX_CHARS_PER_PAGE) -> "PageResult":
        """Return a copy with text truncated to max_chars."""
        truncated = self.text[:max_chars]
        if len(self.text) > max_chars:
            truncated += f"\n\n[Content truncated at {max_chars} chars]"
        return PageResult(
            url=self.url,
            title=self.title,
            text=truncated,
            fetched=self.fetched,
            error=self.error,
        )


@dataclass
class ResearchResult:
    """Aggregate result from crawling a URL and its internal links."""
    ok: bool = False
    root_url: str = ""
    pages: list[PageResult] = field(default_factory=list)
    error: str = ""          # set only when the main page itself fails
    pages_crawled: int = 0
    pages_failed: int = 0
    covered_urls: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def was_partial(self) -> bool:
        """True if some sub-pages failed or we hit the page/char budget."""
        return self.pages_failed > 0 or self.total_chars >= MAX_TOTAL_CHARS * 0.9


# ─── Core helpers ───────────────────────────────────────────────────────────────

_SKIP_EXTS = frozenset({
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".zip", ".tar",
    ".gz", ".rar", ".7z",
    ".css", ".js", ".json", ".xml", ".rss",
    ".png?raw", ".jpg?raw", ".jpeg?raw",
})


def _is_html_url(url: str) -> bool:
    """Return True if the URL likely points to HTML content."""
    parsed = urllib.parse.urlparse(url.lower())
    path = parsed.path

    # No path → assume HTML
    if not path or path == "/":
        return True

    # Skip URLs with binary-like extensions
    for ext in _SKIP_EXTS:
        if path.endswith(ext) or f"{ext}?" in path:
            return False

    # Skip paths with query/fragment that look like file downloads
    # but allow query strings on normal paths
    if "." in path.split("/")[-1] and any(path.endswith(ext) for ext in {".html", ".htm", ".php", ".asp", ".aspx"}):
        return True

    # Default: assume HTML
    return True


def _same_domain(url1: str, url2: str) -> bool:
    """Return True if both URLs share the same domain (ignoring www prefix)."""
    try:
        p1 = urllib.parse.urlparse(url1)
        p2 = urllib.parse.urlparse(url2)
        d1 = p1.netloc.lstrip("www.")
        d2 = p2.netloc.lstrip("www.")
        return d1 == d2 and bool(d1)
    except Exception:
        return False


def _resolve_url(base: str, href: str) -> Optional[str]:
    """Resolve a href relative to base, returning an absolute URL or None."""
    try:
        # Remove fragments
        clean_href = href.split("#")[0].split("?")[0]
        if not clean_href:
            return None
        # Skip javascript: and mailto:
        if clean_href.startswith(("javascript:", "mailto:", "tel:", "file:")):
            return None
        # Skip data: URIs
        if clean_href.startswith("data:"):
            return None
        resolved = urllib.parse.urljoin(base, clean_href)
        parsed = urllib.parse.urlparse(resolved)
        # Only allow http/https
        if parsed.scheme not in ("http", "https"):
            return None
        return resolved
    except Exception:
        return None


def _extract_links(soup: BeautifulSoup, base_url: str, root_url: str) -> list[str]:
    """Extract same-domain HTML links from a parsed page."""
    links = []
    seen: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "")
        absolute = _resolve_url(base_url, href)
        if not absolute:
            continue
        if not _same_domain(absolute, root_url):
            continue
        if not _is_html_url(absolute):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)

    return links


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract the best title from a parsed page."""
    # Try <title> first
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)

    # Try og:title
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()

    # Try <h1>
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    return ""


def _strip_and_extract_text(soup: BeautifulSoup) -> str:
    """Remove boilerplate and return clean text."""
    # Kill noise tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "iframe", "svg", "form", "button"]):
        tag.decompose()

    # Try to find main content
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find("div", class_=lambda c: c and ("content" in c or "post" in c or "entry" in c)) or
        soup.find("div", id=lambda i: i and ("content" in i or "main" in i or "post" in i))
    )

    text_source = main if main else soup.body if soup.body else soup

    # Extract text
    text = text_source.get_text(separator="\n", strip=True)

    # Clean whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


# ─── HTTP client (httpx with requests fallback) ─────────────────────────────────

class _HTTPClient:
    """HTTP client wrapper — uses httpx when available, falls back to requests."""

    def __init__(self):
        self._client: Optional[object] = None
        self._use_httpx = _HTTPX_AVAILABLE

    def __enter__(self):
        if self._use_httpx:
            self._client = httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True)
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.close()  # type: ignore[union-attr]
            self._client = None

    def get(self, url: str) -> tuple[int, str]:
        """Fetch a URL. Returns (status_code, raw_html)."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        if self._use_httpx and self._client:
            try:
                resp = self._client.get(url, headers=headers)  # type: ignore[union-attr]
                return resp.status_code, resp.text
            except httpx.HTTPError as e:
                logger.warning("httpx error fetching %s: %s", url, e)
                return 0, ""

        # Fallback to requests
        try:
            import requests
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            return resp.status_code, resp.text
        except Exception as e:
            logger.warning("requests error fetching %s: %s", url, e)
            return 0, ""


# ─── Main crawler ───────────────────────────────────────────────────────────────

def research_url(url: str) -> ResearchResult:
    """
    Crawl a URL and its internal links, returning structured content.

    Args:
        url: The target URL to research.

    Returns:
        ResearchResult with ok=True if the main page was fetched,
        and pages containing title+text for each successfully crawled page.
        If the main page fails, ok=False and error is set.
    """
    result = ResearchResult(root_url=url)

    # Normalize the starting URL
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme:
            url = "https://" + url.lstrip("/")
            parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return ResearchResult(ok=False, root_url=url, error=f"Invalid URL: {e}")

    root_url = url
    root_domain = parsed.netloc.lstrip("www.")

    # BFS crawl
    visited: set[str] = set()
    to_visit: list[tuple[str, int]] = [(root_url, 0)]  # (url, depth)

    with _HTTPClient() as client:
        while to_visit and len(visited) < MAX_PAGES and result.total_chars < MAX_TOTAL_CHARS:
            current_url, depth = to_visit.pop(0)

            if current_url in visited:
                continue
            visited.add(current_url)

            # Respect depth limit
            if depth > MAX_DEPTH:
                continue

            # Fetch the page
            logger.info("Fetching [%d] %s", depth, current_url)
            status, raw = client.get(current_url)

            if status == 0:
                logger.warning("Failed to fetch %s (status=%d)", current_url, status)
                result.pages_failed += 1
                result.pages.append(PageResult(url=current_url, fetched=False, error="fetch failed"))
                continue

            if status >= 400:
                logger.warning("HTTP %d for %s", status, current_url)
                result.pages_failed += 1
                result.pages.append(PageResult(url=current_url, fetched=False, error=f"HTTP {status}"))
                continue

            # Parse HTML
            try:
                soup = BeautifulSoup(raw, "html.parser")
            except Exception as e:
                logger.warning("Parse error for %s: %s", current_url, e)
                result.pages_failed += 1
                result.pages.append(PageResult(url=current_url, fetched=False, error=f"parse error: {e}"))
                continue

            # Extract content
            title = _extract_title(soup)
            text = _strip_and_extract_text(soup)

            page_result = PageResult(url=current_url, title=title, text=text, fetched=True)
            result.pages.append(page_result)
            result.pages_crawled += 1
            result.covered_urls.append(current_url)

            # Discover new links
            if depth < MAX_DEPTH:
                links = _extract_links(soup, current_url, root_url)
                for link in links:
                    if link not in visited and len(visited) + len(to_visit) < MAX_PAGES:
                        to_visit.append((link, depth + 1))

            # Polite delay
            time.sleep(REQUEST_DELAY)

    # Set overall ok based on main page
    main_page = next((p for p in result.pages if p.url == root_url), None)
    if main_page is None or not main_page.fetched:
        result.ok = False
        result.error = f"Could not fetch the main page: {main_page.error if main_page else 'unknown error'}"
        return result

    result.ok = True
    return result


# ─── Content cleaning ───────────────────────────────────────────────────────────

# Lines that look like nav/menu/boilerplate leftovers after HTML strip.
# Matched as case-insensitive substrings on each line.
_NOISE_LINE_PATTERNS = [
    r"^\s*jobs:.*",
    r"^\s*sign in.*",
    r"^\s*sign up.*",
    r"^\s*log in.*",
    r"^\s*log out.*",
    r"^\s*register.*",
    r"^\s*subscribe.*newsletter.*",
    r"^\s*all rights reserved.*",
    r"^\s*privacy policy.*",
    r"^\s*terms of service.*",
    r"^\s*cookie.*policy.*",
    r"^\s*follow us on.*",
    r"^\s*share (this|on).*",
    r"^\s*skip to (content|main|nav).*",
    r"^\s*skip navigation.*",
    r"^\s*menu\s*$",
    r"^\s*navigation\s*$",
    r"^\s*search\s*$",
    r"^\s*home\s*$",
    r"^\s*back to top.*",
    r"^\s*page last updated.*",
    r"^\s*was this (article|page) helpful.*",
    r"^\s*related (articles?|posts?|links?).*",
    r"^\s*table of contents\s*$",
]
_NOISE_RE = re.compile("|".join(_NOISE_LINE_PATTERNS), re.IGNORECASE)

# A line is considered link-list only if it's just a comma/semicolon/pipe-
# separated list of short tokens (no full sentences, very short average length).
_LINK_LIST_RE = re.compile(r"^[\w\-\./\s|,;:&]{3,80}$")

# Common nav/menu/sidebar section labels — short Title-Case phrases that
# almost never appear as the sole content of a real page. Matched as
# case-insensitive equality (after stripping).
_NAV_LABELS = frozenset({
    "documentation", "docs", "api reference", "api docs", "api",
    "getting started", "tutorials", "tutorial", "guide", "guides",
    "community", "blog", "news", "about", "about us", "contact",
    "contact us", "support", "help", "help center", "faq",
    "products", "pricing", "careers", "jobs",
    "github", "twitter", "facebook", "linkedin", "youtube", "discord",
    "changelog", "release notes", "updates", "downloads",
    "examples", "demo", "demos", "showcase", "showcases",
    "open source", "open-source", "repository", "repo",
    "home", "homepage", "main", "index",
    "documentation portal", "developer portal",
    "navigation", "menu", "footer", "sidebar", "header",
    "previous", "next", "edit on github", "view source",
    "table of contents", "overview", "introduction",
    "install", "installation", "quickstart", "quick start",
    "learn more", "read more", "see more", "view all",
})

# Sentences shorter than this without a verb and without a period are likely
# fragments / list items / nav crumbs — drop them.
_MIN_SENTENCE_CHARS = 25


def _is_noise_line(line: str) -> bool:
    """Return True if the line looks like leftover nav / boilerplate junk."""
    if not line or not line.strip():
        return True
    if _NOISE_RE.match(line):
        return True
    stripped = line.strip()
    # Common nav/menu/sidebar section labels — these are never the only
    # content of a real page. Match as case-insensitive equality.
    if stripped.lower() in _NAV_LABELS:
        return True
    # A line that's just a pipe/comma-separated list of short tokens is a
    # link-list fragment (e.g. "Docs | API | GitHub | Twitter").
    if _LINK_LIST_RE.match(stripped) and len(stripped) < 60 and stripped.count("|") >= 1:
        return True
    if stripped.count("|") >= 2 and len(stripped) < 80:
        return True
    return False


def clean_extracted_text(text: str) -> str:
    """
    Clean text extracted by _strip_and_extract_text:
      - Drop lines that look like nav/menu/boilerplate leftovers
      - Deduplicate repeated lines (common in nav and footer sections)
      - Drop lines that are just punctuation/fragments
      - Collapse 3+ blank lines to 1
    """
    if not text:
        return text

    seen: set[str] = set()
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # Collapse runs of blank lines later
            if out and out[-1] == "":
                continue
            out.append("")
            continue
        if _is_noise_line(line):
            continue
        # Dedupe identical lines (nav links repeat across header/footer/sidebar)
        if line in seen:
            continue
        seen.add(line)
        out.append(line)

    # Collapse any remaining multi-blanks and trim
    cleaned = "\n".join(out).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# ─── Prompt-side budget enforcement ────────────────────────────────────────────


def _truncate_pages_to_budget(
    pages: list[PageResult],
    budget_chars: int,
) -> tuple[list[PageResult], list[str], list[str]]:
    """
    Trim a list of PageResults so the total injected text fits in budget_chars.

    Strategy:
      - Sort: root page first, then sub-pages in crawl order (preserve input order).
      - Take the root page in full up to its share of the budget.
      - Distribute remaining budget across sub-pages.
      - Pages that don't fit are dropped; their URLs are returned in `dropped_urls`.
      - Pages that partially fit are truncated with a marker.
    """
    if not pages:
        return [], [], []

    # Reserve room for headers, page markers, and the coverage note
    # (approx 60 chars per page * page count + 400 chars overhead).
    n = len(pages)
    overhead_per_page = 80
    overhead_total = 400 + overhead_per_page * n
    usable = max(budget_chars - overhead_total, budget_chars // 2)

    kept: list[PageResult] = []
    dropped_urls: list[str] = []
    truncated_urls: list[str] = []

    for i, page in enumerate(pages):
        if not page.fetched or not page.text:
            continue
        # First (root) page gets the lion's share; rest share the remainder.
        if i == 0:
            share = min(len(page.text), usable // 2 + usable // 4)
        else:
            share = min(len(page.text), max(usable // max(n, 1), 500))

        cleaned = clean_extracted_text(page.text)
        if len(cleaned) <= share:
            new_page = PageResult(
                url=page.url, title=page.title, text=cleaned,
                fetched=page.fetched, error=page.error,
            )
            kept.append(new_page)
        else:
            truncated = cleaned[:share].rsplit(" ", 1)[0]  # break on word boundary
            new_page = PageResult(
                url=page.url, title=page.title,
                text=truncated + f"\n[...truncated, page is {len(cleaned)} chars]",
                fetched=page.fetched, error=page.error,
            )
            kept.append(new_page)
            truncated_urls.append(page.url)

    # Drop pages that don't fit at all (only if we already exceeded budget).
    while kept:
        total_kept_chars = sum(len(p.text) for p in kept)
        if total_kept_chars <= budget_chars - overhead_total:
            break
        # Drop the last (lowest priority) page
        dropped = kept.pop()
        if dropped.fetched:
            dropped_urls.append(dropped.url)

    # Recompute final list (we mutated kept)
    return kept, dropped_urls, truncated_urls


def format_research_context(result: ResearchResult) -> str:
    """
    Format a ResearchResult as a readable context block for injection into
    LLM messages.

    The format is:
    [Website research: <root URL>]
    (Crawled N page(s), K failed)
    === Page 1: <url> — <title> ===
    <text>
    ...

    The prompt-side cap is MAX_INJECT_CHARS (12k). When pages are dropped or
    truncated, a short coverage note is included so the LLM can be honest
    about what was covered. Diagnostic messages go to the logger only.
    """
    if not result.ok:
        # Failure: return a compact, human-readable note for the LLM.
        # The full error is logged for the operator, not sent to the chat.
        logger.warning("Research failed for %s: %s", result.root_url, result.error)
        return (
            f"[Website research failed for {result.root_url}: {result.error}. "
            f"Tell the user you couldn't fetch that URL.]"
        )

    # Clean every fetched page up front.
    cleaned_pages: list[PageResult] = []
    for page in result.pages:
        if page.fetched and page.text:
            cleaned = clean_extracted_text(page.text)
            cleaned_pages.append(PageResult(
                url=page.url, title=page.title, text=cleaned,
                fetched=True, error=page.error,
            ))
        elif not page.fetched:
            cleaned_pages.append(page)

    # Enforce the prompt-side budget. This may drop or truncate pages.
    kept, dropped_urls, truncated_urls = _truncate_pages_to_budget(
        cleaned_pages, MAX_INJECT_CHARS
    )

    if dropped_urls or truncated_urls:
        logger.info(
            "Research budget: kept %d pages, dropped %d, truncated %d, "
            "inject budget %d",
            len(kept), len(dropped_urls), len(truncated_urls), MAX_INJECT_CHARS,
        )

    lines = [f"[Website research: {result.root_url}]"]
    lines.append(
        f"(Crawled {result.pages_crawled} page(s), {result.pages_failed} failed)"
    )

    for i, page in enumerate(kept, 1):
        title_part = f" — {page.title}" if page.title else ""
        lines.append(f"\n=== Page {i}: {page.url}{title_part} ===")
        lines.append(page.text)

    total_injected = sum(len(p.text) for p in kept)
    lines.append(f"\n[End of research — {total_injected} chars injected]")

    # Coverage note — machine-readable, for the LLM to use in its reply.
    # Not for human display; tell the LLM what was and wasn't covered.
    coverage_notes: list[str] = []
    if dropped_urls:
        coverage_notes.append(
            f"{len(dropped_urls)} page(s) omitted: {', '.join(dropped_urls)}"
        )
    if truncated_urls:
        coverage_notes.append(
            f"{len(truncated_urls)} page(s) partially shown: "
            f"{', '.join(truncated_urls)}"
        )
    if result.pages_failed > 0:
        failed_urls = [p.url for p in result.pages if not p.fetched]
        coverage_notes.append(
            f"{len(failed_urls)} page(s) failed to load: "
            f"{', '.join(failed_urls)}"
        )
    if coverage_notes:
        lines.append("\n[Coverage note: " + " | ".join(coverage_notes) + "]")

    output = "\n".join(lines)

    # Hard backstop: even with budget enforcement, never exceed the cap.
    # If somehow we did, log it (not print!) and slice.
    if len(output) > MAX_INJECT_CHARS:
        logger.warning(
            "Research output exceeded inject budget (%d > %d), truncating",
            len(output), MAX_INJECT_CHARS,
        )
        output = output[:MAX_INJECT_CHARS] + "\n[...truncated to fit budget]"

    return output


def detect_research_trigger(text: str) -> list[str]:
    """
    Extract URLs from text that indicate a research request.

    Looks for:
      - Full URLs (http:// or https://)
      - Bare domains (www.example.com, example.com, example.com/path)

    Returns a list of detected URLs (deduplicated, ordered by appearance).
    Bare domains are skipped if they overlap with a previously-matched full URL.

    Known file extensions (.md, .json, .py, etc.) are NOT treated as TLDs
    to avoid false positives like "see CLAUDE.md" being misidentified as
    a domain name.
    """
    urls: list[str] = []
    # Track the character ranges already covered by a full-URL match
    # so the bare-domain regex doesn't double-count them.
    covered_ranges: list[tuple[int, int]] = []

    # Regex for full URLs
    url_pattern = re.compile(
        r"https?://[^\s<>\"')\]]+",
        re.IGNORECASE,
    )
    for m in url_pattern.finditer(text):
        url = m.group(0).rstrip(".,;:!?)\"'>")
        if url not in urls:
            urls.append(url)
        covered_ranges.append((m.start(), m.end()))

    # File extensions that should NOT be treated as TLDs.
    # This prevents "see CLAUDE.md" from being misidentified as a domain.
    _FILE_EXT_BLACKLIST = frozenset({
        # Documentation / config
        "md", "markdown", "rst",
        "json", "jsonc", "yaml", "yml", "toml", "ini", "cfg", "conf",
        "xml", "properties",
        # Source code
        "py", "pyw", "js", "ts", "tsx", "jsx", "mjs", "cjs",
        "java", "kt", "scala", "go", "rs", "rb", "php",
        "c", "cpp", "cc", "cxx", "h", "hpp", "cs",
        "swift", "m", "mm",
        "sh", "bash", "zsh", "fish", "ps1", "psm1",
        "bat", "cmd", "reg",
        # Web
        "html", "htm", "xhtml", "css", "scss", "sass", "less",
        "vue", "svelte",
        # Data / logs
        "csv", "tsv", "sql", "db", "sqlite",
        "log", "txt", "text",
        # Build / packaging
        "makefile", "dockerfile", "gradle", "bazel",
        "zip", "tar", "gz", "rar", "7z",
        "exe", "msi", "dmg", "app", "deb", "rpm", "apk",
        "jar", "war", "ear",
        # Images / media
        "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp",
        "mp3", "mp4", "avi", "mov", "mkv", "webm",
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        # Other common non-TLDs
        "lock", "sum", "env", "example", "sample", "test", "spec",
        "ignore", "editorconfig", "gitattributes",
        # Generic names that are commonly file names, not domains
        "config", "settings", "local", "global", "default",
        "readme", "changelog", "license", "contributing",
        "todo", "notes", "draft", "tmp", "temp",
    })

    def _looks_like_file_ext(tld: str) -> bool:
        """Return True if tld looks like a file extension, not a real TLD."""
        tld_lower = tld.lower()
        if tld_lower in _FILE_EXT_BLACKLIST:
            return True
        # Single-letter TLDs like "e" don't exist in practice
        if len(tld) < 2:
            return True
        return False

    # Regex for bare domains (www.example.com, example.com, example.com/path).
    # The domain part matches one or more dot-separated labels, each ending
    # with a 2+ char TLD-like segment. The path is optional and starts with /.
    domain_pattern = re.compile(
        r"(?<![\w@./:\\])(?:www\.)?(?:[a-zA-Z0-9][a-zA-Z0-9-]*\.)+[a-zA-Z]{2,}(?:/[^\s]*)?\b",
    )
    for m in domain_pattern.finditer(text):
        m_start, m_end = m.start(), m.end()
        # Skip if this span overlaps with an already-collected full URL
        if any(s <= m_start < e or s < m_end <= e or (m_start <= s and m_end >= e)
               for s, e in covered_ranges):
            continue
        domain = m.group(0).rstrip("/")
        # Extract the TLD (last dot-separated segment before any path)
        path_sep = domain.find("/")
        host_part = domain if path_sep == -1 else domain[:path_sep]
        tld = host_part.rsplit(".", 1)[-1].lower()

        # Reject if the "TLD" is actually a file extension
        if _looks_like_file_ext(tld):
            continue

        # Reject single-label bare domains (e.g. just "example" with no dot)
        if "." not in host_part:
            continue

        if domain.startswith("www."):
            candidate = "https://" + domain
        else:
            candidate = "https://www." + domain
        if candidate not in urls:
            urls.append(candidate)

    return urls


# ─── Pre-processing hook ───────────────────────────────────────────────────────


def maybe_research(text: str) -> tuple[bool, str]:
    """
    Check if a user message contains a research trigger and run research.

    Triggers:
      - A URL is detected in the text
      - Phrases like "this link", "this website", "do your research",
        "research this", "everything on this", "digest", "deep dive"

    Returns:
        (True, research_context)  — if research was performed
        (False, "")               — if no research trigger was found
    """
    # Check for research trigger phrases
    trigger_phrases = [
        "do your research", "do research", "do some research",
        "deep dive", "deep research", "research this",
        "this link", "this website", "this page", "this url",
        "everything on this", "give me everything", "digest this",
        "crawl", "scrape", "extract all",
    ]
    text_lower = text.lower()
    has_trigger_phrase = any(phrase in text_lower for phrase in trigger_phrases)

    # Check for URLs
    urls = detect_research_trigger(text)

    if not urls and not has_trigger_phrase:
        return False, ""

    if not urls:
        # Trigger phrase but no URL — let the LLM handle it
        return False, ""

    # Run research on the first detected URL
    target_url = urls[0]
    logger.info("Research trigger detected: %s", target_url)

    try:
        result = research_url(target_url)
        context = format_research_context(result)
        return True, context
    except Exception as e:
        logger.error("Research failed for %s: %s", target_url, e)
        return False, ""
