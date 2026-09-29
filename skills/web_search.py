"""Web search skill for Fairy 2.0."""
import json
import re
import urllib.parse
from typing import Optional

from config import GEMINI_KEY, log

try:
    import ddgs as _ddgs_mod
    _DDG = True
    _DDG_NEW = True
except ImportError:
    _DDG_NEW = False
    try:
        import duckduckgo_search
        _DDG = True
    except Exception:
        _DDG = False


# ─────────────────────────────────────────────────────────────────
# Sites with multiple valid URLs — pick the most relevant one
# ─────────────────────────────────────────────────────────────────

# Maps a domain root → the canonical "home" URL we prefer for navigation.
# When multiple results share the same domain, we rank the one whose path
# best matches the query rather than blindly taking the first result.
_CANONICAL_HOMES = {
    "youtube.com": "https://www.youtube.com/",
    "reddit.com": "https://www.reddit.com/",
    "wikipedia.org": "https://www.wikipedia.org/",
    "twitter.com": "https://twitter.com/",
    "x.com": "https://x.com/",
    "github.com": "https://github.com/",
    "stackoverflow.com": "https://stackoverflow.com/",
    "amazon.com": "https://www.amazon.com/",
    "instagram.com": "https://www.instagram.com/",
    "tiktok.com": "https://www.tiktok.com/",
    "twitch.tv": "https://www.twitch.tv/",
    "netflix.com": "https://www.netflix.com/",
    "spotify.com": "https://open.spotify.com/",
    "discord.com": "https://discord.com/",
    "facebook.com": "https://www.facebook.com/",
    "linkedin.com": "https://www.linkedin.com/",
    "imdb.com": "https://www.imdb.com/",
    "rottentomatoes.com": "https://www.rottentomatoes.com/",
}


def _domain_root(url: str) -> str:
    """Extract base domain (e.g. 'youtube.com') from a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower().lstrip("www.")
        return host
    except Exception:
        return ""


def _result_relevance_score(result: dict, query: str) -> float:
    """Score a search result by how well it matches the query.

    Considers: query words in title, query words in URL path, body snippet.
    Higher is better.
    """
    title = (result.get("title") or "").lower()
    url = (result.get("url") or result.get("href") or "").lower()
    body = (result.get("body") or "").lower()
    query_words = re.split(r"\W+", query.lower())
    score = 0.0
    for word in query_words:
        if not word:
            continue
        if word in title:
            score += 3.0
        if word in url:
            score += 1.5
        if word in body:
            score += 0.5
    return score


def pick_best_url(results: list, query: str) -> Optional[str]:
    """Given a list of search results, pick the single best URL to open.

    Strategy:
    1. Score all results by query relevance.
    2. If multiple results share the same domain root (e.g. multiple YouTube
       pages), pick the one with the highest relevance score, not the first.
    3. Return the URL of the top-scored result overall.
    """
    if not results:
        return None

    scored = []
    for r in results:
        url = r.get("url") or r.get("href") or ""
        if not url:
            continue
        score = _result_relevance_score(r, query)
        domain = _domain_root(url)
        scored.append((score, url, domain, r))

    if not scored:
        return None

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _gemini_search(query: str, mode: str = "search") -> Optional[list]:
    if not GEMINI_KEY:
        return None
    try:
        import google.genai as genai
        client = genai.Client(api_key=GEMINI_KEY)
        prompt = query
        if mode == "news":
            prompt = f"Latest news about: {query}"
        elif mode == "price":
            prompt = f"Current price and where to buy: {query}"
        elif mode == "compare":
            prompt = f"Compare specs and prices: {query}"
        elif mode == "research":
            prompt = f"Deep research summary with sources: {query}"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"tools": [{"google_search": {}}]},
        )
        text = response.text if hasattr(response, "text") else str(response)
        return [{"title": "Gemini Result", "body": text, "url": ""}]
    except Exception as e:
        # Gemini 2.5 Flash is rate-limited / unavailable to some users; fallback to DDG.
        if "404" in str(e) or "not available" in str(e).lower() or "NOT_FOUND" in str(e):
            return None
        log(f"Gemini search error: {e}")
        return None


def _ddg_search(query: str, max_results: int = 8) -> list:
    """Run a DuckDuckGo search and return normalised result dicts."""
    if not _DDG:
        return [{"title": "DuckDuckGo not installed", "body": "pip install ddgs", "url": ""}]
    try:
        if _DDG_NEW:
            from ddgs import DDGS
        else:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results)
            results = []
            for r in raw:
                url = r.get("href") or r.get("url") or ""
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": url,
                })
            return results
    except Exception as e:
        return [{"title": "DDG Error", "body": str(e), "url": ""}]


def web_search(query: str, max_results: int = 8, mode: str = "search") -> str:
    """Unified web search.

    mode: search | news | price | compare | research

    Returns JSON string with keys:
      source, mode, results (list), best_url (the single best URL to open).

    best_url is selected by query-relevance scoring — for sites with multiple
    URLs (YouTube, Reddit, Wikipedia, etc.) this picks the most relevant page
    rather than the first result.
    """
    gemini = _gemini_search(query, mode)
    if gemini:
        return json.dumps({
            "source": "gemini",
            "mode": mode,
            "results": gemini,
            "best_url": "",  # Gemini doesn't return URLs
        })

    results = _ddg_search(query, max_results)
    best_url = pick_best_url(results, query)

    return json.dumps({
        "source": "duckduckgo",
        "mode": mode,
        "results": results,
        "best_url": best_url or "",
    })


def search_news(query: str, max_results: int = 5) -> str:
    return web_search(query, max_results, mode="news")


def search_price(query: str, max_results: int = 5) -> str:
    return web_search(query, max_results, mode="price")


def search_research(query: str, max_results: int = 5) -> str:
    return web_search(query, max_results, mode="research")


def search_compare(query: str, max_results: int = 5) -> str:
    return web_search(query, max_results, mode="compare")
