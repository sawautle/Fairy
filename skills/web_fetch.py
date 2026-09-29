import requests
from bs4 import BeautifulSoup

def web_fetch(url: str) -> str:
    """
    Fetch a URL and extract readable text.
    Falls back to raw text if parsing fails.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Kill noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        
        # Try to find main content first
        main = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
        
        # Clean whitespace
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)
        
        # Cap context window burn
        if len(text) > 8000:
            text = text[:8000] + "\n\n[Content truncated at 8000 chars]"
        
        return text
        
    except Exception as e:
        return f"Fetch error for {url}: {e}"