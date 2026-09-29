"""Deep research skill for Fairy 2.0."""
import json
from skills.web_search import web_search


def deep_research(topic: str, depth: int = 3) -> str:
    """Multi-pass research on a topic."""
    try:
        broad = json.loads(web_search(topic, max_results=depth, mode="research"))
        results = broad.get("results", [])
        sub_queries = [r["title"] for r in results[:3] if r.get("title")]
        for q in sub_queries:
            try:
                extra = json.loads(web_search(q, max_results=2))
                results.extend(extra.get("results", []))
            except Exception:
                pass
        return json.dumps({"topic": topic, "sources": len(results), "results": results[:10]})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})