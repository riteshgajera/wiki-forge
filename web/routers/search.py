"""Search API router."""
from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from kb.storage.wiki_manager import WikiManager
from kb.utils.config import settings

router = APIRouter()


@router.get("/")
async def search(q: str, k: int = 5, mode: str = "text") -> JSONResponse:
    if not q.strip():
        return JSONResponse({"results": [], "query": q})

    wiki = WikiManager(settings.wiki_dir)
    terms = q.lower().split()
    scored = []

    for path, content in wiki.iter_articles():
        import re
        content_lower = content.lower()
        score = sum(content_lower.count(t) for t in terms)
        if score > 0:
            title_m = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
            title = title_m.group(1).strip() if title_m else path.stem
            idx = content_lower.find(terms[0])
            snippet = content[max(0, idx - 80):idx + 250].strip() if idx >= 0 else content[:250]
            scored.append({
                "slug": path.stem,
                "subdir": path.parent.name,
                "title": title,
                "score": float(score),
                "snippet": snippet,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return JSONResponse({"results": scored[:k], "query": q, "total": len(scored)})
