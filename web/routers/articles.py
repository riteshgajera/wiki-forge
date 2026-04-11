"""Articles API router."""
from __future__ import annotations
import re
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from kb.storage.wiki_manager import WikiManager
from kb.utils.config import settings

router = APIRouter()


@router.get("/")
async def list_articles(subdir: str | None = None, limit: int = 50) -> JSONResponse:
    wiki = WikiManager(settings.wiki_dir)
    articles = []
    for path in wiki.list_articles(subdir)[:limit]:
        content = path.read_text(encoding="utf-8")
        title_m = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else path.stem
        articles.append({
            "slug": path.stem,
            "title": title,
            "subdir": path.parent.name,
            "size": len(content),
        })
    return JSONResponse({"articles": articles, "total": len(articles)})


@router.get("/{subdir}/{slug}")
async def get_article(subdir: str, slug: str) -> JSONResponse:
    wiki = WikiManager(settings.wiki_dir)
    content = wiki.read_article(slug, subdir)
    if content is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"slug": slug, "subdir": subdir, "content": content})
