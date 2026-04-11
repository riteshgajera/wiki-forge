"""
FastAPI web application for the WIKI Forge.
Provides search, article viewing, and agent triggering via HTMX.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from kb.utils.config import settings
from kb.utils.logging import get_logger, setup_logging

setup_logging(settings.log_level, settings.log_format)
logger = get_logger("web.app")

app = FastAPI(
    title="Wiki Forge",
    description="Local-first multi-agent knowledge base",
    version="0.1.0",
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

from web.routers import articles, search, agents  # noqa: E402
app.include_router(articles.router, prefix="/api/articles", tags=["articles"])
app.include_router(search.router,   prefix="/api/search",   tags=["search"])
app.include_router(agents.router,   prefix="/api/agents",   tags=["agents"])


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)

@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def apple_icon() -> Response:
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    from kb.storage.metadata_store import MetadataStore
    from kb.storage.wiki_manager import WikiManager
    from kb.storage.wiki_logger import WikiLogger

    store = MetadataStore(settings.db_path)
    stats = store.stats()
    wiki = WikiManager(settings.wiki_dir)
    article_count = len(wiki.list_articles())
    wlog = WikiLogger(settings.wiki_dir)
    recent_log = wlog.recent_log(5)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
        "article_count": article_count,
        "wiki_dir": settings.wiki_dir,
        "settings_provider": settings.llm.provider,
        "recent_log": recent_log,
    })


@app.get("/articles", response_class=HTMLResponse)
async def articles_page(request: Request, subdir: str = "") -> HTMLResponse:
    from kb.storage.wiki_manager import WikiManager

    wiki = WikiManager(settings.wiki_dir)
    articles_data = []
    target_subdir = subdir or None

    for path in wiki.list_articles(target_subdir)[:200]:
        content = path.read_text(encoding="utf-8", errors="replace")
        title_m = re.search(r'^title:\s*"?(.+?)"?\s*$', content, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else path.stem
        tags_m = re.search(r'^tags:\s*\[(.+?)\]', content, re.MULTILINE)
        tags = [t.strip().strip("'\"") for t in tags_m.group(1).split(",") if t.strip()] if tags_m else []
        articles_data.append({
            "slug": path.stem,
            "title": title,
            "subdir": path.parent.name,
            "tags": tags[:3],
        })

    return templates.TemplateResponse("articles.html", {
        "request": request,
        "articles": articles_data,
        "subdirs": ["concepts", "entities", "summaries", "analysis"],
        "active_subdir": subdir,
    })


@app.get("/articles/{subdir}/{slug}", response_class=HTMLResponse)
async def article_detail(request: Request, subdir: str, slug: str) -> HTMLResponse:
    from kb.storage.wiki_manager import WikiManager

    wiki = WikiManager(settings.wiki_dir)
    content: str | None = None

    # Root-level wiki files (log, index, wip, WIKI) served under subdir="wiki"
    if subdir == "wiki":
        for candidate in [slug, slug.upper(), slug.lower()]:
            p = wiki.wiki_dir / f"{candidate}.md"
            if p.exists():
                content = p.read_text(encoding="utf-8")
                break
    else:
        content = wiki.read_article(slug, subdir)
        # Fallback: search all subdirs
        if content is None:
            for sd in ("concepts", "entities", "summaries", "analysis", "_meta"):
                content = wiki.read_article(slug, sd)
                if content:
                    subdir = sd
                    break

    if content is None:
        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;padding:40px'>"
            f"<h2>Not found</h2><p><code>{subdir}/{slug}.md</code> does not exist.</p>"
            f"<p><a href='/articles'>← All articles</a></p></body></html>",
            status_code=404,
        )

    html_content = _markdown_to_html(content)
    return templates.TemplateResponse("article.html", {
        "request": request,
        "slug": slug,
        "subdir": subdir,
        "content": html_content,
        "raw": content,
    })


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = "") -> HTMLResponse:
    from kb.storage.wiki_manager import WikiManager

    results = []
    if q.strip():
        wiki = WikiManager(settings.wiki_dir)
        terms = q.lower().split()
        for path, content in wiki.iter_articles():
            score = sum(content.lower().count(t) for t in terms)
            if score > 0:
                title_m = re.search(r'^title:\s*"?(.+?)"?\s*$', content, re.MULTILINE)
                title = title_m.group(1).strip() if title_m else path.stem
                idx = content.lower().find(terms[0])
                snippet = content[max(0, idx-80):idx+250].strip() if idx >= 0 else content[:250]
                results.append({
                    "title": title, "slug": path.stem,
                    "subdir": path.parent.name,
                    "score": float(score), "snippet": snippet,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:10]

    return templates.TemplateResponse("search.html", {
        "request": request,
        "query": q,
        "results": results,
    })


@app.get("/run", response_class=HTMLResponse)
async def run_page(request: Request) -> HTMLResponse:
    from kb.storage.metadata_store import MetadataStore
    store = MetadataStore(settings.db_path)
    return templates.TemplateResponse("run.html", {
        "request": request,
        "stats": store.stats(),
        "raw_dir": settings.raw_dir,
        "wiki_dir": settings.wiki_dir,
    })


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "wiki_dir": settings.wiki_dir,
        "db": settings.db_path,
        "provider": settings.llm.provider,
        "model": settings.llm.model,
    }


def _markdown_to_html(md: str) -> str:
    """Convert markdown to HTML with correct [[wikilink]] routing."""
    html = md
    # Strip frontmatter
    html = re.sub(r"^---\n.*?---\n", "", html, flags=re.DOTALL)

    # Code blocks first (protect content from further transforms)
    code_blocks: list[str] = []
    def stash_code(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"__CODE_{len(code_blocks)-1}__"
    html = re.sub(r"```[\w]*\n(.*?)```", stash_code, html, flags=re.DOTALL)
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)

    # Headers
    for n in range(4, 0, -1):
        html = re.sub(rf"^{'#'*n} (.+)$", rf"<h{n}>\1</h{n}>", html, flags=re.MULTILINE)

    # Bold / italic
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*",     r"<em>\1</em>", html)

    # Wikilinks: [[subdir/slug|Display]] or [[slug|Display]] or [[slug]]
    def wl_full(m: re.Match) -> str:
        path_part, display = m.group(1).strip(), m.group(2).strip()
        href = f"/articles/{path_part}" if "/" in path_part else f"/articles/concepts/{path_part}"
        return f'<a href="{href}" class="wikilink">{display}</a>'

    def wl_simple(m: re.Match) -> str:
        path_part = m.group(1).strip()
        if "/" in path_part:
            label = path_part.split("/")[-1]
            return f'<a href="/articles/{path_part}" class="wikilink">{label}</a>'
        return f'<a href="/articles/concepts/{path_part}" class="wikilink">{path_part}</a>'

    html = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", wl_full, html)
    html = re.sub(r"\[\[([^\]]+)\]\]",             wl_simple, html)

    # Markdown links
    html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)

    # Blockquotes
    html = re.sub(r"^> (.+)$", r"<blockquote>\1</blockquote>", html, flags=re.MULTILINE)

    # HR
    html = re.sub(r"^---+$", r"<hr>", html, flags=re.MULTILINE)

    # Tables (basic pipe tables)
    def render_table(m: re.Match) -> str:
        lines = [l.strip() for l in m.group(0).strip().splitlines()]
        rows, is_header = [], True
        for line in lines:
            if re.match(r"^\|[\-| :]+\|$", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if is_header else "td"
            rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            is_header = False
        if not rows:
            return ""
        return f'<table class="wiki-table"><thead>{rows[0]}</thead><tbody>{"".join(rows[1:])}</tbody></table>'
    html = re.sub(r"(\|.+\|\n?)+", render_table, html)

    # Lists
    html = re.sub(r"^  - (.+)$", r"<li class='nested'>\1</li>", html, flags=re.MULTILINE)
    html = re.sub(r"^- (.+)$",   r"<li>\1</li>",                html, flags=re.MULTILINE)
    html = re.sub(r"(<li>.*?</li>\n?)+", lambda m: f"<ul>{m.group(0)}</ul>", html, flags=re.DOTALL)
    html = re.sub(r"^\d+\. (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)

    # Paragraphs
    parts = re.split(r"\n{2,}", html)
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^<(h[1-6]|ul|ol|table|pre|blockquote|hr|__CODE)", part):
            out.append(part)
        else:
            out.append(f"<p>{part}</p>")

    result = "\n".join(out)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        result = result.replace(f"__CODE_{i}__", f"<pre><code>{block}</code></pre>")

    return result
