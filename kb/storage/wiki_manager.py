"""
File-based wiki manager.
Handles reading, writing, listing, and versioning of Obsidian-compatible markdown files.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from kb.utils.helpers import slugify
from kb.utils.logging import get_logger

logger = get_logger("storage.wiki")


FRONTMATTER_TEMPLATE = """\
---
title: "{title}"
type: {page_type}
aliases: {aliases}
tags: {tags}
created: {created}
updated: {updated}
source: {source}
confidence: {confidence}
depth: {depth}
status: {status}
l0_summary: "{l0_summary}"
related: {related}
contradicts: {contradicts}
---

"""


class WikiManager:
    """Manages the /wiki directory tree of markdown articles."""

    SUBDIRS = ("concepts", "summaries", "index", "_meta")

    def __init__(self, wiki_dir: str | Path) -> None:
        self.wiki_dir = Path(wiki_dir)
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        for sub in self.SUBDIRS:
            (self.wiki_dir / sub).mkdir(parents=True, exist_ok=True)

    def write_article(
        self,
        title: str,
        content: str,
        subdir: str = "concepts",
        slug: str | None = None,
        source: str = "",
        tags: list[str] | None = None,
        aliases: list[str] | None = None,
        confidence: float = 1.0,
        backup: bool = True,
        page_type: str = "concept",
        depth: str = "L1",
        status: str = "active",
        l0_summary: str = "",
        related: list[str] | None = None,
        contradicts: list[str] | None = None,
    ) -> Path:
        """Write a wiki article. Backs up existing version before overwriting."""
        slug = slug or slugify(title)
        target = self.wiki_dir / subdir / f"{slug}.md"
        target.parent.mkdir(parents=True, exist_ok=True)

        if backup and target.exists():
            bak_dir = self.wiki_dir / "_meta" / "backups" / subdir
            bak_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            shutil.copy2(target, bak_dir / f"{slug}.{ts}.md.bak")
            logger.debug("wiki_backup", path=str(target))

        now = datetime.now(timezone.utc).isoformat()
        # Auto-generate L0 summary from first sentence of content if not provided
        if not l0_summary and content:
            import re
            body = re.sub(r"^#.+$", "", content, flags=re.MULTILINE)
            sentences = [s.strip() for s in body.split(".") if len(s.strip()) > 15]
            l0_summary = (sentences[0][:120] + "…") if sentences else title

        frontmatter = FRONTMATTER_TEMPLATE.format(
            title=title.replace('"', '\\"'),
            page_type=page_type,
            aliases=str(aliases or []),
            tags=str(tags or []),
            created=now,
            updated=now,
            source=source,
            confidence=round(confidence, 3),
            depth=depth,
            status=status,
            l0_summary=l0_summary.replace('"', '\\"')[:120],
            related=str(related or []),
            contradicts=str(contradicts or []),
        )

        full_content = frontmatter + content
        target.write_text(full_content, encoding="utf-8")
        logger.info("wiki_write", path=str(target), size=len(full_content))
        return target

    def read_article(self, slug: str, subdir: str = "concepts") -> str | None:
        path = self.wiki_dir / subdir / f"{slug}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def article_exists(self, slug: str, subdir: str = "concepts") -> bool:
        return (self.wiki_dir / subdir / f"{slug}.md").exists()

    def list_articles(self, subdir: str | None = None) -> list[Path]:
        if subdir:
            return sorted((self.wiki_dir / subdir).glob("*.md"))
        return sorted(p for p in self.wiki_dir.rglob("*.md")
                       if "_meta/backups" not in str(p))

    def delete_article(self, slug: str, subdir: str = "concepts") -> bool:
        path = self.wiki_dir / subdir / f"{slug}.md"
        if path.exists():
            path.unlink()
            return True
        return False

    def iter_articles(self) -> Iterator[tuple[Path, str]]:
        """Iterate (path, content) for all wiki articles."""
        for path in self.list_articles():
            try:
                yield path, path.read_text(encoding="utf-8")
            except OSError:
                continue

    def write_index(self, content: str) -> Path:
        """Write the main index / TOC file."""
        target = self.wiki_dir / "index" / "README.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def get_all_wikilinks(self) -> dict[str, list[str]]:
        """Return a map of {article_slug: [linked_slugs]} by scanning [[wikilink]] syntax."""
        import re
        pattern = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
        result: dict[str, list[str]] = {}
        for path, content in self.iter_articles():
            slug = path.stem
            links = [slugify(m) for m in pattern.findall(content)]
            if links:
                result[slug] = links
        return result
