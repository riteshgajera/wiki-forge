"""
WikiLogger: manages the two special wiki navigation files.
  - log.md  : append-only chronological operation timeline
  - index.md: content-oriented catalog with L0 summaries
  - wip.md  : work-in-progress tracker
  - sources.md: source registry
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kb.utils.logging import get_logger

logger = get_logger("storage.wiki_logger")


def _now_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WikiLogger:
    """Manages log.md, index.md, wip.md, and sources.md."""

    def __init__(self, wiki_dir: str | Path) -> None:
        self.wiki_dir = Path(wiki_dir)
        self.log_path = self.wiki_dir / "log.md"
        self.index_path = self.wiki_dir / "index.md"
        self.wip_path = self.wiki_dir / "wip.md"
        self.sources_path = self.wiki_dir / "_meta" / "sources.md"
        self._ensure_files()

    def _ensure_files(self) -> None:
        (self.wiki_dir / "_meta").mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text(
                "# Operation Log\n\n"
                "> Append-only. Format: `## [YYYY-MM-DD] operation | title`\n"
                "> Parse last 5: `grep '^## \\[' wiki/log.md | tail -5`\n\n",
                encoding="utf-8",
            )
        if not self.wip_path.exists():
            self.wip_path.write_text(
                "# Work in Progress\n\n"
                "> Updated during active sessions. Captures state between sessions.\n\n"
                "## Current focus\n\n_Nothing active._\n\n"
                "## Pending sources\n\n## Open questions\n\n## Next actions\n",
                encoding="utf-8",
            )
        if not self.sources_path.exists():
            self.sources_path.write_text(
                "# Source Registry\n\n"
                "| Source | Type | Ingested | Pages touched | Status |\n"
                "|--------|------|----------|---------------|--------|\n",
                encoding="utf-8",
            )

    # ── log.md ────────────────────────────────────────────────────────────────

    def log(self, operation: str, title: str, detail: str = "") -> None:
        """Append one entry to log.md. Parseable format: ## [date] op | title"""
        date = _now_date()
        entry = f"## [{date}] {operation} | {title}\n"
        if detail:
            entry += f"\n{detail}\n"
        entry += "\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.debug("log_entry", operation=operation, title=title[:60])

    def recent_log(self, n: int = 10) -> list[str]:
        """Return the last n log entry header lines."""
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        headers = [l for l in lines if re.match(r"^## \[", l)]
        return headers[-n:]

    # ── index.md ──────────────────────────────────────────────────────────────

    def rebuild_index(self, wiki_manager: Any) -> None:
        """Rebuild index.md from all wiki articles. Called after every compile."""
        import re as _re

        categories: dict[str, list[dict]] = {
            "concepts": [], "entities": [], "summaries": [], "analysis": [], "other": []
        }

        for path in wiki_manager.list_articles():
            if path.name in ("WIKI.md", "index.md", "wip.md", "README.md"):
                continue
            content = path.read_text(encoding="utf-8")
            fm = _parse_frontmatter(content)
            title = fm.get("title", path.stem)
            l0 = fm.get("l0_summary", "")
            if not l0:
                # Extract first non-heading sentence as L0
                body = _re.sub(r"^---\n.*?---\n", "", content, flags=_re.DOTALL)
                body = _re.sub(r"^#+.+$", "", body, flags=_re.MULTILINE)
                sentences = [s.strip() for s in body.split(".") if len(s.strip()) > 20]
                l0 = (sentences[0][:120] + "…") if sentences else ""

            tags = fm.get("tags", [])
            updated_raw = fm.get("updated", "")
            if hasattr(updated_raw, "isoformat"):
                updated = updated_raw.isoformat()[:10]
            else:
                updated = str(updated_raw)[:10] if updated_raw else ""
            subdir = path.parent.name
            slug = path.stem
            cat = subdir if subdir in categories else "other"
            rel_link = f"{subdir}/{slug}" if subdir != "wiki" else slug
            categories[cat].append({
                "link": rel_link,
                "title": title,
                "l0": l0,
                "tags": tags if isinstance(tags, list) else [tags],
                "updated": updated,
            })

        lines = [
            "# Knowledge Base Index\n",
            f"> Auto-generated {_now_date()} · "
            f"{sum(len(v) for v in categories.values())} pages\n",
            "> Read this first when answering queries — scan L0 summaries to find relevant pages.\n",
        ]

        order = ["concepts", "entities", "summaries", "analysis", "other"]
        for cat in order:
            pages = categories[cat]
            if not pages:
                continue
            lines.append(f"\n## {cat.title()} ({len(pages)})\n")
            lines.append("| Page | Summary | Tags | Updated |")
            lines.append("|------|---------|------|---------|")
            for p in sorted(pages, key=lambda x: x["title"]):
                tag_str = ", ".join(str(t) for t in p["tags"][:3])
                lines.append(
                    f"| [[{p['link']}|{p['title']}]] "
                    f"| {p['l0'][:80]} "
                    f"| {tag_str} "
                    f"| {p['updated']} |"
                )

        self.index_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("index_rebuilt", pages=sum(len(v) for v in categories.values()))

    def search_index(self, query: str, top_k: int = 10) -> list[dict]:
        """Fast L0 keyword search over index.md."""
        if not self.index_path.exists():
            return []
        content = self.index_path.read_text(encoding="utf-8")
        terms = query.lower().split()
        results = []
        for line in content.splitlines():
            if not line.startswith("|"):
                continue
            score = sum(line.lower().count(t) for t in terms)
            if score > 0:
                results.append({"line": line, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── sources.md ────────────────────────────────────────────────────────────

    def register_source(
        self, source_path: str, doc_type: str, pages_touched: int, status: str = "done"
    ) -> None:
        date = _now_date()
        row = f"| `{source_path}` | {doc_type} | {date} | {pages_touched} | {status} |\n"
        with open(self.sources_path, "a", encoding="utf-8") as f:
            f.write(row)

    # ── wip.md ────────────────────────────────────────────────────────────────

    def update_wip(self, focus: str = "", pending: list[str] | None = None,
                   questions: list[str] | None = None, next_actions: list[str] | None = None) -> None:
        """Overwrite wip.md with current state."""
        lines = [
            "# Work in Progress\n",
            f"> Last updated: {_now_date()}\n",
            "\n## Current focus\n",
            focus or "_Nothing active._",
        ]
        if pending:
            lines.append("\n## Pending sources\n")
            for p in pending:
                lines.append(f"- {p}")
        if questions:
            lines.append("\n## Open questions\n")
            for q in questions:
                lines.append(f"- {q}")
        if next_actions:
            lines.append("\n## Next actions\n")
            for a in next_actions:
                lines.append(f"- [ ] {a}")
        self.wip_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from markdown content."""
    import yaml
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1)) or {}
        except Exception:
            pass
    return {}
