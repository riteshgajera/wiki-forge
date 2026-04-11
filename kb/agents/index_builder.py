"""
IndexBuilderAgent: builds and maintains the wiki index (TOC, MOC files).
Runs after all articles are compiled to produce navigation structure.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kb.agents.base import AgentInput, BaseAgent
from kb.utils.logging import get_logger

logger = get_logger("agent.index_builder")

SYSTEM_PROMPT = """\
You are a wiki information architect. Build clear, navigable index files for a knowledge base.

RULES:
- Group articles logically by domain/topic.
- Use Obsidian [[wikilink]] syntax for all links.
- Output ONLY valid JSON. No markdown, no explanation.\
"""

CLUSTER_PROMPT = """\
Given this list of wiki articles with their topics, group them into logical sections \
for a Table of Contents. Each section should have 2-10 articles.

ARTICLES:
{articles_json}

Return EXACTLY this JSON:
{{
  "sections": [
    {{
      "title": "Section Title",
      "description": "What this section covers",
      "articles": ["slug1", "slug2"]
    }}
  ],
  "confidence": 0.9
}}\
"""


class IndexBuilderAgent(BaseAgent):
    """Builds the wiki index and map-of-content files."""

    name = "index_builder"

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        # inp.metadata["wiki_articles"] should be list of {slug, title, topics, subdir}
        articles = inp.metadata.get("wiki_articles", [])

        if not articles:
            return {"sections": [], "toc_content": "", "confidence": 1.0}

        # For large wikis, cluster with LLM; for small ones, group by topic
        if len(articles) <= 30:
            sections = self._simple_group(articles)
            confidence = 0.95
        else:
            articles_json = json.dumps(articles[:80], indent=2)  # LLM context limit
            prompt = CLUSTER_PROMPT.format(articles_json=articles_json)
            result = self._call_llm_json(prompt, SYSTEM_PROMPT)
            sections = result.get("sections", [])
            confidence = result.get("confidence", 0.8)

        toc = self._render_toc(sections, articles)
        return {
            "sections": sections,
            "toc_content": toc,
            "article_count": len(articles),
            "confidence": confidence,
        }

    def _simple_group(self, articles: list[dict]) -> list[dict]:
        """Group articles by first topic tag without LLM."""
        groups: dict[str, list[str]] = {}
        for art in articles:
            topics = art.get("topics", ["general"])
            key = topics[0] if topics else "general"
            groups.setdefault(key, []).append(art["slug"])
        return [
            {"title": k.replace("-", " ").title(), "description": f"Articles about {k}", "articles": v}
            for k, v in sorted(groups.items())
        ]

    def _render_toc(self, sections: list[dict], all_articles: list[dict]) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug_to_title = {a["slug"]: a.get("title", a["slug"]) for a in all_articles}
        slug_to_subdir = {a["slug"]: a.get("subdir", "concepts") for a in all_articles}

        lines = [
            "# Knowledge Base Index",
            f"\n> Auto-generated on {now} · {len(all_articles)} articles\n",
            "## Quick Navigation\n",
        ]
        for section in sections:
            lines.append(f"### {section['title']}")
            if section.get("description"):
                lines.append(f"\n*{section['description']}*\n")
            for slug in section.get("articles", []):
                title = slug_to_title.get(slug, slug)
                subdir = slug_to_subdir.get(slug, "concepts")
                lines.append(f"- [[{subdir}/{slug}|{title}]]")
            lines.append("")

        return "\n".join(lines)

    def validate(self, result: dict[str, Any]) -> bool:
        return "toc_content" in result and "confidence" in result

    @staticmethod
    def build_article_list(wiki_manager: Any) -> list[dict[str, Any]]:
        """
        Helper: scan wiki directory and return article metadata list.
        Pass a WikiManager instance.
        """
        import re
        articles = []
        for path in wiki_manager.list_articles():
            content = path.read_text(encoding="utf-8")
            # Extract frontmatter fields
            title_match = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
            tags_match = re.search(r"^tags:\s*(.+)$", content, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else path.stem
            tags_raw = tags_match.group(1).strip() if tags_match else "[]"
            try:
                tags = json.loads(tags_raw)
            except Exception:
                tags = []
            subdir = path.parent.name
            articles.append({
                "slug": path.stem,
                "title": title,
                "topics": tags,
                "subdir": subdir,
            })
        return articles
