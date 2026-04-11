"""
LinkerAgent: generates [[wikilinks]] and backlinks between wiki articles.
"""
from __future__ import annotations

import json
from typing import Any

from kb.agents.base import AgentInput, BaseAgent

SYSTEM_PROMPT = """\
You are a wiki editor specializing in creating well-connected knowledge graphs. \
Your task is to identify where [[wikilinks]] should be inserted into a draft article \
and which existing wiki articles should link back to this one.

RULES:
- Only suggest links to concepts that genuinely appear in the draft article.
- Each term should be linked only on its FIRST occurrence.
- target_file should be a snake_case slug (e.g., "transformer_architecture").
- Output ONLY valid JSON. No preamble, no explanation.\
"""

USER_PROMPT = """\
Given this draft article and the existing concept index, generate wikilinks and backlinks.

ARTICLE SLUG: {slug}
ARTICLE TITLE: {title}

DRAFT ARTICLE:
---
{article}
---

EXISTING WIKI CONCEPTS (slug -> title):
{concept_index}

Return EXACTLY this JSON (no other text):
{{
  "wikilinks": [
    {{
      "term": "original term in article",
      "target_file": "target_slug",
      "context_snippet": "short phrase around the term for verification"
    }}
  ],
  "suggested_backlinks": [
    {{
      "from_file": "existing_slug",
      "reason": "why this article should link here"
    }}
  ],
  "new_concepts_needed": ["concept that should have its own article"],
  "confidence": 0.82
}}\
"""


class LinkerAgent(BaseAgent):
    """Generates [[wikilinks]] and cross-references for a wiki article draft."""

    name = "linker"

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        from kb.utils.helpers import slugify

        summary = inp.get_prior("summarizer")
        concepts = inp.get_prior("concept_extractor")

        title = summary.get("title", inp.metadata.get("rel_path", ""))
        slug = inp.metadata.get("slug", slugify(title))
        article = inp.metadata.get("article_draft", inp.content)

        # Build concept index from wiki (injected via metadata) or prior extraction
        concept_index: dict[str, str] = inp.metadata.get("concept_index", {})
        if not concept_index:
            # Fall back to concepts extracted in this session
            for c in concepts.get("concepts", []):
                from kb.utils.helpers import slugify as sl
                concept_index[sl(c["name"])] = c["name"]

        index_str = json.dumps(concept_index, indent=2)[:3000]
        article_text = self._truncate(article, max_chars=3000)

        prompt = USER_PROMPT.format(
            slug=slug,
            title=title,
            article=article_text,
            concept_index=index_str,
        )
        return self._call_llm_json(prompt, SYSTEM_PROMPT)

    def validate(self, result: dict[str, Any]) -> bool:
        return (
            "wikilinks" in result
            and isinstance(result["wikilinks"], list)
            and "confidence" in result
        )

    @staticmethod
    def apply_wikilinks(article: str, wikilinks: list[dict[str, str]]) -> str:
        """Apply wikilink suggestions to an article string."""
        for link in wikilinks:
            term = link.get("term", "")
            target = link.get("target_file", "")
            if not term or not target:
                continue
            # Replace first occurrence only
            linked = f"[[{target}|{term}]]" if target != term else f"[[{term}]]"
            article = article.replace(term, linked, 1)
        return article
