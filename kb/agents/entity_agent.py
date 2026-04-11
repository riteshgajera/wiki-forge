"""
EntityAgent: creates and maintains entity pages.

Entity pages are the persistent records for named entities —
people, organizations, models, datasets, tools.
Each entity has exactly ONE page that accumulates all mentions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kb.agents.base import AgentInput, BaseAgent
from kb.utils.helpers import slugify

SYSTEM_PROMPT = """\
You are a wiki entity manager. Entity pages accumulate information about \
named entities (people, organizations, models, datasets, tools) across \
all ingested sources. You write clean, factual entity pages in Obsidian markdown.

Output ONLY valid JSON. No preamble.\
"""

ENTITY_PAGE_PROMPT = """\
Create or update an entity page for this entity.

ENTITY INFO:
{entity_info}

EXISTING PAGE CONTENT (empty if new):
{existing_content}

SOURCE CONTEXT:
{source_context}

Return EXACTLY this JSON:
{{
  "title": "Entity Name",
  "type": "person|organization|model|dataset|tool|place",
  "aliases": ["alt name", "abbreviation"],
  "l0_summary": "One sentence description for the index",
  "definition": "2-3 sentence description",
  "key_facts": ["fact 1", "fact 2"],
  "relationships": [
    {{"relation": "created_by|part_of|successor_to|related_to", "entity": "Other Entity"}}
  ],
  "appearances": ["[[summaries/source-slug|Source Title]]"],
  "tags": ["tag1", "tag2"],
  "confidence": 0.9
}}\
"""


class EntityAgent(BaseAgent):
    """Creates and updates entity pages in /wiki/entities/."""

    name = "entity"

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        entity_info = inp.metadata.get("entity_info", {})
        existing_content = inp.metadata.get("existing_content", "")
        source_context = inp.get_prior("summarizer").get("summary", inp.content[:500])

        prompt = ENTITY_PAGE_PROMPT.format(
            entity_info=json.dumps(entity_info, indent=2),
            existing_content=self._truncate(existing_content, 1500),
            source_context=source_context[:800],
        )
        return self._call_llm_json(prompt, SYSTEM_PROMPT)

    def validate(self, result: dict[str, Any]) -> bool:
        return "title" in result and "definition" in result and "confidence" in result

    @staticmethod
    def render_entity_page(result: dict[str, Any], source_slug: str = "") -> str:
        """Render the entity page markdown from agent output."""
        now = datetime.now(timezone.utc).isoformat()
        title = result.get("title", "")
        etype = result.get("type", "other")
        aliases = result.get("aliases", [])
        l0 = result.get("l0_summary", result.get("definition", "")[:120])
        definition = result.get("definition", "")
        key_facts = result.get("key_facts", [])
        relationships = result.get("relationships", [])
        appearances = result.get("appearances", [])
        if source_slug and f"[[summaries/{source_slug}]]" not in str(appearances):
            appearances.append(f"[[summaries/{source_slug}|{source_slug}]]")
        tags = result.get("tags", [etype])
        confidence = result.get("confidence", 0.9)

        frontmatter = (
            f"---\n"
            f"title: \"{title}\"\n"
            f"type: entity\n"
            f"entity_type: {etype}\n"
            f"aliases: {json.dumps(aliases)}\n"
            f"l0_summary: \"{l0}\"\n"
            f"tags: {json.dumps(tags)}\n"
            f"created: {now}\n"
            f"updated: {now}\n"
            f"confidence: {confidence}\n"
            f"depth: L1\n"
            f"status: active\n"
            f"---\n\n"
        )

        body_lines = [f"# {title}\n", f"{definition}\n"]

        if key_facts:
            body_lines.append("\n## Key Facts\n")
            for f in key_facts:
                body_lines.append(f"- {f}")

        if relationships:
            body_lines.append("\n## Relationships\n")
            for r in relationships:
                entity = r.get("entity", "")
                relation = r.get("relation", "related_to")
                slug = slugify(entity)
                body_lines.append(f"- *{relation}* [[entities/{slug}|{entity}]]")

        if appearances:
            body_lines.append("\n## Appearances\n")
            for a in appearances:
                body_lines.append(f"- {a}")

        return frontmatter + "\n".join(body_lines)
