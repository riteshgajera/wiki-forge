"""
ConceptExtractorAgent: extracts concepts, entities, and relationships from documents.
"""
from __future__ import annotations

import json
from typing import Any

from kb.agents.base import AgentInput, BaseAgent

SYSTEM_PROMPT = """\
You are a knowledge ontology builder. Your task is to extract structured concepts, \
named entities, and semantic relationships from technical documents.

RULES:
- Extract only concepts explicitly present in the text.
- Concepts should be noun phrases (not verbs or sentences).
- Relationships must connect two concepts from your extracted list.
- Output ONLY valid JSON. No preamble, no explanation, no markdown.\
"""

USER_PROMPT = """\
Extract key concepts and relationships from this document.

DOCUMENT SUMMARY: {summary}

FULL TEXT:
---
{text}
---

Return EXACTLY this JSON (no other text):
{{
  "concepts": [
    {{
      "name": "Concept Name",
      "type": "algorithm|model|framework|dataset|method|tool|theory|person|organization|other",
      "definition": "One sentence definition from the text",
      "aliases": ["alt name 1", "alt name 2"]
    }}
  ],
  "relationships": [
    {{
      "from": "Concept A",
      "relation": "is_a|part_of|uses|extends|related_to|created_by|applied_to",
      "to": "Concept B"
    }}
  ],
  "primary_domain": "machine-learning|nlp|computer-vision|systems|databases|other",
  "confidence": 0.8
}}\
"""


class ConceptExtractorAgent(BaseAgent):
    """Extracts concepts, entities, and relationships from a document."""

    name = "concept_extractor"

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        summary = inp.get_prior("summarizer").get("summary", "")
        text = self._truncate(inp.content, max_chars=5000)
        prompt = USER_PROMPT.format(summary=summary, text=text)
        result = self._call_llm_json(prompt, SYSTEM_PROMPT)
        # Normalise concept names (strip whitespace)
        for c in result.get("concepts", []):
            c["name"] = c.get("name", "").strip()
        return result

    def validate(self, result: dict[str, Any]) -> bool:
        return (
            "concepts" in result
            and isinstance(result["concepts"], list)
            and "confidence" in result
        )
