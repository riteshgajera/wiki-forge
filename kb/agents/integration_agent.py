"""
IntegrationAgent: the core of the Karpathy pattern.

When a new source is ingested, this agent does NOT just create a new page.
It reads the existing wiki, finds relevant entity and concept pages,
and UPDATES them with information from the new source.

A single source ingest typically touches 10-15 existing pages.
"""
from __future__ import annotations

import json
from typing import Any

from kb.agents.base import AgentInput, BaseAgent

SYSTEM_PROMPT = """\
You are a wiki integration specialist. Your job is to identify which existing \
wiki pages need to be updated when a new source is ingested, and what changes \
to make to each one.

The goal: knowledge accumulates in the wiki. Entity pages grow richer. \
Concept pages gain new examples and references. Contradictions are flagged. \
Cross-references are added.

Output ONLY valid JSON. No preamble.\
"""

INTEGRATION_PROMPT = """\
A new source has been ingested. Determine which existing wiki pages need updating.

NEW SOURCE SUMMARY:
{summary}

NEW CONCEPTS EXTRACTED:
{concepts}

EXISTING WIKI INDEX (L0 summaries):
{index_content}

For each existing page that the new source meaningfully touches, specify the update.
Types of updates:
- add_reference: add this source as a reference to an existing page
- update_definition: the source provides a better/richer definition
- add_example: the source provides a concrete example
- flag_contradiction: the source contradicts something on the page
- add_entity: the source introduces a new entity that should have a page
- strengthen_link: add a [[wikilink]] to connect this source's summary to the page

Return EXACTLY this JSON (no other text):
{{
  "page_updates": [
    {{
      "slug": "existing-page-slug",
      "subdir": "concepts|entities|summaries",
      "update_type": "add_reference|update_definition|add_example|flag_contradiction|strengthen_link",
      "description": "What specifically to add or change",
      "content_snippet": "The exact markdown snippet to append or insert"
    }}
  ],
  "new_entity_pages": [
    {{
      "name": "Entity Name",
      "type": "person|organization|model|dataset|tool|place",
      "definition": "One sentence definition",
      "slug": "entity-slug",
      "first_source": "source summary slug"
    }}
  ],
  "contradictions": [
    {{
      "existing_slug": "page-slug",
      "claim_in_wiki": "What the wiki currently says",
      "claim_in_source": "What the new source says",
      "severity": "minor|major"
    }}
  ],
  "confidence": 0.85
}}\
"""


class IntegrationAgent(BaseAgent):
    """
    Integrates a new source into the existing wiki.
    Core of the Karpathy compounding pattern.
    """

    name = "integration"

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        summary = inp.get_prior("summarizer")
        concepts = inp.get_prior("concept_extractor")
        index_content = inp.metadata.get("wiki_index_content", "")

        if not index_content:
            # No existing wiki — skip integration, nothing to update
            return {
                "page_updates": [],
                "new_entity_pages": [],
                "contradictions": [],
                "confidence": 1.0,
            }

        # Truncate index to fit context
        index_snippet = index_content[:4000]
        concepts_str = json.dumps(concepts.get("concepts", [])[:10], indent=2)
        summary_str = json.dumps({
            "title": summary.get("title", ""),
            "summary": summary.get("summary", ""),
            "key_points": summary.get("key_points", []),
            "topics": summary.get("topics", []),
        }, indent=2)

        prompt = INTEGRATION_PROMPT.format(
            summary=summary_str,
            concepts=concepts_str,
            index_content=index_snippet,
        )
        return self._call_llm_json(prompt, SYSTEM_PROMPT)

    def validate(self, result: dict[str, Any]) -> bool:
        return (
            "page_updates" in result
            and isinstance(result["page_updates"], list)
            and "confidence" in result
        )


class ContradictionDetectorAgent(BaseAgent):
    """
    Detects when a new source contradicts existing wiki claims.
    Runs as part of integration, but can also run standalone during lint.
    """

    name = "contradiction_detector"

    SYSTEM = """\
You are a fact-checker for a knowledge base. Given a new document and existing \
wiki pages, identify genuine contradictions — places where the new source \
makes claims that conflict with what the wiki currently says.

Be precise. Only flag real contradictions, not just different framings.
Output ONLY valid JSON.\
"""

    PROMPT = """\
Identify contradictions between this new source and existing wiki pages.

NEW SOURCE:
{source_text}

EXISTING WIKI PAGES TO CHECK:
{existing_pages}

Return EXACTLY this JSON:
{{
  "contradictions": [
    {{
      "wiki_page": "slug",
      "wiki_claim": "exact text from wiki",
      "source_claim": "what the source says instead",
      "type": "factual|temporal|definitional|methodological",
      "severity": "minor|major",
      "resolution_suggestion": "How to reconcile"
    }}
  ],
  "consistency_score": 0.95,
  "confidence": 0.88
}}\
"""

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        source_text = self._truncate(inp.content, 3000)
        existing_pages = inp.metadata.get("relevant_page_contents", "")
        if not existing_pages:
            return {"contradictions": [], "consistency_score": 1.0, "confidence": 1.0}

        prompt = self.PROMPT.format(
            source_text=source_text,
            existing_pages=self._truncate(existing_pages, 3000),
        )
        return self._call_llm_json(prompt, self.SYSTEM)

    def validate(self, result: dict[str, Any]) -> bool:
        return "contradictions" in result and "confidence" in result
