"""
ResearchAgent: identifies knowledge gaps and suggests enrichment sources.
Runs after indexing to find missing concept articles.
"""
from __future__ import annotations

import json
from typing import Any

from kb.agents.base import AgentInput, BaseAgent

SYSTEM_PROMPT = """\
You are a research librarian analyzing a knowledge base for gaps and improvement opportunities.

Your goal is to identify:
1. Concepts that are referenced (linked) but have no article
2. Topics that should exist given the domain but are missing
3. Articles that need more depth or external references

Output ONLY valid JSON. No preamble.\
"""

USER_PROMPT = """\
Analyze this knowledge base for gaps and enrichment opportunities.

EXISTING ARTICLES:
{existing_articles}

CONCEPTS REFERENCED BUT NOT YET DOCUMENTED:
{orphan_links}

PRIMARY DOMAIN: {domain}

Return EXACTLY this JSON:
{{
  "gaps": [
    {{
      "concept": "Missing Concept Name",
      "priority": "high|medium|low",
      "reason": "Why this is needed",
      "suggested_sources": ["search query or URL"]
    }}
  ],
  "enrichment_suggestions": [
    {{
      "article_slug": "existing-article",
      "suggestion": "What to add or improve"
    }}
  ],
  "domain_summary": "Brief assessment of coverage",
  "coverage_score": 0.7,
  "confidence": 0.85
}}\
"""


class ResearchAgent(BaseAgent):
    """Identifies knowledge gaps and suggests enrichment opportunities."""

    name = "research"

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        existing = inp.metadata.get("existing_articles", [])
        orphan_links = inp.metadata.get("orphan_links", [])
        domain = inp.metadata.get("primary_domain", "general")

        existing_str = json.dumps(existing[:50], indent=2)
        orphan_str = json.dumps(orphan_links[:30], indent=2)

        prompt = USER_PROMPT.format(
            existing_articles=existing_str,
            orphan_links=orphan_str,
            domain=domain,
        )
        return self._call_llm_json(prompt, SYSTEM_PROMPT)

    def validate(self, result: dict[str, Any]) -> bool:
        return "gaps" in result and "confidence" in result
