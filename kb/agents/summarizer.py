"""
SummarizerAgent: produces structured summaries from document chunks.
"""
from __future__ import annotations

from typing import Any

from kb.agents.base import AgentInput, BaseAgent

SYSTEM_PROMPT = """\
You are a knowledge base curator specializing in producing clear, accurate, \
structured summaries of technical and research documents.

RULES:
- Be factual and concise. No hallucinations.
- Extract only information present in the provided text.
- Output ONLY valid JSON. No preamble, no markdown code fences, no explanation.
- Confidence score reflects how complete and clear the source text is (0.0-1.0).\
"""

USER_PROMPT = """\
Summarize the following document. Identify its key points, topics, and type.

DOCUMENT:
---
{text}
---

Return EXACTLY this JSON structure (no other text):
{{
  "title": "Descriptive title for this document",
  "summary": "2-4 sentence summary of the main content",
  "key_points": ["point 1", "point 2", "point 3"],
  "topics": ["topic1", "topic2"],
  "document_type": "article|paper|tutorial|reference|dataset|code|other",
  "audience": "beginner|intermediate|expert|general",
  "confidence": 0.85
}}\
"""


class SummarizerAgent(BaseAgent):
    """Produces a structured summary of a parsed document."""

    name = "summarizer"

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        text = self._truncate(inp.content, max_chars=5000)
        prompt = USER_PROMPT.format(text=text)
        return self._call_llm_json(prompt, SYSTEM_PROMPT)

    def validate(self, result: dict[str, Any]) -> bool:
        required = {"title", "summary", "key_points", "confidence"}
        return required.issubset(result.keys()) and isinstance(result["key_points"], list)
