"""
LintingQAAgent: validates wiki articles for quality, consistency, and completeness.
Produces a confidence score and actionable improvement suggestions.
"""
from __future__ import annotations

import re
from typing import Any

from kb.agents.base import AgentInput, BaseAgent

SYSTEM_PROMPT = """\
You are a wiki quality assurance reviewer. Your job is to evaluate knowledge base \
articles for accuracy, clarity, structure, and completeness.

EVALUATION CRITERIA:
1. Has a clear title and summary
2. Uses [[wikilinks]] to connect related concepts
3. Avoids vague or unsupported claims
4. Has logical structure (intro, body, conclusion or key points)
5. No broken links (links should match known concept slugs)
6. Appropriate length (200-1500 words for concept articles)
7. Has metadata frontmatter (title, tags, source)

Output ONLY valid JSON. No preamble.\
"""

USER_PROMPT = """\
Review this wiki article for quality.

KNOWN CONCEPT SLUGS (for link validation):
{known_slugs}

ARTICLE:
---
{article}
---

Return EXACTLY this JSON:
{{
  "score": 0.85,
  "title_ok": true,
  "has_summary": true,
  "has_wikilinks": true,
  "has_frontmatter": true,
  "word_count": 450,
  "issues": [
    {{
      "type": "missing_link|broken_link|unclear_claim|too_short|too_long|missing_summary|style",
      "description": "Specific description of the issue",
      "severity": "low|medium|high",
      "line_hint": "Optional: quote of the problematic text"
    }}
  ],
  "suggestions": ["Suggestion 1", "Suggestion 2"],
  "approved": true,
  "confidence": 0.9
}}\
"""


class LintingQAAgent(BaseAgent):
    """Reviews wiki articles and produces a quality score with actionable feedback."""

    name = "linting_qa"

    def _execute(self, inp: AgentInput) -> dict[str, Any]:
        article = self._truncate(inp.content, max_chars=4000)
        known_slugs = inp.metadata.get("known_slugs", [])
        slugs_str = ", ".join(known_slugs[:100]) if known_slugs else "none provided"

        prompt = USER_PROMPT.format(article=article, known_slugs=slugs_str)
        result = self._call_llm_json(prompt, SYSTEM_PROMPT)

        # Supplement with rule-based checks (fast, no LLM needed)
        rule_issues = self._rule_based_checks(inp.content)
        result.setdefault("issues", [])
        result["issues"].extend(rule_issues)

        # Recalculate score if rule issues are high severity
        high_sev = sum(1 for i in result["issues"] if i.get("severity") == "high")
        if high_sev > 0:
            result["score"] = max(0.0, float(result.get("score", 0.8)) - 0.1 * high_sev)
            result["approved"] = result["score"] >= 0.7

        return result

    def _rule_based_checks(self, content: str) -> list[dict[str, str]]:
        """Fast rule-based checks that don't require LLM."""
        issues = []
        word_count = len(content.split())

        if word_count < 100:
            issues.append({
                "type": "too_short",
                "description": f"Article is only {word_count} words. Minimum 100 recommended.",
                "severity": "high",
            })
        if word_count > 2000:
            issues.append({
                "type": "too_long",
                "description": f"Article is {word_count} words. Consider splitting at 1500.",
                "severity": "low",
            })
        if not re.search(r"^---\n", content):
            issues.append({
                "type": "missing_summary",
                "description": "Missing YAML frontmatter block.",
                "severity": "medium",
            })
        if not re.search(r"\[\[.+\]\]", content):
            issues.append({
                "type": "missing_link",
                "description": "Article has no [[wikilinks]]. Add links to related concepts.",
                "severity": "medium",
            })
        return issues

    def validate(self, result: dict[str, Any]) -> bool:
        return (
            "score" in result
            and "issues" in result
            and "approved" in result
            and "confidence" in result
        )

    @staticmethod
    def format_report(output: AgentOutput) -> str:  # type: ignore[name-defined]
        """Format a human-readable lint report."""
        r = output.result
        score = r.get("score", 0.0)
        approved = r.get("approved", False)
        status = "✅ APPROVED" if approved else "❌ NEEDS REVIEW"

        lines = [
            f"# Lint Report",
            f"**Status**: {status}  |  **Score**: {score:.2f}  |  **Words**: {r.get('word_count', '?')}",
            "",
        ]
        issues = r.get("issues", [])
        if issues:
            lines.append("## Issues")
            for iss in issues:
                sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(iss.get("severity", "low"), "⚪")
                lines.append(f"- {sev_icon} **{iss.get('type', '?')}**: {iss.get('description', '')}")

        suggestions = r.get("suggestions", [])
        if suggestions:
            lines.append("\n## Suggestions")
            for s in suggestions:
                lines.append(f"- {s}")

        return "\n".join(lines)
