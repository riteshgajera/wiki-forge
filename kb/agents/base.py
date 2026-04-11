"""
BaseAgent: abstract base class for all knowledge base agents.
Provides retry logic, validation, confidence scoring, and structured logging.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from kb.services.llm.base import LLMProvider
from kb.utils.helpers import safe_json_parse
from kb.utils.logging import get_logger


@dataclass
class AgentInput:
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_prior(self, agent_name: str) -> dict[str, Any]:
        """Retrieve a prior agent's output from metadata."""
        return self.metadata.get("prior_results", {}).get(agent_name, {})


@dataclass
class AgentOutput:
    agent: str
    doc_id: str
    result: dict[str, Any]
    confidence: float
    approved: bool = False
    needs_review: bool = False
    error: str | None = None
    tokens_used: int = 0

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.result)


class BaseAgent(ABC):
    """
    Abstract base for all KB agents.

    Subclasses must implement:
      - name (class attribute)
      - _execute(inp) -> dict
      - validate(result) -> bool  (optional override)
    """

    name: str = "base"
    max_retries: int = 3
    confidence_threshold: float = 0.7

    def __init__(self, llm: LLMProvider, confidence_threshold: float | None = None) -> None:
        self.llm = llm
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        self.logger = get_logger(f"agent.{self.name}")

    def run(self, inp: AgentInput) -> AgentOutput:
        """Run the agent with retry/backoff. Never raises — returns error output."""
        self.logger.info("agent_start", agent=self.name, doc_id=inp.doc_id)
        last_exc: Exception | None = None
        total_tokens = 0

        for attempt in range(self.max_retries):
            try:
                result, tokens = self._execute_with_tokens(inp)

                if not self.validate(result):
                    raise ValueError(f"Validation failed: {list(result.keys())}")

                total_tokens += tokens
                confidence = float(result.get("confidence", 1.0))
                approved = confidence >= self.confidence_threshold
                needs_review = not approved

                self.logger.info(
                    "agent_success",
                    agent=self.name,
                    doc_id=inp.doc_id,
                    confidence=confidence,
                    approved=approved,
                    attempt=attempt + 1,
                )
                return AgentOutput(
                    agent=self.name,
                    doc_id=inp.doc_id,
                    result=result,
                    confidence=confidence,
                    approved=approved,
                    needs_review=needs_review,
                    tokens_used=total_tokens,
                )

            except Exception as e:
                last_exc = e
                self.logger.warning(
                    "agent_retry",
                    agent=self.name,
                    doc_id=inp.doc_id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        self.logger.error(
            "agent_failed",
            agent=self.name,
            doc_id=inp.doc_id,
            error=str(last_exc),
        )
        return AgentOutput(
            agent=self.name,
            doc_id=inp.doc_id,
            result={},
            confidence=0.0,
            approved=False,
            error=str(last_exc),
        )

    def _execute_with_tokens(self, inp: AgentInput) -> tuple[dict, int]:
        """Wrapper that tracks token usage."""
        result = self._execute(inp)
        return result, 0  # Subclasses may override to track tokens

    @abstractmethod
    def _execute(self, inp: AgentInput) -> dict[str, Any]: ...

    def validate(self, result: dict[str, Any]) -> bool:
        """Override to add result-specific validation."""
        return isinstance(result, dict) and "confidence" in result

    def _call_llm_json(
        self,
        prompt: str,
        system: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Call LLM and parse JSON response.
        Raises ValueError if response cannot be parsed as JSON.
        """
        resp = self.llm.complete(prompt, system=system, temperature=temperature)
        parsed = safe_json_parse(resp.content)
        if parsed is None:
            raise ValueError(
                f"LLM returned non-JSON response (first 200 chars): {resp.content[:200]}"
            )
        return parsed

    def _truncate(self, text: str, max_chars: int = 6000) -> str:
        """Truncate text to avoid context length issues."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n\n[... truncated {len(text)-max_chars} chars ...]"
