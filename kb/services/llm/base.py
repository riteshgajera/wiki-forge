"""Abstract base for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int = 0
    provider: str = ""


@dataclass
class EmbedResponse:
    embedding: list[float]
    model: str
    tokens_used: int = 0


class LLMProvider(ABC):
    """Unified interface for LLM backends."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def embed(self, text: str) -> EmbedResponse: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...
