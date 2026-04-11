"""OpenAI provider implementation."""
from __future__ import annotations

import openai

from kb.services.llm.base import EmbedResponse, LLMProvider, LLMResponse
from kb.utils.logging import get_logger

logger = get_logger("llm.openai")


class OpenAIProvider(LLMProvider):
    """LLM provider backed by OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> None:
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.embed_model = embed_model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug("openai_complete", model=self.model, prompt_len=len(prompt))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=self.model,
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
            provider=self.provider_name,
        )

    def embed(self, text: str) -> EmbedResponse:
        resp = self.client.embeddings.create(
            model=self.embed_model,
            input=text[:8191],  # token limit
        )
        return EmbedResponse(
            embedding=resp.data[0].embedding,
            model=self.embed_model,
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
        )
