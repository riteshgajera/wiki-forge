"""Ollama local LLM provider."""
from __future__ import annotations

import json

import requests

from kb.services.llm.base import EmbedResponse, LLMProvider, LLMResponse
from kb.utils.logging import get_logger

logger = get_logger("llm.ollama")


class OllamaProvider(LLMProvider):
    """LLM provider backed by local Ollama server."""

    def __init__(
        self,
        model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.embed_model = embed_model
        self.base_url = base_url.rstrip("/")
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.default_temperature,
                "num_predict": max_tokens or self.default_max_tokens,
            },
        }
        if system:
            payload["system"] = system

        logger.debug("ollama_complete", model=self.model, prompt_len=len(prompt))
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            content=data.get("response", ""),
            model=self.model,
            tokens_used=data.get("eval_count", 0),
            provider=self.provider_name,
        )

    def embed(self, text: str) -> EmbedResponse:
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text[:4096]},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return EmbedResponse(
            embedding=data.get("embedding", []),
            model=self.embed_model,
        )

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
