"""Factory for creating LLM providers from config."""
from __future__ import annotations

from functools import lru_cache

from kb.services.llm.base import LLMProvider
from kb.utils.logging import get_logger

logger = get_logger("llm.factory")


def create_provider(
    provider: str = "ollama",
    model: str = "llama3.2",
    openai_api_key: str = "",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.2,
    max_tokens: int = 2000,
    embed_model: str = "nomic-embed-text",
) -> LLMProvider:
    """Instantiate the correct LLM provider from config values."""
    if provider == "openai":
        from kb.services.llm.openai_provider import OpenAIProvider
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY required when provider=openai")
        logger.info("llm_provider_init", provider="openai", model=model)
        return OpenAIProvider(
            api_key=openai_api_key,
            model=model,
            embed_model=embed_model or "text-embedding-3-small",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "ollama":
        from kb.services.llm.ollama_provider import OllamaProvider
        logger.info("llm_provider_init", provider="ollama", model=model, base_url=base_url)
        return OllamaProvider(
            model=model,
            embed_model=embed_model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Choose 'openai' or 'ollama'.")


@lru_cache(maxsize=1)
def get_default_provider() -> LLMProvider:
    """Return a cached provider built from global settings."""
    from kb.utils.config import settings
    cfg = settings.llm
    return create_provider(
        provider=cfg.provider,
        model=cfg.model,
        openai_api_key=cfg.openai_api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        embed_model=cfg.embed_model,
    )
