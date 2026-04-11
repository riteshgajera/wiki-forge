"""
Configuration management using pydantic-settings.
Supports .env file, environment variables, and YAML config.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM__", extra="ignore")

    provider: str = "ollama"
    model: str = "llama3.2"
    openai_api_key: str = ""
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 2000
    embed_model: str = "nomic-embed-text"


class ProcessingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROCESSING__", extra="ignore")

    chunk_size: int = 1500
    chunk_overlap: int = 200
    confidence_threshold: float = 0.7
    auto_approve_threshold: float = 0.85
    max_retries: int = 3
    batch_size: int = 10


class IngestionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGESTION__", extra="ignore")

    supported_extensions: list[str] = [
        ".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".rst", ".html"
    ]
    watch_interval_seconds: int = 30


class WebConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WEB__", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Sub-configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    # Paths
    raw_dir: str = "./raw"
    wiki_dir: str = "./wiki"
    db_path: str = "./data/kb.db"
    vector_store_path: str = "./data/vectors"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_dir)

    @property
    def wiki_path(self) -> Path:
        return Path(self.wiki_dir)

    @property
    def db_path_obj(self) -> Path:
        return Path(self.db_path)

    @property
    def vector_store_path_obj(self) -> Path:
        return Path(self.vector_store_path)

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for d in [self.raw_path, self.wiki_path,
                  self.db_path_obj.parent, self.vector_store_path_obj]:
            Path(d).mkdir(parents=True, exist_ok=True)


def _load_yaml_config(yaml_path: Path) -> dict[str, Any]:
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    # Load YAML defaults first, then override with env
    yaml_config = _load_yaml_config(Path("config/default.yaml"))
    # Flatten yaml config for env-style override
    settings = Settings(**{
        k: v for k, v in yaml_config.items()
        if k not in ("llm", "paths", "processing", "ingestion", "web", "logging")
    })
    return settings


# Module-level singleton
settings = get_settings()
