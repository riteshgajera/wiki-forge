"""
Plugin registry: allows dynamic registration and discovery of agents.
Built-in agents are registered at import time.
External plugins can register via register_agent().
"""
from __future__ import annotations

from typing import Type

from kb.agents.base import BaseAgent
from kb.utils.logging import get_logger

logger = get_logger("tools.plugin_registry")

_registry: dict[str, Type[BaseAgent]] = {}


def register_agent(name: str, cls: Type[BaseAgent]) -> None:
    """Register an agent class under a given name."""
    if name in _registry:
        logger.warning("agent_overwrite", name=name)
    _registry[name] = cls
    logger.debug("agent_registered", name=name, cls=cls.__name__)


def get_agent_class(name: str) -> Type[BaseAgent]:
    """Return agent class by name. Raises KeyError if not found."""
    if name not in _registry:
        raise KeyError(
            f"Agent {name!r} not registered. Available: {list_agents()}"
        )
    return _registry[name]


def list_agents() -> list[str]:
    """Return list of registered agent names."""
    return sorted(_registry.keys())


def _register_builtins() -> None:
    from kb.agents.summarizer import SummarizerAgent
    from kb.agents.concept_extractor import ConceptExtractorAgent
    from kb.agents.linker import LinkerAgent
    from kb.agents.index_builder import IndexBuilderAgent
    from kb.agents.linting import LintingQAAgent
    from kb.agents.research import ResearchAgent

    register_agent("summarizer", SummarizerAgent)
    register_agent("concept_extractor", ConceptExtractorAgent)
    register_agent("linker", LinkerAgent)
    register_agent("index_builder", IndexBuilderAgent)
    register_agent("linting_qa", LintingQAAgent)
    register_agent("research", ResearchAgent)


_register_builtins()
