"""Unit tests for agents using a mock LLM provider."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from kb.agents.base import AgentInput, AgentOutput, BaseAgent
from kb.services.llm.base import EmbedResponse, LLMProvider, LLMResponse


# ─── Mock LLM ─────────────────────────────────────────────────────────────────

class MockLLMProvider(LLMProvider):
    """LLM provider that returns preset responses for testing."""

    def __init__(self, response_data: dict | None = None) -> None:
        self._response = response_data or {"confidence": 0.9}

    @property
    def provider_name(self) -> str:
        return "mock"

    def complete(self, prompt, system="", temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(self._response),
            model="mock-model",
            tokens_used=100,
        )

    def embed(self, text: str) -> EmbedResponse:
        # Return a deterministic 4-dim embedding for testing
        return EmbedResponse(
            embedding=[0.1, 0.2, 0.3, 0.4],
            model="mock-embed",
        )


class FailingLLMProvider(LLMProvider):
    """LLM that always raises an exception."""

    @property
    def provider_name(self) -> str:
        return "failing"

    def complete(self, *args, **kwargs) -> LLMResponse:
        raise ConnectionError("LLM unavailable")

    def embed(self, text: str) -> EmbedResponse:
        raise ConnectionError("LLM unavailable")


# ─── BaseAgent ────────────────────────────────────────────────────────────────

class ConcreteAgent(BaseAgent):
    name = "concrete"

    def _execute(self, inp: AgentInput) -> dict:
        return self._call_llm_json("prompt", "system")


class TestBaseAgent:
    def test_run_success(self):
        llm = MockLLMProvider({"summary": "ok", "confidence": 0.9})
        agent = ConcreteAgent(llm)
        inp = AgentInput(doc_id="doc1", content="test content")

        # Override validate to accept any dict with confidence
        output = agent.run(inp)
        assert output.success
        assert output.confidence == pytest.approx(0.9)
        assert output.approved  # 0.9 >= 0.7 threshold

    def test_run_retries_on_failure(self):
        llm = FailingLLMProvider()
        agent = ConcreteAgent(llm)
        agent.max_retries = 2  # Reduce retries for speed

        # Monkey-patch sleep to avoid waiting
        import kb.agents.base as base_mod
        original_sleep = __import__("time").sleep
        __import__("time").sleep = lambda x: None

        try:
            inp = AgentInput(doc_id="doc2", content="content")
            output = agent.run(inp)
            assert not output.success
            assert output.error is not None
        finally:
            __import__("time").sleep = original_sleep

    def test_low_confidence_not_approved(self):
        llm = MockLLMProvider({"confidence": 0.3})
        agent = ConcreteAgent(llm)
        inp = AgentInput(doc_id="doc3", content="content")
        output = agent.run(inp)
        assert not output.approved
        assert output.needs_review

    def test_truncate(self):
        llm = MockLLMProvider()
        agent = ConcreteAgent(llm)
        long_text = "x" * 10000
        result = agent._truncate(long_text, max_chars=1000)
        assert len(result) <= 1100  # truncated + suffix
        assert "truncated" in result

    def test_agent_input_get_prior(self):
        inp = AgentInput(
            doc_id="d1",
            content="c",
            metadata={"prior_results": {"summarizer": {"summary": "hello"}}},
        )
        assert inp.get_prior("summarizer") == {"summary": "hello"}
        assert inp.get_prior("nonexistent") == {}


# ─── SummarizerAgent ──────────────────────────────────────────────────────────

class TestSummarizerAgent:
    @pytest.fixture
    def agent(self):
        from kb.agents.summarizer import SummarizerAgent
        llm = MockLLMProvider({
            "title": "Test Document",
            "summary": "This is a test summary.",
            "key_points": ["Point 1", "Point 2"],
            "topics": ["testing"],
            "document_type": "article",
            "audience": "general",
            "confidence": 0.88,
        })
        return SummarizerAgent(llm)

    def test_summarizer_output_structure(self, agent):
        inp = AgentInput(doc_id="s1", content="Some document content here.")
        output = agent.run(inp)
        assert output.success
        assert "title" in output.result
        assert "summary" in output.result
        assert "key_points" in output.result
        assert isinstance(output.result["key_points"], list)

    def test_summarizer_validates_correctly(self, agent):
        valid = {"title": "T", "summary": "S", "key_points": ["p"], "confidence": 0.9}
        assert agent.validate(valid)

    def test_summarizer_rejects_missing_fields(self, agent):
        invalid = {"title": "T", "confidence": 0.9}  # missing summary and key_points
        assert not agent.validate(invalid)


# ─── ConceptExtractorAgent ────────────────────────────────────────────────────

class TestConceptExtractorAgent:
    @pytest.fixture
    def agent(self):
        from kb.agents.concept_extractor import ConceptExtractorAgent
        llm = MockLLMProvider({
            "concepts": [
                {"name": "Transformer", "type": "model", "definition": "A neural architecture.",
                 "aliases": ["transformer model"]},
            ],
            "relationships": [{"from": "Transformer", "relation": "uses", "to": "Attention"}],
            "primary_domain": "machine-learning",
            "confidence": 0.85,
        })
        return ConceptExtractorAgent(llm)

    def test_concept_extractor_output(self, agent):
        inp = AgentInput(doc_id="c1", content="Transformers use attention mechanisms.")
        output = agent.run(inp)
        assert output.success
        assert len(output.result["concepts"]) >= 1
        assert output.result["concepts"][0]["name"] == "Transformer"

    def test_concept_names_stripped(self, agent):
        from kb.agents.concept_extractor import ConceptExtractorAgent
        llm = MockLLMProvider({
            "concepts": [{"name": "  Spaced Name  ", "type": "model",
                          "definition": "def", "aliases": []}],
            "relationships": [],
            "primary_domain": "ml",
            "confidence": 0.8,
        })
        agent2 = ConceptExtractorAgent(llm)
        inp = AgentInput(doc_id="c2", content="content")
        output = agent2.run(inp)
        assert output.result["concepts"][0]["name"] == "Spaced Name"


# ─── LinkerAgent ──────────────────────────────────────────────────────────────

class TestLinkerAgent:
    @pytest.fixture
    def agent(self):
        from kb.agents.linker import LinkerAgent
        llm = MockLLMProvider({
            "wikilinks": [
                {"term": "attention", "target_file": "attention-mechanism",
                 "context_snippet": "uses attention to"},
            ],
            "suggested_backlinks": [],
            "new_concepts_needed": [],
            "confidence": 0.8,
        })
        return LinkerAgent(llm)

    def test_linker_output(self, agent):
        inp = AgentInput(
            doc_id="l1",
            content="The model uses attention to process tokens.",
            metadata={"concept_index": {"attention-mechanism": "Attention Mechanism"}},
        )
        output = agent.run(inp)
        assert output.success
        assert "wikilinks" in output.result

    def test_apply_wikilinks(self):
        from kb.agents.linker import LinkerAgent
        article = "The transformer uses attention for processing."
        wikilinks = [{"term": "attention", "target_file": "attention-mechanism"}]
        result = LinkerAgent.apply_wikilinks(article, wikilinks)
        assert "[[attention-mechanism|attention]]" in result

    def test_apply_wikilinks_same_slug(self):
        from kb.agents.linker import LinkerAgent
        article = "Uses transformer architecture."
        wikilinks = [{"term": "transformer", "target_file": "transformer"}]
        result = LinkerAgent.apply_wikilinks(article, wikilinks)
        assert "[[transformer]]" in result


# ─── LintingQAAgent ───────────────────────────────────────────────────────────

class TestLintingQAAgent:
    @pytest.fixture
    def agent(self):
        from kb.agents.linting import LintingQAAgent
        llm = MockLLMProvider({
            "score": 0.85,
            "title_ok": True,
            "has_summary": True,
            "has_wikilinks": True,
            "has_frontmatter": True,
            "word_count": 300,
            "issues": [],
            "suggestions": ["Add more examples"],
            "approved": True,
            "confidence": 0.9,
        })
        return LintingQAAgent(llm)

    def test_lint_passes(self, agent):
        article = (
            "---\ntitle: Test\ntags: []\n---\n\n"
            "# Test Article\n\nThis is a test article about [[transformers]].\n"
            + "word " * 150
        )
        inp = AgentInput(doc_id="lint1", content=article,
                         metadata={"known_slugs": ["transformers"]})
        output = agent.run(inp)
        assert output.success
        assert output.result["approved"]

    def test_lint_detects_short_article(self, agent):
        from kb.agents.linting import LintingQAAgent
        short_content = "---\ntitle: Test\n---\n\nShort."
        issues = agent._rule_based_checks(short_content)
        assert any(i["type"] == "too_short" for i in issues)

    def test_lint_detects_missing_frontmatter(self, agent):
        content = "# No frontmatter\n\nJust a header and text."
        issues = agent._rule_based_checks(content)
        assert any(i["type"] == "missing_summary" for i in issues)

    def test_lint_detects_missing_wikilinks(self, agent):
        content = "---\ntitle: T\n---\n\n" + "word " * 150
        issues = agent._rule_based_checks(content)
        assert any(i["type"] == "missing_link" for i in issues)

    def test_format_report(self, agent):
        inp = AgentInput(doc_id="rep1", content="---\ntitle: T\n---\n\n" + "w " * 200)
        output = agent.run(inp)
        report = agent.format_report(output)
        assert "Lint Report" in report


# ─── Pipeline Engine ──────────────────────────────────────────────────────────

class TestPipelineEngine:
    def test_pipeline_runs_in_order(self):
        from kb.pipelines.engine import Pipeline, PipelineStep

        execution_order = []

        class OrderAgent(BaseAgent):
            def __init__(self, step_name, llm):
                super().__init__(llm)
                self.name = step_name

            def _execute(self, inp):
                execution_order.append(self.name)
                return {"confidence": 0.9}

        llm = MockLLMProvider()
        steps = [
            PipelineStep("step_a", OrderAgent("step_a", llm), depends_on=[]),
            PipelineStep("step_b", OrderAgent("step_b", llm), depends_on=["step_a"]),
            PipelineStep("step_c", OrderAgent("step_c", llm), depends_on=["step_b"]),
        ]
        pipeline = Pipeline(steps)
        result = pipeline.run(AgentInput(doc_id="p1", content="content"))

        assert not result.halted
        assert execution_order.index("step_a") < execution_order.index("step_b")
        assert execution_order.index("step_b") < execution_order.index("step_c")

    def test_pipeline_halts_on_error(self):
        from kb.pipelines.engine import Pipeline, PipelineStep

        class FailAgent(BaseAgent):
            name = "fail"

            def _execute(self, inp):
                raise RuntimeError("Critical failure")

        steps = [
            PipelineStep("fail", FailAgent(FailingLLMProvider()),
                         depends_on=[], confidence_threshold=0.5),
        ]
        pipeline = Pipeline(steps)

        import time
        time.sleep = lambda x: None  # Skip backoff

        result = pipeline.run(AgentInput(doc_id="halt", content="content"))
        assert result.halted

    def test_pipeline_prior_results_injected(self):
        from kb.pipelines.engine import Pipeline, PipelineStep

        received_prior = {}

        class RecorderAgent(BaseAgent):
            name = "recorder"

            def _execute(self, inp):
                received_prior.update(inp.metadata.get("prior_results", {}))
                return {"confidence": 0.9}

        llm = MockLLMProvider({"confidence": 0.9})

        class FirstAgent(BaseAgent):
            name = "first"

            def _execute(self, inp):
                return {"result": "from_first", "confidence": 0.9}

        steps = [
            PipelineStep("first", FirstAgent(llm), depends_on=[]),
            PipelineStep("recorder", RecorderAgent(llm), depends_on=["first"]),
        ]
        pipeline = Pipeline(steps)
        pipeline.run(AgentInput(doc_id="p2", content="content"))
        assert "first" in received_prior
        assert received_prior["first"].get("result") == "from_first"
