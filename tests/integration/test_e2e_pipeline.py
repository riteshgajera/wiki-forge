"""
Integration tests: end-to-end pipeline with mock LLM.
No real LLM calls — uses MockLLMProvider throughout.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kb.services.llm.base import EmbedResponse, LLMProvider, LLMResponse


class MockLLM(LLMProvider):
    """Smart mock that routes on system prompt keywords."""

    @property
    def provider_name(self) -> str:
        return "mock"

    def complete(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        s = system.lower()
        if "ontology" in s or "ontology builder" in s:
            data = {
                "concepts": [
                    {"name": "Attention Mechanism", "type": "method",
                     "definition": "Weighs importance of tokens.", "aliases": ["attention"]},
                    {"name": "Transformer", "type": "model",
                     "definition": "Architecture using attention.", "aliases": []},
                ],
                "relationships": [{"from": "Transformer", "relation": "uses", "to": "Attention Mechanism"}],
                "primary_domain": "machine-learning",
                "confidence": 0.85,
            }
        elif "wiki editor" in s or "wikilink" in s or "connected" in s:
            data = {
                "wikilinks": [{"term": "attention", "target_file": "attention-mechanism",
                                "context_snippet": "uses attention to"}],
                "suggested_backlinks": [],
                "new_concepts_needed": ["Self-Attention"],
                "confidence": 0.80,
            }
        elif "quality" in s or "review" in s or "lint" in s or "assurance" in s:
            data = {
                "score": 0.82, "title_ok": True, "has_summary": True,
                "has_wikilinks": True, "has_frontmatter": True, "word_count": 250,
                "issues": [], "suggestions": ["Add references"], "approved": True, "confidence": 0.9,
            }
        elif "information architect" in s or "index" in s:
            data = {
                "sections": [{"title": "Machine Learning", "description": "ML articles",
                               "articles": ["attention-mechanism-in-transformers"]}],
                "confidence": 0.95,
            }
        elif "curator" in s or "summarize" in s:
            data = {
                "title": "Attention Mechanism in Transformers",
                "summary": "This document explains the attention mechanism.",
                "key_points": ["Attention weighs token importance", "Multi-head in parallel"],
                "topics": ["machine-learning", "transformers"],
                "document_type": "article",
                "audience": "intermediate",
                "confidence": 0.88,
            }
        else:  # fallback
            data = {
                "title": "Attention Mechanism in Transformers",
                "summary": "This document explains the attention mechanism.",
                "key_points": ["Attention weighs token importance", "Multi-head attention runs in parallel"],
                "topics": ["machine-learning", "transformers", "attention"],
                "document_type": "article",
                "audience": "intermediate",
                "confidence": 0.88,
            }
        return LLMResponse(content=json.dumps(data), model="mock", tokens_used=50)

    def embed(self, text: str) -> EmbedResponse:
        import hashlib, struct
        h = hashlib.md5(text.encode()).digest()
        floats = [struct.unpack("f", h[i:i+4])[0] for i in range(0, 16, 4)]
        norm = sum(f**2 for f in floats) ** 0.5 or 1.0
        return EmbedResponse(embedding=[f/norm for f in floats], model="mock-embed")

# ─── End-to-End Pipeline Test ─────────────────────────────────────────────────

class TestEndToEndPipeline:
    @pytest.fixture
    def setup_env(self, tmp_path):
        """Create temp raw + wiki directories with sample documents."""
        raw_dir = tmp_path / "raw"
        wiki_dir = tmp_path / "wiki"
        db_path = tmp_path / "data" / "kb.db"
        raw_dir.mkdir()
        wiki_dir.mkdir()

        # Write sample documents
        (raw_dir / "attention.md").write_text(
            "# Attention Mechanism\n\n"
            "The attention mechanism allows neural networks to focus on relevant parts "
            "of the input sequence. It computes weighted sums of value vectors, where "
            "weights are determined by compatibility between query and key vectors.\n\n"
            "Transformers use multi-head attention to process tokens in parallel, "
            "making them highly efficient for sequence-to-sequence tasks.\n"
        )
        (raw_dir / "transformers.md").write_text(
            "# Transformer Architecture\n\n"
            "The Transformer is a deep learning model introduced in 'Attention is All You Need'. "
            "It relies entirely on attention mechanisms, dispensing with recurrence entirely. "
            "The encoder-decoder structure processes sequences of arbitrary length.\n"
        )

        return tmp_path, raw_dir, wiki_dir, db_path

    def test_ingestion_scans_files(self, setup_env):
        tmp_path, raw_dir, wiki_dir, db_path = setup_env
        from kb.ingestion.ingestion_engine import IngestionEngine
        from kb.utils.config import Settings

        cfg = Settings(raw_dir=str(raw_dir), wiki_dir=str(wiki_dir), db_path=str(db_path))
        engine = IngestionEngine(cfg)
        queued = engine.scan(raw_dir)

        assert len(queued) == 2
        paths = {r.path for r in queued}
        assert "attention.md" in paths
        assert "transformers.md" in paths

    def test_ingestion_skips_unchanged(self, setup_env):
        tmp_path, raw_dir, wiki_dir, db_path = setup_env
        from kb.ingestion.ingestion_engine import IngestionEngine
        from kb.utils.config import Settings

        cfg = Settings(raw_dir=str(raw_dir), wiki_dir=str(wiki_dir), db_path=str(db_path))
        engine = IngestionEngine(cfg)

        first_scan = engine.scan(raw_dir)
        assert len(first_scan) == 2

        # Mark all as done
        for rec in first_scan:
            engine.mark_done(rec.id, "wiki/test.md", 0.9)

        second_scan = engine.scan(raw_dir)
        assert len(second_scan) == 0  # Nothing new

    def test_full_pipeline_single_document(self, setup_env):
        tmp_path, raw_dir, wiki_dir, db_path = setup_env
        from kb.agents.base import AgentInput
        from kb.agents.summarizer import SummarizerAgent
        from kb.agents.concept_extractor import ConceptExtractorAgent
        from kb.agents.linker import LinkerAgent
        from kb.agents.linting import LintingQAAgent
        from kb.pipelines.engine import Pipeline, PipelineStep

        llm = MockLLM()
        steps = [
            PipelineStep("summarizer", SummarizerAgent(llm), depends_on=[]),
            PipelineStep("concept_extractor", ConceptExtractorAgent(llm),
                         depends_on=["summarizer"]),
            PipelineStep("linker", LinkerAgent(llm),
                         depends_on=["summarizer", "concept_extractor"]),
            PipelineStep("linting_qa", LintingQAAgent(llm),
                         depends_on=["linker"]),
        ]
        pipeline = Pipeline(steps, auto_approve_threshold=0.7)

        content = (raw_dir / "attention.md").read_text()
        inp = AgentInput(doc_id="test-doc-001", content=content,
                         metadata={"rel_path": "attention.md"})

        result = pipeline.run(inp)

        assert not result.halted
        assert "summarizer" in result.outputs
        assert "concept_extractor" in result.outputs
        assert "linker" in result.outputs
        assert "linting_qa" in result.outputs
        assert result.overall_confidence > 0

        # Verify summarizer extracted title
        summary = result.get_output("summarizer")
        assert "title" in summary
        assert "key_points" in summary

        # Verify concept extractor found concepts
        concepts = result.get_output("concept_extractor")
        assert len(concepts["concepts"]) >= 1

    def test_wiki_written_after_pipeline(self, setup_env):
        tmp_path, raw_dir, wiki_dir, db_path = setup_env
        from kb.storage.wiki_manager import WikiManager

        wiki = WikiManager(wiki_dir)
        path = wiki.write_article(
            title="Attention Mechanism",
            content="## Summary\nCore concept in transformers.",
            tags=["ml", "transformers"],
            confidence=0.88,
        )

        assert path.exists()
        content = path.read_text()
        assert "Attention Mechanism" in content
        assert "ml" in content
        assert "0.88" in content

    def test_metadata_store_tracks_state(self, setup_env):
        tmp_path, raw_dir, wiki_dir, db_path = setup_env
        from kb.storage.metadata_store import DocumentRecord, MetadataStore

        store = MetadataStore(db_path)
        rec = DocumentRecord(id="int001", path="attention.md", status="pending")
        store.upsert(rec)
        store.update_status("int001", "processing")
        store.update_agent_output("int001", "summarizer", {"title": "Attention"}, 0.88)
        store.update_status("int001", "done")

        final = store.get("int001")
        assert final.status == "done"
        assert "summarizer" in final.agent_outputs

    def test_index_builder_generates_toc(self, setup_env):
        tmp_path, raw_dir, wiki_dir, db_path = setup_env
        from kb.agents.index_builder import IndexBuilderAgent
        from kb.agents.base import AgentInput
        from kb.storage.wiki_manager import WikiManager

        wiki = WikiManager(wiki_dir)
        wiki.write_article("Attention Mechanism", "Content here.", tags=["ml"])
        wiki.write_article("Transformer Architecture", "More content.", tags=["ml"])

        llm = MockLLM()
        agent = IndexBuilderAgent(llm)
        articles = IndexBuilderAgent.build_article_list(wiki)
        assert len(articles) >= 2

        inp = AgentInput(doc_id="__index__", content="",
                         metadata={"wiki_articles": articles})
        output = agent.run(inp)
        assert output.success
        toc = output.result.get("toc_content", "")
        assert "Knowledge Base Index" in toc


# ─── LLM Provider Tests ───────────────────────────────────────────────────────

class TestLLMFactory:
    def test_factory_raises_on_unknown_provider(self):
        from kb.services.llm.factory import create_provider
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider(provider="unknown_provider")

    def test_factory_raises_openai_without_key(self):
        from kb.services.llm.factory import create_provider
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_provider(provider="openai", openai_api_key="")

    def test_ollama_provider_structure(self):
        from kb.services.llm.ollama_provider import OllamaProvider
        provider = OllamaProvider(model="llama3.2")
        assert provider.provider_name == "ollama"
        assert provider.model == "llama3.2"

    def test_openai_provider_structure(self):
        from kb.services.llm.openai_provider import OpenAIProvider
        provider = OpenAIProvider(api_key="sk-fake-key-for-testing")
        assert provider.provider_name == "openai"
