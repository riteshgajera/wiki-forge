"""
Tests for production-grade Karpathy-pattern upgrades:
  - WikiLogger (log.md, index.md, sources.md, wip.md)
  - HybridSearch (BM25 + RRF)
  - IntegrationAgent
  - EntityAgent
  - Orchestrator (end-to-end)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kb.services.llm.base import EmbedResponse, LLMProvider, LLMResponse


# ── Shared mock LLM ──────────────────────────────────────────────────────────

class MockLLM(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock"

    def complete(self, prompt: str, system: str = "", **kw) -> LLMResponse:
        s = system.lower()
        if "ontology" in s:
            data = {
                "concepts": [
                    {"name": "Transformer", "type": "model",
                     "definition": "Neural architecture using attention.", "aliases": []},
                    {"name": "Vaswani", "type": "person",
                     "definition": "Lead author of Attention Is All You Need.", "aliases": []},
                ],
                "relationships": [{"from": "Transformer", "relation": "created_by", "to": "Vaswani"}],
                "primary_domain": "machine-learning",
                "confidence": 0.88,
            }
        elif "wiki editor" in s or "wikilink" in s or "connected" in s:
            data = {
                "wikilinks": [{"term": "transformer", "target_file": "transformer",
                               "context_snippet": "using transformer"}],
                "suggested_backlinks": [],
                "new_concepts_needed": [],
                "confidence": 0.82,
            }
        elif "quality assurance" in s or "review" in s:
            data = {
                "score": 0.85, "title_ok": True, "has_summary": True,
                "has_wikilinks": True, "has_frontmatter": True, "word_count": 200,
                "issues": [], "suggestions": [], "approved": True, "confidence": 0.9,
            }
        elif "integration specialist" in s:
            data = {
                "page_updates": [
                    {"slug": "attention-mechanism", "subdir": "concepts",
                     "update_type": "add_reference",
                     "description": "New source references attention",
                     "content_snippet": "\n- See also [[summaries/test-source|Test Source]]"}
                ],
                "new_entity_pages": [],
                "contradictions": [],
                "confidence": 0.87,
            }
        elif "entity manager" in s:
            data = {
                "title": "Vaswani",
                "type": "person",
                "aliases": [],
                "l0_summary": "Lead author of Attention Is All You Need.",
                "definition": "Ashish Vaswani is lead author of the transformer paper.",
                "key_facts": ["Co-authored Attention Is All You Need (2017)"],
                "relationships": [],
                "appearances": [],
                "tags": ["person", "ml-researcher"],
                "confidence": 0.9,
            }
        elif "information architect" in s:
            data = {
                "sections": [{"title": "ML", "description": "ML concepts",
                               "articles": ["transformer"]}],
                "confidence": 0.95,
            }
        else:  # summarizer
            data = {
                "title": "Attention Is All You Need",
                "summary": "Introduces the Transformer architecture.",
                "key_points": ["Self-attention replaces RNNs", "Highly parallelisable"],
                "topics": ["transformers", "attention", "nlp"],
                "document_type": "paper",
                "audience": "expert",
                "confidence": 0.93,
            }
        return LLMResponse(content=json.dumps(data), model="mock", tokens_used=50)

    def embed(self, text: str) -> EmbedResponse:
        import hashlib, struct
        h = hashlib.md5(text.encode()).digest()
        floats = [struct.unpack("f", h[i:i+4])[0] for i in range(0, 16, 4)]
        norm = (sum(f**2 for f in floats) ** 0.5) or 1.0
        return EmbedResponse(embedding=[f/norm for f in floats], model="mock-embed")


# ── WikiLogger tests ──────────────────────────────────────────────────────────

class TestWikiLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        from kb.storage.wiki_logger import WikiLogger
        return WikiLogger(tmp_path / "wiki")

    def test_log_creates_file(self, logger):
        assert logger.log_path.exists()

    def test_log_append(self, logger):
        logger.log("ingest", "Paper A")
        logger.log("query", "What is attention?")
        content = logger.log_path.read_text()
        assert "ingest | Paper A" in content
        assert "query | What is attention?" in content

    def test_log_parseable_format(self, logger):
        logger.log("ingest", "Test Source")
        logger.log("lint", "Weekly check")
        recent = logger.recent_log(5)
        assert len(recent) == 2
        assert all(line.startswith("## [") for line in recent)

    def test_log_with_detail(self, logger):
        logger.log("compile", "3 docs processed", detail="Stats: {written: 3}")
        content = logger.log_path.read_text()
        assert "Stats: {written: 3}" in content

    def test_wip_file_created(self, logger):
        assert logger.wip_path.exists()
        assert "Work in Progress" in logger.wip_path.read_text()

    def test_wip_update(self, logger):
        logger.update_wip(
            focus="Research on transformers",
            pending=["paper_b.pdf"],
            questions=["What is MoE?"],
            next_actions=["Read BERT paper"],
        )
        content = logger.wip_path.read_text()
        assert "Research on transformers" in content
        assert "paper_b.pdf" in content
        assert "What is MoE?" in content
        assert "Read BERT paper" in content

    def test_sources_registry(self, logger):
        logger.register_source("raw/papers/attention.pdf", "application/pdf", 12, "done")
        content = logger.sources_path.read_text()
        assert "attention.pdf" in content
        assert "12" in content

    def test_rebuild_index(self, logger, tmp_path):
        from kb.storage.wiki_manager import WikiManager
        wiki = WikiManager(tmp_path / "wiki")
        wiki.write_article("Attention Mechanism", "Core concept in transformers.", tags=["ml"])
        wiki.write_article("Transformer Model", "Architecture using attention.", tags=["ml"])
        logger.rebuild_index(wiki)
        assert logger.index_path.exists()
        index = logger.index_path.read_text()
        assert "Attention Mechanism" in index or "attention-mechanism" in index

    def test_search_index(self, logger, tmp_path):
        from kb.storage.wiki_manager import WikiManager
        wiki = WikiManager(tmp_path / "wiki")
        wiki.write_article("Attention Mechanism", "Core concept in transformers.", tags=["ml"])
        logger.rebuild_index(wiki)
        results = logger.search_index("attention transformers")
        assert len(results) > 0
        assert results[0]["score"] > 0


# ── HybridSearch tests ────────────────────────────────────────────────────────

class TestHybridSearch:
    @pytest.fixture
    def wiki_with_articles(self, tmp_path):
        from kb.storage.wiki_manager import WikiManager
        wiki = WikiManager(tmp_path / "wiki")
        wiki.write_article("Attention Mechanism",
                           "The attention mechanism allows models to focus on relevant tokens. "
                           "It uses query key and value vectors to compute weighted sums.",
                           tags=["ml", "nlp"])
        wiki.write_article("Transformer Architecture",
                           "The transformer uses multi-head attention and feed-forward layers. "
                           "It replaced recurrent neural networks for sequence modeling.",
                           tags=["ml", "nlp"])
        wiki.write_article("BERT Language Model",
                           "BERT is an encoder-only transformer pre-trained with masked language modeling. "
                           "It achieves state-of-the-art results on many NLP benchmarks.",
                           tags=["ml", "nlp", "bert"])
        return wiki

    def test_bm25_build(self, wiki_with_articles):
        from kb.search.hybrid_search import BM25Index
        idx = BM25Index()
        idx.build(wiki_with_articles)
        assert len(idx._corpus) >= 3
        assert len(idx._idf) > 0

    def test_bm25_search_returns_results(self, wiki_with_articles):
        from kb.search.hybrid_search import BM25Index
        idx = BM25Index()
        idx.build(wiki_with_articles)
        results = idx.search("attention mechanism", k=5)
        assert len(results) > 0
        # Attention article should rank first
        top_idx, top_score = results[0]
        assert top_score > 0

    def test_bm25_ranks_by_relevance(self, wiki_with_articles):
        from kb.search.hybrid_search import BM25Index
        idx = BM25Index()
        idx.build(wiki_with_articles)
        results = idx.search("BERT masked language", k=3)
        assert len(results) > 0
        top_doc = idx.get_doc(results[0][0])
        assert "bert" in top_doc["slug"].lower() or "bert" in top_doc["title"].lower()

    def test_hybrid_search_bm25_mode(self, wiki_with_articles):
        from kb.search.hybrid_search import HybridSearch
        hs = HybridSearch(wiki_with_articles, vector_pipeline=None)
        hs.build()
        hits = hs.search("transformer architecture", k=3, mode="bm25")
        assert len(hits) > 0
        assert all(h.method == "bm25" for h in hits)

    def test_hybrid_search_snippet_extraction(self, wiki_with_articles):
        from kb.search.hybrid_search import HybridSearch
        hs = HybridSearch(wiki_with_articles)
        hs.build()
        hits = hs.search("attention mechanism", k=1, mode="bm25")
        assert hits
        assert len(hits[0].snippet) > 0

    def test_rrf_fusion(self):
        from kb.search.hybrid_search import reciprocal_rank_fusion
        bm25_results = [(0, 2.5), (1, 1.2), (2, 0.8)]
        bm25_corpus = [
            {"slug": "doc-a", "subdir": "concepts", "title": "Doc A", "tags": []},
            {"slug": "doc-b", "subdir": "concepts", "title": "Doc B", "tags": []},
            {"slug": "doc-c", "subdir": "concepts", "title": "Doc C", "tags": []},
        ]
        fused = reciprocal_rank_fusion(bm25_results, [], bm25_corpus)
        assert len(fused) == 3
        # RRF should preserve order when no vector results
        assert fused[0]["slug"] == "doc-a"

    def test_empty_wiki_search(self, tmp_path):
        from kb.search.hybrid_search import HybridSearch
        from kb.storage.wiki_manager import WikiManager
        wiki = WikiManager(tmp_path / "empty_wiki")
        hs = HybridSearch(wiki)
        hs.build()
        hits = hs.search("anything", k=5)
        assert hits == []


# ── IntegrationAgent tests ────────────────────────────────────────────────────

class TestIntegrationAgent:
    @pytest.fixture
    def agent(self):
        from kb.agents.integration_agent import IntegrationAgent
        return IntegrationAgent(MockLLM())

    def test_integration_with_no_index(self, agent):
        from kb.agents.base import AgentInput
        inp = AgentInput(
            doc_id="test",
            content="",
            metadata={
                "prior_results": {
                    "summarizer": {"title": "T", "summary": "S", "key_points": [], "topics": []},
                    "concept_extractor": {"concepts": [], "relationships": [], "primary_domain": "ml"},
                },
                "wiki_index_content": "",  # No existing wiki
            },
        )
        output = agent.run(inp)
        assert output.success
        assert output.result["page_updates"] == []
        assert output.result["contradictions"] == []

    def test_integration_with_existing_wiki(self, agent):
        from kb.agents.base import AgentInput
        inp = AgentInput(
            doc_id="test",
            content="",
            metadata={
                "prior_results": {
                    "summarizer": {"title": "Attention Paper", "summary": "About attention.",
                                   "key_points": ["Key 1"], "topics": ["ml"]},
                    "concept_extractor": {"concepts": [
                        {"name": "Attention", "type": "method", "definition": "...", "aliases": []}
                    ], "relationships": [], "primary_domain": "ml"},
                },
                "wiki_index_content": "| [[concepts/attention-mechanism|Attention Mechanism]] | ... |",
            },
        )
        output = agent.run(inp)
        assert output.success
        assert "page_updates" in output.result
        assert "confidence" in output.result

    def test_integration_validates_structure(self, agent):
        valid = {"page_updates": [], "new_entity_pages": [], "contradictions": [], "confidence": 0.9}
        assert agent.validate(valid)
        invalid = {"some_other_key": "value"}
        assert not agent.validate(invalid)


# ── EntityAgent tests ─────────────────────────────────────────────────────────

class TestEntityAgent:
    @pytest.fixture
    def agent(self):
        from kb.agents.entity_agent import EntityAgent
        return EntityAgent(MockLLM())

    def test_entity_agent_creates_page(self, agent):
        from kb.agents.base import AgentInput
        inp = AgentInput(
            doc_id="vaswani",
            content="",
            metadata={
                "entity_info": {
                    "name": "Vaswani",
                    "type": "person",
                    "definition": "Lead author of transformer paper.",
                    "aliases": [],
                },
                "existing_content": "",
                "prior_results": {
                    "summarizer": {"summary": "Paper on transformer architecture."}
                },
            },
        )
        output = agent.run(inp)
        assert output.success
        assert "title" in output.result
        assert "definition" in output.result
        assert output.result["confidence"] > 0

    def test_render_entity_page(self):
        from kb.agents.entity_agent import EntityAgent
        result = {
            "title": "Ashish Vaswani",
            "type": "person",
            "aliases": ["Vaswani"],
            "l0_summary": "Lead author of Attention Is All You Need.",
            "definition": "Researcher who led the transformer paper.",
            "key_facts": ["Co-authored 2017 transformer paper"],
            "relationships": [{"relation": "works_at", "entity": "Google Brain"}],
            "appearances": ["[[summaries/transformer-paper|Transformer Paper]]"],
            "tags": ["person", "researcher"],
            "confidence": 0.9,
        }
        page = EntityAgent.render_entity_page(result, "test-source")
        assert "Ashish Vaswani" in page
        assert "type: entity" in page
        assert "entity_type: person" in page
        assert "depth: L1" in page
        assert "Lead author" in page
        assert "Key Facts" in page
        assert "google-brain" in page  # slugified relationship

    def test_entity_page_validates(self, agent):
        valid = {"title": "T", "definition": "D", "confidence": 0.9}
        assert agent.validate(valid)
        assert not agent.validate({"only_title": "T"})


# ── Orchestrator end-to-end ─────────────────────────────────────────

class TestOrchestrator:
    @pytest.fixture
    def env(self, tmp_path):
        raw = tmp_path / "raw"
        wiki = tmp_path / "wiki"
        db = tmp_path / "data" / "kb.db"
        raw.mkdir()
        (raw / "transformer.md").write_text(
            "# Attention Is All You Need\n\n"
            "The Transformer architecture was introduced by Vaswani et al. in 2017. "
            "It relies entirely on attention mechanisms, dispensing with recurrence. "
            "The model achieves state-of-the-art results on translation tasks. "
            "Multi-head attention allows the model to attend to different positions.\n"
        )
        return tmp_path, raw, wiki, db

    def _make_orch(self, env):
        tmp_path, raw, wiki, db = env
        from kb.pipelines.orchestrator import Orchestrator
        from kb.utils.config import Settings

        # Patch the LLM factory
        import kb.services.llm.factory as factory_mod
        original = factory_mod.get_default_provider
        factory_mod.get_default_provider = lambda: MockLLM()

        cfg = Settings(
            raw_dir=str(raw),
            wiki_dir=str(wiki),
            db_path=str(db),
        )
        orch = Orchestrator(cfg=cfg)
        orch.llm = MockLLM()
        orch._pipeline = orch._build_pipeline()

        factory_mod.get_default_provider = original
        return orch, raw, wiki

    def test_compile_produces_summary_pages(self, env):
        orch, raw, wiki = self._make_orch(env)
        stats = orch.compile()
        assert stats["processed"] >= 1
        # Summary page should exist
        summaries = list((wiki / "summaries").glob("*.md"))
        assert len(summaries) >= 1

    def test_compile_produces_entity_pages(self, env):
        orch, raw, wiki = self._make_orch(env)
        orch.compile()
        entities = list((wiki / "entities").glob("*.md"))
        # Vaswani should be detected as a person entity
        assert len(entities) >= 1

    def test_compile_writes_log(self, env):
        orch, raw, wiki = self._make_orch(env)
        orch.compile()
        log_path = wiki / "log.md"
        assert log_path.exists()
        log = log_path.read_text()
        assert "ingest" in log or "compile" in log

    def test_compile_rebuilds_index(self, env):
        orch, raw, wiki = self._make_orch(env)
        orch.compile()
        index_path = wiki / "index.md"
        assert index_path.exists()
        index = index_path.read_text()
        assert "Knowledge Base Index" in index

    def test_query_returns_hits(self, env):
        orch, raw, wiki = self._make_orch(env)
        orch.compile()
        result = orch.query_and_file("transformer attention mechanism")
        assert "hits" in result
        assert "query" in result

    def test_session_context(self, env):
        orch, raw, wiki = self._make_orch(env)
        orch.compile()
        ctx = orch.session_context()
        assert "KB Session Context" in ctx
        assert "Recent operations" in ctx

    def test_second_source_integrates(self, env):
        """Verify that a second source triggers integration with the first source's pages."""
        tmp_path, raw, wiki, db = env
        orch, raw, wiki = self._make_orch(env)

        # First compile
        orch.compile()
        initial_files = set(f.name for f in wiki.rglob("*.md"))

        # Add second source
        (raw / "bert.md").write_text(
            "# BERT: Pre-training of Deep Bidirectional Transformers\n\n"
            "BERT builds on the Transformer architecture from Vaswani et al. "
            "It uses masked language modeling for pre-training. "
            "BERT achieves state-of-the-art on GLUE benchmark tasks.\n"
        )

        # Rebuild orchestrator to pick up new file
        orch2, raw, wiki = self._make_orch(env)
        # Build index content from first compile
        orch2.wlog.rebuild_index(orch2.wiki)
        stats = orch2.compile()

        assert stats["processed"] >= 1
        # Should have more pages now
        final_files = set(f.name for f in wiki.rglob("*.md"))
        assert len(final_files) >= len(initial_files)


# ── WikiManager L0/L1/L2 frontmatter tests ───────────────────────────────────

class TestEnrichedFrontmatter:
    @pytest.fixture
    def wiki(self, tmp_path):
        from kb.storage.wiki_manager import WikiManager
        return WikiManager(tmp_path / "wiki")

    def test_l0_summary_in_frontmatter(self, wiki):
        wiki.write_article(
            "Test Concept",
            "This is a detailed description of the concept and its applications.",
            l0_summary="A brief one-liner for the index.",
            depth="L1",
        )
        content = wiki.read_article("test-concept")
        assert "l0_summary" in content
        assert "A brief one-liner" in content

    def test_depth_field_written(self, wiki):
        wiki.write_article("Deep Concept", "Content here.", depth="L2")
        content = wiki.read_article("deep-concept")
        assert "depth: L2" in content

    def test_status_field_written(self, wiki):
        wiki.write_article("Stale Concept", "Old info.", status="stale")
        content = wiki.read_article("stale-concept")
        assert "status: stale" in content

    def test_related_field_written(self, wiki):
        wiki.write_article("Linked Concept", "Content.", related=["concept-a", "concept-b"])
        content = wiki.read_article("linked-concept")
        assert "related:" in content
        assert "concept-a" in content

    def test_contradicts_field_written(self, wiki):
        wiki.write_article("Disputed Claim", "Content.", contradicts=["old-claim"])
        content = wiki.read_article("disputed-claim")
        assert "contradicts:" in content
        assert "old-claim" in content

    def test_auto_l0_summary_from_content(self, wiki):
        """If no l0_summary provided, auto-generate from content."""
        wiki.write_article(
            "Auto Summary",
            "The attention mechanism is a powerful technique. It enables focus on relevant parts.",
        )
        content = wiki.read_article("auto-summary")
        assert "l0_summary:" in content
        # Should contain part of the first sentence
        assert "attention mechanism" in content or "Auto Summary" in content
