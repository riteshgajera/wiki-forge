"""Unit tests for storage layer, helpers, and parsers."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from kb.storage.metadata_store import DocumentRecord, MetadataStore
from kb.storage.wiki_manager import WikiManager
from kb.utils.helpers import chunk_text, hash_file, hash_text, safe_json_parse, slugify


# ─── Helpers ──────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_hash_text_deterministic(self):
        assert hash_text("hello") == hash_text("hello")
        assert hash_text("hello") != hash_text("world")

    def test_hash_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        h1 = hash_file(f)
        assert len(h1) == 64  # SHA-256 hex
        f.write_text("changed")
        h2 = hash_file(f)
        assert h1 != h2

    def test_slugify(self):
        assert slugify("Transformer Architecture") == "transformer-architecture"
        assert slugify("GPT-4 Overview!") == "gpt-4-overview"
        assert slugify("  spaces  ") == "spaces"

    def test_chunk_text_short(self):
        text = "Short text"
        chunks = chunk_text(text, chunk_size=1500)
        assert chunks == [text]

    def test_chunk_text_splits(self):
        text = "A" * 3000
        chunks = chunk_text(text, chunk_size=1500, overlap=100)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 1600  # chunk_size + some boundary tolerance

    def test_chunk_text_overlap(self):
        text = "word " * 600  # 3000 chars
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 3

    def test_safe_json_parse_clean(self):
        result = safe_json_parse('{"key": "value", "confidence": 0.9}')
        assert result == {"key": "value", "confidence": 0.9}

    def test_safe_json_parse_fenced(self):
        text = '```json\n{"key": "val"}\n```'
        result = safe_json_parse(text)
        assert result == {"key": "val"}

    def test_safe_json_parse_invalid(self):
        result = safe_json_parse("not json at all")
        assert result is None

    def test_safe_json_parse_embedded(self):
        text = 'Here is the result: {"a": 1} end'
        result = safe_json_parse(text)
        assert result == {"a": 1}


# ─── MetadataStore ────────────────────────────────────────────────────────────

class TestMetadataStore:
    @pytest.fixture
    def store(self, tmp_path):
        return MetadataStore(tmp_path / "test.db")

    def test_upsert_and_get(self, store):
        rec = DocumentRecord(id="abc123", path="raw/test.txt", status="pending")
        store.upsert(rec)
        fetched = store.get("abc123")
        assert fetched is not None
        assert fetched.id == "abc123"
        assert fetched.path == "raw/test.txt"
        assert fetched.status == "pending"

    def test_update_status(self, store):
        rec = DocumentRecord(id="xyz", path="raw/doc.pdf")
        store.upsert(rec)
        store.update_status("xyz", "done")
        assert store.get("xyz").status == "done"

    def test_list_by_status(self, store):
        store.upsert(DocumentRecord(id="a1", path="a.txt", status="pending"))
        store.upsert(DocumentRecord(id="b2", path="b.txt", status="done"))
        store.upsert(DocumentRecord(id="c3", path="c.txt", status="pending"))
        pending = store.list_by_status("pending")
        assert len(pending) == 2
        assert all(r.status == "pending" for r in pending)

    def test_upsert_updates_existing(self, store):
        rec = DocumentRecord(id="upd", path="f.txt", status="pending")
        store.upsert(rec)
        rec.status = "done"
        rec.confidence = 0.9
        store.upsert(rec)
        fetched = store.get("upd")
        assert fetched.status == "done"
        assert fetched.confidence == pytest.approx(0.9)

    def test_get_by_path(self, store):
        store.upsert(DocumentRecord(id="id1", path="articles/foo.md"))
        fetched = store.get_by_path("articles/foo.md")
        assert fetched is not None
        assert fetched.id == "id1"

    def test_stats(self, store):
        store.upsert(DocumentRecord(id="s1", path="a.txt", status="done"))
        store.upsert(DocumentRecord(id="s2", path="b.txt", status="done"))
        store.upsert(DocumentRecord(id="s3", path="c.txt", status="failed"))
        stats = store.stats()
        assert stats["done"] == 2
        assert stats["failed"] == 1

    def test_update_agent_output(self, store):
        store.upsert(DocumentRecord(id="ao1", path="x.txt"))
        store.update_agent_output("ao1", "summarizer", {"summary": "test"}, 0.85)
        rec = store.get("ao1")
        assert "summarizer" in rec.agent_outputs
        assert rec.confidence == pytest.approx(0.85)


# ─── WikiManager ──────────────────────────────────────────────────────────────

class TestWikiManager:
    @pytest.fixture
    def wiki(self, tmp_path):
        return WikiManager(tmp_path / "wiki")

    def test_write_and_read(self, wiki):
        path = wiki.write_article("Test Article", "## Content\nHello world")
        assert path.exists()
        content = wiki.read_article("test-article")
        assert content is not None
        assert "Hello world" in content
        assert "title: \"Test Article\"" in content or "title: Test Article" in content

    def test_article_not_found(self, wiki):
        assert wiki.read_article("nonexistent") is None

    def test_article_exists(self, wiki):
        wiki.write_article("Exists", "content")
        assert wiki.article_exists("exists")
        assert not wiki.article_exists("doesnt-exist")

    def test_list_articles(self, wiki):
        wiki.write_article("Article One", "content 1")
        wiki.write_article("Article Two", "content 2")
        articles = wiki.list_articles()
        assert len(articles) >= 2

    def test_backup_on_overwrite(self, wiki):
        wiki.write_article("Backed Up", "version 1", backup=True)
        wiki.write_article("Backed Up", "version 2", backup=True)
        backups = list((wiki.wiki_dir / "_meta" / "backups").rglob("*.bak"))
        assert len(backups) >= 1

    def test_write_with_tags(self, wiki):
        wiki.write_article("Tagged", "content", tags=["ml", "nlp"])
        content = wiki.read_article("tagged")
        assert "ml" in content

    def test_write_index(self, wiki):
        path = wiki.write_index("# Index\n- [[article1]]")
        assert path.exists()
        assert "# Index" in path.read_text()

    def test_get_all_wikilinks(self, wiki):
        wiki.write_article("Source", "See [[target-article]] for more.")
        links = wiki.get_all_wikilinks()
        assert "source" in links
        assert "target-article" in links["source"]


# ─── Parsers ──────────────────────────────────────────────────────────────────

class TestParsers:
    def test_text_parser(self, tmp_path):
        from kb.ingestion.parsers.document_parsers import TextParser
        f = tmp_path / "test.md"
        f.write_text("# Hello\nThis is content.")
        parser = TextParser()
        assert parser.can_parse(f)
        doc = parser.parse(f)
        assert "Hello" in doc.content
        assert doc.error == ""

    def test_text_parser_unsupported(self, tmp_path):
        from kb.ingestion.parsers.document_parsers import TextParser
        f = tmp_path / "test.docx"
        f.write_bytes(b"fake docx")
        assert not TextParser().can_parse(f)

    def test_parser_registry_routes(self, tmp_path):
        from kb.ingestion.parsers.document_parsers import ParserRegistry
        registry = ParserRegistry()
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        doc = registry.parse(f)
        assert "hello world" in doc.content

    def test_parser_registry_unknown(self, tmp_path):
        from kb.ingestion.parsers.document_parsers import ParserRegistry
        f = tmp_path / "test.xyz"
        f.write_bytes(b"unknown")
        doc = ParserRegistry().parse(f)
        assert doc.error != ""
