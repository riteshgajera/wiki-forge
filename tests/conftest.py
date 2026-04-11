"""Shared pytest fixtures."""
import pytest
from pathlib import Path


@pytest.fixture
def sample_raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc1.md").write_text(
        "# Machine Learning Basics\n\n"
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without explicit programming.\n"
    )
    (raw / "doc2.txt").write_text(
        "Natural language processing (NLP) focuses on the interaction "
        "between computers and human language.\n"
    )
    return raw


@pytest.fixture
def sample_wiki_dir(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    return wiki
