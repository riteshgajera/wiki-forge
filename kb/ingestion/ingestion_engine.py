"""
Ingestion engine: scans /raw for new/changed files, fingerprints them,
updates the metadata store, and returns a list of documents ready for processing.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from kb.ingestion.parsers.document_parsers import ParsedDocument, ParserRegistry
from kb.storage.metadata_store import DocumentRecord, MetadataStore
from kb.utils.config import Settings, settings as default_settings
from kb.utils.helpers import hash_file
from kb.utils.logging import get_logger

logger = get_logger("ingestion.engine")


class IngestionEngine:
    """Scans raw directory, fingerprints files, and queues new/changed docs."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        self.store = MetadataStore(self.cfg.db_path)
        self.parser = ParserRegistry()
        self.supported_exts = set(self.cfg.ingestion.supported_extensions)

    def scan(self, raw_dir: str | Path | None = None) -> list[DocumentRecord]:
        """
        Scan directory for files. Returns list of newly queued records.
        Skips unchanged files (same hash already in DB with status=done).
        """
        raw_path = Path(raw_dir or self.cfg.raw_dir)
        if not raw_path.exists():
            logger.warning("raw_dir_missing", path=str(raw_path))
            return []

        queued: list[DocumentRecord] = []

        for file_path in sorted(raw_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.supported_exts:
                continue

            rel_path = str(file_path.relative_to(raw_path))

            try:
                doc_hash = hash_file(file_path)
            except OSError as e:
                logger.error("hash_error", path=rel_path, error=str(e))
                continue

            # Check if already processed with same hash
            existing = self.store.get(doc_hash)
            if existing and existing.status == "done":
                logger.debug("skip_unchanged", path=rel_path, hash=doc_hash[:8])
                continue

            # Check if path exists with different hash (modified file)
            existing_by_path = self.store.get_by_path(rel_path)
            if existing_by_path and existing_by_path.id != doc_hash:
                logger.info("file_changed", path=rel_path, old_hash=existing_by_path.id[:8])

            record = DocumentRecord(
                id=doc_hash,
                path=rel_path,
                status="pending",
                file_size=file_path.stat().st_size,
            )
            self.store.upsert(record)
            queued.append(record)
            logger.info("file_queued", path=rel_path, hash=doc_hash[:8])

        logger.info("scan_complete", total_queued=len(queued), raw_dir=str(raw_path))
        return queued

    def parse_document(self, record: DocumentRecord, raw_dir: str | Path | None = None) -> ParsedDocument:
        """Parse a queued document record into a ParsedDocument."""
        raw_path = Path(raw_dir or self.cfg.raw_dir)
        file_path = raw_path / record.path
        parsed = self.parser.parse(file_path)
        # Attach record metadata
        parsed.metadata["doc_id"] = record.id
        parsed.metadata["rel_path"] = record.path
        return parsed

    def mark_processing(self, doc_id: str) -> None:
        self.store.update_status(doc_id, "processing")

    def mark_done(self, doc_id: str, wiki_path: str, confidence: float) -> None:
        rec = self.store.get(doc_id)
        if rec:
            rec.status = "done"
            rec.wiki_path = wiki_path
            rec.confidence = confidence
            self.store.upsert(rec)

    def mark_failed(self, doc_id: str, error: str) -> None:
        self.store.update_status(doc_id, "failed", error_msg=error)

    def mark_review(self, doc_id: str) -> None:
        self.store.update_status(doc_id, "review")

    def get_pending(self) -> list[DocumentRecord]:
        return self.store.list_by_status("pending")

    def get_failed(self) -> list[DocumentRecord]:
        return self.store.list_by_status("failed")

    def get_review_queue(self) -> list[DocumentRecord]:
        return self.store.list_by_status("review")

    def stats(self) -> dict[str, int]:
        return self.store.stats()

    def watch(
        self,
        raw_dir: str | Path | None = None,
        on_new: Callable[[list[DocumentRecord]], None] | None = None,
    ) -> None:
        """Watch for new files continuously (blocking)."""
        raw_path = Path(raw_dir or self.cfg.raw_dir)
        interval = self.cfg.ingestion.watch_interval_seconds
        logger.info("watch_start", path=str(raw_path), interval=interval)
        try:
            while True:
                new_docs = self.scan(raw_path)
                if new_docs and on_new:
                    on_new(new_docs)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("watch_stopped")
