"""
Production Orchestrator — implements the full Karpathy LLM Wiki pattern.

Key upgrades over basic orchestrator:
  1. Each source touches 10-15 existing wiki pages (IntegrationAgent)
  2. Entity pages created and maintained
  3. log.md updated after every operation
  4. index.md rebuilt after every compile
  5. Contradiction detection
  6. Source registry maintained
  7. Answers can be filed back as wiki pages
  8. Hybrid BM25+vector search rebuilt after compile
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kb.agents.base import AgentInput
from kb.agents.entity_agent import EntityAgent
from kb.agents.index_builder import IndexBuilderAgent
from kb.agents.integration_agent import IntegrationAgent
from kb.ingestion.ingestion_engine import IngestionEngine
from kb.pipelines.engine import Pipeline, PipelineResult, PipelineStep
from kb.services.llm.factory import get_default_provider
from kb.storage.metadata_store import DocumentRecord
from kb.storage.wiki_logger import WikiLogger
from kb.storage.wiki_manager import WikiManager
from kb.utils.config import Settings, settings as default_settings
from kb.utils.helpers import slugify
from kb.utils.logging import get_logger

logger = get_logger("orchestrator.production")


class Orchestrator:
    """
    Full Karpathy-pattern orchestrator.
    The wiki is a persistent, compounding artifact.
    """

    def __init__(
        self,
        cfg: Settings | None = None,
        review_callback: Any | None = None,
        interactive: bool = False,
    ) -> None:
        self.cfg = cfg or default_settings
        self.cfg.ensure_dirs()
        self.review_callback = review_callback
        self.interactive = interactive

        self.ingestion = IngestionEngine(self.cfg)
        self.wiki = WikiManager(self.cfg.wiki_dir)
        self.wlog = WikiLogger(self.cfg.wiki_dir)
        self.llm = get_default_provider()
        self._pipeline = self._build_pipeline()
        self._search: Any | None = None

        logger.info("orchestrator_init",
                    provider=self.cfg.llm.provider,
                    model=self.cfg.llm.model)

    def _build_pipeline(self) -> Pipeline:
        from kb.agents.summarizer import SummarizerAgent
        from kb.agents.concept_extractor import ConceptExtractorAgent
        from kb.agents.linker import LinkerAgent
        from kb.agents.linting import LintingQAAgent

        thresh = self.cfg.processing.confidence_threshold
        auto = self.cfg.processing.auto_approve_threshold

        return Pipeline(
            steps=[
                PipelineStep("summarizer", SummarizerAgent(self.llm, thresh), depends_on=[]),
                PipelineStep("concept_extractor", ConceptExtractorAgent(self.llm, thresh),
                             depends_on=["summarizer"]),
                PipelineStep("linker", LinkerAgent(self.llm, thresh),
                             depends_on=["summarizer", "concept_extractor"]),
                PipelineStep("linting_qa", LintingQAAgent(self.llm, thresh),
                             depends_on=["linker"]),
            ],
            review_callback=self.review_callback,
            auto_approve_threshold=auto,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compile(
        self,
        raw_dir: str | None = None,
        force: bool = False,
        batch_size: int | None = None,
        guided: bool = False,
    ) -> dict[str, Any]:
        """Full compile cycle. Returns stats."""
        raw_path = raw_dir or self.cfg.raw_dir
        batch = batch_size or self.cfg.processing.batch_size

        queued = self.ingestion.scan(raw_path)
        if force:
            done = self.ingestion.store.list_by_status("done")
            queued = queued + done

        if not queued:
            logger.info("compile_nothing_to_do")
            return {"processed": 0, "written": 0, "review": 0, "failed": 0,
                    "entities_created": 0, "pages_integrated": 0}

        logger.info("compile_start", queued=len(queued))
        stats: dict[str, int] = {
            "processed": 0, "written": 0, "review": 0, "failed": 0,
            "entities_created": 0, "pages_integrated": 0,
        }

        # Read current index for integration agent context
        index_content = ""
        if self.wlog.index_path.exists():
            index_content = self.wlog.index_path.read_text(encoding="utf-8")

        for i in range(0, len(queued), batch):
            for record in queued[i:i + batch]:
                if guided:
                    self._guided_ingest_prompt(record)
                outcome, entity_count, pages_count = self._process_document(
                    record, raw_path, index_content
                )
                stats[outcome] = stats.get(outcome, 0) + 1
                stats["processed"] += 1
                stats["entities_created"] += entity_count
                stats["pages_integrated"] += pages_count

        # Rebuild index and search after all documents processed
        pages_touched = stats["written"] + stats["pages_integrated"] + stats["entities_created"]
        if pages_touched > 0:
            self._rebuild_index()
            self._rebuild_search()

        self.wlog.log(
            "compile",
            f"{stats['written']} written, {stats['pages_integrated']} pages integrated",
            detail=f"Stats: {json.dumps(stats)}"
        )
        logger.info("compile_complete", **stats)
        return stats

    def query_and_file(
        self,
        query: str,
        file_answer: bool = False,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Query the wiki using hybrid search.
        Optionally file the answer back as an analysis page.
        """
        search = self._get_search()
        hits = search.search(query, k=k, mode="hybrid")

        # Load full content of top hits
        context_pages = []
        for hit in hits[:3]:
            content = self.wiki.read_article(hit.slug, hit.subdir)
            if content:
                context_pages.append({
                    "title": hit.title,
                    "slug": hit.slug,
                    "content": content[:2000],
                })

        answer = self._synthesize_answer(query, context_pages)

        result = {
            "query": query,
            "hits": [{"title": h.title, "slug": h.slug, "score": h.score,
                       "snippet": h.snippet, "method": h.method} for h in hits],
            "answer": answer,
            "filed": False,
        }

        if file_answer and answer:
            filed_path = self._file_answer(query, answer, hits)
            result["filed"] = True
            result["filed_path"] = str(filed_path)
            self.wlog.log("query", query, detail=f"Answer filed to {filed_path}")
        else:
            self.wlog.log("query", query)

        return result

    def lint_wiki(self, check_contradictions: bool = False) -> dict[str, Any]:
        """Run full lint pass. Returns issues found."""
        from kb.agents.linting import LintingQAAgent

        agent = LintingQAAgent(self.llm)
        all_slugs = [p.stem for p in self.wiki.list_articles()]
        issues: list[dict] = []
        orphans: list[str] = []
        stale: list[str] = []

        # Find orphans (no inbound wikilinks)
        wikilink_graph = self.wiki.get_all_wikilinks()
        inbound: dict[str, int] = {}
        for links in wikilink_graph.values():
            for target in links:
                inbound[target] = inbound.get(target, 0) + 1
        for path in self.wiki.list_articles():
            slug = path.stem
            if slug not in inbound and slug not in ("index", "README", "WIKI", "wip", "log"):
                orphans.append(slug)

        # Run LLM lint on each article
        for path in self.wiki.list_articles()[:30]:  # cap for speed
            if path.stem in ("index", "README", "WIKI", "wip"):
                continue
            content = path.read_text(encoding="utf-8")
            inp = AgentInput(doc_id=path.stem, content=content,
                             metadata={"known_slugs": all_slugs})
            output = agent.run(inp)
            if output.success:
                for iss in output.result.get("issues", []):
                    iss["article"] = path.stem
                    issues.append(iss)

        self.wlog.log(
            "lint",
            f"{len(issues)} issues, {len(orphans)} orphans",
            detail=f"Orphans: {orphans[:10]}"
        )

        return {
            "issues": issues,
            "orphans": orphans,
            "stale": stale,
            "total_articles": len(list(self.wiki.list_articles())),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_document(
        self, record: DocumentRecord, raw_dir: str, index_content: str
    ) -> tuple[str, int, int]:
        """Process one document. Returns (outcome, entities_created, pages_integrated)."""
        self.ingestion.mark_processing(record.id)
        entities_created = 0
        pages_integrated = 0

        try:
            parsed = self.ingestion.parse_document(record, raw_dir)
            if not parsed.content:
                self.ingestion.mark_failed(record.id, "Empty content")
                return "failed", 0, 0

            concept_index = self._build_concept_index()
            inp = AgentInput(
                doc_id=record.id,
                content=parsed.content,
                metadata={
                    "rel_path": record.path,
                    "mime_type": parsed.mime_type,
                    "concept_index": concept_index,
                    "wiki_index_content": index_content,
                },
            )

            result: PipelineResult = self._pipeline.run(inp)
            if result.halted:
                self.ingestion.mark_failed(record.id, result.halt_reason)
                return "failed", 0, 0

            # Write summary page
            summary_path = self._write_summary_page(record, parsed, result)
            summary_slug = summary_path.stem

            # Run integration agent: update existing pages
            if index_content:
                pages_integrated = self._run_integration(record, result, summary_slug)

            # Create entity pages for newly discovered entities
            entities_created = self._create_entity_pages(result, summary_slug)

            # Register source
            self.wlog.register_source(
                record.path, parsed.mime_type,
                pages_touched=pages_integrated + entities_created + 1
            )

            if result.needs_review:
                self.ingestion.mark_review(record.id)
                # Still log the ingest attempt
                summary = result.get_output("summarizer")
                title = summary.get("title", record.path)
                self.wlog.log("ingest", title,
                              detail=f"Status: review | Confidence: {result.overall_confidence:.2f}")
                return "review", entities_created, pages_integrated

            self.ingestion.mark_done(
                record.id, str(summary_path), result.overall_confidence
            )
            # Log the ingest
            summary = result.get_output("summarizer")
            title = summary.get("title", record.path)
            self.wlog.log(
                "ingest", title,
                detail=f"Source: {record.path} | Confidence: {result.overall_confidence:.2f} | "
                       f"Entities: {entities_created} | Pages integrated: {pages_integrated}"
            )
            return "written", entities_created, pages_integrated

        except Exception as e:
            logger.error("process_error", doc_id=record.id, error=str(e))
            self.ingestion.mark_failed(record.id, str(e))
            return "failed", 0, 0

    def _run_integration(
        self, record: DocumentRecord, result: PipelineResult, summary_slug: str
    ) -> int:
        """Run IntegrationAgent to update existing pages. Returns pages touched."""
        try:
            integration_agent = IntegrationAgent(self.llm, self.cfg.processing.confidence_threshold)
            inp = AgentInput(
                doc_id=record.id,
                content="",
                metadata={
                    "prior_results": {
                        "summarizer": result.get_output("summarizer"),
                        "concept_extractor": result.get_output("concept_extractor"),
                    },
                    "wiki_index_content": self.wlog.index_path.read_text(encoding="utf-8")
                    if self.wlog.index_path.exists() else "",
                },
            )
            output = integration_agent.run(inp)
            if not output.success:
                return 0

            pages_touched = 0
            for update in output.result.get("page_updates", []):
                slug = update.get("slug", "")
                subdir = update.get("subdir", "concepts")
                snippet = update.get("content_snippet", "")
                if not slug or not snippet:
                    continue
                # Append the snippet to the existing page
                existing = self.wiki.read_article(slug, subdir)
                if existing:
                    updated = existing.rstrip() + f"\n\n{snippet}\n"
                    path = self.wiki.wiki_dir / subdir / f"{slug}.md"
                    path.write_text(updated, encoding="utf-8")
                    pages_touched += 1

            # Flag contradictions in frontmatter
            for contradiction in output.result.get("contradictions", []):
                slug = contradiction.get("existing_slug", "")
                if slug:
                    self._flag_contradiction(slug, contradiction)

            return pages_touched

        except Exception as e:
            logger.warning("integration_error", error=str(e))
            return 0

    def _create_entity_pages(self, result: PipelineResult, summary_slug: str) -> int:
        """Create entity pages for all entities in concept extractor output."""
        concepts = result.get_output("concept_extractor")
        entity_agent = EntityAgent(self.llm)
        count = 0

        for concept in concepts.get("concepts", []):
            if concept.get("type") not in ("person", "organization", "model", "dataset", "tool"):
                continue
            slug = slugify(concept["name"])
            existing = self.wiki.read_article(slug, "entities")

            inp = AgentInput(
                doc_id=slug,
                content="",
                metadata={
                    "entity_info": concept,
                    "existing_content": existing or "",
                    "prior_results": {"summarizer": result.get_output("summarizer")},
                },
            )
            output = entity_agent.run(inp)
            if output.success:
                from kb.agents.entity_agent import EntityAgent as EA
                page_content = EA.render_entity_page(output.result, summary_slug)
                # Write just the body (without frontmatter, since render_entity_page includes it)
                path = self.wiki.wiki_dir / "entities" / f"{slug}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(page_content, encoding="utf-8")
                count += 1

        return count

    def _write_summary_page(
        self, record: DocumentRecord, parsed: Any, result: PipelineResult
    ) -> Path:
        """Write a per-source summary page to /summaries/."""
        summary = result.get_output("summarizer")
        concepts = result.get_output("concept_extractor")

        title = summary.get("title", Path(record.path).stem)
        slug = slugify(title)
        tags = summary.get("topics", [])

        lines = [
            f"# {title}\n",
            f"{summary.get('summary', '')}\n",
            f"\n**Source**: `{record.path}`\n",
            "\n## Key Points\n",
        ]
        for pt in summary.get("key_points", []):
            lines.append(f"- {pt}")

        concept_list = concepts.get("concepts", [])
        if concept_list:
            lines.append("\n## Concepts Discussed\n")
            for c in concept_list[:8]:
                name = c.get("name", "")
                defn = c.get("definition", "")
                s = slugify(name)
                lines.append(f"- **[[concepts/{s}|{name}]]**: {defn}")

        path = self.wiki.write_article(
            title=title,
            content="\n".join(lines),
            subdir="summaries",
            slug=slug,
            source=record.path,
            tags=tags,
            confidence=result.overall_confidence,
            page_type="summary",
            depth="L1",
            l0_summary=summary.get("summary", "")[:120],
        )
        return path

    def _synthesize_answer(self, query: str, context_pages: list[dict]) -> str:
        """Use LLM to synthesise an answer from retrieved pages."""
        if not context_pages:
            return "No relevant pages found in the wiki."
        context = "\n\n---\n\n".join(
            f"### {p['title']}\n{p['content'][:1500]}" for p in context_pages
        )
        prompt = (
            f"Based on these wiki pages, answer the following question.\n\n"
            f"**Question**: {query}\n\n"
            f"**Wiki context**:\n{context}\n\n"
            f"Provide a clear, well-structured answer with citations to the wiki pages."
        )
        resp = self.llm.complete(prompt)
        return resp.content

    def _file_answer(self, query: str, answer: str, hits: list) -> Path:
        """File a query answer as an analysis page."""
        from datetime import date
        date_str = date.today().isoformat()
        slug = f"{date_str}-{slugify(query)[:40]}"
        refs = "\n".join(f"- [[{h.subdir}/{h.slug}|{h.title}]]" for h in hits[:5])
        content = (
            f"# {query}\n\n"
            f"> Query answered {date_str}\n\n"
            f"{answer}\n\n"
            f"## Sources\n\n{refs}\n"
        )
        path = self.wiki.write_article(
            title=query,
            content=content,
            subdir="analysis",
            slug=slug,
            page_type="analysis",
            depth="L1",
            tags=["analysis", "query"],
            l0_summary=answer[:120],
        )
        return path

    def _flag_contradiction(self, slug: str, contradiction: dict) -> None:
        """Add contradiction note to an existing wiki page."""
        subdir = "concepts"
        for sd in ("concepts", "entities", "summaries"):
            if (self.wiki.wiki_dir / sd / f"{slug}.md").exists():
                subdir = sd
                break
        existing = self.wiki.read_article(slug, subdir)
        if existing and "## Contradictions" not in existing:
            note = (
                f"\n\n## Contradictions\n\n"
                f"- **{contradiction.get('type', 'factual')}**: "
                f"{contradiction.get('wiki_claim', '')} ← "
                f"{contradiction.get('source_claim', '')}\n"
            )
            path = self.wiki.wiki_dir / subdir / f"{slug}.md"
            path.write_text(existing.rstrip() + note, encoding="utf-8")

    def _rebuild_index(self) -> None:
        self.wlog.rebuild_index(self.wiki)
        logger.info("index_rebuilt")

    def _rebuild_search(self) -> None:
        try:
            from kb.search.hybrid_search import HybridSearch
            self._search = HybridSearch(self.wiki)
            self._search.build()
            logger.info("search_index_rebuilt", docs=self._search.bm25._corpus.__len__())
        except Exception as e:
            logger.warning("search_rebuild_error", error=str(e))

    def _get_search(self) -> Any:
        if self._search is None:
            self._rebuild_search()
        return self._search

    def _build_concept_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for path in self.wiki.list_articles():
            content = path.read_text(encoding="utf-8")
            m = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
            title = m.group(1).strip().strip('"') if m else path.stem
            index[path.stem] = title
        return index

    def _guided_ingest_prompt(self, record: DocumentRecord) -> None:
        """Print guided ingest info for interactive sessions."""
        from rich.console import Console
        Console().print(
            f"\n[cyan]Next:[/cyan] [bold]{record.path}[/bold]  "
            f"[dim][{record.id[:8]}][/dim]"
        )

    def session_context(self, n: int = 10) -> str:
        """Return recent log entries for session startup context."""
        recent = self.wlog.recent_log(n)
        stats = self.ingestion.stats()
        articles = len(list(self.wiki.list_articles()))
        lines = [
            "# KB Session Context\n",
            f"Wiki articles: {articles}  |  "
            + "  |  ".join(f"{k}: {v}" for k, v in stats.items()),
            "\n## Recent operations\n",
        ] + recent
        return "\n".join(lines)
