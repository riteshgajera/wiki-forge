# WIKI Schema & Operating Manual

> This document governs how the knowledge base is structured and maintained.
> Every LLM session should read this first. It is the source of truth for conventions.

## Philosophy

The wiki is a **persistent, compounding artifact**. When you add a new source,
you do not just create a new page — you integrate the knowledge into the existing
structure, updating entity pages, revising concept summaries, noting contradictions,
and strengthening cross-references. Knowledge accumulates here. It does not need
to be re-derived on every query.

The human curates sources and asks questions. The LLM does everything else.

---

## Directory Structure

```
wiki/
  WIKI.md              ← this file (read first every session)
  index.md             ← content catalog (one-line summary per page)
  log.md               ← append-only operation timeline
  wip.md               ← current work in progress
  concepts/            ← concept and topic pages
  entities/            ← people, organizations, models, datasets
  summaries/           ← per-source summary pages
  analysis/            ← filed query answers and synthesised reports
  _meta/
    sources.md         ← registry of all ingested sources
    CLAUDE.md          ← session startup context for LLM agents
```

---

## Page Types

### Concept pages (`/concepts/`)
Definitions, explanations, topic overviews. Updated each time a source adds
meaningful information about the concept. Should have:
- A clear one-paragraph definition at the top
- Key properties / characteristics
- Relationships to other concepts (with [[wikilinks]])
- Sources that discuss this concept (as a backlink list at the bottom)

### Entity pages (`/entities/`)
Named entities: people, organizations, models, datasets, tools.
Each entity gets exactly **one** page that accumulates all mentions across sources.
Fields: name, type, aliases, description, appearances, relationships.

### Summary pages (`/summaries/`)
One page per ingested source. Written immediately on ingest.
Contains: title, source path, date ingested, key points, extracted concepts,
entities mentioned, contradictions with existing wiki.

### Analysis pages (`/analysis/`)
Filed answers to queries. When a query produces a valuable synthesis,
it is saved here so it compounds into the knowledge base.
Named as: `YYYY-MM-DD-query-slug.md`

---

## Frontmatter Standard (L0/L1/L2)

Every wiki page MUST have YAML frontmatter with these fields:

```yaml
---
title: "Page Title"
type: concept|entity|summary|analysis|index
tags: [tag1, tag2]
created: 2026-04-07T10:00:00Z
updated: 2026-04-07T10:00:00Z
source: raw/articles/filename.md
confidence: 0.88
depth: L1
status: active|stale|contradiction
related: [concept-a, entity-b]
contradicts: []
---
```

### Depth levels (for context-efficient retrieval)

| Level | What it contains | When to load |
|-------|-----------------|--------------|
| L0 | Title + one-sentence summary (in frontmatter) | Always — used by index.md |
| L1 | Full page content | Standard queries |
| L2 | Deep reference with all sources, raw quotes | Deep-dive research |

The `l0_summary` frontmatter field provides the L0 layer without reading the body.

---

## Operations

### Ingest (`kb ingest` then `kb compile`)

When a new source arrives:

1. Parse and summarize the source
2. Write a summary page to `/summaries/`
3. Extract all entities → create or UPDATE entity pages in `/entities/`
4. Identify concepts discussed → UPDATE relevant concept pages in `/concepts/`
5. Check for contradictions with existing wiki content → flag in frontmatter
6. Update `/index.md` with new pages
7. Append to `/log.md`: `## [date] ingest | Source Title`
8. Update `/_meta/sources.md`

A single source ingest typically touches **10-15 wiki pages**.

### Query (`kb query`)

1. Read `index.md` to identify relevant pages (L0 scan)
2. Load identified pages (L1)
3. Synthesize answer with citations
4. If the answer is valuable → file to `/analysis/` with `kb query --file`
5. Append to `log.md`: `## [date] query | Question`

### Lint (`kb lint`)

Check for:
- Orphan pages (no inbound links)
- Stale pages (confidence decayed, source > 90 days old)
- Contradiction pages that need resolution
- Concepts mentioned but lacking their own page
- Missing entity pages for named entities in summaries
- `log.md` gaps (sources that were ingested but not integrated)

---

## Wikilink Conventions

- Use `[[slug|Display Name]]` for all cross-references
- Link on **first occurrence** only within a page
- Entity names → link to `/entities/entity-slug`
- Concept names → link to `/concepts/concept-slug`
- Source summaries → link with `[[summaries/source-slug|Source Title]]`

---

## log.md Format

Each entry MUST start with `## [YYYY-MM-DD] operation | title` for parseability.

```markdown
## [2026-04-07] ingest | Attention Is All You Need
## [2026-04-07] query | What is multi-head attention?
## [2026-04-07] lint | Weekly health check
## [2026-04-08] analysis | Transformer vs RNN comparison filed
```

To get last 5 entries: `grep "^## \[" wiki/log.md | tail -5`

---

## index.md Format

```markdown
# Index

## Concepts (N)
| Page | Summary | Tags | Updated |
|------|---------|------|---------|
| [[concepts/attention-mechanism\|Attention Mechanism]] | Computes weighted sums... | ml, nlp | 2026-04-07 |

## Entities (N)
...

## Summaries (N)
...

## Analysis (N)
...
```

---

## Contradiction Handling

When a new source contradicts an existing wiki claim:

1. Add `contradicts: [page-slug]` to both pages' frontmatter
2. Add a `## Contradictions` section to both pages noting the disagreement
3. Set `status: contradiction` on the affected section
4. The lint agent flags these for human review

---

## Knowledge Decay

Pages age. Confidence decays for:
- Claims sourced from documents > 90 days old without corroboration
- Entity pages with no updates in 60 days
- Concept pages that newer sources have not referenced

The lint agent flags decayed pages with `status: stale`.
