"""kb session — show session context for LLM startup (recent log + wiki state)."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from kb.utils.config import settings

app = typer.Typer(help="Show session context (recent log + wiki state).")
console = Console()


@app.callback(invoke_without_command=True)
def run(
    n: int = typer.Option(10, "--last", "-n", help="Number of recent log entries"),
    generate_claude_md: bool = typer.Option(
        False, "--claude-md", help="Generate wiki/CLAUDE.md session startup file"
    ),
) -> None:
    """
    Show recent KB activity and wiki state.

    Use this to orient a new LLM session:
      kb session            — print context to terminal
      kb session --claude-md — write to wiki/CLAUDE.md for agent pickup
    """
    from kb.pipelines.orchestrator import Orchestrator
    from kb.storage.wiki_logger import WikiLogger
    from kb.storage.metadata_store import MetadataStore
    from pathlib import Path

    wlog = WikiLogger(settings.wiki_dir)
    store = MetadataStore(settings.db_path)

    recent = wlog.recent_log(n)
    stats = store.stats()
    wiki_path = Path(settings.wiki_dir)
    articles = len(list(wiki_path.rglob("*.md"))) if wiki_path.exists() else 0

    context = _build_context(recent, stats, articles)

    console.print(Panel(Markdown(context), title="[bold cyan]KB Session Context[/bold cyan]"))

    if generate_claude_md:
        claude_md_path = wiki_path / "CLAUDE.md"
        claude_md_path.write_text(context, encoding="utf-8")
        console.print(f"\n[green]✓ Written:[/green] {claude_md_path}")
        console.print("[dim]The LLM agent can read this file to orient itself at session start.[/dim]")


def _build_context(recent: list[str], stats: dict, articles: int) -> str:
    stat_line = "  |  ".join(f"{k}: {v}" for k, v in stats.items())
    lines = [
        "# KB Session Context\n",
        f"**Wiki**: {articles} articles  |  {stat_line}\n",
        "\n## How this wiki works\n",
        "Read `wiki/WIKI.md` for full schema. Quick summary:",
        "- `/wiki/concepts/` — concept and topic pages",
        "- `/wiki/entities/` — people, models, orgs, datasets",
        "- `/wiki/summaries/` — one page per ingested source",
        "- `/wiki/analysis/` — filed query answers",
        "- `wiki/index.md` — catalog (read this first when querying)",
        "- `wiki/log.md` — operation timeline",
        "\n## Recent operations\n",
    ] + recent + [
        "\n## Your next actions\n",
        "1. Read `wiki/index.md` to see what's in the wiki",
        "2. Read `wiki/wip.md` for work in progress",
        "3. Use `kb query <question>` to search",
        "4. Use `kb ingest` then `kb compile` to add new sources",
        "5. Use `kb lint` to health-check the wiki",
    ]
    return "\n".join(lines)
