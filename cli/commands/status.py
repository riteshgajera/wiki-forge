"""kb status — show processing pipeline status."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from kb.utils.config import settings

app = typer.Typer(help="Show knowledge base processing status.")
console = Console()


@app.callback(invoke_without_command=True)
def run(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show per-document details"),
    status_filter: str = typer.Option(None, "--status", "-s",
                                      help="Filter by status: pending|processing|done|failed|review"),
) -> None:
    """Show KB ingestion and processing status."""
    from kb.storage.metadata_store import MetadataStore
    from kb.storage.wiki_manager import WikiManager

    store = MetadataStore(settings.db_path)
    stats = store.stats()

    # Summary panel
    total = sum(stats.values())
    console.print(f"\n[bold]Knowledge Base Status[/bold]  (db: {settings.db_path})\n")

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Bar")

    status_styles = {
        "done": "green", "pending": "yellow", "review": "yellow",
        "processing": "blue", "failed": "red",
    }
    for status, count in sorted(stats.items()):
        bar = "█" * min(count, 40)
        style = status_styles.get(status, "white")
        table.add_row(
            f"[{style}]{status}[/{style}]",
            f"[{style}]{count}[/{style}]",
            f"[dim]{bar}[/dim]",
        )
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "")
    console.print(table)

    # Wiki article count
    wiki_path = settings.wiki_path
    if wiki_path.exists():
        article_count = len(list(wiki_path.rglob("*.md")))
        console.print(f"\n[cyan]Wiki articles:[/cyan] {article_count}  (in {settings.wiki_dir})")

    if detailed:
        _show_detailed(store, status_filter)


def _show_detailed(store, status_filter: str | None) -> None:
    records = store.list_all() if not status_filter else store.list_by_status(status_filter)

    table = Table(title="Documents", show_header=True)
    table.add_column("Status", style="dim", width=12)
    table.add_column("Path", style="cyan", max_width=50)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Wiki Path", style="dim", max_width=40)

    for rec in records[:50]:
        conf = f"{rec.confidence:.2f}" if rec.confidence else "-"
        table.add_row(rec.status, rec.path, conf, rec.wiki_path or "-")

    if len(records) > 50:
        console.print(f"[dim]Showing 50 of {len(records)} records.[/dim]")
    console.print(table)
