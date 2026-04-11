"""kb ingest — scan raw directory and queue documents for processing."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from kb.ingestion.ingestion_engine import IngestionEngine
from kb.utils.config import settings

app = typer.Typer(help="Ingest raw documents into the processing queue.")
console = Console()


@app.callback(invoke_without_command=True)
def run(
    path: str = typer.Option(None, "--path", "-p", help="Directory to scan (default: raw_dir from config)"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch directory for changes"),
    show_stats: bool = typer.Option(True, "--stats/--no-stats", help="Show ingestion stats"),
) -> None:
    """Scan raw directory for new/changed files and queue them for processing."""
    raw_path = path or settings.raw_dir
    engine = IngestionEngine(settings)

    console.print(f"[bold blue]Scanning[/bold blue] {raw_path} ...")
    queued = engine.scan(raw_path)

    if not queued:
        console.print("[green]✓ Nothing new to ingest.[/green]")
    else:
        console.print(f"[green]✓ Queued [bold]{len(queued)}[/bold] documents.[/green]")
        if show_stats:
            _print_queued_table(queued)

    if watch:
        console.print(f"\n[yellow]Watching {raw_path} every {settings.ingestion.watch_interval_seconds}s ...[/yellow]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

        def on_new(docs):
            console.print(f"[blue]New docs:[/blue] {len(docs)} queued")

        engine.watch(raw_path, on_new=on_new)


def _print_queued_table(records: list) -> None:
    table = Table(title="Queued Documents", show_header=True)
    table.add_column("Path", style="cyan", max_width=60)
    table.add_column("Hash (8)", style="dim")
    table.add_column("Size", justify="right")
    for r in records[:20]:
        size = f"{r.file_size // 1024}KB" if r.file_size > 1024 else f"{r.file_size}B"
        table.add_row(r.path, r.id[:8], size)
    if len(records) > 20:
        table.add_row(f"... and {len(records)-20} more", "", "")
    console.print(table)
