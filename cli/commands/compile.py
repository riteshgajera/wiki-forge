"""kb compile — run the full agent pipeline to generate wiki articles."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from kb.utils.config import settings

app = typer.Typer(help="Run the agent pipeline and generate wiki articles.")
console = Console()


@app.callback(invoke_without_command=True)
def run(
    path: str = typer.Option(None, "--path", "-p", help="Raw directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Reprocess already-done documents"),
    batch: int = typer.Option(None, "--batch", "-b", help="Batch size"),
    interactive: bool = typer.Option(False, "--interactive", "-i",
                                     help="Prompt for human approval on low-confidence outputs"),
    guided: bool = typer.Option(False, "--guided", "-g",
                                help="One source at a time guided mode (Karpathy pattern)"),
) -> None:
    """
    Compile raw documents into wiki articles.

    Runs: ingest → summarize → extract → link → integrate → entity pages → lint → write wiki
    New sources update EXISTING pages — not just create new ones.
    """
    from kb.pipelines.orchestrator import Orchestrator

    review_cb = _interactive_review if interactive else None

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=True) as progress:
        progress.add_task("Initialising orchestrator ...", total=None)
        orch = Orchestrator(cfg=settings, review_callback=review_cb, interactive=guided)

    console.print("[bold blue]Compiling knowledge base ...[/bold blue]")
    if guided:
        console.print("[cyan]Guided mode: one source at a time[/cyan]")

    stats = orch.compile(raw_dir=path, force=force, batch_size=batch, guided=guided)

    console.print("\n[bold green]✓ Compile complete[/bold green]")
    console.print(f"  Processed       : [bold]{stats['processed']}[/bold]")
    console.print(f"  Written         : [bold green]{stats['written']}[/bold green]")
    console.print(f"  Entities created: [bold cyan]{stats['entities_created']}[/bold cyan]")
    console.print(f"  Pages integrated: [bold cyan]{stats['pages_integrated']}[/bold cyan]")
    console.print(f"  Review queue    : [bold yellow]{stats['review']}[/bold yellow]")
    console.print(f"  Failed          : [bold red]{stats['failed']}[/bold red]")
    console.print(f"\nWiki: [cyan]{settings.wiki_dir}[/cyan]  |  Log: [dim]wiki/log.md[/dim]")


def _interactive_review(doc_id: str, output) -> bool:
    import json as _json
    console.print(f"\n[yellow]Low confidence ({output.confidence:.2f}): {doc_id}[/yellow]")
    if output.result:
        console.print(_json.dumps(output.result, indent=2)[:600])
    return typer.confirm("Approve?", default=False)

