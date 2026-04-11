"""kb query — hybrid search + optional answer filing."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from kb.utils.config import settings

app = typer.Typer(help="Search the knowledge base.")
console = Console()


@app.callback(invoke_without_command=True)
def run(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    mode: str = typer.Option("hybrid", "--mode", "-m",
                              help="Search mode: hybrid|bm25|vector"),
    output: str = typer.Option(None, "--output", "-o", help="Save results to markdown file"),
    generate: bool = typer.Option(False, "--generate", "-g",
                                   help="LLM-synthesised answer"),
    file_answer: bool = typer.Option(False, "--file", "-f",
                                      help="File answer back to wiki/analysis/ (Karpathy pattern)"),
) -> None:
    """
    Search the knowledge base using hybrid BM25 + vector search.
    Use --file to save valuable answers back to the wiki (they compound into the knowledge base).
    """
    from kb.pipelines.orchestrator import Orchestrator

    wiki_path = Path(settings.wiki_dir)
    if not wiki_path.exists():
        console.print("[red]Wiki not found. Run 'kb compile' first.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Searching:[/bold blue] {query!r}  (mode={mode})")

    orch = Orchestrator(cfg=settings)
    result = orch.query_and_file(query, file_answer=(generate and file_answer), k=top_k)

    hits = result["hits"]
    if not hits:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[green]Found {len(hits)} result(s):[/green]\n")
    for i, h in enumerate(hits, 1):
        method_badge = f"[dim]{h.get('method', 'hybrid')}[/dim]"
        console.print(Panel(
            f"[bold]{h['title']}[/bold]\n"
            f"[dim]{h['slug']}[/dim]  {method_badge}\n\n"
            f"{h.get('snippet', '')[:200]}",
            title=f"[cyan]#{i} — score: {h['score']:.4f}[/cyan]",
            expand=False,
        ))

    if generate or file_answer:
        answer = result.get("answer", "")
        if answer:
            console.print("\n[bold blue]Synthesised Answer:[/bold blue]")
            console.print(Markdown(answer))
            if result.get("filed"):
                console.print(f"\n[green]✓ Filed to wiki:[/green] {result.get('filed_path')}")

    if output:
        lines = [f"# Query: {query}\n\n## Results\n"]
        for h in hits:
            lines.append(f"### {h['title']}\n**Score**: {h['score']:.4f}\n\n{h.get('snippet', '')}\n")
        if result.get("answer"):
            lines.insert(1, f"\n## Answer\n\n{result['answer']}\n")
        Path(output).write_text("\n".join(lines), encoding="utf-8")
        console.print(f"\n[green]Report saved: {output}[/green]")
