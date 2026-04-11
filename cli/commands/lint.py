"""kb lint — validate wiki articles for quality."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kb.utils.config import settings

app = typer.Typer(help="Lint and validate wiki articles.")
console = Console()


@app.callback(invoke_without_command=True)
def run(
    path: str = typer.Option(None, "--path", "-p", help="Specific article or directory"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to auto-fix rule-based issues"),
    min_score: float = typer.Option(0.7, "--min-score", help="Minimum passing score"),
    output: str = typer.Option(None, "--output", "-o", help="Save report to file"),
) -> None:
    """
    Run the linting/QA agent against wiki articles.

    Checks for:
      - Missing frontmatter
      - Missing [[wikilinks]]
      - Article too short/long
      - Broken wikilinks
      - Style issues
    """
    from kb.agents.linting import LintingQAAgent
    from kb.agents.base import AgentInput
    from kb.services.llm.factory import get_default_provider
    from kb.storage.wiki_manager import WikiManager

    wiki = WikiManager(settings.wiki_dir)
    llm = get_default_provider()
    agent = LintingQAAgent(llm, confidence_threshold=min_score)

    # Collect all article slugs for link validation
    all_slugs = [p.stem for p in wiki.list_articles()]

    target_path = Path(path) if path else Path(settings.wiki_dir)
    if target_path.is_file():
        articles = [target_path]
    else:
        articles = list(target_path.rglob("*.md"))

    if not articles:
        console.print("[yellow]No articles found to lint.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold blue]Linting[/bold blue] {len(articles)} article(s) ...\n")

    results = []
    for art_path in articles:
        content = art_path.read_text(encoding="utf-8")
        inp = AgentInput(
            doc_id=art_path.stem,
            content=content,
            metadata={"known_slugs": all_slugs},
        )
        output_obj = agent.run(inp)
        results.append((art_path, output_obj))

    _print_lint_table(results, min_score)

    # Optionally save report
    if output:
        report_lines = ["# KB Lint Report\n"]
        for art_path, out in results:
            from kb.agents.linting import LintingQAAgent as LQA
            report_lines.append(f"## {art_path.stem}\n")
            report_lines.append(LQA.format_report(out) + "\n---\n")
        Path(output).write_text("\n".join(report_lines), encoding="utf-8")
        console.print(f"\n[green]Report saved: {output}[/green]")

    # Exit code
    failed = sum(1 for _, o in results if o.result.get("score", 0) < min_score)
    if failed:
        console.print(f"\n[red]{failed} article(s) below minimum score {min_score}[/red]")
        raise typer.Exit(1)
    else:
        console.print("\n[green]✓ All articles pass lint checks.[/green]")


def _print_lint_table(results: list, min_score: float) -> None:
    table = Table(title="Lint Results", show_header=True)
    table.add_column("Article", style="cyan", max_width=40)
    table.add_column("Score", justify="right")
    table.add_column("Issues", justify="right")
    table.add_column("Status", justify="center")

    for art_path, output in results:
        score = output.result.get("score", 0.0)
        issue_count = len(output.result.get("issues", []))
        status = "[green]✓ PASS[/green]" if score >= min_score else "[red]✗ FAIL[/red]"
        score_str = f"{score:.2f}"
        if score < min_score:
            score_str = f"[red]{score_str}[/red]"
        else:
            score_str = f"[green]{score_str}[/green]"
        table.add_row(art_path.stem, score_str, str(issue_count), status)

    console.print(table)
