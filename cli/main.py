"""
KB CLI — main entry point.
Usage: kb <command> [options]
"""
from __future__ import annotations

import typer
from rich.console import Console

from kb.utils.config import settings
from kb.utils.logging import setup_logging

app = typer.Typer(
    name="kb",
    help="Local-first multi-agent knowledge base system",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _init() -> None:
    setup_logging(settings.log_level, settings.log_format)
    settings.ensure_dirs()


# Import and register sub-commands
from cli.commands import ingest as ingest_cmd  # noqa: E402
from cli.commands import compile as compile_cmd  # noqa: E402
from cli.commands import query as query_cmd  # noqa: E402
from cli.commands import lint as lint_cmd  # noqa: E402
from cli.commands import status as status_cmd  # noqa: E402
from cli.commands import session as session_cmd  # noqa: E402

app.add_typer(ingest_cmd.app, name="ingest")
app.add_typer(compile_cmd.app, name="compile")
app.add_typer(query_cmd.app, name="query")
app.add_typer(lint_cmd.app, name="lint")
app.add_typer(status_cmd.app, name="status")
app.add_typer(session_cmd.app, name="session")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """KB — local-first multi-agent knowledge base."""
    level = "DEBUG" if verbose else settings.log_level
    setup_logging(level, settings.log_format)
    settings.ensure_dirs()


if __name__ == "__main__":
    app()
