from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .report import render_table, to_json
from .scan import scan_repo

app = typer.Typer(
    add_completion=False,
    help="PII detection for Go codebases. Phase 1: regex + entropy baseline.",
)


# Keeps subcommands explicit (`piiclf scan ...`) instead of letting Typer
# collapse a lone command into the top level. Later phases add generate/train/evaluate.
@app.callback()
def main() -> None: ...


@app.command()
def scan(
    path: Annotated[Path, typer.Argument(help="Repository root to scan.")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Disable suppression rules and show the raw candidate stream. "
            "Used to measure how much precision the rules alone contribute.",
        ),
    ] = False,
    min_confidence: Annotated[float, typer.Option("--min-confidence")] = 0.0,
) -> None:
    """Scan a Go repository for PII and credential candidates. Values are always masked."""
    root = path.expanduser().resolve()
    if not root.is_dir():
        typer.secho(f"Not a directory: {root}", fg="red", err=True)
        raise typer.Exit(2)

    findings, scanned = scan_repo(root, apply_suppression=not raw)
    findings = [f for f in findings if f.confidence >= min_confidence]

    if json_out:
        typer.echo(to_json(findings))
    else:
        render_table(findings, scanned)


if __name__ == "__main__":
    app()
