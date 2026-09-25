from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from . import gold as gold_mod
from .report import render_table, to_json
from .scan import scan_repo

CORPUS_ROOT = Path("data/corpus")
GOLD_ROOT = Path("data/gold")

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


@app.command()
def sample(
    set_name: Annotated[str, typer.Option("--set")] = "eval",
    budget: Annotated[int, typer.Option("--budget", help="How many candidates to draw.")] = 150,
    seed: Annotated[int, typer.Option("--seed")] = 20260924,
    corpus_root: Annotated[Path, typer.Option("--corpus-root")] = CORPUS_ROOT,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = GOLD_ROOT,
    show_strata: Annotated[bool, typer.Option("--show-strata")] = True,
) -> None:
    """Draw a stratified candidate sample for labeling."""
    typer.echo(f"Collecting candidates across the {set_name} corpus ...")
    cands = gold_mod.collect_candidates(set_name, corpus_root)
    picked, sizes = gold_mod.sample_candidates(cands, budget, seed)

    if show_strata:
        typer.echo(f"\n{'stratum':44} {'pop':>6} {'drawn':>6}")
        drawn = {s: 0 for s in sizes}
        for c in picked:
            drawn[c.stratum] += 1
        for s in sorted(sizes, key=lambda k: -sizes[k]):
            mark = "  <-- kept by suppression" if s.endswith("|kept") else ""
            typer.echo(f"{s:44} {sizes[s]:6} {drawn[s]:6}{mark}")

    path = gold_mod.write_jsonl(gold_mod.candidate_rows(picked), out_dir / "candidates.jsonl")
    typer.echo(
        f"\n{len(cands)} raw candidates in {len(sizes)} strata; drew {len(picked)}.\nWrote {path}"
    )


@app.command()
def label(
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = GOLD_ROOT,
    limit: Annotated[int, typer.Option("--limit", help="Stop after N items.")] = 0,
) -> None:
    """Verify Claude's proposed labels, one keystroke each. Resumable."""
    from .labeler import run_session

    candidates = gold_dir / "candidates.jsonl"
    if not candidates.exists():
        typer.secho(f"No {candidates}. Run `piiclf sample` first.", fg="red", err=True)
        raise typer.Exit(2)

    run_session(
        candidates_path=candidates,
        gold_path=gold_dir / "gold.jsonl",
        prelabels_path=gold_dir / "prelabels.jsonl",
        limit=limit or None,
    )



if __name__ == "__main__":
    app()
