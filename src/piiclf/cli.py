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
    model_dir: Annotated[Path | None, typer.Option("--model",
        help="Use the fine-tuned model instead of the regex baseline.")] = None,
) -> None:
    """Scan a Go repository for PII and credential candidates. Values are always masked."""
    root = path.expanduser().resolve()
    if not root.is_dir():
        typer.secho(f"Not a directory: {root}", fg="red", err=True)
        raise typer.Exit(2)

    if model_dir is not None:
        if not model_dir.exists():
            typer.secho(f"No model at {model_dir} \u2014 run `piiclf train` first.", fg="red", err=True)
            raise typer.Exit(2)
        from .model_scan import scan_repo_with_model

        findings, scanned = scan_repo_with_model(root, model_dir, min_confidence, log=typer.echo)
    else:
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


@app.command()
def evaluate(
    set_name: Annotated[str, typer.Option("--set")] = "eval",
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = GOLD_ROOT,
    corpus_root: Annotated[Path, typer.Option("--corpus-root")] = CORPUS_ROOT,
) -> None:
    """Score the baseline against the hand-verified gold set."""
    from .evaluate import render, score, stratum_populations

    gold = gold_dir / "gold.jsonl"
    if not gold.exists():
        typer.secho(f"No {gold}. Run `piiclf label` first.", fg="red", err=True)
        raise typer.Exit(2)

    typer.echo("Recomputing stratum populations (needed to reweight the sample) ...")
    pops = stratum_populations(set_name, corpus_root)
    render(score(gold, pops), pops)


@app.command()
def generate(
    set_name: Annotated[str, typer.Option("--set", help="Corpus to inject into.")] = "train",
    target: Annotated[int, typer.Option("--target", help="Training windows to emit.")] = 30_000,
    seed: Annotated[int, typer.Option("--seed")] = 20260925,
    corpus_root: Annotated[Path, typer.Option("--corpus-root")] = CORPUS_ROOT,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("data/synth"),
    check: Annotated[int, typer.Option("--check", help="Windows to BIO-verify.")] = 300,
) -> None:
    """Generate the synthetic BIO-labeled training corpus."""
    from .dataset import build

    if set_name == "eval":
        typer.secho(
            "Refusing: the eval corpus is reserved for the gold set. Training on it "
            "would invalidate every Phase 4 number.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

    typer.echo(f"Injecting into the {set_name} corpus ...")
    summary = build(corpus_root, set_name, out_dir, target=target, seed=seed)

    for split, info in summary["splits"].items():
        typer.echo(f"\n{split}: {info['windows']} windows from {info['files_available']} files "
                   f"(vocab: {info['vocab']})")
        for label, n in list(info["label_counts"].items())[:8]:
            typer.echo(f"    {label:24} {n}")

    typer.echo(f"\nVerifying BIO alignment on {check} windows ...")
    failures = _verify_bio(out_dir / "train.jsonl", check)
    if failures:
        typer.secho(f"{len(failures)} alignment failures:", fg="red", err=True)
        for f in failures[:5]:
            typer.secho(f"  {f}", fg="red", err=True)
        raise typer.Exit(1)
    typer.secho(f"Alignment verified on {check} windows: 0 failures.", fg="green")


def _verify_bio(path: Path, limit: int) -> list[str]:
    """Round-trip every span through tokenization and back to characters."""
    from transformers import AutoTokenizer

    from .gold import read_jsonl
    from .tokenize_align import align, decode_spans, verify_alignment

    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    failures: list[str] = []

    for row in read_jsonl(path)[:limit]:
        try:
            aligned = align(row["text"], row["spans"], tok)
            verify_alignment(row["text"], row["spans"], aligned)
            # The round trip is the real test: labels must decode back to text
            # that contains the value that was injected.
            got = decode_spans(aligned["offset_mapping"], aligned["labels"])
            for span in row["spans"]:
                match = [d for d in got if d["label"] == span["label"]
                         and d["end"] > span["start"] and d["start"] < span["end"]]
                if not match:
                    continue
                recovered = row["text"][match[0]["start"]: match[0]["end"]]
                if span["text"] not in recovered:
                    failures.append(
                        f"{row['path']}: {span['label']} decoded {recovered!r} "
                        f"missing {span['text']!r}"
                    )
        except AssertionError as exc:
            failures.append(f"{row['path']}: {exc}")

    return failures


@app.command()
def train(
    train_path: Annotated[Path, typer.Option("--train")] = Path("data/synth/train.jsonl"),
    val_path: Annotated[Path, typer.Option("--val")] = Path("data/synth/val.jsonl"),
    out_dir: Annotated[Path, typer.Option("--out")] = Path("checkpoints/bio-modernbert"),
    epochs: Annotated[int, typer.Option("--epochs")] = 3,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
    lr: Annotated[float, typer.Option("--lr")] = 5e-5,
    max_length: Annotated[int, typer.Option("--max-length")] = 512,
    limit: Annotated[int, typer.Option("--limit", help="Use only the first N train windows.")] = 0,
    val_limit: Annotated[int, typer.Option("--val-limit")] = 0,
    markers: Annotated[bool, typer.Option("--markers", help="Phase 5: add structural prefixes.")] = False,
    seed: Annotated[int, typer.Option("--seed")] = 20260925,
) -> None:
    """Fine-tune the BIO token classifier on the synthetic corpus."""
    from .train import Config, run

    if not train_path.exists():
        typer.secho(f"No {train_path}. Run `piiclf generate` first.", fg="red", err=True)
        raise typer.Exit(2)

    cfg = Config(
        train_path=train_path,
        val_path=val_path,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        limit=limit or None,
        val_limit=val_limit or None,
        markers=markers,
        seed=seed,
    )
    result = run(cfg, log=typer.echo)
    typer.secho(f"\nBest macro-F1 (exact): {result['best_macro_f1_exact']:.3f}", fg="green")
    typer.echo(f"Checkpoint + history: {out_dir}")


@app.command("score-gold")
def score_gold(
    model_dir: Annotated[Path, typer.Option("--model")] = Path("checkpoints/bio-modernbert"),
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = GOLD_ROOT,
    corpus_root: Annotated[Path, typer.Option("--corpus-root")] = CORPUS_ROOT,
    set_name: Annotated[str, typer.Option("--set")] = "eval",
    survivors_only: Annotated[bool, typer.Option("--survivors-only",
        help="Score only spans that survive the baseline's own suppression \u2014 "
             "the only scope comparable to the structural-layer control.")] = False,
) -> None:
    """Score the trained model against the human-verified gold set."""
    from .score_gold import render, score

    gold = gold_dir / "gold.jsonl"
    for path, what in ((gold, "gold set (run `piiclf label`)"), (model_dir, "model (run `piiclf train`)")):
        if not path.exists():
            typer.secho(f"Missing {path} \u2014 {what}", fg="red", err=True)
            raise typer.Exit(2)

    typer.echo("Scoring the model on human-verified real Go ...")
    render(score(gold, corpus_root, set_name, model_dir, survivors_only, typer.echo), log=typer.echo)


if __name__ == "__main__":
    app()
