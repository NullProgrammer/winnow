"""Score the baseline against the hand-verified gold set.

Reports **span-level** precision per detector stratum and per entity. Token-level
numbers would look far better and mean nothing.

Two honesty rules, both from the fact that this gold set is small:

- Every figure carries its denominator and a Wilson interval. A 0/11 is not
  "0% precision", it's "below roughly 25% with 95% confidence".
- Point estimates with n < 10 are marked not-reportable rather than printed as
  if they were measurements.

Precision comes from the candidate audit. **Recall is not computed here** — it
needs the exhaustive file sweep, because a candidate sample can only ever show
what the detector proposed, never what it missed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from .gold import collect_candidates, read_jsonl

POSITIVE = {"TRUE", "WRONG_SPAN"}  # right entity; span correctness reported separately
MIN_REPORTABLE = 10


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sanely at k=0 and k=n, unlike normal approx."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def stratum_populations(set_name: str, corpus_root: Path) -> dict[str, int]:
    """Full candidate counts per stratum, needed to reweight the sample.

    Equal allocation means sample shares don't match population shares, so an
    unweighted overall precision would badly over-represent tiny strata.
    """
    counts: dict[str, int] = defaultdict(int)
    for c in collect_candidates(set_name, corpus_root):
        counts[c.stratum] += 1
    return dict(counts)


def score(gold_path: Path, populations: dict[str, int]) -> dict:
    rows = read_jsonl(gold_path)

    by_stratum: dict[str, dict] = defaultdict(lambda: {"k": 0, "n": 0, "unsure": 0, "wrong_span": 0})
    for r in rows:
        s = by_stratum[r["stratum"]]
        if r["label"] == "UNSURE":
            s["unsure"] += 1
            continue
        s["n"] += 1
        if r["label"] in POSITIVE:
            s["k"] += 1
        if r["label"] == "WRONG_SPAN":
            s["wrong_span"] += 1

    # Reweight to the population: P = sum(N_s * p_s) / sum(N_s).
    by_entity: dict[str, dict] = defaultdict(lambda: {"num": 0.0, "den": 0.0, "k": 0, "n": 0})
    for stratum, s in by_stratum.items():
        if not stratum.endswith("|kept") or s["n"] == 0:
            continue
        entity = _entity_of(stratum, rows)
        N = populations.get(stratum, 0)
        e = by_entity[entity]
        e["num"] += N * (s["k"] / s["n"])
        e["den"] += N
        e["k"] += s["k"]
        e["n"] += s["n"]

    return {"strata": dict(by_stratum), "entities": dict(by_entity)}


def _entity_of(stratum: str, rows: list[dict]) -> str:
    for r in rows:
        if r["stratum"] == stratum:
            return r["entity"]
    return "?"


def render(result: dict, populations: dict[str, int], console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()

    t = Table(title="Baseline precision on suppression survivors (span-level, per detector)")
    for col, kw in (
        ("Stratum", {}), ("pop", {"justify": "right"}), ("n", {"justify": "right"}),
        ("correct", {"justify": "right"}), ("precision", {"justify": "right"}),
        ("95% CI", {"justify": "right"}), ("unsure", {"justify": "right"}),
    ):
        t.add_column(col, **kw)

    for stratum in sorted(result["strata"]):
        if not stratum.endswith("|kept"):
            continue
        s = result["strata"][stratum]
        if s["n"] == 0:
            continue
        lo, hi = wilson(s["k"], s["n"])
        p = f"{s['k'] / s['n']:.0%}" if s["n"] >= MIN_REPORTABLE else "[dim]n<10[/dim]"
        t.add_row(
            stratum.removesuffix("|kept"),
            str(populations.get(stratum, 0)),
            str(s["n"]),
            str(s["k"]),
            p,
            f"{lo:.0%}–{hi:.0%}",
            str(s["unsure"]) or "0",
        )
    console.print(t)

    t2 = Table(title="Per entity, reweighted to the full candidate population")
    for col, kw in (
        ("Entity", {}), ("pop", {"justify": "right"}), ("labeled", {"justify": "right"}),
        ("precision", {"justify": "right"}), ("95% CI", {"justify": "right"}),
    ):
        t2.add_column(col, **kw)

    for entity in sorted(result["entities"]):
        e = result["entities"][entity]
        if e["den"] == 0:
            continue
        lo, hi = wilson(e["k"], e["n"])
        p = f"{e['num'] / e['den']:.0%}" if e["n"] >= MIN_REPORTABLE else "[dim]n<10[/dim]"
        t2.add_row(entity, f"{int(e['den'])}", str(e["n"]), p, f"{lo:.0%}–{hi:.0%}")
    console.print(t2)

    kept = {k: v for k, v in result["strata"].items() if k.endswith("|kept") and v["n"]}
    tk = sum(v["k"] for v in kept.values())
    tn = sum(v["n"] for v in kept.values())
    lo, hi = wilson(tk, tn)
    console.print(
        f"\n[bold]Overall on survivors: {tk}/{tn} = {tk / tn:.0%}[/bold]  (95% CI {lo:.0%}–{hi:.0%})"
    )
    console.print(
        "[dim]Recall is not estimated here — it requires the exhaustive file sweep, "
        "since a candidate sample cannot show what the detector never proposed.[/dim]"
    )
