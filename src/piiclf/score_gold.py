"""Score the trained model against the human-verified gold set.

**What this can and cannot measure**, because the distinction decides whether
the number means anything.

The gold set adjudicated 150 spans that the *regex baseline* proposed. A model
that proposes some different span has no ground truth for it — scoring that as
a false positive would be unfair, and dropping it would be biased. So this does
not compute model precision.

What it does compute, and what actually matters:

- **False-positive suppression.** 147 of the 150 gold spans are human-verified
  NOT-PII — real mistakes the regex made on real Go. The baseline flags 100% of
  them by construction, since they are its own output. How many does the model
  also flag? Lower is better, and this is the central claim of the project.
- **Recall on verified positives.** Only 3 spans are genuine PII, so this is a
  spot check with a tiny denominator, reported with its n rather than as a rate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .gold import _read, read_jsonl
from .tokenize_align import ENTITIES, align, decode_spans

WINDOW = 700  # chars of context each side; keeps us inside 512 tokens


def _window(text: str, start: int, end: int) -> tuple[str, int]:
    lo = max(0, start - WINDOW)
    hi = min(len(text), end + WINDOW)
    nl = text.rfind("\n", 0, lo)
    lo = 0 if nl == -1 else nl + 1
    return text[lo:hi], lo


def score(gold_path: Path, corpus_root: Path, set_name: str, model_dir: Path,
          survivors_only: bool = False, log=print) -> dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    rows = read_jsonl(gold_path)
    cache: dict[tuple[str, str], str | None] = {}
    results = {"positives": [], "negatives": [], "skipped": 0}
    by_detector: dict[str, dict] = defaultdict(lambda: {"flagged": 0, "total": 0})

    for r in rows:
        if r["label"] == "UNSURE":
            results["skipped"] += 1
            continue
        # Survivors are the spans a user of the regex scanner actually sees.
        # Scoring against all 150 includes spans the baseline already
        # suppresses itself, which inflates the model's apparent advantage and
        # makes the number incomparable to the structural-layer control.
        if survivors_only and not r["survives_suppression"]:
            continue

        key = (r["repo"], r["path"])
        if key not in cache:
            cache[key] = _read(corpus_root / set_name / r["repo"] / r["path"])
        text = cache[key]
        if text is None or r["end"] > len(text):
            results["skipped"] += 1
            continue

        chunk, offset = _window(text, r["start"], r["end"])
        aligned = align(chunk, [], tok, 512)
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor([aligned["input_ids"]]).to(device),
                attention_mask=torch.tensor([aligned["attention_mask"]]).to(device),
            ).logits
        preds = logits.argmax(-1)[0].cpu().tolist()
        spans = decode_spans(aligned["offset_mapping"], preds, chunk)

        want_lo, want_hi = r["start"] - offset, r["end"] - offset
        hit = next(
            (s for s in spans if s["end"] > want_lo and s["start"] < want_hi),
            None,
        )
        record = {
            "path": f"{r['repo']}/{r['path']}",
            "detector": r["detector"],
            "gold_entity": r["entity"],
            "model_entity": hit["label"] if hit else None,
            "flagged": hit is not None,
        }

        if r["label"] in ("TRUE", "WRONG_SPAN"):
            results["positives"].append(record)
        else:
            results["negatives"].append(record)
            d = by_detector[r["detector"]]
            d["total"] += 1
            d["flagged"] += 1 if hit else 0

    results["by_detector"] = dict(by_detector)
    results["survivors_only"] = survivors_only
    return results


def render(results: dict, log=print) -> None:
    neg = results["negatives"]
    pos = results["positives"]
    flagged = sum(1 for n in neg if n["flagged"])

    scope = "SURVIVORS ONLY (what a user actually sees)" if results.get("survivors_only") \
        else "ALL gold negatives (includes ones the baseline self-suppresses)"
    log(f"\n=== False-positive suppression \u2014 {scope} ===")
    log(f"  {len(neg)} spans the regex baseline flagged and a human ruled NOT PII.")
    log(f"  Baseline flags: {len(neg)}/{len(neg)} (100% — they are its own output)")
    log(f"  Model flags:    {flagged}/{len(neg)} ({flagged/max(1,len(neg)):.0%})")
    log(f"  Suppressed:     {len(neg)-flagged}/{len(neg)} "
        f"({(len(neg)-flagged)/max(1,len(neg)):.0%})")

    log("\n  by detector (which regex mistakes the model still repeats):")
    for det, d in sorted(results["by_detector"].items(), key=lambda kv: -kv[1]["flagged"]):
        log(f"    {det:22} {d['flagged']:3}/{d['total']:3} flagged")

    log(f"\n=== Recall on verified positives (n={len(pos)}, spot check only) ===")
    for p in pos:
        mark = "FOUND" if p["flagged"] else "MISSED"
        log(f"  {mark:6} {p['gold_entity']:8} as {str(p['model_entity']):8} {p['path'][:52]}")

    if results["skipped"]:
        log(f"\n  {results['skipped']} spans skipped (unsure, or file changed)")
