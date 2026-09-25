"""Gold evaluation set: sampling, schema, and IO.

Reads `data/corpus/<set>.lock.json` with plain json so nothing here depends on
the corpus fetch tooling — that script is a deletable leaf.

Two sampling modes, and the distinction matters:

- **candidates** measures *precision*. It draws from the detector's own output,
  so it can only ever tell us how often the detector is right — never what it
  missed.
- **sweep** measures *recall*. Whole files are labeled exhaustively, so the
  *absence* of a label is meaningful. This is the only part reusable for
  scoring a future model, since a model may propose spans the candidate audit
  never adjudicated.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .baseline import detect, in_test_path
from .scan import GENERATED, MAX_FILE_BYTES, iter_go_files

LABELS = ("TRUE", "WRONG_SPAN", "WRONG_ENTITY", "FALSE", "UNSURE")

# Detectors collapsed into one stratum: 11 provider patterns that are all the
# same high-precision mechanism, so splitting them would waste budget.
PROVIDER_DETECTORS = frozenset({
    "aws-access-key", "github-pat", "github-pat-fine", "slack-token",
    "stripe-key", "google-api-key", "anthropic-key", "openai-key",
    "sendgrid-key", "jwt", "private-key-pem",
})


@dataclass(frozen=True, slots=True)
class Candidate:
    repo: str
    path: str
    line: int
    col: int
    start: int
    end: int
    entity: str
    detector: str
    confidence: float
    survives_suppression: bool
    in_test_path: bool
    stratum: str
    line_sha256: str
    span_len: int
    # Context is kept only in the intermediate candidates file, never in the
    # committed gold set.
    context: str = ""
    value: str = ""


@dataclass
class GoldRecord:
    repo: str
    path: str
    line: int
    col: int
    start: int
    end: int
    entity: str
    detector: str
    stratum: str
    survives_suppression: bool
    label: str
    line_sha256: str
    span_len: int
    corrected_entity: str | None = None
    corrected_start: int | None = None
    corrected_end: int | None = None
    labeler: str = "human"
    latency_ms: int = 0
    is_foil: bool = False
    notes: str = ""


def stratum_of(detector: str, survives: bool) -> str:
    """Stratify on (detector x survives-suppression).

    Both halves are needed. Sampling only the raw stream over-represents
    test-path findings, which suppression drops — measured on the eval corpus,
    285 raw PASSWORD candidates collapse to 3 survivors. Sampling only
    survivors would never reveal true positives the rules threw away.
    """
    base = "provider-key" if detector in PROVIDER_DETECTORS else detector
    return f"{base}|{'kept' if survives else 'dropped'}"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_lock(set_name: str, corpus_root: Path) -> dict:
    return json.loads((corpus_root / f"{set_name}.lock.json").read_text(encoding="utf-8"))


def _read(path: Path) -> str | None:
    """Canonical read. Pinned because offsets are a function of the decoder."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    return None if GENERATED.search(text[:4096]) else text


def _context(text: str, line_no: int, width: int = 2) -> str:
    lines = text.splitlines()
    lo = max(0, line_no - 1 - width)
    hi = min(len(lines), line_no + width)
    return "\n".join(f"{i + 1:6} | {lines[i]}" for i in range(lo, hi))


def collect_candidates(set_name: str, corpus_root: Path) -> list[Candidate]:
    """Every raw, pre-dedupe candidate across the corpus, flagged for survival."""
    out: list[Candidate] = []
    for repo in read_lock(set_name, corpus_root)["repos"]:
        repo_dir = corpus_root / set_name / repo["name"]
        for fp in iter_go_files(repo_dir):
            text = _read(fp)
            if text is None:
                continue
            rel = fp.relative_to(repo_dir).as_posix()

            raw = detect(text, rel, apply_suppression=False, dedupe=False)
            if not raw:
                continue
            kept = {
                (f.entity, f.start, f.end)
                for f in detect(text, rel, apply_suppression=True, dedupe=False)
            }
            lines = text.splitlines()

            for f in raw:
                survives = (f.entity, f.start, f.end) in kept
                line_text = lines[f.line - 1] if f.line - 1 < len(lines) else ""
                out.append(
                    Candidate(
                        repo=repo["name"],
                        path=rel,
                        line=f.line,
                        col=f.column,
                        start=f.start,
                        end=f.end,
                        entity=f.entity.value,
                        detector=f.detector,
                        confidence=f.confidence,
                        survives_suppression=survives,
                        in_test_path=in_test_path(rel),
                        stratum=stratum_of(f.detector, survives),
                        line_sha256=_sha256_text(line_text),
                        span_len=f.end - f.start,
                        context=_context(text, f.line),
                        value=f.value,
                    )
                )
    return out


def allocate(strata: dict[str, list], budget: int) -> dict[str, int]:
    """Equal allocation with redistribution, capped by stratum size.

    Equal — not proportional — because the goal is a usable precision estimate
    for *each* detector. Proportional allocation would spend the whole budget
    on `copyright` (4,972 of 11,030 raw candidates) and leave PASSWORD
    unmeasurable, which is the exact failure we're trying to avoid.
    """
    quotas = {k: 0 for k in strata}
    remaining = budget
    active = {k for k, v in strata.items() if v}

    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for k in sorted(active):
            if remaining <= 0:
                break
            room = len(strata[k]) - quotas[k]
            take = min(share, room, remaining)
            if take > 0:
                quotas[k] += take
                remaining -= take
                progressed = True
            if quotas[k] >= len(strata[k]):
                active.discard(k)
        if not progressed:
            break

    # Tiny strata are kept deliberately: a population of 1 drawn as a census
    # is exactly what guarantees the rare survivor cells get measured.
    return {k: v for k, v in quotas.items() if v > 0}


def sample_candidates(
    cands: list[Candidate], budget: int, seed: int
) -> tuple[list[Candidate], dict[str, int]]:
    rng = random.Random(seed)
    strata: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        strata[c.stratum].append(c)

    quotas = allocate(strata, budget)
    picked: list[Candidate] = []
    for name, quota in sorted(quotas.items()):
        picked.extend(rng.sample(strata[name], quota))
    rng.shuffle(picked)
    return picked, {k: len(v) for k, v in strata.items()}


def write_jsonl(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def candidate_rows(cands: list[Candidate]) -> list[dict]:
    return [asdict(c) for c in cands]
