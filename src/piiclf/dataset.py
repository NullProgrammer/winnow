"""Build the synthetic training corpus.

Files are split **by file**, not by window: two windows from the same file share
surrounding code, so splitting by window would leak train content into
validation and inflate the score.

The two splits also use **disjoint value vocabularies** (see generators.VOCAB).
Train draws `@gmail.com` / `AKIA…`, validation draws `@fastmail.com` /
`ASIA…`. A model that memorised a domain list therefore scores at chance on
validation, which is the whole point — otherwise memorisation reads as
generalisation.

Text is stored with character spans rather than token ids so the corpus stays
tokenizer-agnostic: Phase 4 compares ModernBERT against a code-pretrained
control, and baked-in ids would make that impossible.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .gold import _read
from .inject import Example, Span, inject_with_check
from .scan import iter_go_files

# ~1400 chars lands comfortably under 512 tokens for Go, which the MPS
# benchmark showed is 2.4x faster than 1024 for the same data.
MAX_WINDOW_CHARS = 1400
CONTEXT_PAD = 300


def _snap(text: str, pos: int, forward: bool) -> int:
    """Move to the nearest line boundary so windows never cut mid-line."""
    if forward:
        nl = text.find("\n", pos)
        return len(text) if nl == -1 else nl + 1
    nl = text.rfind("\n", 0, pos)
    return 0 if nl == -1 else nl + 1


def windows(ex: Example, max_chars: int = MAX_WINDOW_CHARS) -> list[Example]:
    """Cut an injected file into line-aligned windows around its spans.

    Spans that fall outside a window are dropped from it rather than clipped —
    a half-labeled entity is worse than an unlabeled one.
    """
    marks = sorted(ex.spans + ex.distractors, key=lambda s: s.start)
    if not marks:
        return []

    clusters: list[list[Span]] = [[marks[0]]]
    for m in marks[1:]:
        if m.end - clusters[-1][0].start <= max_chars - 2 * CONTEXT_PAD:
            clusters[-1].append(m)
        else:
            clusters.append([m])

    out: list[Example] = []
    for cluster in clusters:
        lo = _snap(ex.text, max(0, cluster[0].start - CONTEXT_PAD), forward=False)
        hi = _snap(ex.text, min(len(ex.text), cluster[-1].end + CONTEXT_PAD), forward=True)
        if hi - lo > max_chars:
            hi = _snap(ex.text, lo + max_chars, forward=False)
        chunk = ex.text[lo:hi]

        def shift(items: list[Span]) -> list[Span]:
            return [
                Span(s.start - lo, s.end - lo, s.label, s.text)
                for s in items
                if s.start >= lo and s.end <= hi
            ]

        spans, distractors = shift(ex.spans), shift(ex.distractors)
        if not spans and not distractors:
            continue
        out.append(Example(ex.repo, ex.path, chunk, spans, distractors))
    return out


def build(
    corpus_root: Path,
    set_name: str,
    out_dir: Path,
    target: int = 30_000,
    seed: int = 20260925,
    val_fraction: float = 0.1,
) -> dict:
    """Generate the train/validation corpora and return a summary."""
    lock = json.loads((corpus_root / f"{set_name}.lock.json").read_text(encoding="utf-8"))

    files: list[tuple[str, Path, str]] = []
    for repo in lock["repos"]:
        d = corpus_root / set_name / repo["name"]
        for fp in iter_go_files(d):
            files.append((repo["name"], fp, fp.relative_to(d).as_posix()))

    rng = random.Random(seed)
    rng.shuffle(files)
    cut = int(len(files) * (1 - val_fraction))
    partitions = {"train": files[:cut], "val": files[cut:]}

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"seed": seed, "corpus": set_name, "splits": {}}

    for split, subset in partitions.items():
        # Validation uses the held-out value vocabulary.
        vocab_split = "train" if split == "train" else "eval"
        quota = target if split == "train" else max(1, int(target * val_fraction))
        srng = random.Random(seed + (0 if split == "train" else 1))

        written = 0
        counts: dict[str, int] = {}
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for repo, fp, rel in _cycle(subset):
                if written >= quota:
                    break
                text = _read(fp)
                if text is None:
                    continue
                ex = inject_with_check(repo, rel, text, srng, vocab_split)
                if ex is None:
                    continue
                for w in windows(ex):
                    if written >= quota:
                        break
                    fh.write(
                        json.dumps(
                            {
                                "repo": w.repo,
                                "path": w.path,
                                "text": w.text,
                                "spans": [asdict(s) for s in w.spans],
                                "distractors": [asdict(s) for s in w.distractors],
                            }
                        )
                        + "\n"
                    )
                    written += 1
                    for s in w.spans:
                        counts[s.label] = counts.get(s.label, 0) + 1
                    for s in w.distractors:
                        counts[s.label] = counts.get(s.label, 0) + 1

        summary["splits"][split] = {
            "windows": written,
            "files_available": len(subset),
            "vocab": vocab_split,
            "label_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "path": str(path),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _cycle(items: list):
    """Repeat the file list, so a small corpus can still fill a large quota.

    Each pass re-injects different synthetic values into the same real code,
    which is the intended augmentation — the code is the scaffolding, the
    injected values are the signal.
    """
    while items:
        yield from items
