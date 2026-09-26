"""Run the fine-tuned model over a real repository.

The regex path in `scan.py` proposes candidates from patterns; this proposes
them from the model instead. Output goes through the same masking in
`report.py`, so values are never echoed.

Files are cut into line-aligned character windows with overlap, because a Go
file is routinely longer than 512 tokens. Predictions from the overlapping
region are deduplicated by span, keeping the highest-confidence one — without
the overlap, an entity sitting on a window boundary would be split and missed.
"""

from __future__ import annotations

from pathlib import Path

from .entities import Entity, Finding
from .gold import _read
from .scan import iter_go_files
from .tokenize_align import align, decode_spans

WINDOW_CHARS = 1400
OVERLAP_CHARS = 250


def _windows(text: str) -> list[tuple[str, int]]:
    """Line-aligned windows with overlap. Returns (chunk, char_offset)."""
    if len(text) <= WINDOW_CHARS:
        return [(text, 0)]
    out: list[tuple[str, int]] = []
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + WINDOW_CHARS)
        nl = text.rfind("\n", pos + 1, end)
        if nl != -1 and end < len(text):
            end = nl + 1
        out.append((text[pos:end], pos))
        if end >= len(text):
            break
        pos = max(pos + 1, end - OVERLAP_CHARS)
    return out


def scan_repo_with_model(
    root: Path, model_dir: Path, min_confidence: float = 0.0, log=lambda *_: None
) -> tuple[list[Finding], int]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    files = iter_go_files(root)
    log(f"scanning {len(files)} Go files with {model_dir} on {device} ...")

    findings: list[Finding] = []
    scanned = 0

    for n, fp in enumerate(files, 1):
        text = _read(fp)
        if text is None:
            continue
        scanned += 1
        rel = fp.relative_to(root).as_posix()
        seen: dict[tuple[int, int, str], Finding] = {}

        for chunk, base in _windows(text):
            enc = align(chunk, [], tok, 512)
            with torch.no_grad():
                logits = model(
                    input_ids=torch.tensor([enc["input_ids"]]).to(device),
                    attention_mask=torch.tensor([enc["attention_mask"]]).to(device),
                ).logits[0]
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(-1).cpu().tolist()
            conf = probs.max(dim=-1).values.cpu().tolist()

            for span in decode_spans(enc["offset_mapping"], preds, chunk):
                # Confidence of the token that opened the span.
                idx = next(
                    (
                        i
                        for i, (s, e) in enumerate(enc["offset_mapping"])
                        if e > s and e > span["start"] and s < span["end"]
                    ),
                    None,
                )
                c = conf[idx] if idx is not None else 0.0
                if c < min_confidence:
                    continue

                abs_start, abs_end = base + span["start"], base + span["end"]
                line = text.count("\n", 0, abs_start) + 1
                col = abs_start - (text.rfind("\n", 0, abs_start) + 1) + 1
                key = (abs_start, abs_end, span["label"])
                cand = Finding(
                    entity=Entity(span["label"]),
                    value=text[abs_start:abs_end],
                    start=abs_start,
                    end=abs_end,
                    line=line,
                    column=col,
                    detector="model",
                    confidence=round(c, 3),
                    path=rel,
                )
                if key not in seen or cand.confidence > seen[key].confidence:
                    seen[key] = cand

        findings.extend(seen.values())
        if n % 200 == 0:
            log(f"  {n}/{len(files)} files, {len(findings)} findings")

    findings.sort(key=lambda f: (f.path, f.start))
    return findings, scanned
