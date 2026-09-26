"""Character spans -> BIO token labels.

The other place in this project where a silent off-by-one would poison
everything, so the rules are spelled out rather than assumed:

**Overlap, not containment.** A token is labeled if it overlaps the span at all
(`tok_end > span_start and tok_start < span_end`). ModernBERT uses byte-level
BPE, so a token's offsets *include its leading space* — a containment test
(`tok_start >= span_start`) therefore mislabels the first token of almost every
span.

**Special tokens get -100.** `[CLS]`/`[SEP]`/padding all report offsets `(0, 0)`
and would otherwise be labeled `B-` for any span starting at 0.

**Weights are 1/span_length.** Plain token cross-entropy makes one ~500-token
PEM block contribute as much gradient as ~100 emails, which teaches exactly the
shape prior we're trying to remove ("long base64 -> Key"). Weighting each token
by the inverse of its span's token count makes every *entity* count once.

Alignment happens here rather than in the dataset file so the stored corpus stays
tokenizer-agnostic — Phase 4 wants to compare ModernBERT against a code-pretrained
control, which would be impossible with token ids baked in.
"""

from __future__ import annotations

ENTITIES = ("EMAIL", "IP", "KEY", "NAME", "PASSWORD", "USERNAME")

LABELS: tuple[str, ...] = ("O",) + tuple(
    f"{prefix}-{entity}" for entity in ENTITIES for prefix in ("B", "I")
)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

IGNORE_INDEX = -100


def align(
    text: str,
    spans: list[dict],
    tokenizer,
    max_length: int = 512,
) -> dict:
    """Tokenize `text` and project char spans onto BIO token labels.

    `spans` are dicts with `start`, `end`, `label` (an entity name). Returns a
    dict of input_ids / attention_mask / labels / weights.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = enc["offset_mapping"]
    o_id = LABEL2ID["O"]
    labels = [o_id if e > s else IGNORE_INDEX for s, e in offsets]
    weights = [1.0 if e > s else 0.0 for s, e in offsets]

    # Single pass per span using a moving pointer, rather than rescanning every
    # offset for every span. This runs on every training example, so the old
    # O(spans x tokens) scan was pure overhead.
    entity_spans = [s for s in sorted(spans, key=lambda s: s["start"]) if s["label"] in ENTITIES]
    cursor = 0
    for span in entity_spans:
        start, end, entity = span["start"], span["end"], span["label"]
        # Offsets are monotonic, so skip tokens that end before this span opens.
        while cursor < len(offsets) and (offsets[cursor][1] <= start or offsets[cursor][1] <= offsets[cursor][0]):
            cursor += 1

        member = []
        i = cursor
        while i < len(offsets):
            s, e = offsets[i]
            if e > s:
                if s >= end:
                    break
                if e > start:
                    member.append(i)
            i += 1

        if not member:
            continue
        b_id = LABEL2ID[f"B-{entity}"]
        i_id = LABEL2ID[f"I-{entity}"]
        w = 1.0 / len(member)
        for rank, idx in enumerate(member):
            labels[idx] = b_id if rank == 0 else i_id
            weights[idx] = w

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "weights": weights,
        "offset_mapping": offsets,
    }


def verify_alignment(text: str, spans: list[dict], aligned: dict) -> None:
    """Assert each in-window span produced a B- tag covering its own text.

    A hard assertion: the failure mode otherwise is a model that trains happily
    on labels pointing at the wrong characters.
    """
    offsets = aligned["offset_mapping"]
    labels = aligned["labels"]
    last_char = max((e for s, e in offsets if e > s), default=0)

    for span in spans:
        if span["label"] not in ENTITIES:
            continue
        if span["start"] >= last_char:
            continue  # truncated out of this window
        want = LABEL2ID[f"B-{span['label']}"]
        idx = [i for i, lab in enumerate(labels) if lab == want]
        assert idx, f"no B-{span['label']} emitted for span at {span['start']}"

        covered = [
            (offsets[i][0], offsets[i][1])
            for i, lab in enumerate(labels)
            if lab in (want, LABEL2ID[f"I-{span['label']}"])
        ]
        assert covered, f"span {span} produced no labeled tokens"
        lo = min(s for s, _ in covered)
        hi = max(e for _, e in covered)
        # Byte-level BPE may pull in a leading space, so the labeled region can
        # start slightly earlier than the span — but it must fully contain it.
        assert lo <= span["start"] and hi >= min(span["end"], last_char), (
            f"labeled region [{lo},{hi}) does not cover span "
            f"[{span['start']},{span['end']}) for {span['label']}"
        )


# Byte-level BPE folds the preceding space — and often the opening quote — into
# a span's first token, so a decoded span starts at ` "` rather than at the
# value. Without trimming, exact-match scoring fails on nearly every EMAIL and
# IP (measured: overlap F1 0.99 vs exact 0.27). The annotation rubric already
# says spans exclude quotes, so trimming matches the label definition.
TRIM_CHARS = " \t\r\n\"'`"


def decode_spans(
    offsets: list[tuple[int, int]], label_ids: list[int], text: str | None = None
) -> list[dict]:
    """BIO token labels -> character spans. Inverse of `align`, for scoring.

    Pass `text` to trim whitespace and quote delimiters off the span edges.
    """
    out: list[dict] = []
    cur: dict | None = None

    for (s, e), lid in zip(offsets, label_ids):
        if e <= s or lid == IGNORE_INDEX:
            continue
        label = ID2LABEL.get(lid, "O")
        if label == "O":
            if cur:
                out.append(cur)
                cur = None
            continue
        prefix, entity = label.split("-", 1)
        if prefix == "B" or cur is None or cur["label"] != entity:
            if cur:
                out.append(cur)
            cur = {"start": s, "end": e, "label": entity}
        else:
            cur["end"] = e

    if cur:
        out.append(cur)

    if text is None:
        return out

    trimmed: list[dict] = []
    for span in out:
        s, e = span["start"], span["end"]
        while s < e and text[s] in TRIM_CHARS:
            s += 1
        while e > s and text[e - 1] in TRIM_CHARS:
            e -= 1
        if e > s:
            trimmed.append({"start": s, "end": e, "label": span["label"]})
    return trimmed
