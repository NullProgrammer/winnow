"""Inject synthetic PII into real Go files and record where it landed.

Replacements are applied **left-to-right with a running delta**, and the order
genuinely matters. Right-to-left looks appealing because unprocessed sites keep
their original offsets — but it's wrong here: replacing a site then replacing
one to its *left* shifts every span already recorded to the right, silently
corrupting them. The span-equality gate in `verify` caught exactly that.

So: walk ascending, keep `delta` = (chars inserted - chars removed) so far, and
place each replacement at `site.start + delta`.

Positives produce labeled spans. Negatives produce **distractors**: text that
must receive O tags. Distractors are recorded (not labeled) so Phase 4 can
report a per-class false-positive rate against the classes Phase 2 measured.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .generators import Value, generate
from .sites import Site, find_sites

# Struct tags are metadata and never carry user data, so nothing is injected
# there — a tag is a hard-negative *position*, which the model sees for free by
# the tags already present in the file.
INJECTABLE_KINDS = ("string", "raw_string", "comment")

COMMENT_PREFIX = "// "


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    label: str  # entity name, or the negative's `why` for distractors
    # The value the generator produced. Carried so verification compares the
    # span against what was *intended*, not against the text it just read —
    # which would make the assertion vacuous. Synthetic throughout, so there's
    # no secret-in-a-file concern here (unlike the gold set).
    text: str


@dataclass
class Example:
    repo: str
    path: str
    text: str
    spans: list[Span] = field(default_factory=list)        # positives -> BIO
    distractors: list[Span] = field(default_factory=list)  # negatives -> expected O


def _fits(value: Value, site: Site) -> bool:
    return site.kind in value.kinds and site.kind in INJECTABLE_KINDS


def _replacement(value: Value, site: Site) -> tuple[str, int]:
    """Return (text to substitute, offset of the value within that text)."""
    if site.kind == "comment":
        # The site span covers the whole comment including `//`, so rebuild it
        # rather than destroying the marker.
        return COMMENT_PREFIX + value.text, len(COMMENT_PREFIX)
    return value.text, 0


def inject_file(
    repo: str,
    path: str,
    text: str,
    rng: random.Random,
    split: str,
    max_per_file: int = 4,
    negative_rate: float = 0.4,
) -> Example | None:
    """Replace a few sites in one file with synthetic values.

    Returns None when the file has no usable site, which is common for tiny or
    generated files.
    """
    sites = [s for s in find_sites(text, path) if s.kind in INJECTABLE_KINDS]
    if not sites:
        return None

    rng.shuffle(sites)
    chosen: list[tuple[Site, Value]] = []
    used: list[tuple[int, int]] = []

    for site in sites:
        if len(chosen) >= max_per_file:
            break
        # Sites can nest (a string inside a larger construct); overlapping
        # replacements would corrupt each other's offsets.
        if any(site.start < e and site.end > s for s, e in used):
            continue
        for _ in range(6):  # a few draws to find a value that fits this kind
            value = generate(rng, split, negative_rate)
            if _fits(value, site):
                chosen.append((site, value))
                used.append((site.start, site.end))
                break

    if not chosen:
        return None

    # Ascending, carrying the cumulative length change so far.
    chosen.sort(key=lambda cv: cv[0].start)

    out = text
    delta = 0
    spans: list[Span] = []
    distractors: list[Span] = []

    for site, value in chosen:
        sub, offset = _replacement(value, site)
        lo, hi = site.start + delta, site.end + delta
        out = out[:lo] + sub + out[hi:]
        delta += len(sub) - (site.end - site.start)

        # label_start/label_len let a compound value label only part of itself:
        # in `Joseph Watson <jw@x.us>` the name is not PII per the rubric but
        # the email is, and the model needs to see that mix.
        start = lo + offset + value.label_start
        labeled = value.labeled_text
        span = Span(start, start + len(labeled), value.entity or value.why, labeled)
        (spans if value.entity else distractors).append(span)

    spans.sort(key=lambda s: s.start)
    distractors.sort(key=lambda s: s.start)
    return Example(repo=repo, path=path, text=out, spans=spans, distractors=distractors)


def verify(example: Example) -> None:
    """Assert every recorded span addresses exactly the value that was injected.

    A hard assertion on purpose. A silent off-by-one here would poison every
    training label downstream, and the resulting model failure would look like
    a modelling problem rather than a data bug.

    The comparison is against `span.text` — what the generator produced — and
    never against a value re-read from `example.text`, which would compare the
    output to itself and always pass.
    """
    for s in example.spans + example.distractors:
        assert 0 <= s.start < s.end <= len(example.text), (
            f"span out of range: {s} in {example.path}"
        )
        got = example.text[s.start : s.end]
        assert got == s.text, f"span mismatch in {example.path}: {got!r} != {s.text!r}"

    # Overlapping labels would make BIO tagging ambiguous.
    ordered = sorted(example.spans + example.distractors, key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        assert a.end <= b.start, f"overlapping spans in {example.path}: {a} / {b}"


def inject_with_check(
    repo: str, path: str, text: str, rng: random.Random, split: str, **kw
) -> Example | None:
    """inject_file plus the span-equality gate, which is what callers want."""
    ex = inject_file(repo, path, text, rng, split, **kw)
    if ex is not None:
        verify(ex)
    return ex
