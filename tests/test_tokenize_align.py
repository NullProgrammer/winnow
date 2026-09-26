from __future__ import annotations

import pytest

from piiclf.tokenize_align import (
    ENTITIES,
    ID2LABEL,
    IGNORE_INDEX,
    LABEL2ID,
    LABELS,
    align,
    decode_spans,
    verify_alignment,
)

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


class TestLabelSet:
    def test_thirteen_labels(self):
        assert len(LABELS) == 1 + 2 * len(ENTITIES) == 13

    def test_ids_round_trip(self):
        assert all(ID2LABEL[LABEL2ID[label]] == label for label in LABELS)

    def test_o_is_zero(self):
        assert LABEL2ID["O"] == 0


class TestAlign:
    def test_single_span_gets_b_then_i(self, tok):
        text = 'email := "adaeze.okafor@gmail.com"\n'
        start = text.index("adaeze")
        spans = [{"start": start, "end": start + len("adaeze.okafor@gmail.com"), "label": "EMAIL"}]
        out = align(text, spans, tok)
        tags = [ID2LABEL[i] for i in out["labels"] if i != IGNORE_INDEX]
        assert "B-EMAIL" in tags
        assert tags.count("B-EMAIL") == 1, "one span must yield exactly one B-"

    def test_special_tokens_are_ignored(self, tok):
        out = align('x := "a"\n', [], tok)
        assert IGNORE_INDEX in out["labels"], "CLS/SEP must be masked"
        for i, (s, e) in enumerate(out["offset_mapping"]):
            if e <= s:
                assert out["labels"][i] == IGNORE_INDEX

    def test_leading_space_does_not_lose_the_first_token(self, tok):
        # Byte-level BPE folds the preceding space into the token, so a
        # containment test would drop the span's first token entirely.
        text = "// contact yuki.nakamura@fastmail.com today\n"
        start = text.index("yuki")
        spans = [{"start": start, "end": start + len("yuki.nakamura@fastmail.com"), "label": "EMAIL"}]
        out = align(text, spans, tok)
        verify_alignment(text, spans, out)

    def test_distractors_stay_o(self, tok):
        text = '// Copyright 2019 Henrik Lindqvist\n'
        spans = [{"start": 3, "end": len(text) - 1, "label": "person-attribution"}]
        out = align(text, spans, tok)
        tags = {ID2LABEL[i] for i in out["labels"] if i != IGNORE_INDEX}
        assert tags == {"O"}, "a distractor must not produce entity labels"

    def test_multiple_spans_all_labeled(self, tok):
        text = 'a := "mateo.salazar@hey.com"\nb := "203.0.113.9"\nc := "Yuki Nakamura"\n'
        spans = []
        for value, label in (
            ("mateo.salazar@hey.com", "EMAIL"),
            ("203.0.113.9", "IP"),
            ("Yuki Nakamura", "NAME"),
        ):
            s = text.index(value)
            spans.append({"start": s, "end": s + len(value), "label": label})
        out = align(text, spans, tok)
        verify_alignment(text, spans, out)
        tags = [ID2LABEL[i] for i in out["labels"] if i != IGNORE_INDEX]
        assert {"B-EMAIL", "B-IP", "B-NAME"} <= set(tags)

    def test_non_ascii_before_span(self, tok):
        text = '// ünïcödé héader\nemail := "sigrid.bergstrom@gmx.net"\n'
        v = "sigrid.bergstrom@gmx.net"
        spans = [{"start": text.index(v), "end": text.index(v) + len(v), "label": "EMAIL"}]
        out = align(text, spans, tok)
        verify_alignment(text, spans, out)


class TestWeights:
    def test_weights_sum_to_one_per_span(self, tok):
        text = 'k := "AKIAQ7X4MZLP2VNRT8KD"\n'
        v = "AKIAQ7X4MZLP2VNRT8KD"
        spans = [{"start": text.index(v), "end": text.index(v) + len(v), "label": "KEY"}]
        out = align(text, spans, tok)
        entity_w = [
            w for w, lab in zip(out["weights"], out["labels"])
            if lab not in (IGNORE_INDEX, LABEL2ID["O"])
        ]
        assert entity_w
        assert abs(sum(entity_w) - 1.0) < 1e-6, entity_w

    def test_long_span_does_not_outweigh_a_short_one(self, tok):
        """The PEM problem: without 1/len weighting, one long key contributes
        as much gradient as ~100 emails and teaches 'long base64 -> Key'."""
        long_v = "A" * 400
        short_v = "yuki@hey.com"
        out_long = align(f'k := "{long_v}"\n', [{"start": 6, "end": 6 + len(long_v), "label": "KEY"}], tok)
        out_short = align(f'e := "{short_v}"\n', [{"start": 6, "end": 6 + len(short_v), "label": "EMAIL"}], tok)

        def total(o):
            return sum(
                w for w, lab in zip(o["weights"], o["labels"])
                if lab not in (IGNORE_INDEX, LABEL2ID["O"])
            )

        assert abs(total(out_long) - total(out_short)) < 1e-6


class TestDecodeSpans:
    def test_round_trip_recovers_the_value(self, tok):
        text = 'e := "folake.oyelaran@fastmail.com"\n'
        v = "folake.oyelaran@fastmail.com"
        spans = [{"start": text.index(v), "end": text.index(v) + len(v), "label": "EMAIL"}]
        out = align(text, spans, tok)
        got = decode_spans(out["offset_mapping"], out["labels"])
        assert len(got) == 1
        assert got[0]["label"] == "EMAIL"
        assert v in text[got[0]["start"] : got[0]["end"]]

    def test_adjacent_different_entities_do_not_merge(self, tok):
        offsets = [(0, 0), (0, 5), (5, 10), (0, 0)]
        labels = [IGNORE_INDEX, LABEL2ID["B-EMAIL"], LABEL2ID["B-NAME"], IGNORE_INDEX]
        got = decode_spans(offsets, labels)
        assert [g["label"] for g in got] == ["EMAIL", "NAME"]

    def test_trims_leading_space_and_quotes(self):
        """Measured impact: without trimming, EMAIL exact-F1 was 0.27 against
        an overlap-F1 of 0.99 — the model found every email but the decoded
        span began at ` "` because byte-level BPE folds those into the first
        token."""
        text = 'e := "yuki@hey.com"\n'
        # A token range that includes the space, the opening quote, and the
        # closing quote — what the tokenizer actually produces.
        offsets = [(0, 0), (4, 6), (6, 18), (18, 19), (0, 0)]
        labels = [
            IGNORE_INDEX,
            LABEL2ID["B-EMAIL"],
            LABEL2ID["I-EMAIL"],
            LABEL2ID["I-EMAIL"],
            IGNORE_INDEX,
        ]
        untrimmed = decode_spans(offsets, labels)
        assert text[untrimmed[0]["start"] : untrimmed[0]["end"]] == ' "yuki@hey.com"'

        trimmed = decode_spans(offsets, labels, text)
        assert text[trimmed[0]["start"] : trimmed[0]["end"]] == "yuki@hey.com"

    def test_trimming_does_not_eat_a_clean_span(self):
        text = "abc yuki@hey.com def"
        offsets = [(4, 16)]
        got = decode_spans(offsets, [LABEL2ID["B-EMAIL"]], text)
        assert text[got[0]["start"] : got[0]["end"]] == "yuki@hey.com"

    def test_all_delimiter_span_is_dropped(self):
        got = decode_spans([(0, 3)], [LABEL2ID["B-EMAIL"]], '   ')
        assert got == []

    def test_empty_input(self):
        assert decode_spans([], []) == []

    def test_all_o_yields_nothing(self):
        assert decode_spans([(0, 3), (3, 6)], [LABEL2ID["O"], LABEL2ID["O"]]) == []


class TestVerifyCatchesBadAlignment:
    def test_wrong_span_offset_is_rejected(self, tok):
        text = 'e := "yuki@hey.com"\n'
        good = [{"start": 6, "end": 18, "label": "EMAIL"}]
        aligned = align(text, good, tok)
        # Same alignment, but claim the span sits somewhere it doesn't.
        with pytest.raises(AssertionError):
            verify_alignment(text, [{"start": 0, "end": 4, "label": "EMAIL"}], aligned)
