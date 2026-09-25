from __future__ import annotations

import random

import pytest

from piiclf.inject import Example, Span, inject_file, inject_with_check, verify

GO = '''package main

// a leading comment
import "fmt"

const apiKey = "placeholder"
var host = "0.0.0.0"

type User struct {
	Email string `json:"email"`
}

func run(u User) {
	msg := "hello"
	note := "second"
	other := "third"
	fmt.Println(msg, note, other, apiKey, host)
}
'''


def rng(seed: int = 7) -> random.Random:
    return random.Random(seed)


class TestInjection:
    def test_produces_spans_and_text(self):
        ex = inject_with_check("r", "main.go", GO, rng(), "train")
        assert ex is not None
        assert ex.text != GO
        assert ex.spans or ex.distractors

    def test_every_span_addresses_its_injected_value(self):
        # The gate for the whole phase. Run many seeds: the offset arithmetic
        # only breaks on specific multi-replacement orderings.
        for seed in range(300):
            ex = inject_with_check("r", "main.go", GO, random.Random(seed), "train")
            if ex is None:
                continue
            for s in ex.spans + ex.distractors:
                assert ex.text[s.start : s.end] == s.text, (seed, s)

    def test_right_to_left_keeps_multiple_spans_valid(self):
        # Specifically exercises >1 replacement in one file, which is where
        # left-to-right offset shifting would corrupt the labels.
        multi = 0
        for seed in range(300):
            ex = inject_with_check("r", "main.go", GO, random.Random(seed), "train", max_per_file=4)
            if ex is None:
                continue
            if len(ex.spans) + len(ex.distractors) >= 2:
                multi += 1
                for s in ex.spans + ex.distractors:
                    assert ex.text[s.start : s.end] == s.text
        assert multi > 50, f"only {multi} multi-injection cases; test is too weak"

    def test_comment_marker_is_preserved(self):
        for seed in range(200):
            ex = inject_with_check("r", "main.go", GO, random.Random(seed), "train")
            if ex is None:
                continue
            for line in ex.text.splitlines():
                st = line.strip()
                # No line should have become bare text where a comment was.
                if st.startswith("Copyright") or st.startswith("Author:"):
                    pytest.fail(f"comment marker lost: {line!r}")

    def test_injected_file_still_parses_as_go(self):
        from piiclf.sites import find_sites

        for seed in range(100):
            ex = inject_with_check("r", "main.go", GO, random.Random(seed), "train")
            if ex is None:
                continue
            # find_sites would raise or return nothing coherent on a shredded file.
            assert find_sites(ex.text, "main.go")

    def test_positives_and_distractors_are_separated(self):
        entities = {"EMAIL", "KEY", "PASSWORD", "NAME", "USERNAME", "IP"}
        for seed in range(200):
            ex = inject_with_check("r", "main.go", GO, random.Random(seed), "train")
            if ex is None:
                continue
            assert all(s.label in entities for s in ex.spans)
            assert all(s.label not in entities for s in ex.distractors)

    def test_no_struct_tag_injection(self):
        for seed in range(200):
            ex = inject_with_check("r", "main.go", GO, random.Random(seed), "train")
            if ex is None:
                continue
            assert 'json:"email"' in ex.text, "struct tag was overwritten"

    def test_negative_rate_extremes(self):
        all_pos = inject_with_check("r", "main.go", GO, rng(), "train", negative_rate=0.0)
        assert all_pos is not None and not all_pos.distractors
        all_neg = inject_with_check("r", "main.go", GO, rng(), "train", negative_rate=1.0)
        assert all_neg is not None and not all_neg.spans

    def test_file_with_no_sites_returns_none(self):
        assert inject_file("r", "a.go", "package main\n", rng(), "train") is None

    def test_max_per_file_respected(self):
        for seed in range(100):
            ex = inject_with_check("r", "main.go", GO, random.Random(seed), "train", max_per_file=2)
            if ex is None:
                continue
            assert len(ex.spans) + len(ex.distractors) <= 2


class TestVerifyCatchesCorruption:
    def test_verify_rejects_a_wrong_offset(self):
        # Proves the gate isn't vacuous: it must fail on a deliberately bad span.
        ex = Example(repo="r", path="a.go", text='x := "value"\n',
                     spans=[Span(6, 11, "KEY", "value")])
        verify(ex)  # correct as constructed
        bad = Example(repo="r", path="a.go", text='x := "value"\n',
                      spans=[Span(5, 10, "KEY", "value")])
        with pytest.raises(AssertionError, match="span mismatch"):
            verify(bad)

    def test_verify_rejects_out_of_range(self):
        ex = Example(repo="r", path="a.go", text="short",
                     spans=[Span(0, 99, "KEY", "short")])
        with pytest.raises(AssertionError, match="out of range"):
            verify(ex)

    def test_verify_rejects_overlapping_spans(self):
        ex = Example(
            repo="r", path="a.go", text="abcdef",
            spans=[Span(0, 4, "KEY", "abcd"), Span(2, 6, "EMAIL", "cdef")],
        )
        with pytest.raises(AssertionError, match="overlapping"):
            verify(ex)
