from __future__ import annotations

from piiclf.dataset import MAX_WINDOW_CHARS, _snap, windows
from piiclf.inject import Example, Span


def ex(text: str, spans: list[Span], distractors: list[Span] | None = None) -> Example:
    return Example("r", "a.go", text, spans, distractors or [])


class TestSnap:
    def test_backward_snaps_to_line_start(self):
        text = "aaa\nbbb\nccc"
        assert _snap(text, 5, forward=False) == 4

    def test_forward_snaps_past_newline(self):
        text = "aaa\nbbb\nccc"
        assert _snap(text, 5, forward=True) == 8

    def test_clamps_at_boundaries(self):
        text = "aaa\nbbb"
        assert _snap(text, 1, forward=False) == 0
        assert _snap(text, 5, forward=True) == len(text)


class TestWindows:
    def test_no_marks_yields_nothing(self):
        assert windows(ex("package main\n", [])) == []

    def test_single_span_produces_one_window_containing_it(self):
        text = "package main\n" + "// filler\n" * 20 + 'e := "yuki@hey.com"\n'
        v = "yuki@hey.com"
        s = text.index(v)
        out = windows(ex(text, [Span(s, s + len(v), "EMAIL", v)]))
        assert len(out) == 1
        w = out[0]
        assert w.text[w.spans[0].start : w.spans[0].end] == v

    def test_offsets_are_rebased_into_the_window(self):
        text = "// pad\n" * 60 + 'e := "a@b.com"\n'
        v = "a@b.com"
        s = text.index(v)
        w = windows(ex(text, [Span(s, s + len(v), "EMAIL", v)]))[0]
        assert w.spans[0].start < s, "offsets were not rebased"
        assert w.text[w.spans[0].start : w.spans[0].end] == v

    def test_far_apart_spans_split_into_separate_windows(self):
        filler = "// pad line here\n" * 200
        text = 'a := "one@x.com"\n' + filler + 'b := "two@y.com"\n'
        s1 = text.index("one@x.com")
        s2 = text.index("two@y.com")
        out = windows(ex(text, [
            Span(s1, s1 + 9, "EMAIL", "one@x.com"),
            Span(s2, s2 + 9, "EMAIL", "two@y.com"),
        ]))
        assert len(out) == 2
        for w in out:
            for sp in w.spans:
                assert w.text[sp.start : sp.end] == sp.text

    def test_nearby_spans_share_one_window(self):
        text = 'a := "one@x.com"\nb := "two@y.com"\n'
        s1, s2 = text.index("one@x.com"), text.index("two@y.com")
        out = windows(ex(text, [
            Span(s1, s1 + 9, "EMAIL", "one@x.com"),
            Span(s2, s2 + 9, "EMAIL", "two@y.com"),
        ]))
        assert len(out) == 1
        assert len(out[0].spans) == 2

    def test_windows_respect_the_char_cap(self):
        text = "// pad\n" * 500 + 'e := "a@b.com"\n'
        s = text.index("a@b.com")
        for w in windows(ex(text, [Span(s, s + 7, "EMAIL", "a@b.com")])):
            assert len(w.text) <= MAX_WINDOW_CHARS

    def test_windows_start_and_end_on_line_boundaries(self):
        text = "// pad\n" * 80 + 'e := "a@b.com"\n' + "// tail\n" * 80
        s = text.index("a@b.com")
        for w in windows(ex(text, [Span(s, s + 7, "EMAIL", "a@b.com")])):
            assert w.text.endswith("\n")
            assert text[: text.index(w.text)].endswith("\n") or text.startswith(w.text)

    def test_distractors_are_carried_and_rebased(self):
        text = "// pad\n" * 40 + '// Copyright 2019 Henrik Lindqvist\n'
        v = "Copyright 2019 Henrik Lindqvist"
        s = text.index(v)
        out = windows(ex(text, [], [Span(s, s + len(v), "person-attribution", v)]))
        assert len(out) == 1
        d = out[0].distractors[0]
        assert out[0].text[d.start : d.end] == v

    def test_every_carried_span_is_addressable(self):
        # The invariant that matters: rebasing must never corrupt a span.
        text = "".join(f'v{i} := "user{i}@hey.com"\n' for i in range(40))
        spans = []
        for i in range(40):
            v = f"user{i}@hey.com"
            s = text.index(v)
            spans.append(Span(s, s + len(v), "EMAIL", v))
        for w in windows(ex(text, spans)):
            for sp in w.spans:
                assert w.text[sp.start : sp.end] == sp.text
