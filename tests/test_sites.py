from __future__ import annotations

from piiclf.sites import find_sites, verify_offsets

GO = '''package main

// Author: someone here
import "fmt"

const apiKey = "secret-value"

type User struct {
	Email string `json:"email"`
}

func handle(u User) {
	msg := "hello there"
	raw := `multi
line`
	m := map[string]string{"contact": "a@b.com"}
	fmt.Println(msg, raw, m, apiKey)
}
'''


def sites_by_kind(text: str, path: str = "main.go"):
    out: dict[str, list] = {}
    for s in find_sites(text, path):
        out.setdefault(s.kind, []).append(s)
    return out


class TestFindSites:
    def test_finds_each_kind(self):
        got = sites_by_kind(GO)
        assert set(got) == {"string", "raw_string", "struct_tag", "comment"}

    def test_spans_exclude_the_delimiters(self):
        # The rubric says spans exclude quotes; the grammar's *_content nodes
        # give that for free. Checked by asserting the delimiter sits just
        # OUTSIDE the span — a struct tag's content legitimately contains
        # quotes inside it (`json:"email"`), so checking the span's own edges
        # would be wrong.
        delims = {"string": '"', "raw_string": "`", "struct_tag": "`"}
        for s in find_sites(GO, "main.go"):
            if s.kind not in delims:
                continue
            d = delims[s.kind]
            assert GO[s.start - 1] == d, f"{s.kind} open delimiter inside span"
            assert GO[s.end] == d, f"{s.kind} close delimiter inside span"

    def test_string_content_is_exact(self):
        spans = {GO[s.start : s.end] for s in find_sites(GO, "main.go") if s.kind == "string"}
        assert "secret-value" in spans
        assert "hello there" in spans

    def test_struct_tag_is_not_classified_as_raw_string(self):
        got = sites_by_kind(GO)
        assert [GO[s.start : s.end] for s in got["struct_tag"]] == ['json:"email"']
        assert all("json:" not in GO[s.start : s.end] for s in got["raw_string"])

    def test_ident_is_captured(self):
        idents = {GO[s.start : s.end]: s.ident for s in find_sites(GO, "main.go")}
        assert idents["secret-value"] == "apiKey"
        assert idents["hello there"] == "msg"

    def test_enclosing_function(self):
        funcs = {GO[s.start : s.end]: s.func for s in find_sites(GO, "main.go")}
        assert funcs["hello there"] == "handle"
        assert funcs["secret-value"] is None  # package level

    def test_test_path_flag(self):
        assert all(s.in_test for s in find_sites(GO, "pkg/thing_test.go"))
        assert not any(s.in_test for s in find_sites(GO, "pkg/thing.go"))

    def test_empty_and_malformed_input(self):
        assert find_sites("", "a.go") == []
        # tree-sitter error-recovers rather than raising, so this must not crash.
        find_sites("package main\nfunc broken( {{{", "a.go")


class TestOffsetCorrectness:
    def test_non_ascii_earlier_in_file_does_not_shift_spans(self):
        # The bug this module exists to prevent: tree-sitter returns byte
        # offsets, so a multi-byte char earlier in the file would push every
        # later span right if bytes were used as char indices.
        text = 'package main\n\n// Copyright 2020 José Müller — all rights\nconst k = "target"\n'
        spans = [text[s.start : s.end] for s in find_sites(text, "a.go")]
        assert "target" in spans

    def test_emoji_before_target(self):
        text = 'package main\n\n// 🎉🎉 launch\nconst k = "target"\n'
        assert "target" in [text[s.start : s.end] for s in find_sites(text, "a.go")]

    def test_cjk_string_before_target(self):
        text = 'package main\n\nvar a = "日本語のテキスト"\nconst k = "target"\n'
        spans = [text[s.start : s.end] for s in find_sites(text, "a.go")]
        assert "target" in spans
        assert "日本語のテキスト" in spans

    def test_verify_offsets_passes_on_real_shapes(self):
        for text in (GO, 'package main\n// ünïcödé\nvar x = "y"\n'):
            verify_offsets(text, find_sites(text, "a.go"))

    def test_ascii_fast_path_and_slow_path_agree(self):
        ascii_text = 'package main\n\nconst k = "target"\n'
        wide_text = 'package main\n\n// ü\nconst k = "target"\n'
        a = [ascii_text[s.start : s.end] for s in find_sites(ascii_text, "a.go")]
        b = [wide_text[s.start : s.end] for s in find_sites(wide_text, "a.go")]
        assert "target" in a and "target" in b
