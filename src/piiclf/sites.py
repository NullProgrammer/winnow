"""Find syntactically valid injection sites in Go source, via tree-sitter.

**The coordinate problem is the whole reason this module exists.** tree-sitter
reports *byte* offsets; Python strings are indexed by *code point*; and the HF
tokenizer's `offset_mapping` is *character*-based. Any non-ASCII earlier in a
file — an accented name in a copyright header, a CJK string, an emoji in a
comment — silently shifts every downstream label if the three are mixed. So
every `Site` here carries **character** offsets into the decoded `str`, converted
once and asserted, and nothing outside this module ever sees a byte offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

from .baseline import in_test_path

# Content nodes exclude the surrounding quotes, which matches the annotation
# rubric ("spans exclude the quotes") for free.
STRING_CONTENT = {"interpreted_string_literal_content", "raw_string_literal_content"}

# Where a value can be swapped in. `struct_tag` is separated from `raw_string`
# because a tag is metadata, never user data — it's a hard-negative position.
KINDS = ("string", "raw_string", "struct_tag", "comment")

# Nodes whose name identifies what a string is *for*, used both to place values
# realistically and as the structural signal Phase 5 ablates.
_IDENT_PARENTS = {
    "const_spec", "var_spec", "short_var_declaration", "assignment_statement",
    "keyed_element", "field_declaration", "argument_list", "call_expression",
}


@dataclass(frozen=True, slots=True)
class Site:
    start: int          # character offset into the decoded str
    end: int            # character offset, exclusive
    kind: str
    ident: str | None   # nearest declaring identifier or map key, if any
    func: str | None    # enclosing function name
    in_test: bool

    @property
    def length(self) -> int:
        return self.end - self.start


@lru_cache(maxsize=1)
def go_parser() -> Parser:
    return Parser(Language(tsgo.language()))


def _byte_to_char(raw: bytes) -> callable:
    """Build a byte-offset -> char-offset converter for one file.

    Fast path: an all-ASCII file needs no conversion at all, which is the
    overwhelming majority of Go source. Otherwise build an explicit index so
    conversion stays O(1) per site rather than re-decoding a prefix each time.
    """
    if raw.isascii():
        return lambda b: b

    mapping: dict[int, int] = {}
    char = 0
    byte = 0
    for ch in raw.decode("utf-8", errors="replace"):
        mapping[byte] = char
        byte += len(ch.encode("utf-8"))
        char += 1
    mapping[byte] = char

    def convert(b: int) -> int:
        # Offsets always land on a character boundary for node starts/ends;
        # walk back only if a replacement char shifted things.
        while b not in mapping and b > 0:
            b -= 1
        return mapping.get(b, 0)

    return convert


def _enclosing_func(node) -> str | None:
    n = node
    while n is not None:
        if n.type in ("function_declaration", "method_declaration"):
            name = n.child_by_field_name("name")
            return name.text.decode("utf-8", "replace") if name else None
        n = n.parent
    return None


def _nearest_ident(node) -> str | None:
    """The identifier that names this string: a var/const name, field, or map key."""
    n = node.parent
    depth = 0
    while n is not None and depth < 6:
        if n.type in _IDENT_PARENTS:
            for field in ("name", "left", "key"):
                got = n.child_by_field_name(field)
                if got is not None:
                    return got.text.decode("utf-8", "replace").strip('"`')
            # const/var specs hold a bare identifier list
            for c in n.children:
                if c.type == "identifier":
                    return c.text.decode("utf-8", "replace")
            if n.type == "call_expression":
                fn = n.child_by_field_name("function")
                if fn is not None:
                    return fn.text.decode("utf-8", "replace")
        n = n.parent
        depth += 1
    return None


def find_sites(text: str, path: str = "") -> list[Site]:
    """Every replaceable string literal, struct tag, and comment in one file."""
    raw = text.encode("utf-8")
    tree = go_parser().parse(raw)
    to_char = _byte_to_char(raw)
    test = in_test_path(path)
    out: list[Site] = []

    def visit(node) -> None:
        kind: str | None = None

        if node.type in STRING_CONTENT:
            parent = node.parent
            grand = parent.parent if parent is not None else None
            if node.type == "raw_string_literal_content":
                kind = "struct_tag" if grand is not None and grand.type == "field_declaration" else "raw_string"
            else:
                kind = "string"
        elif node.type == "comment":
            kind = "comment"

        if kind is not None:
            start, end = to_char(node.start_byte), to_char(node.end_byte)
            if end > start:
                out.append(
                    Site(
                        start=start,
                        end=end,
                        kind=kind,
                        ident=_nearest_ident(node),
                        func=_enclosing_func(node),
                        in_test=test,
                    )
                )
            # A comment has no string children worth descending into.
            if kind == "comment":
                return

        for c in node.children:
            visit(c)

    visit(tree.root_node)
    return out


def verify_offsets(text: str, sites: list[Site]) -> None:
    """Assert every site's span is addressable in the original str.

    Deliberately an assertion, not a warning: a silent off-by-one here would
    corrupt every training label downstream and be nearly impossible to trace.
    """
    for s in sites:
        assert 0 <= s.start < s.end <= len(text), f"span out of range: {s}"
        # Only interpreted string literals are single-line. Raw strings and
        # struct tags use backticks and may span lines, which is valid Go.
        if s.kind == "string":
            assert "\n" not in text[s.start : s.end], f"interpreted string crosses a newline: {s}"
