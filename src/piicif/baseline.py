"""Regex + entropy baseline for PII in Go source.

This is deliberately the *honest* baseline the model has to beat. Where it is
weak — NAME in particular, which regex can only find in author/copyright
comments — that weakness is the measurement, not a bug to paper over.
"""

from __future__ import annotations

import re
from bisect import bisect_right

from .entities import Entity, Finding
from .entropy import has_low_variety, is_sequential, looks_random

# --- provider-prefixed credentials -------------------------------------------
# Shape patterns only. No real credential values appear in this file.
PROVIDER_KEY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("github-pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("github-pat-fine", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("stripe-key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("sendgrid-key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("private-key-pem", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
)

# --- Go assignment contexts ---------------------------------------------------
# Covers `x := "v"`, `x = "v"`, `const X = "v"`, struct literal field `X: "v"`,
# and map keys `"x": "v"`. Go raw strings use backticks, so both quote styles.
_GO_STRING = r"""(?:"(?P<dq>[^"\n\\]*(?:\\.[^"\n\\]*)*)"|`(?P<bq>[^`]*)`)"""

_KEY_IDENT = r"(?:api[_-]?key|apikey|access[_-]?key|secret[_-]?key|secret|token|credential|auth|private[_-]?key|client[_-]?secret|signing[_-]?key)"
_PASSWORD_IDENT = r"(?:password|passwd|pwd|pass_?phrase)"
_USERNAME_IDENT = r"(?:user_?name|user|login|account)"

KEY_ASSIGN = re.compile(
    rf"""(?ix) \b [A-Za-z0-9_]* {_KEY_IDENT} [A-Za-z0-9_]* \s* (?::=|=|:) \s* {_GO_STRING}"""
)
PASSWORD_ASSIGN = re.compile(
    rf"""(?ix) \b [A-Za-z_]* {_PASSWORD_IDENT} [A-Za-z0-9_]* \s* (?::=|=|:) \s* {_GO_STRING}"""
)
USERNAME_ASSIGN = re.compile(
    rf"""(?ix) \b [A-Za-z_]* {_USERNAME_IDENT} [A-Za-z0-9_]* \s* (?::=|=|:) \s* {_GO_STRING}"""
)

# Credentials embedded in connection strings: scheme://user:password@host
DSN_CREDS = re.compile(
    r"\b(?P<scheme>postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|ftp|https?)://"
    r"(?P<user>[^:/@\s\"'`]+):(?P<pw>[^@/\s\"'`]+)@"
)

EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
AT_HANDLE = re.compile(r"(?:^|\s)@(?P<handle>[A-Za-z0-9][A-Za-z0-9_\-]{2,38})\b")

# NAME is only tractable for regex inside attribution comments.
NAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("author-tag", re.compile(r"(?i)@author\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){1,2})")),
    ("author-comment", re.compile(r"(?i)//\s*(?:author|maintainer|written by|created by)\s*:?\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){1,2})")),
    ("copyright", re.compile(r"(?i)copyright\s*(?:\(c\)|©)?\s*(?:\d{4}(?:\s*-\s*\d{4})?)?\s*,?\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){1,2})")),
)

# --- suppression vocab --------------------------------------------------------
PLACEHOLDER_SUBSTRINGS = frozenset({
    "changeme", "change_me", "change-me", "placeholder", "example", "dummy",
    "sample", "redacted", "yourpassword", "your_password", "your-password",
    "your_api_key", "your-api-key", "yourapikey", "yourkey", "your_token",
    "insert_", "replace_", "secret_here", "todo", "fixme", "notreal", "fake",
    "foobar", "hunter2", "password123", "test123", "s3cret", "mysecret",
    "xxxx", "aaaa", "0000", "1234", "abcd",
})

PLACEHOLDER_EXACT = frozenset({
    "", "password", "passwd", "secret", "token", "key", "apikey", "api_key",
    "user", "username", "admin", "root", "test", "testing", "guest", "foo",
    "bar", "baz", "none", "null", "nil", "n/a", "na", "empty", "default",
    "postgres", "mysql", "mongo", "redis", "localhost", "unknown", "string",
})

PLACEHOLDER_DOMAINS = frozenset({
    "example.com", "example.org", "example.net", "test.com", "domain.com",
    "email.com", "yourdomain.com", "company.com", "acme.com", "foo.com",
    "localhost", "invalid", "sample.com", "mail.com", "somewhere.com",
})

# Paths whose contents are fixtures, not real data.
TEST_PATH_MARKERS = ("_test.go", "/testdata/", "/test/", "/tests/", "/mocks/",
                     "/mock/", "/fixtures/", "/examples/", "/example/", "_mock.go",
                     "_example.go", "/generated/", ".pb.go")

# PII-shaped but structurally not secrets.
GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
UUID = re.compile(r"(?i)\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
ENV_VAR_NAME = re.compile(r"\A[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\Z")
GO_SUM_HASH = re.compile(r"\Ah1:[A-Za-z0-9+/=]+\Z")
URL_VALUE = re.compile(r"(?i)\A(?:https?|wss?|ftp|grpc|file)://|\A//")
TEMPLATE_REF = re.compile(r"\A(?:\$\{[^}]*\}|\{\{[^}]*\}\}|<[^>]*>|%[sv]|\$[A-Z_]+)\Z")


def _is_placeholder(value: str) -> bool:
    v = value.strip().lower()
    if v in PLACEHOLDER_EXACT or TEMPLATE_REF.match(value.strip()):
        return True
    if any(s in v for s in PLACEHOLDER_SUBSTRINGS):
        return True
    if set(v) <= {"*", "x", "•", "."} and v:
        return True
    return has_low_variety(value) or is_sequential(value)


def _is_non_secret_shape(value: str) -> bool:
    return bool(
        GIT_SHA.match(value)
        or UUID.match(value)
        or ENV_VAR_NAME.match(value)
        or GO_SUM_HASH.match(value)
    )


def _is_plausible_key(value: str) -> bool:
    """Shape gate for the generic (non-provider-prefixed) key detector.

    Found by running the baseline over real Go: long GraphQL query/mutation
    literals and URLs assigned to identifiers like `authQuery` or `tokenURL`
    sail past the entropy test, because prose-ish text sits around 4.0
    bits/char. A credential is a single whitespace-free token of bounded
    length, so that alone removes the whole class.
    """
    if URL_VALUE.match(value):
        return False
    if any(ch.isspace() for ch in value):
        return False
    return 16 <= len(value) <= 200


def in_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return any(m in p for m in TEST_PATH_MARKERS)


class _LineIndex:
    """Char offset -> (line, column), both 1-based."""

    __slots__ = ("_starts",)

    def __init__(self, text: str) -> None:
        starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                starts.append(i + 1)
        self._starts = starts

    def locate(self, offset: int) -> tuple[int, int]:
        line = bisect_right(self._starts, offset)
        return line, offset - self._starts[line - 1] + 1


def _string_group(m: re.Match[str]) -> tuple[str, int, int] | None:
    """Return (value, start, end) for whichever quote style matched."""
    for name in ("dq", "bq"):
        if m.group(name) is not None:
            return m.group(name), m.start(name), m.end(name)
    return None


def detect(text: str, path: str = "", *, apply_suppression: bool = True) -> list[Finding]:
    """Find PII candidates in Go source.

    `apply_suppression=False` returns the raw candidate stream. Phase 5 needs
    that to measure how much of the precision comes from the suppression rules
    alone versus from the model.
    """
    idx = _LineIndex(text)
    out: list[Finding] = []

    def add(entity: Entity, value: str, start: int, end: int, detector: str, confidence: float) -> None:
        line, column = idx.locate(start)
        out.append(Finding(entity, value, start, end, line, column, detector, confidence, path))

    test_ctx = apply_suppression and in_test_path(path)

    for detector, pattern in PROVIDER_KEY_PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(0)
            if apply_suppression and _is_placeholder(value):
                continue
            # A provider prefix is strong evidence even in a test file, so we
            # keep it and only discount confidence.
            add(Entity.KEY, value, m.start(), m.end(), detector, 0.70 if test_ctx else 0.95)

    for m in KEY_ASSIGN.finditer(text):
        got = _string_group(m)
        if not got:
            continue
        value, s, e = got
        if apply_suppression and (_is_placeholder(value) or _is_non_secret_shape(value)):
            continue
        if apply_suppression and not _is_plausible_key(value):
            continue
        if apply_suppression and not looks_random(value):
            continue
        if test_ctx:
            continue
        add(Entity.KEY, value, s, e, "key-assignment", 0.60)

    for m in PASSWORD_ASSIGN.finditer(text):
        got = _string_group(m)
        if not got:
            continue
        value, s, e = got
        if apply_suppression and (_is_placeholder(value) or _is_non_secret_shape(value)):
            continue
        # Passphrases may contain spaces, so only newlines and absurd lengths
        # are disqualifying here — unlike keys, which allow no whitespace at all.
        if apply_suppression and ("\n" in value or len(value) > 100):
            continue
        if test_ctx:
            continue
        add(Entity.PASSWORD, value, s, e, "password-assignment", 0.75)

    # Tracked so the email detector can skip them. A DSN's `user:pass@host`
    # matches the email pattern, and EMAIL's mask reveals the first character —
    # which would leak a character of the password that PASSWORD's mask hides.
    dsn_spans: list[tuple[int, int]] = []

    for m in DSN_CREDS.finditer(text):
        dsn_spans.append((m.start(), m.end()))
        pw, s, e = m.group("pw"), m.start("pw"), m.end("pw")
        if apply_suppression and _is_placeholder(pw):
            continue
        if not test_ctx:
            add(Entity.PASSWORD, pw, s, e, "dsn-credentials", 0.85)
        user, us, ue = m.group("user"), m.start("user"), m.end("user")
        if not (apply_suppression and _is_placeholder(user)) and not test_ctx:
            add(Entity.USERNAME, user, us, ue, "dsn-credentials", 0.70)

    for m in EMAIL.finditer(text):
        if any(m.start() < de and m.end() > ds for ds, de in dsn_spans):
            continue
        value = m.group(0)
        domain = value.rsplit("@", 1)[-1].lower()
        if apply_suppression and (domain in PLACEHOLDER_DOMAINS or _is_placeholder(value)):
            continue
        if apply_suppression and value.lower().startswith(("noreply@", "no-reply@", "donotreply@")):
            continue
        add(Entity.EMAIL, value, m.start(), m.end(), "email", 0.60 if test_ctx else 0.90)

    for m in IPV4.finditer(text):
        value = m.group(0)
        octets = [int(o) for o in value.split(".")]
        if any(o > 255 for o in octets):
            continue
        if apply_suppression and _is_reserved_ip(octets):
            continue
        # A dotted quad on a line mentioning a version is almost always a version.
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line_text = text[line_start : line_end if line_end != -1 else len(text)]
        if apply_suppression and re.search(r"(?i)\bversion\b|\bv\d", line_text):
            continue
        add(Entity.IP, value, m.start(), m.end(), "ipv4", 0.55 if test_ctx else 0.80)

    for m in USERNAME_ASSIGN.finditer(text):
        got = _string_group(m)
        if not got:
            continue
        value, s, e = got
        if apply_suppression and (_is_placeholder(value) or _is_non_secret_shape(value)):
            continue
        if test_ctx:
            continue
        add(Entity.USERNAME, value, s, e, "username-assignment", 0.55)

    for m in AT_HANDLE.finditer(text):
        value = m.group("handle")
        if apply_suppression and (_is_placeholder(value) or value.lower() in {"param", "return", "author", "todo", "deprecated"}):
            continue
        if test_ctx:
            continue
        add(Entity.USERNAME, value, m.start("handle"), m.end("handle"), "at-handle", 0.45)

    for detector, pattern in NAME_PATTERNS:
        for m in pattern.finditer(text):
            value = m.group("name")
            if apply_suppression and _is_placeholder(value):
                continue
            add(Entity.NAME, value, m.start("name"), m.end("name"), detector, 0.75)

    return _dedupe(out)


def _is_reserved_ip(octets: list[int]) -> bool:
    a, b = octets[0], octets[1]
    return (
        a == 10
        or a == 127
        or a == 0
        or (a == 172 and 16 <= b <= 31)
        or (a == 192 and b == 168)
        or (a == 169 and b == 254)
        or 224 <= a <= 239
        or a >= 240
        or octets == [255, 255, 255, 255]
    )


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Keep the highest-confidence finding per (entity, span)."""
    best: dict[tuple[Entity, int, int], Finding] = {}
    for f in findings:
        k = (f.entity, f.start, f.end)
        if k not in best or f.confidence > best[k].confidence:
            best[k] = f
    return sorted(best.values(), key=lambda f: (f.start, f.entity.value))
