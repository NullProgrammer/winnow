"""Synthetic PII values, and the hard negatives that make them learnable.

Two things make this file the heart of Phase 3.

**The positive classes follow Saurabh's rubric, not intuition.** Attribution is
not PII, so a name in a copyright header is a *negative* here and a name in a
data field is a positive. A public handle is a negative; a DSN account name is a
positive. Generating what the rubric calls PII — rather than what merely looks
like it — is the whole point.

**Hard negatives are drawn from measured false positives.** Every entry in
NEGATIVES below is a class the Phase 2 gold set caught the regex baseline
failing on, in real Go: `The X Authors`, T-SQL variables, SNMP OIDs, PEM
fixtures, snake_case filenames, URLs in key-named variables. These aren't
imagined edge cases, which is why they're worth ~40% of generated spans.

**Vocabularies are held out, not rows.** Train and eval draw from disjoint
domains, key prefixes, and placeholder words, so a model that memorised
`@gmail.com` scores no better than chance on the eval split. Holding out rows
instead would let memorisation pass for generalisation.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

# Site kinds a value can plausibly occupy (see sites.py).
STR = ("string",)
STR_RAW = ("string", "raw_string")
STR_COMMENT = ("string", "comment")
COMMENT = ("comment",)


@dataclass(frozen=True, slots=True)
class Value:
    text: str
    entity: str | None  # None means hard negative
    why: str            # provenance, so we can analyse which classes the model misses
    kinds: tuple[str, ...]


# --- held-out vocabularies ----------------------------------------------------
VOCAB: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "domains": ("gmail.com", "outlook.com", "proton.me", "icloud.com", "zoho.com"),
        "key_prefixes": ("AKIA", "ghp_", "sk_live_", "AIza"),
        "placeholders": ("changeme", "your_api_key", "insert_secret", "todo_fill_in"),
        "surnames": ("Okafor", "Lindqvist", "Batista", "Moreau", "Haruna", "Vasquez"),
        "givens": ("Adaeze", "Henrik", "Luciana", "Camille", "Ibrahim", "Rosalind"),
    },
    "eval": {
        "domains": ("yahoo.com", "fastmail.com", "hey.com", "gmx.net", "mail.ru"),
        "key_prefixes": ("ASIA", "github_pat_", "rk_live_", "SG."),
        "placeholders": ("replace_me", "yourpassword", "secret_here", "fixme_later"),
        "surnames": ("Nakamura", "Petrov", "Oyelaran", "Bergström", "Salazar", "Dlamini"),
        "givens": ("Yuki", "Anastasia", "Folake", "Sigrid", "Mateo", "Thandiwe"),
    },
}

ALNUM = string.ascii_letters + string.digits


def _rand(rng: random.Random, n: int, alphabet: str = ALNUM) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _person(rng: random.Random, v: dict) -> tuple[str, str]:
    return rng.choice(v["givens"]), rng.choice(v["surnames"])


# --- positives ----------------------------------------------------------------

def _email(rng: random.Random, v: dict) -> Value:
    given, sur = _person(rng, v)
    local = rng.choice(
        [
            f"{given.lower()}.{sur.lower()}",
            f"{given[0].lower()}{sur.lower()}",
            f"{given.lower()}{rng.randint(2, 89)}",
            f"{given.lower()}_{sur.lower()}",
        ]
    )
    return Value(f"{local}@{rng.choice(v['domains'])}", "EMAIL", "personal-email", STR_COMMENT)


def _key(rng: random.Random, v: dict) -> Value:
    p = rng.choice(v["key_prefixes"])
    body = _rand(rng, rng.randint(24, 40)) if not p.startswith("AKIA") else _rand(
        rng, 16, string.ascii_uppercase + string.digits
    )
    return Value(f"{p}{body}", "KEY", "provider-credential", STR_RAW)


def _password(rng: random.Random, v: dict) -> Value:
    if rng.random() < 0.25:
        # Passphrase: allowed to contain spaces, unlike a key.
        words = ("harbour", "lantern", "gravel", "muster", "tundra", "cobalt", "kestrel")
        return Value(
            " ".join(rng.sample(words, 4)), "PASSWORD", "passphrase", STR
        )
    body = _rand(rng, rng.randint(10, 18)) + rng.choice("!@#$%^&*")
    return Value(body, "PASSWORD", "generated-password", STR)


def _name(rng: random.Random, v: dict) -> Value:
    """A person's name as application DATA — not attribution.

    Per the rubric, a name in a copyright header or `@author` comment is not
    PII, so NAME positives only ever occupy data positions.
    """
    given, sur = _person(rng, v)
    return Value(f"{given} {sur}", "NAME", "person-as-data", STR)


def _username(rng: random.Random, v: dict) -> Value:
    """An account name on a live system — the rubric's only USERNAME positive."""
    given, sur = _person(rng, v)
    return Value(
        rng.choice([f"svc_{sur.lower()}", f"{given[0].lower()}{sur.lower()}", f"{sur.lower()}_admin"]),
        "USERNAME",
        "account-name",
        STR,
    )


def _ip(rng: random.Random, v: dict) -> Value:
    """Routable public address, skipping every reserved and documentation range."""
    while True:
        a = rng.randint(1, 223)
        b, c, d = (rng.randint(0, 255) for _ in range(3))
        if a in (10, 127, 0) or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
            continue
        if (a == 169 and b == 254) or (a == 192 and b == 0 and c == 2):
            continue
        if (a == 198 and b in (51, 18, 19)) or (a == 203 and b == 0 and c == 113):
            continue
        return Value(f"{a}.{b}.{c}.{d}", "IP", "public-address", STR_COMMENT)


POSITIVES = {
    "EMAIL": _email,
    "KEY": _key,
    "PASSWORD": _password,
    "NAME": _name,
    "USERNAME": _username,
    "IP": _ip,
}


# --- hard negatives, every one a measured Phase 2 false positive --------------

def _org_attribution(rng: random.Random, v: dict) -> Value:
    org = rng.choice(("Go", "Kubernetes", "Gitea", "etcd", "Prometheus", "Caddy"))
    return Value(f"Copyright {rng.randint(2012, 2026)} The {org} Authors", None, "org-attribution", COMMENT)


def _person_attribution(rng: random.Random, v: dict) -> Value:
    """A real person's name in a copyright header. NOT PII per the rubric."""
    given, sur = _person(rng, v)
    form = rng.choice(
        [f"Copyright {rng.randint(2012, 2026)} {given} {sur}", f"Author: {given} {sur}", f"@author {given} {sur}"]
    )
    return Value(form, None, "person-attribution", COMMENT)


def _handle(rng: random.Random, v: dict) -> Value:
    given, sur = _person(rng, v)
    h = f"{given[0].lower()}{sur.lower()}"
    return Value(
        rng.choice([f"owner: @{h}", f"reported by @{h}", f"TODO(@{h}): fix"]),
        None,
        "public-handle",
        COMMENT,
    )


def _placeholder_email(rng: random.Random, v: dict) -> Value:
    d = rng.choice(("example.com", "test.com", "example.org", "domain.com", "yourdomain.com", "fake.local"))
    return Value(f"{rng.choice(('user', 'test', 'jane', 'admin'))}@{d}", None, "placeholder-email", STR_COMMENT)


def _noreply_email(rng: random.Random, v: dict) -> Value:
    return Value(
        rng.choice(["noreply@github.com", f"{_rand(rng, 6).lower()}@users.noreply.github.com"]),
        None, "noreply-email", STR,
    )


def _org_mailbox(rng: random.Random, v: dict) -> Value:
    return Value(
        f"{rng.choice(('info', 'support', 'security', 'dev'))}@{rng.choice(('gitea.com', 'apache.org'))}",
        None, "org-mailbox", STR_COMMENT,
    )


def _env_var_name(rng: random.Random, v: dict) -> Value:
    return Value(
        rng.choice(("STRIPE_SECRET_KEY", "DB_PASSWORD", "AWS_ACCESS_KEY_ID", "GITHUB_TOKEN")),
        None, "env-var-name", STR,
    )


def _git_sha(rng: random.Random, v: dict) -> Value:
    return Value(_rand(rng, 40, "0123456789abcdef"), None, "git-sha", STR)


def _uuid(rng: random.Random, v: dict) -> Value:
    h = "0123456789abcdef"
    parts = [_rand(rng, n, h) for n in (8, 4, 4, 4, 12)]
    return Value("-".join(parts), None, "uuid", STR)


def _snake_filename(rng: random.Random, v: dict) -> Value:
    return Value(
        rng.choice(("authorized_principals", "known_hosts_file", "service_account_token")),
        None, "snake-case-filename", STR,
    )


def _url_in_key_var(rng: random.Random, v: dict) -> Value:
    return Value(
        f"https://{rng.choice(('api', 'auth'))}.{rng.choice(('vendor.io', 'service.dev'))}/v1/token",
        None, "url-as-secret", STR,
    )


def _sql_variable(rng: random.Random, v: dict) -> Value:
    name = rng.choice(("ErrorMessage", "SqlStatement", "Columns", "PCounters", "MajorMinorVersion"))
    form = rng.choice([f"DECLARE @{name} AS nvarchar(500)", f"SET @{name} = N''", f"FROM @{name} AS x"])
    return Value(form, None, "tsql-variable", STR_RAW)


def _snmp_oid(rng: random.Random, v: dict) -> Value:
    return Value(
        "." + ".".join(str(rng.randint(0, 9)) for _ in range(rng.randint(6, 11))),
        None, "snmp-oid", STR,
    )


def _reserved_ip(rng: random.Random, v: dict) -> Value:
    return Value(
        rng.choice(("127.0.0.1", "10.0.0.1", "192.168.1.1", "0.0.0.0", "192.0.2.1", "203.0.113.45", "8.8.8.8")),
        None, "reserved-or-doc-ip", STR_COMMENT,
    )


def _version_string(rng: random.Random, v: dict) -> Value:
    return Value(
        f"{rng.randint(1, 3)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
        None, "version-string", STR,
    )


def _placeholder_secret(rng: random.Random, v: dict) -> Value:
    return Value(rng.choice(v["placeholders"]), None, "placeholder-secret", STR)


def _template_ref(rng: random.Random, v: dict) -> Value:
    return Value(
        rng.choice(("${DB_PASS}", "{{ .Password }}", "<your-token-here>", "%s")),
        None, "template-ref", STR,
    )


def _service_default(rng: random.Random, v: dict) -> Value:
    return Value(
        rng.choice(("admin", "root", "postgres", "tomcat", "guest", "test")),
        None, "service-default", STR,
    )


def _app_display_name(rng: random.Random, v: dict) -> Value:
    """`DisplayName: "VS Code"` — looked like a name field, was an app."""
    return Value(
        rng.choice(("VS Code", "Gitea App", "Intellij IDEA", "Git Credential Manager")),
        None, "app-display-name", STR,
    )


def _constant_identifier(rng: random.Random, v: dict) -> Value:
    return Value(
        rng.choice(("manage_mfa", "proxy_user", "SignedUser", "change_username", "unauthorized_client")),
        None, "constant-identifier", STR,
    )


def _pem_header(rng: random.Random, v: dict) -> Value:
    """A PEM header is a real key shape; in a test file it's a fixture."""
    kind = rng.choice(("RSA ", "OPENSSH ", "EC ", ""))
    return Value(f"-----BEGIN {kind}PRIVATE KEY-----", None, "pem-test-fixture", STR_RAW)


NEGATIVES = (
    _org_attribution, _person_attribution, _handle, _placeholder_email, _noreply_email,
    _org_mailbox, _env_var_name, _git_sha, _uuid, _snake_filename, _url_in_key_var,
    _sql_variable, _snmp_oid, _reserved_ip, _version_string, _placeholder_secret,
    _template_ref, _service_default, _app_display_name, _constant_identifier, _pem_header,
)


def generate(rng: random.Random, split: str, negative_rate: float = 0.4) -> Value:
    """One synthetic value. `negative_rate` of them are PII-shaped non-PII."""
    v = VOCAB[split]
    if rng.random() < negative_rate:
        return rng.choice(NEGATIVES)(rng, v)
    return POSITIVES[rng.choice(list(POSITIVES))](rng, v)
