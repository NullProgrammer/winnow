# winnow

PII and credential detection for **Go codebases** — not prose.

Most open PII models (Piiranha, GLiNER-PII, Horizon Labs' redactor) are trained on natural language. Code is a different problem: the signal separating a real secret from a placeholder is mostly *structural* — is the string in a `_test.go` file? a struct tag? an `os.Getenv` fallback? a GraphQL literal? This project builds a small encoder model that learns that distinction, with a regex + entropy baseline as the honest bar to beat.

**Status: Phase 1 complete** — the baseline scanner works end to end. No ML yet. Full roadmap in [PLAN.md](PLAN.md).

---

## Install

From scratch on a clean machine:

```bash
# 1. install uv, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. from the repo root
cd winnow

# 3. create the venv (uv downloads Python 3.12 itself if needed)
uv venv --python 3.12 .venv

# 4. install the project and its test deps
uv pip install -e ".[dev]"

# 5. activate — every command below assumes this
source .venv/bin/activate
```

Takes under a minute; Phase 1 pulls only `typer`, `rich`, and `pathspec`.

**If you skip step 5**, prefix commands with the venv path instead — `.venv/bin/piiclf scan ...` and `.venv/bin/python -m pytest`. Nothing here needs the venv activated, it's just shorter.

The ML stack is a separate extra so the scanner stays fast to install. You don't need it for Phase 1:

```bash
uv pip install -e ".[ml]"   # needed from Phase 3 onward (~2.5 GB, pulls torch)
```

### Verify the install

```bash
pytest -q                          # expect: 31 passed
piiclf scan {{repo}}
```

### If `piiclf` isn't in `.venv/bin/`

The `piiclf` command is created **only** by step 4. If it's missing, that step didn't succeed. Two common reasons:

1. **zsh ate the extra.** Unquoted, `[dev]` is a glob pattern and zsh refuses the command before uv runs:
   ```
   $ uv pip install -e .[dev]
   zsh:1: no matches found: .[dev]
   ```
   Always quote it: `uv pip install -e ".[dev]"`. Works unquoted on bash, fails on zsh — which is why this bites on some machines and not others.
2. **Wrong Python.** `requires-python = ">=3.12"`, so the install errors if `uv venv` ran without `--python 3.12`.

**Fallback that needs no install at all** — only `typer`, `rich`, and `pathspec`:

```bash
PYTHONPATH=src python -m piiclf.cli scan {{repo}}
```

Identical behaviour, bypasses the console-script mechanism entirely. Useful for debugging a broken install, or for running from a checkout you'd rather not install.

---

## Usage

```bash
piiclf scan <repo-path>                      # masked table
piiclf scan <repo-path> --json               # machine-readable
piiclf scan <repo-path> --raw                # suppression OFF: raw candidate stream
piiclf scan <repo-path> --min-confidence 0.8 # high-confidence findings only
```

`--raw` exists to measure how much precision comes from the suppression rules alone. Phase 5 needs that number as a control — otherwise a model "win" might just be a regex win in disguise.

---

## How it works

```
walk .go files  →  detect (regex + entropy)  →  suppress  →  mask  →  report
   scan.py              baseline.py            baseline.py   report.py
```

The scanner honors `.gitignore`, skips `vendor/`, `node_modules/`, `third_party/`, and Go files carrying a `// Code generated ... DO NOT EDIT.` header.

### Entity classes

| Entity | How the baseline finds it | Known weakness |
|---|---|---|
| `KEY` | Provider prefixes (AWS, GitHub, Slack, Stripe, Google, Anthropic, OpenAI, SendGrid), JWTs, PEM blocks, plus high-entropy strings in assignment position | Generic high-entropy detection is the weak arm |
| `PASSWORD` | Keyword-adjacent assignment, DSN `user:pass@host` | Misses unnamed credentials |
| `EMAIL` | Standard pattern + placeholder-domain filtering | Strong already |
| `IP` | Dotted quad with octet validation | Strong already |
| `USERNAME` | Keyword-adjacent assignment, `@handle` in comments | Low precision — same problem StarPII has |
| `NAME` | **Only** author/copyright attribution comments | Near-total blind spot. This is the clearest case for the model |

### Masking

**Values are masked unconditionally and there is deliberately no unmask flag.** A scan that echoed secrets would just relocate the leak into your terminal scrollback and shell history.

| Entity | Rendered as | Rationale |
|---|---|---|
| `KEY` | `AKIA…[20 chars]` | Provider prefix identifies what to rotate; never a tail |
| `PASSWORD` | `…[11 chars]` | Length only |
| `EMAIL` | `j***@a***.io` | Enough to recognize, not to harvest |
| `IP` | `93.x.x.x` | First octet only |
| `NAME` | `F*** L***` | Initials |
| `USERNAME` | `s***` | First character |

---

## Testing scenarios

### 1. Unit tests

```bash
pytest -q
```

Expect **31 passed**. Every value in the test suite is synthetic and shape-valid but non-functional — no real credential appears anywhere in this repo.

Coverage: provider-key detection, placeholder suppression, non-secret shapes (git SHA / UUID / env-var name), reserved IPs, version-string disambiguation, test-path suppression, raw-vs-suppressed behavior, 1-based line/column correctness, and the invariant that **no mask ever contains its full input value**.

### 2. Canary file — does it detect all 6 classes?

The most useful manual test. Create a file with one of everything plus hard negatives.

**Every value below is synthetic and non-functional** — it exists only to match a detector's *shape*. The names are schematic placeholders, not real people. The `postgres://` DSN connects nowhere: `.internal` is a reserved special-use TLD that does not resolve on the public internet, and the username, password, and database name are all invented. Nothing here authenticates against anything.

```bash
mkdir -p /tmp/piiclf-canary && cat > /tmp/piiclf-canary/main.go <<'EOF'
package main

// Author: Firstname Lastname
// Copyright (c) 2024 Jane Roe
// questions: jane.doe@acmecorp.io

import "fmt"

const accessKeyID = "AKIAQ7X4MZLP2VNRT8KD"

var (
    dsn       = "postgres://svc_reader:Xk92LmQp4Zt@db.internal:5432/offers"
    upstream  = "93.184.216.34:8080"
    authQuery = "mutation Upsert($in: In!) { upsert(in: $in) { id name } }"
    tokenURL  = "https://auth.vendor.io/oauth2/v1/token"
    fallback  = "changeme"
    envName   = "STRIPE_SECRET_KEY"
    localAddr = "127.0.0.1:9090"
)

func main() { fmt.Println(accessKeyID, dsn, upstream, authQuery, tokenURL, fallback, envName, localAddr) }
EOF

piiclf scan /tmp/piiclf-canary
```

Verified output — 7 findings, all 6 entity classes:

```
┃ Entity   ┃ Masked value    ┃ Location      ┃ Detector        ┃ Conf ┃
│ KEY      │ AKIA…[20 chars] │ main.go:9:22  │ aws-access-key  │ 0.95 │
│ EMAIL    │ j***@a***.io    │ main.go:5:15  │ email           │ 0.90 │
│ PASSWORD │ …[11 chars]     │ main.go:12:37 │ dsn-credentials │ 0.85 │
│ IP       │ 93.x.x.x        │ main.go:13:15 │ ipv4            │ 0.80 │
│ NAME     │ F*** L***       │ main.go:3:12  │ author-comment  │ 0.75 │
│ NAME     │ J*** R***       │ main.go:4:23  │ copyright       │ 0.75 │
│ USERNAME │ s***            │ main.go:12:26 │ dsn-credentials │ 0.70 │
```

**What must NOT appear** — these five are the hard negatives, and all are correctly suppressed:

| Value | Why it must be suppressed |
|---|---|
| `authQuery` GraphQL literal | Prose-ish text sits near 4.0 bits/char, same as random base64 needs |
| `tokenURL` | A URL is not a credential |
| `changeme` | Placeholder vocabulary |
| `STRIPE_SECRET_KEY` | An env var *name*, not its value |
| `127.0.0.1` | Loopback |

### 3. Suppression contribution

```bash
piiclf scan /tmp/piiclf-canary --raw   # expect MORE findings than default
```

The delta between `--raw` and default *is* the suppression layer's contribution. Phase 5 reports this as a control.

### 4. Real repositories

```bash
piiclf scan {{repo}}                       # 574 Go files
piiclf scan {{repo}}                       # 59 Go files
piiclf scan ../golang-modules              # 32 Go files
```

### 5. Masking invariant (the one that matters most)

```bash
piiclf scan {{repo}} --json | grep -c '"masked"'
```

No output of this tool should ever contain a full secret. If you change `report.py`, `TestMasking::test_mask_never_contains_full_value` is the guard.

---

## What running it on real code actually found

Three genuine bugs, all found by execution rather than review:

1. **GraphQL and URL literals read as keys.** The first scan of `clearcar-go-api` flagged 4 KEYs — all false. One URL, three GraphQL query/mutation strings assigned to identifiers like `authQuery`. They beat the entropy gate because prose sits near 4.0 bits/char. Fixed with a shape rule: a credential is a single whitespace-free token of bounded length. **KEY findings went 4 → 0.**

2. **`KEY_ASSIGN` never matched plain `apiKey := "..."`** — the pattern required a character before the keyword. Caught while writing tests.

3. **A DSN password leaked through the email mask.** In `postgres://user:pass@host`, the `pass@host` portion matches the email pattern, and EMAIL's mask reveals the first character — exposing a character of the password that PASSWORD's mask deliberately hides. Fixed by span-overlap suppression.

**No real credentials exist in the scanned repos.** The 8 highest-confidence findings are hardcoded internal email addresses in `clearcar-go-api` — a configuration smell, not a security incident.

And the predicted weakness held: **zero NAME findings across 574 Go files**, because regex can only see attribution comments. That gap is the model's job.

---

## Fine-tuning: where it runs and how long

**Training runs entirely on your local machine.** Model weights download from HuggingFace to local disk, and the GPU does the math. Nothing is uploaded, and no code or training data leaves the machine.

Measured on this host (M4 Max, 40 GPU cores, 64 GB unified memory, torch 2.14.0, MPS), a real forward + backward + optimizer step on ModernBERT-base (149.6M params):

| seq len | batch | ms/step | ex/s | tokens/s | 30k-example epoch |
|---|---|---|---|---|---|
| **512** | 8 | 1066 | 7.5 | 3,840 | **~67 min** |
| 512 | 16 | 2241 | 7.1 | 3,635 | ~70 min |
| 1024 | 4 | 1392 | 2.9 | 2,970 | ~174 min |
| 1024 | 8 | 2629 | 3.0 | 3,072 | ~164 min |

Reading this:

- **"tokens/s" is the fair comparison.** A model reads *tokens* (subword pieces), not lines, and examples at 512 and 1024 tokens aren't the same size — so examples/sec isn't apples-to-apples. At ~3,000–3,800 tokens/sec you're training at roughly the rate of one mid-sized Go file per second.
- **Tokens/sec *drops* at longer sequences** (3,840 → 3,072) because attention cost grows with the square of sequence length. Every token attends to every other token, so each individual token gets more expensive when there are more of them around it. That's why 1024 is 2.4× slower, not 2×.
- **Use seq len 512.** 2.4× faster for the same data.
- **Memory is a non-issue: 2.4 GB of 64 GB.** The run is compute-bound — doubling the batch doubled the step time, leaving per-example throughput flat, which means the GPU is already saturated with arithmetic. Practical upshot: a bigger batch is *free* (better gradients, same wall clock), but it will not make training faster.
- **An epoch is not a finished model.** Token classification needs 3–5 passes, so one usable model is ~3.4 h at seq 512. Phase 5's ablation needs 6–10 runs.

Gotcha worth recording: `reference_compile` is a **config attribute, not a `from_pretrained` kwarg**. Passing it directly raises `TypeError`. Load via `AutoConfig`, set `config.reference_compile = False`, then pass `config=`.

---

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Scaffold | ✅ |
| 1 | Regex + entropy baseline, scanner CLI | ✅ |
| 2 | Eval sets — 300–500 hand-labeled real spans | next |
| 3 | Synthetic data generation via tree-sitter injection | |
| 4 | Fine-tune flat BIO token classifier | |
| 5 | Span classifier + structural-context ablation | |
| 6 | Hybrid arbitration, writeup, then extend to Node/TS | |

Phase 2 comes before any training on purpose: a held-out *synthetic* split would score ~0.99 and mean nothing, because we'd be grading ourselves against our own label function. The hand-labeled set is the headline metric.

See [PLAN.md](PLAN.md) for the full plan, decided tradeoffs, and open risks.

---

## Policy

- Training data: permissively licensed public repos plus synthetic only. **Never ACV source.**
- **No real secrets in any fixture or eval set**, regardless of provenance.
- Values are always masked; there is no unmask flag. If a scan surfaces a real credential, rotate via `security@acvauctions.com` and do not paste the value anywhere.
- All dependencies are permissively licensed (BSD / Apache-2.0 / MIT).


<img width="720" height="379" alt="Screenshot 2026-09-24 at 4 25 49 PM" src="https://github.com/user-attachments/assets/97b05759-cc05-49d0-ac7b-06cbf86e5ba6" />

