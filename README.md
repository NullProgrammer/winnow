# hackathon-pii-classifier

PII and credential detection for **Go codebases** — not prose.

Most open PII models (Piiranha, GLiNER-PII, Horizon Labs' redactor) are trained on natural language. Code is a different problem: the signal separating real PII from a placeholder is mostly *structural* — is the string in a `_test.go` file? a struct tag? a copyright header? a T-SQL variable? This project builds a small encoder model that learns that distinction, with a regex + entropy baseline as the honest bar to beat.

**Status: Phases 0–3 complete.** The baseline is measured, a hand-verified gold set exists, and 30k synthetic training windows are generated. Phase 4 (fine-tuning) is next. Full roadmap and results in [PLAN.md](PLAN.md); adjudication rules in [docs/ANNOTATION.md](docs/ANNOTATION.md).

**Headline so far: the regex baseline scores 3/56 = 5% precision** (95% CI 2–15%) on a hand-verified sample. Only the `email` detector has non-zero precision. Regex finds PII-*shaped* text; ~95% of it isn't PII.

---

## Install

From scratch on a clean machine:

```bash
# 1. install uv, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. from the repo root
cd hackathon-pii-classifier

# 3. create the venv (uv downloads Python 3.12 itself if needed)
uv venv --python 3.12 .venv

# 4. install the project and its test deps
uv pip install -e ".[dev]"

# 5. activate — every command below assumes this
source .venv/bin/activate
```

Under a minute. This gets you `scan`, `sample`, `label`, and `evaluate` — everything except the ML pipeline.

### The ML extra (needed from Phase 3 onward)

```bash
uv pip install -e ".[ml]"
```

~2.5 GB, mostly torch. Pulls `torch`, `transformers`, `tree-sitter`, `tree-sitter-go`, `faker`. Required for `piiclf generate` and for training. **Not** required to scan a repo or score the gold set.

> If you only need the tokenizer and not training (e.g. to re-verify BIO alignment), `uv pip install transformers` alone is enough and far smaller.

### Two install gotchas that have actually bitten

1. **zsh eats the extras.** Unquoted, `[dev]` is a glob pattern and zsh refuses the command before uv ever runs:
   ```
   $ uv pip install -e .[dev]
   zsh:1: no matches found: .[dev]
   ```
   Always quote it: `uv pip install -e ".[dev]"`. Works unquoted on bash, fails on zsh.
2. **Rebuilding `.venv` wipes the ML extra.** `uv venv` recreates the venv from empty, so after any rebuild you must re-run *both* installs. A missing `transformers` silently **skips** the alignment test module rather than failing it.

> **Run every command from inside `hackathon-pii-classifier/`.** The venv lives here, so from a parent directory `.venv/bin/piiclf` is "no such file or directory" and bare `piiclf` is "command not found". The `{{REPO_PATH}}/...` and `data/...` paths are relative to this directory too.
>
> To run from anywhere, use the binary's absolute path — `/path/to/hackathon-pii-classifier/.venv/bin/piiclf` — or alias it in your shell profile.

### Verify the install

```bash
pytest -q                              # expect: 140 passed
piiclf --help                          # lists all 5 commands
piiclf scan {{REPO_PATH}}   # works with no corpus fetched
```

**Fallback if `piiclf` isn't on PATH** — needs no install at all, only `typer`/`rich`/`pathspec`:

```bash
PYTHONPATH=src python -m piiclf.cli scan {{REPO_PATH}}
```

---

## Command reference

| Command | Needs | What it does |
|---|---|---|
| `python scripts/fetch_corpus.py --list` | — | Show the configured repo sets |
| `python scripts/fetch_corpus.py --set eval` | git | Clone the eval corpus, verify licenses, write the lockfile |
| `python scripts/fetch_corpus.py --set train` | git | Same for the training corpus (disjoint from eval) |
| `piiclf scan <path>` | core | Scan a Go repo for PII candidates. Values always masked |
| `piiclf sample` | core + corpus | Draw a stratified candidate sample for labeling |
| `piiclf label` | core + sample | Verify proposed labels, one keystroke each. Resumable |
| `piiclf evaluate` | core + gold | Score the baseline against the gold set |
| `piiclf generate` | **`[ml]`** | Build the synthetic BIO-labeled training corpus |

### `piiclf scan <path>`

```bash
piiclf scan {{REPO_PATH}}                        # masked table
piiclf scan {{REPO_PATH}} --json                 # machine-readable
piiclf scan {{REPO_PATH}} --raw                  # suppression OFF: raw candidate stream
piiclf scan {{REPO_PATH}} --min-confidence 0.8   # high-confidence findings only
```

`--raw` exists to measure how much precision comes from the suppression rules alone. Phase 5 needs that as a control — otherwise a model "win" might just be a regex win in disguise.

### `piiclf sample`

```bash
piiclf sample                                    # 150 candidates from the eval corpus
piiclf sample --budget 300 --seed 42             # bigger sample, different draw
piiclf sample --set eval --out-dir data/gold     # explicit paths
```

Stratifies by **(detector × survives-suppression)** and prints the allocation table. Equal allocation, not proportional — proportional would spend the whole budget on `copyright` and leave PASSWORD unmeasurable. Writes `data/gold/candidates.jsonl`.

### `piiclf label`

```bash
piiclf label              # verify everything outstanding
piiclf label --limit 25   # short sitting
```

Keys: `t` TRUE · `s` WRONG_SPAN · `e` WRONG_ENTITY · `f` FALSE · `u` UNSURE · `?` help · `q`/`Ctrl+C`/`Ctrl+X`/`Ctrl+Z` save and quit.

Resumable — already-labeled spans are skipped, autosaves every 10, and saves on *any* exit path including a crash. Reads proposed labels from `data/gold/prelabels.jsonl` if present, writes `data/gold/gold.jsonl`.

Planted foils (deliberately wrong proposals) and per-decision latency are recorded; the session summary flags the pass as untrustworthy if too many foils slipped through.

### `piiclf evaluate`

```bash
piiclf evaluate
```

Span-level precision per detector and per entity, reweighted to the full candidate population, with Wilson confidence intervals. Point estimates below n=10 print `n<10` rather than pretending to be measurements. Recall is *not* reported — that needs an exhaustive file sweep, since a candidate sample can only show what the detector proposed.

### `piiclf generate`

```bash
piiclf generate                     # 30k train + 3k val windows
piiclf generate --target 10000      # smaller run (~22 min/epoch to train)
piiclf generate --check 1000        # verify BIO alignment on more windows
piiclf generate --set eval          # refused on purpose
```

Injects synthetic PII into the **train** corpus, cuts line-aligned windows, and verifies BIO alignment. Writes `data/synth/{train,val}.jsonl` and `summary.json`.

Passing `--set eval` is **refused**: the eval corpus is reserved for the gold set, and training on it would invalidate every Phase 4 number.

---

## End-to-end walkthrough

Everything from a clean clone, in order:

```bash
# install
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]" && uv pip install -e ".[ml]"
pytest -q                                          # 140 passed

# Phase 1 — scan a real repo with the regex baseline
piiclf scan {{REPO_PATH}}

# Phase 2 — fetch the eval corpus, sample, label, score
python scripts/fetch_corpus.py --set eval          # 9 repos, 7229 Go files
piiclf sample --budget 150                         # -> data/gold/candidates.jsonl
piiclf label                                       # interactive, ~20 min
piiclf evaluate                                    # -> the precision table

# Phase 3 — fetch the training corpus and generate training data
python scripts/fetch_corpus.py --set train         # 5 repos, 2038 Go files
piiclf generate --target 30000                     # -> data/synth/train.jsonl
```

`piiclf label` is the only step that needs a human. Everything else is reproducible from the committed lockfiles and seeds.

---

## Data directories

| Path | Committed? | What |
|---|---|---|
| `corpus.json` | yes | Repo definitions. **Edit this, not Python**, to change the corpus |
| `data/corpus/*.lock.json` | **yes** | Pinned SHAs + verified licenses — the reproducibility record |
| `data/corpus/<set>/` | no (149 MB) | Cloned repos |
| `data/gold/gold.jsonl` | **yes** | Hand-verified labels. **No raw values stored** |
| `data/gold/candidates.jsonl` | no | Intermediate sample (contains values) |
| `data/synth/` | no | Generated training windows |

Lockfiles and the gold set are committed because they *are* the experiment. `.gitignore` ignores directory *contents* with negations for those two, because git won't descend into a fully ignored directory.

---

## How it works

```
Phase 1   walk .go files -> detect (regex + entropy) -> suppress -> mask -> report
              scan.py            baseline.py          baseline.py  report.py

Phase 2   corpus -> candidates -> stratified sample -> human verify -> score
          fetch_corpus.py           gold.py            labeler.py   evaluate.py

Phase 3   corpus -> tree-sitter sites -> inject values -> window -> BIO labels
                       sites.py         generators.py   dataset.py  tokenize_align.py
                                          inject.py
```

The scanner honors `.gitignore`, skips `vendor/`, `node_modules/`, `third_party/`, and files carrying a `// Code generated ... DO NOT EDIT.` header.

### Entity classes and measured baseline precision

| Entity | Precision | Known weakness |
|---|---|---|
| `EMAIL` | **27%** (3/11) | The only detector that works |
| `IP` | 0% (0/11) | Test placeholders, public DNS, RFC section numbers |
| `NAME` | 0% (0/11) | Attribution isn't PII, so regex finds nothing of value |
| `USERNAME` | 0% (0/21) | Constants, SQL variables, service defaults |
| `KEY` | not reportable | Provider prefixes work; generic entropy doesn't |
| `PASSWORD` | 0% (0/2) | n too small to measure |

### Masking

**Values are masked unconditionally and there is deliberately no unmask flag.** A scan that echoed secrets would just relocate the leak into terminal scrollback and shell history.

| Entity | Rendered as | Rationale |
|---|---|---|
| `KEY` | `AKIA…[20 chars]` | Provider prefix identifies what to rotate; never a tail |
| `PASSWORD` | `…[11 chars]` | Length only |
| `EMAIL` | `j***@a***.io` | Enough to recognize, not to harvest |
| `IP` | `93.x.x.x` | First octet only |
| `NAME` | `F*** L***` | Initials |
| `USERNAME` | `s***` | First character |

---

## Testing

```bash
pytest -q                                # 140 tests
pytest tests/test_baseline.py -q         # detectors and suppression
pytest tests/test_sites.py -q            # tree-sitter offsets, incl. non-ASCII
pytest tests/test_inject.py -q           # the span-equality gate
pytest tests/test_tokenize_align.py -q   # BIO alignment (needs transformers)
```

Every value in the suite is synthetic and shape-valid but non-functional — **no real credential appears anywhere in this repo.**

### Canary file — does it detect all 6 classes?

The most useful manual test. Every value below is synthetic: the names are schematic placeholders, and the `postgres://` DSN connects nowhere (`.internal` is a reserved TLD that doesn't resolve).

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

**What must NOT appear** — the hard negatives, all correctly suppressed:

| Value | Why |
|---|---|
| `authQuery` GraphQL literal | Prose sits near 4.0 bits/char, same as random base64 |
| `tokenURL` | A URL is not a credential |
| `changeme` | Placeholder vocabulary |
| `STRIPE_SECRET_KEY` | An env var *name*, not its value |
| `127.0.0.1` | Loopback |

---

## Fine-tuning: where it runs and how long

**Training runs entirely on your local machine.** Weights download from HuggingFace to local disk and the GPU does the math. Nothing is uploaded; no code or training data leaves the machine.

Measured on an M4 Max (40 GPU cores, 64 GB unified, torch 2.14, MPS), real forward + backward + optimizer step on ModernBERT-base (149.6M params):

| seq len | batch | ms/step | ex/s | tokens/s | 30k-window epoch |
|---|---|---|---|---|---|
| **512** | 8 | 1066 | 7.5 | 3,840 | **~67 min** |
| 512 | 16 | 2241 | 7.1 | 3,635 | ~70 min |
| 1024 | 4 | 1392 | 2.9 | 2,970 | ~174 min |
| 1024 | 8 | 2629 | 3.0 | 3,072 | ~164 min |

- **Use seq len 512** — 2.4× faster than 1024 for the same data, because attention cost grows with the square of sequence length.
- **Memory is a non-issue: 2.4 GB of 64 GB.** The run is compute-bound, so a bigger batch is free (better gradients, same wall clock) but won't make it faster.
- **An epoch is not a finished model.** Token classification needs 3–5 passes, so one usable model is ~3.4 h at 30k windows. Start at `--target 10000` (~22 min/epoch) to prove the pipeline first.

Gotcha worth recording: `reference_compile` is a **config attribute, not a `from_pretrained` kwarg**. Passing it directly raises `TypeError`. Load via `AutoConfig`, set `config.reference_compile = False`, then pass `config=`.

---

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Scaffold | ✅ |
| 1 | Regex + entropy baseline, scanner CLI | ✅ |
| 2 | Gold eval set — 150 hand-verified labels | ✅ precision; recall sweep skipped (see PLAN.md) |
| 3 | Synthetic data generation — 30k windows | ✅ |
| 4 | Fine-tune flat BIO token classifier | next |
| 5 | Span classifier + structural-context ablation | |
| 6 | Hybrid arbitration, writeup, then Node/TS | |

---

## Policy

- Training data: permissively licensed public repos + synthetic only.
- **No real secrets in any fixture, test, or eval set**, regardless of provenance.
- All dependencies are permissively licensed (BSD / Apache-2.0 / MIT / MPL-2.0).
