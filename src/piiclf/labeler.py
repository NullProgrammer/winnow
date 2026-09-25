"""Interactive verification of pre-labeled candidates.

Claude proposes a label for each candidate; a human accepts or corrects it with
one keystroke. That's ~4x faster than labeling cold, at the cost of a
rubber-stamping risk — so the session tracks decision latency and scores the
planted foils, and refuses to call a pass valid if the foils weren't caught.

Resumable: already-labeled spans in gold.jsonl are skipped, so a 150-item
session can be done across sittings.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

from .entities import Entity
from .gold import GoldRecord, read_jsonl, write_jsonl

KEYS = {
    "t": "TRUE",
    "s": "WRONG_SPAN",
    "e": "WRONG_ENTITY",
    "f": "FALSE",
    "u": "UNSURE",
}

# Raw tty mode means Ctrl+C/X/Z arrive as bytes, not signals — so they have to
# be handled explicitly or they read as an unrecognized key.
EXIT_KEYS = {"q", "\x03", "\x18", "\x1a"}  # q, Ctrl+C, Ctrl+X, Ctrl+Z

HELP = """
  t  TRUE          right entity, right boundaries
  s  WRONG_SPAN    right entity, wrong boundaries
  e  WRONG_ENTITY  real PII, wrong class
  f  FALSE         not PII at all
  u  UNSURE        excluded from scoring, reported separately
  ?  show this
  q / Ctrl+C / Ctrl+X / Ctrl+Z   save and quit (resumes here next time)
"""


def _read_key() -> str:
    """Single keypress without Enter, falling back to line input."""
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()
    except Exception:
        # Not a tty (piped input, some CI shells). Treat EOF as an exit key
        # rather than letting it raise — the caller saves on the way out.
        try:
            return (input().strip().lower() or "?")[:1]
        except EOFError:
            return "\x03"


def _key_of(row: dict) -> tuple:
    return (row["repo"], row["path"], row["start"], row["end"], row["entity"], row["detector"])


def run_session(
    candidates_path: Path,
    gold_path: Path,
    prelabels_path: Path | None = None,
    limit: int | None = None,
) -> dict:
    from rich.console import Console

    console = Console()

    cands = read_jsonl(candidates_path)
    prelabels = {}
    if prelabels_path and prelabels_path.exists():
        for row in read_jsonl(prelabels_path):
            prelabels[_key_of(row)] = row

    done = read_jsonl(gold_path) if gold_path.exists() else []
    seen = {_key_of(r) for r in done}
    todo = [c for c in cands if _key_of(c) not in seen]
    if limit:
        todo = todo[:limit]

    if not todo:
        console.print("[green]Nothing left to label.[/green]")
        return _summarize(done, console)

    console.print(
        f"[bold]{len(todo)} to label[/bold] ({len(done)} already done).\n"
        f"[dim]t=true  f=false  s=wrong span  e=wrong entity  u=unsure  "
        f"?=help  q/Ctrl+C/Ctrl+X/Ctrl+Z=save & quit[/dim]\n"
    )

    records: list = list(done)

    def save() -> None:
        write_jsonl([asdict(r) if not isinstance(r, dict) else r for r in records], gold_path)

    stopped = False
    try:
        for i, c in enumerate(todo, 1):
            rec = _label_one(c, prelabels.get(_key_of(c), {}), i, len(todo), console)
            if rec is None:
                stopped = True
                break
            records.append(rec)
            if i % 10 == 0:
                save()
    except (KeyboardInterrupt, EOFError):
        stopped = True
    finally:
        # Always persist. A crash or interrupt would otherwise lose up to the
        # whole autosave window, and resume is the whole point.
        save()

    if stopped:
        console.print(
            f"\n[yellow]Saved {len(records)} records to {gold_path}[/yellow]\n"
            f"[dim]{len(cands) - len(records)} left. Run `piiclf label` again to resume here.[/dim]"
        )
    else:
        console.print(f"\n[green]Saved {len(records)} records to {gold_path}[/green]")
    return _summarize(records, console)


def _label_one(c: dict, pre: dict, i: int, total: int, console) -> GoldRecord | None:
    """Collect one verdict. Returns None if the labeler asked to stop."""
    from rich.panel import Panel
    from rich.syntax import Syntax

    proposed = pre.get("label")
    kept = c["survives_suppression"]

    console.print(
        Panel(
            Syntax(c["context"], "go", theme="ansi_dark", line_numbers=False),
            title=f"[{i}/{total}] {c['repo']}/{c['path']}:{c['line']}",
            subtitle=f"{c['entity']} via {c['detector']}"
            + ("  [dim](suppression keeps this)[/dim]" if kept else "  [dim](suppression drops this)[/dim]"),
        )
    )
    console.print(f"  span: [bold yellow]{c['value']!r}[/bold yellow]")
    if proposed:
        console.print(f"  Claude says: [bold cyan]{proposed}[/bold cyan] — {pre.get('reason', '')}")

    t0 = time.perf_counter()
    while True:
        console.print("  verdict> ", end="")
        k = _read_key()
        console.print(k)
        if k == "?":
            console.print(HELP)
            continue
        if k in EXIT_KEYS:
            return None
        if k in KEYS:
            break
        console.print("  [red]unrecognized — press ? for help[/red]")

    latency = int((time.perf_counter() - t0) * 1000)
    label = KEYS[k]

    corrected_entity = None
    if label == "WRONG_ENTITY":
        opts = [e.value for e in Entity]
        console.print("  correct entity: " + "  ".join(f"{n}={v}" for n, v in enumerate(opts, 1)))
        console.print("  > ", end="")
        sel = _read_key()
        console.print(sel)
        if sel.isdigit() and 1 <= int(sel) <= len(opts):
            corrected_entity = opts[int(sel) - 1]

    corrected_start = corrected_end = None
    if label == "WRONG_SPAN":
        console.print("  paste the correct span text (Enter to skip): ", end="")
        try:
            want = input().strip()
        except EOFError:
            want = ""
        if want:
            # Locate within the candidate's own line so the offset stays absolute.
            idx = c["context"].find(want)
            if idx >= 0:
                corrected_start = c["start"] - (c["col"] - 1) + idx
                corrected_end = corrected_start + len(want)

    return GoldRecord(
        repo=c["repo"],
        path=c["path"],
        line=c["line"],
        col=c["col"],
        start=c["start"],
        end=c["end"],
        entity=c["entity"],
        detector=c["detector"],
        stratum=c["stratum"],
        survives_suppression=kept,
        label=label,
        line_sha256=c["line_sha256"],
        span_len=c["span_len"],
        corrected_entity=corrected_entity,
        corrected_start=corrected_start,
        corrected_end=corrected_end,
        latency_ms=latency,
        is_foil=bool(pre.get("is_foil")),
        notes=f"claude_proposed={proposed}" if proposed else "",
    )


def _summarize(records: list, console) -> dict:
    rows = [asdict(r) if not isinstance(r, dict) else r for r in records]
    if not rows:
        return {}

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    lat = sorted(r.get("latency_ms", 0) for r in rows)
    median = lat[len(lat) // 2] if lat else 0
    fast = sum(1 for x in lat if x < 1000)

    foils = [r for r in rows if r.get("is_foil")]
    caught = [
        r for r in foils
        if r["label"] != r.get("notes", "").split("claude_proposed=")[-1]
    ]

    console.print("\n[bold]Session summary[/bold]")
    console.print("  " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    console.print(f"  median decision {median} ms; {fast}/{len(lat)} under 1s")
    if foils:
        console.print(f"  foils caught: {len(caught)}/{len(foils)}")
        if len(caught) < len(foils) * 0.8:
            console.print(
                "  [red bold]Too many foils missed — this pass isn't trustworthy. "
                "Delete gold.jsonl and redo it.[/red bold]"
            )
    return {
        "counts": counts,
        "median_latency_ms": median,
        "foils": len(foils),
        "caught": len(caught) if foils else 0,
    }
