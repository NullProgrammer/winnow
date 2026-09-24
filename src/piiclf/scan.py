from __future__ import annotations

import re
from pathlib import Path

import pathspec

from .baseline import detect
from .entities import Finding

# Generated Go files are machine output; findings in them are noice.
GENERATED = re.compile(r"^// Code generated .* DO NOT EDIT\.$", re.MULTILINE)

ALWAYS_SKIP_DIRS = {"vendor", "node_modules", ".git", "third_party", ".idea", ".vscode"}
MAX_FILE_BYTES = 2_000_000

def load_gitignore(root: Path) -> pathspec.PathSpec:
    patterns: list[str] = []
    gi = root / ".gitignore"
    if gi.is_file():
        patterns = gi.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

def iter_go_file(root: Path) -> list[Path]:
    spec = load_gitignore(root)
    out: list[Path] = []
    for p in root.rglob("*.go"):
        if any(part in ALWAYS_SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if spec.match_file(rel):
            continue
        out.append(p)
    return sorted(out)

def scan_repo(
    root: Path, *, apply_suppression: bool = True, skip_generated: bool = True
) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    scanned = 0

    for path in iter_go_file(root):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            # A BOM would otherwise decode to a leading U+FEFF character and 
            # shift every stored char offset by one, invalidating gold-set spans.
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue

        if skip_generated and GENERATED.search(text[:4096]):
            continue

        scanned += 1
        rel = path.relative_to(root).as_posix()
        findings.extend(detect(text, rel, apply_suppression=apply_suppression))
    return findings, scanned
