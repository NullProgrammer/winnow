#!/usr/bin/env python3
"""Clone the repo corpus defined in corpus.json and lock what was fetched.

Standalone setup tooling, deliberately *outside* the package: nothing in
`src/piiclf/` imports it, and it can be deleted once the corpus is fetched and
Phase 3 is done without touching the library or the CLI.

Downstream code reads `data/corpus/<set>.lock.json` directly rather than
importing from here, which is what keeps this a leaf.

    python scripts/fetch_corpus.py --list
    python scripts/fetch_corpus.py --set eval

Licenses are classified from the actual LICENSE text at fetch time rather than
trusted from a config field, since a repo can relicense between runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from piiclf.scan import ALWAYS_SKIP_DIRS, MAX_FILE_BYTES, iter_go_files  # noqa: E402

CONFIG_PATH = REPO_ROOT / "corpus.json"
DEFAULT_ROOT = REPO_ROOT / "data" / "corpus"

LICENSE_FILENAMES = (
    "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENSE-MIT",
    "COPYING", "COPYING.md",
)


def classify_license(text: str) -> str | None:
    """Identify a license from its text, or None if unrecognized.

    Conservative on purpose — an unrecognized license is refused rather than
    guessed at, since guessing wrong is the expensive direction.
    """
    t = " ".join(text.split()).lower()

    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "apache software license" in t and "version 1.1" in t:
        return "Apache-1.1"
    if "mozilla public license" in t:
        return "MPL-2.0" if "version 2.0" in t else "MPL-1.1"
    if "permission is hereby granted, free of charge" in t:
        # MIT and X11 share this opening; X11 adds an advertising clause.
        return "X11" if "x consortium" in t else "MIT"
    if "permission to use, copy, modify, and/or distribute this software" in t:
        return "ISC"
    if "redistribution and use in source and binary forms" in t:
        if "neither the name of" in t or "endorse or promote" in t:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    if "free and unencumbered software released into the public domain" in t:
        return "Unlicense"
    if "do what the fuck you want to public license" in t:
        return "WTFPL"
    if "altered source versions must be plainly marked as such" in t:
        return "Zlib"
    if "academic free license" in t:
        return "AFL-2.1"
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _find_license(repo_dir: Path) -> tuple[str | None, str | None]:
    for fn in LICENSE_FILENAMES:
        p = repo_dir / fn
        if p.is_file():
            raw = p.read_bytes()
            return classify_license(raw.decode("utf-8", errors="replace")), _sha256(raw)
    return None, None


def _git(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch_set(set_name: str, root: Path, config: dict) -> dict:
    approved = set(config["approved_licenses"])
    dest = root / set_name
    dest.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    rejected: list[dict] = []

    for repo in config["sets"][set_name]["repos"]:
        repo_dir = dest / repo["name"]
        if not (repo_dir / ".git").is_dir():
            _git(["clone", "--depth", "1", "--quiet", repo["url"], str(repo_dir)])

        license_id, license_hash = _find_license(repo_dir)
        if license_id not in approved:
            rejected.append({**repo, "license": license_id})
            continue

        gi = repo_dir / ".gitignore"
        entries.append({
            "name": repo["name"],
            "url": repo["url"],
            "role": repo.get("role"),
            "sha": _git(["rev-parse", "HEAD"], cwd=repo_dir),
            "license": license_id,
            "license_sha256": license_hash,
            "gitignore_sha256": _sha256(gi.read_bytes()) if gi.is_file() else None,
            "go_files": len(iter_go_files(repo_dir)),
        })

    return {
        "set": set_name,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Recorded because these decide which files are in scope, making them
        # part of the corpus definition rather than incidental config.
        "scan_config": {
            "encoding": "utf-8-sig",
            "errors": "replace",
            "skip_generated": True,
            "max_file_bytes": MAX_FILE_BYTES,
            "always_skip_dirs": sorted(ALWAYS_SKIP_DIRS),
        },
        "repos": entries,
        "rejected": rejected,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", dest="set_name", help="which set to fetch (eval or train)")
    ap.add_argument("--list", action="store_true", help="show configured sets and exit")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if args.list or not args.set_name:
        for set_name, spec in config["sets"].items():
            print(f"\n{set_name} ({len(spec['repos'])} repos) — {spec['purpose']}")
            for r in spec["repos"]:
                print(f"  {r['name']:22} {r.get('role', ''):12} {r['url']}")
        print(f"\napproved licenses: {', '.join(config['approved_licenses'])}")
        return 0

    if args.set_name not in config["sets"]:
        print(f"Unknown set: {args.set_name}. Choose: {', '.join(config['sets'])}", file=sys.stderr)
        return 2

    print(f"Fetching {args.set_name} into {args.root}/{args.set_name} ...")
    lock = fetch_set(args.set_name, args.root, config)

    path = args.root / f"{args.set_name}.lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    for r in lock["repos"]:
        print(f"  {r['name']:22} {r['license']:14} {r['go_files']:5} go files  {r['sha'][:10]}")
    for r in lock["rejected"]:
        print(f"  REJECTED {r['name']} — license {r['license']!r} not approved", file=sys.stderr)

    total = sum(r["go_files"] for r in lock["repos"])
    print(f"\n{len(lock['repos'])} repos, {total} Go files. Lockfile: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
