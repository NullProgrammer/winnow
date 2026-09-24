"""Masked rendering of findings.

Values are masked unconditionally and there is deliberately no unmask flag.
For a credential the only actionable detail is which provider issued it and
where it lives — never the value itself. A scan that echoed secrets back would
just relocate the leak into your terminal scrollback and shell history.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from .entities import Entity, Finding

# Known provider prefixes worth preserving, since they identify what to rotate.
_PREFIX = re.compile(
    r"\A(AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ABIA|ACCA|gh[pousr]_|github_pat_"
    r"|xox[baprs]-|[sr]k_(?:live|test)_|AIza|sk-ant-|sk-proj-|sk-|SG\.|eyJ)"
)


def mask(entity: Entity, value: str) -> str:
    if entity is Entity.KEY:
        m = _PREFIX.match(value)
        if value.startswith("-----BEGIN"):
            return "-----BEGIN …PRIVATE KEY----- [PEM block]"
        prefix = m.group(1) if m else value[:2]
        return f"{prefix}…[{len(value)} chars]"

    if entity is Entity.PASSWORD:
        return f"…[{len(value)} chars]"

    if entity is Entity.EMAIL:
        local, _, domain = value.partition("@")
        host, _, tld = domain.rpartition(".")
        return f"{local[:1]}***@{host[:1]}***.{tld}"

    if entity is Entity.IP:
        return f"{value.split('.')[0]}.x.x.x"

    if entity is Entity.NAME:
        return " ".join(f"{p[:1]}***" for p in value.split()[:3])

    return f"{value[:1]}***"


def to_json(findings: list[Finding]) -> str:
    return json.dumps(
        [
            {
                "entity": f.entity.value,
                "masked": mask(f.entity, f.value),
                "path": f.path,
                "line": f.line,
                "column": f.column,
                "detector": f.detector,
                "confidence": round(f.confidence, 2),
            }
            for f in findings
        ],
        indent=2,
    )


def render_table(findings: list[Finding], files_scanned: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if not findings:
        console.print(f"[green]No findings[/green] across {files_scanned} Go files.")
        return

    table = Table(title=f"PII findings ({len(findings)} across {files_scanned} Go files)")
    table.add_column("Entity", style="bold")
    table.add_column("Masked value")
    table.add_column("Location", style="cyan")
    table.add_column("Detector", style="dim")
    table.add_column("Conf", justify="right")

    colors = {
        Entity.KEY: "red",
        Entity.PASSWORD: "red",
        Entity.EMAIL: "yellow",
        Entity.NAME: "yellow",
        Entity.USERNAME: "blue",
        Entity.IP: "magenta",
    }

    for f in sorted(findings, key=lambda x: (-x.confidence, x.path, x.line)):
        table.add_row(
            f"[{colors[f.entity]}]{f.entity.value}[/]",
            mask(f.entity, f.value),
            f.location,
            f.detector,
            f"{f.confidence:.2f}",
        )

    console.print(table)

    counts = Counter(f.entity.value for f in findings)
    console.print("  ".join(f"[bold]{k}[/bold] {v}" for k, v in sorted(counts.items())))

    if any(f.entity in (Entity.KEY, Entity.PASSWORD) and f.confidence >= 0.85 for f in findings):
        console.print(
            "\n[red bold]High-confidence credential(s) found.[/red bold] "
            "If any is real, rotate it via security@acvauctions.com. Do not paste the value anywhere."
        )
