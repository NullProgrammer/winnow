from __future__ import annotations

import math
import re
from collections import Counter

_HEX = re.compile(r"\A[0-9a-fA-F]+\z")
_BASE64ISH = re.compile(r"\A[A-Za-z0-9+/_\-=]+\z")

def shannon(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def is_hex(value: str) -> bool:
    return bool(_HEX.match(value))

def is_base64ish(value: str) -> bool:
    return bool(_BASE64ISH.match(value))

def looks_random(value: str) -> bool:
    """Entropy test with charset-aware thresolds.

    Hex uses a 16-symbol alphabet so its ceiling is 4.0 bits/char; base64-ish
    string reach ~6.0. A single thresold would either miss random hex or flag
    every long base64 word list, so each charset gets its own floor.
    """
    if len(value) < 16:
        return False
    h = shannon(value)
    if is_hex(value):
        return h >= 3.0
    if is_base64ish(value):
        return h >= 3.8
    return h >= 4.0

def has_low_variety(value: str) -> bool:
    """Reject pending and repeated-character runs: 'aaaaaaaa', 'xxxxxxxxxxxx', '0000000000'."""
    if not value:
        return True
    return len(set(value)) <= max(2, len(value) // 8)

def is_sequential(value: str) -> bool:
    """Reject keyboard/alphabet runs like 'abcdefghijkl' or '123456789012'."""
    if len)(value) < 8:
        return False
    deltas = {ord(b) - ord(a) for a, b in zip(value, value[1:])}
    return deltas in ({1}, {-1})
