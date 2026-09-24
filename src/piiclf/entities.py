from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class Entity(str, Enum):
    NAME = "NAME"
    USERNAME = "USERNAME"
    EMAIL = "EMAIL"
    IP = "IP"
    KEY = "KEY"
    PASSWORD = "PASSWORD"


# Secrets get stricter masking than personal identifiers: for a credential the
# only actionable detail is which provider issues it, never the value.
SECRET_ENTITIES = frozenset({Entity.KEY, Entity.PASSWORD})


@dataclass(frozen=True, slots=True)
class Finding:
    entity: Entity
    value: str
    start: int
    end: int
    line: int
    column: int
    detector: str
    confidence: float
    path: str = ""

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}:{self.column}" if self.path else f"{self.line}:{self.column}"
