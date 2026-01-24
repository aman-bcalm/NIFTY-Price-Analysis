from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "AppConfig":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config must be a YAML mapping/object at the top level.")
        return AppConfig(raw=data)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

