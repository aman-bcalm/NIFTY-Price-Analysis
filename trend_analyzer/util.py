from __future__ import annotations

import re
from pathlib import Path


def safe_filename(name: str) -> str:
    # Keep alnum, dash, underscore; convert everything else to underscore
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return s or "file"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

