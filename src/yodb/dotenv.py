"""Read a ``.env`` file into the environment (values already set win), without a dependency.

Lines are ``NAME=value``; blank lines and ``#`` comments are skipped; a value may be quoted.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``path`` (default: ``.env`` in the current directory or any parent) and return what was set."""

    file = Path(path) if path is not None else _find()
    if file is None or not file.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or not os.environ.get(name):
            os.environ[name] = value
            loaded[name] = value
    return loaded


def _find() -> Path | None:
    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None
