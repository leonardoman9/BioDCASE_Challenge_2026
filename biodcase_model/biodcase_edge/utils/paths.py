from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (base or project_root() / path).resolve()

