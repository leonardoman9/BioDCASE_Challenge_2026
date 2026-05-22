"""BioDCASE 2026 TinyML bird classification project."""

from __future__ import annotations

import os
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_root / "outputs" / "matplotlib"))

__version__ = "0.1.0"
