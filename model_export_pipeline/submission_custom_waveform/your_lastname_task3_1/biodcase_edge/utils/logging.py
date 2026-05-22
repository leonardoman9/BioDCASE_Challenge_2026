from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_run_dir(log_root: str | Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(log_root) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

