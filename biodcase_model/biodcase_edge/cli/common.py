from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from biodcase_edge.utils.paths import project_root


def config_dir() -> Path:
    return project_root() / "config"


def load_config(config_name: str, overrides: Sequence[str] | None = None) -> DictConfig:
    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    normalize_paths(cfg)
    return cfg


def parse_config_args(description: str, argv: Iterable[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-name", default="baseline", help="Hydra config name without .yaml")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.config_name, args.overrides


def normalize_paths(cfg: DictConfig) -> None:
    root = project_root()
    for key in ("output_dir",):
        if key in cfg.project:
            cfg.project[key] = str(_resolve(cfg.project[key], root))
    if "logging" in cfg and "log_dir" in cfg.logging:
        cfg.logging.log_dir = str(_resolve(cfg.logging.log_dir, root))
    if "data" in cfg:
        for key in ("dataset_dir", "class_map_path"):
            if key in cfg.data:
                cfg.data[key] = str(_resolve(cfg.data[key], root))
    if "distillation" in cfg and "soft_labels_path" in cfg.distillation:
        cfg.distillation.soft_labels_path = str(_resolve(cfg.distillation.soft_labels_path, root))
    if "init_checkpoint" in cfg and cfg.init_checkpoint:
        cfg.init_checkpoint = str(_resolve(cfg.init_checkpoint, root))
    if "export" in cfg:
        for key in ("checkpoint", "output_path"):
            if key in cfg.export and cfg.export[key]:
                cfg.export[key] = str(_resolve(cfg.export[key], root))


def save_resolved_config(cfg: DictConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def _resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (root / path).resolve()
