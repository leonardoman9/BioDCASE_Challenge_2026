from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from biodcase_edge.cli.common import load_config, parse_config_args
from biodcase_edge.data.dataset import build_class_map, collect_records
from biodcase_edge.utils import configure_logging, write_json

log = logging.getLogger(__name__)


def main(argv=None) -> None:
    config_name, overrides = parse_config_args("Extract teacher soft labels for BioDCASE", argv)
    cfg = load_config(config_name, overrides)
    configure_logging(str(cfg.logging.level))

    class_map = build_class_map(cfg.data.dataset_dir, cfg.data.class_map_path)
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    teacher_cfg = cfg.distillation.get("teacher", {})
    fallback_to_hard = bool(teacher_cfg.get("fallback_to_hard_labels", True))
    species_map = dict(teacher_cfg.get("species_map", {}))

    analyzer = _load_birdnet_analyzer()
    if analyzer is None:
        if not fallback_to_hard:
            raise RuntimeError("BirdNET is unavailable and fallback_to_hard_labels=false")
        log.warning("BirdNET is unavailable; writing hard-label fallback soft labels.")

    soft_labels: dict[str, list[float]] = {}
    processed = 0
    for split in ("train", "validation"):
        for record in collect_records(cfg.data.dataset_dir, split, class_map):
            key = str(record.path.relative_to(cfg.data.dataset_dir))
            if record.class_name == "Background":
                vector = _hard_vector(record.label, len(class_names))
            elif analyzer is None:
                vector = _hard_vector(record.label, len(class_names))
            else:
                vector = _teacher_vector(
                    analyzer,
                    record.path,
                    class_names,
                    species_map,
                    hard_label=record.label,
                    fallback_to_hard=fallback_to_hard,
                    confidence_threshold=float(teacher_cfg.get("confidence_threshold", 0.05)),
                )
            soft_labels[key] = vector
            processed += 1
            if processed % 250 == 0:
                log.info("Processed %s files", processed)

    output_dir = Path(cfg.distillation.soft_labels_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "teacher": "BirdNET" if analyzer is not None else "hard_label_fallback",
            "num_classes": len(class_names),
            "class_names": class_names,
            "confidence_threshold": float(teacher_cfg.get("confidence_threshold", 0.05)),
            "files_processed": processed,
            "background_policy": str(teacher_cfg.get("background_policy", "hard_background")),
        },
        "soft_labels": soft_labels,
    }
    write_json(payload, output_dir / "soft_labels.json")
    log.info("Soft labels saved to %s", output_dir / "soft_labels.json")


def _hard_vector(label: int, num_classes: int) -> list[float]:
    vector = [0.0] * num_classes
    vector[label] = 1.0
    return vector


def _load_birdnet_analyzer() -> Any | None:
    try:
        from birdnetlib import Recording  # noqa: F401
        from birdnetlib.analyzer import Analyzer

        return Analyzer()
    except Exception as exc:
        log.warning("Could not initialize BirdNET analyzer: %s", exc)
        return None


def _teacher_vector(
    analyzer: Any,
    path: Path,
    class_names: list[str],
    species_map: dict[str, str],
    hard_label: int,
    fallback_to_hard: bool,
    confidence_threshold: float,
) -> list[float]:
    try:
        from birdnetlib import Recording

        recording = Recording(analyzer, str(path), min_conf=confidence_threshold)
        recording.analyze()
        detections = getattr(recording, "detections", []) or []
    except Exception as exc:
        log.warning("BirdNET failed for %s: %s", path, exc)
        return _hard_vector(hard_label, len(class_names)) if fallback_to_hard else _uniform(len(class_names))

    scores = [0.0] * len(class_names)
    mapped = {species_map.get(name, name).lower(): idx for idx, name in enumerate(class_names)}
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = str(det.get("common_name") or det.get("scientific_name") or det.get("label") or "").lower()
        confidence = float(det.get("confidence", det.get("score", 0.0)))
        if label in mapped:
            scores[mapped[label]] = max(scores[mapped[label]], confidence)

    total = sum(scores)
    if total <= 0:
        return _hard_vector(hard_label, len(class_names)) if fallback_to_hard else _uniform(len(class_names))
    return [score / total for score in scores]


def _uniform(num_classes: int) -> list[float]:
    return [1.0 / num_classes] * num_classes


if __name__ == "__main__":
    main()

