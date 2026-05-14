from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from biodcase_edge.cli.common import load_config, parse_config_args
from biodcase_edge.data.dataset import build_class_map, collect_records
from biodcase_edge.utils import configure_logging, write_json
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def main(argv=None) -> None:
    config_name, overrides = parse_config_args("Extract teacher soft labels for BioDCASE", argv)
    cfg = load_config(config_name, overrides)
    configure_logging(str(cfg.logging.level))

    class_map = build_class_map(cfg.data.dataset_dir, cfg.data.class_map_path)
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    background_idx = class_map.get("Background")
    teacher_cfg = cfg.distillation.get("teacher", {})
    species_map = dict(teacher_cfg.get("species_map", {}))

    species_list_path = teacher_cfg.get("species_list_path", None)
    analyzer = _load_birdnet_analyzer(species_list_path)
    if analyzer is None:
        log.warning("BirdNET is unavailable; writing weak-background soft labels.")

    split_records = {
        split: collect_records(cfg.data.dataset_dir, split, class_map)
        for split in ("train", "validation")
    }
    total_records = sum(len(records) for records in split_records.values())

    soft_labels: dict[str, list[float]] = {}
    processed = 0
    progress = tqdm(
        total=total_records,
        desc="Extract soft labels",
        unit="file",
        dynamic_ncols=True,
    )
    for split, records in split_records.items():
        for record in records:
            key = str(record.path.relative_to(cfg.data.dataset_dir))
            if analyzer is None:
                vector = _default_teacher_vector(len(class_names), background_idx, error=True)
            else:
                vector = _teacher_vector(
                    analyzer,
                    record.path,
                    class_names,
                    species_map,
                    background_idx=background_idx,
                    confidence_threshold=float(teacher_cfg.get("confidence_threshold", 0.05)),
                )
            soft_labels[key] = vector
            processed += 1
            progress.set_postfix(split=split, file=record.path.name[:36], refresh=False)
            progress.update(1)
            if processed % 250 == 0:
                log.info("Processed %s/%s files", processed, total_records)
    progress.close()

    out_path = Path(cfg.distillation.soft_labels_path)
    if out_path.suffix == ".json":
        output_file = out_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        output_file = out_path / "soft_labels.json"
    payload = {
        "metadata": {
            "teacher": "BirdNET" if analyzer is not None else "background_absorption_fallback",
            "num_classes": len(class_names),
            "class_names": class_names,
            "confidence_threshold": float(teacher_cfg.get("confidence_threshold", 0.05)),
            "files_processed": processed,
            "background_policy": "background_absorption",
        },
        "soft_labels": soft_labels,
    }
    write_json(payload, output_file)
    log.info("Soft labels saved to %s", output_file)


def _hard_vector(label: int, num_classes: int) -> list[float]:
    vector = [0.0] * num_classes
    vector[label] = 1.0
    return vector


def _load_birdnet_analyzer(species_list_path: str | Path | None = None) -> Any | None:
    try:
        from birdnetlib import Recording  # noqa: F401
        from birdnetlib.analyzer import Analyzer

        kwargs = {}
        if species_list_path is not None:
            path = Path(species_list_path)
            if path.exists():
                kwargs["custom_species_list_path"] = str(path)
                log.info("BirdNET using custom species list: %s", path)
            else:
                log.warning("Species list not found: %s — using full BirdNET vocabulary", path)
        return Analyzer(**kwargs)
    except Exception as exc:
        log.warning("Could not initialize BirdNET analyzer: %s", exc)
        return None


def _teacher_vector(
    analyzer: Any,
    path: Path,
    class_names: list[str],
    species_map: dict[str, str],
    background_idx: int | None,
    confidence_threshold: float,
) -> list[float]:
    try:
        from birdnetlib import Recording

        recording = Recording(analyzer, str(path), min_conf=confidence_threshold)
        recording.analyze()
        detections = getattr(recording, "detections", []) or []
    except Exception as exc:
        log.warning("BirdNET failed for %s: %s", path, exc)
        return _default_teacher_vector(len(class_names), background_idx, error=True)

    scores = [0.0] * len(class_names)
    mapped = {species_map.get(name, name).lower(): idx for idx, name in enumerate(class_names)}
    found_target = False
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = str(det.get("common_name") or det.get("scientific_name") or det.get("label") or "").lower()
        confidence = float(det.get("confidence", det.get("score", 0.0)))
        if label in mapped:
            scores[mapped[label]] = max(scores[mapped[label]], confidence)
            found_target = True
        elif background_idx is not None:
            scores[background_idx] = max(scores[background_idx], confidence * 0.1)

    if not found_target and detections and background_idx is not None:
        scores[background_idx] = max(scores[background_idx], 0.3)
    if not detections and background_idx is not None:
        scores[background_idx] = max(scores[background_idx], 0.1)

    if sum(scores) <= 0:
        return _default_teacher_vector(len(class_names), background_idx, error=False)

    return scores


def _uniform(num_classes: int) -> list[float]:
    return [1.0 / num_classes] * num_classes


def _default_teacher_vector(
    num_classes: int,
    background_idx: int | None,
    *,
    error: bool,
) -> list[float]:
    labels = [0.0] * num_classes
    if background_idx is not None:
        labels[background_idx] = 0.1 if error else 0.0
    return labels


if __name__ == "__main__":
    main()
