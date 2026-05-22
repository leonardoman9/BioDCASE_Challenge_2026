from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_classification_report(
    targets: Sequence[int],
    preds: Sequence[int],
    class_names: Sequence[str],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    from sklearn.metrics import classification_report

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(
        targets,
        preds,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        targets,
        preds,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
    )
    (output_dir / f"{prefix}_classification_report.txt").write_text(report_text, encoding="utf-8")
    write_json(report_dict, output_dir / f"{prefix}_classification_report.json")
    return report_dict


def save_confusion_matrix(
    targets: Sequence[int],
    preds: Sequence[int],
    class_names: Sequence[str],
    output_dir: str | Path,
    prefix: str,
) -> Any:
    from sklearn.metrics import confusion_matrix

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    with (output_dir / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])
    return matrix
