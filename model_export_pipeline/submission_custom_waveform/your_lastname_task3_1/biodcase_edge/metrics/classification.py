from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationSummary:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    weighted_precision: float
    weighted_recall: float


def summarize_classification(targets: Sequence[int], preds: Sequence[int]) -> ClassificationSummary:
    targets_arr = np.asarray(targets)
    preds_arr = np.asarray(preds)
    return ClassificationSummary(
        accuracy=float(accuracy_score(targets_arr, preds_arr)),
        macro_f1=float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(targets_arr, preds_arr, average="weighted", zero_division=0)),
        weighted_precision=float(precision_score(targets_arr, preds_arr, average="weighted", zero_division=0)),
        weighted_recall=float(recall_score(targets_arr, preds_arr, average="weighted", zero_division=0)),
    )

