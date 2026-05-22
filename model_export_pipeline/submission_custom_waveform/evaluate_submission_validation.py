from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import soundfile
import yaml
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SOLUTION_DIR = SCRIPT_DIR / "mannini_task3_1"
DEFAULT_DATASET_DIR = (
    REPO_ROOT / "model_export_pipeline" / "eval_data" / "BioDCASE2026_TinyML_Development_Dataset" / "Validation"
)
DEFAULT_OUTPUT = SOLUTION_DIR / "validation_inference_scores.yaml"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_wavs(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("*/*.wav"))


def resolve_label_key(label_dict: dict[str, int], dataset_label: str) -> str:
    if dataset_label in label_dict:
        return dataset_label
    normalized = dataset_label.replace(" ", "_")
    if normalized in label_dict:
        return normalized
    raise KeyError(dataset_label)


def main() -> None:
    dataset_dir = DEFAULT_DATASET_DIR.resolve()
    output_path = DEFAULT_OUTPUT.resolve()

    if str(SOLUTION_DIR) not in sys.path:
        sys.path.insert(0, str(SOLUTION_DIR))

    inference_module = load_module("submission_inference_handler", SOLUTION_DIR / "inference_handler.py")

    old_cwd = Path.cwd()
    try:
        os.chdir(SOLUTION_DIR)
        cfg = yaml.safe_load(open("config_submission.yaml", "r"))
        inference_handler = inference_module.InferenceHandler(cfg["inference_handler"])

        y_targets: list[int] = []
        y_predictions_model: list[np.ndarray] = []
        y_predictions_tflite: list[np.ndarray] = []

        wav_files = collect_wavs(dataset_dir)
        total = len(wav_files)

        for idx, wav_path in enumerate(wav_files, start=1):
            label_name = resolve_label_key(inference_handler.get_label_dict(), wav_path.parent.name)
            y_targets.append(inference_handler.get_label_dict()[label_name])
            waveform, fs = soundfile.read(wav_path)
            y_hat_model, y_hat_tflite = inference_handler.infer(waveform, fs)
            y_predictions_model.extend(y_hat_model)
            if y_hat_tflite is None:
                raise RuntimeError("Expected TFLite predictions, got None")
            y_predictions_tflite.extend(y_hat_tflite)

            if idx % 50 == 0 or idx == total:
                print(f"processed {idx}/{total}: {wav_path.name}")

        y_targets_np = np.asarray(y_targets)
        y_predictions_model_np = np.asarray(y_predictions_model)
        y_predictions_tflite_np = np.asarray(y_predictions_tflite)

        results = {
            "dataset_dir": str(dataset_dir),
            "num_samples": int(total),
            "inference_score_dict": {
                "accuracy_inference": round(
                    float(np.mean(y_targets_np == np.argmax(y_predictions_model_np, axis=-1))), 4
                ),
                "accuracy_tflite": round(
                    float(np.mean(y_targets_np == np.argmax(y_predictions_tflite_np, axis=-1))), 4
                ),
                "roc_auc_inference": round(
                    float(
                        roc_auc_score(
                            y_targets_np,
                            softmax(y_predictions_model_np, axis=1),
                            multi_class="ovr",
                            average="macro",
                        )
                    ),
                    4,
                ),
                "roc_auc_tflite": round(
                    float(
                        roc_auc_score(
                            y_targets_np,
                            softmax(y_predictions_tflite_np, axis=1),
                            multi_class="ovr",
                            average="macro",
                        )
                    ),
                    4,
                ),
                "prediction_agreement_model_vs_tflite": round(
                    float(
                        np.mean(
                            np.argmax(y_predictions_model_np, axis=-1)
                            == np.argmax(y_predictions_tflite_np, axis=-1)
                        )
                    ),
                    4,
                ),
                "model_size_inference_bytes": inference_handler.get_model_size(),
                "model_size_tflite_bytes": inference_handler.get_tflite_model_file().stat().st_size,
                "macs_inference": inference_handler.get_macs_model() or "N/A",
                "macs_tflite": inference_handler.get_macs_tflite() or "N/A",
                "num_params_inference": inference_handler.get_num_params_model() or "N/A",
                "num_params_tflite": inference_handler.get_num_params_tflite() or "N/A",
            },
        }

        output_path.write_text(yaml.safe_dump(results, sort_keys=False), encoding="utf-8")
        print(yaml.safe_dump(results, sort_keys=False))
        print(f"saved: {output_path}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
