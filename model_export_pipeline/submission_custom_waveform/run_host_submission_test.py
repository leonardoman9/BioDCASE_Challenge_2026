from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import yaml
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CHALLENGE_REPO = REPO_ROOT / "BioDCASE-Tiny-2026"
SOLUTION_DIR = SCRIPT_DIR / "mannini_task3_1"
STAGING_ROOT = SCRIPT_DIR / "_staging" / "BioDCASE-Tiny-2026"
MPLCONFIGDIR = SCRIPT_DIR / "_matplotlib"


def clean_modules() -> None:
    for name in list(sys.modules):
        if name == "inference_handler" or name == "feature_handler" or name.startswith("biodcase_edge"):
            sys.modules.pop(name, None)


def install_soundfile_fallback() -> None:
    if "soundfile" in sys.modules:
        return

    from scipy.io import wavfile
    import numpy as np

    module = types.ModuleType("soundfile")

    def read(path):
        sample_rate, audio = wavfile.read(path)
        if np.issubdtype(audio.dtype, np.integer):
            max_abs = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_abs
        else:
            audio = audio.astype(np.float32)
        return audio, sample_rate

    module.read = read
    sys.modules["soundfile"] = module


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def stage_solution() -> Path:
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    shutil.copytree(CHALLENGE_REPO, STAGING_ROOT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    for item in SOLUTION_DIR.iterdir():
        if item.name in {"config_submission.yaml", "your_submission_model", "your_generated_code"}:
            dst = STAGING_ROOT / "submission" / item.name
        else:
            dst = STAGING_ROOT / item.name

        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)

    return STAGING_ROOT / "submission"


def load_stage_module(module_name: str, path: Path):
    clean_modules()
    install_soundfile_fallback()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_host_inference(submission_dir: Path) -> dict:
    stage_root = submission_dir.parent
    if str(stage_root) not in sys.path:
        sys.path.insert(0, str(stage_root))

    inference_module = load_stage_module("inference_handler_stage", stage_root / "inference_handler.py")
    soundfile = sys.modules["soundfile"]

    old_cwd = Path.cwd()
    try:
        os.chdir(submission_dir)
        cfg = yaml.safe_load(open("config_submission.yaml", "r"))
        inference_handler = inference_module.InferenceHandler(cfg["inference_handler"])
        report_dir = Path(cfg["report_dir"])
        report_dir.mkdir(exist_ok=True)
        inference_scores_file = report_dir / "inference_scores.yaml"
        if inference_scores_file.exists():
            inference_scores_file.unlink()

        inference_handler.info()

        y_targets = []
        y_predictions_model = []
        y_predictions_tflite = []
        test_files = sorted(Path(cfg["test_file_dir"]).glob(f"**/*{cfg['test_files_ext']}"))
        for test_file in test_files:
            y_targets.append(inference_handler.get_label_dict()[test_file.parent.stem])
            waveform, fs = soundfile.read(test_file)
            y_hat_model, y_hat_tflite = inference_handler.infer(waveform, fs)
            y_predictions_model.extend(y_hat_model)
            if y_hat_tflite is not None:
                y_predictions_tflite.extend(y_hat_tflite)

        y_targets = np.asarray(y_targets)
        y_predictions_model = np.asarray(y_predictions_model)
        y_predictions_tflite = np.asarray(y_predictions_tflite)

        results = {
            "inference_score_dict": {
                "accuracy_inference": round(float(np.mean(y_targets == np.argmax(y_predictions_model, axis=-1))), 4),
                "accuracy_tflite": round(float(np.mean(y_targets == np.argmax(y_predictions_tflite, axis=-1))), 4),
                "roc_auc_inference": round(float(roc_auc_score(y_targets, softmax(y_predictions_model, axis=1), multi_class="ovr", average="macro")), 4),
                "roc_auc_tflite": round(float(roc_auc_score(y_targets, softmax(y_predictions_tflite, axis=1), multi_class="ovr", average="macro")), 4),
                "prediction_agreement_model_vs_tflite": round(
                    float(
                        np.mean(
                            np.argmax(y_predictions_model, axis=-1)
                            == np.argmax(y_predictions_tflite, axis=-1)
                        )
                    ),
                    4,
                ),
                "model_size_inference_bytes": inference_handler.get_model_size(),
                "model_size_tflite_bytes": inference_handler.get_tflite_model_file().stat().st_size if inference_handler.get_tflite_model_file() else "N/A",
                "macs_inference": inference_handler.get_macs_model() or "N/A",
                "macs_tflite": inference_handler.get_macs_tflite() or "N/A",
                "num_params_inference": inference_handler.get_num_params_model() or "N/A",
                "num_params_tflite": inference_handler.get_num_params_tflite() or "N/A",
            }
        }
        yaml.safe_dump(results, open(inference_scores_file, "w"), sort_keys=False)
        return {
            "submission_dir": str(submission_dir),
            "tflite_path": str(inference_handler.get_tflite_model_file()) if inference_handler.get_tflite_model_file() is not None else None,
            "inference_scores_file": str(inference_scores_file),
            "results": results,
        }
    finally:
        os.chdir(old_cwd)


def main() -> None:
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    submission_dir = stage_solution()
    result = run_host_inference(submission_dir)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
