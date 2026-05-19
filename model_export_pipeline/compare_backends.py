from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch

from pytorch_to_onnx import instantiate_model, load_checkpoint, resolve_cfg


REPO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPO_ROOT / "biodcase_model"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biodcase_edge.data.audio import load_waveform  # noqa: E402
from biodcase_edge.data.dataset import AudioRecord, collect_records  # noqa: E402
from biodcase_edge.metrics.classification import summarize_classification  # noqa: E402


DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "biodcase_best_06717.ckpt"
DEFAULT_DATASET_DIR = SCRIPT_DIR / "eval_data" / "BioDCASE2026_TinyML_Development_Dataset"
DEFAULT_CLASS_MAP = SCRIPT_DIR / "eval_data" / "class_map.json"
DEFAULT_ONNX = SCRIPT_DIR / "exports" / "biodcase_best_06717.onnx"
DEFAULT_TFLITES = [
    SCRIPT_DIR / "exports" / "biodcase_best_06717_float32.tflite",
    SCRIPT_DIR / "exports" / "biodcase_best_06717_dynamic.tflite",
]
DEFAULT_OUTPUT = SCRIPT_DIR / "exports" / "backend_comparison_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch, ONNX, and TFLite on the same validation set")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Lightning checkpoint")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Dataset root containing Validation/")
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP), help="Class map JSON")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="ONNX model path")
    parser.add_argument(
        "--tflite",
        action="append",
        default=None,
        help="TFLite model path. Repeatable. If omitted, uses local float32/dynamic artifacts when present.",
    )
    parser.add_argument("--split", default="validation", choices=("validation", "val", "test"), help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick smoke test")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT), help="Where to save the comparison JSON")
    return parser.parse_args()


def load_class_map(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def compute_model_spectrogram(model: torch.nn.Module, waveform_batch: torch.Tensor) -> torch.Tensor:
    x = waveform_batch
    if x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)
    elif x.dim() != 2:
        raise ValueError(f"Unexpected waveform batch shape: {tuple(x.shape)}")

    if model.spectrogram_type == "mel":
        x = model.mel_transform(x)
    elif model.spectrogram_type == "linear_stft":
        x = model.stft_transform(x)
    elif model.spectrogram_type == "linear_triangular":
        x = model.stft_transform(x)
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, model.linear_filterbank.T.to(x.device))
        x = x.permute(0, 2, 1)
        x = model.amplitude_to_db(x)
    elif model.spectrogram_type in ("combined_log_linear", "fully_learnable"):
        x_mag_filt = model.combined_log_linear_spec(x)
        x = model.amplitude_to_db(x_mag_filt**2)
    else:
        raise ValueError(f"Unsupported spectrogram_type for export comparison: {model.spectrogram_type}")

    if x.dim() != 3:
        raise ValueError(f"Unexpected spectrogram shape: {tuple(x.shape)}")
    return x


def adapt_spectrogram_for_shape(spec: np.ndarray, target_shape: list[int]) -> np.ndarray:
    candidates = [spec]
    if spec.ndim == 3:
        candidates.append(np.transpose(spec, (0, 2, 1)))
        candidates.append(spec[:, None, :, :])
        candidates.append(np.transpose(spec[:, None, :, :], (0, 3, 2, 1)))

    normalized_target = [int(x) for x in target_shape]
    for candidate in candidates:
        if list(candidate.shape) == normalized_target:
            return candidate.astype(np.float32, copy=False)

    raise ValueError(f"Could not adapt spectrogram shape {list(spec.shape)} to runtime shape {normalized_target}")


def run_pytorch(model: torch.nn.Module, waveform_batch: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(waveform_batch)
    return logits.detach().cpu().numpy()


def run_onnx(session: ort.InferenceSession, spec_batch: np.ndarray) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    adapted = adapt_spectrogram_for_shape(spec_batch, list(input_meta.shape))
    outputs = session.run(None, {input_meta.name: adapted})
    return np.asarray(outputs[0], dtype=np.float32)


def run_tflite(interpreter: tf.lite.Interpreter, spec_batch: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    adapted = adapt_spectrogram_for_shape(spec_batch, list(input_detail["shape"]))

    if input_detail["dtype"] == np.int8:
        scale, zero_point = input_detail["quantization"]
        if scale == 0:
            raise ValueError("Invalid TFLite int8 input quantization scale=0")
        adapted = np.round(adapted / scale + zero_point).astype(np.int8)
    else:
        adapted = adapted.astype(np.float32, copy=False)

    interpreter.set_tensor(input_detail["index"], adapted)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])

    if output_detail["dtype"] == np.int8:
        scale, zero_point = output_detail["quantization"]
        output = scale * (output.astype(np.float32) - zero_point)
    return np.asarray(output, dtype=np.float32)


def backend_name_from_path(path: Path) -> str:
    stem = path.stem
    return stem.replace("biodcase_best_06717_", "tflite_")


def summarize_backend(targets: list[int], preds: list[int]) -> dict[str, float]:
    summary = summarize_classification(targets, preds)
    return {
        "accuracy": summary.accuracy,
        "macro_f1": summary.macro_f1,
        "weighted_f1": summary.weighted_f1,
        "weighted_precision": summary.weighted_precision,
        "weighted_recall": summary.weighted_recall,
    }


def compare_logits(reference: np.ndarray, candidate: np.ndarray, ref_preds: np.ndarray, cand_preds: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(candidate - reference)
    return {
        "prediction_agreement": float(np.mean(cand_preds == ref_preds)),
        "mean_abs_logit_diff": float(abs_diff.mean()),
        "max_abs_logit_diff": float(abs_diff.max()),
    }


def resolve_tflite_paths(args: argparse.Namespace) -> list[Path]:
    if args.tflite:
        return [Path(p).expanduser().resolve() for p in args.tflite]
    return [p.resolve() for p in DEFAULT_TFLITES if p.exists()]


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    class_map_path = Path(args.class_map).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    tflite_paths = resolve_tflite_paths(args)

    checkpoint = load_checkpoint(checkpoint_path)
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"]).cpu().eval()

    class_map = load_class_map(class_map_path)
    records = collect_records(dataset_dir, args.split, class_map)
    if args.limit is not None:
        records = records[: args.limit]

    onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tflite_runtimes = []
    for path in tflite_paths:
        interpreter = tf.lite.Interpreter(
            model_path=str(path),
            experimental_delegates=[],
            num_threads=1,
        )
        interpreter.allocate_tensors()
        tflite_runtimes.append((backend_name_from_path(path), path, interpreter))

    targets: list[int] = []
    pytorch_preds: list[int] = []
    onnx_preds: list[int] = []
    tflite_preds: dict[str, list[int]] = {name: [] for name, _, _ in tflite_runtimes}
    pytorch_logits_all: list[np.ndarray] = []
    onnx_logits_all: list[np.ndarray] = []
    tflite_logits_all: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in tflite_runtimes}

    total = len(records)
    for idx, record in enumerate(records, start=1):
        waveform = load_waveform(record.path, int(cfg["data"]["sample_rate"]), float(cfg["data"]["clip_duration"]))
        waveform_batch = waveform.unsqueeze(0)
        spec_batch = compute_model_spectrogram(model, waveform_batch).detach().cpu().numpy().astype(np.float32)

        pytorch_logits = run_pytorch(model, waveform_batch)
        onnx_logits = run_onnx(onnx_session, spec_batch)

        pytorch_logits_all.append(pytorch_logits[0])
        onnx_logits_all.append(onnx_logits[0])
        pytorch_preds.append(int(np.argmax(pytorch_logits[0])))
        onnx_preds.append(int(np.argmax(onnx_logits[0])))

        for name, _, interpreter in tflite_runtimes:
            logits = run_tflite(interpreter, spec_batch)
            tflite_logits_all[name].append(logits[0])
            tflite_preds[name].append(int(np.argmax(logits[0])))

        targets.append(int(record.label))

        if idx % 50 == 0 or idx == total:
            print(f"processed {idx}/{total}: {record.path.name}")

    pytorch_logits_np = np.stack(pytorch_logits_all)
    onnx_logits_np = np.stack(onnx_logits_all)
    pytorch_preds_np = np.asarray(pytorch_preds)
    onnx_preds_np = np.asarray(onnx_preds)

    results: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "num_samples": total,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "tflite_models": {name: str(path) for name, path, _ in tflite_runtimes},
        "metrics": {
            "pytorch": summarize_backend(targets, pytorch_preds),
            "onnx": summarize_backend(targets, onnx_preds),
        },
        "comparisons": {
            "onnx_vs_pytorch": compare_logits(pytorch_logits_np, onnx_logits_np, pytorch_preds_np, onnx_preds_np),
        },
    }

    for name, _, _ in tflite_runtimes:
        logits_np = np.stack(tflite_logits_all[name])
        preds_np = np.asarray(tflite_preds[name])
        results["metrics"][name] = summarize_backend(targets, tflite_preds[name])
        results["comparisons"][f"{name}_vs_pytorch"] = compare_logits(
            pytorch_logits_np,
            logits_np,
            pytorch_preds_np,
            preds_np,
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nsaved: {output_json}")


if __name__ == "__main__":
    main()
