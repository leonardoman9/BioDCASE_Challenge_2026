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
from biodcase_edge.data.dataset import collect_records  # noqa: E402
from biodcase_edge.metrics.classification import summarize_classification  # noqa: E402


DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "biodcase_best_06717.ckpt"
DEFAULT_DATASET_DIR = SCRIPT_DIR / "eval_data" / "BioDCASE2026_TinyML_Development_Dataset"
DEFAULT_CLASS_MAP = SCRIPT_DIR / "eval_data" / "class_map.json"
DEFAULT_ONNX = SCRIPT_DIR / "exports" / "biodcase_best_06717_waveform_static.onnx"
DEFAULT_TFLITE = SCRIPT_DIR / "exports" / "biodcase_best_06717_waveform_float32.tflite"
DEFAULT_OUTPUT = SCRIPT_DIR / "exports" / "backend_comparison_waveform_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch, ONNX, and waveform-input TFLite on the same validation set")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Lightning checkpoint")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Dataset root containing Validation/")
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP), help="Class map JSON")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Waveform ONNX model path")
    parser.add_argument("--tflite", default=str(DEFAULT_TFLITE), help="Waveform TFLite model path")
    parser.add_argument("--split", default="validation", choices=("validation", "val", "test"), help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick smoke test")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT), help="Where to save the comparison JSON")
    return parser.parse_args()


def load_class_map(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def adapt_waveform_for_onnx(waveform_batch: np.ndarray, target_shape: list[int]) -> np.ndarray:
    candidates = [waveform_batch]
    if waveform_batch.ndim == 2:
        candidates.append(waveform_batch[:, None, :])
        candidates.append(waveform_batch[:, :, None])
    elif waveform_batch.ndim == 3:
        candidates.append(np.transpose(waveform_batch, (0, 2, 1)))

    normalized_target = [int(x) for x in target_shape]
    for candidate in candidates:
        if list(candidate.shape) == normalized_target:
            return candidate.astype(np.float32, copy=False)
    raise ValueError(f"Could not adapt waveform shape {list(waveform_batch.shape)} to ONNX runtime shape {normalized_target}")


def adapt_waveform_for_tflite(waveform_batch: np.ndarray, target_shape: list[int]) -> np.ndarray:
    candidates = [waveform_batch]
    if waveform_batch.ndim == 2:
        candidates.append(waveform_batch[:, :, None])
        candidates.append(waveform_batch[:, None, :])
    elif waveform_batch.ndim == 3:
        candidates.append(np.transpose(waveform_batch, (0, 2, 1)))

    normalized_target = [int(x) for x in target_shape]
    for candidate in candidates:
        if list(candidate.shape) == normalized_target:
            return candidate.astype(np.float32, copy=False)
    raise ValueError(f"Could not adapt waveform shape {list(waveform_batch.shape)} to TFLite runtime shape {normalized_target}")


def run_pytorch(model: torch.nn.Module, waveform_batch: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(waveform_batch)
    return logits.detach().cpu().numpy()


def run_onnx(session: ort.InferenceSession, waveform_batch: np.ndarray) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    adapted = adapt_waveform_for_onnx(waveform_batch, list(input_meta.shape))
    outputs = session.run(None, {input_meta.name: adapted})
    return np.asarray(outputs[0], dtype=np.float32)


def run_tflite(interpreter: tf.lite.Interpreter, waveform_batch: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    adapted = adapt_waveform_for_tflite(waveform_batch, list(input_detail["shape"]))

    input_dtype = input_detail["dtype"]
    output_dtype = output_detail["dtype"]

    if input_dtype == np.int8:
        input_scale, input_zero_point = input_detail["quantization"]
        if input_scale == 0.0:
            raise ValueError("Invalid int8 input quantization scale 0.0")
        adapted = np.clip(adapted / input_scale + input_zero_point, -128, 127).astype(np.int8)
    elif input_dtype == np.float32:
        adapted = adapted.astype(np.float32, copy=False)
    else:
        raise NotImplementedError(f"Unsupported TFLite input dtype: {input_dtype}")

    interpreter.set_tensor(input_detail["index"], adapted)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])

    if output_dtype == np.int8:
        output_scale, output_zero_point = output_detail["quantization"]
        output = (output.astype(np.float32) - output_zero_point) * output_scale
        return np.asarray(output, dtype=np.float32)
    if output_dtype == np.float32:
        return np.asarray(output, dtype=np.float32)
    raise NotImplementedError(f"Unsupported TFLite output dtype: {output_dtype}")


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


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    class_map_path = Path(args.class_map).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()
    tflite_path = Path(args.tflite).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    checkpoint = load_checkpoint(checkpoint_path)
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"]).cpu().eval()

    class_map = load_class_map(class_map_path)
    records = collect_records(dataset_dir, args.split, class_map)
    if args.limit is not None:
        records = records[: args.limit]

    onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), experimental_delegates=[], num_threads=1)
    interpreter.allocate_tensors()

    targets: list[int] = []
    pytorch_preds: list[int] = []
    onnx_preds: list[int] = []
    tflite_preds: list[int] = []
    pytorch_logits_all: list[np.ndarray] = []
    onnx_logits_all: list[np.ndarray] = []
    tflite_logits_all: list[np.ndarray] = []

    total = len(records)
    for idx, record in enumerate(records, start=1):
        waveform = load_waveform(record.path, int(cfg["data"]["sample_rate"]), float(cfg["data"]["clip_duration"]))
        waveform_batch_torch = waveform
        waveform_batch_np = waveform.detach().cpu().numpy().astype(np.float32)

        pytorch_logits = run_pytorch(model, waveform_batch_torch)
        onnx_logits = run_onnx(onnx_session, waveform_batch_np)
        tflite_logits = run_tflite(interpreter, waveform_batch_np)

        pytorch_logits_all.append(pytorch_logits[0])
        onnx_logits_all.append(onnx_logits[0])
        tflite_logits_all.append(tflite_logits[0])

        pytorch_preds.append(int(np.argmax(pytorch_logits[0])))
        onnx_preds.append(int(np.argmax(onnx_logits[0])))
        tflite_preds.append(int(np.argmax(tflite_logits[0])))
        targets.append(int(record.label))

        if idx % 50 == 0 or idx == total:
            print(f"processed {idx}/{total}: {record.path.name}")

    pytorch_logits_np = np.stack(pytorch_logits_all)
    onnx_logits_np = np.stack(onnx_logits_all)
    tflite_logits_np = np.stack(tflite_logits_all)
    pytorch_preds_np = np.asarray(pytorch_preds)
    onnx_preds_np = np.asarray(onnx_preds)
    tflite_preds_np = np.asarray(tflite_preds)

    results: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "num_samples": total,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "tflite": str(tflite_path),
        "metrics": {
            "pytorch": summarize_backend(targets, pytorch_preds),
            "onnx_waveform": summarize_backend(targets, onnx_preds),
            "tflite_waveform_float32": summarize_backend(targets, tflite_preds),
        },
        "comparisons": {
            "onnx_waveform_vs_pytorch": compare_logits(pytorch_logits_np, onnx_logits_np, pytorch_preds_np, onnx_preds_np),
            "tflite_waveform_float32_vs_pytorch": compare_logits(
                pytorch_logits_np, tflite_logits_np, pytorch_preds_np, tflite_preds_np
            ),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nsaved: {output_json}")


if __name__ == "__main__":
    main()
