from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PIPELINE_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PIPELINE_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import tensorflow as tf


DEFAULT_SAVED_MODEL = PIPELINE_ROOT / ".conversion_work" / "run" / "output" / "saved_model"
DEFAULT_FEATURES = SCRIPT_DIR / "representative_data" / "validation_frontend_features.npy"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "biodcase_best_06717_classifier_only_int8.tflite"
DEFAULT_METADATA = SCRIPT_DIR / "artifacts" / "biodcase_best_06717_classifier_only_int8.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize the classifier-only SavedModel to int8 TFLite")
    parser.add_argument("--saved-model", default=str(DEFAULT_SAVED_MODEL), help="SavedModel directory")
    parser.add_argument("--features-npy", default=str(DEFAULT_FEATURES), help="Representative frontend features [.npy]")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output TFLite path")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Output metadata json path")
    parser.add_argument("--num-calibration-samples", type=int, default=128, help="Max feature samples for calibration")
    parser.add_argument("--inference-io", choices=("int8", "float32"), default="int8", help="Input/output dtype")
    parser.add_argument("--allow-select-tf-ops", action="store_true", help="Retry with SELECT_TF_OPS on failure")
    parser.add_argument("--disable-new-quantizer", action="store_true", help="Use legacy quantizer")
    parser.add_argument("--disable-per-channel", action="store_true", help="Disable per-channel quantization")
    return parser.parse_args()


def get_saved_model_input(saved_model_dir: Path) -> tuple[str, list[int]]:
    loaded = tf.saved_model.load(str(saved_model_dir))
    signature = loaded.signatures["serving_default"]
    _, kw = signature.structured_input_signature
    if len(kw) != 1:
        raise ValueError(f"Expected single SavedModel input, got {list(kw.keys())}")
    name, spec = next(iter(kw.items()))
    return name, [int(v) for v in spec.shape]


def adapt_feature_sample(sample: np.ndarray, target_shape: list[int]) -> np.ndarray:
    if sample.ndim != 2:
        raise ValueError(f"Expected single sample [features, frames], got {sample.shape}")
    batched = sample[None, ...]
    candidates = [
        batched,
        np.transpose(batched, (0, 2, 1)),
        batched[:, None, :, :],
        np.transpose(batched[:, None, :, :], (0, 3, 2, 1)),
    ]
    normalized_target = [int(v) for v in target_shape]
    for candidate in candidates:
        if list(candidate.shape) == normalized_target:
            return candidate.astype(np.float32, copy=False)
    raise ValueError(f"Could not adapt sample shape {list(sample.shape)} to SavedModel input {normalized_target}")


def representative_dataset(features_npy: Path, target_shape: list[int], limit: int):
    features = np.load(features_npy).astype(np.float32, copy=False)
    if features.ndim != 3:
        raise ValueError(f"Expected features array [N, F, T], got {features.shape}")

    count = min(limit, int(features.shape[0]))
    if count <= 0:
        raise ValueError("Representative feature file is empty")

    def generator():
        for idx in range(count):
            x = adapt_feature_sample(features[idx], target_shape)
            yield [x]

    return generator, count


def extract_quantization(detail: dict) -> tuple[float, int]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return float(scale), int(zero_point)


def verify_model(model_path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), experimental_delegates=[], num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details:
        shape = [int(x) for x in detail["shape"]]
        if detail["dtype"] == np.int8:
            data = np.zeros(shape, dtype=np.int8)
        else:
            data = np.zeros(shape, dtype=np.float32)
        interpreter.set_tensor(detail["index"], data)

    interpreter.invoke()

    result = {
        "tflite_path": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "inputs": [],
        "outputs": [],
    }

    for detail in input_details:
        scale, zero_point = extract_quantization(detail)
        result["inputs"].append(
            {
                "name": detail["name"],
                "shape": [int(x) for x in detail["shape"]],
                "dtype": str(detail["dtype"]),
                "quantization": [scale, zero_point],
            }
        )

    for detail in output_details:
        scale, zero_point = extract_quantization(detail)
        tensor = interpreter.get_tensor(detail["index"])
        result["outputs"].append(
            {
                "name": detail["name"],
                "shape": [int(x) for x in detail["shape"]],
                "dtype": str(detail["dtype"]),
                "quantization": [scale, zero_point],
                "min": float(np.min(tensor)),
                "max": float(np.max(tensor)),
            }
        )

    return result


def convert(args: argparse.Namespace, use_select_tf_ops: bool) -> tuple[bytes, int, list[int], str]:
    saved_model_dir = Path(args.saved_model).expanduser().resolve()
    input_name, input_shape = get_saved_model_input(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    rep_dataset, num_samples = representative_dataset(
        Path(args.features_npy).expanduser().resolve(),
        input_shape,
        args.num_calibration_samples,
    )
    converter.representative_dataset = rep_dataset

    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if use_select_tf_ops:
        supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
    converter.target_spec.supported_ops = supported_ops
    if args.inference_io == "int8":
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    converter.experimental_new_quantizer = not args.disable_new_quantizer
    if args.disable_per_channel:
        converter._experimental_disable_per_channel = True

    return converter.convert(), num_samples, input_shape, input_name


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    use_select_tf_ops = False
    try:
        tflite_bytes, num_calibration_samples, input_shape, input_name = convert(args, use_select_tf_ops=False)
    except Exception as exc:
        if not args.allow_select_tf_ops:
            raise
        print(f"builtin int8 conversion failed, retrying with SELECT_TF_OPS: {exc}")
        use_select_tf_ops = True
        tflite_bytes, num_calibration_samples, input_shape, input_name = convert(args, use_select_tf_ops=True)

    output_path.write_bytes(tflite_bytes)
    metadata = verify_model(output_path)
    metadata["used_select_tf_ops"] = use_select_tf_ops
    metadata["num_calibration_samples"] = num_calibration_samples
    metadata["inference_io"] = args.inference_io
    metadata["used_new_quantizer"] = not args.disable_new_quantizer
    metadata["disable_per_channel"] = args.disable_per_channel
    metadata["saved_model_input_name"] = input_name
    metadata["saved_model_input_shape"] = input_shape
    metadata["representative_features_file"] = str(Path(args.features_npy).expanduser().resolve())
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"tflite:   {output_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
