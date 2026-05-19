from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import tensorflow as tf

from compare_waveform_backends import DEFAULT_DATASET_DIR

REPO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPO_ROOT / "biodcase_model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biodcase_edge.data.audio import load_waveform


DEFAULT_SAVED_MODEL = SCRIPT_DIR / ".conversion_work" / "run" / "output" / "saved_model"
DEFAULT_OUTPUT = SCRIPT_DIR / "exports" / "biodcase_best_06717_waveform_int8.tflite"
DEFAULT_METADATA = SCRIPT_DIR / "exports" / "biodcase_best_06717_waveform_int8.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a waveform SavedModel to full-int8 TFLite")
    parser.add_argument("--saved-model", default=str(DEFAULT_SAVED_MODEL), help="Path to the SavedModel directory")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Dataset root containing Validation/")
    parser.add_argument("--split", default="Validation", help="Dataset split directory to use for representative samples")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output .tflite path")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Output metadata .json path")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Expected sample rate")
    parser.add_argument("--clip-duration", type=float, default=3.0, help="Expected clip duration in seconds")
    parser.add_argument("--num-calibration-samples", type=int, default=128, help="Number of real waveforms to use for calibration")
    parser.add_argument("--allow-select-tf-ops", action="store_true", help="Allow SELECT_TF_OPS fallback if builtin int8 fails")
    parser.add_argument("--inference-io", choices=("int8", "float32"), default="int8", help="Input/output dtype for the exported TFLite model")
    parser.add_argument("--disable-new-quantizer", action="store_true", help="Use the legacy TFLite quantizer")
    parser.add_argument("--disable-per-channel", action="store_true", help="Disable per-channel quantization if the converter is unstable")
    return parser.parse_args()


def list_waveforms(dataset_dir: Path, split: str) -> list[Path]:
    root = dataset_dir / split
    if not root.exists():
        alt = dataset_dir / split.capitalize()
        if alt.exists():
            root = alt
    if not root.exists():
        raise FileNotFoundError(f"Split directory not found under {dataset_dir}: {split}")
    return sorted(root.glob("**/*.wav"))


def representative_dataset(
    wav_paths: list[Path],
    sample_rate: int,
    clip_duration: float,
):
    def generator():
        for path in wav_paths:
            waveform = load_waveform(path, sample_rate, clip_duration)
            x = waveform.detach().cpu().numpy().astype(np.float32)
            x = x.transpose(0, 1)[:, :, None]
            yield [x]

    return generator


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
        dtype = detail["dtype"]
        if dtype == np.int8:
            data = np.zeros(shape, dtype=np.int8)
        else:
            data = np.zeros(shape, dtype=np.float32)
        interpreter.set_tensor(detail["index"], data)

    interpreter.invoke()

    metadata = {
        "tflite_path": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "inputs": [],
        "outputs": [],
    }
    for detail in input_details:
        scale, zero_point = extract_quantization(detail)
        metadata["inputs"].append(
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
        metadata["outputs"].append(
            {
                "name": detail["name"],
                "shape": [int(x) for x in detail["shape"]],
                "dtype": str(detail["dtype"]),
                "quantization": [scale, zero_point],
                "min": float(np.min(tensor)),
                "max": float(np.max(tensor)),
            }
        )
    return metadata


def convert(args: argparse.Namespace, use_select_tf_ops: bool) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(Path(args.saved_model).resolve()))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    wav_paths = list_waveforms(Path(args.dataset_dir).resolve(), args.split)
    wav_paths = wav_paths[: args.num_calibration_samples]
    if not wav_paths:
        raise RuntimeError("No calibration waveforms found")
    converter.representative_dataset = representative_dataset(
        wav_paths=wav_paths,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
    )

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
    return converter.convert()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    use_select_tf_ops = False
    try:
        tflite_bytes = convert(args, use_select_tf_ops=False)
    except Exception as exc:
        if not args.allow_select_tf_ops:
            raise
        print(f"builtin int8 conversion failed, retrying with SELECT_TF_OPS: {exc}")
        use_select_tf_ops = True
        tflite_bytes = convert(args, use_select_tf_ops=True)

    output_path.write_bytes(tflite_bytes)
    metadata = verify_model(output_path)
    metadata["used_select_tf_ops"] = use_select_tf_ops
    metadata["num_calibration_samples"] = args.num_calibration_samples
    metadata["inference_io"] = args.inference_io
    metadata["used_new_quantizer"] = not args.disable_new_quantizer
    metadata["disable_per_channel"] = args.disable_per_channel
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"tflite:   {output_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
