from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf


DEFAULT_MODELS = [
    Path("model_export_pipeline/embedded_classifier_only/artifacts/biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.tflite"),
    Path("model_export_pipeline/submission_custom_waveform/your_lastname_task3_1/your_submission_model/biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.tflite"),
]


def is_simd_friendly_pair(rows: int, cols: int) -> bool:
    # Conservative embedded-SIMD heuristic: inner dimensions should avoid tiny
    # odd tails and preferably align to 4-lane int8 kernels.
    if rows <= 0 or cols <= 0:
        return False
    if rows == 1 or cols == 1:
        return max(rows, cols) % 4 == 0
    return rows % 4 == 0 and cols % 4 == 0


def collect_shape_patterns(interpreter: tf.lite.Interpreter) -> Counter[tuple[int, int]]:
    patterns: Counter[tuple[int, int]] = Counter()
    for detail in interpreter.get_tensor_details():
        shape = [int(x) for x in detail["shape"] if int(x) > 0]
        if len(shape) < 2:
            continue
        patterns[(shape[-2], shape[-1])] += 1
    return patterns


def tensor_bytes(detail: dict) -> int:
    shape = [int(x) for x in detail["shape"] if int(x) > 0]
    dtype = np.dtype(detail["dtype"])
    size = int(np.prod(shape)) if shape else 1
    return size * dtype.itemsize


def audit_model(path: Path, top_k: int) -> None:
    print()
    print("=" * 78)
    print(f"Model: {path}")
    print("=" * 78)

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    tensors = interpreter.get_tensor_details()

    print(f"File size: {path.stat().st_size:,} bytes")
    print("Inputs:")
    for item in inputs:
        print(f"  {item['name']}: shape={list(item['shape'])} dtype={np.dtype(item['dtype']).name} quant={item.get('quantization')}")
    print("Outputs:")
    for item in outputs:
        print(f"  {item['name']}: shape={list(item['shape'])} dtype={np.dtype(item['dtype']).name} quant={item.get('quantization')}")

    total_tensor_bytes = sum(tensor_bytes(t) for t in tensors)
    print(f"Tensor metadata count: {len(tensors)}")
    print(f"Naive tensor byte sum: {total_tensor_bytes:,} bytes")

    patterns = collect_shape_patterns(interpreter)
    print()
    print("Dimension patterns from tensor trailing dims:")
    for (rows, cols), count in patterns.most_common(top_k):
        marker = "" if is_simd_friendly_pair(rows, cols) else "  WARN not 4-lane friendly"
        print(f"  {rows}x{cols:<8} count={count:<5}{marker}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TFLite tensor shapes for embedded-SIMD friendliness")
    parser.add_argument("models", nargs="*", help="TFLite files to inspect")
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.models] if args.models else DEFAULT_MODELS
    for path in paths:
        if path.exists():
            audit_model(path, args.top_k)
        else:
            print(f"missing: {path}")


if __name__ == "__main__":
    main()
