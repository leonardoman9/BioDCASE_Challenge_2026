from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX = SCRIPT_DIR / "exports" / "biodcase_best_06717.onnx"
DEFAULT_OUTPUTS = {
    "float32": SCRIPT_DIR / "exports" / "biodcase_best_06717_float32.tflite",
    "float16": SCRIPT_DIR / "exports" / "biodcase_best_06717_float16.tflite",
    "dynamic": SCRIPT_DIR / "exports" / "biodcase_best_06717_dynamic.tflite",
    "int8": SCRIPT_DIR / "exports" / "biodcase_best_06717_int8.tflite",
}
DOCKER_IMAGE = "biodcase-onnx2tflite:tf215"
WORK_ROOT = SCRIPT_DIR / ".conversion_work"


DOCKERFILE_TEXT = """\
FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \\
    python3-pip \\
    cmake \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
    tensorflow==2.15.0 \\
    tf_keras \\
    onnx==1.15.0 \\
    onnxruntime \\
    onnxsim \\
    onnx_graphsurgeon \\
    psutil \\
    ai-edge-litert \\
    sng4onnx \\
    onnx2tf
"""


CONVERTER_TEXT = """\
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf


WORKSPACE = Path("/workspace")
INPUT_ONNX = WORKSPACE / "input" / "model.onnx"
INPUT_SPEC = WORKSPACE / "input" / "spec.json"
OUTPUT_DIR = WORKSPACE / "output"
SAVED_MODEL_DIR = OUTPUT_DIR / "saved_model"
FINAL_TFLITE = OUTPUT_DIR / "model.tflite"
METADATA_PATH = OUTPUT_DIR / "conversion_metadata.json"
AUTO_PROFILE_PATH = OUTPUT_DIR / "model_auto.json"
CALIBRATION_DIR = OUTPUT_DIR / "calibration_data"

ARTIFACT_CANDIDATES = {
    "float32": ["model_float32.tflite"],
    "float16": ["model_float16.tflite"],
    "dynamic": ["model_dynamic_range_quant.tflite"],
    "int8": [
        "model_full_integer_quant.tflite",
        "model_integer_quant.tflite",
        "model_integer_quant_with_int16_act.tflite",
    ],
}


def calibration_shape(input_shape: list[int], batch: int = 100) -> list[int]:
    if len(input_shape) == 3:
        _, features, frames = input_shape
        return [batch, frames, features]
    if len(input_shape) == 4:
        _, channels, features, frames = input_shape
        return [batch, frames, features, channels]
    raise ValueError(f"Unsupported input rank for calibration export: {input_shape}")


def write_calibration_file(input_shape: list[int]) -> Path:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    data = np.random.normal(0.0, 1.0, calibration_shape(input_shape)).astype(np.float32)
    data = np.clip(data, -3.0, 3.0)
    path = CALIBRATION_DIR / "input.npy"
    np.save(path, data)
    return path


def build_onnx2tf_command(spec: dict, profile_path: Path | None = None) -> list[str]:
    mode = spec["mode"]
    cmd = ["onnx2tf", "-i", str(INPUT_ONNX), "-o", str(SAVED_MODEL_DIR)]
    if mode == "dynamic":
        cmd.append("-odrqt")
    elif mode == "int8":
        calib_path = write_calibration_file(spec["input_shape"])
        cmd.extend(
            [
                "-oiqt",
                "-qt",
                "per-channel",
                "-iqd",
                "int8",
                "-oqd",
                "int8",
                "-cind",
                spec["input_name"],
                str(calib_path),
                "[0.0]",
                "[1.0]",
            ]
        )
    if profile_path is not None:
        cmd.extend(["-prf", str(profile_path)])
    return cmd


def run_onnx2tf_once(spec: dict, profile_path: Path | None = None) -> None:
    cmd = build_onnx2tf_command(spec, profile_path)
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_onnx2tf(spec: dict) -> None:
    try:
        run_onnx2tf_once(spec)
        return
    except subprocess.CalledProcessError:
        generated_profile = SAVED_MODEL_DIR / "model_auto.json"
        if not generated_profile.exists():
            raise
        AUTO_PROFILE_PATH.write_text(generated_profile.read_text(encoding="utf-8"), encoding="utf-8")
        print(
            f"onnx2tf failed once, retrying with auto-generated parameter replacement: {AUTO_PROFILE_PATH}",
            flush=True,
        )
        if SAVED_MODEL_DIR.exists():
            shutil.rmtree(SAVED_MODEL_DIR)
        run_onnx2tf_once(spec, AUTO_PROFILE_PATH)


def find_artifact(mode: str) -> Path:
    for relative_name in ARTIFACT_CANDIDATES[mode]:
        candidate = SAVED_MODEL_DIR / relative_name
        if candidate.exists():
            return candidate
    produced = sorted(str(path.relative_to(SAVED_MODEL_DIR)) for path in SAVED_MODEL_DIR.glob("*.tflite"))
    raise FileNotFoundError(
        f"No TFLite artifact for mode={mode}. Produced artifacts: {produced}"
    )


def verify_tflite(model_path: Path) -> dict:
    def int_list(values) -> list[int]:
        return [int(v) for v in values]

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_delegates=[],
        num_threads=1,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details:
        dtype = detail["dtype"]
        shape = detail["shape"]
        if dtype == np.int8:
            data = np.random.randint(-128, 128, shape, dtype=np.int8)
        else:
            data = np.random.normal(0.0, 1.0, shape).astype(np.float32)
            data = np.clip(data, -3.0, 3.0)
        interpreter.set_tensor(detail["index"], data)

    interpreter.invoke()

    outputs = []
    for detail in output_details:
        out = interpreter.get_tensor(detail["index"])
        outputs.append(
            {
                "name": detail["name"],
                "shape": int_list(detail["shape"]),
                "dtype": str(detail["dtype"]),
                "min": float(out.min()),
                "max": float(out.max()),
            }
        )

    result = {
        "selected_tflite": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "inputs": [
            {
                "name": d["name"],
                "shape": int_list(d["shape"]),
                "dtype": str(d["dtype"]),
                "quantization": [float(d["quantization"][0]), int(d["quantization"][1])],
            }
            for d in input_details
        ],
        "outputs": [
            {
                "name": d["name"],
                "shape": int_list(d["shape"]),
                "dtype": str(d["dtype"]),
                "quantization": [float(d["quantization"][0]), int(d["quantization"][1])],
            }
            for d in output_details
        ],
        "output_ranges": outputs,
    }
    METADATA_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    spec = json.loads(INPUT_SPEC.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_onnx2tf(spec)
    selected = find_artifact(spec["mode"])
    shutil.copy2(selected, FINAL_TFLITE)
    result = verify_tflite(FINAL_TFLITE)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an ONNX classifier to TFLite using Docker")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Input ONNX path")
    parser.add_argument("--output", default=None, help="Output TFLite path")
    parser.add_argument(
        "--quantize",
        choices=("float32", "float16", "dynamic", "int8"),
        default="float32",
        help="Conversion mode",
    )
    parser.add_argument(
        "--rebuild-image",
        action="store_true",
        help="Force rebuild of the Docker conversion image",
    )
    return parser.parse_args()


def inspect_onnx_input(onnx_path: Path) -> tuple[str, list[int]]:
    model = onnx.load(str(onnx_path))
    if len(model.graph.input) != 1:
        raise ValueError("This script currently supports single-input ONNX models only")

    tensor = model.graph.input[0]
    dims: list[int] = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_value <= 0:
            raise ValueError(
                f"ONNX input shape must be static for TFLite export. Found non-static dim in {tensor.name}."
            )
        dims.append(int(dim.dim_value))
    return tensor.name, dims


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def image_exists(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def build_image(build_dir: Path, rebuild: bool) -> None:
    if image_exists(DOCKER_IMAGE) and not rebuild:
        print(f"docker image already present: {DOCKER_IMAGE}")
        return

    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "Dockerfile").write_text(DOCKERFILE_TEXT, encoding="utf-8")
    subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE, str(build_dir)],
        check=True,
    )


def prepare_workspace(
    work_dir: Path,
    onnx_path: Path,
    output_path: Path,
    input_name: str,
    input_shape: list[int],
    mode: str,
) -> None:
    ensure_clean_dir(work_dir)
    (work_dir / "input").mkdir(parents=True, exist_ok=True)
    (work_dir / "output").mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, work_dir / "input" / "model.onnx")
    (work_dir / "input" / "spec.json").write_text(
        json.dumps(
            {
                "input_name": input_name,
                "input_shape": input_shape,
                "mode": mode,
                "output_filename": output_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (work_dir / "convert_inside_container.py").write_text(CONVERTER_TEXT, encoding="utf-8")


def run_container(work_dir: Path) -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            "TF_LITE_DISABLE_XNNPACK=1",
            "-v",
            f"{work_dir}:/workspace",
            DOCKER_IMAGE,
            "python",
            "/workspace/convert_inside_container.py",
        ],
        check=True,
    )


def finalize_outputs(work_dir: Path, output_path: Path) -> Path:
    produced = work_dir / "output" / "model.tflite"
    if not produced.exists():
        raise FileNotFoundError(f"TFLite model not produced: {produced}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, output_path)

    metadata_src = work_dir / "output" / "conversion_metadata.json"
    if metadata_src.exists():
        shutil.copy2(metadata_src, output_path.with_suffix(".json"))
    return output_path


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    return DEFAULT_OUTPUTS[args.quantize].resolve()


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    output_path = resolve_output_path(args)
    if not onnx_path.exists():
        raise SystemExit(f"ONNX file not found: {onnx_path}")

    input_name, input_shape = inspect_onnx_input(onnx_path)
    print(f"onnx input:  {input_name} {input_shape}")
    print(f"mode:        {args.quantize}")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    build_dir = WORK_ROOT / "docker_build"
    work_dir = WORK_ROOT / "run"

    build_image(build_dir, rebuild=args.rebuild_image)
    prepare_workspace(work_dir, onnx_path, output_path, input_name, input_shape, args.quantize)
    run_container(work_dir)
    final_path = finalize_outputs(work_dir, output_path)

    print(f"tflite:      {final_path}")
    print(f"metadata:    {final_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
