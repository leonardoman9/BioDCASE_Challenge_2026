from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
QUANT_SCRIPT = SCRIPT_DIR / "quantize_classifier_only_int8.py"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"


VARIANTS = [
    {
        "name": "int8_io_new_quantizer",
        "flags": [],
    },
    {
        "name": "int8_io_legacy_quantizer",
        "flags": ["--disable-new-quantizer"],
    },
    {
        "name": "float32_io_legacy_quantizer",
        "flags": ["--inference-io", "float32", "--disable-new-quantizer"],
    },
    {
        "name": "int8_io_legacy_no_per_channel",
        "flags": ["--disable-new-quantizer", "--disable-per-channel"],
    },
    {
        "name": "float32_io_legacy_no_per_channel",
        "flags": ["--inference-io", "float32", "--disable-new-quantizer", "--disable-per-channel"],
    },
    {
        "name": "int8_io_legacy_select_tf_ops",
        "flags": ["--disable-new-quantizer", "--allow-select-tf-ops"],
    },
    {
        "name": "float32_io_legacy_select_tf_ops",
        "flags": ["--inference-io", "float32", "--disable-new-quantizer", "--allow-select-tf-ops"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try multiple classifier-only int8 quantization variants")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--summary-json", default=str(ARTIFACTS_DIR / "quantization_variant_summary.json"), help="Summary output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in VARIANTS:
        name = variant["name"]
        output = ARTIFACTS_DIR / f"biodcase_best_06717_classifier_only_{name}.tflite"
        metadata = ARTIFACTS_DIR / f"biodcase_best_06717_classifier_only_{name}.json"
        cmd = [
            args.python,
            str(QUANT_SCRIPT),
            "--output",
            str(output),
            "--metadata",
            str(metadata),
            *variant["flags"],
        ]
        print(f"running variant: {name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        results.append(
            {
                "name": name,
                "returncode": proc.returncode,
                "output_exists": output.exists(),
                "metadata_exists": metadata.exists(),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "command": cmd,
            }
        )
        print(f"variant {name}: returncode={proc.returncode} output_exists={output.exists()}")

    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
