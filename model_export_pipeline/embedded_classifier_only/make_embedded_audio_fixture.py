from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
PROJECT_ROOT = REPO_ROOT / "biodcase_model"
SUBMISSION_DIR = PIPELINE_ROOT / "submission_custom_waveform" / "your_lastname_task3_1"
FIRMWARE_MAIN_DIR = SUBMISSION_DIR / "your_generated_code" / "src" / "main"
DEFAULT_WAVS = [
    PROJECT_ROOT
    / "BioDCASE2026_TinyML_Development_Dataset"
    / "Validation"
    / "Background"
    / "BioDCASE2026_TinyML_VAL_0001_Background.wav",
    PROJECT_ROOT
    / "BioDCASE2026_TinyML_Development_Dataset"
    / "Validation"
    / "Common Chaffinch"
    / "BioDCASE2026_TinyML_VAL_0051_Common_Chaffinch.wav",
    PROJECT_ROOT
    / "BioDCASE2026_TinyML_Development_Dataset"
    / "Validation"
    / "Eurasian Blackbird"
    / "BioDCASE2026_TinyML_VAL_0151_Eurasian_Blackbird.wav",
    PROJECT_ROOT
    / "BioDCASE2026_TinyML_Development_Dataset"
    / "Validation"
    / "Eurasian Blue Tit"
    / "BioDCASE2026_TinyML_VAL_0251_Eurasian_Blue_Tit.wav",
    PROJECT_ROOT
    / "BioDCASE2026_TinyML_Development_Dataset"
    / "Validation"
    / "Song Thrush"
    / "BioDCASE2026_TinyML_VAL_0450_Song_Thrush.wav",
]
DEFAULT_EXPECTED = SUBMISSION_DIR / "your_generated_code" / "audio_fixture_expected.json"
DEFAULT_BACKBONE = SUBMISSION_DIR / "your_submission_model" / "biodcase_backbone_streaming_float32.tflite"
DEFAULT_STEP = SUBMISSION_DIR / "your_submission_model" / "biodcase_streaming_step_float32.tflite"

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedded_classifier_only.models import build_frozen_frontend
from embedded_classifier_only.validate_stateful_tflite import (
    get_tflite_interpreter,
    run_backbone_tflite,
    run_step_tflite_loop,
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def crc32_bytes(data: bytes) -> str:
    return f"0x{zlib.crc32(data) & 0xFFFFFFFF:08X}"


def format_float_array(values: np.ndarray, per_line: int = 6) -> str:
    flat = values.astype(np.float32, copy=False).reshape(-1)
    lines = []
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line]
        lines.append("  " + ", ".join(f"{float(value):.9e}f" for value in chunk) + ",")
    return "\n".join(lines)


def prepare_waveform(wav_path: Path) -> tuple[np.ndarray, int]:
    module = load_module("submission_feature_handler", SUBMISSION_DIR / "feature_handler.py")
    handler = module.FeatureHandler(
        target_sample_rate=24000,
        target_wav_length_sec=3,
        normalize_peak=True,
        add_batch_dimension=True,
        channel_last=True,
    )
    waveform, sample_rate = sf.read(wav_path)
    prepared = handler.extract(waveform, fs=sample_rate).reshape(-1).astype(np.float32)
    if prepared.size != 72000:
        raise ValueError(f"Expected 72000 samples, got {prepared.size}")
    return prepared, int(sample_rate)


def compute_expected(
    waveform: np.ndarray,
    backbone_tflite: Path,
    step_tflite: Path,
) -> dict:
    frontend = build_frozen_frontend().cpu().eval()
    with torch.no_grad():
        features = frontend(torch.from_numpy(waveform).reshape(1, -1)).numpy().astype(np.float32)
        mean = features.mean(axis=(1, 2), keepdims=True)
        std = features.std(axis=(1, 2), ddof=1, keepdims=True) + 1e-5
        normalized_features = ((features - mean) / std).astype(np.float32)

    backbone = get_tflite_interpreter(backbone_tflite)
    step = get_tflite_interpreter(step_tflite)
    embeddings = run_backbone_tflite(backbone, normalized_features)
    logits = run_step_tflite_loop(step, embeddings).astype(np.float32)

    return {
        "audio_crc32": crc32_bytes(waveform.astype(np.float32).tobytes()),
        "normalized_feature_crc32": crc32_bytes(normalized_features.astype(np.float32).tobytes()),
        "backbone_output_crc32": crc32_bytes(embeddings.astype(np.float32).tobytes()),
        "final_logits_crc32": crc32_bytes(logits.astype(np.float32).tobytes()),
        "logits": [float(value) for value in logits.reshape(-1)],
        "predicted_index": int(np.argmax(logits, axis=1)[0]),
    }


def write_fixture_files(records: list[dict]) -> None:
    header = f"""#pragma once

#include <cstddef>

namespace audio_fixture {{

struct Fixture {{
  const char* name;
  const char* source_path;
  int source_sample_rate;
  const float* waveform;
  size_t waveform_samples;
  const char* expected_audio_crc32;
  const char* expected_input_crc32;
  const char* expected_backbone_crc32;
  const char* expected_final_logits_crc32;
  int expected_predicted_index;
}};

extern const Fixture kFixtures[];
extern const size_t kNumFixtures;

}}  // namespace audio_fixture
"""
    arrays = []
    table_rows = []
    for index, record in enumerate(records):
        wav_path = Path(record["wav_path"])
        waveform = record["waveform"]
        expected = record["expected"]
        array_name = f"kFixtureWaveform{index}"
        arrays.append(
            f"""const float {array_name}[] = {{
{format_float_array(waveform)}
}};
"""
        )
        table_rows.append(
            f"""  {{
      "{wav_path.name}",
      "{wav_path.as_posix()}",
      {record["source_sample_rate"]},
      {array_name},
      {waveform.size},
      "{expected["audio_crc32"]}",
      "{expected["normalized_feature_crc32"]}",
      "{expected["backbone_output_crc32"]}",
      "{expected["final_logits_crc32"]}",
      {expected["predicted_index"]},
  }}"""
        )

    arrays_text = "".join(arrays)
    table_text = ",\n".join(table_rows)
    source = f"""#include "audio_fixture.h"

namespace audio_fixture {{

{arrays_text}

const Fixture kFixtures[] = {{
{table_text}
}};

const size_t kNumFixtures = sizeof(kFixtures) / sizeof(kFixtures[0]);

}}  // namespace audio_fixture
"""
    (FIRMWARE_MAIN_DIR / "audio_fixture.h").write_text(header, encoding="utf-8")
    (FIRMWARE_MAIN_DIR / "audio_fixture.cpp").write_text(source, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an embedded C waveform fixture from a real validation WAV")
    parser.add_argument("--wav", action="append", default=None)
    parser.add_argument("--backbone-tflite", default=str(DEFAULT_BACKBONE))
    parser.add_argument("--step-tflite", default=str(DEFAULT_STEP))
    parser.add_argument("--expected-json", default=str(DEFAULT_EXPECTED))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_paths = [Path(path).expanduser().resolve() for path in args.wav] if args.wav else [path.resolve() for path in DEFAULT_WAVS]
    records = []
    expected_records = []
    for wav_path in wav_paths:
        old_cwd = Path.cwd()
        try:
            os.chdir(SUBMISSION_DIR)
            waveform, source_sample_rate = prepare_waveform(wav_path)
        finally:
            os.chdir(old_cwd)

        expected = compute_expected(waveform, Path(args.backbone_tflite), Path(args.step_tflite))
        expected.update(
            {
                "wav_path": str(wav_path),
                "source_sample_rate": source_sample_rate,
                "prepared_sample_rate": 24000,
                "prepared_num_samples": int(waveform.size),
            }
        )
        records.append(
            {
                "wav_path": str(wav_path),
                "waveform": waveform,
                "source_sample_rate": source_sample_rate,
                "expected": expected,
            }
        )
        expected_records.append(expected)

    write_fixture_files(records)
    Path(args.expected_json).write_text(json.dumps(expected_records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(expected_records, indent=2))
    print(f"wrote: {FIRMWARE_MAIN_DIR / 'audio_fixture.h'}")
    print(f"wrote: {FIRMWARE_MAIN_DIR / 'audio_fixture.cpp'}")
    print(f"wrote: {Path(args.expected_json).resolve()}")


if __name__ == "__main__":
    main()
