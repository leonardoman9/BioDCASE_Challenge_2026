from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
PROJECT_ROOT = REPO_ROOT / "biodcase_model"
SUBMISSION_DIR = PIPELINE_ROOT / "submission_custom_waveform" / "your_lastname_task3_1"
DEFAULT_DATASET = PROJECT_ROOT / "BioDCASE2026_TinyML_Development_Dataset" / "Validation"
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


def resolve_label_key(label_dict: dict[str, int], dataset_label: str) -> str:
    if dataset_label in label_dict:
        return dataset_label
    normalized = dataset_label.replace(" ", "_")
    if normalized in label_dict:
        return normalized
    raise KeyError(dataset_label)


def normalize_frontend_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=(1, 2), keepdims=True)
    std = features.std(axis=(1, 2), ddof=1, keepdims=True) + 1e-5
    return ((features - mean) / std).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch, host waveform TFLite, and embedded split TFLite on real WAVs")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--backbone-tflite", default=str(DEFAULT_BACKBONE))
    parser.add_argument("--step-tflite", default=str(DEFAULT_STEP))
    parser.add_argument("--limit", type=int, default=0, help="0 means all WAVs")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    wavs = sorted(dataset_dir.glob("*/*.wav"))
    if args.limit > 0:
        wavs = wavs[: args.limit]

    inference_module = load_module("submission_inference_handler_compare", SUBMISSION_DIR / "inference_handler.py")
    cfg = yaml.safe_load((SUBMISSION_DIR / "config_submission.yaml").read_text())

    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))

    old_cwd = Path.cwd()
    try:
        os.chdir(SUBMISSION_DIR)
        inference_handler = inference_module.InferenceHandler(cfg["inference_handler"])
    finally:
        os.chdir(old_cwd)

    frontend = build_frozen_frontend().cpu().eval()
    backbone = get_tflite_interpreter(Path(args.backbone_tflite))
    step = get_tflite_interpreter(Path(args.step_tflite))
    label_dict = inference_handler.get_label_dict()

    targets = []
    pytorch_logits = []
    host_tflite_logits = []
    split_tflite_logits = []

    with torch.no_grad():
        for index, wav_path in enumerate(wavs, start=1):
            label_name = resolve_label_key(label_dict, wav_path.parent.name)
            targets.append(label_dict[label_name])

            waveform_raw, sample_rate = sf.read(wav_path)
            waveform_features = inference_handler.feature_handler.extract(waveform_raw, fs=sample_rate)
            torch_logits, host_logits = inference_handler.infer(waveform_raw, sample_rate)
            pytorch_logits.append(torch_logits[0].astype(np.float32))
            host_tflite_logits.append(host_logits[0].astype(np.float32))

            waveform = waveform_features.reshape(1, -1).astype(np.float32)
            frontend_features = frontend(torch.from_numpy(waveform)).numpy().astype(np.float32)
            normalized = normalize_frontend_features(frontend_features)
            embeddings = run_backbone_tflite(backbone, normalized)
            split_logits = run_step_tflite_loop(step, embeddings)
            split_tflite_logits.append(split_logits[0].astype(np.float32))

            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(wavs)):
                print(f"processed {index}/{len(wavs)}: {wav_path.name}")

    targets_np = np.asarray(targets)
    pytorch_np = np.stack(pytorch_logits)
    host_np = np.stack(host_tflite_logits)
    split_np = np.stack(split_tflite_logits)

    pytorch_argmax = pytorch_np.argmax(axis=1)
    host_argmax = host_np.argmax(axis=1)
    split_argmax = split_np.argmax(axis=1)

    print("")
    print(f"samples:                         {len(wavs)}")
    print(f"accuracy_pytorch:                {np.mean(targets_np == pytorch_argmax):.4f}")
    print(f"accuracy_host_waveform_tflite:   {np.mean(targets_np == host_argmax):.4f}")
    print(f"accuracy_embedded_split_tflite:  {np.mean(targets_np == split_argmax):.4f}")
    print(f"agreement_pytorch_vs_host:       {np.mean(pytorch_argmax == host_argmax):.4f}")
    print(f"agreement_pytorch_vs_split:      {np.mean(pytorch_argmax == split_argmax):.4f}")
    print(f"agreement_host_vs_split:         {np.mean(host_argmax == split_argmax):.4f}")
    print(f"max_abs_diff_pytorch_vs_host:    {np.max(np.abs(pytorch_np - host_np)):.8g}")
    print(f"max_abs_diff_pytorch_vs_split:   {np.max(np.abs(pytorch_np - split_np)):.8g}")
    print(f"mean_abs_diff_pytorch_vs_split:  {np.mean(np.abs(pytorch_np - split_np)):.8g}")

    mismatch_indices = np.flatnonzero(pytorch_argmax != split_argmax)
    print(f"pytorch_vs_split_mismatches:     {len(mismatch_indices)}")
    for mismatch_index in mismatch_indices[:20]:
        print(
            "mismatch",
            mismatch_index,
            wavs[mismatch_index].name,
            "target",
            int(targets_np[mismatch_index]),
            "pytorch",
            int(pytorch_argmax[mismatch_index]),
            "split",
            int(split_argmax[mismatch_index]),
        )


if __name__ == "__main__":
    main()
