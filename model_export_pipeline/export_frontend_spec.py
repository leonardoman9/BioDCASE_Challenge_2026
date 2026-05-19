from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pytorch_to_onnx import instantiate_model, load_checkpoint, resolve_cfg


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "biodcase_best_06717.ckpt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "frontend_specs" / "biodcase_best_06717"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained waveform frontend specification for deployment")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Lightning checkpoint path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for JSON and .npy files")
    return parser.parse_args()


def tensor_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def infer_spectrogram_shape(model: torch.nn.Module, sample_rate: int, clip_duration: float) -> tuple[int, int]:
    waveform = torch.zeros(1, 1, int(sample_rate * clip_duration), dtype=torch.float32)
    with torch.no_grad():
        spec = model(waveform)
    raise RuntimeError("This helper should not be called directly on the full model")


def compute_frontend_spec(model: torch.nn.Module, cfg: dict[str, Any], checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    if getattr(model, "spectrogram_type", None) != "combined_log_linear":
        raise ValueError(f"Expected combined_log_linear frontend, got {getattr(model, 'spectrogram_type', None)!r}")

    spec_module = getattr(model, "combined_log_linear_spec", None)
    if spec_module is None:
        raise ValueError("Model does not expose combined_log_linear_spec")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]["params"]
    sample_rate = int(data_cfg["sample_rate"])
    clip_duration = float(data_cfg["clip_duration"])
    num_samples = int(sample_rate * clip_duration)

    with torch.no_grad():
        filter_bank = spec_module._get_current_filter_bank(device=torch.device("cpu")).detach().cpu()
        center_freqs_hz = spec_module._get_triangular_filter_centers(device=torch.device("cpu")).detach().cpu()
        window = torch.hann_window(int(spec_module.n_fft), device=torch.device("cpu")).detach().cpu()
        dummy_waveform = torch.zeros(1, num_samples, dtype=torch.float32)
        spec = spec_module(dummy_waveform).detach().cpu()
        spec_db = model.amplitude_to_db(spec**2).detach().cpu()

    output_dir.mkdir(parents=True, exist_ok=True)
    filter_bank_path = output_dir / "filter_bank.npy"
    center_freqs_path = output_dir / "filter_centers_hz.npy"
    window_path = output_dir / "hann_window.npy"
    spectrogram_example_path = output_dir / "zero_waveform_frontend_output.npy"
    spectrogram_db_example_path = output_dir / "zero_waveform_frontend_output_db.npy"
    json_path = output_dir / "frontend_spec.json"

    np.save(filter_bank_path, filter_bank.numpy())
    np.save(center_freqs_path, center_freqs_hz.numpy())
    np.save(window_path, window.numpy())
    np.save(spectrogram_example_path, spec.numpy())
    np.save(spectrogram_db_example_path, spec_db.numpy())

    amplitude_to_db = getattr(model, "amplitude_to_db", None)
    top_db = getattr(amplitude_to_db, "top_db", None)

    raw_breakpoint = getattr(spec_module, "breakpoint", None)
    raw_transition = getattr(spec_module, "transition_width", None)

    result = {
        "checkpoint": checkpoint_path,
        "model_target": cfg["model"]["target"],
        "spectrogram_type": model.spectrogram_type,
        "sample_rate": sample_rate,
        "clip_duration_seconds": clip_duration,
        "waveform_num_samples": num_samples,
        "n_fft": int(spec_module.n_fft),
        "hop_length": int(spec_module.hop_length),
        "n_freq_bins_stft": int(spec_module.n_fft // 2 + 1),
        "n_filters": int(spec_module.n_filters),
        "expected_input_features": int(getattr(model, "expected_input_features", spec_module.n_filters)),
        "f_min_hz": tensor_to_float(spec_module.f_min),
        "f_max_hz": tensor_to_float(spec_module.f_max),
        "trainable_filterbank": bool(getattr(spec_module, "trainable_filterbank", False)),
        "raw_breakpoint_parameter": tensor_to_float(raw_breakpoint) if raw_breakpoint is not None else None,
        "raw_transition_width_parameter": tensor_to_float(raw_transition) if raw_transition is not None else None,
        "effective_breakpoint_hz": tensor_to_float(spec_module.effective_breakpoint()),
        "effective_transition_width": tensor_to_float(spec_module.effective_transition_width()),
        "amplitude_to_db": {
            "stype": "power",
            "top_db": float(top_db) if top_db is not None else None,
            "power_input": True,
            "applied_outside_spec_module": True,
        },
        "window": {
            "type": "hann",
            "size": int(spec_module.n_fft),
            "path": window_path.name,
        },
        "filter_bank": {
            "shape": list(filter_bank.shape),
            "path": filter_bank_path.name,
        },
        "filter_centers_hz": {
            "shape": list(center_freqs_hz.shape),
            "path": center_freqs_path.name,
        },
        "reference_frontend_output": {
            "shape": list(spec.shape),
            "path": spectrogram_example_path.name,
            "description": "Raw filtered magnitude output before AmplitudeToDB, using a zero waveform",
        },
        "reference_frontend_output_db": {
            "shape": list(spec_db.shape),
            "path": spectrogram_db_example_path.name,
            "description": "Frontend output after AmplitudeToDB(spec**2), before per-sample normalization",
        },
        "spectrogram_after_db_shape": [int(spec.shape[1]), int(spec.shape[2])],
        "original_model_params": {
            "sample_rate": model_cfg["sample_rate"],
            "hidden_dim": model_cfg["hidden_dim"],
            "n_mel_bins": model_cfg["n_mel_bins"],
            "n_linear_filters": model_cfg["n_linear_filters"],
            "f_min": model_cfg["f_min"],
            "f_max": model_cfg["f_max"],
            "n_fft": model_cfg["n_fft"],
            "hop_length": model_cfg["hop_length"],
            "breakpoint_config_hz": model_cfg["breakpoint"],
            "transition_width_config": model_cfg["transition_width"],
            "matchbox": model_cfg["matchbox"],
        },
    }

    json_path.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    checkpoint = load_checkpoint(checkpoint_path)
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"]).cpu().eval()

    result = compute_frontend_spec(model, cfg, checkpoint_path, output_dir)
    print(json.dumps(result, indent=2, default=json_default))
    print(f"\nsaved: {output_dir / 'frontend_spec.json'}")


if __name__ == "__main__":
    main()
