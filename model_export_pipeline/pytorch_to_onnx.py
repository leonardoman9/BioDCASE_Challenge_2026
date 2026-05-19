from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from waveform_exportable import WaveformEndToEndExportableWrapper, WaveformFrontendExportable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPO_ROOT / "biodcase_model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a BioDCASE PyTorch checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        default=str(SCRIPT_DIR / "checkpoints" / "local_export_smoke.ckpt"),
        help="Path to the Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "exports" / "model.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--input-kind",
        choices=("spectrogram", "waveform", "waveform_exportable"),
        default="spectrogram",
        help="Export the classifier from spectrogram input, or try end-to-end waveform export",
    )
    parser.add_argument(
        "--dynamic-frames",
        action="store_true",
        help="Keep the spectrogram frame dimension dynamic. Default is static, which is safer for TFLite Micro.",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Export batch dimension as static 1. Recommended for TFLite/TFLite Micro.",
    )
    parser.add_argument(
        "--spectrogram-rank",
        type=int,
        choices=(3, 4),
        default=3,
        help="Spectrogram input rank for classifier export. 3D avoids NCHW/NHWC squeeze issues in onnx2tf.",
    )
    return parser.parse_args()


def default_frontend_spec_path() -> Path:
    return SCRIPT_DIR / "frontend_specs" / "biodcase_best_06717" / "frontend_spec.json"


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def resolve_cfg(checkpoint: dict[str, Any]) -> dict[str, Any]:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise ValueError("Checkpoint does not contain hyper_parameters")

    if "model" in hyper_parameters and "data" in hyper_parameters:
        return hyper_parameters

    cfg = hyper_parameters.get("cfg")
    if isinstance(cfg, dict):
        return cfg

    raise ValueError("Could not recover model/data config from checkpoint hyper_parameters")


def resolve_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    fc_weight = state_dict.get("model.fc.weight")
    if fc_weight is None:
        raise ValueError("Could not infer num_classes: missing model.fc.weight in checkpoint state_dict")
    return int(fc_weight.shape[0])


def instantiate_model(cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    target = model_cfg["target"]
    params = dict(model_cfg.get("params") or {})
    params["num_classes"] = resolve_num_classes(state_dict)

    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    model = model_cls(**params)

    model_state = {key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"warning: missing keys while loading checkpoint: {missing}")
    if unexpected:
        print(f"warning: unexpected keys while loading checkpoint: {unexpected}")
    model.eval()
    return model


class SpectrogramClassifierWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) == 1:
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + 1e-5
            x = (x - mean) / std
            x = x.squeeze(1)
        elif x.dim() == 3:
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True) + 1e-5
            x = (x - mean) / std
        else:
            raise ValueError(f"Unexpected spectrogram shape: {tuple(x.shape)}")

        x = self.model._align_feature_dimension(x)
        x = self.model.phi(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.model.gru(x)
        x = self.model.projection(x)
        x, _ = self.model.keyword_attention(x)
        x = self.model.fc(x)
        return x


def infer_spectrogram_shape(model: nn.Module, cfg: dict[str, Any]) -> tuple[int, int]:
    sample_rate = int(cfg["data"]["sample_rate"])
    clip_duration = float(cfg["data"]["clip_duration"])
    waveform = torch.zeros(1, 1, int(sample_rate * clip_duration), dtype=torch.float32)
    x = waveform.squeeze(1)

    with torch.no_grad():
        if model.spectrogram_type == "mel":
            spec = model.mel_transform(x)
        elif model.spectrogram_type == "linear_stft":
            spec = model.stft_transform(x)
        elif model.spectrogram_type == "linear_triangular":
            spec = model.stft_transform(x)
            spec = spec.permute(0, 2, 1)
            spec = torch.matmul(spec, model.linear_filterbank.T.to(spec.device))
            spec = spec.permute(0, 2, 1)
            spec = model.amplitude_to_db(spec)
        elif model.spectrogram_type in ("combined_log_linear", "fully_learnable"):
            spec = model.amplitude_to_db(model.combined_log_linear_spec(x) ** 2)
        else:
            raise ValueError(f"Unsupported spectrogram_type for export: {model.spectrogram_type}")

    if spec.dim() != 3:
        raise ValueError(f"Unexpected inferred spectrogram shape: {tuple(spec.shape)}")
    return int(spec.shape[1]), int(spec.shape[2])


def export_onnx(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    opset: int,
    input_kind: str,
    dynamic_frames: bool,
    spectrogram_rank: int,
    static_batch: bool,
) -> None:
    sample_rate = int(cfg["data"]["sample_rate"])
    clip_duration = float(cfg["data"]["clip_duration"])
    num_samples = int(sample_rate * clip_duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_kind == "waveform":
        export_model: nn.Module = model
        dummy = torch.zeros(1, 1, num_samples, dtype=torch.float32)
        input_name = "waveform"
        dynamic_axes = None if static_batch else {"waveform": {0: "batch"}, "logits": {0: "batch"}}
    elif input_kind == "waveform_exportable":
        frontend_spec_path = default_frontend_spec_path()
        if not frontend_spec_path.exists():
            raise FileNotFoundError(
                f"Frontend spec not found: {frontend_spec_path}. Run export_frontend_spec.py first."
            )
        frontend = WaveformFrontendExportable.from_exported_spec(frontend_spec_path)
        export_model = WaveformEndToEndExportableWrapper(model, frontend)
        dummy = torch.zeros(1, 1, num_samples, dtype=torch.float32)
        input_name = "waveform"
        dynamic_axes = None if static_batch else {"waveform": {0: "batch"}, "logits": {0: "batch"}}
    else:
        export_model = SpectrogramClassifierWrapper(model)
        n_features, n_frames = infer_spectrogram_shape(model, cfg)
        if spectrogram_rank == 4:
            dummy = torch.zeros(1, 1, n_features, n_frames, dtype=torch.float32)
        else:
            dummy = torch.zeros(1, n_features, n_frames, dtype=torch.float32)
        input_name = "spectrogram"
        dynamic_axes = None
        if dynamic_frames:
            frame_axis = 3 if spectrogram_rank == 4 else 2
            dynamic_axes = {"spectrogram": {0: "batch", frame_axis: "frames"}, "logits": {0: "batch"}}

    export_model.eval()
    with torch.no_grad():
        _ = export_model(dummy)

    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=[input_name],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    print(f"checkpoint: {checkpoint_path}")
    print(f"exported:   {output_path}")
    print(f"input_kind: {input_kind}")
    print(f"input_shape:{tuple(dummy.shape)}")
    print(f"dynamic:    {bool(dynamic_axes)}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    checkpoint = load_checkpoint(checkpoint_path)
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"])
    export_onnx(
        model,
        cfg,
        checkpoint_path,
        output_path,
        args.opset,
        args.input_kind,
        args.dynamic_frames,
        args.spectrogram_rank,
        args.static_batch,
    )


if __name__ == "__main__":
    main()
