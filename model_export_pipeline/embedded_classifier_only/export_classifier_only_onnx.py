from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent / "biodcase_model"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedded_classifier_only.models import (
    ClassifierOnlyPreNormalizedWrapper,
    ClassifierOnlyUnrolledGRUWrapper,
    ClassifierOnlyUnrolledGRUPreNormalizedWrapper,
    ClassifierOnlyWrapper,
    FrozenFrontendFeatureExtractor,
    build_frozen_frontend,
)
from pytorch_to_onnx import instantiate_model, load_checkpoint, resolve_cfg


DEFAULT_CHECKPOINT = PIPELINE_ROOT / "checkpoints" / "biodcase_best_06717.ckpt"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "biodcase_best_06717_classifier_only.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the post-frontend classifier branch to ONNX")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Best checkpoint")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--unrolled-gru", action="store_true", help="Export the primitive-op GRU variant")
    parser.add_argument(
        "--external-normalization",
        action="store_true",
        help="Assume sample-wise feature normalization happens outside the graph",
    )
    return parser.parse_args()


def infer_feature_shape(sample_rate: int, clip_duration: float) -> tuple[int, int]:
    frontend = FrozenFrontendFeatureExtractor(build_frozen_frontend()).cpu().eval()
    waveform = torch.zeros(1, 1, int(sample_rate * clip_duration), dtype=torch.float32)
    with torch.no_grad():
        features = frontend(waveform)
    if features.dim() != 3:
        raise ValueError(f"Unexpected frontend feature shape: {tuple(features.shape)}")
    return int(features.shape[1]), int(features.shape[2])


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"]).cpu().eval()
    if args.unrolled_gru and args.external_normalization:
        export_model = ClassifierOnlyUnrolledGRUPreNormalizedWrapper(model)
    elif args.unrolled_gru:
        export_model = ClassifierOnlyUnrolledGRUWrapper(model)
    elif args.external_normalization:
        export_model = ClassifierOnlyPreNormalizedWrapper(model)
    else:
        export_model = ClassifierOnlyWrapper(model)
    export_model = export_model.cpu().eval()

    sample_rate = int(cfg["data"]["sample_rate"])
    clip_duration = float(cfg["data"]["clip_duration"])
    n_features, n_frames = infer_feature_shape(sample_rate, clip_duration)
    dummy = torch.zeros(1, n_features, n_frames, dtype=torch.float32)

    with torch.no_grad():
        _ = export_model(dummy)

    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["frontend_features"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=False,
    )

    print(f"checkpoint:   {checkpoint_path}")
    print(f"exported:     {output_path}")
    print(f"input_shape:  {tuple(dummy.shape)}")
    print(f"unrolled_gru: {args.unrolled_gru}")
    print(f"external_normalization: {args.external_normalization}")


if __name__ == "__main__":
    main()
