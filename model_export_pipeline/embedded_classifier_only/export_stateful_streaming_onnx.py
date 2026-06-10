from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent / "biodcase_model"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedded_classifier_only.models import prepare_frontend_features
from pytorch_to_onnx import instantiate_model, load_checkpoint, resolve_cfg


DEFAULT_CHECKPOINT = PIPELINE_ROOT / "checkpoints" / "biodcase_best_06717.ckpt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "stateful_streaming"


class BackboneStreamingWrapper(nn.Module):
    """
    Frontend features -> per-frame CNN embeddings.

    Input:  [1, features, frames]
    Output: [1, frames, channels]
    """

    def __init__(self, model: nn.Module, normalize_input: bool) -> None:
        super().__init__()
        self.model = model
        self.normalize_input = normalize_input

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = prepare_frontend_features(features, normalize=self.normalize_input)
        x = self.model._align_feature_dimension(x)
        x = self.model.phi(x)
        return x.permute(0, 2, 1).contiguous()


class StreamingStepWrapper(nn.Module):
    """
    One recurrent step with explicit state.

    Inputs:
      frame:    [1, cnn_channels]
      hidden:   [1, hidden_dim]
      att_num:  [1, hidden_dim]
      att_den:  [1, 1]
      att_max:  [1, 1]

    Outputs:
      new_hidden, new_att_num, new_att_den, new_att_max, logits
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        frame: torch.Tensor,
        hidden: torch.Tensor,
        att_num: torch.Tensor,
        att_den: torch.Tensor,
        att_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gru = self.model.gru
        weight_ih = gru.weight_ih_l0.float()
        weight_hh = gru.weight_hh_l0.float()
        bias_ih = gru.bias_ih_l0.float()
        bias_hh = gru.bias_hh_l0.float()

        w_ir, w_iz, w_in = weight_ih.chunk(3, dim=0)
        w_hr, w_hz, w_hn = weight_hh.chunk(3, dim=0)
        b_ir, b_iz, b_in = bias_ih.chunk(3, dim=0)
        b_hr, b_hz, b_hn = bias_hh.chunk(3, dim=0)

        r_t = torch.sigmoid(F.linear(frame, w_ir, b_ir) + F.linear(hidden, w_hr, b_hr))
        z_t = torch.sigmoid(F.linear(frame, w_iz, b_iz) + F.linear(hidden, w_hz, b_hz))
        n_t = torch.tanh(F.linear(frame, w_in, b_in) + r_t * F.linear(hidden, w_hn, b_hn))
        new_hidden = (1.0 - z_t) * n_t + z_t * hidden

        projected = self.model.projection(new_hidden)
        score = self.model.keyword_attention.attention(projected)
        new_att_max = torch.maximum(att_max, score)
        old_scale = torch.exp(att_max - new_att_max)
        new_scale = torch.exp(score - new_att_max)
        new_att_num = att_num * old_scale + projected * new_scale
        new_att_den = att_den * old_scale + new_scale
        context = new_att_num / new_att_den
        logits = self.model.fc(context)
        return new_hidden, new_att_num, new_att_den, new_att_max, logits


def infer_frontend_shape(cfg: dict) -> tuple[int, int]:
    sample_rate = int(cfg["data"]["sample_rate"])
    clip_duration = float(cfg["data"]["clip_duration"])
    waveform_len = int(sample_rate * clip_duration)
    # For the exported frozen frontend, this model uses 64 bins and 301 frames.
    # Keep this explicit to avoid importing the frontend in this lightweight exporter.
    if sample_rate == 24000 and waveform_len == 72000:
        return 64, 301
    raise ValueError(f"Unsupported feature shape inference for sample_rate={sample_rate}, waveform_len={waveform_len}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stateful streaming backbone and recurrent step to ONNX")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply sample-wise normalization inside backbone. Omit for prenormalized embedded frontend features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(Path(args.checkpoint))
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"]).cpu().eval()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features, n_frames = infer_frontend_shape(cfg)
    backbone = BackboneStreamingWrapper(model, normalize_input=args.normalize).cpu().eval()
    features = torch.zeros(1, n_features, n_frames, dtype=torch.float32)
    with torch.no_grad():
        embeddings = backbone(features)

    _, _, cnn_channels = embeddings.shape
    hidden_dim = int(model.hidden_dim)
    step = StreamingStepWrapper(model).cpu().eval()
    frame = torch.zeros(1, cnn_channels, dtype=torch.float32)
    hidden = torch.zeros(1, hidden_dim, dtype=torch.float32)
    att_num = torch.zeros(1, hidden_dim, dtype=torch.float32)
    att_den = torch.zeros(1, 1, dtype=torch.float32)
    att_max = torch.full((1, 1), -1.0e9, dtype=torch.float32)

    backbone_path = output_dir / "biodcase_backbone_streaming.onnx"
    step_path = output_dir / "biodcase_streaming_step.onnx"

    torch.onnx.export(
        backbone,
        features,
        str(backbone_path),
        input_names=["frontend_features"],
        output_names=["frame_embeddings"],
        opset_version=args.opset,
        dynamo=False,
    )
    torch.onnx.export(
        step,
        (frame, hidden, att_num, att_den, att_max),
        str(step_path),
        input_names=["frame", "hidden", "att_num", "att_den", "att_max"],
        output_names=["new_hidden", "new_att_num", "new_att_den", "new_att_max", "logits"],
        opset_version=args.opset,
        dynamo=False,
    )

    print(f"backbone:       {backbone_path}")
    print(f"streaming_step: {step_path}")
    print(f"features:       {tuple(features.shape)}")
    print(f"embeddings:     {tuple(embeddings.shape)}")
    print(f"hidden_dim:     {hidden_dim}")
    print(f"normalize:      {args.normalize}")


if __name__ == "__main__":
    main()
