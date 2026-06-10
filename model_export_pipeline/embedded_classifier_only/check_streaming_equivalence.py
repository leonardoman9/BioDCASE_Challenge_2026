from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
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
DEFAULT_FEATURES = SCRIPT_DIR / "representative_data_prenorm_full" / "validation_frontend_features.npy"


def gru_step(gru: torch.nn.GRU, x_t: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    weight_ih = gru.weight_ih_l0.float()
    weight_hh = gru.weight_hh_l0.float()
    bias_ih = gru.bias_ih_l0.float()
    bias_hh = gru.bias_hh_l0.float()

    w_ir, w_iz, w_in = weight_ih.chunk(3, dim=0)
    w_hr, w_hz, w_hn = weight_hh.chunk(3, dim=0)
    b_ir, b_iz, b_in = bias_ih.chunk(3, dim=0)
    b_hr, b_hz, b_hn = bias_hh.chunk(3, dim=0)

    r_t = torch.sigmoid(F.linear(x_t, w_ir, b_ir) + F.linear(hidden, w_hr, b_hr))
    z_t = torch.sigmoid(F.linear(x_t, w_iz, b_iz) + F.linear(hidden, w_hz, b_hz))
    n_t = torch.tanh(F.linear(x_t, w_in, b_in) + r_t * F.linear(hidden, w_hn, b_hn))
    return (1.0 - z_t) * n_t + z_t * hidden


def classifier_logits_full(model: torch.nn.Module, features: torch.Tensor, normalize: bool) -> torch.Tensor:
    x = prepare_frontend_features(features, normalize=normalize)
    x = model._align_feature_dimension(x)
    x = model.phi(x)
    x = x.permute(0, 2, 1).contiguous()
    x, _ = model.gru(x)
    x = model.projection(x)
    x, _ = model.keyword_attention(x)
    return model.fc(x)


def classifier_logits_streaming(model: torch.nn.Module, features: torch.Tensor, normalize: bool) -> torch.Tensor:
    x = prepare_frontend_features(features, normalize=normalize)
    x = model._align_feature_dimension(x)
    x = model.phi(x)
    x = x.permute(0, 2, 1).contiguous()

    batch_size, time_steps, _ = x.shape
    hidden = torch.zeros(batch_size, model.hidden_dim, dtype=x.dtype, device=x.device)

    # Exact streaming softmax accumulation for attention over projected GRU states.
    max_score = torch.full((batch_size, 1), -float("inf"), dtype=x.dtype, device=x.device)
    numerator = torch.zeros(batch_size, model.hidden_dim, dtype=x.dtype, device=x.device)
    denominator = torch.zeros(batch_size, 1, dtype=x.dtype, device=x.device)

    for step in range(time_steps):
        hidden = gru_step(model.gru, x[:, step, :], hidden)
        projected = model.projection(hidden)
        score = model.keyword_attention.attention(projected)

        new_max = torch.maximum(max_score, score)
        old_scale = torch.exp(max_score - new_max)
        new_scale = torch.exp(score - new_max)
        numerator = numerator * old_scale + projected * new_scale
        denominator = denominator * old_scale + new_scale
        max_score = new_max

    context = numerator / denominator
    return model.fc(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check full-sequence classifier vs exact streaming GRU+attention")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply sample-wise normalization inside the check. Omit for prenormalized representative features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(Path(args.checkpoint))
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"]).cpu().eval()

    features_np = np.load(args.features)
    features = torch.from_numpy(features_np[: args.num_samples]).float()
    if features.dim() == 4 and features.size(-1) == 1:
        features = features.squeeze(-1)
    if features.shape[1] != model.expected_input_features and features.shape[2] == model.expected_input_features:
        features = features.permute(0, 2, 1).contiguous()

    with torch.no_grad():
        full = classifier_logits_full(model, features, normalize=args.normalize)
        streaming = classifier_logits_streaming(model, features, normalize=args.normalize)

    diff = (full - streaming).abs()
    print(f"samples:      {features.shape[0]}")
    print(f"feature_shape:{tuple(features.shape)}")
    print(f"normalize:    {args.normalize}")
    print(f"max_abs_diff: {diff.max().item():.8g}")
    print(f"mean_abs_diff:{diff.mean().item():.8g}")
    print(f"full_argmax:  {full.argmax(dim=1).tolist()}")
    print(f"stream_argmax:{streaming.argmax(dim=1).tolist()}")
    print(f"agreement:    {int((full.argmax(dim=1) == streaming.argmax(dim=1)).sum())}/{features.shape[0]}")


if __name__ == "__main__":
    main()
