from __future__ import annotations

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

from waveform_exportable import WaveformFrontendExportable


class FrozenFrontendFeatureExtractor(nn.Module):
    """
    Waveform -> frozen frontend features.

    Output contract:
      [batch, 64, 301] float32 dB spectrogram
    """

    def __init__(self, frontend: WaveformFrontendExportable) -> None:
        super().__init__()
        self.frontend = frontend

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        elif waveform.dim() != 3:
            raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")

        x = waveform.squeeze(1)
        return self.frontend(x)


def prepare_frontend_features(features: torch.Tensor, normalize: bool) -> torch.Tensor:
    if features.dim() == 4 and features.size(1) == 1:
        x = features.squeeze(1)
    elif features.dim() == 3:
        x = features
    else:
        raise ValueError(f"Unexpected frontend feature shape: {tuple(features.shape)}")

    if normalize:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-5
        x = (x - mean) / std
    return x


def run_classifier_from_features(
    classifier_model: nn.Module,
    gru_module: nn.Module,
    features: torch.Tensor,
    *,
    normalize_input: bool,
) -> torch.Tensor:
    x = prepare_frontend_features(features, normalize=normalize_input)
    x = classifier_model._align_feature_dimension(x)
    x = classifier_model.phi(x)
    x = x.permute(0, 2, 1).contiguous()
    x, _ = gru_module(x)
    x = classifier_model.projection(x)
    x, _ = classifier_model.keyword_attention(x)
    x = classifier_model.fc(x)
    return x


class ClassifierOnlyWrapper(nn.Module):
    """
    Frontend-features -> logits.

    The frozen waveform frontend is outside this graph. Sample-wise
    normalization and feature alignment stay inside to preserve parity.
    """

    def __init__(self, classifier_model: nn.Module) -> None:
        super().__init__()
        self.classifier_model = classifier_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return run_classifier_from_features(
            self.classifier_model,
            self.classifier_model.gru,
            features,
            normalize_input=True,
        )


class ClassifierOnlyPreNormalizedWrapper(nn.Module):
    """
    Frontend-features -> logits, assuming sample-wise normalization already
    happened outside the graph.
    """

    def __init__(self, classifier_model: nn.Module) -> None:
        super().__init__()
        self.classifier_model = classifier_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return run_classifier_from_features(
            self.classifier_model,
            self.classifier_model.gru,
            features,
            normalize_input=False,
        )


class PrimitiveGRUInference(nn.Module):
    """
    Inference-only GRU rewritten with primitive ops.

    This avoids exporting the fused GRU op and exposes only matmul/add/sigmoid/
    tanh/mul primitives to downstream converters.
    """

    def __init__(self, gru: nn.GRU) -> None:
        super().__init__()
        if gru.num_layers != 1 or gru.bidirectional or not gru.bias or not gru.batch_first:
            raise ValueError("PrimitiveGRUInference currently supports only 1-layer, batch_first, biased, unidirectional GRU")

        self.input_size = int(gru.input_size)
        self.hidden_size = int(gru.hidden_size)

        self.register_buffer("weight_ih", gru.weight_ih_l0.detach().clone().float())
        self.register_buffer("weight_hh", gru.weight_hh_l0.detach().clone().float())
        self.register_buffer("bias_ih", gru.bias_ih_l0.detach().clone().float())
        self.register_buffer("bias_hh", gru.bias_hh_l0.detach().clone().float())

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(x.shape)}")
        batch_size, time_steps, _ = x.shape
        if h0 is None:
            hidden = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            if h0.dim() == 3:
                hidden = h0[0]
            elif h0.dim() == 2:
                hidden = h0
            else:
                raise ValueError(f"Unexpected h0 shape: {tuple(h0.shape)}")

        w_ir, w_iz, w_in = self.weight_ih.chunk(3, dim=0)
        w_hr, w_hz, w_hn = self.weight_hh.chunk(3, dim=0)
        b_ir, b_iz, b_in = self.bias_ih.chunk(3, dim=0)
        b_hr, b_hz, b_hn = self.bias_hh.chunk(3, dim=0)

        outputs = []
        for step in range(time_steps):
            x_t = x[:, step, :]
            r_t = torch.sigmoid(F.linear(x_t, w_ir, b_ir) + F.linear(hidden, w_hr, b_hr))
            z_t = torch.sigmoid(F.linear(x_t, w_iz, b_iz) + F.linear(hidden, w_hz, b_hz))
            n_t = torch.tanh(F.linear(x_t, w_in, b_in) + r_t * F.linear(hidden, w_hn, b_hn))
            hidden = (1.0 - z_t) * n_t + z_t * hidden
            outputs.append(hidden)

        output = torch.stack(outputs, dim=1)
        return output, hidden.unsqueeze(0)


class ClassifierOnlyUnrolledGRUWrapper(nn.Module):
    """
    Same post-frontend classifier path, but with the recurrent block rewritten
    as primitive ops for converter-friendly quantization attempts.
    """

    def __init__(self, classifier_model: nn.Module) -> None:
        super().__init__()
        self.classifier_model = classifier_model
        self.primitive_gru = PrimitiveGRUInference(classifier_model.gru)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return run_classifier_from_features(
            self.classifier_model,
            self.primitive_gru,
            features,
            normalize_input=True,
        )


class ClassifierOnlyUnrolledGRUPreNormalizedWrapper(nn.Module):
    """
    Primitive-op GRU variant that expects already normalized frontend
    features, so the quantized graph does not contain a sample-wise DIV.
    """

    def __init__(self, classifier_model: nn.Module) -> None:
        super().__init__()
        self.classifier_model = classifier_model
        self.primitive_gru = PrimitiveGRUInference(classifier_model.gru)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return run_classifier_from_features(
            self.classifier_model,
            self.primitive_gru,
            features,
            normalize_input=False,
        )


def default_frontend_spec_path() -> Path:
    return PIPELINE_ROOT / "frontend_specs" / "biodcase_best_06717" / "frontend_spec.json"


def build_frozen_frontend(spec_path: Path | None = None) -> WaveformFrontendExportable:
    return WaveformFrontendExportable.from_exported_spec(spec_path or default_frontend_spec_path())
