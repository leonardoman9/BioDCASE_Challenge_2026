from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenAmplitudeToDB(nn.Module):
    def __init__(self, stype: str = "power", top_db: float | None = 80.0, amin: float = 1e-10, reference: float = 1.0) -> None:
        super().__init__()
        if stype not in {"power", "magnitude"}:
            raise ValueError(f"Unsupported stype: {stype!r}")
        multiplier = 10.0 if stype == "power" else 20.0
        db_multiplier = math.log10(max(reference, amin))
        self.stype = stype
        self.top_db = top_db
        self.amin = amin
        self.multiplier = multiplier
        self.db_multiplier = db_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db = x_db - self.multiplier * self.db_multiplier

        if self.top_db is not None:
            shape = x_db.size()
            packed_channels = shape[-3] if x_db.dim() > 2 else 1
            x_db = x_db.reshape(-1, packed_channels, shape[-2], shape[-1])
            cutoff = (x_db.amax(dim=(-3, -2, -1)) - self.top_db).view(-1, 1, 1, 1)
            x_db = torch.maximum(x_db, cutoff)
            x_db = x_db.reshape(shape)
        return x_db


class WaveformFrontendExportable(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        filter_bank: torch.Tensor,
        window: torch.Tensor,
        amplitude_to_db: FrozenAmplitudeToDB,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.pad = self.n_fft // 2
        self.amplitude_to_db = amplitude_to_db

        filter_bank = filter_bank.detach().clone().float()
        window = window.detach().clone().float()
        if filter_bank.dim() != 2:
            raise ValueError(f"filter_bank must be 2D, got shape {tuple(filter_bank.shape)}")
        if window.dim() != 1 or window.numel() != self.n_fft:
            raise ValueError(f"window must be 1D of length {self.n_fft}, got shape {tuple(window.shape)}")

        self.register_buffer("filter_bank", filter_bank)
        self.register_buffer("window", window)
        self.register_buffer("real_kernel", self._build_stft_kernel(sign=-1.0, use_cos=True))
        self.register_buffer("imag_kernel", self._build_stft_kernel(sign=-1.0, use_cos=False))

    def _build_stft_kernel(self, sign: float, use_cos: bool) -> torch.Tensor:
        freq_bins = self.n_fft // 2 + 1
        n = torch.arange(self.n_fft, dtype=torch.float32)
        k = torch.arange(freq_bins, dtype=torch.float32).unsqueeze(1)
        phase = 2.0 * math.pi * k * n.unsqueeze(0) / float(self.n_fft)
        base = torch.cos(phase) if use_cos else sign * torch.sin(phase)
        kernel = base * self.window.unsqueeze(0)
        return kernel.unsqueeze(1).contiguous()

    @classmethod
    def from_exported_spec(cls, spec_json_path: Path | str) -> "WaveformFrontendExportable":
        spec_path = Path(spec_json_path).expanduser().resolve()
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        root = spec_path.parent

        filter_bank = torch.from_numpy(np.load(root / spec["filter_bank"]["path"])).float()
        window = torch.from_numpy(np.load(root / spec["window"]["path"])).float()

        amp_cfg = spec["amplitude_to_db"]
        amplitude_to_db = FrozenAmplitudeToDB(
            stype=str(amp_cfg["stype"]),
            top_db=float(amp_cfg["top_db"]) if amp_cfg["top_db"] is not None else None,
            amin=1e-10,
            reference=1.0,
        )
        return cls(
            sample_rate=int(spec["sample_rate"]),
            n_fft=int(spec["n_fft"]),
            hop_length=int(spec["hop_length"]),
            filter_bank=filter_bank,
            window=window,
            amplitude_to_db=amplitude_to_db,
        )

    def stft_magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform shape [batch, time] or [batch, 1, time], got {tuple(waveform.shape)}")

        x = waveform.unsqueeze(1)
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        real = F.conv1d(x, self.real_kernel, stride=self.hop_length)
        imag = F.conv1d(x, self.imag_kernel, stride=self.hop_length)
        magnitude = torch.sqrt(real * real + imag * imag)
        return magnitude

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        magnitude = self.stft_magnitude(waveform)
        filtered = torch.matmul(self.filter_bank, magnitude)
        spectrogram_db = self.amplitude_to_db(filtered * filtered)
        return spectrogram_db


class WaveformEndToEndExportableWrapper(nn.Module):
    def __init__(self, classifier_model: nn.Module, frontend: WaveformFrontendExportable) -> None:
        super().__init__()
        self.classifier_model = classifier_model
        self.frontend = frontend

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        elif waveform.dim() != 3:
            raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")

        x = waveform.squeeze(1)
        x = self.frontend(x)

        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-5
        x = (x - mean) / std

        x = self.classifier_model._align_feature_dimension(x)
        x = self.classifier_model.phi(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.classifier_model.gru(x)
        x = self.classifier_model.projection(x)
        x, _ = self.classifier_model.keyword_attention(x)
        x = self.classifier_model.fc(x)
        return x
