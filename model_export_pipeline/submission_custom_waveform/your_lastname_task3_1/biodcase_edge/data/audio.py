from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from scipy.io import wavfile


def load_waveform(
    path: str | Path,
    sample_rate: int,
    clip_duration: float,
    normalize: bool = True,
) -> torch.Tensor:
    source_sr, audio = wavfile.read(str(path))
    waveform = _numpy_audio_to_tensor(audio, path)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if source_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)

    target_len = int(round(sample_rate * clip_duration))
    current_len = waveform.size(1)
    if current_len > target_len:
        waveform = waveform[:, :target_len]
    elif current_len < target_len:
        waveform = F.pad(waveform, (0, target_len - current_len))

    if normalize:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak.clamp_min(1e-8)

    return waveform.float()


def _numpy_audio_to_tensor(audio: np.ndarray, path: str | Path) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim == 2:
        audio = audio.T
    else:
        raise ValueError(f"Expected mono/stereo audio, got shape {audio.shape} for {path}")

    if np.issubdtype(audio.dtype, np.integer):
        max_abs = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_abs
    else:
        audio = audio.astype(np.float32)

    return torch.from_numpy(audio)


def augment_waveform(
    waveform: torch.Tensor,
    noise_std: float = 0.0,
    gain_min: float = 1.0,
    gain_max: float = 1.0,
    max_shift_samples: int = 0,
) -> torch.Tensor:
    if gain_min != 1.0 or gain_max != 1.0:
        gain = torch.empty(1).uniform_(gain_min, gain_max).item()
        waveform = waveform * gain

    if max_shift_samples > 0:
        shift = int(torch.randint(-max_shift_samples, max_shift_samples + 1, (1,)).item())
        if shift != 0:
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

    if noise_std > 0:
        waveform = waveform + torch.randn_like(waveform) * noise_std

    return waveform.clamp(-1.0, 1.0)
