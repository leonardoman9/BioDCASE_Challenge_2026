from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly


@dataclass
class WaveformSpec:
    target_sample_rate: int = 24000
    target_wav_length_sec: int = 3
    normalize_peak: bool = True
    add_batch_dimension: bool = True
    channel_last: bool = True

    @property
    def target_num_samples(self) -> int:
        return int(self.target_sample_rate * self.target_wav_length_sec)


class FeatureHandler:
    """
    Custom waveform handler for submission.

    This replaces the baseline mel feature extraction path with the waveform
    contract expected by the frozen-frontend model:
      [batch, time, channels] = [1, 72000, 1]
    """

    def __init__(self, cfg: dict | None = None, **kwargs) -> None:
        merged = {}
        if cfg:
            merged.update(cfg)
        merged.update(kwargs)
        self.spec = WaveformSpec(**merged)

    def extract(self, waveform, fs: int | None = None) -> np.ndarray:
        x = np.asarray(waveform)
        if x.ndim == 2:
            # soundfile.read returns [samples, channels]
            if x.shape[1] <= 8:
                x = x.mean(axis=1)
            else:
                x = x.mean(axis=0)
        elif x.ndim != 1:
            raise ValueError(f"Expected mono/stereo waveform, got shape {x.shape}")

        if np.issubdtype(x.dtype, np.integer):
            max_abs = float(np.iinfo(x.dtype).max)
            x = x.astype(np.float32) / max_abs
        else:
            x = x.astype(np.float32)

        if fs is not None and fs != self.spec.target_sample_rate:
            up = self.spec.target_sample_rate
            down = int(fs)
            common = gcd(up, down)
            x = resample_poly(x, up // common, down // common).astype(np.float32)

        target_len = self.spec.target_num_samples
        if x.shape[0] > target_len:
            x = x[:target_len]
        elif x.shape[0] < target_len:
            x = np.pad(x, (0, target_len - x.shape[0]))

        if self.spec.normalize_peak:
            peak = float(np.max(np.abs(x)))
            if peak > 0.0:
                x = x / peak

        if self.spec.add_batch_dimension:
            x = x[None, :]

        if self.spec.channel_last:
            x = x[..., None]
        else:
            x = x[:, None, :]

        return x.astype(np.float32, copy=False)
