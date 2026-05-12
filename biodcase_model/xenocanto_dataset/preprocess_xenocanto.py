#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torchaudio
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
from tqdm.auto import tqdm


DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_CLIP_DURATION = 3.0
DEFAULT_LOWCUT = 150.0
DEFAULT_HIGHCUT = 10_000.0
DEFAULT_MIN_PEAK_DISTANCE = 1.0
DEFAULT_HEIGHT_PERCENTILE = 75.0
DEFAULT_TOP_K_WINDOWS = 1
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Species:
    class_name: str
    scientific_name: str
    genus: str
    species: str

    @property
    def slug(self) -> str:
        return self.class_name.strip().lower().replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Xeno-canto raw audio into BioDCASE-style 3s WAV snippets. "
            "Pipeline: decode -> mono -> 24kHz -> energy-based peak detection -> 3s snippet -> PCM16 export."
        )
    )
    parser.add_argument(
        "--input-root",
        default=str(SCRIPT_DIR),
        help="Root folder containing raw_audio/ and species.json.",
    )
    parser.add_argument(
        "--raw-audio-dir",
        default=None,
        help="Raw audio directory. Defaults to <input-root>/raw_audio.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for processed WAV snippets.",
    )
    parser.add_argument(
        "--species-file",
        default=str(SCRIPT_DIR / "species.json"),
        help="Species JSON mapping file.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional Xeno-canto recordings.jsonl manifest for richer metadata linking.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sample rate.",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=DEFAULT_CLIP_DURATION,
        help="Snippet duration in seconds.",
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=DEFAULT_LOWCUT,
        help="Band-pass low cutoff in Hz.",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=DEFAULT_HIGHCUT,
        help="Band-pass high cutoff in Hz.",
    )
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=DEFAULT_MIN_PEAK_DISTANCE,
        help="Minimum distance between detected peaks in seconds.",
    )
    parser.add_argument(
        "--height-percentile",
        type=float,
        default=DEFAULT_HEIGHT_PERCENTILE,
        help="Peak envelope percentile threshold after prominence sorting.",
    )
    parser.add_argument(
        "--max-files-per-species",
        type=int,
        default=None,
        help="Optional cap on processed recordings per species.",
    )
    parser.add_argument(
        "--species",
        action="append",
        help="Restrict processing to one class/scientific name/slug. Can be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate existing output snippets.",
    )
    parser.add_argument(
        "--delete-source-after-success",
        action="store_true",
        help="Delete the raw source file after a snippet is successfully exported.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file failures and fallback decisions.",
    )
    return parser.parse_args()


def load_species(path: Path) -> list[Species]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Species(**item) for item in data]


def load_manifest_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record = payload.get("recording", {})
            record_id = str(record.get("id") or record.get("xc_id") or "")
            if record_id:
                index[record_id] = payload
    return index


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.98)
    if not 0 < low < high < 1:
        raise ValueError(
            f"Invalid band-pass params: lowcut={lowcut}, highcut={highcut}, fs={fs}"
        )
    return butter(order, [low, high], btype="band")


def apply_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        return filtfilt(b, a, data)
    except ValueError:
        return data


def compute_adaptive_prominence(y: np.ndarray, sr: int, lowcut: float, highcut: float) -> float:
    y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
    frame_length = max(int(sr * 0.05), 1)
    hop_length = max(int(sr * 0.01), 1)
    if len(y_filtered) < frame_length:
        return float(np.max(np.abs(y_filtered)) if len(y_filtered) else 0.0)
    envelope_frames = frame_signal(np.abs(y_filtered), frame_length, hop_length)
    envelope = envelope_frames.mean(axis=0)
    median_env = np.median(envelope)
    mad_env = np.median(np.abs(envelope - median_env))
    return float(median_env + 1.5 * mad_env)


def frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(y) < frame_length:
        return np.empty((frame_length, 0), dtype=y.dtype)
    num_frames = 1 + (len(y) - frame_length) // hop_length
    shape = (frame_length, num_frames)
    strides = (y.strides[0], y.strides[0] * hop_length)
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)


def choose_best_window(
    y: np.ndarray,
    sr: int,
    clip_duration: float,
    lowcut: float,
    highcut: float,
    min_peak_distance: float,
    height_percentile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    duration = len(y) / sr if sr else 0.0
    target_samples = int(round(clip_duration * sr))
    if len(y) <= target_samples:
        return pad_or_trim(y, target_samples), {
            "strategy": "pad_short_audio",
            "start_time_sec": 0.0,
            "end_time_sec": min(duration, clip_duration),
            "peak_time_sec": None,
        }

    adaptive_prominence = compute_adaptive_prominence(y, sr, lowcut, highcut)
    y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
    frame_length = max(int(sr * 0.05), 1)
    hop_length = max(int(sr * 0.01), 1)
    envelope_frames = frame_signal(np.abs(y_filtered), frame_length, hop_length)

    if envelope_frames.shape[1] > 0:
        envelope = envelope_frames.mean(axis=0)
        min_peak_distance_frames = max(int(min_peak_distance / (hop_length / sr)), 1)
        peaks, properties = find_peaks(
            envelope,
            prominence=adaptive_prominence,
            distance=min_peak_distance_frames,
        )
        if len(peaks) > 0:
            sorted_indices = np.argsort(-properties["prominences"])
            sorted_peaks = peaks[sorted_indices]
            height_threshold = np.percentile(envelope[sorted_peaks], height_percentile)
            selected_peaks = [int(p) for p in sorted_peaks if envelope[p] >= height_threshold]
            if selected_peaks:
                peak_frame = selected_peaks[0]
                peak_time = peak_frame * hop_length / sr
                start_time = max(0.0, peak_time - clip_duration / 2)
                start_sample = int(round(start_time * sr))
                start_sample = min(start_sample, max(len(y) - target_samples, 0))
                end_sample = start_sample + target_samples
                return pad_or_trim(y[start_sample:end_sample], target_samples), {
                    "strategy": "peak_centered",
                    "start_time_sec": start_sample / sr,
                    "end_time_sec": end_sample / sr,
                    "peak_time_sec": peak_time,
                    "adaptive_prominence": adaptive_prominence,
                }

    # Fallback: pick the maximum-energy 3s window on the filtered signal.
    energy = moving_window_energy(y_filtered, target_samples)
    best_start = int(np.argmax(energy)) if len(energy) else 0
    best_end = min(best_start + target_samples, len(y))
    return pad_or_trim(y[best_start:best_end], target_samples), {
        "strategy": "max_energy_fallback",
        "start_time_sec": best_start / sr,
        "end_time_sec": best_end / sr,
        "peak_time_sec": None,
        "adaptive_prominence": adaptive_prominence,
    }


def moving_window_energy(y: np.ndarray, window_samples: int) -> np.ndarray:
    if len(y) < window_samples:
        return np.array([float(np.mean(y ** 2))], dtype=np.float64)
    squared = y.astype(np.float64) ** 2
    cumsum = np.concatenate([[0.0], np.cumsum(squared)])
    window_energy = cumsum[window_samples:] - cumsum[:-window_samples]
    return window_energy / max(window_samples, 1)


def pad_or_trim(y: np.ndarray, target_samples: int) -> np.ndarray:
    if len(y) == target_samples:
        return y.astype(np.float32, copy=False)
    if len(y) > target_samples:
        return y[:target_samples].astype(np.float32, copy=False)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: len(y)] = y.astype(np.float32, copy=False)
    return padded


def to_pcm16(y: np.ndarray) -> np.ndarray:
    clipped = np.clip(y, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def load_audio_mono_resampled(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
    mono = waveform.squeeze(0).detach().cpu().numpy()
    if mono.size == 0:
        raise ValueError("empty waveform")
    peak = np.max(np.abs(mono))
    if peak > 1.0:
        mono = mono / peak
    return mono.astype(np.float32, copy=False), sample_rate


def extract_recording_id(path: Path) -> str:
    stem = path.stem
    if stem.startswith("XC"):
        return stem.split("_", 1)[0][2:]
    return stem


def build_output_name(source_path: Path, clip_index: int = 1) -> str:
    return f"{source_path.stem}_clip_{clip_index:03d}.wav"


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    raw_audio_dir = Path(args.raw_audio_dir) if args.raw_audio_dir else (input_root / "raw_audio")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    species_list = load_species(Path(args.species_file))
    if args.species:
        requested = {item.lower() for item in args.species}
        species_list = [
            item
            for item in species_list
            if item.class_name.lower() in requested
            or item.scientific_name.lower() in requested
            or item.slug in requested
        ]
        if not species_list:
            raise SystemExit(f"No species matched: {', '.join(args.species)}")

    manifest_index = load_manifest_index(Path(args.manifest_path) if args.manifest_path else None)
    processed_manifest_path = metadata_dir / "processed_snippets.jsonl"
    summary_path = metadata_dir / "summary.json"

    clip_samples = int(round(args.sample_rate * args.clip_duration))
    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "raw_audio_dir": str(raw_audio_dir),
        "output_dir": str(output_dir),
        "sample_rate": args.sample_rate,
        "clip_duration": args.clip_duration,
        "clip_samples": clip_samples,
        "lowcut": args.lowcut,
        "highcut": args.highcut,
        "snippets_per_recording": 1,
        "species": {},
    }

    species_counts: dict[str, int] = {}
    planned_files_by_species: dict[str, list[Path]] = {}
    total_planned = 0
    for species in species_list:
        slug_dir = raw_audio_dir / species.slug
        if not slug_dir.exists():
            planned_files_by_species[species.class_name] = []
            continue
        files = sorted(
            path
            for path in slug_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        )
        if args.max_files_per_species is not None:
            files = files[: args.max_files_per_species]
        planned_files_by_species[species.class_name] = files
        total_planned += len(files)

    global_bar = tqdm(
        total=total_planned,
        desc="All species",
        unit="file",
        dynamic_ncols=True,
        leave=True,
    )

    with processed_manifest_path.open("w", encoding="utf-8") as manifest:
        for species in species_list:
            files = planned_files_by_species[species.class_name]
            slug_dir = raw_audio_dir / species.slug
            if not slug_dir.exists():
                summary["species"][species.class_name] = {
                    "scientific_name": species.scientific_name,
                    "raw_recordings_found": 0,
                    "processed_snippets": 0,
                    "failed_recordings": 0,
                    "missing_raw_dir": True,
                }
                continue

            class_output_dir = output_dir / species.class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)

            processed = 0
            failed = 0
            progress = tqdm(
                files,
                desc=species.class_name[:28],
                unit="file",
                dynamic_ncols=True,
                leave=False,
            )
            for audio_path in progress:
                output_name = build_output_name(audio_path, clip_index=1)
                output_path = class_output_dir / output_name
                if output_path.exists() and not args.overwrite:
                    processed += 1
                    global_bar.update(1)
                    global_bar.set_postfix_str(f"species={species.slug} processed={processed} failed={failed}")
                    progress.set_postfix_str(f"processed={processed} failed={failed}")
                    continue

                try:
                    waveform, sample_rate = load_audio_mono_resampled(audio_path, args.sample_rate)
                    snippet, segment_meta = choose_best_window(
                        y=waveform,
                        sr=sample_rate,
                        clip_duration=args.clip_duration,
                        lowcut=args.lowcut,
                        highcut=args.highcut,
                        min_peak_distance=args.min_peak_distance,
                        height_percentile=args.height_percentile,
                    )
                    wavfile.write(output_path, args.sample_rate, to_pcm16(snippet))
                    record_id = extract_recording_id(audio_path)
                    source_meta = manifest_index.get(record_id)
                    payload = {
                        "class_name": species.class_name,
                        "scientific_name": species.scientific_name,
                        "recording_id": record_id,
                        "source_audio_path": str(audio_path),
                        "output_audio_path": str(output_path),
                        "sample_rate": args.sample_rate,
                        "clip_duration": args.clip_duration,
                        "segment": segment_meta,
                        "source_manifest": source_meta,
                    }
                    manifest.write(json.dumps(payload, sort_keys=True) + "\n")
                    manifest.flush()
                    if args.delete_source_after_success:
                        audio_path.unlink(missing_ok=True)
                    processed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    if args.verbose:
                        print(f"FAILED {audio_path}: {exc}")
                global_bar.update(1)
                global_bar.set_postfix_str(f"species={species.slug} processed={processed} failed={failed}")
                progress.set_postfix_str(f"processed={processed} failed={failed}")

            progress.close()
            summary["species"][species.class_name] = {
                "scientific_name": species.scientific_name,
                "raw_recordings_found": len(files),
                "processed_snippets": processed,
                "failed_recordings": failed,
            }
            species_counts[species.class_name] = processed

    global_bar.close()
    summary["total_processed_snippets"] = int(sum(species_counts.values()))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote snippet manifest: {processed_manifest_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
