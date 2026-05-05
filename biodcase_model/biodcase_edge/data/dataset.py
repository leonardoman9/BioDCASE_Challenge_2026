from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .audio import augment_waveform, load_waveform

AUDIO_EXTENSIONS = {".wav", ".WAV"}
SPLIT_DIRS = {"train": "Train", "validation": "Validation", "val": "Validation", "test": "Validation"}


@dataclass(frozen=True)
class AudioRecord:
    path: Path
    label: int
    class_name: str
    split: str


def discover_classes(dataset_dir: str | Path) -> list[str]:
    train_dir = Path(dataset_dir) / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    return sorted(path.name for path in train_dir.iterdir() if path.is_dir() and not path.name.startswith("."))


def build_class_map(dataset_dir: str | Path, class_map_path: str | Path | None = None) -> dict[str, int]:
    classes = discover_classes(dataset_dir)
    class_map = {name: idx for idx, name in enumerate(classes)}
    if class_map_path is not None:
        path = Path(class_map_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(class_map, indent=2), encoding="utf-8")
    return class_map


def load_class_map(dataset_dir: str | Path, class_map_path: str | Path | None = None) -> dict[str, int]:
    if class_map_path is not None and Path(class_map_path).exists():
        return json.loads(Path(class_map_path).read_text(encoding="utf-8"))
    return build_class_map(dataset_dir, class_map_path)


def collect_records(dataset_dir: str | Path, split: str, class_map: dict[str, int]) -> list[AudioRecord]:
    split_key = split.lower()
    if split_key not in SPLIT_DIRS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {sorted(SPLIT_DIRS)}")
    split_name = SPLIT_DIRS[split_key]
    split_dir = Path(dataset_dir) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    records: list[AudioRecord] = []
    for class_name, label in sorted(class_map.items(), key=lambda item: item[1]):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix in AUDIO_EXTENSIONS and path.is_file():
                records.append(AudioRecord(path=path, label=label, class_name=class_name, split=split_name))
    return records


class BioDCASEDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        class_map: dict[str, int] | None = None,
        class_map_path: str | Path | None = None,
        sample_rate: int = 24000,
        clip_duration: float = 3.0,
        augment: bool = False,
        augmentation: dict[str, Any] | None = None,
        soft_labels_path: str | Path | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.class_map = class_map or load_class_map(self.dataset_dir, class_map_path)
        self.idx_to_class = {idx: name for name, idx in self.class_map.items()}
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.augment = augment
        self.augmentation = augmentation or {}
        self.return_metadata = return_metadata
        self.records = collect_records(self.dataset_dir, split, self.class_map)
        self.soft_labels = self._load_soft_labels(soft_labels_path)

        if not self.records:
            raise RuntimeError(f"No WAV files found for split={split} in {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        waveform = load_waveform(record.path, self.sample_rate, self.clip_duration)
        if self.augment:
            max_shift = int(self.augmentation.get("max_shift_seconds", 0.0) * self.sample_rate)
            waveform = augment_waveform(
                waveform,
                noise_std=float(self.augmentation.get("noise_std", 0.0)),
                gain_min=float(self.augmentation.get("gain_min", 1.0)),
                gain_max=float(self.augmentation.get("gain_max", 1.0)),
                max_shift_samples=max_shift,
            )

        label = torch.tensor(record.label, dtype=torch.long)
        if self.soft_labels is not None:
            soft, mask = self._soft_label_for(record)
            payload = (waveform, label, soft, mask)
        else:
            payload = (waveform, label)

        if self.return_metadata:
            return (*payload, {"path": str(record.path), "class_name": record.class_name})
        return payload

    def _load_soft_labels(self, path: str | Path | None) -> dict[str, list[float]] | None:
        if path is None:
            return None
        path = Path(path)
        if path.is_dir():
            path = path / "soft_labels.json"
        if not path.exists():
            raise FileNotFoundError(f"Soft-label file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        labels = data.get("soft_labels", data)
        if not isinstance(labels, dict):
            raise ValueError(f"Soft-label JSON must contain a mapping, got {type(labels)!r}")
        return labels

    def _soft_label_for(self, record: AudioRecord) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.soft_labels is not None
        candidates = [
            str(record.path),
            record.path.name,
            record.path.stem,
            str(record.path.relative_to(self.dataset_dir)),
        ]
        vector = None
        for key in candidates:
            if key in self.soft_labels:
                vector = self.soft_labels[key]
                break

        if vector is None:
            vector = [0.0] * len(self.class_map)
            vector[record.label] = 1.0
            mask = torch.tensor(False)
        else:
            mask = torch.tensor(True)

        tensor = torch.tensor(vector, dtype=torch.float32)
        if tensor.numel() != len(self.class_map):
            raise ValueError(
                f"Soft-label vector for {record.path.name} has {tensor.numel()} classes; "
                f"expected {len(self.class_map)}"
            )
        total = tensor.sum()
        if total > 0:
            tensor = tensor / total
        return tensor, mask

