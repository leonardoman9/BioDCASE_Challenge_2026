from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as L
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .dataset import BioDCASEDataset, build_class_map, collect_records


class BioDCASEDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_dir = Path(cfg.data.dataset_dir)
        self.class_map_path = Path(cfg.data.class_map_path)
        self.class_map: dict[str, int] = {}
        self.train_dataset: Optional[BioDCASEDataset] = None
        self.val_dataset: Optional[BioDCASEDataset] = None

    def prepare_data(self) -> None:
        self.class_map_path.parent.mkdir(parents=True, exist_ok=True)
        self.class_map = build_class_map(self.dataset_dir, self.class_map_path)

    def setup(self, stage: str | None = None) -> None:
        if not self.class_map:
            self.class_map = build_class_map(self.dataset_dir, self.class_map_path)

        data_cfg = self.cfg.data
        aug_cfg = OmegaConf.to_container(data_cfg.get("augmentation", {}), resolve=True) or {}
        soft_labels_path = None
        if bool(self.cfg.distillation.get("enabled", False)):
            soft_labels_path = self.cfg.distillation.soft_labels_path

        if stage in (None, "fit", "validate"):
            self.train_dataset = BioDCASEDataset(
                dataset_dir=self.dataset_dir,
                split="train",
                class_map=self.class_map,
                sample_rate=int(data_cfg.sample_rate),
                clip_duration=float(data_cfg.clip_duration),
                augment=bool(data_cfg.get("use_augmentation", False)),
                augmentation=aug_cfg,
                soft_labels_path=soft_labels_path,
            )
            self.val_dataset = BioDCASEDataset(
                dataset_dir=self.dataset_dir,
                split="validation",
                class_map=self.class_map,
                sample_rate=int(data_cfg.sample_rate),
                clip_duration=float(data_cfg.clip_duration),
                augment=False,
                soft_labels_path=soft_labels_path,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized")
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            persistent_workers=bool(self.cfg.data.get("persistent_workers", False))
            and int(self.cfg.data.num_workers) > 0,
            drop_last=bool(self.cfg.data.get("drop_last", False)),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized")
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            persistent_workers=bool(self.cfg.data.get("persistent_workers", False))
            and int(self.cfg.data.num_workers) > 0,
        )

    @property
    def class_names(self) -> list[str]:
        if not self.class_map:
            self.class_map = build_class_map(self.dataset_dir, self.class_map_path)
        return [name for name, _ in sorted(self.class_map.items(), key=lambda item: item[1])]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def split_counts(self) -> dict[str, dict[str, int]]:
        if not self.class_map:
            self.class_map = build_class_map(self.dataset_dir, self.class_map_path)
        result: dict[str, dict[str, int]] = {}
        for split in ("train", "validation"):
            records = collect_records(self.dataset_dir, split, self.class_map)
            counts = {name: 0 for name in self.class_names}
            for record in records:
                counts[record.class_name] += 1
            result[split] = counts
        return result

