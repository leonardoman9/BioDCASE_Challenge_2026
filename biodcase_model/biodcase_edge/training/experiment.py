from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import MulticlassAccuracy

from biodcase_edge.losses import DistillationLoss, FocalDistillationLoss, FocalLoss
from biodcase_edge.metrics import summarize_classification
from biodcase_edge.utils.reporting import (
    save_classification_report,
    save_confusion_matrix,
    write_json,
)


def _to_container(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return OmegaConf.to_container(value, resolve=True) or {}


class BioDCASEExperiment(L.LightningModule):
    def __init__(self, cfg: DictConfig, class_names: list[str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        model_params = _to_container(cfg.model.params)
        model_params["num_classes"] = self.num_classes
        self.model = instantiate({"_target_": cfg.model.target, **model_params})

        self.loss_fn = self._build_supervised_loss()
        self.distillation_loss_fn = self._build_distillation_loss() if cfg.distillation.enabled else None

        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(waveform)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        waveform, labels, soft_labels, soft_mask = self._unpack_batch(batch)
        logits = self(waveform)
        loss = self._compute_loss(logits, labels, soft_labels, soft_mask, stage="train")
        acc = self.train_acc(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        waveform, labels, soft_labels, soft_mask = self._unpack_batch(batch)
        logits = self(waveform)
        loss = self._compute_loss(logits, labels, soft_labels, soft_mask, stage="val")
        preds = logits.argmax(dim=1)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(labels.detach().cpu())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_targets.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).numpy().tolist()
        targets = torch.cat(self._val_targets).numpy().tolist()
        summary = summarize_classification(targets, preds)
        self.log("val/accuracy", summary.accuracy, prog_bar=True)
        self.log("val/macro_f1", summary.macro_f1, prog_bar=True)
        self.log("val/weighted_f1", summary.weighted_f1, prog_bar=False)
        self.log("val/weighted_precision", summary.weighted_precision, prog_bar=False)
        self.log("val/weighted_recall", summary.weighted_recall, prog_bar=False)
        self.log("val_accuracy", summary.accuracy, prog_bar=False)
        self.log("val_macro_f1", summary.macro_f1, prog_bar=False)
        self.log("val_weighted_f1", summary.weighted_f1, prog_bar=False)

        # Backward-compatible names for previous CSV analysis scripts.
        self.log("val/weighted_f1_epoch", summary.weighted_f1, prog_bar=False)
        self.log("val/weighted_precision_epoch", summary.weighted_precision, prog_bar=False)
        self.log("val/weighted_recall_epoch", summary.weighted_recall, prog_bar=False)
        for name, value in self._frontend_parameter_metrics().items():
            self.log(name, value, prog_bar=False)

        if not self.trainer.sanity_checking:
            output_dir = self._output_dir()
            save_classification_report(targets, preds, self.class_names, output_dir, "val")
            save_confusion_matrix(targets, preds, self.class_names, output_dir, "val")
            write_json(
                {
                    "accuracy": summary.accuracy,
                    "macro_f1": summary.macro_f1,
                    "weighted_f1": summary.weighted_f1,
                    "weighted_precision": summary.weighted_precision,
                    "weighted_recall": summary.weighted_recall,
                    "epoch": int(self.current_epoch),
                },
                output_dir / "latest_val_metrics.json",
            )

    def configure_optimizers(self):
        optim_cfg = self.cfg.optimizer
        lr = float(optim_cfg.lr)
        weight_decay = float(optim_cfg.get("weight_decay", 0.0))
        param_groups = self._parameter_groups(lr, weight_decay)
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=0.0,
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            eps=float(optim_cfg.get("eps", 1e-8)),
        )

        scheduler_cfg = self.cfg.get("scheduler")
        if not scheduler_cfg or not scheduler_cfg.get("enabled", False):
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "min")),
            factor=float(scheduler_cfg.get("factor", 0.6)),
            patience=int(scheduler_cfg.get("patience", 7)),
            min_lr=float(scheduler_cfg.get("min_lr", 5e-7)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": str(scheduler_cfg.get("monitor", "val/loss")),
                "interval": "epoch",
            },
        }

    def _build_supervised_loss(self) -> nn.Module:
        loss_cfg = self.cfg.loss
        if loss_cfg.type == "cross_entropy":
            return nn.CrossEntropyLoss()
        if loss_cfg.type in ("focal", "focal_distillation"):
            class_weights = loss_cfg.get("class_weights")
            if class_weights in (None, "none", "auto"):
                class_weights = None
            return FocalLoss(
                alpha=class_weights if class_weights is not None else 1.0,
                gamma=float(loss_cfg.get("gamma", 2.0)),
            )
        raise ValueError(f"Unsupported loss type: {loss_cfg.type}")

    def _build_distillation_loss(self) -> nn.Module:
        dist_cfg = self.cfg.distillation
        loss_cfg = self.cfg.loss
        if loss_cfg.type == "focal_distillation":
            class_weights = loss_cfg.get("class_weights")
            if class_weights in (None, "none", "auto"):
                class_weights = None
            return FocalDistillationLoss(
                alpha=float(dist_cfg.alpha),
                gamma=float(loss_cfg.get("gamma", 2.0)),
                temperature=float(dist_cfg.temperature),
                class_weights=class_weights,
            )
        return DistillationLoss(alpha=float(dist_cfg.alpha), temperature=float(dist_cfg.temperature))

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_labels: torch.Tensor | None,
        soft_mask: torch.Tensor | None,
        stage: str,
    ) -> torch.Tensor:
        if self.distillation_loss_fn is None or soft_labels is None or soft_mask is None:
            return self.loss_fn(logits, labels)

        mask = soft_mask.view(-1).bool()
        if mask.any():
            total, hard, soft = self.distillation_loss_fn(logits[mask], labels[mask], soft_labels[mask])
            self.log(f"{stage}/hard_loss", hard, on_epoch=True, batch_size=int(mask.sum()))
            self.log(f"{stage}/soft_loss", soft, on_epoch=True, batch_size=int(mask.sum()))
            if (~mask).any():
                supervised = self.loss_fn(logits[~mask], labels[~mask])
                return total + supervised
            return total
        return self.loss_fn(logits, labels)

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            waveform, labels, soft_labels, soft_mask = batch
            return (
                waveform.float().to(self.device, non_blocking=True),
                labels.long().to(self.device, non_blocking=True),
                soft_labels.float().to(self.device, non_blocking=True),
                soft_mask.bool().to(self.device, non_blocking=True),
            )
        if len(batch) == 2:
            waveform, labels = batch
            return (
                waveform.float().to(self.device, non_blocking=True),
                labels.long().to(self.device, non_blocking=True),
                None,
                None,
            )
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    def _parameter_groups(self, base_lr: float, base_weight_decay: float) -> list[dict[str, Any]]:
        breakpoint_lr = float(self.cfg.optimizer.get("breakpoint_lr", base_lr))
        transition_lr = float(self.cfg.optimizer.get("transition_width_lr", base_lr))
        breakpoint_weight_decay = float(self.cfg.optimizer.get("breakpoint_weight_decay", 0.0))
        transition_weight_decay = float(self.cfg.optimizer.get("transition_width_weight_decay", 0.0))
        groups: dict[str, list[torch.nn.Parameter]] = {
            "main": [],
            "breakpoint": [],
            "transition": [],
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "combined_log_linear_spec.breakpoint" in name:
                groups["breakpoint"].append(param)
            elif "combined_log_linear_spec.transition_width" in name:
                groups["transition"].append(param)
            else:
                groups["main"].append(param)

        result = []
        if groups["main"]:
            result.append({"params": groups["main"], "lr": base_lr, "weight_decay": base_weight_decay})
        if groups["breakpoint"]:
            result.append(
                {"params": groups["breakpoint"], "lr": breakpoint_lr, "weight_decay": breakpoint_weight_decay}
            )
        if groups["transition"]:
            result.append(
                {"params": groups["transition"], "lr": transition_lr, "weight_decay": transition_weight_decay}
            )
        return result

    def _frontend_parameter_metrics(self) -> dict[str, torch.Tensor]:
        spec = getattr(self.model, "combined_log_linear_spec", None)
        if spec is None:
            return {}
        metrics: dict[str, torch.Tensor] = {}
        if hasattr(spec, "effective_breakpoint"):
            metrics["frontend/breakpoint_hz"] = spec.effective_breakpoint().detach()
        if hasattr(spec, "effective_transition_width"):
            metrics["frontend/transition_width"] = spec.effective_transition_width().detach()
        return metrics

    def _output_dir(self) -> Path:
        if self.trainer.logger and getattr(self.trainer.logger, "log_dir", None):
            return Path(self.trainer.logger.log_dir)
        return Path(self.cfg.project.output_dir)
