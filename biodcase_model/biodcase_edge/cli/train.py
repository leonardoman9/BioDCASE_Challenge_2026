from __future__ import annotations

import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from biodcase_edge.cli.common import load_config, parse_config_args, save_resolved_config
from biodcase_edge.data import BioDCASEDataModule
from biodcase_edge.training import BioDCASEExperiment
from biodcase_edge.utils import configure_logging, setup_run_dir, write_json

log = logging.getLogger(__name__)


def compute_balanced_class_weights(counts: dict[str, int], class_names: list[str]) -> list[float]:
    total = sum(counts.values())
    num_classes = len(class_names)
    weights = []
    for name in class_names:
        count = max(1, counts.get(name, 0))
        weights.append(total / (num_classes * count))
    return weights


def main(argv=None) -> None:
    config_name, overrides = parse_config_args("Train BioDCASE TinyML model", argv)
    cfg = load_config(config_name, overrides)

    run_dir = setup_run_dir(cfg.logging.log_dir, cfg.project.name)
    configure_logging(str(cfg.logging.level), run_dir / "train.log")
    save_resolved_config(cfg, run_dir / "resolved_config.yaml")

    if cfg.project.get("seed") is not None:
        L.seed_everything(int(cfg.project.seed), workers=True)

    datamodule = BioDCASEDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup("fit")
    class_names = datamodule.class_names
    split_counts = datamodule.split_counts()
    write_json({"class_names": class_names, "counts": split_counts}, run_dir / "dataset_summary.json")

    if cfg.loss.get("class_weights") == "auto":
        cfg.loss.class_weights = compute_balanced_class_weights(split_counts["train"], class_names)
        log.info("Computed balanced focal class weights: %s", cfg.loss.class_weights)

    experiment = BioDCASEExperiment(cfg, class_names=class_names)
    logger = CSVLogger(save_dir=str(run_dir), name="csv", version="")

    callbacks = []
    checkpoint_callback = None
    if cfg.trainer.get("enable_checkpointing", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best-{epoch:03d}-{val_macro_f1:.4f}",
            monitor=str(cfg.checkpoint.monitor),
            mode=str(cfg.checkpoint.mode),
            save_top_k=int(cfg.checkpoint.save_top_k),
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
    if cfg.early_stopping.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.early_stopping.monitor),
                mode=str(cfg.early_stopping.mode),
                patience=int(cfg.early_stopping.patience),
                min_delta=float(cfg.early_stopping.min_delta),
            )
        )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.pop("enable_checkpointing", None)
    trainer = L.Trainer(
        **trainer_kwargs,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(run_dir),
        enable_checkpointing=bool(cfg.trainer.get("enable_checkpointing", True)),
    )

    log.info("Starting training. Run dir: %s", run_dir)
    trainer.fit(experiment, datamodule=datamodule)

    metrics = {
        key: float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
        for key, value in trainer.callback_metrics.items()
        if hasattr(value, "item") or isinstance(value, (float, int))
    }
    metrics["run_dir"] = str(run_dir)
    metrics["num_classes"] = len(class_names)
    metrics["total_params"] = sum(p.numel() for p in experiment.model.parameters())
    metrics["trainable_params"] = sum(p.numel() for p in experiment.model.parameters() if p.requires_grad)
    if checkpoint_callback is not None:
        metrics["best_checkpoint"] = str(checkpoint_callback.best_model_path)
        if checkpoint_callback.best_model_score is not None:
            metrics["best_model_score"] = float(checkpoint_callback.best_model_score.detach().cpu().item())
    write_json(metrics, run_dir / "results.json")
    (run_dir / "model_summary.txt").write_text(
        "\n".join(
            [
                f"Model: {experiment.model.__class__.__name__}",
                f"Total parameters: {metrics['total_params']:,}",
                f"Trainable parameters: {metrics['trainable_params']:,}",
                f"Best checkpoint: {checkpoint_callback.best_model_path if checkpoint_callback else ''}",
                f"Best score: {checkpoint_callback.best_model_score if checkpoint_callback else ''}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Training complete. Results saved to %s", run_dir)


if __name__ == "__main__":
    main()
