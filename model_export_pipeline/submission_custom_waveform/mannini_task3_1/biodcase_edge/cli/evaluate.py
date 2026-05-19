from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as L
from omegaconf import OmegaConf

from biodcase_edge.cli.common import load_config
from biodcase_edge.data import BioDCASEDataModule
from biodcase_edge.training import BioDCASEExperiment
from biodcase_edge.utils import configure_logging, write_json

log = logging.getLogger(__name__)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BioDCASE checkpoint on validation split")
    parser.add_argument("checkpoint", help="Lightning checkpoint path")
    parser.add_argument("--config-name", default="baseline")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_name, args.overrides)
    configure_logging(str(cfg.logging.level))
    datamodule = BioDCASEDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup("validate")

    model = BioDCASEExperiment.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        class_names=datamodule.class_names,
        strict=False,
    )
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.pop("max_epochs", None)
    trainer_kwargs.pop("enable_checkpointing", None)
    trainer = L.Trainer(**trainer_kwargs, logger=False, enable_checkpointing=False)
    results = trainer.validate(model, datamodule=datamodule)
    out = Path(cfg.project.output_dir) / "evaluation_results.json"
    write_json({"checkpoint": args.checkpoint, "results": results}, out)
    log.info("Evaluation results saved to %s", out)


if __name__ == "__main__":
    main()

