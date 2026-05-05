from __future__ import annotations

import logging

from biodcase_edge.cli.common import load_config, parse_config_args
from biodcase_edge.data import BioDCASEDataModule
from biodcase_edge.utils import configure_logging, write_json

log = logging.getLogger(__name__)


def main(argv=None) -> None:
    config_name, overrides = parse_config_args("Inspect BioDCASE dataset", argv)
    cfg = load_config(config_name, overrides)
    configure_logging(str(cfg.logging.level))
    dm = BioDCASEDataModule(cfg)
    dm.prepare_data()
    summary = {"class_names": dm.class_names, "counts": dm.split_counts()}
    write_json(summary, cfg.project.output_dir + "/dataset_inspection.json")
    log.info("Classes: %s", dm.class_names)
    log.info("Counts: %s", summary["counts"])


if __name__ == "__main__":
    main()

