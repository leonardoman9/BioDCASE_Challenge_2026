from __future__ import annotations

import argparse
from pathlib import Path

import torch

from biodcase_edge.cli.common import load_config
from biodcase_edge.data.dataset import load_class_map
from biodcase_edge.training import BioDCASEExperiment


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export a BioDCASE Lightning checkpoint to ONNX")
    parser.add_argument("checkpoint")
    parser.add_argument("--config-name", default="export")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_name, args.overrides)
    class_map = load_class_map(cfg.data.dataset_dir, cfg.data.class_map_path)
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    experiment = BioDCASEExperiment.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        class_names=class_names,
        strict=False,
    )
    experiment.eval()
    model = experiment.model
    output_path = Path(args.output_path or cfg.export.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, int(cfg.data.sample_rate * cfg.data.clip_duration), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(cfg.export.opset_version),
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()

