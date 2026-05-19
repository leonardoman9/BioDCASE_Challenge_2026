from __future__ import annotations

import argparse
import json

import torch

from biodcase_edge.cli.common import load_config
from biodcase_edge.data.audio import load_waveform
from biodcase_edge.data.dataset import load_class_map
from biodcase_edge.training import BioDCASEExperiment


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict one WAV file with a BioDCASE checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("audio_path")
    parser.add_argument("--config-name", default="baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_name, args.overrides)
    class_map = load_class_map(cfg.data.dataset_dir, cfg.data.class_map_path)
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    model = BioDCASEExperiment.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        class_names=class_names,
        strict=False,
    )
    model.eval()

    waveform = load_waveform(args.audio_path, int(cfg.data.sample_rate), float(cfg.data.clip_duration)).unsqueeze(0)
    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top_k = min(args.top_k, len(class_names))
    values, indices = torch.topk(probs, k=top_k)
    result = {
        "audio_path": args.audio_path,
        "prediction": class_names[int(indices[0])],
        "top_k": [
            {"class_name": class_names[int(idx)], "probability": float(prob)}
            for prob, idx in zip(values, indices)
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

