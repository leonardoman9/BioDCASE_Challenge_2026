from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent / "biodcase_model"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedded_classifier_only.models import (
    ClassifierOnlyPreNormalizedWrapper,
    ClassifierOnlyUnrolledGRUPreNormalizedWrapper,
    FrozenFrontendFeatureExtractor,
    build_frozen_frontend,
    prepare_frontend_features,
)
from pytorch_to_onnx import instantiate_model, load_checkpoint, resolve_cfg

from biodcase_edge.data.audio import load_waveform
from biodcase_edge.data.dataset import collect_records


DEFAULT_CHECKPOINT = PIPELINE_ROOT / "checkpoints" / "biodcase_best_06717.ckpt"
DEFAULT_DATASET_DIR = PIPELINE_ROOT / "eval_data" / "BioDCASE2026_TinyML_Development_Dataset"
DEFAULT_CLASS_MAP = PIPELINE_ROOT / "eval_data" / "class_map.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "representative_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump post-frontend tensors for classifier-only export work")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Best checkpoint")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Dataset root")
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP), help="Class map json")
    parser.add_argument("--split", default="validation", choices=("validation", "val", "train", "test"))
    parser.add_argument("--limit", type=int, default=128, help="Maximum number of examples to dump")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument(
        "--external-normalization",
        action="store_true",
        help="Normalize frontend features outside the exported graph",
    )
    parser.add_argument(
        "--unrolled-gru",
        action="store_true",
        help="Use the primitive-op GRU wrapper when producing reference logits",
    )
    return parser.parse_args()


def load_class_map(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    class_map_path = Path(args.class_map).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"]).cpu().eval()
    feature_extractor = FrozenFrontendFeatureExtractor(build_frozen_frontend()).cpu().eval()
    if args.unrolled_gru:
        classifier_wrapper = ClassifierOnlyUnrolledGRUPreNormalizedWrapper(model).cpu().eval()
    else:
        classifier_wrapper = ClassifierOnlyPreNormalizedWrapper(model).cpu().eval()

    class_map = load_class_map(class_map_path)
    records = collect_records(dataset_dir, args.split, class_map)
    if args.limit is not None:
        records = records[: args.limit]

    features_all: list[np.ndarray] = []
    labels_all: list[int] = []
    paths_all: list[str] = []
    logits_all: list[np.ndarray] = []

    sample_rate = int(cfg["data"]["sample_rate"])
    clip_duration = float(cfg["data"]["clip_duration"])

    for idx, record in enumerate(records, start=1):
        waveform = load_waveform(record.path, sample_rate, clip_duration).unsqueeze(0)
        with torch.no_grad():
            feature_tensor = feature_extractor(waveform)
            if args.external_normalization:
                feature_tensor = prepare_frontend_features(feature_tensor, normalize=True)
                logits = classifier_wrapper(feature_tensor).cpu().numpy().astype(np.float32)
            else:
                logits = model(waveform).cpu().numpy().astype(np.float32)
            features = feature_tensor.cpu().numpy().astype(np.float32)

        features_all.append(features[0])
        logits_all.append(logits[0])
        labels_all.append(int(record.label))
        paths_all.append(str(record.path))

        if idx % 50 == 0 or idx == len(records):
            print(f"processed {idx}/{len(records)}: {record.path.name}")

    features_np = np.stack(features_all)
    logits_np = np.stack(logits_all)
    labels_np = np.asarray(labels_all, dtype=np.int64)

    features_path = output_dir / f"{args.split}_frontend_features.npy"
    logits_path = output_dir / f"{args.split}_pytorch_logits.npy"
    labels_path = output_dir / f"{args.split}_labels.npy"
    manifest_path = output_dir / f"{args.split}_manifest.json"

    np.save(features_path, features_np)
    np.save(logits_path, logits_np)
    np.save(labels_path, labels_np)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "num_samples": int(features_np.shape[0]),
        "feature_shape": list(features_np.shape[1:]),
        "feature_dtype": str(features_np.dtype),
        "sample_rate": sample_rate,
        "clip_duration": clip_duration,
        "frontend_spec": str((PIPELINE_ROOT / "frontend_specs" / "biodcase_best_06717" / "frontend_spec.json").resolve()),
        "external_normalization": bool(args.external_normalization),
        "unrolled_gru": bool(args.unrolled_gru),
        "feature_file": features_path.name,
        "logits_file": logits_path.name,
        "labels_file": labels_path.name,
        "source_paths": paths_all,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"saved features: {features_path}")
    print(f"saved logits:   {logits_path}")
    print(f"saved labels:   {labels_path}")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
