#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Species:
    class_name: str
    scientific_name: str
    genus: str
    species: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Train/Validation splits from preprocessed Xeno-canto snippets. "
            "Uses hard links by default to avoid duplicating WAV files."
        )
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Processed Xeno-canto directory containing class folders and metadata/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory that will contain Train/ and Validation/.",
    )
    parser.add_argument(
        "--species-file",
        default=str(SCRIPT_DIR / "species.json"),
        help="Species JSON file.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction for bird classes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split.",
    )
    parser.add_argument(
        "--background-dataset-dir",
        default=None,
        help=(
            "Optional BioDCASE dataset root. If provided, Background is pulled from "
            "<root>/Train/Background and <root>/Validation/Background."
        ),
    )
    parser.add_argument(
        "--copy-instead-of-link",
        action="store_true",
        help="Copy files instead of creating hard links.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the output directory if it already exists.",
    )
    return parser.parse_args()


def load_species(path: Path) -> list[Species]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Species(**item) for item in data]


def ensure_clean_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {path}. Pass --overwrite to rebuild it.")
        shutil.rmtree(path)
    (path / "Train").mkdir(parents=True, exist_ok=True)
    (path / "Validation").mkdir(parents=True, exist_ok=True)
    (path / "metadata").mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_instead: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_instead:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def collect_species_files(processed_dir: Path, class_name: str) -> list[Path]:
    class_dir = processed_dir / class_name
    if not class_dir.exists():
        return []
    return sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav")


def split_paths(paths: list[Path], val_fraction: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    if len(paths) <= 1:
        return paths, []
    shuffled = paths[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1
    val_paths = sorted(shuffled[:val_count])
    train_paths = sorted(shuffled[val_count:])
    return train_paths, val_paths


def build_summary_entry(train_paths: list[Path], val_paths: list[Path]) -> dict[str, Any]:
    return {
        "train_count": len(train_paths),
        "validation_count": len(val_paths),
        "total_count": len(train_paths) + len(val_paths),
    }


def main() -> int:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    species_list = load_species(Path(args.species_file))
    rng = random.Random(args.seed)

    ensure_clean_output(output_dir, overwrite=args.overwrite)

    split_manifest_path = output_dir / "metadata" / "split_manifest.jsonl"
    summary_path = output_dir / "metadata" / "split_summary.json"
    class_map_path = output_dir / "metadata" / "class_map.json"

    summary: dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "copy_instead_of_link": bool(args.copy_instead_of_link),
        "classes": {},
    }
    class_names: list[str] = []

    bird_file_map: dict[str, tuple[list[Path], list[Path]]] = {}
    total_files = 0
    for item in species_list:
        files = collect_species_files(processed_dir, item.class_name)
        train_paths, val_paths = split_paths(files, args.val_fraction, rng)
        bird_file_map[item.class_name] = (train_paths, val_paths)
        total_files += len(files)

    background_train: list[Path] = []
    background_val: list[Path] = []
    if args.background_dataset_dir:
        background_root = Path(args.background_dataset_dir)
        background_train = sorted((background_root / "Train" / "Background").glob("*.wav"))
        background_val = sorted((background_root / "Validation" / "Background").glob("*.wav"))
        total_files += len(background_train) + len(background_val)

    progress = tqdm(total=total_files, desc="Build splits", unit="file", dynamic_ncols=True)
    with split_manifest_path.open("w", encoding="utf-8") as manifest:
        for item in species_list:
            class_names.append(item.class_name)
            train_paths, val_paths = bird_file_map[item.class_name]
            train_dir = output_dir / "Train" / item.class_name
            val_dir = output_dir / "Validation" / item.class_name
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)

            for src in train_paths:
                dst = train_dir / src.name
                link_or_copy(src, dst, args.copy_instead_of_link)
                manifest.write(json.dumps({"split": "Train", "class_name": item.class_name, "source": str(src), "target": str(dst)}) + "\n")
                progress.update(1)
            for src in val_paths:
                dst = val_dir / src.name
                link_or_copy(src, dst, args.copy_instead_of_link)
                manifest.write(json.dumps({"split": "Validation", "class_name": item.class_name, "source": str(src), "target": str(dst)}) + "\n")
                progress.update(1)

            summary["classes"][item.class_name] = build_summary_entry(train_paths, val_paths)

        if background_train or background_val:
            class_names.append("Background")
            train_dir = output_dir / "Train" / "Background"
            val_dir = output_dir / "Validation" / "Background"
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            for src in background_train:
                dst = train_dir / src.name
                link_or_copy(src, dst, args.copy_instead_of_link)
                manifest.write(json.dumps({"split": "Train", "class_name": "Background", "source": str(src), "target": str(dst)}) + "\n")
                progress.update(1)
            for src in background_val:
                dst = val_dir / src.name
                link_or_copy(src, dst, args.copy_instead_of_link)
                manifest.write(json.dumps({"split": "Validation", "class_name": "Background", "source": str(src), "target": str(dst)}) + "\n")
                progress.update(1)
            summary["classes"]["Background"] = build_summary_entry(background_train, background_val)

    progress.close()

    class_map = {name: idx for idx, name in enumerate(sorted(class_names))}
    class_map_path.write_text(json.dumps(class_map, indent=2), encoding="utf-8")
    summary["class_map"] = class_map
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote split manifest: {split_manifest_path}")
    print(f"Wrote split summary: {summary_path}")
    print(f"Wrote class map: {class_map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
