# BioDCASE 2026 TinyML Bird Classifier

This project trains a compact semi-learnable bird-sound classifier for the BioDCASE 2026 TinyML challenge. It is adapted from the `rf4423` WrenNet pipeline and uses the fixed BioDCASE development split directly.

## Dataset

Expected layout:

```text
BioDCASE2026_TinyML_Development_Dataset/
├── Train/
│   ├── Background/
│   ├── Common Chaffinch/
│   └── ...
└── Validation/
    ├── Background/
    ├── Common Chaffinch/
    └── ...
```

Current local audit:

- Train: 2200 files.
- Validation: 549 files.
- `Validation/Great Tit` has 49 files.
- Audio is loaded as mono, resampled or checked to 24 kHz, normalized, and padded/truncated to 3 seconds.

## Setup

Use Python 3.11. PyTorch/torchaudio wheels may not be available for very new Python versions.

```bash
cd biodcase_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inspect Dataset

```bash
python -m biodcase_edge.cli.inspect_dataset --config-name baseline
```

This writes `outputs/dataset_inspection.json` and `outputs/class_map.json`.

## Train

Debug smoke run:

```bash
python -m biodcase_edge.cli.train --config-name debug
```

Baseline compact semi-learnable model:

```bash
python -m biodcase_edge.cli.train --config-name baseline
```

Parameter-matched 57k variant for the 11-class BioDCASE task:

```bash
python -m biodcase_edge.cli.train --config-name baseline_57k
```

Focal-loss variant:

```bash
python -m biodcase_edge.cli.train --config-name focal
```

Focal distillation, after soft labels exist:

```bash
python -m biodcase_edge.cli.extract_soft_labels --config-name distillation
python -m biodcase_edge.cli.train --config-name distillation
```

## Model Defaults

The baseline model uses:

- `combined_log_linear` semi-learnable front-end
- 24 kHz waveform input
- 3 second clips
- 64 spectral filters
- `hidden_dim=64` in `baseline`, approximately 53.6k parameters with 11 classes
- `hidden_dim=70` in `baseline_57k`, approximately 57.5k parameters with 11 classes
- Matchbox base filters: 32
- Matchbox layers: 3
- 11 output classes

The front-end learns two global parameters: breakpoint and transition width.

## Logs

Each training run writes to:

```text
logs/<experiment>/<timestamp>/
```

Artifacts include:

- `train.log`
- `resolved_config.yaml`
- `dataset_summary.json`
- `results.json`
- `model_summary.txt`
- `latest_val_metrics.json`
- `val_classification_report.txt`
- `val_confusion_matrix.csv`
- `checkpoints/`

## Evaluate

```bash
python -m biodcase_edge.cli.evaluate logs/.../checkpoints/best-...ckpt --config-name baseline
```

## Predict One File

```bash
python -m biodcase_edge.cli.predict logs/.../checkpoints/best-...ckpt path/to/audio.wav --config-name baseline
```

## Export ONNX

```bash
python -m biodcase_edge.cli.export_onnx logs/.../checkpoints/best-...ckpt --config-name export
```

ONNX export is provided as an intermediate artifact. ESP32/TFLite Micro conversion may still need the official BioDCASE baseline export path.

## Docker

```bash
docker compose run --rm inspect
docker compose run --rm train
docker compose run --rm soft-labels
```

The dataset is mounted through the local project directory, not baked into the image.

## Tests

```bash
pytest
```

The tests cover dataset counts, config loading, model forward shape, and loss smoke checks.
