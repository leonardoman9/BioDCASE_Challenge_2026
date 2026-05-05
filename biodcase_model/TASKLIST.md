# BioDCASE 2026 TinyML Project Tasklist

## Goal

Build a self-contained project inside `biodcase_model` for the BioDCASE 2026 TinyML bird-sound challenge. The project must train and evaluate a compact semi-learnable WrenNet-style classifier on the provided `Train` and `Validation` splits, support optional BirdNET distillation, log reproducible experiment artifacts, and provide Docker and test tooling.

The dataset directory is kept intact:

```text
BioDCASE2026_TinyML_Development_Dataset/
├── Train/
└── Validation/
```

Initial dataset audit:

- Train: 2200 WAV files, 200 per class.
- Validation: 549 WAV files.
- `Validation/Great Tit` currently has 49 WAV files instead of the expected 50.
- Classes: 10 bird species plus `Background`.

## Implementation Plan

- [x] Create a clean Python package `biodcase_edge`.
- [x] Add a BioDCASE-native dataset loader that reads fixed `Train` and `Validation` splits.
- [x] Save and reuse a stable `class_map.json`.
- [x] Validate dataset counts through `inspect_dataset`; audio integrity checks still need a dependency-backed runtime.
- [x] Port the compact semi-learnable model pieces from `rf4423`:
  - [x] `Improved_Phi_GRU_ATT`
  - [x] Matchbox/GRU/attention modules
  - [x] combined log-linear semi-learnable spectral front-end
- [x] Adapt model defaults to BioDCASE:
  - [x] 24 kHz audio
  - [x] 3 second clips
  - [x] 11 classes
  - [x] 64 spectral filters
  - [x] `hidden_dim=64`
  - [x] `matchbox.base_filters=32`
  - [x] `matchbox.num_layers=3`
- [x] Implement training with PyTorch Lightning:
  - [x] train loop
  - [x] validation loop
  - [x] checkpoints
  - [x] early stopping
  - [x] CSV/file logging
- [x] Implement losses:
  - [x] cross entropy
  - [x] focal loss
  - [x] focal distillation loss
- [x] Add Hydra configs:
  - [x] `baseline.yaml`
  - [x] `focal.yaml`
  - [x] `distillation.yaml`
  - [x] `debug.yaml`
  - [x] `export.yaml`
  - [x] `baseline_57k.yaml`
- [x] Add BirdNET soft-label extraction pipeline:
  - [x] species-name mapping
  - [x] teacher metadata
  - [x] `Background` handling
  - [x] JSON output suitable for distillation training
- [x] Add command-line entry points:
  - [x] train
  - [x] evaluate
  - [x] predict
  - [x] inspect dataset
  - [x] extract soft labels
  - [x] export ONNX
- [x] Add reproducible logging artifacts:
  - [x] `results.json`
  - [x] `model_summary.txt`
  - [x] classification report
  - [x] confusion matrix CSV
  - [x] resolved config
  - [x] class map
- [x] Add Docker support:
  - [x] `Dockerfile`
  - [x] `.dockerignore`
  - [x] `docker-compose.yml`
- [x] Add tests:
  - [x] dataset count and class-map tests
  - [x] dataloader shape test
  - [x] model forward test
  - [x] loss smoke test
  - [x] CLI config-load smoke test
- [x] Add project documentation:
  - [x] `README.md`
  - [x] `requirements.txt`
  - [x] usage examples

## Challenge Delivery Insights

The local `SUBMISSIONS_INSTRUCTIONS.md` file is still BioDCASE 2025 material: it mentions the 2025 deadline, the 2025 CMT portal, and the previous generic DCASE package structure. For Task 3 2026, the current BioDCASE page points to the official `birdnet-team/BioDCASE-Tiny-2026` baseline repository for the final rules. That repository currently states that the final submission rules are still being updated and expected around mid-May 2026.

The 2026 Task 3 delivery target should therefore be treated as embedded/deployment-oriented, not just a Python experiment or CSV prediction task. The package is expected to include an inference model and submission metadata, with `.tflite` strongly preferred because classification performance, model size, inference time, and peak memory are ranking criteria.

Practical implications for this project:

- ONNX export is useful for debugging/intermediate portability, but it is not enough for final Task 3 delivery.
- A deployable `.tflite` or baseline-compatible inference artifact must be added.
- The semi-learnable filter is already inside the PyTorch checkpoint, but for embedded delivery it should be frozen after training and exported as a fixed filterbank or implemented in the baseline feature-extraction path.
- The safest deployable design is `baseline feature extraction + frozen learned filterbank + compact TFLite model`, rather than trying to export dynamic `torch.stft` and filter generation into TFLite Micro.
- A physical ESP32-S3-Korvo-2 board is useful but not required. Without the board, the minimum confidence checks are TFLite conversion, local TFLite inference, operator inspection, and successful compilation through the official baseline Docker/ESP-IDF flow.
- BirdNET distillation should not be the primary official system unless external-resource eligibility is confirmed and declared to organizers. Use a no-external-data supervised system as the primary submission.

## Submission Plan

- [ ] Track the official 2026 Task 3 submission rules once the baseline repository updates them.
- [ ] Add `config/final_train.yaml` for final training on the complete development set after model selection on the fixed Train/Validation split.
- [ ] Add a submission-safe export path:
  - [ ] freeze learned front-end parameters (`breakpoint`, `transition_width`, final filterbank)
  - [ ] save the frozen filterbank and feature parameters as reproducible artifacts
  - [ ] export the classifier path to `.tflite`
  - [ ] compare PyTorch vs exported inference on representative samples
- [ ] Add a baseline-compatibility path:
  - [ ] inspect official BioDCASE-Tiny-2026 feature extraction and model loading hooks
  - [ ] integrate the frozen filterbank into the baseline feature extraction or generated firmware path
  - [ ] compile the generated embedded project with the official Docker/ESP-IDF flow
  - [ ] collect available model-size and memory/inference reports
- [ ] Add submission utilities:
  - [ ] `biodcase_edge.cli.benchmark_resources`
  - [ ] `biodcase_edge.cli.package_submission`
  - [ ] metadata YAML generator/template
  - [ ] technical-report outline/template
  - [ ] optional evaluation-set prediction CSV writer if the final rules still require CSV outputs
- [ ] Prepare a final zip package with one to four systems, following the final official structure.

Expected package contents, pending the final 2026 rules:

- inference model artifact for each submitted system, ideally `.tflite`
- optional feature-extraction artifact/code if the submitted system differs from the baseline features
- metadata YAML for each submitted system
- technical report PDF, maximum 4+1 pages
- optional system output CSV if requested by the final Task 3 instructions

Recommended system slots:

1. `baseline_57k`: no external data, semi-learnable frozen filterbank, focal loss, `hidden_dim=70`.
2. `baseline`: no external data, smaller semi-learnable model, `hidden_dim=64`.
3. BirdNET distillation variant only if external model use is allowed and declared.
4. Optional resource-optimized or accuracy-optimized variant after deployment checks.

## Verification Status

- [x] Python syntax compilation passed with `python3 -m compileall -q biodcase_edge tests`.
- [x] Runtime tests were initially blocked by missing ML dependencies; after dependency installation the runtime smoke checks pass.
- [x] After installing dependencies, `pytest` passes: 7 tests passed.
- [x] `python -m biodcase_edge.cli.train --config-name debug` passes with finite loss.
- [x] Added `baseline_57k.yaml`; for 11 BioDCASE classes, `hidden_dim=70` gives approximately 57.5k parameters, while `hidden_dim=64` gives approximately 53.6k.
- [x] `baseline_57k` fast-dev smoke run passes with finite train/validation loss.
- [ ] `.tflite` export has not been implemented yet.
- [ ] Official baseline compilation/deployment compatibility has not been verified yet.

## Initial Experiment Strategy

1. Start with a supervised baseline: semi-learnable combined log-linear front-end plus focal loss.
2. Verify end-to-end training on `debug.yaml`.
3. Train the full baseline on the fixed BioDCASE splits.
4. Add BirdNET soft labels and run focal distillation only after the baseline is reliable.
5. Add final training on the full development set after selecting settings on validation.
6. Make the selected model deployment-safe before investing in many larger variants.
7. Compare `hidden_dim=64`, `hidden_dim=70`, and larger variants only after the compact model is exportable and baseline-compatible.

## Open Risks

- BirdNET species names may not map exactly to all BioDCASE class names.
- `Background` is a real BioDCASE class, not a BirdNET class; distillation must handle it explicitly.
- BirdNET and any other teacher model count as external trained models; official use requires eligibility and organizer notification.
- ESP32 deployment constraints require TensorFlow Lite Micro or baseline-specific conversion beyond ONNX export.
- The current GRU/attention architecture may not be TFLite Micro friendly. A submission-safe variant may need to remove GRU or replace it with Conv/DepthwiseConv/Pooling/FC blocks.
- Exporting dynamic waveform-to-logits with `torch.stft` inside the graph is risky for TFLite Micro. Prefer frozen feature extraction plus a compact exported classifier.
- Without a physical Korvo-2 board, compatibility can only be approximated by TFLite conversion, local TFLite inference, operator inspection, and official baseline compilation.
- Current validation split has one missing `Great Tit` file relative to the expected 50.
