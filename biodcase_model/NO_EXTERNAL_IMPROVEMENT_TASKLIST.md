# No-External-Data Model Improvement Tasklist

## Scope

Improve the BioDCASE 2026 Task 3 model using only the provided development dataset:

```text
BioDCASE2026_TinyML_Development_Dataset/
├── Train/
└── Validation/
```

This track must not use Xeno-canto, BirdNET, Macaulay Library, pretrained teacher models, or any other external dataset/model. The goal is to produce the safest official submission candidate: reproducible, rule-clean, compact, and deployable.

Current reference run:

- Config: `baseline_57k`
- Parameters: 57,552
- Best checkpoint: epoch 21
- Best validation macro-F1: 44.34%
- Best validation accuracy: 45.36%
- Main failure modes:
  - `Common Chaffinch`, `Eurasian Blackcap`, `Common Chiffchaff`, `Great Tit`, and `Song Thrush` are weak.
  - Many samples are over-predicted as `Great Spotted Woodpecker` or `Eurasian Blackbird`.

## Phase 1: Metrics, Reports, And Reproducibility

- [x] Fix validation metric logging so checkpointing monitors epoch-level `val_macro_f1`.
- [x] Save `best_checkpoint` and `best_model_score` in `results.json`.
- [ ] Add Average Precision metrics, because the Task 3 page lists average precision as a classification metric.
  - [ ] macro AP
  - [ ] weighted AP
  - [ ] per-class AP
  - [ ] checkpoint monitor option for `val_macro_ap`
- [ ] Add a run plotting CLI:
  - [ ] loss curves
  - [ ] accuracy curve
  - [ ] macro-F1 curve
  - [ ] AP curve
  - [ ] confusion matrix PNG
  - [ ] per-class F1 bar chart
- [ ] Add a run comparison CLI that summarizes multiple log directories into one CSV.
- [ ] Add learned front-end reporting:
  - [ ] final `breakpoint`
  - [ ] final `transition_width`
  - [ ] frozen filterbank `.npz`
  - [ ] filterbank plot PNG

## Phase 2: Data Quality And Preprocessing

- [ ] Add an audio audit CLI:
  - [ ] per-file RMS/peak/DC offset
  - [ ] silent or near-silent clips
  - [ ] clipping detection
  - [ ] duration/sample-rate/channel sanity checks
  - [ ] per-class summary
- [ ] Inspect misclassified validation examples from weak classes.
- [ ] Add optional waveform preprocessing configs:
  - [ ] remove DC offset
  - [ ] RMS normalization alternative to peak normalization
  - [ ] high-pass filter option
  - [ ] pre-emphasis option
- [ ] Test whether peak normalization is hurting background/noisy field recordings.
- [ ] Add deterministic multi-crop validation:
  - [ ] keep default single 3-second clip for official comparability
  - [ ] optional shifted crops for diagnosis only

## Phase 3: Augmentation Improvements

- [ ] Expand augmentation beyond the current gain/noise/shift:
  - [ ] time masking on spectrogram
  - [ ] frequency masking on spectrogram
  - [ ] random time stretch or resample jitter within a small range
  - [ ] random band-pass or EQ tilt
  - [ ] background/noise mixing using only BioDCASE `Background` train clips
- [ ] Add per-class stronger augmentation for weak classes.
- [ ] Add augmentation ablation configs:
  - [ ] no augmentation
  - [ ] waveform-only augmentation
  - [ ] SpecAugment-only
  - [ ] waveform + SpecAugment
  - [ ] background-mix augmentation

## Phase 4: Losses, Sampling, And Calibration

- [ ] Compare focal loss against cross entropy.
- [ ] Add and test label smoothing.
- [ ] Sweep focal gamma:
  - [ ] `gamma=0.5`
  - [ ] `gamma=1.0`
  - [ ] `gamma=1.5`
  - [ ] `gamma=2.0`
- [ ] Add class-balanced sampler even though class counts are almost balanced, to test whether batch composition affects weak classes.
- [ ] Add confusion-aware oversampling for weak classes, using only Train labels.
- [ ] Add probability calibration diagnostics:
  - [ ] confidence histogram
  - [ ] expected calibration error
  - [ ] per-class threshold diagnostics for AP reporting

## Phase 5: Model And Front-End Sweeps

- [ ] Sweep compact model sizes:
  - [ ] `hidden_dim=64` (~53.6k params)
  - [ ] `hidden_dim=70` (~57.6k params)
  - [ ] `hidden_dim=80`
  - [ ] smaller deployment-safe variant
- [ ] Sweep front-end settings:
  - [ ] `n_fft=512`, `hop_length=240`
  - [ ] `n_fft=1024`, `hop_length=240`
  - [ ] `n_fft=1024`, `hop_length=320`
  - [ ] `n_linear_filters=48`, `64`, `80`
  - [ ] `f_min=50`, `100`, `150`
- [ ] Test fixed front-end baselines:
  - [ ] mel
  - [ ] linear triangular
  - [ ] combined log-linear trainable
  - [ ] combined log-linear frozen after warmup
- [ ] Test architecture variants:
  - [ ] current Matchbox + GRU + attention
  - [ ] Matchbox + pooling + classifier
  - [ ] Conv-only deployment-safe classifier
  - [ ] smaller GRU or no projection

## Phase 6: Experiment Schedule

Run these first, in order:

1. `baseline_57k` fixed metrics rerun.
2. `baseline_57k` with `gamma=1.0`.
3. `baseline_57k` with cross entropy + label smoothing.
4. `baseline_57k` with stronger background-mix augmentation.
5. `baseline` smaller 53.6k params with the best loss/augmentation setting.
6. One deployment-safe no-GRU variant.

For each run, record:

- run directory
- best checkpoint path
- best macro-F1
- best macro AP
- best accuracy
- model parameter count
- confusion matrix
- weak-class F1 values
- notes on training stability

## Phase 7: Final No-External Submission Candidate

- [ ] Select best no-external config on the fixed Validation split.
- [ ] Train final model on the complete development set only after model selection.
- [ ] Freeze and export the learned filterbank.
- [ ] Prepare no-external metadata YAML.
- [ ] Prepare no-external technical report section.
- [ ] Keep this system as submission slot 1 unless a later external-data system is clearly better and allowed.

## Guardrails

- Do not use evaluation data for training, feature selection, threshold tuning, or statistics.
- Do not use Xeno-canto, BirdNET, Macaulay Library, pretrained bird models, or any external labels in this track.
- Keep all improvements reproducible through Hydra configs.
- Preserve the fixed Train/Validation split for comparable development results.
- Any final full-development training must happen only after settings are selected.
