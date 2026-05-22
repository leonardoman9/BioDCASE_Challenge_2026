## Embedded classifier-only branch

This workspace is the new embedded path after freezing the waveform fallback.

Target architecture:

```text
waveform
-> frozen custom frontend
-> frontend features [1, 64, 301]
-> classifier-only TFLite
```

What stays outside the embedded `.tflite`:

- STFT equivalent
- frozen filter bank
- dB conversion

What stays inside the embedded `.tflite`:

- per-sample normalization
- feature alignment
- `phi` / CNN backbone
- GRU
- attention
- final FC

Why this branch exists:

- the waveform monolithic `float32` and `float16` exports are correct
- the waveform monolithic `int8` path is currently blocked by converter crashes
- the embedded target now needs a simpler graph for full-int8 attempts

Current contract:

- frontend output tensor: `float32`
- shape: `[batch, 64, 301]`
- semantic: post-frontend dB spectrogram before sample normalization

Scripts in this folder:

- `models.py`
  - reusable wrappers for frozen frontend extraction and classifier-only export
- `export_classifier_only_onnx.py`
  - exports the post-frontend branch to ONNX
- `prepare_representative_dataset.py`
  - dumps real post-frontend tensors for calibration and parity work
- `compare_classifier_only_backends.py`
  - parity check for `PyTorch -> ONNX -> TFLite` on post-frontend features

Already verified:

1. classifier-only ONNX export
2. classifier-only TFLite float32 conversion
3. parity on full validation set:
   - PyTorch accuracy: `0.6520947176684881`
   - ONNX accuracy: `0.6520947176684881`
   - TFLite float32 accuracy: `0.6520947176684881`
   - prediction agreement vs PyTorch: `1.0`

Current local artifacts:

- `artifacts/biodcase_best_06717_classifier_only_unrolled_prenorm.onnx`
- `artifacts/biodcase_best_06717_classifier_only_unrolled_prenorm_float32.tflite`
- `artifacts/biodcase_best_06717_classifier_only_unrolled_prenorm_float32.json`
- `artifacts/biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.tflite`
- `artifacts/biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.json`
- `artifacts/backend_comparison_classifier_only_unrolled_prenorm_float32_int8_rerun.json`
- `artifacts/quantization_variant_summary.json`

Current conclusion:

1. the post-frontend float32 branch preserves parity with PyTorch
2. the selected embedded candidate is `classifier_only_unrolled_prenorm_int8_legacy_nopc`
3. its accuracy is lower than float32, but it is the best deploy-oriented int8 compromise found so far
4. the matching custom ESP-IDF source tree is generated into `submission_custom_waveform/mannini_task3_1/your_generated_code/src`
