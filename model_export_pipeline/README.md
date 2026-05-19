## Model export pipeline

Workspace for exporting and validating the current BioDCASE model.

### Final model reference

- best training checkpoint: fine-tune without distillation
- best run checkpoint path on the training machine:
  - `/mnt/sda4/Progetti/BioDCASE_Challenge_2026/biodcase_model/logs/biodcase_finetune_from_xenocanto_57k/2026-05-13_15-00-15/checkpoints/best-epoch=019-val_macro_f1=0.6717.ckpt`

### What is implemented

- `export_frontend_spec.py`
  - exports the frozen frontend specification from the best checkpoint
- `waveform_exportable.py`
  - exportable waveform frontend and end-to-end wrapper
- `pytorch_to_onnx.py`
  - PyTorch -> ONNX exporter
- `onnx_to_tflite.py`
  - ONNX -> TFLite conversion through Docker/onnx2tf
- `saved_model_to_tflite_int8.py`
  - direct SavedModel -> TFLite quantization experiments
- `compare_backends.py`
  - parity test for spectrogram-input exports
- `compare_waveform_backends.py`
  - parity test for waveform-input exports

### Verified paths

- spectrogram -> ONNX -> TFLite
- waveform -> ONNX -> TFLite float32
- waveform -> ONNX -> TFLite float16

### Validation status

- waveform float32 parity vs PyTorch: verified
- waveform float16 parity vs PyTorch: verified
- waveform dynamic-range quantization: works, but degrades
- waveform full-int8 monolithic export: currently blocked

See:

- `INT8_STATUS.md`
- `submission_custom_waveform/`

### Notes

- checkpoints, large exports, local datasets, and converter workdirs are intentionally kept out of git
- the folder is meant to keep the scripts, configs, specs, and submission scaffolding needed to reproduce the work
