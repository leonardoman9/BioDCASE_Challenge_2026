## Custom waveform submission package

Local solution folder:

- `mannini_task3_1/`

Contents:

- `config_submission.yaml`: submission config for the custom waveform path
- `feature_handler.py`: waveform preprocessing to `[1, 72000, 1]`
- `inference_handler.py`: custom PyTorch/TFLite inference handler
- `biodcase_edge/`: vendored model package needed to load the PyTorch artifact
- `your_submission_model/`: model artifacts
- `host_inference_scores.yaml`: host-side smoke test results on the challenge demo files

Local host-side test:

```bash
./biodcase_model/.venv/bin/python model_export_pipeline/submission_custom_waveform/run_host_submission_test.py
```

This stages the official challenge repo into:

- `model_export_pipeline/submission_custom_waveform/_staging/BioDCASE-Tiny-2026/`

Then it overlays the custom submission files and runs the host-side inference test only.

Current scope:

- host-side submission path: working
- embedded code generation / int8 deployment: not implemented yet
