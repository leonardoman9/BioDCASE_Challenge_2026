## Custom waveform submission package

Local solution folder:

- `your_lastname_task3_1/`

Contents:

- `config_submission.yaml`: submission config for the custom waveform path
- `feature_handler.py`: waveform preprocessing to `[1, 72000, 1]`
- `inference_handler.py`: custom PyTorch/TFLite inference handler
- `requirements_submission_host.txt`: minimal Python deps for host-side inference
- `requirements_submission_full.txt`: deps for official `submission_test.py` and embedded build helpers
- `biodcase_edge/`: vendored model package needed to load the PyTorch artifact
- `your_submission_model/`: model artifacts
- `your_generated_code/src/`: custom ESP-IDF source tree for the embedded candidate
- `your_lastname_task3_1/DEPLOYMENT_NOTES.md`: current host/deploy fallback status
- `host_inference_scores.yaml`: host-side smoke test results on the challenge demo files
- `your_lastname_task3_1.meta.yaml`: submission metadata draft

Suggested environment setup:

```bash
source biodcase_model/.venv/bin/activate
pip install -r model_export_pipeline/submission_custom_waveform/your_lastname_task3_1/requirements_submission_host.txt
```

If you want to run the official challenge `submission_test.py` or the embedded
build helpers, install the full set instead:

```bash
source biodcase_model/.venv/bin/activate
pip install -r model_export_pipeline/submission_custom_waveform/your_lastname_task3_1/requirements_submission_full.txt
```

Local host-side test:

```bash
./biodcase_model/.venv/bin/python model_export_pipeline/submission_custom_waveform/run_host_submission_test.py
```

The runner creates a temporary local overlay of the official challenge submission
pipeline, injects the custom submission files, and runs the host-side inference
test only.

Validation-set evaluation of the actual submission package:

```bash
./biodcase_model/.venv/bin/python model_export_pipeline/submission_custom_waveform/evaluate_submission_validation.py
```

This writes:

- `your_lastname_task3_1/validation_inference_scores.yaml`

Current scope:

- host-side submission path: working
- primary host-side artifact: waveform float32 TFLite
- host-side fallback artifact: waveform float16 TFLite
- embedded candidate: classifier-only int8 TFLite
- embedded code path: implemented in `your_lastname_task3_1/your_generated_code/src`
