# Submission model artifact freeze

Python requirements shipped with this package:

- `requirements_submission_host.txt`
  - enough for custom host-side inference and parity checks
- `requirements_submission_full.txt`
  - enough for the official `submission_test.py` path and embedded build helpers

Use the full file instead of the upstream `requirements_pytorch.txt` if you want
to avoid the deprecated `sklearn` package entry in the official repository.

Primary host-side artifact:

- `your_submission_model/biodcase_best_06717_submission.tflite`
  - waveform input
  - float32
  - official host submission artifact used by `config_submission.yaml`
  - input: `[1, 72000, 1]` waveform, output: `[1, 11]` logits
  - size: 4,755,032 bytes

Primary PyTorch checkpoint:

- `your_submission_model/biodcase_best_06717_submission.pth`
  - source checkpoint loaded by `inference_handler.py`
  - size: 346,725 bytes
  - parameter count reported by the validation scorer: 57,552

Embedded deployment artifacts:

- `your_submission_model/biodcase_backbone_streaming_float32.tflite`
  - frontend-feature input
  - float32
  - input: `[1, 301, 64]`, output: `[1, 301, 32]`
  - size: 205,268 bytes
- `your_submission_model/biodcase_streaming_step_float32.tflite`
  - one-frame recurrent/attention step
  - float32
  - inputs: frame plus explicit hidden/attention state
  - outputs: updated state plus `[1, 11]` logits
  - size: 118,072 bytes
  - invoked once per frontend frame by `your_generated_code/src`

Removed fallback artifact:

- `your_submission_model/biodcase_best_06717_submission_float16.tflite`
  - waveform input
  - float16 weights
  - not selected by `config_submission.yaml`
  - removed from the final submission package

Removed legacy embedded candidate:

- `your_submission_model/biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.tflite`
  - classifier-only int8
  - expects frontend-frozen features after external sample-wise normalization
  - host-side verified, but too large/awkward as an unrolled recurrent graph for
    the current ESP32-S3 TFLite Micro runtime
  - removed from the final submission package

Current status:

- custom waveform host-side submission path: working
- waveform monolithic float32: verified
- stateful streaming float32 split: verified host-side against PyTorch and the
  host waveform TFLite
- custom embedded `src`: generated, Docker-buildable, flashed on ESP32-S3, and
  monitor-tested with parser-compatible timing blocks
- legacy int8 and float16 files: not selected for final scoring and removed
  from the final package

Practical decision:

- keep the current waveform float32 file as the stable submission reference
- use the stateful streaming split for embedded deployment
- keep only the final host and embedded artifacts in the package
