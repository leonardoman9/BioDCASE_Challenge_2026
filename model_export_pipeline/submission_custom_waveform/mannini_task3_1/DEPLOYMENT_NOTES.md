# Waveform submission deployment notes

Primary host-side artifact:

- `your_submission_model/biodcase_best_06717_submission.tflite`
  - waveform input
  - float32
  - reference artifact used by the current custom submission test

Host-side fallback artifact:

- `your_submission_model/biodcase_best_06717_submission_float16.tflite`
  - waveform input
  - float16 weights
  - smaller than float32 while preserving parity well

Chosen embedded candidate:

- `your_submission_model/biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.tflite`
  - classifier-only int8
  - expects frontend-frozen features after external sample-wise normalization
  - paired with the custom ESP-IDF source tree in `your_generated_code/src`

Current status:

- custom waveform host-side submission path: working
- waveform monolithic float32: verified
- waveform monolithic float16: verified
- classifier-only int8: verified host-side and selected as embedded candidate
- custom embedded `src`: generated and Docker-buildable
- real device flash/monitor benchmark: still pending

Practical decision:

- keep the current waveform float32 file as the stable submission reference
- keep the waveform float16 file as the host-side fallback
- use the classifier-only int8 path for embedded work
- keep new embedded work under `model_export_pipeline/embedded_classifier_only/`
