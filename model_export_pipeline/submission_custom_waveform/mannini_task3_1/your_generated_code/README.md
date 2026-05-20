Custom embedded source tree for the frozen-frontend waveform submission.

This `src/` project targets the chosen embedded candidate:

- model: `biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.tflite`
- runtime contract:
  - waveform input on device
  - frozen frontend on device
  - pre-normalized int8 classifier-only TFLite inside firmware

The firmware keeps the challenge parser-compatible log structure:

- setup timing block
- preprocessing timing block
- inference timing block

To compile later with the official toolchain:

```bash
python compile_embedded_src_code.py
```

Or from the submission folder, after adapting the serial device if needed:

```bash
python submission_test.py
```

Note: the current challenge `submission_test.py` skips build if `your_generated_code/src`
already exists, so compile/flash is best done explicitly via the toolchain helpers.
