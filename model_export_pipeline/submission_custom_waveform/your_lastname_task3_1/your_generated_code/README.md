Custom embedded source tree for the frozen-frontend waveform submission.

This `src/` project targets the chosen embedded candidate:

- backbone model: `biodcase_backbone_streaming_float32.tflite`
- streaming-step model: `biodcase_streaming_step_float32.tflite`
- runtime contract:
  - waveform input on device
  - frozen frontend on device
  - float32 backbone TFLite inside firmware
  - float32 recurrent/attention step TFLite inside firmware, invoked once per
    frontend frame with explicit state buffers

The firmware keeps the challenge parser-compatible log structure:

- setup timing block
- preprocessing timing block
- inference timing block

To compile manually from `src/` with the ESP-IDF Docker image:

```bash
docker run --rm -v "$PWD":/project -w /project espressif/idf:release-v5.4 idf.py build
```

Or from the submission folder, after adapting the serial device if needed:

```bash
python submission_test.py
```

Note: the current challenge `submission_test.py` skips build if `your_generated_code/src`
already exists, so compile/flash is best done explicitly via the toolchain helpers.
