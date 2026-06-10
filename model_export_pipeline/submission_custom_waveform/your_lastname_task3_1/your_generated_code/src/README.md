Custom ESP-IDF source tree for the BioDCASE 2026 Task 3 submission.

This firmware implements the embedded runtime path used by the stateful
streaming candidate:

- waveform input on device
- frozen custom frontend on device
- sample-wise normalization on device
- backbone float32 TFLite model on device
- recurrent streaming-step float32 TFLite model on device

The backbone produces one embedding per frontend frame. The streaming-step
model is then invoked once per frame while the firmware keeps the recurrent and
attention state buffers between invocations. This avoids embedding the full
unrolled GRU sequence into one large TFLite graph.

The final benchmark firmware uses one deterministic synthetic waveform input.
Validation WAV fixtures are not compiled into the final runtime; host-side
validation accuracy is measured separately by `submission_test.py`.

The project is intended for build, flash, and monitor through the official
challenge ESP-IDF toolchain. It keeps CSV-style profiler logs for setup,
preprocessing, combined backbone plus streaming-step inference, and memory
reporting. The RAM line consumed by the challenge parser reports the larger
backbone TFLM arena allocation.
