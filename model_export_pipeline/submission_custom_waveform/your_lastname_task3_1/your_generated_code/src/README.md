Custom ESP-IDF source tree for the BioDCASE 2026 Task 3 submission.

This firmware implements the embedded runtime path used by the selected int8
candidate:

- waveform input on device
- frozen custom frontend on device
- sample-wise normalization on device
- classifier-only int8 TFLite model on device

The project is intended for build, flash, and monitor through the official
challenge ESP-IDF toolchain. It keeps the log structure expected by the
challenge parser for setup, preprocessing, inference, and memory reporting.
