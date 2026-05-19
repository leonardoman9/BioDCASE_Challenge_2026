# Waveform Monolithic Quantization Status

Target model:

- checkpoint: `checkpoints/biodcase_best_06717.ckpt`
- ONNX: `exports/biodcase_best_06717_waveform_static.onnx`
- float32 TFLite reference: `exports/biodcase_best_06717_waveform_float32.tflite`

## What was attempted

### 1. `onnx2tf` direct full-int8

Command path:

- `onnx_to_tflite.py --quantize int8`

Result:

- the graph conversion itself starts and reaches full model conversion
- `onnx2tf -oiqt ...` dies with `SIGSEGV`
- no final int8 `.tflite` is produced

Observed failure:

- crash inside container during full integer quantization
- failure happens after SavedModel generation, during the int8 export path

### 2. TensorFlow `SavedModel -> TFLite` full-int8

Script:

- `saved_model_to_tflite_int8.py`

Variants attempted:

- builtin int8, int8 I/O, new quantizer
- builtin int8, int8 I/O, legacy quantizer
- builtin int8, float32 I/O, legacy quantizer
- per-channel disabled
- `SELECT_TF_OPS` allowed

Result:

- all variants crash before producing a `.tflite`
- the crash happens immediately after the converter reports:
  - `Estimated count of arithmetic ops: 651.347 M ops`

Practical conclusion:

- with the current toolchain, the **monolithic waveform graph is not exportable to working int8 TFLite**

## Fallbacks that do work

### Float16 waveform TFLite

Artifact already produced by the converter workdir:

- `.conversion_work/run/output/saved_model/model_float16.tflite`

Validation parity on 549 files:

- accuracy: `0.6520947176684881`
- macro_f1: `0.6485337848710724`
- prediction agreement vs PyTorch: `1.0`
- mean abs logit diff: `0.012916897423565388`

Reference report:

- `exports/backend_comparison_waveform_float16_validation.json`

### Dynamic-range waveform TFLite

Artifact already produced by the converter workdir:

- `.conversion_work/run/output/saved_model/model_dynamic_range_quant.tflite`

Validation parity on 549 files:

- accuracy: `0.6247723132969034`
- macro_f1: `0.6199647459529983`
- prediction agreement vs PyTorch: `0.8142076502732241`

Reference report:

- `exports/backend_comparison_waveform_dynamic_validation.json`

Practical conclusion:

- dynamic-range quantization degrades too much for this model

## Current conclusion

The current status for the **monolithic waveform model** is:

- float32: works
- float16: works and preserves parity well
- dynamic-range quantization: works but degrades noticeably
- full int8: currently blocked by converter crashes

## Most likely next options

1. keep waveform monolith for host/submission and use float16 as a fallback deploy artifact
2. move embedded work to a different graph form, instead of insisting on monolithic waveform int8
3. if strict embedded deploy is mandatory, revisit architecture rather than only the converter
