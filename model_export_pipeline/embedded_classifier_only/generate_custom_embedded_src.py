from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT / "BioDCASE-Tiny-2026" / "biodcase_tiny" / "embedded" / "firmware"
SUBMISSION_DIR = ROOT / "model_export_pipeline" / "submission_custom_waveform" / "mannini_task3_1"
OUTPUT_DIR = SUBMISSION_DIR / "your_generated_code" / "src"
ARTIFACTS_DIR = ROOT / "model_export_pipeline" / "embedded_classifier_only" / "artifacts"
FRONTEND_SPEC_DIR = ROOT / "model_export_pipeline" / "frontend_specs" / "biodcase_best_06717"

MODEL_TFLITE = ARTIFACTS_DIR / "biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.tflite"
MODEL_METADATA = ARTIFACTS_DIR / "biodcase_best_06717_classifier_only_unrolled_prenorm_int8_legacy_nopc.json"
SUBMISSION_EMBEDDED_MODEL = SUBMISSION_DIR / "your_submission_model" / MODEL_TFLITE.name
SUBMISSION_EMBEDDED_METADATA = SUBMISSION_DIR / "your_submission_model" / MODEL_METADATA.name


def chunked(items: list[str], width: int = 12) -> str:
    lines: list[str] = []
    for start in range(0, len(items), width):
        lines.append("  " + ", ".join(items[start : start + width]))
    return ",\n".join(lines)


def format_float_array(name: str, values: np.ndarray, width: int = 8) -> str:
    flat = values.reshape(-1)
    items = [f"{float(value):.9e}f" for value in flat]
    return f"const float {name}[{flat.size}] = {{\n{chunked(items, width)}\n}};\n"


def format_int_array(name: str, values: np.ndarray, c_type: str, width: int = 16) -> str:
    flat = values.reshape(-1)
    items = [str(int(value)) for value in flat]
    return f"const {c_type} {name}[{flat.size}] = {{\n{chunked(items, width)}\n}};\n"


def format_hex_model(model_bytes: bytes, width: int = 16) -> str:
    items = [f"0x{byte:02x}" for byte in model_bytes]
    return chunked(items, width)


def build_filterbank_sparse(filter_bank: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts: list[int] = []
    lengths: list[int] = []
    offsets: list[int] = []
    weights: list[float] = []

    offset = 0
    for row in filter_bank:
        nz = np.flatnonzero(np.abs(row) > 1e-12)
        if nz.size == 0:
            starts.append(0)
            lengths.append(0)
            offsets.append(offset)
            continue
        start = int(nz[0])
        end = int(nz[-1]) + 1
        segment = row[start:end]
        starts.append(start)
        lengths.append(end - start)
        offsets.append(offset)
        weights.extend(float(x) for x in segment)
        offset += segment.size

    return (
        np.asarray(starts, dtype=np.int16),
        np.asarray(lengths, dtype=np.int16),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
    )


def write_model_cpp(main_dir: Path, model_tflite: Path) -> None:
    model_bytes = model_tflite.read_bytes()
    content = f"""/* Auto-generated from {model_tflite.name}. */

#include "model.h"

#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(16)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

const unsigned char g_model[] DATA_ALIGN_ATTRIBUTE = {{
{format_hex_model(model_bytes)}
}};

const int g_model_len = {len(model_bytes)};
"""
    (main_dir / "model.cpp").write_text(content, encoding="utf-8")


def write_frontend_data(main_dir: Path) -> None:
    window = np.load(FRONTEND_SPEC_DIR / "hann_window.npy").astype(np.float32)
    filter_bank = np.load(FRONTEND_SPEC_DIR / "filter_bank.npy").astype(np.float32)
    starts, lengths, offsets, sparse_weights = build_filterbank_sparse(filter_bank)
    weight_count = int(sparse_weights.size)

    header = """#pragma once

#include <cstdint>

namespace custom_frontend {

constexpr int kSampleRate = 24000;
constexpr int kWaveformSamples = 72000;
constexpr int kFftSize = 1024;
constexpr int kFftBins = 513;
constexpr int kHopLength = 240;
constexpr int kNumFrames = 301;
constexpr int kNumFilters = 64;
constexpr float kAmplitudeFloor = 1.0e-5f;
constexpr float kTopDb = 80.0f;
constexpr float kNormalizationEps = 1.0e-5f;

extern const float kHannWindow[kFftSize];
extern const int16_t kFilterStarts[kNumFilters];
extern const int16_t kFilterLengths[kNumFilters];
extern const int32_t kFilterOffsets[kNumFilters];
extern const float kFilterWeights[__WEIGHT_COUNT__];

}  // namespace custom_frontend
"""
    header = header.replace("__WEIGHT_COUNT__", str(weight_count))
    source = f"""#include "frontend_frozen_data.h"

namespace custom_frontend {{

{format_float_array("kHannWindow", window)}
{format_int_array("kFilterStarts", starts, "int16_t")}
{format_int_array("kFilterLengths", lengths, "int16_t")}
{format_int_array("kFilterOffsets", offsets, "int32_t")}
{format_float_array("kFilterWeights", sparse_weights)}

}}  // namespace custom_frontend
"""
    (main_dir / "frontend_frozen_data.h").write_text(header, encoding="utf-8")
    (main_dir / "frontend_frozen_data.cpp").write_text(source, encoding="utf-8")


FRONTEND_HEADER = """#pragma once

#include <span>

#include "esp_err.h"
#include "esp_micro_profiler.h"
#include "frontend_frozen_data.h"
#include "tensorflow/lite/c/common.h"

namespace custom_frontend {

class FrozenFrontend {
 public:
  FrozenFrontend();
  ~FrozenFrontend() = default;

  esp_err_t Init();
  esp_err_t ExtractToModelInput(
      std::span<const float> waveform,
      TfLiteTensor* input_tensor,
      benchmark::MicroProfiler* profiler);

 private:
  int ReflectIndex(int index, int size) const;
  void PeakNormalize(std::span<const float> waveform);
  esp_err_t ComputeFrameFeatures(int frame_index, float* frame_features);

  bool initialized_ = false;
  float peak_scale_ = 1.0f;
  const float* waveform_ptr_ = nullptr;
  float* fft_buffer_ = nullptr;
  float* spectrogram_db_ = nullptr;
};

}  // namespace custom_frontend
"""


FRONTEND_SOURCE = """#include "frontend_frozen.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "dsps_fft2r.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "frontend_frozen_data.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace custom_frontend {
namespace {

constexpr auto TAG = "bm";

inline float ClampFloor(float value) {
  return value < kAmplitudeFloor ? kAmplitudeFloor : value;
}

inline int8_t QuantizeToInt8(float value, float scale, int zero_point) {
  const int quantized = static_cast<int>(std::lrintf(value / scale)) + zero_point;
  return static_cast<int8_t>(std::clamp(quantized, -128, 127));
}

}  // namespace

FrozenFrontend::FrozenFrontend() = default;

esp_err_t FrozenFrontend::Init() {
  if (initialized_) {
    return ESP_OK;
  }
  if (dsps_fft2r_init_fc32(nullptr, kFftSize) != ESP_OK) {
    ESP_LOGE(TAG, "dsps_fft2r_init_fc32 failed");
    return ESP_FAIL;
  }

  fft_buffer_ = static_cast<float*>(
      heap_caps_malloc(sizeof(float) * 2 * kFftSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  spectrogram_db_ = static_cast<float*>(
      heap_caps_malloc(sizeof(float) * kNumFrames * kNumFilters, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

  if (fft_buffer_ == nullptr || spectrogram_db_ == nullptr) {
    ESP_LOGE(TAG, "Failed to allocate frontend buffers");
    return ESP_ERR_NO_MEM;
  }
  initialized_ = true;
  return ESP_OK;
}

int FrozenFrontend::ReflectIndex(int index, int size) const {
  if (size <= 1) {
    return 0;
  }
  while (index < 0 || index >= size) {
    if (index < 0) {
      index = -index;
    }
    if (index >= size) {
      index = 2 * size - 2 - index;
    }
  }
  return index;
}

void FrozenFrontend::PeakNormalize(std::span<const float> waveform) {
  float peak = 0.0f;
  for (float sample : waveform) {
    peak = std::max(peak, std::fabs(sample));
  }
  peak_scale_ = peak > 0.0f ? 1.0f / peak : 1.0f;
}

esp_err_t FrozenFrontend::ComputeFrameFeatures(int frame_index, float* frame_features) {
  const int start_sample = frame_index * kHopLength - (kFftSize / 2);

  for (int sample_idx = 0; sample_idx < kFftSize; ++sample_idx) {
    const int reflected = ReflectIndex(start_sample + sample_idx, kWaveformSamples);
    const float sample = waveform_ptr_[reflected] * peak_scale_ * kHannWindow[sample_idx];
    fft_buffer_[2 * sample_idx] = sample;
    fft_buffer_[2 * sample_idx + 1] = 0.0f;
  }

  if (dsps_fft2r_fc32(fft_buffer_, kFftSize) != ESP_OK) {
    ESP_LOGE(TAG, "dsps_fft2r_fc32 failed");
    return ESP_FAIL;
  }
  if (dsps_bit_rev_fc32(fft_buffer_, kFftSize) != ESP_OK) {
    ESP_LOGE(TAG, "dsps_bit_rev_fc32 failed");
    return ESP_FAIL;
  }

  float magnitudes[kFftBins];
  for (int bin = 0; bin < kFftBins; ++bin) {
    const float real = fft_buffer_[2 * bin];
    const float imag = fft_buffer_[2 * bin + 1];
    magnitudes[bin] = std::sqrt(real * real + imag * imag);
  }

  for (int filter_idx = 0; filter_idx < kNumFilters; ++filter_idx) {
    const int length = kFilterLengths[filter_idx];
    const int start = kFilterStarts[filter_idx];
    const int offset = kFilterOffsets[filter_idx];
    float accum = 0.0f;
    for (int idx = 0; idx < length; ++idx) {
      accum += kFilterWeights[offset + idx] * magnitudes[start + idx];
    }
    frame_features[filter_idx] = 20.0f * std::log10(ClampFloor(accum));
  }

  return ESP_OK;
}

esp_err_t FrozenFrontend::ExtractToModelInput(
    std::span<const float> waveform,
    TfLiteTensor* input_tensor,
    benchmark::MicroProfiler* profiler) {
  if (!initialized_) {
    return ESP_FAIL;
  }
  if (waveform.size() != kWaveformSamples) {
    ESP_LOGE(TAG, "Unexpected waveform size: %d", static_cast<int>(waveform.size()));
    return ESP_ERR_INVALID_SIZE;
  }
  if (input_tensor == nullptr || input_tensor->type != kTfLiteInt8) {
    ESP_LOGE(TAG, "Expected int8 input tensor");
    return ESP_ERR_INVALID_ARG;
  }
  if (input_tensor->bytes != kNumFrames * kNumFilters) {
    ESP_LOGE(TAG, "Unexpected input tensor size: %d", static_cast<int>(input_tensor->bytes));
    return ESP_ERR_INVALID_SIZE;
  }

  waveform_ptr_ = waveform.data();

  auto event = profiler->BeginEvent("PeakNormalize");
  PeakNormalize(waveform);
  profiler->EndEvent(event);

  float max_db = -1.0e30f;
  event = profiler->BeginEvent("FrontendFFT+FB");
  for (int frame_idx = 0; frame_idx < kNumFrames; ++frame_idx) {
    float frame_features[kNumFilters];
    if (ComputeFrameFeatures(frame_idx, frame_features) != ESP_OK) {
      profiler->EndEvent(event);
      return ESP_FAIL;
    }
    for (int filter_idx = 0; filter_idx < kNumFilters; ++filter_idx) {
      const float value = frame_features[filter_idx];
      spectrogram_db_[frame_idx * kNumFilters + filter_idx] = value;
      max_db = std::max(max_db, value);
    }
  }
  profiler->EndEvent(event);

  const float cutoff = max_db - kTopDb;
  event = profiler->BeginEvent("SampleNorm");
  double sum = 0.0;
  const int count = kNumFrames * kNumFilters;
  for (int idx = 0; idx < count; ++idx) {
    if (spectrogram_db_[idx] < cutoff) {
      spectrogram_db_[idx] = cutoff;
    }
    sum += spectrogram_db_[idx];
  }
  const float mean = static_cast<float>(sum / static_cast<double>(count));

  double sq_sum = 0.0;
  for (int idx = 0; idx < count; ++idx) {
    const double centered = static_cast<double>(spectrogram_db_[idx]) - static_cast<double>(mean);
    sq_sum += centered * centered;
  }
  const double variance = sq_sum / static_cast<double>(count - 1);
  const float std = static_cast<float>(std::sqrt(variance)) + kNormalizationEps;
  profiler->EndEvent(event);

  event = profiler->BeginEvent("InputQuantize");
  const float input_scale = input_tensor->params.scale;
  const int input_zero_point = input_tensor->params.zero_point;
  auto* input_data = input_tensor->data.int8;
  for (int idx = 0; idx < count; ++idx) {
    const float normalized = (spectrogram_db_[idx] - mean) / std;
    input_data[idx] = QuantizeToInt8(normalized, input_scale, input_zero_point);
  }
  profiler->EndEvent(event);
  return ESP_OK;
}

}  // namespace custom_frontend
"""


MAIN_SOURCE = """/* Custom embedded benchmark for the frozen-frontend waveform path. */

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <span>

#include "esp_heap_caps.h"

#include "esp_micro_profiler.h"
#include "frontend_frozen.h"
#include "metrics.h"
#include "model.h"
#include "op_resolver.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using Profiler = ::benchmark::MicroProfiler;

constexpr uint32_t kRandomSeed = 0xFB;
constexpr size_t kTensorArenaSize = 6000000;
constexpr size_t kWaveformBytes = custom_frontend::kWaveformSamples * sizeof(float);

uint8_t* tensor_arena = nullptr;

constexpr uint32_t kCrctabLen = 256;
uint32_t crctab[kCrctabLen];

void GenCRC32Table() {
  constexpr uint32_t kPolyN = 0xEDB88320;
  for (size_t index = 0; index < kCrctabLen; index++) {
    crctab[index] = index;
    for (int i = 0; i < 8; i++) {
      if (crctab[index] & 1) {
        crctab[index] = (crctab[index] >> 1) ^ kPolyN;
      } else {
        crctab[index] >>= 1;
      }
    }
  }
}

uint32_t ComputeCRC32(const uint8_t* data, const size_t data_length) {
  uint32_t crc32 = ~0U;
  for (size_t i = 0; i < data_length; i++) {
    const uint32_t index = (crc32 ^ data[i]) & (kCrctabLen - 1);
    crc32 = (crc32 >> 8) ^ crctab[index];
  }
  crc32 ^= ~0U;
  return crc32;
}

void ShowWaveformCRC32(std::span<const float> waveform) {
  GenCRC32Table();
  const uint32_t crc32_value =
      ComputeCRC32(reinterpret_cast<const uint8_t*>(waveform.data()), waveform.size_bytes());
  MicroPrintf("Audio Input CRC32: 0x%X", crc32_value);
}

void ShowFeatureCRC32(const TfLiteTensor* input) {
  GenCRC32Table();
  const auto* input_values = reinterpret_cast<const uint8_t*>(input->data.int8);
  const uint32_t crc32_value = ComputeCRC32(input_values, input->bytes);
  MicroPrintf("Output Features CRC32: 0x%X", crc32_value);
}

void ShowInterpreterInputCRC32(tflite::MicroInterpreter* interpreter) {
  GenCRC32Table();
  for (size_t i = 0; i < interpreter->inputs_size(); ++i) {
    TfLiteTensor* input = interpreter->input_tensor(i);
    uint8_t* input_values = tflite::GetTensorData<uint8_t>(input);
    uint32_t crc32_value = ComputeCRC32(input_values, input->bytes);
    MicroPrintf("Input CRC32: 0x%X", crc32_value);
  }
}

void ShowInterpreterOutputCRC32(tflite::MicroInterpreter* interpreter) {
  GenCRC32Table();
  for (size_t i = 0; i < interpreter->outputs_size(); ++i) {
    TfLiteTensor* output = interpreter->output_tensor(i);
    uint8_t* output_values = tflite::GetTensorData<uint8_t>(output);
    uint32_t crc32_value = ComputeCRC32(output_values, output->bytes);
    MicroPrintf("Output CRC32: 0x%X", crc32_value);
  }
}

void SetRandomWaveform(uint32_t random_seed, std::span<float> waveform) {
  std::mt19937 eng(random_seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& sample : waveform) {
    sample = dist(eng);
  }
}

[[nodiscard]] int Benchmark(const uint8_t* model_data) {
  static Profiler profiler;
  uint32_t seed = kRandomSeed;
  TfLiteStatus status;

  tensor_arena = reinterpret_cast<uint8_t*>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM));
  auto* waveform_buffer = static_cast<float*>(
      heap_caps_malloc(kWaveformBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr || waveform_buffer == nullptr) {
    MicroPrintf("Allocation failed");
    return -1;
  }

  auto waveform = std::span<float>(waveform_buffer, custom_frontend::kWaveformSamples);
  custom_frontend::FrozenFrontend frontend;

  uint32_t event_handle = profiler.BeginEvent("FrontendInit");
  if (frontend.Init() != ESP_OK) {
    MicroPrintf("Frontend init failed");
    profiler.EndEvent(event_handle);
    return -1;
  }
  profiler.EndEvent(event_handle);

  event_handle = profiler.BeginEvent("tflite::GetModel");
  const tflite::Model* model = tflite::GetModel(model_data);
  profiler.EndEvent(event_handle);

  event_handle = profiler.BeginEvent("tflite::CreateOpResolver");
  TflmOpResolver op_resolver;
  status = CreateOpResolver(op_resolver);
  profiler.EndEvent(event_handle);
  if (status != kTfLiteOk) {
    MicroPrintf("CreateOpResolver failed");
    return -1;
  }

  event_handle = profiler.BeginEvent("tflite::MicroInterpreter instantiation");
  tflite::RecordingMicroInterpreter interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
  profiler.EndEvent(event_handle);

  event_handle = profiler.BeginEvent("tflite::MicroInterpreter::AllocateTensors");
  status = interpreter.AllocateTensors();
  profiler.EndEvent(event_handle);
  if (status != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed");
    return -1;
  }

  profiler.LogTicksPerTagCsv();
  MicroPrintf("");
  profiler.ClearEvents();
  MicroPrintf("");

  SetRandomWaveform(seed, waveform);
  ShowWaveformCRC32(waveform);

  TfLiteTensor* input = interpreter.input_tensor(0);
  if (frontend.ExtractToModelInput(waveform, input, &profiler) != ESP_OK) {
    MicroPrintf("Frontend extraction failed");
    return -1;
  }

  MicroPrintf("");
  profiler.LogTicksPerTagCsv();
  MicroPrintf("");
  profiler.ClearEvents();
  ShowFeatureCRC32(input);
  ShowInterpreterInputCRC32(&interpreter);
  MicroPrintf("");

  status = interpreter.Invoke();
  if (status != kTfLiteOk) {
    MicroPrintf("Model interpreter invocation failed: %d", status);
    return -1;
  }

  profiler.LogTicksPerTagCsv();
  MicroPrintf("");
  profiler.ClearEvents();
  ShowInterpreterOutputCRC32(&interpreter);
  MicroPrintf("");
  interpreter.GetMicroAllocator().PrintAllocations();
  return 0;
}

}  // namespace
}  // namespace tflite

extern "C" void app_main() {
  MicroPrintf("\\nConfigured arena size = %d\\n", tflite::kTensorArenaSize);
  auto res = tflite::Benchmark(g_model);
  MicroPrintf("Result=%d", res);
}
"""


def write_frontend_sources(main_dir: Path) -> None:
    (main_dir / "frontend_frozen.h").write_text(FRONTEND_HEADER, encoding="utf-8")
    (main_dir / "frontend_frozen.cpp").write_text(FRONTEND_SOURCE, encoding="utf-8")
    (main_dir / "main.cpp").write_text(MAIN_SOURCE, encoding="utf-8")

    cmake = """idf_component_register(
    SRCS main.cpp metrics.cpp model.cpp esp_micro_profiler.cpp frontend_frozen.cpp frontend_frozen_data.cpp
    PRIV_REQUIRES spi_flash driver esp_timer
    INCLUDE_DIRS "")

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits)
"""
    (main_dir / "CMakeLists.txt").write_text(cmake, encoding="utf-8")


def write_readme() -> None:
    text = f"""Custom embedded source tree for the frozen-frontend waveform submission.

This `src/` project targets the chosen embedded candidate:

- model: `{MODEL_TFLITE.name}`
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
"""
    (SUBMISSION_DIR / "your_generated_code" / "README.md").write_text(text, encoding="utf-8")


def generate() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    shutil.copytree(TEMPLATE_DIR, OUTPUT_DIR, dirs_exist_ok=False)

    main_dir = OUTPUT_DIR / "main"
    write_model_cpp(main_dir, MODEL_TFLITE)
    write_frontend_data(main_dir)
    write_frontend_sources(main_dir)

    if MODEL_TFLITE.exists():
        shutil.copy2(MODEL_TFLITE, SUBMISSION_EMBEDDED_MODEL)
    if MODEL_METADATA.exists():
        shutil.copy2(MODEL_METADATA, SUBMISSION_EMBEDDED_METADATA)

    write_readme()


if __name__ == "__main__":
    generate()
