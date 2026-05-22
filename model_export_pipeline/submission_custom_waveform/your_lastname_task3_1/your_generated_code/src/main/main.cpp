/* Custom embedded benchmark for the frozen-frontend waveform path. */

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
  MicroPrintf("\nConfigured arena size = %d\n", tflite::kTensorArenaSize);
  auto res = tflite::Benchmark(g_model);
  MicroPrintf("Result=%d", res);
}
