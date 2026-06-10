/* Custom embedded benchmark for the frozen-frontend streaming classifier path. */

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using Profiler = ::benchmark::MicroProfiler;

constexpr uint32_t kRandomSeed = 0xFB;
constexpr size_t kBackboneArenaSize = 1000000;
constexpr size_t kStepArenaSize = 100000;
constexpr size_t kWaveformBytes = custom_frontend::kWaveformSamples * sizeof(float);
constexpr int kNumFrames = 301;
constexpr int kNumFilters = 64;
constexpr int kEmbeddingDim = 32;
constexpr int kHiddenDim = 70;
constexpr int kNumClasses = 11;

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

void ShowBufferCRC32(const char* label, const void* data, size_t bytes) {
  GenCRC32Table();
  const uint32_t crc32_value =
      ComputeCRC32(reinterpret_cast<const uint8_t*>(data), bytes);
  MicroPrintf("%s CRC32: 0x%X", label, crc32_value);
}

void ShowInterpreterInputCRC32(tflite::MicroInterpreter* interpreter) {
  for (size_t i = 0; i < interpreter->inputs_size(); ++i) {
    TfLiteTensor* input = interpreter->input_tensor(i);
    ShowBufferCRC32("Input", tflite::GetTensorData<uint8_t>(input), input->bytes);
  }
}

void ShowLogits(std::span<const float> logits) {
  int best_index = 0;
  float best_value = logits[0];
  for (int index = 0; index < static_cast<int>(logits.size()); ++index) {
    MicroPrintf("Logit[%d]: %.6f", index, static_cast<double>(logits[index]));
    if (logits[index] > best_value) {
      best_value = logits[index];
      best_index = index;
    }
  }
  MicroPrintf("Predicted index: %d", best_index);
}

void SetRandomWaveform(uint32_t random_seed, std::span<float> waveform) {
  std::mt19937 eng(random_seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& sample : waveform) {
    sample = dist(eng);
  }
}

bool CheckTensorShape(const TfLiteTensor* tensor, TfLiteType type, std::span<const int> dims) {
  if (tensor == nullptr || tensor->type != type || tensor->dims == nullptr ||
      tensor->dims->size != static_cast<int>(dims.size())) {
    return false;
  }
  for (int i = 0; i < tensor->dims->size; ++i) {
    if (tensor->dims->data[i] != dims[i]) {
      return false;
    }
  }
  return true;
}

float* AllocateFloatBuffer(size_t elements) {
  return static_cast<float*>(
      heap_caps_malloc(elements * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
}

[[nodiscard]] int RunBackbone(
    std::span<float> waveform,
    std::span<float> embeddings,
    custom_frontend::FrozenFrontend* frontend,
    Profiler* profiler) {
  TfLiteStatus status;
  auto* tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kBackboneArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr) {
    MicroPrintf("Backbone arena allocation failed");
    return -1;
  }

  uint32_t event_handle = profiler->BeginEvent("backbone::GetModel");
  const tflite::Model* model = tflite::GetModel(g_backbone_model);
  profiler->EndEvent(event_handle);

  event_handle = profiler->BeginEvent("backbone::CreateOpResolver");
  TflmOpResolver op_resolver;
  status = CreateOpResolver(op_resolver);
  profiler->EndEvent(event_handle);
  if (status != kTfLiteOk) {
    MicroPrintf("Backbone CreateOpResolver failed");
    heap_caps_free(tensor_arena);
    return -1;
  }

  {
    event_handle = profiler->BeginEvent("backbone::Interpreter");
    tflite::RecordingMicroInterpreter interpreter(
        model, op_resolver, tensor_arena, kBackboneArenaSize, nullptr, nullptr);
    profiler->EndEvent(event_handle);

    event_handle = profiler->BeginEvent("backbone::AllocateTensors");
    status = interpreter.AllocateTensors();
    profiler->EndEvent(event_handle);
    if (status != kTfLiteOk) {
      MicroPrintf("Backbone AllocateTensors failed");
      heap_caps_free(tensor_arena);
      return -1;
    }

    TfLiteTensor* input = interpreter.input_tensor(0);
    const int input_dims[] = {1, kNumFrames, kNumFilters};
    if (!CheckTensorShape(input, kTfLiteFloat32, input_dims)) {
      MicroPrintf("Unexpected backbone input tensor");
      heap_caps_free(tensor_arena);
      return -1;
    }

    MicroPrintf("");
    profiler->LogTicksPerTagCsv();
    MicroPrintf("");
    profiler->ClearEvents();

    if (frontend->ExtractToModelInput(waveform, input, profiler) != ESP_OK) {
      MicroPrintf("Frontend extraction failed");
      heap_caps_free(tensor_arena);
      return -1;
    }

    MicroPrintf("");
    profiler->LogTicksPerTagCsv();
    MicroPrintf("");
    profiler->ClearEvents();
    ShowBufferCRC32("Output Features", tflite::GetTensorData<uint8_t>(input), input->bytes);
    ShowInterpreterInputCRC32(&interpreter);

    event_handle = profiler->BeginEvent("backbone::Invoke");
    status = interpreter.Invoke();
    profiler->EndEvent(event_handle);
    if (status != kTfLiteOk) {
      MicroPrintf("Backbone invocation failed: %d", status);
      heap_caps_free(tensor_arena);
      return -1;
    }

    TfLiteTensor* output = interpreter.output_tensor(0);
    const int output_dims[] = {1, kNumFrames, kEmbeddingDim};
    if (!CheckTensorShape(output, kTfLiteFloat32, output_dims) ||
        output->bytes != embeddings.size_bytes()) {
      MicroPrintf("Unexpected backbone output tensor");
      heap_caps_free(tensor_arena);
      return -1;
    }
    std::memcpy(embeddings.data(), output->data.f, output->bytes);

    ShowBufferCRC32("Backbone Output", embeddings.data(), embeddings.size_bytes());
    MicroPrintf("");
    interpreter.GetMicroAllocator().PrintAllocations();
  }

  heap_caps_free(tensor_arena);
  return 0;
}

[[nodiscard]] int RunStreamingStepLoop(
    std::span<const float> embeddings,
    std::span<float> final_logits,
    Profiler* profiler) {
  TfLiteStatus status;
  auto* tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kStepArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  float* hidden = AllocateFloatBuffer(kHiddenDim);
  float* att_num = AllocateFloatBuffer(kHiddenDim);
  float* att_den = AllocateFloatBuffer(1);
  float* att_max = AllocateFloatBuffer(1);
  if (tensor_arena == nullptr || hidden == nullptr || att_num == nullptr ||
      att_den == nullptr || att_max == nullptr) {
    MicroPrintf("Streaming step allocation failed");
    heap_caps_free(tensor_arena);
    heap_caps_free(hidden);
    heap_caps_free(att_num);
    heap_caps_free(att_den);
    heap_caps_free(att_max);
    return -1;
  }
  std::fill(hidden, hidden + kHiddenDim, 0.0f);
  std::fill(att_num, att_num + kHiddenDim, 0.0f);
  att_den[0] = 0.0f;
  att_max[0] = -1.0e9f;

  const tflite::Model* model = tflite::GetModel(g_streaming_step_model);

  TflmOpResolver op_resolver;
  status = CreateOpResolver(op_resolver);
  if (status != kTfLiteOk) {
    MicroPrintf("Step CreateOpResolver failed");
    heap_caps_free(tensor_arena);
    heap_caps_free(hidden);
    heap_caps_free(att_num);
    heap_caps_free(att_den);
    heap_caps_free(att_max);
    return -1;
  }

  {
    tflite::RecordingMicroInterpreter interpreter(
        model, op_resolver, tensor_arena, kStepArenaSize, nullptr, nullptr);

    status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
      MicroPrintf("Step AllocateTensors failed");
      heap_caps_free(tensor_arena);
      heap_caps_free(hidden);
      heap_caps_free(att_num);
      heap_caps_free(att_den);
      heap_caps_free(att_max);
      return -1;
    }

    const int frame_dims[] = {1, kEmbeddingDim};
    const int hidden_dims[] = {1, kHiddenDim};
    const int scalar_dims[] = {1, 1};
    if (!CheckTensorShape(interpreter.input_tensor(0), kTfLiteFloat32, frame_dims) ||
        !CheckTensorShape(interpreter.input_tensor(1), kTfLiteFloat32, hidden_dims) ||
        !CheckTensorShape(interpreter.input_tensor(2), kTfLiteFloat32, hidden_dims) ||
        !CheckTensorShape(interpreter.input_tensor(3), kTfLiteFloat32, scalar_dims) ||
        !CheckTensorShape(interpreter.input_tensor(4), kTfLiteFloat32, scalar_dims)) {
      MicroPrintf("Unexpected step input tensors");
      heap_caps_free(tensor_arena);
      heap_caps_free(hidden);
      heap_caps_free(att_num);
      heap_caps_free(att_den);
      heap_caps_free(att_max);
      return -1;
    }

    uint32_t event_handle = profiler->BeginEvent("step::InvokeLoop");
    for (int frame_index = 0; frame_index < kNumFrames; ++frame_index) {
      std::memcpy(interpreter.input_tensor(0)->data.f,
                  embeddings.data() + frame_index * kEmbeddingDim,
                  kEmbeddingDim * sizeof(float));
      std::memcpy(interpreter.input_tensor(1)->data.f, hidden, kHiddenDim * sizeof(float));
      std::memcpy(interpreter.input_tensor(2)->data.f, att_num, kHiddenDim * sizeof(float));
      std::memcpy(interpreter.input_tensor(3)->data.f, att_den, sizeof(float));
      std::memcpy(interpreter.input_tensor(4)->data.f, att_max, sizeof(float));

      status = interpreter.Invoke();
      if (status != kTfLiteOk) {
        profiler->EndEvent(event_handle);
        MicroPrintf("Step invocation failed at frame %d: %d", frame_index, status);
        heap_caps_free(tensor_arena);
        heap_caps_free(hidden);
        heap_caps_free(att_num);
        heap_caps_free(att_den);
        heap_caps_free(att_max);
        return -1;
      }

      std::memcpy(hidden, interpreter.output_tensor(0)->data.f, kHiddenDim * sizeof(float));
      std::memcpy(att_num, interpreter.output_tensor(1)->data.f, kHiddenDim * sizeof(float));
      std::memcpy(att_den, interpreter.output_tensor(2)->data.f, sizeof(float));
      std::memcpy(att_max, interpreter.output_tensor(3)->data.f, sizeof(float));
      std::memcpy(final_logits.data(), interpreter.output_tensor(4)->data.f,
                  kNumClasses * sizeof(float));
    }
    profiler->EndEvent(event_handle);

    profiler->LogTicksPerTagCsv();
    MicroPrintf("");
    profiler->ClearEvents();
    ShowBufferCRC32("Output", final_logits.data(), final_logits.size_bytes());
    ShowBufferCRC32("FinalLogits", final_logits.data(), final_logits.size_bytes());
    ShowLogits(final_logits);
    MicroPrintf("");
    MicroPrintf("Step arena allocation is smaller than backbone and is omitted from parser RAM.");
  }

  heap_caps_free(tensor_arena);
  heap_caps_free(hidden);
  heap_caps_free(att_num);
  heap_caps_free(att_den);
  heap_caps_free(att_max);
  return 0;
}

[[nodiscard]] int Benchmark() {
  static Profiler profiler;

  auto* waveform_buffer = static_cast<float*>(
      heap_caps_malloc(kWaveformBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  float* embeddings = AllocateFloatBuffer(kNumFrames * kEmbeddingDim);
  float* final_logits = AllocateFloatBuffer(kNumClasses);
  if (waveform_buffer == nullptr || embeddings == nullptr || final_logits == nullptr) {
    MicroPrintf("Benchmark buffer allocation failed");
    heap_caps_free(waveform_buffer);
    heap_caps_free(embeddings);
    heap_caps_free(final_logits);
    return -1;
  }

  auto waveform = std::span<float>(waveform_buffer, custom_frontend::kWaveformSamples);
  auto embedding_span = std::span<float>(embeddings, kNumFrames * kEmbeddingDim);
  auto logits_span = std::span<float>(final_logits, kNumClasses);
  custom_frontend::FrozenFrontend frontend;

  uint32_t event_handle = profiler.BeginEvent("FrontendInit");
  if (frontend.Init() != ESP_OK) {
    MicroPrintf("Frontend init failed");
    profiler.EndEvent(event_handle);
    heap_caps_free(waveform_buffer);
    heap_caps_free(embeddings);
    heap_caps_free(final_logits);
    return -1;
  }
  profiler.EndEvent(event_handle);

  MicroPrintf("");
  MicroPrintf("=== Embedded benchmark input ===");
  MicroPrintf("Input source: deterministic random waveform");
  MicroPrintf("Random seed: 0x%X", kRandomSeed);
  SetRandomWaveform(kRandomSeed, waveform);
  ShowBufferCRC32("Audio Input", waveform.data(), waveform.size_bytes());

  int result = RunBackbone(waveform, embedding_span, &frontend, &profiler);
  if (result == 0) {
    result = RunStreamingStepLoop(embedding_span, logits_span, &profiler);
  }

  heap_caps_free(waveform_buffer);
  heap_caps_free(embeddings);
  heap_caps_free(final_logits);
  return result;
}

}  // namespace
}  // namespace tflite

extern "C" void app_main() {
  MicroPrintf("\nBackbone arena size = %d", tflite::kBackboneArenaSize);
  MicroPrintf("Step arena size = %d\n", tflite::kStepArenaSize);
  auto res = tflite::Benchmark();
  MicroPrintf("Result=%d", res);
}
