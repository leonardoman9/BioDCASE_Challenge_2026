#pragma once

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
