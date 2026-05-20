#include "frontend_frozen.h"

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
