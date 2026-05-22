#pragma once

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
extern const float kFilterWeights[1010];

}  // namespace custom_frontend
