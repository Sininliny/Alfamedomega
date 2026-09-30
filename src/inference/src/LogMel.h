#pragma once

// Log-mel spectrogram matching training's logmel() (alfamedomega_train.py): 48 kHz mono,
// n_fft 2048, hop 512, periodic Hann, centred frames with reflect padding (torch.stft
// defaults), magnitude spectrum, 64 triangular HTK-mel filters from 20 Hz to 16 kHz,
// log(mel + 1e-5). Message thread only.

#include <cstddef>
#include <span>
#include <vector>

namespace alfa::inference
{

inline constexpr std::size_t kMelBands = 64;
inline constexpr std::size_t kMelFftSize = 2048;
inline constexpr std::size_t kMelHop = 512;

/// Returns [kMelBands x frames] row-major (band-major, like the model's [1, 64, frames] input),
/// or an empty vector if the audio is shorter than half an FFT (reflect padding needs more).
[[nodiscard]] std::vector<float> logMel(std::span<const float> mono48k, std::size_t& frames);

} // namespace alfa::inference
