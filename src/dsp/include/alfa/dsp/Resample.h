#pragma once

// Resampling utilities, in two clearly separated groups:
//
//   Real-time safe (audio thread, after prepare()):
//     PolyphaseResampler  streaming Kaiser-windowed-sinc resampler with a fixed or variable
//                         ratio, and random-access reads of a buffer (sample playback).
//
//   Offline (allocates; NEVER call from the audio thread):
//     resampleOffline     whole-buffer high-quality conversion (sample import, IR banks).
//
// Both evaluate the band-limited interpolation y(t) = sum_j x_j s h(s (t - j)) with h a
// Kaiser-windowed sinc and s = min(1, outRate / inRate): when downsampling the kernel is
// stretched by 1 / s so its cutoff follows the output Nyquist (anti-aliasing), at the cost of
// proportionally more taps.

#include <cstdint>
#include <vector>

namespace alfa::dsp
{

/// Streaming resampler. ratio = input samples consumed per output sample
/// (inRate / outRate for rate conversion, the playback speed for pitch shifting).
///
/// Kernel: `taps` points (at ratio <= 1), Kaiser beta, cutoff placed so the transition band ends
/// at Nyquist: with the default 64 taps / beta 8 the passband is flat (+-0.001 dB) to about
/// 0.42 fs and the stopband is below -80 dB from 0.5 fs. The kernel is tabulated at 512 points
/// per zero crossing and linearly interpolated (table error below -120 dB).
/// Timing: output m is aligned with input time m * ratio (an impulse at input 0 peaks at output
/// 0), but it is only produced once latency() further inputs have been pushed: taps / 2 input
/// samples of look-ahead (times the ratio when it is above 1).
class PolyphaseResampler
{
public:
    /// Allocates. taps is rounded up to an even number in 8..256; maxRatio (>= 1) bounds
    /// setRatio and sizes the history for the stretched kernel.
    void prepare(int taps = 64, double maxRatio = 4.0, double kaiserBeta = 8.0);
    void reset() noexcept;

    /// Clamped to [1 / 64, maxRatio]. Takes effect from the next output sample (no glitch: the
    /// read position is continuous).
    void setRatio(double ratio) noexcept;
    [[nodiscard]] double ratio() const noexcept { return ratio_; }

    /// Consumes up to numIn inputs and produces up to maxOut outputs; returns the number of
    /// outputs written, and sets consumed to the number of inputs used. Stops when either the
    /// input is exhausted or the output is full; unconsumed inputs must be passed again.
    int process(const float* in, int numIn, float* out, int maxOut, int& consumed) noexcept;

    /// Random access: band-limited value of data at fractional index position (zeros outside
    /// [0, length)), with the kernel stretched for the given playback ratio (> 1 reads faster
    /// and needs anti-aliasing). Does not touch the streaming state.
    [[nodiscard]] float sampleAt(const float* data, std::int64_t length, double position,
                                 double ratio = 1.0) const noexcept;

    /// Look-ahead in input samples (see the class comment).
    [[nodiscard]] double latency() const noexcept;

    [[nodiscard]] int taps() const noexcept { return 2 * half_; }

private:
    [[nodiscard]] float kernel(float t) const noexcept; ///< h(t) for |t| < half_, 0 outside
    [[nodiscard]] float evaluate() const noexcept;

    std::vector<float> table_;   // h at t = i / kResolution, i = 0 .. half_ * kResolution (+ guard)
    std::vector<float> history_; // input ring (power of two)
    std::size_t mask_ = 0;
    int half_ = 32;
    double maxRatio_ = 4.0;
    double ratio_ = 1.0;
    std::int64_t received_ = 0;  // inputs pushed so far (index of the next input)
    std::int64_t baseIndex_ = 0; // integer part of the next output time (input index units)
    double frac_ = 0.0;          // fractional part, in [0, 1)
};

/// Offline, allocating, NOT real-time safe. Converts n samples at fromRate to toRate
/// (output length round(n * toRate / fromRate)); the signal is treated as zero outside the
/// buffer, and output sample m is aligned with input time m * fromRate / toRate.
/// quality 0..3 selects the Kaiser-windowed sinc: 32 taps / beta 6, 64 / 8, 128 / 10,
/// 256 / 12 (quality 3, the default, gives > 100 dB SNR and a passband to about 0.47 of the
/// lower rate); values outside the range are clamped.
[[nodiscard]] std::vector<float> resampleOffline(const float* in, std::size_t n, double fromRate,
                                                 double toRate, int quality = 3);

} // namespace alfa::dsp
