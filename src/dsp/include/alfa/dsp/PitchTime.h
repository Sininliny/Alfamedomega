#pragma once

// Pitch and time: the PITCH SHIFT family (real-time, instance slots), the phase-vocoder frame
// helper for real-time bin control, and offline phase-vocoder tools that are never used on the
// audio path.

#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/SpectralStft.h"

#include <array>
#include <cstddef>
#include <vector>

namespace alfa::dsp
{

/// Real-time-safe phase-vocoder frame (magnitude / instantaneous frequency analysis and phase
/// accumulating synthesis), for use inside a detail::SpectralStft frame callback.
using PhaseVocoderFrame = detail::PhaseVocoderFrame;

/// PITCH SHIFT family (Family::PitchShift, instance slots, stereo): a delay-line pitch shifter.
///
/// Two Hann grains (sin^2 windows at 50% overlap, so their weights sum to 1) read a delay line
/// at the pitch ratio p. A grain for p > 1 starts at a delay of about (p - 1) * grain length and
/// catches up with the write head as it plays; for p < 1 it starts near the write head and falls
/// behind. When a grain starts, its read position is aligned to the grain that is fading out by
/// normalised cross-correlation over the last few milliseconds (search window 12 ms), so periodic
/// input stays phase continuous across the crossfade: a sinusoid comes out at exactly p * f
/// instead of the sideband comb of a free-running shifter. With "align" off the grains start
/// at fixed delays (the classic dual-tap shifter).
///
/// Latency varies with the settings; latencySamples() reports the mean delay of the wet path:
/// margin + search / 2 + |p - 1| * grain / 2 (about 6.5 ms at p = 1, 17 ms at +7 semitones with
/// 50 ms grains). The dry path is not delayed.
///
/// Feedback sends the (tanh-limited, low-passed) wet signal back into the delay line, so each
/// pass is shifted again (shimmer); it is capped at 0.9 and the tanh keeps the loop bounded.
///
/// Params (slot param i = entry i):
///    0 shift          -24 .. 24 semitones (0)
///    1 fine           -100 .. 100 cents (0)
///    2 grain          10 .. 100 ms, log (50)
///    3 feedback       0 .. 0.9 (0)
///    4 feedback tone  500 .. 20000 Hz, log (20000): one-pole low-pass in the feedback path
///    5 spread         0 .. 50 cents (0): left detuned down, right up by this amount
///    6 align          Toggle (on): correlation-aligned grain starts
///    7 mix            0 .. 1 (1)
///    8 output gain    -24 .. 24 dB (0)
///    9..15 unused
class PitchShiftProcessor final : public Processor
{
public:
    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;
    /// Mean delay of the wet path for the current settings.
    [[nodiscard]] int latencySamples() const noexcept;

private:
    struct Grain
    {
        double read = 0.0; // absolute input time being read
        int age = 0;
        int length = 0;
        bool active = false;
    };

    struct Channel
    {
        std::vector<float> buffer;
        std::array<Grain, 2> grains;
        int next = 0;      // grain slot started next
        int untilNext = 0; // samples until the next grain start
        double semitones = 0.0;
        float feedback = 0.0f; // low-passed feedback signal
    };

    void startGrain(Channel& c, double ratio) noexcept;
    [[nodiscard]] float read(const Channel& c, double position) const noexcept;
    [[nodiscard]] double alignedStart(const Channel& c, const Grain& other, double minDelay) const noexcept;

    std::array<Channel, 2> ch_;
    double sampleRate_ = 48000.0;
    int channels_ = 2;
    int mask_ = 0;
    long long time_ = 0; // absolute sample index of the next write
    int margin_ = 4;
    int searchRange_ = 0;
    int corrLength_ = 0;
    int corrStride_ = 1;

    // targets from setParams, smoothed in process()
    double shiftTarget_ = 0.0; // semitones including fine
    double spreadTarget_ = 0.0;
    double grainTarget_ = 0.05; // seconds
    double grainSeconds_ = 0.05;
    float feedbackTarget_ = 0.0f;
    float feedbackAmount_ = 0.0f;
    float toneCoeff_ = 1.0f;
    bool align_ = true;
    float mixTarget_ = 1.0f;
    float mix_ = 1.0f;
    float gainTarget_ = 1.0f;
    float gain_ = 1.0f;
};

/// Offline tools. They allocate and are meant for rendering, analysis and training data, never
/// for the audio thread.
namespace offline
{

/// Phase-vocoder time stretch with identity phase locking: the output has round(n * ratio)
/// samples at the input's pitch. ratio is clamped to [0.25, 8]; fftSize is rounded to a power of
/// two in [256, 16384] (2048 suits 44.1 / 48 kHz material).
[[nodiscard]] std::vector<float> timeStretch(const float* in, std::size_t n, double ratio,
                                             int fftSize = 2048);

/// Windowed-sinc (Kaiser, 24 zero crossings) resampler: out[i] = x(i * step) for i < outLength,
/// band-limited to min(1, 1 / step) of the input Nyquist. A stand-in until the shared
/// high-quality resampler (Resample.h) is available.
[[nodiscard]] std::vector<float> resample(const float* in, std::size_t n, double step, std::size_t outLength);

/// Pitch shift keeping duration: time stretch by 2^(semitones / 12), then resample back to n
/// samples.
[[nodiscard]] std::vector<float> pitchShiftOffline(const float* in, std::size_t n, double semitones,
                                                   int fftSize = 2048);

} // namespace offline

} // namespace alfa::dsp
