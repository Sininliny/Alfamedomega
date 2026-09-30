#pragma once

#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/NonlinearOversampler.h"

#include <array>

namespace alfa::dsp
{

/// Family::Nonlinear: saturation, clipping, folding, rectification and bit / rate reduction in
/// one processor, with a continuous shape selector so the network can interpolate between
/// shapes.
///
/// Signal path per channel:
///   dry --------------------------------------------------(delay = latency)---+
///   in -> [up 1/2/4x] -> drive * x + bias -> shape(type) - shape(bias) -> [down] -> DC block
///      -> auto gain * trim -> sample hold -> bit quantiser -> tilt tone -> mix -+-> out
///
/// Shapes, in order along "type" (fractional values crossfade the two neighbours), each
/// bounded to [-1, 1]:
///   0 soft saturation (rational tanh), 1 asymmetric tube (the negative side saturates earlier,
///   so even harmonics), 2 cubic soft clip, 3 hard clip, 4 sine folder, 5 triangle folder
///   (Buchla-style parallel folding cells), 6 half-wave and 7 full-wave rectifier (of the soft
///   saturated signal, so they stay bounded at any drive).
/// "character" means: tube asymmetry, sine-folder fold gain, triangle-folder cell count; the
/// other shapes ignore it.
///
/// The static offset shape(bias) is subtracted inside the shaper so a constant bias does not
/// step the output; the asymmetric DC that remains is removed by a 10 Hz DC blocker.
///
/// Oversampling wraps only the shaper (the bit / rate reduction is meant to alias). The FIR
/// halfbands give an integer latency of 0 / 23 / 29 base samples for 1x / 2x / 4x at any
/// sample rate; the dry path is delayed to match, so a partial mix never combs. Changing the
/// factor fades the output out (32 samples), mutes while the new filters fill (2x latency)
/// and fades back in (32 samples).
class NonlinearProcessor final : public Processor
{
public:
    enum Param
    {
        kDrive = 0,    ///< 0..48 dB input gain into the shaper
        kType,         ///< 0..7 shape morph (see above)
        kBias,         ///< -1..1 offset added after the drive, before the shaper
        kCharacter,    ///< 0..1 per-shape character
        kAutoGain,     ///< 0..1 amount of automatic drive compensation
        kTrim,         ///< -24..+12 dB output trim
        kBits,         ///< 1..24 bits (continuous; 24 = off)
        kDownsample,   ///< 1..64 sample-hold factor (fractional; 1 = off)
        kOversampling, ///< Choice 1x / 2x / 4x
        kTone,         ///< -1..1 tilt around 1 kHz: highs +-6 dB, lows -+6 dB (0 = flat)
        kMix,          ///< 0..1 dry / wet
        kParamCount
    };

    static constexpr int kTypeCount = 8;

    static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;
    /// Reduced quality caps oversampling at 1x (switching fades, as a param change does).
    void setQuality(int level) noexcept override { quality_ = level; }
    [[nodiscard]] bool hasReducedQuality() const noexcept override { return true; }

    /// Base-rate latency of the current oversampling factor (0, 23 or 29 samples).
    [[nodiscard]] int latencySamples() const noexcept;

    /// For tests: bypass the DC blocker to measure the shaper's DC content.
    void setDcBlockerEnabled(bool enabled) noexcept { dcBlock_ = enabled; }

    /// The memoryless transfer curve (no oversampling, gain, DC removal), for tests and plots.
    [[nodiscard]] static float shape(float x, float type, float character) noexcept;

private:
    // Half-kernel sizes of the two halfband stages (NonlinearOversampler.h): stage 1 (base <->
    // 2x) has 4 * 11 + 3 = 47 taps, stage 2 (2x <-> 4x) 23 taps.
    static constexpr int kStage1K = 11;
    static constexpr int kStage2K = 5;

    struct Ramp
    {
        float cur = 0.0f;
        float target = 0.0f;
        float step = 0.0f;
        void begin(int n) noexcept { step = (target - cur) / static_cast<float>(n); }
        void snap() noexcept
        {
            cur = target;
            step = 0.0f;
        }
    };

    struct Channel
    {
        detail::NonlinearUpsampler2x<kStage1K> up1;
        detail::NonlinearUpsampler2x<kStage2K> up2;
        detail::NonlinearDownsampler2x<kStage1K> down1;
        detail::NonlinearDownsampler2x<kStage2K> down2;
        float align = 0.0f; // 1 sample at 2x rate, makes the 4x latency integer
        float dcX = 0.0f, dcY = 0.0f;
        float holdPhase = 1.0f, held = 0.0f;
        float toneLp = 0.0f;
        std::array<float, 32> dry{};
        int dryPos = 0;
    };

    void setFactor(int factor) noexcept;
    void clearChannel(Channel& c) noexcept;
    /// Per-sample values shared by both channels.
    struct Sample
    {
        float drive = 1.0f, typeFrac = 0.0f, bias = 0.0f, character = 0.0f, offset = 0.0f;
        int typeIndex = 0;
        float gain = 1.0f, toneLow = 1.0f, toneHigh = 1.0f, mix = 1.0f, fade = 1.0f;
        float q = 1.0f, qLo = 0.0f, qHi = 0.0f;
        bool crush = false, toneOn = false;
    };

    [[nodiscard]] static float shaperAt(float x, const Sample& s) noexcept;
    [[nodiscard]] float tick(Channel& c, float x, const Sample& s) const noexcept;

    double sampleRate_ = 48000.0;
    int channels_ = 1;
    detail::NonlinearHalfband stage1_, stage2_;
    std::array<Channel, 2> ch_{};

    Ramp drive_, type_, bias_, character_, gain_, toneLow_, toneHigh_, mix_;
    float bits_ = 24.0f;
    float holdIncrement_ = 1.0f;
    int factor_ = 1;
    int pendingFactor_ = 1;
    int quality_ = 0;
    float fade_ = 1.0f; // output fade while switching the oversampling factor
    int mute_ = 0;      // samples to stay silent after a switch while the new filters fill
    bool first_ = true;
    bool dcBlock_ = true;
    float dcR_ = 0.999f;
    float toneA_ = 0.1f;
};

} // namespace alfa::dsp
