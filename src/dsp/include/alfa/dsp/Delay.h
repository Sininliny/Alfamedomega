#pragma once

// Delay-based processing families: feedback delay, modulated short delay (chorus / flanger /
// vibrato), phaser and FDN reverb. Building blocks live in detail/DelayLine.h,
// detail/DelayAllpass.h, detail/DelayInterpolation.h and detail/DelayFdn.h.
//
// Mix convention: out = (1 - mix) dry + mix wet. Delay and Reverb default to mix 1 (wet only)
// because the slot graph already carries the dry signal (dryGain); ModDelay and Phaser default
// to 0.5 because their effect is the interference between dry and wet.

#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/DelayAllpass.h"
#include "alfa/dsp/detail/DelayFdn.h"
#include "alfa/dsp/detail/DelayLine.h"

#include <array>

namespace alfa::dsp
{

/// Family::Delay (instance slots, stereo). Feedback delay with per-channel times, ping-pong,
/// band-limited and softly saturated feedback, tape-style wow and an optional allpass diffuser
/// inside the loop.
///
/// Time changes: small changes glide (one-pole, 60 ms, rate-limited to +-25% speed: the read
/// head moves like tape and bends the pitch briefly); jumps larger than 20 ms, or targets that
/// run away faster than the glide can follow, crossfade between the old and new read position
/// over 30 ms. Either way the output derivative stays bounded. The diffuser's delay is taken out
/// of the main read, so echoes land at the set time for any diffusion.
class DelayProcessor final : public Processor
{
public:
    enum Param
    {
        kTimeL,
        kTimeR,
        kLink,
        kFeedback,
        kPingPong,
        kLowCut,
        kHighCut,
        kModDepth,
        kModRate,
        kDiffusion,
        kMix,
    };

    static constexpr float kMaxSeconds = 2.0f;
    static constexpr float kMaxModMs = 5.0f;

    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;

private:
    struct Channel
    {
        detail::DelayLine line;
        detail::SchroederAllpass diffuser;
        detail::OnePole lowpass;
        detail::OnePole highpass; // used as x - lowpass(x)
        // Delays in samples, in double so slow glides on 2 s lines stay smooth and exact.
        double target = 0.0;
        double cur = 0.0;     // active read head
        double next = 0.0;    // incoming head while crossfading
        double apDelay = 0.0; // diffuser delay, subtracted from the main read
        double apTarget = 0.0;
        double apBase = 0.0;
        float fade = 0.0f; // 0..1 crossfade progress
        bool fading = false;
        float lfoOffset = 0.0f;
    };

    [[nodiscard]] float readHeads(Channel& c, float mod) noexcept;

    std::array<Channel, 2> ch_{};
    double sampleRate_ = 48000.0;
    int channels_ = 1;
    double glide_ = 0.0;
    double jump_ = 0.0;
    float fadeInc_ = 0.0f;
    float lfoPhase_ = 0.0f;
    float lfoInc_ = 0.0f;
    // Per-block linear ramps (value at block start, step per sample).
    float fb_ = 0.0f, fbTarget_ = 0.0f;
    float pp_ = 0.0f, ppTarget_ = 0.0f;
    float mix_ = 1.0f, mixTarget_ = 1.0f;
    float depth_ = 0.0f, depthTarget_ = 0.0f; // samples
    float apGain_ = 0.0f;
    bool first_ = true;
};

/// Family::ModDelay (voice or instance). Chorus / flanger / vibrato: 1..4 read heads on one short
/// line per channel, each swept by a sine LFO between base and base + depth (phases spread
/// evenly across the heads, and by stereo spread between channels). Reads use cubic Hermite
/// interpolation; the LFOs are evaluated every 8 samples and the head delays ramp linearly. Feedback (the
/// head average, soft-limited) makes it a flanger. Presets: chorus (7 ms, 3 ms, 0.5 Hz, 3 voices, mix 0.5),
/// flanger (1 ms, 2 ms, 0.2 Hz, feedback +-0.8), vibrato (5 ms, 4 ms, 5 Hz, mix 1).
class ModDelayProcessor final : public Processor
{
public:
    enum Param
    {
        kBase,
        kDepth,
        kRate,
        kVoices,
        kFeedback,
        kSpread,
        kMix,
    };

    static constexpr int kMaxVoices = 4;
    static constexpr float kMaxBaseMs = 40.0f;
    static constexpr float kMaxDepthMs = 10.0f;

    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;

private:
    [[nodiscard]] double headDelay(float phase) const noexcept;

    std::array<detail::DelayLine, 2> line_{};
    std::array<std::array<double, kMaxVoices>, 2> pos_{}; // current head delays (samples)
    std::array<float, kMaxVoices> voiceGain_{};           // smoothed 1/n or 0
    std::array<float, kMaxVoices> voiceGainTarget_{};
    std::array<float, kMaxVoices> voiceOffset_{}; // smoothed LFO phase offsets (cycles)
    std::array<float, kMaxVoices> voiceOffsetTarget_{};
    double sampleRate_ = 48000.0;
    int channels_ = 1;
    float smooth_ = 0.0f;
    double base_ = 0.0, baseTarget_ = 0.0;   // samples
    double depth_ = 0.0, depthTarget_ = 0.0; // samples
    float spread_ = 0.0f, spreadTarget_ = 0.0f;
    float fb_ = 0.0f, fbTarget_ = 0.0f;
    float mix_ = 0.5f, mixTarget_ = 0.5f;
    float lfoPhase_ = 0.0f;
    float lfoInc_ = 0.0f;
    bool first_ = true;
};

/// Family::Phaser (voice or instance). 2..12 first-order allpass stages at one swept frequency
/// f = centre * 2^(depth * lfo), with feedback around the chain and the output mixed with the
/// dry signal. With mix 0.5 and no feedback, notches sit where the chain phase is an odd
/// multiple of 180 degrees: tan(pi f / fs) = tan(pi fc / fs) tan(pi (2k + 1) / (2 N)).
/// Coefficients are recomputed every 8 samples and ramped linearly in between; stage-count
/// changes crossfade over 10 ms.
class PhaserProcessor final : public Processor
{
public:
    enum Param
    {
        kStages,
        kCentre,
        kDepth,
        kRate,
        kFeedback,
        kStereoPhase,
        kMix,
    };

    static constexpr int kMaxStages = 12;

    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;

private:
    [[nodiscard]] float coefficientAt(float phase) const noexcept;

    std::array<std::array<float, kMaxStages>, 2> state_{};
    std::array<float, 2> yPrev_{};
    std::array<float, 2> coef_{};
    double sampleRate_ = 48000.0;
    int channels_ = 1;
    int stages_ = 4;
    int stagesOld_ = 4;
    int stagesTarget_ = 4;
    float stageFade_ = 1.0f; // 1 = settled on stages_
    float stageFadeInc_ = 0.0f;
    float centre_ = 1000.0f, centreTarget_ = 1000.0f;
    float depth_ = 0.0f, depthTarget_ = 0.0f; // octaves
    float offset_ = 0.0f, offsetTarget_ = 0.0f;
    float fb_ = 0.0f, fbTarget_ = 0.0f;
    float mix_ = 0.5f, mixTarget_ = 0.5f;
    float lfoPhase_ = 0.0f;
    float lfoInc_ = 0.0f;
    float smooth_ = 0.0f;
    bool first_ = true;
};

/// Family::Reverb (instance slots). Pre-delay, early reflections (6 taps per channel off the
/// pre-delay line, cross-panned), and a late tail: a 4-stage allpass diffuser per channel into
/// the 8-line FDN (detail::DelayFdn), injected and tapped through orthogonal Hadamard rows so
/// the two outputs are decorrelated. RT60 is set per line from its length; damping shortens the
/// high (up to 10x) and low (up to 5x) decay. Freeze makes the loop lossless and mutes the input.
/// The diffuser adds about 15 ms before the late tail starts.
class ReverbProcessor final : public Processor
{
public:
    enum Param
    {
        kPreDelay,
        kSize,
        kDecay,
        kDamping,
        kLowDamping,
        kDiffusion,
        kModulation,
        kWidth,
        kEarlyLate,
        kFreeze,
        kMix,
    };

    static constexpr float kMaxPreDelayMs = 200.0f;
    static constexpr int kEarlyTaps = 6;

    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;

private:
    std::array<detail::DelayLine, 2> pre_{};
    std::array<detail::AllpassDiffuser, 2> diffuser_{};
    detail::DelayFdn fdn_;
    double sampleRate_ = 48000.0;
    int channels_ = 1;
    float smooth_ = 0.0f;
    float preDelay_ = 0.0f, preDelayTarget_ = 0.0f; // samples
    float size_ = 1.0f, sizeTarget_ = 1.0f;
    float inGain_ = 1.0f, inGainTarget_ = 1.0f;
    float early_ = 1.0f, earlyTarget_ = 1.0f;
    float late_ = 1.0f, lateTarget_ = 1.0f;
    float width_ = 1.0f, widthTarget_ = 1.0f;
    float mix_ = 1.0f, mixTarget_ = 1.0f;
    float rt60_ = 2.0f;
    bool freeze_ = false;
    bool first_ = true;
};

} // namespace alfa::dsp
