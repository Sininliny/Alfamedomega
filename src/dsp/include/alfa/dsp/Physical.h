#pragma once

// Physical-model slot families: a digital-waveguide string (Family::Waveguide) and a modal
// resonator bank (Family::Resonator). Both are excited by the slot input, so any primitive,
// noise burst or earlier slot can bow, pluck or strike them; neither is an instrument preset.
//
// Pitch (both families): hz = base * 2^((pitch + fine / 100) / 12), where base is ctx.f0 in
// voice slots and the absolute "frequency" param in instance slots (ctx.f0 == 0). The result
// is clamped to 20 Hz .. the family's upper limit.
//
// Extending the physical models
// -----------------------------
// The modal side is split into a mode source and a renderer:
//   - detail::PhysicalModeTables holds precomputed ratio tables (computed once, see
//     detail/PhysicalModes.h). ResonatorProcessor turns params into per-mode frequency, decay
//     and gain arrays from them.
//   - detail::ModalBank renders any such arrays (SoA, vectorised over modes, stable rotation
//     form), knowing nothing about where they came from.
// A plate, membrane with other boundary conditions or a 2-D mesh reduced to modes therefore
// either adds tables to PhysicalModeTables behind a *new* param/family (the six-table morph axis
// of Resonator is a trained contract and must not be re-spaced), or becomes a new Processor
// class that fills its own arrays and reuses ModalBank. Neither changes the Processor interface.

#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/FastMath.h"
#include "alfa/dsp/detail/PhysicalModes.h"

#include <array>
#include <vector>

namespace alfa::dsp
{

/// Karplus-Strong / digital-waveguide string. One loop per channel:
///
///   input -> excitation comb (1 - z^-De) / 2 -> + -> delay N -> loss FIR -> fractional allpass
///                                               ^                           -> dispersion x0..4
///                                               +---- loss gain, soft limit <------------+
///   output = mix of dry and the pickup comb (1 - z^-Dp) / 2 on the string signal
///
/// Tuning: the loop's total phase delay at the fundamental equals sr / hz exactly. The loss
/// FIR is symmetric (linear phase, one sample), the dispersion allpasses' phase delay at the
/// fundamental is computed exactly and subtracted, and the first-order fractional allpass is
/// solved in closed form for the remainder (not the low-frequency Thiran approximation), so the
/// fundamental stays within a fraction of a cent up to sr / 5.
///
/// Loss: the per-period gain at the fundamental is 10^(-3 / (T60 hz)); the loss gain divides
/// out the FIR's magnitude at the fundamental so brightness changes the decay of the upper
/// partials only. Where the FIR alone would lose more than that at the fundamental (high pitch,
/// long T60, low brightness) it is opened just enough: the decay param wins. Dispersion (negative-coefficient
/// allpasses) makes upper partials sharp like a stiff string; at high pitch the stage count and coefficient
/// shrink to fit the loop. Tension adds up to +1 semitone (at tension 1) proportional to the string amplitude
/// (clamped at 1), a small pitch glide as loud notes decay. A soft limiter in the loop holds the string below
/// +-4 whatever the drive; below +-2 the loop is exactly linear.
class WaveguideProcessor final : public Processor
{
public:
    enum Param : int
    {
        kPitch = 0,  ///< semitones   -24 .. 24, linear, default 0
        kFine,       ///< cents      -100 .. 100, linear, default 0
        kFrequency,  ///< Hz, instance slots only: 20 .. 8000 log, default 220
        kDecay,      ///< T60 s      0.05 .. 30 log, default 2
        kBrightness, ///< 0 .. 1: loss FIR from (1 + cos w) / 2 (dark) to flat (bright), default 0.5
        kDispersion, ///< 0 .. 1: allpass coefficient 0 .. -0.93 over up to 4 stages, default 0
        kPickup,     ///< 0.02 .. 0.5 of the string length, default 0.2
        kExcitation, ///< 0.02 .. 0.5 of the string length, default 0.13
        kTension,    ///< 0 .. 1 semitone of amplitude-dependent glide, default 0
        kInputGain,  ///< 0 .. 2 linear, default 1
        kPluck,      ///< Toggle: filtered noise burst (one period) on note-on, default off
        kMix,        ///< 0 .. 1 dry/wet, default 1 (wet)
    };

    static constexpr float kMinHz = 20.0f;

    static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;

    /// The fundamental the loop is currently tuned to (after clamping and glide), for tests.
    [[nodiscard]] double tunedHz() const noexcept { return tunedHz_; }
    [[nodiscard]] int dispersionStages() const noexcept { return stages_; }

private:
    static constexpr int kMaxStages = 4;

    struct Channel
    {
        std::vector<float> line; // loop delay, also read by the pickup comb
        std::vector<float> exc;  // input history for the excitation comb
        float fir1 = 0.0f, fir2 = 0.0f;
        float frac = 0.0f;
        std::array<float, kMaxStages> disp{};
        float burstLp = 0.0f;
        detail::Random rng;
    };

    void retune(double hz) noexcept;
    void resetChannel(Channel& ch) noexcept;

    double sampleRate_ = 48000.0;
    int channels_ = 1;
    std::array<Channel, 2> ch_;
    unsigned lineMask_ = 0, excMask_ = 0;
    unsigned pos_ = 0;

    // Targets from the params.
    double baseHz_ = 220.0;
    float t60_ = 2.0f, brightness_ = 0.5f, dispersion_ = 0.0f, tension_ = 0.0f;
    bool pluckOn_ = false;

    // Loop coefficients (retune()).
    double tunedHz_ = 220.0;
    double period_ = 48000.0 / 220.0;
    int delayInt_ = 1;
    float firA0_ = 1.0f, firA1_ = 0.0f;
    float fracA_ = 0.0f;
    float dispA_ = 0.0f;
    int stages_ = 0;
    float loopGain_ = 0.0f;
    double lastRetuneHz_ = -1.0;
    float lastRetuneB_ = -1.0f, lastRetuneD_ = -1.0f, lastRetuneT_ = -1.0f;

    // Per-block ramps (current value, target).
    float pickupD_ = 0.0f, pickupT_ = 0.0f;
    float excD_ = 0.0f, excT_ = 0.0f;
    float inGain_ = 1.0f, inGainT_ = 1.0f;
    float mix_ = 1.0f, mixT_ = 1.0f;
    bool first_ = true;

    float env_ = 0.0f; // string amplitude follower for the tension glide
    bool pluckPending_ = false;
    int burstLeft_ = 0;
    float burstK_ = 0.5f;
};

/// Modal resonator: up to limits.resonatorModes (<= 32) decaying sinusoidal modes excited by
/// the input. Mode k has frequency hz * ratio_k, T60 decay * ratio_k^-tilt and gain
/// ratio_k^(brightness / 6.02) * |sin(pi (k + 1) position)|, gains normalised to unit energy.
/// ratio_k morphs (log-frequency interpolation) between the detail::PhysicalModeSet tables,
/// then is stretched to ratio_k^(1 + inharmonicity) and detuned by spread * u_k cents (u_k a
/// fixed per-mode random in -1..1). Modes at or above 0.48 sr are skipped.
class ResonatorProcessor final : public Processor
{
public:
    enum Param : int
    {
        kPitch = 0,     ///< semitones   -24 .. 24, linear, default 0
        kFine,          ///< cents      -100 .. 100, linear, default 0
        kFrequency,     ///< Hz, instance slots only: 20 .. 8000 log, default 220
        kStructure,     ///< 0 .. 5: string, stiff string, free bar, membrane, square plate, tube
        kInharmonicity, ///< stretch exponent -0.25 .. 0.25, default 0
        kDecay,         ///< T60 of the lowest mode, s, 0.01 .. 30 log, default 1.5
        kDampingTilt,   ///< 0 .. 2: T60_k = T60 ratio_k^-tilt, default 0.5
        kBrightness,    ///< mode gain tilt dB/octave -18 .. 6, default -6
        kPosition,      ///< strike / pickup position 0.01 .. 0.5, default 0.23
        kModeCount,     ///< 0 .. 1 of the tier limit (at least one mode), default 1
        kSpread,        ///< per-mode random detune 0 .. 50 cents, default 0
        kInputGain,     ///< 0 .. 2 linear, default 1
        kMix,           ///< 0 .. 1 dry/wet, default 1 (wet)
    };

    static constexpr float kMinHz = 20.0f;

    static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;

    /// Frequency of mode k in Hz as last set (0 for skipped or unused modes), for tests.
    [[nodiscard]] float modeHz(int k) const noexcept;
    [[nodiscard]] int activeModes() const noexcept { return bank_.activeModes(); }
    [[nodiscard]] int modeLimit() const noexcept { return limit_; }

private:
    double sampleRate_ = 48000.0;
    int channels_ = 1;
    int limit_ = detail::kPhysicalMaxModes;
    const detail::PhysicalModeTables* tables_ = nullptr;
    std::array<float, detail::kPhysicalMaxModes> detune_{}; // fixed -1..1 per mode

    alignas(64) std::array<float, detail::kPhysicalMaxModes> freq_{};
    alignas(64) std::array<float, detail::kPhysicalMaxModes> decay_{};
    alignas(64) std::array<float, detail::kPhysicalMaxModes> gain_{};
    detail::ModalBank bank_;

    std::array<float, kSlotParams> lastParams_{};
    float lastF0_ = -1.0f;
    float t60_ = 1.5f;

    float inGain_ = 1.0f, inGainT_ = 1.0f;
    float mix_ = 1.0f, mixT_ = 1.0f;
    bool first_ = true;
    std::vector<float> scratch_;
};

} // namespace alfa::dsp
