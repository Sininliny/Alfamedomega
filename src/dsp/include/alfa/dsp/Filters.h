#pragma once

// Filter-type processing families (Family.h): FILTER, COMB, FORMANT and BODY.
//
// All four run in voice slots (mono, ctx.f0 = voice pitch) and instance slots (stereo, f0 = 0;
// each channel has its own state). Parameters arrive once per control block and are ramped
// linearly across the next kControlBlock samples (BODY smooths its mode table over ~15 ms
// instead, so a body-type switch glides rather than clicks). The building blocks live in
// detail/FiltersSvf.h (modulation-safe TPT state-variable filter) and detail/FiltersBiquad.h.
//
// Parameter tables: see params() of each class and the .cpp for the mapping details.

#include "alfa/dsp/Family.h"
#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/FiltersBiquad.h"
#include "alfa/dsp/detail/FiltersSvf.h"

#include <array>
#include <cstddef>
#include <vector>

namespace alfa::dsp
{

namespace detail
{

/// A bank of N linear parameter ramps: value_j moves by step_j per sample for the first
/// kControlBlock samples after retarget(), then holds at target_j.
template <std::size_t N>
struct FiltersRamps
{
    std::array<float, N> value{};
    std::array<float, N> target{};
    std::array<float, N> step{};
    int left = 0;

    /// Starts ramping towards target (reached after kControlBlock samples), or jumps.
    void retarget(bool jump) noexcept
    {
        constexpr float inv = 1.0f / static_cast<float>(kControlBlock);
        for (std::size_t j = 0; j < value.size(); ++j)
        {
            if (jump)
                value[j] = target[j];
            step[j] = (target[j] - value[j]) * inv;
        }
        left = jump ? 0 : kControlBlock;
    }

    /// Samples of the next n that still ramp.
    [[nodiscard]] int rampingIn(int n) const noexcept { return n < left ? n : left; }

    /// Advances every ramp by n samples (after each channel of a block was processed).
    void advance(int n) noexcept
    {
        const int s = rampingIn(n);
        left -= s;
        for (std::size_t j = 0; j < value.size(); ++j)
            value[j] = left == 0 ? target[j] : value[j] + step[j] * static_cast<float>(s);
    }
};

} // namespace detail

/// FILTER: multimode TPT state-variable filter (Simper/Cytomic), stable under per-sample
/// cutoff and resonance modulation.
///
///   0 cutoff     20 .. 20000 Hz, log; clamped to 0.49 fs after key tracking
///   1 resonance  0 .. 1: Q from Butterworth (0.707) to 100 (a ring of seconds, just short of
///                self-oscillation); a nonlinear damping term (k += 0.05 band^2) bounds the
///                resonant peak for loud inputs the way an analog filter's saturation does
///   2 mode       continuous morph LP > BP > HP > notch > back to LP at 1 (0, .25, .5, .75)
///   3 slope      12 or 24 dB/oct (24 = two cascaded stages with 4th-order Butterworth Qs;
///                the second stage carries the resonance)
///   4 drive      0 .. 36 dB pre-filter tanh saturation (0 = linear)
///   5 keytrack   0 .. 1: cutoff *= (f0 / 261.63)^keytrack when f0 > 0
///   6 type       multimode, bell, low shelf, high shelf (the EQ types are single-stage, use
///                cutoff as the centre/corner and resonance for Q 0.707 .. 16)
///   7 gain       -24 .. +24 dB for bell and shelves
///   8 mix        dry/wet
class FilterProcessor final : public Processor
{
public:
    static constexpr Family kFamily = Family::Filter;
    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override { return tail_; }

private:
    // Ramp indices: per stage g, k, c0, c1, c2, ch (weight of -k_eff * band), nl; then drive.
    enum : int
    {
        kG = 0,
        kK,
        kC0,
        kC1,
        kC2,
        kCh,
        kNl,
        kStageRamps,
        kDrive = 2 * kStageRamps,
        kDriveMakeup,
        kDriveBlend,
        kWet,
        kRampCount
    };

    double sampleRate_ = 48000.0;
    int channels_ = 1;
    detail::FiltersRamps<kRampCount> ramps_;
    bool primed_ = false;
    int tail_ = 0;
    std::array<std::array<detail::Svf, 2>, 2> svf_{}; // [channel][stage]
    std::array<std::array<float, 2>, 2> lastBand_{};  // [channel][stage], for the damping term
};

/// COMB: feedforward and feedback comb with fractional (4-point Lagrange) delay, a one-pole
/// damping low-pass in the loop and loop-delay compensation, so the comb pitch is exact.
///
///   0 frequency  20 .. 5000 Hz, log (default 261.63 Hz, so keytrack 1 tunes the comb to f0)
///   1 keytrack   0 .. 1: frequency *= (f0 / 261.63)^keytrack when f0 > 0; clamped to
///                20 Hz .. 0.45 fs
///   2 feedback   -0.99 .. 0.99: loop gain of the feedback comb and tap gain of the
///                feedforward comb (negative moves the peaks half a harmonic)
///   3 damping    0 .. 1: loop low-pass from open to 200 Hz (log)
///   4 balance    0 = feedforward, 1 = feedback (outputs are normalised to white-noise power)
///   5 mix        dry/wet
class CombProcessor final : public Processor
{
public:
    static constexpr Family kFamily = Family::Comb;
    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override { return tail_; }

private:
    enum : int
    {
        kDelay = 0,
        kFbDelay,
        kFb,
        kFf,
        kDamp,
        kWff,
        kWfb,
        kWet,
        kRampCount
    };

    struct Channel
    {
        std::vector<float> ffLine; // input history
        std::vector<float> fbLine; // loop history
        float lp = 0.0f;
    };

    double sampleRate_ = 48000.0;
    int channels_ = 1;
    std::size_t mask_ = 0;
    std::size_t write_ = 0;
    std::array<Channel, 2> ch_{};
    detail::FiltersRamps<kRampCount> ramps_;
    bool primed_ = false;
    int tail_ = 0;
};

/// FORMANT: five parallel constant-peak-gain SVF band-passes whose centre, bandwidth and level
/// interpolate along a vowel morph between male (bass) and female (soprano) tables.
///
///   0 vowel   A > E > I > O > U (0, .25, .5, .75, 1), frequencies interpolated in log
///   1 voice   0 = male, 1 = female (tables interpolated)
///   2 shift   -12 .. +12 semitones on all formants (bandwidths scale with them)
///   3 q       x0.25 .. x4 the table Q (log, default x1)
///   4 mix     dry/wet
/// Wet gain is 0 dB at the first formant's peak; formants above 0.45 fs fade out.
class FormantProcessor final : public Processor
{
public:
    static constexpr Family kFamily = Family::Formant;
    static constexpr int kFormants = 5;
    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override { return tail_; }

    /// Centre frequency (Hz) of formant i for the current params (after setParams), for tests.
    [[nodiscard]] float formantHz(int i) const noexcept;

private:
    // Per formant g, k, amplitude (already multiplied by k for the unit-peak band-pass).
    enum : int
    {
        kWet = 3 * kFormants,
        kRampCount
    };

    double sampleRate_ = 48000.0;
    int channels_ = 1;
    detail::FiltersRamps<kRampCount> ramps_;
    bool primed_ = false;
    int tail_ = 0;
    std::array<float, kFormants> hz_{};
    std::array<std::array<detail::Svf, kFormants>, 2> svf_{};
};

/// BODY: acoustic body coloration, a fixed chain of eight peaking biquads (the body's main
/// air/plate/shell modes) and a high shelf (material brightness).
///
///   0 type      guitar, violin, drum shell, box (nominal mode tables in the .cpp)
///   1 size      x0.5 .. x2 the nominal dimensions (log): mode frequencies divide by it
///   2 material  0 = soft/dark (modes tilt down with frequency, lower Q, HF shelf -12 dB) ..
///               1 = hard/bright (tilt up, higher Q, shelf +3 dB)
///   3 damping   0 = ringing (Q x2, peaks +25%) .. 1 = dead (Q x0.5, peaks -25%)
///   4 mix       dry/wet
/// Mode parameters glide with a ~15 ms one-pole so type and size changes never click. At
/// neutral material and damping, pink-noise gain is within about 1 dB of unity for every type.
class BodyProcessor final : public Processor
{
public:
    static constexpr Family kFamily = Family::Body;
    static constexpr int kModes = 8;
    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override { return tail_; }

private:
    static constexpr int kSections = kModes + 1; // modes, then the shelf

    struct Section
    {
        double logHz = 0.0, gainDb = 0.0, logQ = 0.0;
    };

    double sampleRate_ = 48000.0;
    int channels_ = 1;
    bool primed_ = false;
    int tail_ = 0;
    std::array<Section, kSections> current_{};
    std::array<Section, kSections> target_{};
    std::array<detail::BiquadCoeffs, kSections> coeffs_{};
    std::array<std::array<detail::Biquad, kSections>, 2> biquad_{};
    detail::FiltersRamps<1> ramps_; // wet
};

} // namespace alfa::dsp
