#pragma once

// Trapezoidal ("zero-delay-feedback", TPT) state-variable filter after Andrew Simper, "Linear
// Trajectory SVF" (Cytomic technical paper, 2013), and V. Zavalishin, "The Art of VA Filter
// Design".
//
// Why this form: the state is kept as the two trapezoidal integrator memories (ic1, ic2), so
// the filter stays stable when g (cutoff) and k (damping) change every sample, which is what a
// network-driven cutoff does. Direct-form biquads blow up or thump under the same modulation.
//
// One tick produces three signals from the input v0: band (v1) and low (v2), and implicitly
// high = v0 - k v1 - v2. Every standard second-order response is a weighting of v0, v1 and v2
// (plus, for shelves, a pre-scaled g), so SvfCoeffs carries the five numbers
//   g, k, and out = c0 v0 + c1 v1 + c2 v2
// and linear ramps of those five morph between responses without clicks.

#include <algorithm>
#include <cmath>

namespace alfa::dsp::detail
{

/// Prewarped integrator gain g = tan(pi f / fs), f clamped to [1 Hz, 0.49 fs] so g stays finite.
[[nodiscard]] inline float svfGain(double hz, double sampleRate) noexcept
{
    const double f = std::clamp(std::isfinite(hz) ? hz : 1000.0, 1.0, 0.49 * sampleRate);
    return static_cast<float>(std::tan(3.14159265358979323846 * f / sampleRate));
}

struct SvfCoeffs
{
    float g = 0.0f;
    float k = 1.41421356f; ///< 1 / Q
    float c0 = 0.0f;       ///< weight of the input
    float c1 = 0.0f;       ///< weight of band (v1; peak gain 1/k at the cutoff)
    float c2 = 1.0f;       ///< weight of low (v2)
};

/// Response designs (Simper's paper). q > 0; gains in dB for bell and shelves.
namespace svf
{

[[nodiscard]] inline SvfCoeffs lowpass(float g, float q) noexcept
{
    return {g, 1.0f / q, 0.0f, 0.0f, 1.0f};
}

/// Constant peak gain band-pass: 0 dB at the centre for any q.
[[nodiscard]] inline SvfCoeffs bandpass(float g, float q) noexcept
{
    const float k = 1.0f / q;
    return {g, k, 0.0f, k, 0.0f};
}

[[nodiscard]] inline SvfCoeffs highpass(float g, float q) noexcept
{
    const float k = 1.0f / q;
    return {g, k, 1.0f, -k, -1.0f};
}

[[nodiscard]] inline SvfCoeffs notch(float g, float q) noexcept
{
    const float k = 1.0f / q;
    return {g, k, 1.0f, -k, 0.0f};
}

[[nodiscard]] inline SvfCoeffs allpass(float g, float q) noexcept
{
    const float k = 1.0f / q;
    return {g, k, 1.0f, -2.0f * k, 0.0f};
}

[[nodiscard]] inline SvfCoeffs bell(float g, float q, float gainDb) noexcept
{
    const float a = std::pow(10.0f, gainDb / 40.0f);
    const float k = 1.0f / (q * a);
    return {g, k, 1.0f, k * (a * a - 1.0f), 0.0f};
}

[[nodiscard]] inline SvfCoeffs lowShelf(float g, float q, float gainDb) noexcept
{
    const float a = std::pow(10.0f, gainDb / 40.0f);
    const float k = 1.0f / q;
    return {g / std::sqrt(a), k, 1.0f, k * (a - 1.0f), a * a - 1.0f};
}

[[nodiscard]] inline SvfCoeffs highShelf(float g, float q, float gainDb) noexcept
{
    const float a = std::pow(10.0f, gainDb / 40.0f);
    const float k = 1.0f / q;
    return {g * std::sqrt(a), k, a * a, k * (1.0f - a) * a, 1.0f - a * a};
}

} // namespace svf

/// One SVF stage's state. The coefficients live with the caller so they can be ramped or shared
/// between channels.
class Svf
{
public:
    void reset() noexcept { ic1_ = ic2_ = 0.0f; }

    /// Advances one sample; writes band (v1) and low (v2). g >= 0, k > 0.
    void tick(float v0, float g, float k, float& band, float& low) noexcept
    {
        const float a1 = 1.0f / (1.0f + g * (g + k));
        const float v1 = a1 * (ic1_ + g * (v0 - ic2_));
        const float v2 = ic2_ + g * v1;
        ic1_ = 2.0f * v1 - ic1_;
        ic2_ = 2.0f * v2 - ic2_;
        band = v1;
        low = v2;
    }

    [[nodiscard]] float process(float v0, const SvfCoeffs& c) noexcept
    {
        float band = 0.0f, low = 0.0f;
        tick(v0, c.g, c.k, band, low);
        return c.c0 * v0 + c.c1 * band + c.c2 * low;
    }

    [[nodiscard]] bool isFinite() const noexcept { return std::isfinite(ic1_) && std::isfinite(ic2_); }

    /// Zeroes states that decayed below audibility, before they reach the denormal range.
    void flushDenormals() noexcept
    {
        if (std::fabs(ic1_) < 1e-20f)
            ic1_ = 0.0f;
        if (std::fabs(ic2_) < 1e-20f)
            ic2_ = 0.0f;
    }

private:
    float ic1_ = 0.0f;
    float ic2_ = 0.0f;
};

} // namespace alfa::dsp::detail
