#pragma once

// Small building blocks shared by the Convolution and Spatial families: a per-block linear ramp
// for gains, a TPT state-variable filter and a TPT one-pole. All header-only, allocation-free.

#include "alfa/dsp/detail/FastMath.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp::detail
{

/// Moves linearly to a new target over a fixed number of samples (normally one control block),
/// then holds. Used for gains and matrix coefficients, where a per-block step would click.
class LinearRamp
{
public:
    void reset(float v) noexcept
    {
        value_ = v;
        target_ = v;
        step_ = 0.0f;
        remaining_ = 0;
    }

    void setTarget(float target, int samples) noexcept
    {
        target_ = target;
        if (samples <= 0 || target == value_)
        {
            value_ = target;
            remaining_ = 0;
            step_ = 0.0f;
            return;
        }
        remaining_ = samples;
        step_ = (target - value_) / static_cast<float>(samples);
    }

    float next() noexcept
    {
        if (remaining_ > 0)
        {
            value_ += step_;
            if (--remaining_ == 0)
                value_ = target_; // land exactly, so held values are bit-exact
        }
        return value_;
    }

    [[nodiscard]] float value() const noexcept { return value_; }
    [[nodiscard]] float target() const noexcept { return target_; }
    [[nodiscard]] bool ramping() const noexcept { return remaining_ > 0; }

private:
    float value_ = 0.0f;
    float target_ = 0.0f;
    float step_ = 0.0f;
    int remaining_ = 0;
};

/// Topology-preserving-transform state-variable filter (Zavalishin / Cytomic form). Stable for
/// any cutoff below Nyquist and any Q > 0, and cheap to retune every control block.
class TptSvf
{
public:
    struct Out
    {
        float lp, bp, hp;
    };

    void setup(double cutoffHz, double q, double sampleRate) noexcept
    {
        const double fc = std::clamp(cutoffHz, 1.0, 0.49 * sampleRate);
        const double g = std::tan(3.14159265358979323846 * fc / sampleRate);
        k_ = static_cast<float>(1.0 / std::max(q, 0.05));
        const double a1 = 1.0 / (1.0 + g * (g + static_cast<double>(k_)));
        a1_ = static_cast<float>(a1);
        a2_ = static_cast<float>(g * a1);
        a3_ = static_cast<float>(g * g * a1);
    }

    void reset() noexcept
    {
        ic1_ = 0.0f;
        ic2_ = 0.0f;
    }

    Out tick(float x) noexcept
    {
        const float v3 = x - ic2_;
        const float v1 = a1_ * ic1_ + a2_ * v3;
        const float v2 = ic2_ + a2_ * ic1_ + a3_ * v3;
        ic1_ = 2.0f * v1 - ic1_;
        ic2_ = 2.0f * v2 - ic2_;
        return {v2, v1, x - k_ * v1 - v2};
    }

    /// All-pass output of the same section: x - 2k bp.
    [[nodiscard]] float k() const noexcept { return k_; }

private:
    float k_ = 1.41421356f;
    float a1_ = 1.0f, a2_ = 0.0f, a3_ = 0.0f;
    float ic1_ = 0.0f, ic2_ = 0.0f;
};

/// First-order TPT low-pass; the high-pass is x - lp.
class TptOnePole
{
public:
    void setup(double cutoffHz, double sampleRate) noexcept
    {
        const double fc = std::clamp(cutoffHz, 1.0, 0.49 * sampleRate);
        const double g = std::tan(3.14159265358979323846 * fc / sampleRate);
        a_ = static_cast<float>(g / (1.0 + g));
    }

    void reset() noexcept { s_ = 0.0f; }

    float tick(float x) noexcept
    {
        const float v = a_ * (x - s_);
        const float lp = v + s_;
        s_ = lp + v;
        return lp;
    }

private:
    float a_ = 0.0f;
    float s_ = 0.0f;
};

/// Coefficient of a one-pole smoother that covers 63% of a step in `seconds`, applied once per
/// `interval` samples.
[[nodiscard]] inline float smoothingCoeff(double seconds, double sampleRate, int interval = 1) noexcept
{
    if (seconds <= 0.0 || sampleRate <= 0.0)
        return 1.0f;
    return static_cast<float>(1.0 - std::exp(-static_cast<double>(interval) / (seconds * sampleRate)));
}

} // namespace alfa::dsp::detail
