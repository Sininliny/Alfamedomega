#pragma once

// Allpass and comb building blocks for delay, reverb and resonator families.
//
//   SchroederAllpass   v = x + g w,  y = w - g v,  w = v delayed by D   (|H| = 1 for |g| < 1)
//   NestedAllpass      the same with a second allpass inside the delay path (still allpass)
//   DampedComb         feedback comb with a one-pole low-pass in the loop (Freeverb style)
//   AllpassDiffuser    up to kMaxStages Schroeder allpasses in series (reverb input diffuser)
//
// All are prepared with a maximum delay, never allocate afterwards, and read before writing, so
// a delay of D gives exactly D samples. With g = 0 an allpass is a pure delay of D samples.

#include "alfa/dsp/detail/DelayLine.h"

#include <algorithm>
#include <array>

namespace alfa::dsp::detail
{

class SchroederAllpass
{
public:
    void prepare(int maxDelaySamples)
    {
        line_.prepare(maxDelaySamples);
        delay_ = static_cast<float>(maxDelaySamples);
    }

    void reset() noexcept { line_.reset(); }

    /// Delay in samples (fractional allowed, read with linear interpolation; >= 1).
    void setDelay(float samples) noexcept
    {
        delay_ = std::clamp(samples, 1.0f, static_cast<float>(line_.maxDelay()));
    }
    void setGain(float g) noexcept { g_ = std::clamp(g, -0.99f, 0.99f); }

    [[nodiscard]] float delay() const noexcept { return delay_; }
    [[nodiscard]] float gain() const noexcept { return g_; }

    float process(float x) noexcept { return processWithDelay(x, delay_); }

    /// Per-sample delay (for modulated or gliding allpasses).
    float processWithDelay(float x, float delaySamples) noexcept
    {
        const float w = line_.readLinear(delaySamples - 1.0f);
        const float v = x + g_ * w;
        line_.write(v);
        return w - g_ * v;
    }

private:
    DelayLine line_;
    float delay_ = 1.0f;
    float g_ = 0.0f;
};

class NestedAllpass
{
public:
    void prepare(int outerMaxDelay, int innerMaxDelay)
    {
        outer_.prepare(outerMaxDelay);
        inner_.prepare(innerMaxDelay);
        delay_ = static_cast<float>(outerMaxDelay);
    }

    void reset() noexcept
    {
        outer_.reset();
        inner_.reset();
    }

    void setOuter(float delaySamples, float g) noexcept
    {
        delay_ = std::clamp(delaySamples, 1.0f, static_cast<float>(outer_.maxDelay()));
        g_ = std::clamp(g, -0.99f, 0.99f);
    }

    void setInner(float delaySamples, float g) noexcept
    {
        inner_.setDelay(delaySamples);
        inner_.setGain(g);
    }

    float process(float x) noexcept
    {
        const float w = inner_.process(outer_.readLinear(delay_ - 1.0f));
        const float v = x + g_ * w;
        outer_.write(v);
        return w - g_ * v;
    }

private:
    DelayLine outer_;
    SchroederAllpass inner_;
    float delay_ = 1.0f;
    float g_ = 0.0f;
};

class DampedComb
{
public:
    void prepare(int maxDelaySamples)
    {
        line_.prepare(maxDelaySamples);
        delay_ = static_cast<float>(maxDelaySamples);
    }

    void reset() noexcept
    {
        line_.reset();
        damp_.y = 0.0f;
    }

    void setDelay(float samples) noexcept
    {
        delay_ = std::clamp(samples, 1.0f, static_cast<float>(line_.maxDelay()));
    }
    /// Loop gain, hard-limited below 1 so the comb always decays.
    void setFeedback(float g) noexcept { g_ = std::clamp(g, -0.999f, 0.999f); }
    /// Damping 0 (none) .. 1 (the loop low-pass pole at 0.95).
    void setDamping(float d) noexcept { damp_.p = 0.95f * std::clamp(d, 0.0f, 1.0f); }

    /// Returns the delayed signal (the comb output); writes x + g * lowpass(output).
    float process(float x) noexcept
    {
        const float y = line_.readLinear(delay_ - 1.0f);
        line_.write(x + g_ * damp_.lowpass(y));
        flushDenormal(damp_.y);
        return y;
    }

private:
    DelayLine line_;
    OnePole damp_;
    float delay_ = 1.0f;
    float g_ = 0.0f;
};

class AllpassDiffuser
{
public:
    static constexpr int kMaxStages = 8;

    /// Prepares `stages` allpasses with the given delays (samples). The delays become the
    /// maxima; setDelayScale can shorten them later.
    void prepare(const float* delaysSamples, int stages)
    {
        stages_ = std::clamp(stages, 0, kMaxStages);
        for (int i = 0; i < stages_; ++i)
        {
            const auto s = static_cast<std::size_t>(i);
            base_[s] = std::max(1.0f, delaysSamples[i]);
            ap_[s].prepare(static_cast<int>(std::ceil(base_[s])) + 1);
            ap_[s].setDelay(base_[s]);
        }
    }

    void reset() noexcept
    {
        for (int i = 0; i < stages_; ++i)
            ap_[static_cast<std::size_t>(i)].reset();
    }

    /// Per-stage gains; gains beyond the stage count are ignored.
    void setGains(const float* g) noexcept
    {
        for (int i = 0; i < stages_; ++i)
            ap_[static_cast<std::size_t>(i)].setGain(g[i]);
    }

    /// Scales every delay by s in (0, 1].
    void setDelayScale(float s) noexcept
    {
        s = std::clamp(s, 0.01f, 1.0f);
        for (int i = 0; i < stages_; ++i)
        {
            const auto k = static_cast<std::size_t>(i);
            ap_[k].setDelay(base_[k] * s);
        }
    }

    float process(float x) noexcept
    {
        for (int i = 0; i < stages_; ++i)
            x = ap_[static_cast<std::size_t>(i)].process(x);
        return x;
    }

    [[nodiscard]] int stages() const noexcept { return stages_; }

private:
    std::array<SchroederAllpass, kMaxStages> ap_{};
    std::array<float, kMaxStages> base_{};
    int stages_ = 0;
};

} // namespace alfa::dsp::detail
