#pragma once

// Power-of-two ring buffer delay line plus the small loop helpers every delay-based structure
// uses (soft limiter, one-pole filters, block sanity checks).
//
// Read convention: read(d) returns the signal d samples before the most recent write, so
// read(0) is the last written sample. In a feedback loop, read before writing: y = read(D - 1)
// then write(x + g y) gives a loop of exactly D samples. Minimum fractional read delays per
// interpolator (clamped inside): None/Linear/Allpass 0, Cubic 1, Sinc 3. The maximum is the
// length passed to prepare(); reads beyond it are clamped.

#include "alfa/dsp/detail/DelayInterpolation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace alfa::dsp::detail
{

/// Per-reader state of the Thiran allpass interpolator (the only stateful one).
struct AllpassReadState
{
    float y1 = 0.0f;
};

class DelayLine
{
public:
    /// Allocates room for reads up to maxDelaySamples (plus the interpolators' extra taps).
    void prepare(int maxDelaySamples)
    {
        (void)sincTable();
        maxDelay_ = std::max(1, maxDelaySamples);
        std::size_t size = 1;
        while (size < static_cast<std::size_t>(maxDelay_) + SincTable::kTaps + 2)
            size <<= 1;
        buffer_.assign(size, 0.0f);
        mask_ = size - 1;
        write_ = 0;
    }

    void reset() noexcept
    {
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
        write_ = 0;
    }

    [[nodiscard]] int maxDelay() const noexcept { return maxDelay_; }

    void write(float x) noexcept
    {
        write_ = (write_ + 1) & mask_;
        buffer_[write_] = x;
    }

    /// Integer tap: k samples before the most recent write (k clamped to 0..size-1).
    [[nodiscard]] float tap(int k) const noexcept
    {
        return buffer_[(write_ - static_cast<std::size_t>(k)) & mask_];
    }

    [[nodiscard]] float readNone(float d) const noexcept
    {
        return tap(static_cast<int>(clampDelay(d, 0.0f)));
    }

    [[nodiscard]] float readLinear(float d) const noexcept
    {
        d = clampDelay(d, 0.0f);
        const int i = static_cast<int>(d);
        const float f = d - static_cast<float>(i);
        const float y0 = tap(i);
        return y0 + f * (tap(i + 1) - y0);
    }

    [[nodiscard]] float readCubic(float d) const noexcept
    {
        d = clampDelay(d, 1.0f);
        const int i = static_cast<int>(d);
        const float f = d - static_cast<float>(i);
        return hermite(tap(i - 1), tap(i), tap(i + 1), tap(i + 2), f);
    }

    /// Double-precision delays for long lines: a float delay near 2^16 samples only resolves
    /// 1/128 sample, which quantises slow glides and modulation.
    [[nodiscard]] float readLinear(double d) const noexcept
    {
        d = clampDelay(d, 0.0);
        const int i = static_cast<int>(d);
        const auto f = static_cast<float>(d - static_cast<double>(i));
        const float y0 = tap(i);
        return y0 + f * (tap(i + 1) - y0);
    }

    [[nodiscard]] float readCubic(double d) const noexcept
    {
        d = clampDelay(d, 1.0);
        const int i = static_cast<int>(d);
        const auto f = static_cast<float>(d - static_cast<double>(i));
        return hermite(tap(i - 1), tap(i), tap(i + 1), tap(i + 2), f);
    }

    [[nodiscard]] float readSinc(float d) const noexcept
    {
        d = clampDelay(d, static_cast<float>(SincTable::kBefore));
        const int i = static_cast<int>(d);
        const float f = d - static_cast<float>(i);
        float w[SincTable::kTaps];
        sincTable().weights(f, w);
        float acc = 0.0f;
        for (int j = 0; j < SincTable::kTaps; ++j)
            acc += w[j] * tap(i + j - SincTable::kBefore);
        return acc;
    }

    /// First-order Thiran allpass read. Keeps the fractional part in [0.5, 1.5) where the
    /// allpass is well behaved; call once per sample per reader with its own state. Moving d
    /// quickly produces transients, so use it for static or slowly gliding delays.
    [[nodiscard]] float readAllpass(float d, AllpassReadState& s) const noexcept
    {
        d = clampDelay(d, 0.0f);
        int i = static_cast<int>(d);
        float delta = d - static_cast<float>(i);
        if (delta < 0.5f && i > 0)
        {
            --i;
            delta += 1.0f;
        }
        const float a = thiranCoefficient(delta);
        // The allpass input is the line delayed by i; its previous sample is tap(i + 1) now.
        const float y = a * tap(i) + tap(i + 1) - a * s.y1;
        s.y1 = y;
        return y;
    }

    [[nodiscard]] float read(float d, Interpolation mode, AllpassReadState& s) const noexcept
    {
        switch (mode)
        {
        case Interpolation::None:
            return readNone(d);
        case Interpolation::Linear:
            return readLinear(d);
        case Interpolation::Cubic:
            return readCubic(d);
        case Interpolation::Allpass:
            return readAllpass(d, s);
        case Interpolation::Sinc:
            return readSinc(d);
        }
        return readLinear(d);
    }

private:
    template <typename T>
    [[nodiscard]] T clampDelay(T d, T lo) const noexcept
    {
        // Also maps NaN to lo (comparisons with NaN are false).
        if (!(d >= lo))
            return lo;
        const auto hi = static_cast<T>(maxDelay_);
        return d > hi ? hi : d;
    }

    std::vector<float> buffer_;
    std::size_t mask_ = 0;
    std::size_t write_ = 0;
    int maxDelay_ = 1;
};

// Loop helpers ----------------------------------------------------------------------------------

/// Identity for |x| <= 1, then a smooth knee that saturates at +-2 (continuous first derivative).
/// Placed on feedback paths it leaves normal levels untouched and keeps runaway loops bounded.
[[nodiscard]] inline float softLimit(float x) noexcept
{
    const float a = std::fabs(x);
    if (a <= 1.0f)
        return x;
    const float r = a - 1.0f;
    return std::copysign(1.0f + r / (1.0f + r), x);
}

/// Flushes values that have decayed into the denormal range (and stray non-finite states are
/// handled by the processors' block checks).
inline void flushDenormal(float& s) noexcept
{
    if (std::fabs(s) < 1e-20f)
        s = 0.0f;
}

/// One-pole low-pass y += (1 - p)(x - y). coefficientFor gives p for a -3 dB-ish cutoff.
struct OnePole
{
    float p = 0.0f;
    float y = 0.0f;

    [[nodiscard]] static float coefficientFor(double hz, double sampleRate) noexcept
    {
        const double w = 2.0 * 3.14159265358979323846 * std::clamp(hz, 0.0, 0.49 * sampleRate) / sampleRate;
        return static_cast<float>(std::exp(-w));
    }

    float lowpass(float x) noexcept
    {
        y += (1.0f - p) * (x - y);
        return y;
    }

    float highpass(float x) noexcept { return x - lowpass(x); }
};

/// One-pole DC blocker y[n] = x[n] - x[n-1] + R y[n-1] (about 5-10 Hz corner at 48 kHz for
/// R = 0.999).
struct DcBlocker
{
    float r = 0.999f;
    float x1 = 0.0f;
    float y1 = 0.0f;

    void setCorner(double hz, double sampleRate) noexcept
    {
        r = static_cast<float>(std::exp(-2.0 * 3.14159265358979323846 * hz / sampleRate));
    }

    float process(float x) noexcept
    {
        const float y = x - x1 + r * y1;
        x1 = x;
        y1 = y;
        return y;
    }

    void reset() noexcept { x1 = y1 = 0.0f; }
};

/// True when every sample of the block is finite (one branch per block; any NaN or Inf
/// poisons the sum).
[[nodiscard]] inline bool allFinite(const float* x, int n) noexcept
{
    float acc = 0.0f;
    for (int i = 0; i < n; ++i)
        acc += x[i] * 0.0f;
    return acc == 0.0f;
}

} // namespace alfa::dsp::detail
