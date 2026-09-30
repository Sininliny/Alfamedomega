#pragma once

// Second-order IIR section with the RBJ "Audio EQ Cookbook" designs.
//
// Use it for static or slowly moving responses (EQ banks, body models, crossovers). Its
// coefficients are not safe to modulate at audio rate; use FiltersSvf.h for anything a network
// sweeps. Coefficients and state are double: a 40 Hz, Q 20 peak at 192 kHz puts the poles within
// 1e-4 of the unit circle, where float coefficients visibly move the resonance.

#include <algorithm>
#include <cmath>
#include <complex>

namespace alfa::dsp::detail
{

struct BiquadCoeffs
{
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0; ///< normalised so a0 = 1

    /// Magnitude of the response at hz (for tests and gain staging, not the audio path).
    [[nodiscard]] double magnitude(double hz, double sampleRate) const noexcept
    {
        const std::complex<double> z1 = std::polar(1.0, -2.0 * 3.14159265358979323846 * hz / sampleRate);
        const std::complex<double> z2 = z1 * z1;
        return std::abs((b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2));
    }
};

namespace biquad
{

struct Warp
{
    double cosw, alpha;
};

[[nodiscard]] inline Warp warp(double hz, double q, double sampleRate) noexcept
{
    const double f = std::clamp(hz, 1.0, 0.49 * sampleRate);
    const double w = 2.0 * 3.14159265358979323846 * f / sampleRate;
    return {std::cos(w), std::sin(w) / (2.0 * std::max(q, 1e-3))};
}

[[nodiscard]] inline BiquadCoeffs normalise(double b0, double b1, double b2, double a0, double a1,
                                            double a2) noexcept
{
    const double inv = 1.0 / a0;
    return {b0 * inv, b1 * inv, b2 * inv, a1 * inv, a2 * inv};
}

[[nodiscard]] inline BiquadCoeffs lowpass(double hz, double q, double sampleRate) noexcept
{
    const auto [c, al] = warp(hz, q, sampleRate);
    return normalise((1.0 - c) * 0.5, 1.0 - c, (1.0 - c) * 0.5, 1.0 + al, -2.0 * c, 1.0 - al);
}

[[nodiscard]] inline BiquadCoeffs highpass(double hz, double q, double sampleRate) noexcept
{
    const auto [c, al] = warp(hz, q, sampleRate);
    return normalise((1.0 + c) * 0.5, -(1.0 + c), (1.0 + c) * 0.5, 1.0 + al, -2.0 * c, 1.0 - al);
}

/// Constant 0 dB peak gain.
[[nodiscard]] inline BiquadCoeffs bandpass(double hz, double q, double sampleRate) noexcept
{
    const auto [c, al] = warp(hz, q, sampleRate);
    return normalise(al, 0.0, -al, 1.0 + al, -2.0 * c, 1.0 - al);
}

[[nodiscard]] inline BiquadCoeffs notch(double hz, double q, double sampleRate) noexcept
{
    const auto [c, al] = warp(hz, q, sampleRate);
    return normalise(1.0, -2.0 * c, 1.0, 1.0 + al, -2.0 * c, 1.0 - al);
}

[[nodiscard]] inline BiquadCoeffs allpass(double hz, double q, double sampleRate) noexcept
{
    const auto [c, al] = warp(hz, q, sampleRate);
    return normalise(1.0 - al, -2.0 * c, 1.0 + al, 1.0 + al, -2.0 * c, 1.0 - al);
}

[[nodiscard]] inline BiquadCoeffs peaking(double hz, double q, double gainDb, double sampleRate) noexcept
{
    const auto [c, al] = warp(hz, q, sampleRate);
    const double a = std::pow(10.0, gainDb / 40.0);
    return normalise(1.0 + al * a, -2.0 * c, 1.0 - al * a, 1.0 + al / a, -2.0 * c, 1.0 - al / a);
}

[[nodiscard]] inline BiquadCoeffs lowShelf(double hz, double q, double gainDb, double sampleRate) noexcept
{
    const auto [c, al] = warp(hz, q, sampleRate);
    const double a = std::pow(10.0, gainDb / 40.0);
    const double s = 2.0 * std::sqrt(a) * al;
    return normalise(a * ((a + 1.0) - (a - 1.0) * c + s), 2.0 * a * ((a - 1.0) - (a + 1.0) * c),
                     a * ((a + 1.0) - (a - 1.0) * c - s), (a + 1.0) + (a - 1.0) * c + s,
                     -2.0 * ((a - 1.0) + (a + 1.0) * c), (a + 1.0) + (a - 1.0) * c - s);
}

[[nodiscard]] inline BiquadCoeffs highShelf(double hz, double q, double gainDb, double sampleRate) noexcept
{
    const auto [c, al] = warp(hz, q, sampleRate);
    const double a = std::pow(10.0, gainDb / 40.0);
    const double s = 2.0 * std::sqrt(a) * al;
    return normalise(a * ((a + 1.0) + (a - 1.0) * c + s), -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
                     a * ((a + 1.0) + (a - 1.0) * c - s), (a + 1.0) - (a - 1.0) * c + s,
                     2.0 * ((a - 1.0) - (a + 1.0) * c), (a + 1.0) - (a - 1.0) * c - s);
}

} // namespace biquad

/// Transposed direct form II section (the best-behaved direct form for float/double state).
class Biquad
{
public:
    void reset() noexcept { s1_ = s2_ = 0.0; }

    [[nodiscard]] float process(float x, const BiquadCoeffs& c) noexcept
    {
        const double in = x;
        const double y = c.b0 * in + s1_;
        s1_ = c.b1 * in - c.a1 * y + s2_;
        s2_ = c.b2 * in - c.a2 * y;
        return static_cast<float>(y);
    }

    [[nodiscard]] bool isFinite() const noexcept { return std::isfinite(s1_) && std::isfinite(s2_); }

    void flushDenormals() noexcept
    {
        if (std::fabs(s1_) < 1e-30)
            s1_ = 0.0;
        if (std::fabs(s2_) < 1e-30)
            s2_ = 0.0;
    }

private:
    double s1_ = 0.0;
    double s2_ = 0.0;
};

} // namespace alfa::dsp::detail
