#pragma once

// Fractional-delay interpolators shared by the delay-based families (Delay.h) and anything else
// that reads a delay line or a buffer between samples.
//
// Conventions: a read at fractional delay d = i + f (i integer, 0 <= f < 1) blends the sample
// i steps back (y0) with its older neighbour (y1); ym1 is the newer neighbour of y0, y2 the
// older neighbour of y1. So f moves the read from y0 towards y1, i.e. further into the past.

#include "alfa/dsp/detail/FastMath.h"

#include <array>
#include <cmath>

namespace alfa::dsp::detail
{

enum class Interpolation : unsigned char
{
    None,    ///< nearest integer at or below d (use for integer delays)
    Linear,  ///< 2 points; cheap, low-passes by up to cos(w/2) at f = 0.5
    Cubic,   ///< 4-point Catmull-Rom Hermite; good for modulated reads
    Allpass, ///< first-order Thiran; flat magnitude, stateful, for static or slowly moving delays
    Sinc,    ///< 8-point Kaiser-windowed sinc, table based; best quality, 8 MACs per read
};

/// 4-point, 3rd-order Hermite (Catmull-Rom). Unity gain at DC; magnitude <= 1 at every
/// frequency, so it is safe inside feedback loops.
[[nodiscard]] inline float hermite(float ym1, float y0, float y1, float y2, float f) noexcept
{
    const float c1 = 0.5f * (y1 - ym1);
    const float c2 = ym1 - 2.5f * y0 + 2.0f * y1 - 0.5f * y2;
    const float c3 = 0.5f * (y2 - ym1) + 1.5f * (y0 - y1);
    return ((c3 * f + c2) * f + c1) * f + y0;
}

/// Zeroth-order modified Bessel function of the first kind (for Kaiser windows). Converges to
/// double precision in < 30 terms for the betas used here (<= 20).
[[nodiscard]] inline double besselI0(double x) noexcept
{
    double sum = 1.0;
    double term = 1.0;
    const double q = 0.25 * x * x;
    for (int k = 1; k < 64; ++k)
    {
        term *= q / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
        if (term < sum * 1e-17)
            break;
    }
    return sum;
}

/// Kaiser window at position u in [-1, 1] (0 outside).
[[nodiscard]] inline double kaiser(double u, double beta) noexcept
{
    if (u <= -1.0 || u >= 1.0)
        return 0.0;
    return besselI0(beta * std::sqrt(1.0 - u * u)) / besselI0(beta);
}

[[nodiscard]] inline double sincPi(double x) noexcept
{
    if (std::fabs(x) < 1e-12)
        return 1.0;
    const double px = 3.14159265358979323846 * x;
    return std::sin(px) / px;
}

/// Coefficients of the 8-point windowed-sinc fractional delay, tabulated at kPhases fractions
/// (plus one guard row) and linearly interpolated between rows. Row r holds the weights for
/// f = r / kPhases, for the taps at offsets -3..+4 from y0. Each row is normalised to unity DC
/// gain so reads of constant signals are exact.
struct SincTable
{
    static constexpr int kTaps = 8;
    static constexpr int kPhases = 512;
    static constexpr int kBefore = 3; ///< newer taps used before y0; the minimum read delay
    static constexpr double kBeta = 6.0;

    std::array<float, static_cast<std::size_t>((kPhases + 1) * kTaps)> w{};

    SincTable() noexcept
    {
        for (int r = 0; r <= kPhases; ++r)
        {
            const double f = static_cast<double>(r) / kPhases;
            std::array<double, kTaps> row{};
            double sum = 0.0;
            for (int j = 0; j < kTaps; ++j)
            {
                // Tap j sits at offset k = j - 3 samples further back than y0; the ideal read is
                // at offset f, so the kernel is evaluated at k - f.
                const double x = static_cast<double>(j - kBefore) - f;
                row[static_cast<std::size_t>(j)] = sincPi(x) * kaiser(x / 4.0, kBeta);
                sum += row[static_cast<std::size_t>(j)];
            }
            for (int j = 0; j < kTaps; ++j)
                w[static_cast<std::size_t>(r * kTaps + j)] =
                    static_cast<float>(row[static_cast<std::size_t>(j)] / sum);
        }
    }

    /// Writes the 8 weights for fraction f in [0, 1).
    void weights(float f, float* out) const noexcept
    {
        const float pos = f * static_cast<float>(kPhases);
        int r = static_cast<int>(pos);
        r = r < 0 ? 0 : (r >= kPhases ? kPhases - 1 : r);
        const float t = pos - static_cast<float>(r);
        const float* a = w.data() + r * kTaps;
        const float* b = a + kTaps;
        for (int j = 0; j < kTaps; ++j)
            out[j] = a[j] + t * (b[j] - a[j]);
    }
};

/// The shared table (16 KB, built once). DelayLine::prepare touches it so the one-time build
/// never happens on the audio thread.
[[nodiscard]] inline const SincTable& sincTable() noexcept
{
    static const SincTable table;
    return table;
}

/// First-order Thiran allpass coefficient for a fractional delay delta, best in [0.5, 1.5):
/// H(z) = (a + z^-1) / (1 + a z^-1) has a low-frequency delay of delta samples.
[[nodiscard]] inline float thiranCoefficient(float delta) noexcept
{
    return (1.0f - delta) / (1.0f + delta);
}

} // namespace alfa::dsp::detail
