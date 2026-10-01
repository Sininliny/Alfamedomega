#pragma once
#include <array>
#include <cmath>
#include <span>

namespace alfa::dsp::detail
{
// Stateless Kaiser halfband designer extracted from NonlinearOversampler.h's
// exact design arithmetic. Off-thread only. Output values use the caller's type.
template <class Scalar>
bool designNonlinearHalfbandDense(std::span<Scalar> output, int halfK, double beta) noexcept
{
    if (halfK < 0 || halfK > 11 || !std::isfinite(beta) || beta < 0 || beta > 30)
        return false;
    const int m = 2 * halfK + 2, center = 2 * halfK + 1;
    if (output.size() < static_cast<std::size_t>(m))
        return false;
    const auto i0 = [](double x)
    {
        double sum = 1.0, term = 1.0;
        for (int n = 1; n < 40; ++n)
        {
            const double t = x / (2.0 * n);
            term *= t * t;
            sum += term;
        }
        return sum;
    };
    double total = 0.0;
    std::array<double, 24> h{};
    for (int j = 0; j < m; ++j)
    {
        const double d = static_cast<double>(2 * j - center);
        const double pd = 3.14159265358979323846 * d / 2.0;
        const double sinc = std::sin(pd) / pd;
        const double r = d / static_cast<double>(center + 1);
        const double w = i0(beta * std::sqrt(1.0 - r * r)) / i0(beta);
        h[static_cast<std::size_t>(j)] = 0.5 * sinc * w;
        total += h[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < m; ++j)
        output[static_cast<std::size_t>(j)] =
            static_cast<Scalar>(h[static_cast<std::size_t>(j)] * (0.5 / total));
    for (std::size_t j = static_cast<std::size_t>(m); j < output.size(); ++j)
        output[j] = Scalar(0);
    return true;
}
} // namespace alfa::dsp::detail
