#pragma once

// Shared helpers for the DSP tests: an allocation counter for the no-allocation checks, and a
// few signal measurements.

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace alfa::dsp::test
{

/// Counts global operator new calls made on this thread while alive. Use it around audio-path
/// calls: REQUIRE(guard.count() == 0).
class AllocationGuard
{
public:
    AllocationGuard() noexcept;
    ~AllocationGuard();
    AllocationGuard(const AllocationGuard&) = delete;
    AllocationGuard& operator=(const AllocationGuard&) = delete;
    [[nodiscard]] long count() const noexcept;
};

inline double rms(const float* x, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += static_cast<double>(x[i]) * x[i];
    return n != 0 ? std::sqrt(s / static_cast<double>(n)) : 0.0;
}

inline double rms(const std::vector<float>& x)
{
    return rms(x.data(), x.size());
}

/// Magnitude of the DFT of x at frequency hz (single-bin Goertzel-style), normalised so a
/// sinusoid of amplitude A that fills x reads about A.
inline double toneAmplitude(const std::vector<float>& x, double hz, double sampleRate)
{
    std::complex<double> acc{};
    const double w = 2.0 * 3.14159265358979323846 * hz / sampleRate;
    for (std::size_t n = 0; n < x.size(); ++n)
    {
        // Hann window to keep leakage from neighbouring partials down.
        const double win = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * static_cast<double>(n) /
                                                static_cast<double>(x.size()));
        acc += static_cast<double>(x[n]) * win * std::polar(1.0, -w * static_cast<double>(n));
    }
    return 4.0 * std::abs(acc) / static_cast<double>(x.size());
}

/// Fundamental estimate by counting upward zero crossings (good for clean, single-pitch signals).
inline double zeroCrossingHz(const std::vector<float>& x, double sampleRate)
{
    int first = -1, count = 0;
    double firstPos = 0.0, lastPos = 0.0;
    for (std::size_t n = 1; n < x.size(); ++n)
    {
        if (x[n - 1] < 0.0f && x[n] >= 0.0f)
        {
            const double pos = static_cast<double>(n - 1) + x[n - 1] / (x[n - 1] - x[n]);
            if (first < 0)
            {
                first = static_cast<int>(n);
                firstPos = pos;
            }
            lastPos = pos;
            ++count;
        }
    }
    if (count < 2)
        return 0.0;
    return (count - 1) * sampleRate / (lastPos - firstPos);
}

} // namespace alfa::dsp::test
