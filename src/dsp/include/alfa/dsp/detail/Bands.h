#pragma once

// The 64-band log-frequency layout shared by NOISE and the SAMPLER EQ (ControlFrame.h):
// band b is centred at 20 * (nyquist / 20)^(b / 63) Hz; a bin's gain is linear in log-frequency
// between the two neighbouring centres, flat below band 0 (DC included) and above band 63.

#include "alfa/dsp/ControlFrame.h"

#include <cmath>
#include <vector>

namespace alfa::dsp::detail
{

class BandMap
{
public:
    /// Precompute the interpolation for an FFT of fftSize bins at sampleRate. Allocates.
    void prepare(double sampleRate, int fftSize)
    {
        const int bins = fftSize / 2 + 1;
        lo_.assign(static_cast<std::size_t>(bins), 0);
        frac_.assign(static_cast<std::size_t>(bins), 0.0f);
        const double nyquist = 0.5 * sampleRate;
        const double span = std::log(nyquist / kBandLowHz);
        for (int k = 1; k < bins; ++k)
        {
            const double f = static_cast<double>(k) * sampleRate / fftSize;
            double pos = (kBands - 1) * std::log(f / kBandLowHz) / span; // fractional band index
            if (pos < 0.0)
                pos = 0.0;
            if (pos > kBands - 1)
                pos = kBands - 1;
            int b = static_cast<int>(pos);
            if (b >= kBands - 1)
                b = kBands - 2;
            lo_[static_cast<std::size_t>(k)] = b;
            frac_[static_cast<std::size_t>(k)] = static_cast<float>(pos - b);
        }
        bins_ = bins;
    }

    [[nodiscard]] int bins() const noexcept { return bins_; }

    /// Gain of bin k (0 .. fftSize/2) for the given band gains.
    [[nodiscard]] float gain(const float* bands, int k) const noexcept
    {
        const auto i = static_cast<std::size_t>(k);
        const int b = lo_[i];
        const float f = frac_[i];
        return bands[b] + f * (bands[b + 1] - bands[b]);
    }

private:
    std::vector<int> lo_;
    std::vector<float> frac_;
    int bins_ = 0;
};

/// Periodic Hann window of length n: 0.5 - 0.5 cos(2 pi t / n). Two of them at a hop of n/2 sum
/// to exactly 1. Allocates.
inline std::vector<float> periodicHann(int n)
{
    std::vector<float> w(static_cast<std::size_t>(n));
    for (int t = 0; t < n; ++t)
        w[static_cast<std::size_t>(t)] =
            static_cast<float>(0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * t / n));
    return w;
}

} // namespace alfa::dsp::detail
