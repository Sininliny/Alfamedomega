#include "LogMel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>

namespace alfa::inference
{
namespace
{

constexpr double kSampleRate = 48000.0;
constexpr double kMelLowHz = 20.0;
constexpr double kMelHighHz = 16000.0;

void fft(std::vector<std::complex<double>>& a)
{
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i)
    {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1)
    {
        const double angle = -2.0 * std::numbers::pi / static_cast<double>(len);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len)
        {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k)
            {
                const auto u = a[i + k];
                const auto v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= step;
            }
        }
    }
}

double hzToMel(double hz)
{
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double melToHz(double mel)
{
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

// [kMelBands x (kMelFftSize / 2 + 1)], as training builds it.
std::vector<double> melFilterbank()
{
    constexpr std::size_t bins = kMelFftSize / 2 + 1;
    std::vector<double> points(kMelBands + 2);
    const double lo = hzToMel(kMelLowHz), hi = hzToMel(kMelHighHz);
    for (std::size_t i = 0; i < points.size(); ++i)
        points[i] = melToHz(lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(kMelBands + 1));

    std::vector<double> w(kMelBands * bins, 0.0);
    for (std::size_t m = 0; m < kMelBands; ++m)
    {
        const double a = points[m], c = points[m + 1], b = points[m + 2];
        for (std::size_t k = 0; k < bins; ++k)
        {
            const double f = static_cast<double>(k) * kSampleRate / static_cast<double>(kMelFftSize);
            w[m * bins + k] = std::max(0.0, std::min((f - a) / (c - a), (b - f) / (b - c)));
        }
    }
    return w;
}

} // namespace

std::vector<float> logMel(std::span<const float> x, std::size_t& frames)
{
    constexpr std::size_t pad = kMelFftSize / 2;
    constexpr std::size_t bins = kMelFftSize / 2 + 1;
    frames = 0;
    if (x.size() <= pad)
        return {};

    // Reflect padding, as torch.stft(center=True, pad_mode="reflect").
    const auto length = x.size();
    const auto sample = [&](std::ptrdiff_t i) -> double
    {
        const auto n = static_cast<std::ptrdiff_t>(length);
        if (i < 0)
            i = -i;
        if (i >= n)
            i = 2 * (n - 1) - i;
        return static_cast<double>(x[static_cast<std::size_t>(i)]);
    };

    frames = 1 + length / kMelHop;
    static const auto filters = melFilterbank();
    std::vector<double> window(kMelFftSize);
    for (std::size_t n = 0; n < kMelFftSize; ++n)
        window[n] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(n) / kMelFftSize);

    std::vector<float> out(kMelBands * frames);
    std::vector<std::complex<double>> buffer(kMelFftSize);
    std::vector<double> magnitude(bins);
    for (std::size_t t = 0; t < frames; ++t)
    {
        const auto start = static_cast<std::ptrdiff_t>(t * kMelHop) - static_cast<std::ptrdiff_t>(pad);
        for (std::size_t n = 0; n < kMelFftSize; ++n)
            buffer[n] = {sample(start + static_cast<std::ptrdiff_t>(n)) * window[n], 0.0};
        fft(buffer);
        for (std::size_t k = 0; k < bins; ++k)
            magnitude[k] = std::abs(buffer[k]);
        for (std::size_t m = 0; m < kMelBands; ++m)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < bins; ++k)
                sum += filters[m * bins + k] * magnitude[k];
            out[m * frames + t] = static_cast<float>(std::log(sum + 1e-5));
        }
    }
    return out;
}

} // namespace alfa::inference
