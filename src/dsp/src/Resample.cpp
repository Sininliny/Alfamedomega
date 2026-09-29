#include "alfa/dsp/Resample.h"

#include "alfa/dsp/detail/DelayInterpolation.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp
{

namespace
{

constexpr int kResolution = 512;         // real-time table points per zero crossing
constexpr int kOfflineResolution = 4096; // offline table points per zero crossing

/// Normalised cutoff (fraction of Nyquist) that puts the end of the Kaiser transition band at
/// Nyquist: the window's transition width is about (A - 8) / (2.285 N 2 pi) of fs, with the
/// stopband attenuation A recovered from beta (Kaiser's formula, A > 50 dB).
double cutoffFor(int taps, double beta) noexcept
{
    const double attenuation = beta / 0.1102 + 8.7;
    const double width =
        (attenuation - 8.0) / (2.285 * 2.0 * 3.14159265358979323846 * static_cast<double>(taps));
    return std::clamp(1.0 - width, 0.5, 0.99);
}

/// h(t) = fc sinc(fc t) kaiser(t / half), sampled at resolution points per unit t for
/// t = 0 .. half (plus a zero guard), scaled so the integer-spaced samples sum to exactly 1.
std::vector<double> designKernel(int half, double beta, int resolution)
{
    const double fc = cutoffFor(2 * half, beta);
    const auto count = static_cast<std::size_t>(half * resolution + 2);
    std::vector<double> h(count, 0.0);
    for (std::size_t i = 0; i + 1 < count; ++i)
    {
        const double t = static_cast<double>(i) / resolution;
        h[i] = fc * detail::sincPi(fc * t) * detail::kaiser(t / half, beta);
    }
    double sum = h[0];
    for (int j = 1; j < half; ++j)
        sum += 2.0 * h[static_cast<std::size_t>(j * resolution)];
    for (auto& v : h)
        v /= sum;
    return h;
}

template <typename T>
T lookup(const std::vector<T>& table, int half, int resolution, T t) noexcept
{
    const T a = std::fabs(t) * static_cast<T>(resolution);
    if (!(a < static_cast<T>(half * resolution)))
        return T(0);
    const auto i = static_cast<std::size_t>(a);
    const T f = a - static_cast<T>(i);
    return table[i] + f * (table[i + 1] - table[i]);
}

} // namespace

// PolyphaseResampler ----------------------------------------------------------------------------

void PolyphaseResampler::prepare(int taps, double maxRatio, double kaiserBeta)
{
    half_ = std::clamp((taps + 1) / 2, 4, 128);
    maxRatio_ = std::max(1.0, maxRatio);
    const auto h = designKernel(half_, kaiserBeta, kResolution);
    table_.assign(h.size(), 0.0f);
    for (std::size_t i = 0; i < h.size(); ++i)
        table_[i] = static_cast<float>(h[i]);

    const auto reach = static_cast<std::size_t>(std::ceil(static_cast<double>(half_) * maxRatio_));
    std::size_t size = 1;
    while (size < 2 * reach + 8)
        size <<= 1;
    history_.assign(size, 0.0f);
    mask_ = size - 1;
    ratio_ = 1.0;
    reset();
}

void PolyphaseResampler::reset() noexcept
{
    std::fill(history_.begin(), history_.end(), 0.0f);
    received_ = 0;
    baseIndex_ = 0;
    frac_ = 0.0;
}

void PolyphaseResampler::setRatio(double ratio) noexcept
{
    if (!(ratio > 0.0))
        return; // also rejects NaN
    ratio_ = std::clamp(ratio, 1.0 / 64.0, maxRatio_);
}

double PolyphaseResampler::latency() const noexcept
{
    return static_cast<double>(half_) * std::max(1.0, ratio_);
}

float PolyphaseResampler::kernel(float t) const noexcept
{
    return lookup(table_, half_, kResolution, t);
}

float PolyphaseResampler::evaluate() const noexcept
{
    if (ratio_ <= 1.0)
    {
        // Unstretched kernel: consecutive taps sit exactly one zero crossing (kResolution table
        // points) apart, so the table index advances by a constant and the interpolation
        // fraction is shared by all taps on each side of the output time.
        const auto frac = static_cast<float>(frac_);
        const float* h = table_.data();
        float acc = 0.0f;
        // Taps at or before the output time: j = base - k, t = frac + k, k = 0 .. half-1.
        {
            const float a = frac * static_cast<float>(kResolution);
            auto i = static_cast<std::size_t>(a);
            const float f = a - static_cast<float>(i);
            for (int k = 0; k < half_; ++k, i += kResolution)
            {
                const float w = h[i] + f * (h[i + 1] - h[i]);
                acc += w * history_[static_cast<std::size_t>(baseIndex_ - k) & mask_];
            }
        }
        // Taps after it: j = base + k, t = k - frac, k = 1 .. half.
        {
            const float a = (1.0f - frac) * static_cast<float>(kResolution);
            auto i = static_cast<std::size_t>(a);
            const float f = a - static_cast<float>(i);
            for (int k = 1; k <= half_; ++k, i += kResolution)
            {
                const float w = h[i] + f * (h[i + 1] - h[i]);
                acc += w * history_[static_cast<std::size_t>(baseIndex_ + k) & mask_];
            }
        }
        return acc;
    }

    const auto s = static_cast<float>(std::min(1.0, 1.0 / ratio_));
    const auto reach =
        static_cast<std::int64_t>(std::ceil(static_cast<double>(half_) / static_cast<double>(s)));
    const auto frac = static_cast<float>(frac_);
    float acc = 0.0f;
    for (std::int64_t k = -reach + 1; k <= reach; ++k)
    {
        // Input j = base + k sits at t = frac - k from the output time.
        const float x = history_[static_cast<std::size_t>(baseIndex_ + k) & mask_];
        acc += x * kernel(s * (frac - static_cast<float>(k)));
    }
    return s * acc;
}

int PolyphaseResampler::process(const float* in, int numIn, float* out, int maxOut, int& consumed) noexcept
{
    consumed = 0;
    int produced = 0;
    const auto reach =
        static_cast<std::int64_t>(std::ceil(static_cast<double>(half_) * std::max(1.0, ratio_)));
    while (produced < maxOut)
    {
        if (received_ <= baseIndex_ + reach)
        {
            if (consumed >= numIn)
                break;
            float x = in[consumed++];
            if (!std::isfinite(x))
                x = 0.0f; // one bad input must not poison the whole kernel span
            history_[static_cast<std::size_t>(received_) & mask_] = x;
            ++received_;
            continue;
        }
        out[produced++] = evaluate();
        frac_ += ratio_;
        const double whole = std::floor(frac_);
        baseIndex_ += static_cast<std::int64_t>(whole);
        frac_ -= whole;
    }
    return produced;
}

float PolyphaseResampler::sampleAt(const float* data, std::int64_t length, double position,
                                   double ratio) const noexcept
{
    if (!std::isfinite(position) || length <= 0)
        return 0.0f;
    const double s = std::min(1.0, 1.0 / std::clamp(ratio, 1.0 / 64.0, maxRatio_));
    const auto reach = static_cast<std::int64_t>(std::ceil(static_cast<double>(half_) / s));
    const double base = std::floor(position);
    const auto i0 = static_cast<std::int64_t>(base);
    const double frac = position - base;
    const std::int64_t lo = std::max<std::int64_t>(i0 - reach + 1, 0);
    const std::int64_t hi = std::min<std::int64_t>(i0 + reach, length - 1);
    float acc = 0.0f;
    for (std::int64_t j = lo; j <= hi; ++j)
    {
        const auto t = static_cast<float>(s * (frac + static_cast<double>(i0 - j)));
        acc += data[j] * kernel(t);
    }
    return static_cast<float>(s) * acc;
}

// Offline ---------------------------------------------------------------------------------------

std::vector<float> resampleOffline(const float* in, std::size_t n, double fromRate, double toRate,
                                   int quality)
{
    if (in == nullptr || n == 0 || !(fromRate > 0.0) || !(toRate > 0.0))
        return {};
    constexpr int kHalf[] = {16, 32, 64, 128};
    constexpr double kBeta[] = {6.0, 8.0, 10.0, 12.0};
    const auto q = static_cast<std::size_t>(std::clamp(quality, 0, 3));
    const int half = kHalf[q];
    const auto table = designKernel(half, kBeta[q], kOfflineResolution);

    const double step = fromRate / toRate; // input samples per output sample
    const double s = std::min(1.0, toRate / fromRate);
    const auto reach = static_cast<std::int64_t>(std::ceil(static_cast<double>(half) / s));
    const auto outLength = static_cast<std::size_t>(std::llround(static_cast<double>(n) * toRate / fromRate));
    const auto last = static_cast<std::int64_t>(n) - 1;

    std::vector<float> out(outLength, 0.0f);
    for (std::size_t m = 0; m < outLength; ++m)
    {
        const double tau = static_cast<double>(m) * step;
        const auto centre = static_cast<std::int64_t>(std::floor(tau));
        const std::int64_t lo = std::max<std::int64_t>(centre - reach + 1, 0);
        const std::int64_t hi = std::min<std::int64_t>(centre + reach, last);
        double acc = 0.0;
        for (std::int64_t j = lo; j <= hi; ++j)
            acc += static_cast<double>(in[j]) *
                   lookup(table, half, kOfflineResolution, s * (tau - static_cast<double>(j)));
        out[m] = static_cast<float>(s * acc);
    }
    return out;
}

} // namespace alfa::dsp
