#include "alfa/dsp/Partials.h"

#include "alfa/dsp/detail/FastMath.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace alfa::dsp
{

namespace
{
/// sin(2 pi x) for |x| < 2^22 cycles: the same fold and polynomial as detail::sin2pi, written
/// without std::floor or compares. With the default -ftrapping-math and baseline SSE2, GCC does
/// not vectorise detail::sin2pi (floor becomes a branch); this form vectorises.
inline float sin2piVec(float x) noexcept
{
    constexpr float kRound = 12582912.0f;        // 1.5 * 2^23: (x + kRound) - kRound rounds to nearest
    const float r = x - ((x + kRound) - kRound); // [-0.5, 0.5]
    const float m = std::copysign(0.25f - std::fabs(std::fabs(r) - 0.25f), r); // [-0.25, 0.25]
    const float t = detail::kTwoPi * m;
    const float t2 = t * t;
    float p = -2.5052108e-8f;
    p = p * t2 + 2.7557319e-6f;
    p = p * t2 - 1.9841270e-4f;
    p = p * t2 + 8.3333333e-3f;
    p = p * t2 - 1.6666667e-1f;
    p = p * t2 + 1.0f;
    return p * t;
}

/// Keeps a phase in (-1, 1) with a truncation, which vectorises (std::floor does not, see above).
inline float wrapPhase(float p) noexcept
{
    return p - static_cast<float>(static_cast<int>(p));
}
} // namespace

void PartialsPrimitive::prepare(const PrepareSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    invSampleRate_ = 1.0 / spec.sampleRate;
    nyquist_ = static_cast<float>(0.5 * spec.sampleRate);
    hop_ = spec.hop > 0 ? spec.hop : 1;
    invHop_ = 1.0f / static_cast<float>(hop_);
    limit_ = std::clamp(spec.limits.partials, 0, kMaxPartials);
    reset();
}

void PartialsPrimitive::reset() noexcept
{
    phase_.fill(0.0);
    count_ = 0;
    pos_ = 0;
}

double PartialsPrimitive::phase(int m) const noexcept
{
    return (m >= 0 && m < kMaxPartials) ? phase_[static_cast<std::size_t>(m)] : 0.0;
}

void PartialsPrimitive::startHop(const HopContext& ctx, bool collect) noexcept
{
    count_ = 0;
    safeCount_ = 0;
    pos_ = 0;
    f0Start_ = ctx.f0Start;
    f0Delta_ = ctx.f0End - ctx.f0Start;
    if (ctx.cur == nullptr)
        return;
    const PartialsControls& b = ctx.cur->partials;
    const PartialsControls& a = ctx.prev != nullptr ? ctx.prev->partials : b;

    // Phase advance over the hop, sum_{n=0}^{H-1} f0(n) ratio(n) / sr with
    // f0(n) = f0s + df n and ratio(n) = r0 + dr n (df, dr per sample): a quadratic in n.
    const double h = static_cast<double>(hop_);
    const double s1 = h * (h - 1.0) * 0.5;
    const double s2 = (h - 1.0) * h * (2.0 * h - 1.0) / 6.0;
    const double f0s = static_cast<double>(ctx.f0Start);
    const double df = static_cast<double>(ctx.f0End - ctx.f0Start) / h;
    std::array<int, kN> entries{};
    int safeCount = 0;
    int crossingCount = 0;

    for (int m = 0; m < limit_; ++m)
    {
        const auto i = static_cast<std::size_t>(m);
        const float r0 = a.ratio[i];
        const float r1 = b.ratio[i];
        const double start = phase_[i];
        startPhase_[i] = start;

        const double dr = static_cast<double>(r1 - r0) / h;
        const double cycles = (h * f0s * static_cast<double>(r0) +
                               (f0s * dr + df * static_cast<double>(r0)) * s1 + df * dr * s2) *
                              invSampleRate_;
        const double end = start + cycles;
        phase_[i] = end - std::floor(end);

        if (!collect || (a.amp[i] == 0.0f && b.amp[i] == 0.0f))
            continue;

        // |frequency| over the hop's samples n = 0..H-1: f(n) = (f0s + df n)(r0 + dr n) is a
        // quadratic, so its extremes are at the ends or at the vertex.
        const auto freqAt = [&](double n)
        {
            return (f0s + df * n) * (static_cast<double>(r0) + dr * n);
        };
        const double fa = freqAt(0.0);
        const double fb = freqAt(h - 1.0);
        double lo = std::min(fa, fb);
        double hi = std::max(fa, fb);
        const double curv = df * dr;
        if (curv != 0.0)
        {
            const double vertex = -(f0s * dr + df * static_cast<double>(r0)) / (2.0 * curv);
            if (vertex > 0.0 && vertex < h - 1.0)
            {
                lo = std::min(lo, freqAt(vertex));
                hi = std::max(hi, freqAt(vertex));
            }
        }
        const double maxAbs = std::max(std::fabs(lo), std::fabs(hi));
        const double minAbs = (lo <= 0.0 && hi >= 0.0) ? 0.0 : std::min(std::fabs(lo), std::fabs(hi));
        const double nyq = static_cast<double>(nyquist_);
        if (minAbs >= nyq * (1.0 + 1e-5))
            continue; // at or above Nyquist for the whole hop: silent
        const bool crossing = maxAbs >= nyq * (1.0 - 1e-5);
        const int slot = crossing ? kMaxPartials - 1 - crossingCount++ : safeCount++;
        entries[static_cast<std::size_t>(slot)] = m;
    }
    if (!collect)
        return;

    // Audible partials first, then the ones that may cross Nyquist inside the hop (gated per
    // sample in render).
    count_ = safeCount + crossingCount;
    safeCount_ = safeCount;
    for (int k = 0; k < count_; ++k)
    {
        const int m = k < safeCount ? entries[static_cast<std::size_t>(k)]
                                    : entries[static_cast<std::size_t>(kMaxPartials - 1 - (k - safeCount))];
        const auto i = static_cast<std::size_t>(m);
        const auto j = static_cast<std::size_t>(k);
        ph_[j] = static_cast<float>(startPhase_[i]);
        ratio0_[j] = a.ratio[i];
        ratioD_[j] = b.ratio[i] - a.ratio[i];
        amp0_[j] = a.amp[i];
        ampD_[j] = b.amp[i] - a.amp[i];
    }
}

void PartialsPrimitive::beginHop(const HopContext& ctx) noexcept
{
    startHop(ctx, true);
}

void PartialsPrimitive::skipHop(const HopContext& ctx) noexcept
{
    startHop(ctx, false);
}

void PartialsPrimitive::render(float* out, int n) noexcept
{
    const int count = count_;
    if (count == 0)
    {
        std::fill(out, out + n, 0.0f);
        pos_ += n;
        return;
    }

    float* ph = ph_.data();
    const float* r0 = ratio0_.data();
    const float* rd = ratioD_.data();
    const float* a0 = amp0_.data();
    const float* ad = ampD_.data();
    const float nyq = nyquist_;
    const auto invSr = static_cast<float>(invSampleRate_);
    int pos = pos_;

    const int safe = safeCount_;
    for (int i = 0; i < n; ++i, ++pos)
    {
        const float t = static_cast<float>(pos) * invHop_;
        const float f0 = f0Start_ + f0Delta_ * t;
        float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
        for (int k = 0; k < safe; ++k)
        {
            const float f = f0 * (r0[k] + rd[k] * t);
            const float amp = a0[k] + ad[k] * t;
            acc += amp * sin2piVec(ph[k]);
            ph[k] = wrapPhase(ph[k] + f * invSr);
        }
        for (int k = safe; k < count; ++k) // rare: partials that reach Nyquist during the hop
        {
            const float f = f0 * (r0[k] + rd[k] * t);
            if (std::fabs(f) < nyq)
                acc += (a0[k] + ad[k] * t) * sin2piVec(ph[k]);
            ph[k] = wrapPhase(ph[k] + f * invSr);
        }
        out[i] = acc;
    }
    pos_ = pos;
}

} // namespace alfa::dsp
