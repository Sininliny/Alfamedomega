#include "alfa/dsp/Fm.h"

#include "alfa/dsp/detail/FastMath.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace alfa::dsp
{

namespace
{

constexpr float kInvTwoPi = 1.0f / detail::kTwoPi;

inline std::size_t u(int i) noexcept
{
    return static_cast<std::size_t>(i);
}

inline double wrapUnit(double x) noexcept
{
    const double w = x - std::floor(x);
    return w >= 1.0 ? 0.0 : w;
}

} // namespace

void FmPrimitive::prepare(const PrepareSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    hop_ = std::max(1, spec.hop);
    numOps_ = std::clamp(spec.limits.fmOperators, 0, kOps);
    reset();
}

void FmPrimitive::reset() noexcept
{
    phase_.fill(0.0);
    numActive_ = 0;
    activeMask_ = 0;
    g_ = gInc_ = 0.0;
    pos_ = 0;
}

void FmPrimitive::loadRamps(const HopContext& ctx) noexcept
{
    const FmControls& cur = ctx.cur->fm;
    const FmControls& prev = (ctx.prev != nullptr ? ctx.prev : ctx.cur)->fm;
    const double invH = 1.0 / hop_;
    gStart_ = static_cast<double>(ctx.f0Start) / sampleRate_;
    gStep_ = (static_cast<double>(ctx.f0End) - static_cast<double>(ctx.f0Start)) / sampleRate_ * invH;
    for (int i = 0; i < numOps_; ++i)
    {
        ratioStart_[u(i)] = prev.ratio[u(i)];
        ratioStep_[u(i)] = (static_cast<double>(cur.ratio[u(i)]) - prev.ratio[u(i)]) * invH;
    }
}

void FmPrimitive::advanceClosedForm(unsigned mask) noexcept
{
    // sum_{k=0}^{H-1} (g0 + dg k)(r0 + dr k) = H g0 r0 + (g0 dr + r0 dg) S1 + dg dr S2
    const auto h = static_cast<double>(hop_);
    const double s1 = h * (h - 1.0) * 0.5;
    const double s2 = (h - 1.0) * h * (2.0 * h - 1.0) / 6.0;
    for (int i = 0; i < numOps_; ++i)
    {
        if ((mask & (1u << static_cast<unsigned>(i))) == 0)
            continue;
        const double r0 = ratioStart_[u(i)];
        const double dr = ratioStep_[u(i)];
        const double adv = h * gStart_ * r0 + (gStart_ * dr + r0 * gStep_) * s1 + gStep_ * dr * s2;
        phase_[u(i)] = wrapUnit(phase_[u(i)] + adv);
    }
}

void FmPrimitive::beginHop(const HopContext& ctx) noexcept
{
    numActive_ = 0;
    activeMask_ = 0;
    if (ctx.cur == nullptr)
        return;
    loadRamps(ctx);

    const FmControls& cur = ctx.cur->fm;
    const FmControls& prev = (ctx.prev != nullptr ? ctx.prev : ctx.cur)->fm;

    // An operator is needed when it is heard or modulates a needed operator (j > i only, so one
    // ascending pass settles it).
    for (int i = 0; i < numOps_; ++i)
    {
        bool need = cur.level[u(i)] != 0.0f || prev.level[u(i)] != 0.0f;
        for (int j = 0; j < i && !need; ++j)
        {
            if ((activeMask_ & (1u << static_cast<unsigned>(j))) == 0)
                continue;
            const auto e = u(j * kOps + i);
            need = cur.index[e] != 0.0f || prev.index[e] != 0.0f;
        }
        if (need)
        {
            activeMask_ |= 1u << static_cast<unsigned>(i);
            active_[u(numActive_++)] = i;
        }
    }

    // Unused operators only keep time.
    const unsigned all = numOps_ >= 32 ? ~0u : ((1u << static_cast<unsigned>(numOps_)) - 1u);
    advanceClosedForm(all & ~activeMask_);

    const float invH = 1.0f / static_cast<float>(hop_);
    pos_ = 0;
    g_ = gStart_;
    gInc_ = gStep_;
    for (int a = 0; a < numActive_; ++a)
    {
        const int i = active_[u(a)];
        ratio_[u(a)] = ratioStart_[u(i)];
        ratioInc_[u(a)] = ratioStep_[u(i)];
        level_[u(a)] = prev.level[u(i)];
        levelInc_[u(a)] = (cur.level[u(i)] - prev.level[u(i)]) * invH;
        for (int b = a + 1; b < numActive_; ++b)
        {
            const auto e = u(i * kOps + active_[u(b)]);
            const auto c = u(a * kOps + b);
            mod_[c] = prev.index[e] * kInvTwoPi;
            modInc_[c] = (cur.index[e] - prev.index[e]) * kInvTwoPi * invH;
        }
    }
}

void FmPrimitive::render(float* out, int n) noexcept
{
    const int na = numActive_;
    if (na == 0)
    {
        std::fill(out, out + n, 0.0f);
        pos_ += n;
        return;
    }

    // Samples within a block are independent (no feedback), so each operator is evaluated for a
    // whole block at a time: the phase recurrence runs first (double), then the operators from
    // last to first as straight loops over the block.
    alignas(32) float phs[kOps][kBlock];
    alignas(32) float y[kOps][kBlock];
    alignas(32) float kk[kBlock];
    for (int k = 0; k < kBlock; ++k)
        kk[k] = static_cast<float>(k);

    while (n > 0)
    {
        const int bn = std::min(n, kBlock);
        const auto p0 = static_cast<float>(pos_);

        const double dg = gInc_;
        for (int a = 0; a < na; ++a)
        {
            double p = phase_[u(active_[u(a)])];
            double g = g_;
            double r = ratio_[u(a)];
            const double dr = ratioInc_[u(a)];
            float* ph = phs[a];
            for (int k = 0; k < bn; ++k)
            {
                ph[k] = static_cast<float>(p);
                p += g * r;
                if (p >= 1.0 || p < 0.0)
                    p = wrapUnit(p);
                g += dg;
                r += dr;
            }
            phase_[u(active_[u(a)])] = p;
            ratio_[u(a)] = r;
        }
        g_ += dg * bn;

        alignas(32) float acc[kBlock];
        std::fill(acc, acc + bn, 0.0f);
        for (int a = na - 1; a >= 0; --a)
        {
            float* ya = y[a];
            const float* ph = phs[a];
            std::copy(ph, ph + bn, ya);
            for (int b = a + 1; b < na; ++b)
            {
                const float dm = modInc_[u(a * kOps + b)];
                const float m0 = mod_[u(a * kOps + b)] + dm * p0;
                const float* yb = y[b];
#pragma omp simd
                for (int k = 0; k < bn; ++k)
                    ya[k] += (m0 + dm * kk[k]) * yb[k];
            }
#pragma omp simd
            for (int k = 0; k < bn; ++k)
                ya[k] = detail::sin2pi(ya[k]);

            const float dl = levelInc_[u(a)];
            const float l0 = level_[u(a)] + dl * p0;
#pragma omp simd
            for (int k = 0; k < bn; ++k)
                acc[k] += (l0 + dl * kk[k]) * ya[k];
        }
        std::copy(acc, acc + bn, out);

        out += bn;
        n -= bn;
        pos_ += bn;
    }
}

void FmPrimitive::skipHop(const HopContext& ctx) noexcept
{
    numActive_ = 0;
    activeMask_ = 0;
    if (ctx.cur == nullptr)
        return;
    loadRamps(ctx);
    const unsigned all = numOps_ >= 32 ? ~0u : ((1u << static_cast<unsigned>(numOps_)) - 1u);
    advanceClosedForm(all);
}

} // namespace alfa::dsp
