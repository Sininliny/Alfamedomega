#include "alfa/dsp/Osc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace alfa::dsp
{

namespace
{
constexpr int kGuard = 16; // table[kOscTableSize] repeats table[0]; the rest keeps the size aligned
constexpr std::size_t kTableAlloc = static_cast<std::size_t>(kOscTableSize + kGuard);

inline double wrap01(double x) noexcept
{
    return x - std::floor(x);
}
} // namespace

void OscPrimitive::prepare(const PrepareSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    invSampleRate_ = 1.0 / spec.sampleRate;
    hop_ = spec.hop > 0 ? spec.hop : 1;
    invHop_ = 1.0f / static_cast<float>(hop_);

    if (fft_.size() != kOscTableSize)
        fft_.prepare(kOscTableSize);
    spectrum_.resize(static_cast<std::size_t>(kOscTableSize));
    tableA_.resize(kTableAlloc);
    tableB_.resize(kTableAlloc);
    reset();
}

void OscPrimitive::reset() noexcept
{
    tableA_.clear();
    tableB_.clear();
    prev_ = tableA_.data();
    cur_ = tableA_.data();
    prevValid_ = false;
    curValid_ = false;
    started_ = false;
    phase_ = 0.0;
    phase2_ = 0.0;
    detune_ = false;
    pos_ = 0;
}

void OscPrimitive::buildTable(const VoiceFrame* frame, float f0, float* table) noexcept
{
    if (frame == nullptr)
    {
        std::memset(table, 0, kTableAlloc * sizeof(float));
        return;
    }
    const OscControls& controls = frame->osc;

    // Harmonics h = 1..count satisfy h * f0 < Nyquist.
    int count = kOscHarmonics;
    if (f0 > 0.0f)
    {
        const double q = 0.5 * sampleRate_ / static_cast<double>(f0);
        const double c = std::ceil(q) - 1.0;
        count =
            c <= 0.0 ? 0 : (c >= static_cast<double>(kOscHarmonics) ? kOscHarmonics : static_cast<int>(c));
    }

    float* s = spectrum_.data();
    // pffft ordered layout: s[0] = DC, s[1] = Nyquist (bin 1024 = harmonic 1024, cosine part only),
    // s[2k], s[2k+1] = bin k. inverse() computes 2 Re(S_k e^{i theta}), so S_k = (re, im) / 2 gives
    // re cos - im sin.
    s[0] = 0.0f;
    s[1] = count >= kOscHarmonics ? controls.re[static_cast<std::size_t>(kOscHarmonics - 1)] : 0.0f;
    const int below = std::min(count, kOscHarmonics - 1); // harmonics that land on bins 1..1023
    const float* re = controls.re.data();
    const float* im = controls.im.data();
    for (int k = 0; k < below; ++k)
    {
        s[2 * k + 2] = 0.5f * re[k];
        s[2 * k + 3] = 0.5f * im[k];
    }
    if (below < kOscHarmonics - 1)
        std::memset(s + 2 * below + 2, 0,
                    static_cast<std::size_t>(2 * (kOscHarmonics - 1 - below)) * sizeof(float));

    fft_.inverse(s, table);
    table[kOscTableSize] = table[0];
}

void OscPrimitive::rotate(const HopContext& ctx) noexcept
{
    if (ctx.firstHop || !started_)
    {
        // Frame -1 is frame 0: both tables are the current one.
        prev_ = tableA_.data();
        cur_ = tableA_.data();
        prevValid_ = false;
        curValid_ = false;
        prevF0_ = ctx.f0Start;
        started_ = true;
    }
    else
    {
        prev_ = cur_;
        prevValid_ = curValid_;
        prevF0_ = curF0_;
        cur_ = prev_ == tableA_.data() ? tableB_.data() : tableA_.data();
        curValid_ = false;
    }
    // Detuned second read: its ratio and level ramp like every other control. While it sounds,
    // the Nyquist cut must hold for the higher of the two reads.
    const VoiceFrame* prevFrame = ctx.prev != nullptr ? ctx.prev : ctx.cur;
    if (ctx.cur != nullptr && prevFrame != nullptr)
    {
        ratioStart_ = prevFrame->oscDetuneRatio;
        ratioDelta_ = ctx.cur->oscDetuneRatio - ratioStart_;
        levelStart_ = prevFrame->oscDetuneLevel;
        levelDelta_ = ctx.cur->oscDetuneLevel - levelStart_;
    }
    else
    {
        ratioStart_ = 1.0f;
        ratioDelta_ = levelStart_ = levelDelta_ = 0.0f;
    }
    detune_ = levelStart_ != 0.0f || levelStart_ + levelDelta_ != 0.0f;
    const float cutScale = detune_ ? std::max(1.0f, std::max(ratioStart_, ratioStart_ + ratioDelta_)) : 1.0f;
    curF0_ = ctx.f0Start * cutScale;
    f0Start_ = ctx.f0Start;
    f0Delta_ = ctx.f0End - ctx.f0Start;
    pos_ = 0;
}

void OscPrimitive::beginHop(const HopContext& ctx) noexcept
{
    const bool first = ctx.firstHop || !started_;
    rotate(ctx);
    buildTable(ctx.cur, curF0_, cur_);
    curValid_ = true;
    if (first)
    {
        prev_ = cur_;
        prevValid_ = true;
    }
    else if (!prevValid_)
    {
        // The previous hop was skipped, so frame j-1's table was never built.
        buildTable(ctx.prev != nullptr ? ctx.prev : ctx.cur, prevF0_, prev_);
        prevValid_ = true;
    }
}

void OscPrimitive::render(float* out, int n) noexcept
{
    const float* a = prev_;
    const float* b = cur_;
    double phase = phase_;
    int pos = pos_;
    const double invSr = invSampleRate_;
    constexpr double kSize = static_cast<double>(kOscTableSize);

    for (int i = 0; i < n; ++i, ++pos)
    {
        const float t = static_cast<float>(pos) * invHop_;
        const double x = phase * kSize;
        const int idx = static_cast<int>(x);
        const float frac = static_cast<float>(x - static_cast<double>(idx));
        const auto u = static_cast<std::size_t>(idx);
        const float va = a[u] + frac * (a[u + 1] - a[u]);
        const float vb = b[u] + frac * (b[u + 1] - b[u]);
        out[i] = va + t * (vb - va);

        const float f0 = f0Start_ + f0Delta_ * t;
        phase += static_cast<double>(f0) * invSr;
        if (phase >= 1.0)
            phase -= 1.0;
        if (phase >= 1.0 || phase < 0.0) // f0 above the sample rate or negative
            phase = wrap01(phase);
    }
    phase_ = phase;

    if (detune_)
    {
        double phase2 = phase2_;
        int p2 = pos_;
        for (int i = 0; i < n; ++i, ++p2)
        {
            const float t = static_cast<float>(p2) * invHop_;
            const double x = phase2 * kSize;
            const int idx = static_cast<int>(x);
            const float frac = static_cast<float>(x - static_cast<double>(idx));
            const auto u = static_cast<std::size_t>(idx);
            const float va = a[u] + frac * (a[u + 1] - a[u]);
            const float vb = b[u] + frac * (b[u + 1] - b[u]);
            out[i] += (levelStart_ + levelDelta_ * t) * (va + t * (vb - va));

            const float f = (f0Start_ + f0Delta_ * t) * (ratioStart_ + ratioDelta_ * t);
            phase2 = wrap01(phase2 + static_cast<double>(f) * invSr);
        }
        phase2_ = phase2;
    }
    else
    {
        advanceDetuned(n);
    }
    pos_ = pos;
}

void OscPrimitive::advanceDetuned(int n) noexcept
{
    // Keep the detuned phase running while silent so it comes back where a continuous
    // render would have it (sum of per-sample steps, ratio and f0 both linear in t).
    double cycles = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const float t = static_cast<float>(pos_ + i) * invHop_;
        cycles += static_cast<double>((f0Start_ + f0Delta_ * t) * (ratioStart_ + ratioDelta_ * t));
    }
    phase2_ = wrap01(phase2_ + cycles * invSampleRate_);
}

void OscPrimitive::skipHop(const HopContext& ctx) noexcept
{
    rotate(ctx);
    // Nothing was built: the next beginHop rebuilds frame j's table as its previous table.
    curValid_ = false;
    // Sum over n = 0..H-1 of f0(n) / sr with f0(n) = f0Start + (f0End - f0Start) n / H.
    const double h = static_cast<double>(hop_);
    const double cycles = (h * static_cast<double>(ctx.f0Start) +
                           static_cast<double>(ctx.f0End - ctx.f0Start) * (h - 1.0) * 0.5) *
                          invSampleRate_;
    phase_ = wrap01(phase_ + cycles);
    pos_ = 0;
    advanceDetuned(hop_);
    pos_ = 0;
}

} // namespace alfa::dsp
