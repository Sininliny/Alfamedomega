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
    reads_ = std::clamp(spec.limits.oscUnison - 1, 0, kMaxOscDetune);

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
    readOn_.fill(false);
    for (int r = 0; r < kMaxOscDetune; ++r)
        readPhase_[static_cast<std::size_t>(r)] =
            static_cast<double>(r + 1) / static_cast<double>(kMaxOscDetune + 1);
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
    // Unison reads: ratios and levels ramp like every other control. While a read sounds, the
    // Nyquist cut must hold for the highest ratio in use.
    const VoiceFrame* prevFrame = ctx.prev != nullptr ? ctx.prev : ctx.cur;
    float cutScale = 1.0f;
    for (int r = 0; r < reads_; ++r)
    {
        const auto u = static_cast<std::size_t>(r);
        if (ctx.cur == nullptr || prevFrame == nullptr)
        {
            readOn_[u] = false;
            continue;
        }
        ratioStart_[u] = prevFrame->oscDetuneRatio[u];
        ratioDelta_[u] = ctx.cur->oscDetuneRatio[u] - ratioStart_[u];
        levelStart_[u] = prevFrame->oscDetuneLevel[u];
        levelDelta_[u] = ctx.cur->oscDetuneLevel[u] - levelStart_[u];
        readOn_[u] = levelStart_[u] != 0.0f || levelStart_[u] + levelDelta_[u] != 0.0f;
        if (readOn_[u])
        {
            cutScale = std::max(cutScale, std::max(ratioStart_[u], ratioStart_[u] + ratioDelta_[u]));
        }
    }
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

    for (int r = 0; r < reads_; ++r)
    {
        const auto u = static_cast<std::size_t>(r);
        if (!readOn_[u])
        {
            advanceRead(r, n);
            continue;
        }
        double ph = readPhase_[u];
        int p2 = pos_;
        const float ls = levelStart_[u], ld = levelDelta_[u], rs = ratioStart_[u], rd = ratioDelta_[u];
        for (int i = 0; i < n; ++i, ++p2)
        {
            const float t = static_cast<float>(p2) * invHop_;
            const double x = ph * kSize;
            const int idx = static_cast<int>(x);
            const float frac = static_cast<float>(x - static_cast<double>(idx));
            const auto k = static_cast<std::size_t>(idx);
            const float va = a[k] + frac * (a[k + 1] - a[k]);
            const float vb = b[k] + frac * (b[k + 1] - b[k]);
            out[i] += (ls + ld * t) * (va + t * (vb - va));
            const float f = (f0Start_ + f0Delta_ * t) * (rs + rd * t);
            ph = wrap01(ph + static_cast<double>(f) * invSr);
        }
        readPhase_[u] = ph;
    }
    pos_ = pos;
}

void OscPrimitive::advanceRead(int r, int n) noexcept
{
    // Keep a silent read's phase running so it comes back where a continuous render would have
    // it (sum of per-sample steps, ratio and f0 both linear in t).
    const auto u = static_cast<std::size_t>(r);
    double cycles = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const float t = static_cast<float>(pos_ + i) * invHop_;
        cycles += static_cast<double>((f0Start_ + f0Delta_ * t) * (ratioStart_[u] + ratioDelta_[u] * t));
    }
    readPhase_[u] = wrap01(readPhase_[u] + cycles * invSampleRate_);
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
    for (int r = 0; r < reads_; ++r)
        advanceRead(r, hop_);
    pos_ = 0;
}

} // namespace alfa::dsp
