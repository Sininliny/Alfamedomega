#include "alfa/dsp/Filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>

namespace alfa::dsp
{

namespace
{

constexpr float kMiddleC = 261.6256f;
constexpr float kSqrt2 = 1.41421356f;
// 4th-order Butterworth as two sections: k = 2 cos(pi/8), 2 cos(3 pi/8).
constexpr float kButter4K1 = 1.84775907f;
constexpr float kButter4K2 = 0.76536686f;
// Resonance 1 stops at Q 100: a ring of seconds, but always decaying.
constexpr float kMinDamping = 0.01f;
// Nonlinear damping k_eff = k + kNlDamping band^2 (Van der Pol style): negligible at normal
// levels, but a sustained input at the resonant peak settles at a few units instead of 100x.
constexpr float kNlDamping = 0.05f;
constexpr double kMaxTailSeconds = 30.0;

inline std::size_t u(int i) noexcept
{
    return static_cast<std::size_t>(i);
}

ParamTable makeTable(std::initializer_list<ParamInfo> entries) noexcept
{
    ParamTable t{};
    std::size_t i = 0;
    for (const ParamInfo& e : entries)
        t[i++] = e;
    return t;
}

void mapAll(const ParamTable& table, const float* normalised, std::array<float, kSlotParams>& out) noexcept
{
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = table[i].name.empty() ? 0.0f : mapParam(table[i], normalised[i]);
}

/// (f0 / middle C)^amount, or 1 when there is no usable pitch.
float keyTrackFactor(float f0, float amount) noexcept
{
    if (!(f0 > 0.0f) || !std::isfinite(f0) || !(amount > 0.0f))
        return 1.0f;
    return std::pow(f0 / kMiddleC, amount);
}

/// Pade tanh, exact at 0 and saturating to exactly +-1 at |x| = 3 (continuous, cheap).
inline float softClip(float x) noexcept
{
    const float c = std::clamp(x, -3.0f, 3.0f);
    const float c2 = c * c;
    return c * (27.0f + c2) / (27.0f + 9.0f * c2);
}

int tailFromSeconds(double seconds, double sampleRate) noexcept
{
    const double s = std::clamp(std::isfinite(seconds) ? seconds : kMaxTailSeconds, 0.0, kMaxTailSeconds);
    return static_cast<int>(s * sampleRate) + kControlBlock;
}

/// Zeroes a channel's block after a non-finite value was detected (the caller resets state).
void zeroBlock(float* x, int n) noexcept
{
    std::fill_n(x, n, 0.0f);
}

} // namespace

// FILTER ------------------------------------------------------------------------------------

const ParamTable& FilterProcessor::params() noexcept
{
    static const ParamTable table = makeTable({
        {"cutoff", 20.0f, 20000.0f, Curve::Log, 1.0f, "Hz"},
        {"resonance", 0.0f, 1.0f, Curve::Linear, 0.0f, ""},
        {"mode", 0.0f, 1.0f, Curve::Linear, 0.0f, "LP>BP>HP>notch>LP"},
        {"slope", 0.0f, 1.0f, Curve::Choice, 0.0f, "12/24 dB"},
        {"drive", 0.0f, 36.0f, Curve::Linear, 0.0f, "dB"},
        {"keytrack", 0.0f, 1.0f, Curve::Linear, 0.0f, ""},
        {"type", 0.0f, 3.0f, Curve::Choice, 0.0f, "multimode/bell/lowshelf/highshelf"},
        {"gain", -24.0f, 24.0f, Curve::Linear, 0.5f, "dB"},
        {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
    });
    return table;
}

void FilterProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    reset();
}

void FilterProcessor::reset() noexcept
{
    for (auto& ch : svf_)
        for (auto& s : ch)
            s.reset();
    for (auto& ch : lastBand_)
        ch.fill(0.0f);
    primed_ = false;
    // Until the first setParams: a transparent 12 dB low-pass at the top of the range.
    std::array<float, kSlotParams> neutral{};
    for (std::size_t i = 0; i < neutral.size(); ++i)
        neutral[i] = params()[i].defaultNorm;
    setParams(neutral.data(), {});
    primed_ = false;
}

void FilterProcessor::setParams(const float* normalised, const ControlContext& ctx) noexcept
{
    std::array<float, kSlotParams> p{};
    mapAll(params(), normalised, p);
    const float res = p[1];
    const float mode = p[2];
    const bool slope24 = p[3] >= 0.5f;
    const float driveDb = p[4];
    const int type = static_cast<int>(p[6]);
    const float gainDb = p[7];

    const double cutoff = std::clamp(static_cast<double>(p[0] * keyTrackFactor(ctx.f0, p[5])), 10.0, 20000.0);
    const float g = detail::svfGain(cutoff, sampleRate_);

    auto& t = ramps_.target;
    auto setStage = [&t](int base, float gs, float k, float c0, float c1, float c2, float ch, float nl)
    {
        t[u(base + kG)] = gs;
        t[u(base + kK)] = k;
        t[u(base + kC0)] = c0;
        t[u(base + kC1)] = c1;
        t[u(base + kC2)] = c2;
        t[u(base + kCh)] = ch;
        t[u(base + kNl)] = nl;
    };
    // The second stage passes its input through (it keeps running so slope switches are warm).
    auto passStage2 = [&]
    {
        setStage(kStageRamps, g, kSqrt2, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    };

    double ringSeconds = 0.0;
    if (type == 0)
    {
        // Morph weights: a triangle per response on a circle LP, BP, HP, notch.
        std::array<float, 4> w{};
        const float pos = std::clamp(mode, 0.0f, 1.0f) * 4.0f;
        const int seg = std::min(static_cast<int>(pos), 3);
        const float frac = pos - static_cast<float>(seg);
        w[u(seg)] = 1.0f - frac;
        w[u((seg + 1) % 4)] += frac;
        const float wl = w[0], wb = w[1], wh = w[2], wn = w[3];
        const float ch = wh + wn; // high = v0 - k band - low, notch = v0 - k band
        // Band-pass is scaled by the Butterworth k so every response is 0 dB at resonance 0.
        auto multimode = [&](int base, float k, float kButter)
        {
            setStage(base, g, k, ch, wb * kButter, wl - wh, ch, kNlDamping);
            ringSeconds = std::max(ringSeconds, 2.2 / (static_cast<double>(k) * cutoff));
        };
        if (!slope24)
        {
            multimode(0, kMinDamping + (kSqrt2 - kMinDamping) * (1.0f - res), kSqrt2);
            passStage2();
        }
        else
        {
            multimode(0, kButter4K1, kButter4K1);
            multimode(kStageRamps, kMinDamping + (kButter4K2 - kMinDamping) * (1.0f - res), kButter4K2);
        }
    }
    else
    {
        const float q = 0.70710678f * std::pow(16.0f / 0.70710678f, res);
        detail::SvfCoeffs c{};
        if (type == 1)
            c = detail::svf::bell(g, q, gainDb);
        else if (type == 2)
            c = detail::svf::lowShelf(g, q, gainDb);
        else
            c = detail::svf::highShelf(g, q, gainDb);
        setStage(0, c.g, c.k, c.c0, c.c1, c.c2, 0.0f, 0.0f);
        passStage2();
        ringSeconds = 2.2 / (static_cast<double>(c.k) * cutoff);
    }

    const float driveGain = std::pow(10.0f, driveDb / 20.0f);
    t[kDrive] = driveGain;
    t[kDriveMakeup] = 1.0f / std::sqrt(driveGain);
    t[kDriveBlend] = std::clamp(driveDb / 3.0f, 0.0f, 1.0f);
    t[kWet] = std::clamp(p[8], 0.0f, 1.0f);

    ramps_.retarget(!primed_);
    primed_ = true;
    tail_ = tailFromSeconds(ringSeconds, sampleRate_);
}

namespace
{

/// One multimode stage with the nonlinear damping term; r points at the stage's ramps.
inline float runStage(detail::Svf& s, float v0, const float* r, float& lastBand, int kG, int kK, int kC0,
                      int kC1, int kC2, int kCh, int kNl) noexcept
{
    const float kEff = std::min(r[kK] + r[kNl] * lastBand * lastBand, 1.0e4f);
    float band = 0.0f, low = 0.0f;
    s.tick(v0, r[kG], kEff, band, low);
    lastBand = band;
    return r[kC0] * v0 + (r[kC1] - r[kCh] * kEff) * band + r[kC2] * low;
}

} // namespace

void FilterProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    const int ramping = ramps_.rampingIn(n);
    const bool driveOn = ramps_.value[kDriveBlend] > 0.0f || ramps_.target[kDriveBlend] > 0.0f;

    for (int c = 0; c < channels_; ++c)
    {
        float* x = channels[c];
        auto& s = svf_[u(c)];
        auto& lb = lastBand_[u(c)];
        std::array<float, kRampCount> v = ramps_.value;
        float probe = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            if (i < ramping)
                for (std::size_t j = 0; j < v.size(); ++j)
                    v[j] += ramps_.step[j];
            const float in = x[i];
            float d = in;
            if (driveOn)
                d = in + v[kDriveBlend] * (softClip(v[kDrive] * in) * v[kDriveMakeup] - in);
            float y = runStage(s[0], d, v.data(), lb[0], kG, kK, kC0, kC1, kC2, kCh, kNl);
            y = runStage(s[1], y, v.data() + kStageRamps, lb[1], kG, kK, kC0, kC1, kC2, kCh, kNl);
            const float out = in + v[kWet] * (y - in);
            probe += out - out; // NaN if any output was NaN or Inf
            x[i] = out;
        }
        if (!std::isfinite(probe) || !s[0].isFinite() || !s[1].isFinite())
        {
            s[0].reset();
            s[1].reset();
            lb.fill(0.0f);
            zeroBlock(x, n);
            continue;
        }
        for (std::size_t k = 0; k < 2; ++k)
        {
            s[k].flushDenormals();
            if (std::fabs(lb[k]) < 1e-20f)
                lb[k] = 0.0f;
        }
    }
    ramps_.advance(n);
}

// COMB --------------------------------------------------------------------------------------

namespace
{

constexpr float kCombMinHz = 20.0f;
constexpr float kCombMaxFeedback = 0.99f;

/// 4-point Lagrange read at a fractional delay >= 2 samples behind the write position w.
inline float readLagrange(const float* line, std::size_t mask, std::size_t w, float delay) noexcept
{
    const auto i = static_cast<std::size_t>(delay);
    const float d = delay - static_cast<float>(i) + 1.0f; // position within taps i-1 .. i+2, in [1, 2)
    const float dm1 = d - 1.0f, dm2 = d - 2.0f, dm3 = d - 3.0f;
    const float h0 = -dm1 * dm2 * dm3 * (1.0f / 6.0f);
    const float h1 = d * dm2 * dm3 * 0.5f;
    const float h2 = -d * dm1 * dm3 * 0.5f;
    const float h3 = d * dm1 * dm2 * (1.0f / 6.0f);
    const std::size_t p = w - i + 1; // tap at delay i - 1
    return h0 * line[p & mask] + h1 * line[(p - 1) & mask] + h2 * line[(p - 2) & mask] +
           h3 * line[(p - 3) & mask];
}

} // namespace

const ParamTable& CombProcessor::params() noexcept
{
    static const ParamTable table = makeTable({
        {"frequency", kCombMinHz, 5000.0f, Curve::Log, 0.46567f, "Hz"}, // 261.63 Hz
        {"keytrack", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
        {"feedback", -kCombMaxFeedback, kCombMaxFeedback, Curve::Linear, 0.5f, ""},
        {"damping", 0.0f, 1.0f, Curve::Linear, 0.0f, ""},
        {"balance", 0.0f, 1.0f, Curve::Linear, 1.0f, "FF>FB"},
        {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
    });
    return table;
}

void CombProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    // At least the 20 Hz period at the prepared rate (and at 192 kHz, so a slot prepared once
    // can be reused at any supported rate), plus the interpolator's taps.
    const auto needed = static_cast<std::size_t>(std::max(sampleRate_, 192000.0) / kCombMinHz) + 8;
    std::size_t size = 1;
    while (size < needed)
        size <<= 1;
    mask_ = size - 1;
    for (auto& ch : ch_)
    {
        ch.ffLine.assign(size, 0.0f);
        ch.fbLine.assign(size, 0.0f);
    }
    reset();
}

void CombProcessor::reset() noexcept
{
    for (auto& ch : ch_)
    {
        std::fill(ch.ffLine.begin(), ch.ffLine.end(), 0.0f);
        std::fill(ch.fbLine.begin(), ch.fbLine.end(), 0.0f);
        ch.lp = 0.0f;
    }
    write_ = 0;
    primed_ = false;
    std::array<float, kSlotParams> neutral{};
    for (std::size_t i = 0; i < neutral.size(); ++i)
        neutral[i] = params()[i].defaultNorm;
    setParams(neutral.data(), {});
    primed_ = false;
}

void CombProcessor::setParams(const float* normalised, const ControlContext& ctx) noexcept
{
    std::array<float, kSlotParams> p{};
    mapAll(params(), normalised, p);
    const double maxHz = 0.45 * sampleRate_;
    const double hz = std::clamp(static_cast<double>(p[0] * keyTrackFactor(ctx.f0, p[1])),
                                 static_cast<double>(kCombMinHz), maxHz);
    const float fb = std::clamp(p[2], -kCombMaxFeedback, kCombMaxFeedback);
    const float damping = std::clamp(p[3], 0.0f, 1.0f);
    const float balance = std::clamp(p[4], 0.0f, 1.0f);

    const double delay = std::min(sampleRate_ / hz, static_cast<double>(mask_) - 4.0);
    // The loop low-pass y += a (x - y) delays the comb frequency by its phase delay; take that
    // off the delay line so the loop round trip is exactly one period.
    double alpha = 1.0;
    double lpDelay = 0.0;
    if (damping > 1e-4f)
    {
        const double dampHz = std::min(20000.0 * std::pow(0.01, static_cast<double>(damping)), maxHz);
        alpha = 1.0 - std::exp(-2.0 * 3.14159265358979323846 * dampHz / sampleRate_);
        const double w = 2.0 * 3.14159265358979323846 * hz / sampleRate_;
        const double b = 1.0 - alpha;
        lpDelay = std::atan2(b * std::sin(w), 1.0 - b * std::cos(w)) / w;
    }

    auto& t = ramps_.target;
    t[kDelay] = static_cast<float>(std::max(delay, 2.0));
    t[kFbDelay] = static_cast<float>(std::max(delay - lpDelay, 2.0));
    t[kFb] = fb;
    t[kFf] = fb;
    t[kDamp] = static_cast<float>(alpha);
    // Normalised to white-noise power: FF has power gain 1 + b^2, FB 1 / (1 - a^2).
    t[kWff] = (1.0f - balance) / std::sqrt(1.0f + fb * fb);
    t[kWfb] = balance * std::sqrt(1.0f - fb * fb);
    t[kWet] = std::clamp(p[5], 0.0f, 1.0f);
    ramps_.retarget(!primed_);
    primed_ = true;

    double ring = delay;
    const double a = std::fabs(static_cast<double>(fb));
    if (balance > 0.0f && a > 1e-3)
        ring += delay * std::log(1e-3) / std::log(a);
    tail_ = tailFromSeconds(ring / sampleRate_, sampleRate_);
}

void CombProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0 || mask_ == 0)
        return;
    const int ramping = ramps_.rampingIn(n);
    for (int c = 0; c < channels_; ++c)
    {
        float* x = channels[c];
        Channel& ch = ch_[u(c)];
        float* ff = ch.ffLine.data();
        float* fbl = ch.fbLine.data();
        float lp = ch.lp;
        std::size_t w = write_;
        std::array<float, kRampCount> v = ramps_.value;
        float probe = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            if (i < ramping)
                for (std::size_t j = 0; j < v.size(); ++j)
                    v[j] += ramps_.step[j];
            const float in = x[i];
            const float xd = readLagrange(ff, mask_, w, v[kDelay]);
            const float wd = readLagrange(fbl, mask_, w, v[kFbDelay]);
            lp += v[kDamp] * (wd - lp);
            float loop = in + v[kFb] * lp;
            loop = std::fabs(loop) < 1e-20f ? 0.0f : loop; // the decaying loop never goes denormal
            const float ffOut = in + v[kFf] * xd;
            ff[w] = in;
            fbl[w] = loop;
            w = (w + 1) & mask_;
            const float wet = v[kWff] * ffOut + v[kWfb] * loop;
            const float out = in + v[kWet] * (wet - in);
            probe += out - out;
            x[i] = out;
        }
        if (!std::isfinite(probe) || !std::isfinite(lp))
        {
            std::fill(ch.ffLine.begin(), ch.ffLine.end(), 0.0f);
            std::fill(ch.fbLine.begin(), ch.fbLine.end(), 0.0f);
            lp = 0.0f;
            zeroBlock(x, n);
        }
        ch.lp = std::fabs(lp) < 1e-20f ? 0.0f : lp;
    }
    write_ = (write_ + static_cast<std::size_t>(n)) & mask_;
    ramps_.advance(n);
}

// FORMANT -----------------------------------------------------------------------------------

namespace
{

// Classic formant tables (Csound manual appendix, after Peterson & Barney style measurements):
// [voice: bass, soprano][vowel: a e i o u][formant]. Frequencies and bandwidths in Hz, levels
// in dB relative to the first formant.
constexpr float kFormantHz[2][5][5] = {
    {{600, 1040, 2250, 2450, 2750},
     {400, 1620, 2400, 2800, 3100},
     {250, 1750, 2600, 3050, 3340},
     {400, 750, 2400, 2600, 2900},
     {350, 600, 2400, 2675, 2950}},
    {{800, 1150, 2900, 3900, 4950},
     {350, 2000, 2800, 3600, 4950},
     {270, 2140, 2950, 3900, 4950},
     {450, 800, 2830, 3800, 4950},
     {325, 700, 2700, 3800, 4950}},
};
constexpr float kFormantBw[2][5][5] = {
    {{60, 70, 110, 120, 130},
     {40, 80, 100, 120, 120},
     {60, 90, 100, 120, 120},
     {40, 80, 100, 120, 120},
     {40, 80, 100, 120, 120}},
    {{80, 90, 120, 130, 140},
     {60, 100, 120, 150, 200},
     {60, 90, 100, 120, 120},
     {70, 80, 100, 130, 135},
     {50, 60, 170, 180, 200}},
};
constexpr float kFormantDb[2][5][5] = {
    {{0, -7, -9, -9, -20},
     {0, -12, -9, -12, -18},
     {0, -30, -16, -22, -28},
     {0, -11, -21, -20, -40},
     {0, -20, -32, -28, -36}},
    {{0, -6, -32, -20, -50},
     {0, -20, -15, -40, -56},
     {0, -12, -26, -26, -44},
     {0, -11, -22, -22, -50},
     {0, -16, -35, -40, -60}},
};

inline float lerp(float a, float b, float t) noexcept
{
    return a + (b - a) * t;
}

} // namespace

const ParamTable& FormantProcessor::params() noexcept
{
    static const ParamTable table = makeTable({
        {"vowel", 0.0f, 1.0f, Curve::Linear, 0.0f, "A>E>I>O>U"},
        {"voice", 0.0f, 1.0f, Curve::Linear, 0.0f, "male>female"},
        {"shift", -12.0f, 12.0f, Curve::Linear, 0.5f, "st"},
        {"q", 0.25f, 4.0f, Curve::Log, 0.5f, "x"},
        {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
    });
    return table;
}

void FormantProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    reset();
}

void FormantProcessor::reset() noexcept
{
    for (auto& ch : svf_)
        for (auto& s : ch)
            s.reset();
    primed_ = false;
    std::array<float, kSlotParams> neutral{};
    for (std::size_t i = 0; i < neutral.size(); ++i)
        neutral[i] = params()[i].defaultNorm;
    setParams(neutral.data(), {});
    primed_ = false;
}

float FormantProcessor::formantHz(int i) const noexcept
{
    return i >= 0 && i < kFormants ? hz_[u(i)] : 0.0f;
}

void FormantProcessor::setParams(const float* normalised, const ControlContext&) noexcept
{
    std::array<float, kSlotParams> p{};
    mapAll(params(), normalised, p);
    const float pos = std::clamp(p[0], 0.0f, 1.0f) * 4.0f;
    const int seg = std::min(static_cast<int>(pos), 3);
    const float t = pos - static_cast<float>(seg);
    const float voice = std::clamp(p[1], 0.0f, 1.0f);
    const float shift = std::exp2(p[2] / 12.0f);
    const float q = p[3];
    const auto fs = static_cast<float>(sampleRate_);
    const auto a = u(seg), b = u(seg + 1);

    double ring = 0.0;
    auto& tg = ramps_.target;
    for (std::size_t i = 0; i < u(kFormants); ++i)
    {
        auto logHz = [&](std::size_t vc)
        {
            return lerp(std::log(kFormantHz[vc][a][i]), std::log(kFormantHz[vc][b][i]), t);
        };
        auto bw = [&](std::size_t vc)
        {
            return lerp(kFormantBw[vc][a][i], kFormantBw[vc][b][i], t);
        };
        auto db = [&](std::size_t vc)
        {
            return lerp(kFormantDb[vc][a][i], kFormantDb[vc][b][i], t);
        };
        const float hz = std::exp(lerp(logHz(0), logHz(1), voice)) * shift;
        const float bandwidth = lerp(bw(0), bw(1), voice) * shift / q;
        const float k = std::clamp(bandwidth / hz, 0.01f, 2.0f);
        // Fade formants out between 0.4 and 0.45 fs instead of piling them up at the clamp.
        const float fade = std::clamp((0.45f * fs - hz) / (0.05f * fs), 0.0f, 1.0f);
        const float amp = std::pow(10.0f, lerp(db(0), db(1), voice) / 20.0f) * fade;
        hz_[i] = hz;
        tg[3 * i] = detail::svfGain(static_cast<double>(hz), sampleRate_);
        tg[3 * i + 1] = k;
        tg[3 * i + 2] = amp * k; // unit-peak band-pass: band * k
        if (amp > 0.0f)
            ring = std::max(ring, 2.2 / static_cast<double>(k * std::min(hz, 0.45f * fs)));
    }
    tg[kWet] = std::clamp(p[4], 0.0f, 1.0f);
    ramps_.retarget(!primed_);
    primed_ = true;
    tail_ = tailFromSeconds(ring, sampleRate_);
}

void FormantProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    const int ramping = ramps_.rampingIn(n);
    for (int c = 0; c < channels_; ++c)
    {
        float* x = channels[c];
        auto& s = svf_[u(c)];
        std::array<float, kRampCount> v = ramps_.value;
        float probe = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            if (i < ramping)
                for (std::size_t j = 0; j < v.size(); ++j)
                    v[j] += ramps_.step[j];
            const float in = x[i];
            float sum = 0.0f;
            for (std::size_t f = 0; f < u(kFormants); ++f)
            {
                float band = 0.0f, low = 0.0f;
                s[f].tick(in, v[3 * f], v[3 * f + 1], band, low);
                sum += v[3 * f + 2] * band;
            }
            const float out = in + v[kWet] * (sum - in);
            probe += out - out;
            x[i] = out;
        }
        bool finite = std::isfinite(probe);
        for (auto& f : s)
            finite = finite && f.isFinite();
        for (auto& f : s)
            finite ? f.flushDenormals() : f.reset();
        if (!finite)
            zeroBlock(x, n);
    }
    ramps_.advance(n);
}

// BODY --------------------------------------------------------------------------------------

namespace
{

struct BodyType
{
    float hz[BodyProcessor::kModes];
    float q[BodyProcessor::kModes];
    float db[BodyProcessor::kModes];
    float shelfHz;
};

// Nominal mode sets. Guitar: Helmholtz air mode ~100 Hz, top-plate modes 200-900 Hz, then
// weaker plate/brace modes. Violin: A0 air ~275 Hz, CBR ~400, the B1-/B1+ corpus pair ~460/540,
// and the 2-3 kHz bridge hill. Drum shell: dense, high-Q shell modes of a mid tom. Box: a
// wooden cabinet, broader and lower Q. With the neutral material's -4.5 dB high shelf the
// boosts and the shelf roughly cancel: pink-noise gain is within 1 dB of unity for every type.
constexpr BodyType kBodies[4] = {
    {{98, 204, 390, 560, 820, 1250, 2300, 3600},
     {10, 14, 10, 9, 8, 6, 4, 3},
     {9, 10, 5, 4, 5, 3, 3, 2},
     2500},
    {{275, 405, 460, 540, 720, 1100, 2600, 4200},
     {14, 10, 16, 14, 9, 7, 3, 4},
     {8, 4, 9, 9, 4, 3, 6, -3},
     3000},
    {{120, 185, 260, 340, 470, 690, 980, 1500},
     {18, 16, 15, 14, 12, 10, 8, 6},
     {8, 7, 6, 5, 5, 4, 3, 2},
     1800},
    {{140, 230, 310, 420, 610, 890, 1350, 2100}, {8, 9, 9, 8, 7, 6, 5, 4}, {7, 6, 5, 5, 4, 4, 3, 2}, 2000},
};

constexpr double kBodyGlideSeconds = 0.015;

} // namespace

const ParamTable& BodyProcessor::params() noexcept
{
    static const ParamTable table = makeTable({
        {"type", 0.0f, 3.0f, Curve::Choice, 0.0f, "guitar/violin/drum/box"},
        {"size", 0.5f, 2.0f, Curve::Log, 0.5f, "x"},
        {"material", 0.0f, 1.0f, Curve::Linear, 0.5f, "soft>hard"},
        {"damping", 0.0f, 1.0f, Curve::Linear, 0.5f, ""},
        {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
    });
    return table;
}

void BodyProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    reset();
}

void BodyProcessor::reset() noexcept
{
    for (auto& ch : biquad_)
        for (auto& b : ch)
            b.reset();
    primed_ = false;
    std::array<float, kSlotParams> neutral{};
    for (std::size_t i = 0; i < neutral.size(); ++i)
        neutral[i] = params()[i].defaultNorm;
    setParams(neutral.data(), {});
    primed_ = false;
}

void BodyProcessor::setParams(const float* normalised, const ControlContext&) noexcept
{
    std::array<float, kSlotParams> p{};
    mapAll(params(), normalised, p);
    const BodyType& body = kBodies[std::clamp(static_cast<int>(p[0]), 0, 3)];
    const double size = p[1];
    const double material = std::clamp(p[2], 0.0f, 1.0f);
    const double damping = std::clamp(p[3], 0.0f, 1.0f);
    const double maxHz = 0.45 * sampleRate_;

    const double tilt = (material - 0.5) * 6.0; // dB per octave around 500 Hz
    const double qScale = (0.7 + 0.6 * material) * std::exp2(1.0 - 2.0 * damping);
    const double gainScale = 1.25 - 0.5 * damping;
    for (std::size_t m = 0; m < u(kModes); ++m)
    {
        const double nominal = body.hz[m];
        const double hz = std::clamp(nominal / size, 20.0, maxHz);
        const double db = (body.db[m] + tilt * std::log2(nominal / 500.0)) * gainScale;
        target_[m] = {std::log(hz), std::clamp(db, -18.0, 18.0),
                      std::log(std::clamp(body.q[m] * qScale, 0.3, 40.0))};
    }
    target_[u(kModes)] = {std::log(std::clamp(body.shelfHz / size, 20.0, maxHz)), -12.0 + 15.0 * material,
                          std::log(0.70710678)};

    // Glide the mode table (a one-pole per control block), then redesign what moved.
    const double beta = primed_ ? 1.0 - std::exp(-kControlBlock / (kBodyGlideSeconds * sampleRate_)) : 1.0;
    double ring = 0.0;
    for (std::size_t s = 0; s < u(kSections); ++s)
    {
        Section& cur = current_[s];
        const Section& tgt = target_[s];
        const bool moved = !primed_ || std::fabs(cur.logHz - tgt.logHz) > 1e-6 ||
                           std::fabs(cur.gainDb - tgt.gainDb) > 1e-4 || std::fabs(cur.logQ - tgt.logQ) > 1e-6;
        const double hz = std::exp(cur.logHz);
        const double q = std::exp(cur.logQ);
        if (s < u(kModes))
            ring = std::max(ring, 2.2 * q * std::pow(10.0, std::fabs(cur.gainDb) / 40.0) / hz);
        if (!moved)
            continue;
        cur.logHz += beta * (tgt.logHz - cur.logHz);
        cur.gainDb += beta * (tgt.gainDb - cur.gainDb);
        cur.logQ += beta * (tgt.logQ - cur.logQ);
        const double newHz = std::exp(cur.logHz);
        const double newQ = std::exp(cur.logQ);
        coeffs_[s] = s < u(kModes) ? detail::biquad::peaking(newHz, newQ, cur.gainDb, sampleRate_)
                                   : detail::biquad::highShelf(newHz, newQ, cur.gainDb, sampleRate_);
    }

    ramps_.target[0] = std::clamp(p[4], 0.0f, 1.0f);
    ramps_.retarget(!primed_);
    primed_ = true;
    tail_ = tailFromSeconds(ring, sampleRate_);
}

void BodyProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    const int ramping = ramps_.rampingIn(n);
    for (int c = 0; c < channels_; ++c)
    {
        float* x = channels[c];
        auto& bq = biquad_[u(c)];
        float wet = ramps_.value[0];
        float probe = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            if (i < ramping)
                wet += ramps_.step[0];
            const float in = x[i];
            float y = in;
            for (std::size_t s = 0; s < u(kSections); ++s)
                y = bq[s].process(y, coeffs_[s]);
            const float out = in + wet * (y - in);
            probe += out - out;
            x[i] = out;
        }
        bool finite = std::isfinite(probe);
        for (auto& b : bq)
            finite = finite && b.isFinite();
        for (auto& b : bq)
            finite ? b.flushDenormals() : b.reset();
        if (!finite)
            zeroBlock(x, n);
    }
    ramps_.advance(n);
}

} // namespace alfa::dsp
