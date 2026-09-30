#include "alfa/dsp/Physical.h"

#include "alfa/dsp/detail/Fft.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace alfa::dsp
{

namespace
{

constexpr double kLn1000 = 6.907755278982137; // T60: amplitude falls to 10^-3
constexpr float kFlush = 1.0e-20f;
constexpr double kMaxLoopGain = 0.99999;

[[nodiscard]] inline float flushTiny(float x) noexcept
{
    return std::fabs(x) < kFlush ? 0.0f : x;
}

[[nodiscard]] inline double flushTinyD(double x) noexcept
{
    return std::fabs(x) < static_cast<double>(kFlush) ? 0.0 : x;
}

/// Phase delay in samples at w (rad/sample) of the first-order allpass (a + z^-1) / (1 + a z^-1).
/// Its phase is -w + 2 arg(1 + a e^{iw}); 1 + a cos w > 0 for |a| < 1, so no unwrapping.
[[nodiscard]] double allpassDelay(double a, double w) noexcept
{
    return 1.0 - 2.0 * std::atan2(a * std::sin(w), 1.0 + a * std::cos(w)) / w;
}

[[nodiscard]] float semitoneRatio(float semitones) noexcept
{
    return std::exp2(semitones / 12.0f);
}

[[nodiscard]] float baseFrequency(const ControlContext& ctx, float absoluteHz) noexcept
{
    return std::isfinite(ctx.f0) && ctx.f0 > 0.0f ? ctx.f0 : absoluteHz;
}

[[nodiscard]] float readFractional(const std::vector<float>& buf, unsigned mask, unsigned pos,
                                   float delay) noexcept
{
    const auto i = static_cast<unsigned>(delay);
    const float f = delay - static_cast<float>(i);
    const float a = buf[(pos - i) & mask];
    const float b = buf[(pos - i - 1u) & mask];
    return a + f * (b - a);
}

} // namespace

// Mode tables ---------------------------------------------------------------------------------

namespace detail
{

namespace
{

// The 32 smallest zeros j_mn of J_m (any m >= 0), ascending: the modes of an ideal circular
// membrane with a fixed rim. Tabulated (computed offline from Bessel's integral and bisection,
// checked against Abramowitz & Stegun table 9.5) because std::cyl_bessel_j is missing on libc++.
constexpr std::array<double, kPhysicalMaxModes> kBesselZeros = {
    2.4048255577,  3.8317059702,  5.1356223018,  5.5200781103,  6.3801618959,  7.0155866698,  7.5883424345,
    8.4172441404,  8.6537279129,  8.7714838160,  9.7610231300,  9.9361095242,  10.1734681351, 11.0647094885,
    11.0863700192, 11.6198411721, 11.7915344390, 12.2250922640, 12.3386041975, 13.0152007217, 13.3236919363,
    13.3543004774, 13.5892901705, 14.3725366716, 14.4755006866, 14.7959517824, 14.8212687270, 14.9309177085,
    15.5898478845, 15.7001740797, 16.0377741909, 16.2234661603,
};

PhysicalModeTables buildTables()
{
    PhysicalModeTables t;
    auto set = [&t](PhysicalModeSet s, int k, double ratio)
    {
        const auto si = static_cast<std::size_t>(s);
        const auto ki = static_cast<std::size_t>(k);
        t.ratio[si][ki] = static_cast<float>(ratio);
        t.log2Ratio[si][ki] = static_cast<float>(std::log2(ratio));
    };

    // Free-free beam: roots of cos(b) cosh(b) = 1; beyond the fifth, (2k + 3) pi / 2 is exact to
    // float precision.
    constexpr std::array<double, 5> beamRoots = {4.730040745, 7.853204624, 10.99560784, 14.13716549,
                                                 17.27875966};

    // Simply supported square plate: f ~ m^2 + n^2 for m, n >= 1. Distinct sums, ascending.
    std::array<int, kPhysicalMaxModes> plate{};
    {
        int found = 0;
        for (int sum = 2; found < kPhysicalMaxModes; ++sum)
        {
            bool hit = false;
            for (int m = 1; m * m < sum && !hit; ++m)
            {
                const int rest = sum - m * m;
                const auto n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(rest))));
                hit = n >= 1 && n * n == rest;
            }
            if (hit)
                plate[static_cast<std::size_t>(found++)] = sum;
        }
    }

    for (int k = 0; k < kPhysicalMaxModes; ++k)
    {
        const auto ki = static_cast<std::size_t>(k);
        const double n = k + 1;
        set(PhysicalModeSet::IdealString, k, n);
        set(PhysicalModeSet::StiffString, k,
            n * std::sqrt((1.0 + kStiffStringB * n * n) / (1.0 + kStiffStringB)));
        const double beta = k < 5 ? beamRoots[ki] : (2.0 * k + 3.0) * std::numbers::pi / 2.0;
        set(PhysicalModeSet::FreeBar, k, (beta / beamRoots[0]) * (beta / beamRoots[0]));
        set(PhysicalModeSet::Membrane, k, kBesselZeros[ki] / kBesselZeros[0]);
        set(PhysicalModeSet::SquarePlate, k, plate[ki] / 2.0);
        set(PhysicalModeSet::Tube, k, 2.0 * n - 1.0);
    }
    return t;
}

} // namespace

const PhysicalModeTables& physicalModeTables()
{
    static const PhysicalModeTables tables = buildTables();
    return tables;
}

// ModalBank -------------------------------------------------------------------------------------

void ModalBank::reset() noexcept
{
    for (int c = 0; c < kMaxChannels; ++c)
        resetChannel(c);
}

void ModalBank::resetChannel(int channel) noexcept
{
    const auto c = static_cast<std::size_t>(channel);
    re_[c].fill(0.0f);
    im_[c].fill(0.0f);
}

void ModalBank::setModes(const float* freq, const float* decay, const float* gain, int count,
                         float maxFreq) noexcept
{
    count = std::clamp(count, 0, kPhysicalMaxModes);
    int highest = -1;
    int active = 0;
    for (int k = 0; k < kPhysicalMaxModes; ++k)
    {
        const auto i = static_cast<std::size_t>(k);
        const float w = k < count ? freq[k] : 0.0f;
        const float g = k < count ? gain[k] : 0.0f;
        const bool on = w > 0.0f && w < maxFreq && g != 0.0f && std::isfinite(g);
        if (!on)
        {
            c_[i] = 0.0f;
            s_[i] = 0.0f;
            g_[i] = 0.0f;
            continue;
        }
        // r = e^-a by series: exact where it matters (a ~ 1e-6 at 30 s decays, where 1 - e^-a
        // must keep its relative precision) and good to 3e-5 at the clamp a = 0.5.
        const float a = std::clamp(std::isfinite(decay[k]) ? decay[k] : 0.5f, 1.0e-6f, 0.5f);
        const float r = 1.0f - a * (1.0f - a * (0.5f - a * (1.0f / 6.0f - a * (1.0f / 24.0f - a / 120.0f))));
        // cos from the half-angle sine keeps its relative precision near theta = 0 (low modes at
        // high sample rates), which keeps the pole radius, and so the decay, exact.
        const float u = sin2pi(0.5f * w);
        c_[i] = r * (1.0f - 2.0f * u * u);
        s_[i] = r * sin2pi(w);
        g_[i] = g;
        highest = k;
        ++active;
    }
    active_ = active;
    const int padded = std::min(kPhysicalMaxModes, (highest + 8) & ~7);
    // Modes past the processed range keep no stale state for when they come back.
    for (int k = padded; k < padded_; ++k)
    {
        for (auto& ch : re_)
            ch[static_cast<std::size_t>(k)] = 0.0f;
        for (auto& ch : im_)
            ch[static_cast<std::size_t>(k)] = 0.0f;
    }
    padded_ = padded;
}

bool ModalBank::process(int channel, const float* in, float* out, int n) noexcept
{
    const auto ch = static_cast<std::size_t>(channel);
    float* re = re_[ch].data();
    float* im = im_[ch].data();
    const float* c = c_.data();
    const float* s = s_.data();
    const float* g = g_.data();
    const int m = padded_;

    for (int t = 0; t < n; ++t)
    {
        const float x = in[t];
        float y = 0.0f;
#pragma omp simd reduction(+ : y)
        for (int k = 0; k < m; ++k)
        {
            const float r0 = re[k];
            const float i0 = im[k];
            re[k] = r0 * c[k] - i0 * s[k] + g[k] * x;
            const float i1 = r0 * s[k] + i0 * c[k];
            im[k] = i1;
            y += i1;
        }
        out[t] = y;
    }

    // Once per block: flush decayed state, catch non-finite state, and bound each mode (only
    // reached when an input far above unity drives a very long decay at resonance).
    constexpr float kLimit = 64.0f;
    float sum = 0.0f;
    for (int k = 0; k < m; ++k)
    {
        re[k] = std::clamp(flushTiny(re[k]), -kLimit, kLimit);
        im[k] = std::clamp(flushTiny(im[k]), -kLimit, kLimit);
        sum += std::fabs(re[k]) + std::fabs(im[k]);
    }
    if (!std::isfinite(sum) || std::isnan(re[0]) || std::isnan(im[0]))
    {
        resetChannel(channel);
        std::fill_n(out, n, 0.0f);
        return false;
    }
    return true;
}

} // namespace detail

// WaveguideProcessor ----------------------------------------------------------------------------

const ParamTable& WaveguideProcessor::params() noexcept
{
    static constexpr ParamTable table = {{
        {"pitch", -24.0f, 24.0f, Curve::Linear, 0.5f, "st"},
        {"fine", -100.0f, 100.0f, Curve::Linear, 0.5f, "ct"},
        {"frequency", 20.0f, 8000.0f, Curve::Log, 0.40022f, "Hz"},
        {"decay", 0.05f, 30.0f, Curve::Log, 0.57668f, "s"},
        {"brightness", 0.0f, 1.0f, Curve::Linear, 0.5f, ""},
        {"dispersion", 0.0f, 1.0f, Curve::Linear, 0.0f, ""},
        {"pickup", 0.02f, 0.5f, Curve::Linear, 0.375f, ""},
        {"excitation", 0.02f, 0.5f, Curve::Linear, 0.22917f, ""},
        {"tension", 0.0f, 1.0f, Curve::Linear, 0.0f, "st"},
        {"input", 0.0f, 2.0f, Curve::Linear, 0.5f, ""},
        {"pluck", 0.0f, 1.0f, Curve::Toggle, 0.0f, ""},
        {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
        {},
        {},
        {},
        {},
    }};
    return table;
}

void WaveguideProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    // Longest loop is one period at kMinHz, and both combs reach at most half a period back.
    const int length = detail::nextPowerOfTwo(static_cast<int>(std::ceil(sampleRate_ / kMinHz)) + 8);
    lineMask_ = static_cast<unsigned>(length - 1);
    const int excLength = detail::nextPowerOfTwo(static_cast<int>(std::ceil(0.5 * sampleRate_ / kMinHz)) + 8);
    excMask_ = static_cast<unsigned>(excLength - 1);
    std::uint32_t seed = 0x1234567u;
    for (auto& ch : ch_)
    {
        ch.line.assign(static_cast<std::size_t>(length), 0.0f);
        ch.exc.assign(static_cast<std::size_t>(excLength), 0.0f);
        ch.rng.setSeed(seed);
        seed = seed * 747796405u + 2891336453u;
    }
    reset();
}

void WaveguideProcessor::resetChannel(Channel& ch) noexcept
{
    std::fill(ch.line.begin(), ch.line.end(), 0.0f);
    std::fill(ch.exc.begin(), ch.exc.end(), 0.0f);
    ch.fir1 = ch.fir2 = 0.0;
    ch.frac = 0.0;
    ch.disp.fill(0.0);
    ch.burstLp = 0.0f;
}

void WaveguideProcessor::reset() noexcept
{
    for (auto& ch : ch_)
        resetChannel(ch);
    pos_ = 0;
    env_ = 0.0f;
    pluckPending_ = true;
    burstLeft_ = 0;
    first_ = true;
    lastRetuneHz_ = -1.0;
}

void WaveguideProcessor::retune(double hz) noexcept
{
    const double w = 2.0 * std::numbers::pi * hz / sampleRate_;
    const double period = sampleRate_ / hz;

    // Loss FIR a1 z^0 + a0 z^-1 + a1 z^-2: linear phase (exactly one sample), magnitude
    // a0 + 2 a1 cos w, from flat (brightness 1) to (1 + cos w) / 2 (brightness 0).
    // The decay wins over brightness: at high pitch and long T60 the FIR's loss at the
    // fundamental alone could exceed the loss allowed per period, so a1 is capped to keep the
    // fundamental's T60 reachable with a loss gain below 1.
    const double perPeriod = std::exp(-kLn1000 / (static_cast<double>(t60_) * hz));
    const double maxLoss = std::max(0.0, 1.0 - perPeriod / kMaxLoopGain);
    const double a1 = std::min(0.25 * (1.0 - static_cast<double>(brightness_)),
                               maxLoss / (2.0 * std::max(1.0e-12, 1.0 - std::cos(w))));
    const double a0 = 1.0 - 2.0 * a1;
    const double firMag = std::max(1.0e-3, a0 + 2.0 * a1 * std::cos(w));

    // Everything but the dispersion needs 1 (FIR) + 1 (shortest integer delay) + 0.5 (shortest
    // fractional delay) samples; dispersion gets what is left, fewer and weaker stages at high
    // pitch.
    const double budget = period - 2.5 - 0.01;
    int stages = dispersion_ > 1.0e-4f ? kMaxStages : 0;
    while (stages > 0 && static_cast<double>(stages) > budget)
        --stages;
    double a = -0.93 * static_cast<double>(dispersion_);
    if (stages > 0 && stages * allpassDelay(a, w) > budget)
    {
        double lo = a, hi = 0.0; // delay grows as a -> -1
        for (int i = 0; i < 40; ++i)
        {
            const double mid = 0.5 * (lo + hi);
            (stages * allpassDelay(mid, w) > budget ? lo : hi) = mid;
        }
        a = hi;
    }
    const double dispDelay = stages > 0 ? stages * allpassDelay(a, w) : 0.0;

    // The fractional allpass takes 0.5 .. 1.5 samples, solved exactly at the fundamental:
    // a = sin(w (1 - d) / 2) / sin(w (1 + d) / 2) has phase delay d at w.
    const double rest = period - 1.0 - dispDelay;
    const double integer = std::max(1.0, std::floor(rest - 0.5));
    const double d = rest - integer;
    const double fracA = std::sin(0.5 * w * (1.0 - d)) / std::sin(0.5 * w * (1.0 + d));

    if (stages < stages_)
        for (auto& ch : ch_)
            for (int s = stages; s < kMaxStages; ++s)
                ch.disp[static_cast<std::size_t>(s)] = 0.0f;
    stages_ = stages;
    dispA_ = static_cast<float>(a);
    firA0_ = static_cast<float>(a0);
    firA1_ = static_cast<float>(a1);
    fracA_ = static_cast<float>(std::clamp(fracA, -0.999, 0.999));
    delayInt_ = static_cast<int>(integer);
    loopGain_ = static_cast<float>(std::clamp(perPeriod / firMag, 0.0, kMaxLoopGain));
    tunedHz_ = hz;
    period_ = period;
}

void WaveguideProcessor::setParams(const float* p, const ControlContext& ctx) noexcept
{
    const auto& t = params();
    auto map = [&](int i)
    {
        return mapParam(t[static_cast<std::size_t>(i)], p[i]);
    };

    const float offset = map(kPitch) + map(kFine) / 100.0f;
    const double maxHz = sampleRate_ / 5.0; // leaves room for the loop's fixed delays
    baseHz_ = std::clamp(static_cast<double>(baseFrequency(ctx, map(kFrequency)) * semitoneRatio(offset)),
                         static_cast<double>(kMinHz), maxHz);
    t60_ = map(kDecay);
    brightness_ = map(kBrightness);
    dispersion_ = map(kDispersion);
    tension_ = map(kTension);
    pluckOn_ = map(kPluck) >= 0.5f;

    const double glide = static_cast<double>(tension_ * std::min(env_, 1.0f)) / 12.0;
    const double hz = std::min(baseHz_ * std::exp2(glide), maxHz);
    if (std::fabs(hz - lastRetuneHz_) > 1.0e-9 * hz || brightness_ != lastRetuneB_ ||
        dispersion_ != lastRetuneD_ || t60_ != lastRetuneT_)
    {
        retune(hz);
        lastRetuneHz_ = hz;
        lastRetuneB_ = brightness_;
        lastRetuneD_ = dispersion_;
        lastRetuneT_ = t60_;
    }

    const auto period = static_cast<float>(period_);
    pickupT_ = map(kPickup) * period;
    excT_ = map(kExcitation) * period;
    inGainT_ = map(kInputGain);
    mixT_ = map(kMix);

    if (pluckPending_)
    {
        pluckPending_ = false;
        if (pluckOn_)
        {
            burstLeft_ = std::max(1, static_cast<int>(std::lround(period_)));
            burstK_ = 0.1f + 0.9f * brightness_;
        }
    }

    if (first_)
    {
        pickupD_ = pickupT_;
        excD_ = excT_;
        inGain_ = inGainT_;
        mix_ = mixT_;
        first_ = false;
    }
}

void WaveguideProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    const float inv = 1.0f / static_cast<float>(n);
    const float dPick = (pickupT_ - pickupD_) * inv;
    const float dExc = (excT_ - excD_) * inv;
    const float dIn = (inGainT_ - inGain_) * inv;
    const float dMix = (mixT_ - mix_) * inv;
    const auto delay = static_cast<unsigned>(delayInt_);
    const double a0 = firA0_, a1 = firA1_, fa = fracA_, da = dispA_, g = loopGain_;
    const int stages = stages_;
    float peak = 0.0f;

    for (int c = 0; c < channels_; ++c)
    {
        Channel& ch = ch_[static_cast<std::size_t>(c)];
        float* x = channels[c];
        float* line = ch.line.data();
        float* exc = ch.exc.data();
        unsigned pos = pos_;
        int burst = burstLeft_;
        bool bad = false;
        float energy = 0.0f;

        for (int t = 0; t < n; ++t)
        {
            const auto ft = static_cast<float>(t);
            const float dry = x[t];
            bad = bad || !std::isfinite(dry);
            float e = (bad ? 0.0f : dry) * (inGain_ + dIn * ft);
            if (burst > 0)
            {
                ch.burstLp += burstK_ * (ch.rng.nextBipolar() - ch.burstLp);
                e += ch.burstLp;
                --burst;
            }
            exc[pos & excMask_] = e;
            const float excited = 0.5f * (e - readFractional(ch.exc, excMask_, pos, excD_ + dExc * ft));

            const double v = line[(pos - delay) & lineMask_];
            const double y = a1 * (v + ch.fir2) + a0 * ch.fir1;
            ch.fir2 = ch.fir1;
            ch.fir1 = v;
            double od = fa * y + ch.frac;
            ch.frac = y - fa * od;
            for (int s = 0; s < stages; ++s)
            {
                double& st = ch.disp[static_cast<std::size_t>(s)];
                const double z = da * od + st;
                st = od - da * z;
                od = z;
            }
            float o = flushTiny(static_cast<float>(od * g));
            const float ao = std::fabs(o);
            if (ao > 2.0f)
                o = std::copysign(4.0f - 4.0f / ao, o); // C1-continuous, asymptote 4

            const float str = o + excited;
            line[pos & lineMask_] = str;
            energy = std::max(energy, std::fabs(str));
            const float wet = 0.5f * (str - readFractional(ch.line, lineMask_, pos, pickupD_ + dPick * ft));
            x[t] = dry + (mix_ + dMix * ft) * (wet - dry);
            ++pos;
        }

        ch.fir1 = flushTinyD(ch.fir1);
        ch.fir2 = flushTinyD(ch.fir2);
        ch.frac = flushTinyD(ch.frac);
        for (auto& s : ch.disp)
            s = flushTinyD(s);
        if (bad || !std::isfinite(energy) || !std::isfinite(ch.frac))
        {
            resetChannel(ch);
            std::fill_n(x, n, 0.0f);
            energy = 0.0f;
        }
        peak = std::max(peak, energy);
    }

    pos_ += static_cast<unsigned>(n);
    burstLeft_ = std::max(0, burstLeft_ - n);
    pickupD_ = pickupT_;
    excD_ = excT_;
    inGain_ = inGainT_;
    mix_ = mixT_;
    // Amplitude envelope for the tension glide: fast attack, 50 ms release. A block (32
    // samples) is far shorter than a low string's period, so the raw block peak swings within
    // every period; following it directly frequency-modulated the loop, and the constant
    // retuning bled energy (a 30 s T60 decayed 15 dB too fast with tension and dispersion).
    if (peak > env_)
        env_ += 0.5f * (peak - env_);
    else
        env_ *= static_cast<float>(std::exp(-static_cast<double>(n) / (0.05 * sampleRate_)));
}

int WaveguideProcessor::tailSamples() const noexcept
{
    return static_cast<int>(std::min(1.0e9, static_cast<double>(t60_) * sampleRate_ + period_));
}

// ResonatorProcessor ----------------------------------------------------------------------------

const ParamTable& ResonatorProcessor::params() noexcept
{
    static constexpr ParamTable table = {{
        {"pitch", -24.0f, 24.0f, Curve::Linear, 0.5f, "st"},
        {"fine", -100.0f, 100.0f, Curve::Linear, 0.5f, "ct"},
        {"frequency", 20.0f, 8000.0f, Curve::Log, 0.40022f, "Hz"},
        {"structure", 0.0f, 5.0f, Curve::Linear, 0.0f, ""},
        {"inharmonicity", -0.25f, 0.25f, Curve::Linear, 0.5f, ""},
        {"decay", 0.01f, 30.0f, Curve::Log, 0.62583f, "s"},
        {"damping", 0.0f, 2.0f, Curve::Linear, 0.25f, ""},
        {"brightness", -18.0f, 6.0f, Curve::Linear, 0.5f, "dB/oct"},
        {"position", 0.01f, 0.5f, Curve::Linear, 0.44898f, ""},
        {"modes", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
        {"spread", 0.0f, 50.0f, Curve::Linear, 0.0f, "ct"},
        {"input", 0.0f, 2.0f, Curve::Linear, 0.5f, ""},
        {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
        {},
        {},
        {},
    }};
    return table;
}

void ResonatorProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    limit_ = std::clamp(spec.limits.resonatorModes, 1, detail::kPhysicalMaxModes);
    tables_ = &detail::physicalModeTables();
    detail::Random rng(0xC0FFEEu);
    for (auto& d : detune_)
        d = rng.nextBipolar();
    scratch_.assign(static_cast<std::size_t>(kControlBlock), 0.0f);
    reset();
}

void ResonatorProcessor::reset() noexcept
{
    bank_.reset();
    first_ = true;
    lastF0_ = -1.0f;
}

void ResonatorProcessor::setParams(const float* p, const ControlContext& ctx) noexcept
{
    const auto& t = params();
    auto map = [&](int i)
    {
        return mapParam(t[static_cast<std::size_t>(i)], p[i]);
    };

    inGainT_ = map(kInputGain);
    mixT_ = map(kMix);
    if (first_)
    {
        inGain_ = inGainT_;
        mix_ = mixT_;
    }

    // The mode set only changes when its inputs do; skip the per-mode maths otherwise.
    bool same = !first_ && ctx.f0 == lastF0_;
    for (int i = 0; i < kSlotParams && same; ++i)
        same = p[i] == lastParams_[static_cast<std::size_t>(i)];
    first_ = false;
    if (same)
        return;
    std::copy_n(p, kSlotParams, lastParams_.begin());
    lastF0_ = ctx.f0;

    const float offset = map(kPitch) + map(kFine) / 100.0f;
    const float hz =
        std::clamp(baseFrequency(ctx, map(kFrequency)) * semitoneRatio(offset), kMinHz, 24000.0f);
    const float structure = std::clamp(map(kStructure), 0.0f, 5.0f);
    const int lo = std::min(static_cast<int>(structure), detail::kPhysicalModeSetCount - 2);
    const float morph = structure - static_cast<float>(lo);
    const float* logA = tables_->log2Ratio[static_cast<std::size_t>(lo)].data();
    const float* logB = tables_->log2Ratio[static_cast<std::size_t>(lo + 1)].data();
    const float stretch = 1.0f + map(kInharmonicity);
    t60_ = map(kDecay);
    const auto baseDecay = static_cast<float>(kLn1000 / (static_cast<double>(t60_) * sampleRate_));
    const float tilt = map(kDampingTilt);
    const float gainExp = map(kBrightness) / 6.0206f; // dB/octave -> exponent of the ratio
    const float position = map(kPosition);
    const float detune = map(kSpread) / 1200.0f;
    const int limit = quality_ > 0 ? std::max(1, limit_ / 2) : limit_;
    const auto count =
        std::clamp(static_cast<int>(std::lround(map(kModeCount) * static_cast<float>(limit))), 1, limit);
    const auto cyclesPerSample = static_cast<float>(static_cast<double>(hz) / sampleRate_);

    float* freq = freq_.data();
    float* decay = decay_.data();
    float* gain = gain_.data();
    const float* dt = detune_.data();
#pragma omp simd
    for (int k = 0; k < detail::kPhysicalMaxModes; ++k)
    {
        const float l = (logA[k] + morph * (logB[k] - logA[k])) * stretch + detune * dt[k];
        freq[k] = cyclesPerSample * detail::physicalExp2(l);
        decay[k] = baseDecay * detail::physicalExp2(tilt * l);
        gain[k] = detail::physicalExp2(gainExp * l) *
                  std::fabs(detail::sin2pi(0.5f * static_cast<float>(k + 1) * position));
    }

    // Unit energy over the audible modes, so level barely depends on mode count or structure.
    constexpr float kMaxFreq = 0.48f;
    float energy = 0.0f;
    for (int k = 0; k < count; ++k)
        if (freq[k] < kMaxFreq)
            energy += gain[k] * gain[k];
    const float norm = energy > 1.0e-12f ? 1.0f / std::sqrt(energy) : 0.0f;
    for (int k = 0; k < count; ++k)
        gain[k] *= norm;

    bank_.setModes(freq, decay, gain, count, kMaxFreq);
}

void ResonatorProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    const float inv = 1.0f / static_cast<float>(n);
    const float dIn = (inGainT_ - inGain_) * inv;
    const float dMix = (mixT_ - mix_) * inv;
    float* s = scratch_.data();

    for (int c = 0; c < channels_; ++c)
    {
        float* x = channels[c];
        float check = 0.0f;
        for (int t = 0; t < n; ++t)
            check += x[t] * 0.0f; // NaN / Inf poison the sum
        if (!std::isfinite(check))
        {
            bank_.resetChannel(c);
            std::fill_n(x, n, 0.0f);
            continue;
        }
        for (int done = 0; done < n; done += kControlBlock)
        {
            const int m = std::min(kControlBlock, n - done);
            for (int t = 0; t < m; ++t)
                s[t] = x[done + t] * (inGain_ + dIn * static_cast<float>(done + t));
            bank_.process(c, s, s, m);
            for (int t = 0; t < m; ++t)
            {
                const float dry = x[done + t];
                x[done + t] = dry + (mix_ + dMix * static_cast<float>(done + t)) * (s[t] - dry);
            }
        }
    }
    inGain_ = inGainT_;
    mix_ = mixT_;
}

int ResonatorProcessor::tailSamples() const noexcept
{
    return static_cast<int>(std::min(1.0e9, 1.1 * static_cast<double>(t60_) * sampleRate_));
}

float ResonatorProcessor::modeHz(int k) const noexcept
{
    if (k < 0 || k >= detail::kPhysicalMaxModes)
        return 0.0f;
    const auto i = static_cast<std::size_t>(k);
    if (bank_.coefficientMagnitude(k) == 0.0f)
        return 0.0f;
    return freq_[i] * static_cast<float>(sampleRate_);
}

} // namespace alfa::dsp
