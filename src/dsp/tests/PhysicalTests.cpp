#include "TestHelpers.h"
#include "alfa/dsp/Physical.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr std::array<double, 3> kRates = {44100.0, 48000.0, 96000.0};

using Params = std::array<float, kSlotParams>;

/// Normalised value for a physical one (inverse of mapParam for Linear and Log).
float norm(const ParamInfo& info, float value)
{
    if (info.curve == Curve::Log)
        return std::log(value / info.min) / std::log(info.max / info.min);
    return (value - info.min) / (info.max - info.min);
}

template <typename P>
Params defaults()
{
    Params p{};
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = P::params()[i].defaultNorm;
    return p;
}

template <typename P>
void set(Params& p, int index, float value)
{
    p[static_cast<std::size_t>(index)] = norm(P::params()[static_cast<std::size_t>(index)], value);
}

ProcessSpec makeSpec(double sr, int channels = 1, Tier tier = Tier::Studio)
{
    ProcessSpec s;
    s.sampleRate = sr;
    s.channels = channels;
    s.limits = limitsFor(tier);
    return s;
}

/// Runs mono: setParams every control block, then process. input may be shorter than n (zero
/// padded).
std::vector<float> run(Processor& proc, const Params& p, const ControlContext& ctx, long long n,
                       const std::vector<float>& input = {})
{
    std::vector<float> out(static_cast<std::size_t>(n), 0.0f);
    std::copy_n(input.begin(), std::min(input.size(), out.size()), out.begin());
    for (long long i = 0; i < n; i += kControlBlock)
    {
        const auto m = static_cast<int>(std::min<long long>(kControlBlock, n - i));
        proc.setParams(p.data(), ctx);
        float* ch[1] = {out.data() + i};
        proc.process(ch, m);
    }
    return out;
}

/// Hann-windowed DTFT magnitude at hz, by a recursive oscillator (fast enough for searches).
double dtftMag(const float* x, std::size_t n, double hz, double sr)
{
    const double w = 2.0 * kPi * hz / sr;
    const double cw = std::cos(w), sw = std::sin(w);
    const double cv = std::cos(2.0 * kPi / static_cast<double>(n)),
                 sv = std::sin(2.0 * kPi / static_cast<double>(n));
    double re = 0.0, im = 0.0, pc = 1.0, ps = 0.0, hc = 1.0, hs = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(x[i]) * (0.5 - 0.5 * hc);
        re += v * pc;
        im -= v * ps;
        const double npc = pc * cw - ps * sw;
        ps = pc * sw + ps * cw;
        pc = npc;
        const double nhc = hc * cv - hs * sv;
        hs = hc * sv + hs * cv;
        hc = nhc;
    }
    return std::hypot(re, im);
}

/// Frequency of the spectral peak within +-range (relative) of guess: grid, then golden section.
double peakHz(const float* x, std::size_t n, double sr, double guess, double range = 0.01)
{
    const double bin = sr / static_cast<double>(n);
    const double lo0 = guess * (1.0 - range), hi0 = guess * (1.0 + range);
    const double step = std::min(0.25 * bin, (hi0 - lo0) / 8.0);
    double best = guess, bestMag = -1.0;
    for (double f = lo0; f <= hi0; f += step)
    {
        const double m = dtftMag(x, n, f, sr);
        if (m > bestMag)
        {
            bestMag = m;
            best = f;
        }
    }
    double a = best - step, b = best + step;
    const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = dtftMag(x, n, c, sr), fd = dtftMag(x, n, d, sr);
    for (int i = 0; i < 40; ++i)
    {
        if (fc > fd)
        {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = dtftMag(x, n, c, sr);
        }
        else
        {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = dtftMag(x, n, d, sr);
        }
    }
    return 0.5 * (a + b);
}

double cents(double hz, double ref)
{
    return 1200.0 * std::log2(hz / ref);
}

bool allFinite(const std::vector<float>& x)
{
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}

float peakAbs(const std::vector<float>& x)
{
    float m = 0.0f;
    for (float v : x)
        m = std::max(m, std::fabs(v));
    return m;
}

std::vector<float> noise(long long n, float amp, std::uint32_t seed = 1)
{
    alfa::dsp::detail::Random r(seed);
    std::vector<float> x(static_cast<std::size_t>(n));
    for (auto& v : x)
        v = amp * r.nextBipolar();
    return x;
}

std::vector<float> slice(const std::vector<float>& x, double sr, double from, double to)
{
    const auto a = static_cast<std::ptrdiff_t>(from * sr);
    const auto b = static_cast<std::ptrdiff_t>(to * sr);
    return {x.begin() + a, x.begin() + b};
}

/// A plucked-string patch: pluck on, silent input.
Params pluckedString(float t60, float brightness, float dispersion = 0.0f)
{
    using W = WaveguideProcessor;
    auto p = defaults<W>();
    set<W>(p, W::kDecay, t60);
    set<W>(p, W::kBrightness, brightness);
    set<W>(p, W::kDispersion, dispersion);
    set<W>(p, W::kPickup, 0.5f);
    p[W::kPluck] = 1.0f;
    return p;
}

std::vector<float> impulse(long long n)
{
    std::vector<float> x(static_cast<std::size_t>(n), 0.0f);
    x[0] = 1.0f;
    return x;
}

} // namespace

// Waveguide ---------------------------------------------------------------------------------------

TEST_CASE("WAVEGUIDE is in tune from 30 Hz to 4 kHz at every sample rate", "[dsp][physical]")
{
    for (double sr : kRates)
        for (float dispersion : {0.0f, 0.8f})
            for (float hz : {30.0f, 61.7f, 110.0f, 261.6f, 523.3f, 1000.0f, 1975.5f, 3000.0f, 4000.0f})
            {
                WaveguideProcessor wg;
                wg.prepare(makeSpec(sr));
                auto p = pluckedString(8.0f, 0.3f, dispersion);
                ControlContext ctx;
                ctx.f0 = hz;
                const auto out = run(wg, p, ctx, static_cast<long long>(0.6 * sr));
                const auto w = slice(out, sr, 0.1, 0.6);
                const double measured = peakHz(w.data(), w.size(), sr, hz);
                INFO("sr " << sr << " hz " << hz << " dispersion " << dispersion << " measured " << measured);
                REQUIRE(std::fabs(cents(measured, hz)) < 3.0);
                REQUIRE(std::fabs(cents(wg.tunedHz(), hz)) < 0.01);
            }
}

TEST_CASE("WAVEGUIDE pitch and fine offsets, and the absolute frequency in instance slots", "[dsp][physical]")
{
    using W = WaveguideProcessor;
    WaveguideProcessor wg;
    wg.prepare(makeSpec(48000.0));
    auto p = pluckedString(4.0f, 0.3f);
    set<W>(p, W::kPitch, 7.0f);
    set<W>(p, W::kFine, -30.0f);
    ControlContext ctx;
    ctx.f0 = 200.0f;
    wg.setParams(p.data(), ctx);
    REQUIRE(std::fabs(cents(wg.tunedHz(), 200.0 * std::exp2(6.7 / 12.0))) < 0.05);

    set<W>(p, W::kPitch, 0.0f);
    set<W>(p, W::kFine, 0.0f);
    set<W>(p, W::kFrequency, 440.0f);
    ctx.f0 = 0.0f;
    wg.setParams(p.data(), ctx);
    REQUIRE(std::fabs(cents(wg.tunedHz(), 440.0)) < 0.05);
}

TEST_CASE("WAVEGUIDE T60 matches the decay param within 10%", "[dsp][physical]")
{
    for (double sr : kRates)
        for (float t60 : {0.5f, 1.0f, 5.0f})
            for (float brightness : {1.0f, 0.2f})
            {
                WaveguideProcessor wg;
                wg.prepare(makeSpec(sr));
                const auto p = pluckedString(t60, brightness);
                ControlContext ctx;
                ctx.f0 = 220.0f;
                const double t0 = 0.1, t1 = 0.1 + 0.4 * static_cast<double>(t60);
                const auto out = run(wg, p, ctx, static_cast<long long>((t1 + 0.2) * sr));
                const double a0 = toneAmplitude(slice(out, sr, t0, t0 + 0.2), 220.0, sr);
                const double a1 = toneAmplitude(slice(out, sr, t1, t1 + 0.2), 220.0, sr);
                const double drop = 20.0 * std::log10(a0 / a1);
                const double measured = (t1 - t0) * 60.0 / drop;
                INFO("sr " << sr << " t60 " << t60 << " brightness " << brightness << " measured "
                           << measured);
                REQUIRE(std::fabs(measured / static_cast<double>(t60) - 1.0) < 0.1);
            }
}

TEST_CASE("WAVEGUIDE pluck sounds with silent input; without it silence stays silent", "[dsp][physical]")
{
    using W = WaveguideProcessor;
    WaveguideProcessor wg;
    wg.prepare(makeSpec(48000.0));
    ControlContext ctx;
    ctx.f0 = 330.0f;
    auto p = pluckedString(2.0f, 0.5f);
    const auto plucked = run(wg, p, ctx, 24000);
    REQUIRE(rms(plucked) > 0.02);

    p[W::kPluck] = 0.0f;
    wg.reset();
    const auto silent = run(wg, p, ctx, 24000);
    REQUIRE(peakAbs(silent) == 0.0f);

    // The pluck fires once per reset (note-on), not on every block.
    p[W::kPluck] = 1.0f;
    const auto late = run(wg, p, ctx, 4800);
    REQUIRE(peakAbs(late) == 0.0f);
    REQUIRE(wg.tailSamples() >= static_cast<int>(2.0 * 48000.0));
}

TEST_CASE("WAVEGUIDE rings when excited by the input and brightness shapes the spectrum", "[dsp][physical]")
{
    using W = WaveguideProcessor;
    auto measure = [](float brightness)
    {
        WaveguideProcessor wg;
        wg.prepare(makeSpec(48000.0));
        auto p = defaults<W>();
        set<W>(p, W::kBrightness, brightness);
        set<W>(p, W::kDecay, 2.0f);
        ControlContext ctx;
        ctx.f0 = 200.0f;
        const auto out = run(wg, p, ctx, 48000, impulse(1));
        const auto w = slice(out, 48000.0, 0.3, 0.8);
        return toneAmplitude(w, 3000.0, 48000.0) / toneAmplitude(w, 200.0, 48000.0);
    };
    const double dark = measure(0.0f);
    const double bright = measure(1.0f);
    REQUIRE(bright > 0.0);
    REQUIRE(bright > 10.0 * dark);
}

TEST_CASE("WAVEGUIDE dispersion stretches the partials sharp", "[dsp][physical]")
{
    WaveguideProcessor wg;
    wg.prepare(makeSpec(48000.0));
    ControlContext ctx;
    ctx.f0 = 110.0f;
    auto p = pluckedString(4.0f, 1.0f, 1.0f);
    p[WaveguideProcessor::kPickup] = norm(WaveguideProcessor::params()[WaveguideProcessor::kPickup], 0.13f);
    const auto out = run(wg, p, ctx, 48000);
    const auto w = slice(out, 48000.0, 0.1, 0.9);
    REQUIRE(wg.dispersionStages() == 4);
    const double f1 = peakHz(w.data(), w.size(), 48000.0, 110.0);
    const double f5 = peakHz(w.data(), w.size(), 48000.0, 5.0 * 110.0 * 1.02, 0.03);
    REQUIRE(std::fabs(cents(f1, 110.0)) < 3.0);
    REQUIRE(f5 / f1 > 5.02);
}

TEST_CASE("WAVEGUIDE stays stable at a 30 s decay for 60 s with every feature on", "[dsp][physical]")
{
    using W = WaveguideProcessor;
    const double sr = 48000.0;
    WaveguideProcessor wg;
    wg.prepare(makeSpec(sr));
    auto p = pluckedString(30.0f, 1.0f, 1.0f);
    p[W::kTension] = 1.0f;
    ControlContext ctx;
    ctx.f0 = 98.0f;
    const auto out = run(wg, p, ctx, static_cast<long long>(60.0 * sr));
    REQUIRE(allFinite(out));
    REQUIRE(peakAbs(out) <= 4.0f);
    // The fundamental falls 60 dB over 30 s (dispersed upper partials go round the loop more
    // often and fall faster; the tension glide moves the early pitch, hence the peak search).
    auto fundamental = [&](double from)
    {
        const auto w = slice(out, sr, from, from + 1.0);
        return dtftMag(w.data(), w.size(), peakHz(w.data(), w.size(), sr, 98.0, 0.03), sr);
    };
    const double early = fundamental(1.0);
    const double mid = fundamental(31.0);
    const double late = fundamental(59.0);
    const double drop = 20.0 * std::log10(early / mid);
    REQUIRE(drop > 54.0);
    REQUIRE(drop < 66.0);
    REQUIRE(late < mid);
}

TEST_CASE("WAVEGUIDE extreme dispersion across the whole range produces no NaN", "[dsp][physical]")
{
    using W = WaveguideProcessor;
    for (double sr : kRates)
    {
        WaveguideProcessor wg;
        wg.prepare(makeSpec(sr));
        auto p = pluckedString(30.0f, 1.0f, 1.0f);
        p[W::kTension] = 1.0f;
        p[W::kInputGain] = 1.0f;
        const auto in = noise(static_cast<long long>(sr), 1.0f);
        for (float hz : {5.0f, 20.0f, 100.0f, 1000.0f, 5000.0f, 12000.0f, 30000.0f})
        {
            ControlContext ctx;
            ctx.f0 = hz;
            wg.reset();
            const auto out = run(wg, p, ctx, static_cast<long long>(sr), in);
            INFO("sr " << sr << " hz " << hz);
            REQUIRE(allFinite(out));
            REQUIRE(peakAbs(out) < 10.0f);
            REQUIRE(wg.tunedHz() <= sr / 5.0 + 1e-6);
        }
    }
}

TEST_CASE("WAVEGUIDE NaN / Inf input and NaN params give finite output and recover", "[dsp][physical]")
{
    WaveguideProcessor wg;
    wg.prepare(makeSpec(48000.0));
    ControlContext ctx;
    ctx.f0 = 220.0f;
    auto p = defaults<WaveguideProcessor>();
    std::vector<float> in = noise(4800, 0.5f);
    in[100] = std::nanf("");
    in[2000] = INFINITY;
    auto out = run(wg, p, ctx, 4800, in);
    REQUIRE(allFinite(out));

    out = run(wg, p, ctx, 9600, noise(9600, 0.5f, 7));
    REQUIRE(allFinite(out));
    REQUIRE(rms(out) > 1.0e-3);

    Params bad;
    bad.fill(std::nanf(""));
    ControlContext badCtx;
    badCtx.f0 = std::nanf("");
    out = run(wg, bad, badCtx, 4800, noise(4800, 0.5f));
    REQUIRE(allFinite(out));
    badCtx.f0 = INFINITY;
    out = run(wg, bad, badCtx, 4800, noise(4800, 0.5f));
    REQUIRE(allFinite(out));
}

TEST_CASE("WAVEGUIDE extreme and randomly jumping params stay finite and bounded", "[dsp][physical]")
{
    for (double sr : kRates)
    {
        WaveguideProcessor wg;
        wg.prepare(makeSpec(sr));
        ControlContext ctx;
        ctx.f0 = 440.0f;
        Params p{};
        auto out = run(wg, p, ctx, 9600, noise(9600, 1.0f));
        REQUIRE(allFinite(out));
        p.fill(1.0f);
        out = run(wg, p, ctx, 9600, noise(9600, 1.0f));
        REQUIRE(allFinite(out));
        REQUIRE(peakAbs(out) < 10.0f);

        alfa::dsp::detail::Random r(99);
        const auto in = noise(static_cast<long long>(sr), 1.0f, 3);
        std::vector<float> buf(kControlBlock);
        float peak = 0.0f;
        bool finite = true;
        for (std::size_t i = 0; i + kControlBlock <= in.size(); i += kControlBlock)
        {
            for (auto& v : p)
                v = 0.5f + 0.5f * r.nextBipolar();
            ctx.f0 = 20.0f * std::exp2(9.0f * (0.5f + 0.5f * r.nextBipolar()));
            std::copy_n(in.begin() + static_cast<std::ptrdiff_t>(i), kControlBlock, buf.begin());
            wg.setParams(p.data(), ctx);
            float* ch[1] = {buf.data()};
            wg.process(ch, kControlBlock);
            for (float v : buf)
            {
                finite = finite && std::isfinite(v);
                peak = std::max(peak, std::fabs(v));
            }
        }
        REQUIRE(finite);
        REQUIRE(peak < 10.0f);
    }
}

TEST_CASE("WAVEGUIDE stereo channels keep their own state", "[dsp][physical]")
{
    WaveguideProcessor wg;
    wg.prepare(makeSpec(48000.0, 2));
    auto p = defaults<WaveguideProcessor>();
    set<WaveguideProcessor>(p, WaveguideProcessor::kFrequency, 300.0f);
    ControlContext ctx; // instance slot: f0 = 0
    std::vector<float> l(9600, 0.0f), r(9600, 0.0f);
    l[0] = 1.0f;
    for (std::size_t i = 0; i < l.size(); i += kControlBlock)
    {
        wg.setParams(p.data(), ctx);
        float* ch[2] = {l.data() + i, r.data() + i};
        wg.process(ch, kControlBlock);
    }
    REQUIRE(rms(l) > 1.0e-3);
    REQUIRE(peakAbs(r) == 0.0f);
    REQUIRE(std::fabs(zeroCrossingHz(slice(l, 48000.0, 0.05, 0.2), 48000.0) - 300.0) < 30.0);
}

TEST_CASE("WAVEGUIDE setParams and process do not allocate", "[dsp][physical][realtime]")
{
    WaveguideProcessor wg;
    wg.prepare(makeSpec(48000.0, 2));
    auto p = pluckedString(5.0f, 0.5f, 1.0f);
    p[WaveguideProcessor::kTension] = 1.0f;
    std::vector<float> l(kControlBlock, 0.1f), r(kControlBlock, -0.1f);
    ControlContext ctx;
    ctx.f0 = 220.0f;
    AllocationGuard guard;
    for (int b = 0; b < 200; ++b)
    {
        ctx.f0 = 100.0f + static_cast<float>(b);
        wg.setParams(p.data(), ctx);
        float* ch[2] = {l.data(), r.data()};
        wg.process(ch, kControlBlock);
    }
    wg.reset();
    REQUIRE(guard.count() == 0);
}

// Resonator ---------------------------------------------------------------------------------------

TEST_CASE("RESONATOR mode tables hold the textbook ratios", "[dsp][physical]")
{
    using alfa::dsp::detail::PhysicalModeSet;
    const auto& t = alfa::dsp::detail::physicalModeTables();
    auto ratio = [&](PhysicalModeSet s, int k)
    {
        return static_cast<double>(t.ratio[static_cast<std::size_t>(s)][static_cast<std::size_t>(k)]);
    };
    auto near = [](double a, double b)
    {
        return std::fabs(a / b - 1.0) < 1.0e-3;
    };
    REQUIRE(near(ratio(PhysicalModeSet::IdealString, 6), 7.0));
    REQUIRE(near(ratio(PhysicalModeSet::FreeBar, 1), 2.756));
    REQUIRE(near(ratio(PhysicalModeSet::FreeBar, 2), 5.404));
    REQUIRE(near(ratio(PhysicalModeSet::FreeBar, 3), 8.933));
    REQUIRE(near(ratio(PhysicalModeSet::Membrane, 1), 1.594));
    REQUIRE(near(ratio(PhysicalModeSet::Membrane, 2), 2.136));
    REQUIRE(near(ratio(PhysicalModeSet::Membrane, 3), 2.296));
    REQUIRE(near(ratio(PhysicalModeSet::SquarePlate, 1), 2.5));
    REQUIRE(near(ratio(PhysicalModeSet::SquarePlate, 2), 4.0));
    REQUIRE(near(ratio(PhysicalModeSet::SquarePlate, 3), 5.0));
    REQUIRE(near(ratio(PhysicalModeSet::Tube, 3), 7.0));
    REQUIRE(ratio(PhysicalModeSet::StiffString, 9) > 10.0);
    for (int s = 0; s < alfa::dsp::detail::kPhysicalModeSetCount; ++s)
        for (int k = 1; k < alfa::dsp::detail::kPhysicalMaxModes; ++k)
            REQUIRE(ratio(static_cast<PhysicalModeSet>(s), k) >
                    ratio(static_cast<PhysicalModeSet>(s), k - 1));
}

TEST_CASE("RESONATOR spectral peaks sit on the table ratios for every structure", "[dsp][physical]")
{
    using R = ResonatorProcessor;
    const auto& t = alfa::dsp::detail::physicalModeTables();
    for (double sr : kRates)
        for (int s = 0; s < alfa::dsp::detail::kPhysicalModeSetCount; ++s)
        {
            ResonatorProcessor res;
            res.prepare(makeSpec(sr));
            auto p = defaults<R>();
            set<R>(p, R::kFrequency, 150.0f);
            p[R::kStructure] = static_cast<float>(s) / 5.0f;
            set<R>(p, R::kDecay, 3.0f);
            set<R>(p, R::kDampingTilt, 0.0f);
            set<R>(p, R::kBrightness, 0.0f);
            set<R>(p, R::kPosition, 0.137f);
            const auto out = run(res, p, ControlContext{}, static_cast<long long>(sr), impulse(1));
            for (int k = 0; k < 6; ++k)
            {
                const double expected =
                    150.0 *
                    static_cast<double>(t.ratio[static_cast<std::size_t>(s)][static_cast<std::size_t>(k)]);
                INFO("sr " << sr << " structure " << s << " mode " << k);
                REQUIRE(std::fabs(static_cast<double>(res.modeHz(k)) / expected - 1.0) < 1.0e-4);
                const double measured = peakHz(out.data(), out.size(), sr, expected, 0.004);
                REQUIRE(std::fabs(cents(measured, expected)) < 2.0);
            }
        }
}

TEST_CASE("RESONATOR T60 of the lowest mode matches the decay param within 10%", "[dsp][physical]")
{
    using R = ResonatorProcessor;
    for (double sr : kRates)
        for (float t60 : {0.3f, 1.0f, 4.0f})
        {
            ResonatorProcessor res;
            res.prepare(makeSpec(sr));
            auto p = defaults<R>();
            set<R>(p, R::kFrequency, 200.0f);
            set<R>(p, R::kDecay, t60);
            const double t0 = 0.05, t1 = 0.05 + 0.5 * static_cast<double>(t60);
            const auto out =
                run(res, p, ControlContext{}, static_cast<long long>((t1 + 0.15) * sr), impulse(1));
            const double a0 = toneAmplitude(slice(out, sr, t0, t0 + 0.1), 200.0, sr);
            const double a1 = toneAmplitude(slice(out, sr, t1, t1 + 0.1), 200.0, sr);
            const double measured = (t1 - t0) * 60.0 / (20.0 * std::log10(a0 / a1));
            INFO("sr " << sr << " t60 " << t60 << " measured " << measured);
            REQUIRE(std::fabs(measured / static_cast<double>(t60) - 1.0) < 0.1);
        }
}

TEST_CASE("RESONATOR damping tilt shortens the high modes", "[dsp][physical]")
{
    using R = ResonatorProcessor;
    auto decayRatio = [](float tilt)
    {
        const double sr = 48000.0;
        ResonatorProcessor res;
        res.prepare(makeSpec(sr));
        auto p = defaults<R>();
        set<R>(p, R::kFrequency, 200.0f);
        set<R>(p, R::kDecay, 2.0f);
        set<R>(p, R::kDampingTilt, tilt);
        set<R>(p, R::kBrightness, 0.0f);
        set<R>(p, R::kPosition, 0.137f);
        const auto out = run(res, p, ControlContext{}, 48000, impulse(1));
        const auto a = slice(out, sr, 0.0, 0.2), b = slice(out, sr, 0.5, 0.7);
        const double lowDrop = toneAmplitude(a, 200.0, sr) / toneAmplitude(b, 200.0, sr);
        const double highDrop = toneAmplitude(a, 1600.0, sr) / toneAmplitude(b, 1600.0, sr);
        return std::log(highDrop) / std::log(lowDrop); // = T60_low / T60_high
    };
    REQUIRE(std::fabs(decayRatio(0.0f) - 1.0) < 0.1);
    REQUIRE(std::fabs(decayRatio(1.0f) - 8.0) < 0.8); // ratio 8 at tilt 1
}

TEST_CASE("RESONATOR respects the tier's mode limit and the count fraction", "[dsp][physical]")
{
    using R = ResonatorProcessor;
    for (Tier tier : {Tier::Lite, Tier::Standard, Tier::Pro, Tier::Studio})
    {
        ResonatorProcessor res;
        res.prepare(makeSpec(48000.0, 1, tier));
        const int limit = limitsFor(tier).resonatorModes;
        REQUIRE(res.modeLimit() == limit);
        auto p = defaults<R>();
        set<R>(p, R::kFrequency, 100.0f);
        set<R>(p, R::kBrightness, 0.0f);
        set<R>(p, R::kPosition, 0.137f);
        const auto out = run(res, p, ControlContext{}, 24000, impulse(1));
        REQUIRE(res.activeModes() == limit);
        const double last = toneAmplitude(out, 100.0 * limit, 48000.0);
        const double beyond = toneAmplitude(out, 100.0 * (limit + 1), 48000.0);
        REQUIRE(last > 1.0e-3);
        REQUIRE(beyond < last * 1.0e-2);

        p[R::kModeCount] = 0.5f;
        res.setParams(p.data(), ControlContext{});
        REQUIRE(res.activeModes() == limit / 2);
        p[R::kModeCount] = 0.0f;
        res.setParams(p.data(), ControlContext{});
        REQUIRE(res.activeModes() == 1);
    }
}

TEST_CASE("RESONATOR skips modes at or above Nyquist", "[dsp][physical]")
{
    using R = ResonatorProcessor;
    const double sr = 44100.0;
    ResonatorProcessor res;
    res.prepare(makeSpec(sr));
    auto p = defaults<R>();
    set<R>(p, R::kFrequency, 2000.0f);
    set<R>(p, R::kBrightness, 6.0f);
    set<R>(p, R::kPosition, 0.137f);
    const auto out = run(res, p, ControlContext{}, 44100, impulse(1));
    REQUIRE(res.activeModes() == 10); // 2 .. 20 kHz; 22 kHz >= 0.48 sr
    REQUIRE(res.modeHz(10) == 0.0f);
    // Mode 12 (24 kHz) would alias to 20.1 kHz, mode 11 (22 kHz) to 22.1 kHz.
    REQUIRE(toneAmplitude(out, 20100.0, sr) < 1.0e-3 * toneAmplitude(out, 20000.0, sr));
    REQUIRE(allFinite(out));
}

TEST_CASE("RESONATOR structure morph, stretch and spread move the modes smoothly", "[dsp][physical]")
{
    using R = ResonatorProcessor;
    ResonatorProcessor res;
    res.prepare(makeSpec(48000.0));
    auto p = defaults<R>();
    set<R>(p, R::kFrequency, 100.0f);
    p[R::kStructure] = 0.1f; // halfway between string and stiff string
    res.setParams(p.data(), ControlContext{});
    const auto& t = alfa::dsp::detail::physicalModeTables();
    const double geo = std::sqrt(static_cast<double>(t.ratio[0][9]) * static_cast<double>(t.ratio[1][9]));
    REQUIRE(std::fabs(res.modeHz(9) / (100.0 * geo) - 1.0) < 1.0e-4);

    p[R::kStructure] = 0.0f;
    set<R>(p, R::kInharmonicity, 0.1f);
    res.setParams(p.data(), ControlContext{});
    REQUIRE(std::fabs(res.modeHz(3) / (100.0 * std::pow(4.0, 1.1)) - 1.0) < 1.0e-4);

    set<R>(p, R::kInharmonicity, 0.0f);
    set<R>(p, R::kSpread, 50.0f);
    res.setParams(p.data(), ControlContext{});
    for (int k = 0; k < 8; ++k)
        REQUIRE(std::fabs(cents(res.modeHz(k), 100.0 * (k + 1))) <= 50.01);

    // Key tracking: voice f0 and pitch offset.
    set<R>(p, R::kSpread, 0.0f);
    set<R>(p, R::kPitch, 12.0f);
    ControlContext ctx;
    ctx.f0 = 110.0f;
    res.setParams(p.data(), ctx);
    REQUIRE(std::fabs(res.modeHz(0) / 220.0f - 1.0f) < 1.0e-4f);
}

TEST_CASE("RESONATOR stays stable at a 30 s decay for 60 s under constant excitation", "[dsp][physical]")
{
    using R = ResonatorProcessor;
    const double sr = 48000.0;
    ResonatorProcessor res;
    res.prepare(makeSpec(sr));
    auto p = defaults<R>();
    set<R>(p, R::kFrequency, 30.0f);
    set<R>(p, R::kDecay, 30.0f);
    set<R>(p, R::kDampingTilt, 0.0f);
    p[R::kInputGain] = 1.0f;
    // One second of loud noise, then 59 s of ringing.
    const auto out = run(res, p, ControlContext{}, static_cast<long long>(60.0 * sr), noise(48000, 1.0f));
    REQUIRE(allFinite(out));
    const double early = rms(slice(out, sr, 2.0, 3.0));
    const double late = rms(slice(out, sr, 32.0, 33.0));
    REQUIRE(early > 0.0);
    const double drop = 20.0 * std::log10(early / late);
    REQUIRE(drop > 50.0);
    REQUIRE(drop < 70.0);
}

TEST_CASE("RESONATOR silence in gives silence out", "[dsp][physical]")
{
    ResonatorProcessor res;
    res.prepare(makeSpec(48000.0));
    const auto out = run(res, defaults<ResonatorProcessor>(), ControlContext{}, 9600);
    REQUIRE(peakAbs(out) == 0.0f);
    REQUIRE(res.tailSamples() >= static_cast<int>(1.5 * 48000.0));
}

TEST_CASE("RESONATOR NaN / Inf input and NaN params give finite output and recover", "[dsp][physical]")
{
    ResonatorProcessor res;
    res.prepare(makeSpec(48000.0, 2));
    auto p = defaults<ResonatorProcessor>();
    std::vector<float> l = noise(4800, 0.5f), r = noise(4800, 0.5f, 5);
    l[77] = std::nanf("");
    r[3000] = -INFINITY;
    for (std::size_t i = 0; i < l.size(); i += kControlBlock)
    {
        res.setParams(p.data(), ControlContext{});
        float* ch[2] = {l.data() + i, r.data() + i};
        res.process(ch, kControlBlock);
    }
    REQUIRE(allFinite(l));
    REQUIRE(allFinite(r));

    ResonatorProcessor mono;
    mono.prepare(makeSpec(48000.0));
    auto out = run(mono, p, ControlContext{}, 4800, std::vector<float>{std::nanf("")});
    REQUIRE(allFinite(out));
    out = run(mono, p, ControlContext{}, 4800, noise(4800, 0.5f));
    REQUIRE(rms(out) > 1.0e-3);

    Params bad;
    bad.fill(std::nanf(""));
    ControlContext badCtx;
    badCtx.f0 = std::nanf("");
    out = run(mono, bad, badCtx, 4800, noise(4800, 0.5f));
    REQUIRE(allFinite(out));
}

TEST_CASE("RESONATOR extreme and randomly jumping params stay finite and bounded", "[dsp][physical]")
{
    for (double sr : kRates)
    {
        ResonatorProcessor res;
        res.prepare(makeSpec(sr));
        Params p{};
        ControlContext ctx;
        ctx.f0 = 440.0f;
        auto out = run(res, p, ctx, 9600, noise(9600, 1.0f));
        REQUIRE(allFinite(out));
        p.fill(1.0f);
        out = run(res, p, ctx, 9600, noise(9600, 1.0f));
        REQUIRE(allFinite(out));
        REQUIRE(peakAbs(out) < 1.0e4f);

        alfa::dsp::detail::Random r(42);
        const auto in = noise(static_cast<long long>(sr), 1.0f, 3);
        std::vector<float> buf(kControlBlock);
        bool finite = true;
        float peak = 0.0f;
        for (std::size_t i = 0; i + kControlBlock <= in.size(); i += kControlBlock)
        {
            for (auto& v : p)
                v = 0.5f + 0.5f * r.nextBipolar();
            ctx.f0 = 20.0f * std::exp2(10.0f * (0.5f + 0.5f * r.nextBipolar()));
            std::copy_n(in.begin() + static_cast<std::ptrdiff_t>(i), kControlBlock, buf.begin());
            res.setParams(p.data(), ctx);
            float* ch[1] = {buf.data()};
            res.process(ch, kControlBlock);
            for (float v : buf)
            {
                finite = finite && std::isfinite(v);
                peak = std::max(peak, std::fabs(v));
            }
        }
        REQUIRE(finite);
        REQUIRE(peak < 1.0e4f);
    }
}

TEST_CASE("RESONATOR setParams and process do not allocate", "[dsp][physical][realtime]")
{
    ResonatorProcessor res;
    res.prepare(makeSpec(48000.0, 2));
    auto p = defaults<ResonatorProcessor>();
    std::vector<float> l(kControlBlock, 0.1f), r(kControlBlock, -0.1f);
    ControlContext ctx;
    AllocationGuard guard;
    for (int b = 0; b < 200; ++b)
    {
        ctx.f0 = 100.0f + static_cast<float>(b);
        p[ResonatorProcessor::kStructure] = static_cast<float>(b % 50) / 50.0f;
        res.setParams(p.data(), ctx);
        float* ch[2] = {l.data(), r.data()};
        res.process(ch, kControlBlock);
    }
    res.reset();
    REQUIRE(guard.count() == 0);
}

TEST_CASE("PHYSICAL param tables are complete and defaults are in range", "[dsp][physical]")
{
    for (const ParamTable* t : {&WaveguideProcessor::params(), &ResonatorProcessor::params()})
        for (const auto& info : *t)
        {
            REQUIRE(info.defaultNorm >= 0.0f);
            REQUIRE(info.defaultNorm <= 1.0f);
            if (!info.name.empty() && info.curve == Curve::Log)
                REQUIRE(info.min > 0.0f);
        }
    // Documented neutral values.
    const auto& w = WaveguideProcessor::params();
    REQUIRE(
        std::fabs(mapParam(w[WaveguideProcessor::kFrequency], w[WaveguideProcessor::kFrequency].defaultNorm) -
                  220.0f) < 0.5f);
    REQUIRE(std::fabs(mapParam(w[WaveguideProcessor::kDecay], w[WaveguideProcessor::kDecay].defaultNorm) -
                      2.0f) < 0.01f);
    const auto& r = ResonatorProcessor::params();
    REQUIRE(std::fabs(mapParam(r[ResonatorProcessor::kDecay], r[ResonatorProcessor::kDecay].defaultNorm) -
                      1.5f) < 0.01f);
    REQUIRE(
        std::fabs(mapParam(r[ResonatorProcessor::kPosition], r[ResonatorProcessor::kPosition].defaultNorm) -
                  0.23f) < 0.001f);
}

// Benchmarks --------------------------------------------------------------------------------------

namespace
{
template <typename P>
void bench(const char* name, const Params& p, bool pluckReset, float f0Base)
{
    constexpr int kVoices = 16;
    constexpr int kBlock = 128;
    constexpr int kBlocks = 48000 / kBlock * 2; // 2 s
    std::vector<std::unique_ptr<P>> voices;
    std::vector<std::vector<float>> buf(kVoices, std::vector<float>(kBlock));
    for (int v = 0; v < kVoices; ++v)
    {
        voices.push_back(std::make_unique<P>());
        voices.back()->prepare(makeSpec(48000.0));
    }
    alfa::dsp::detail::Random r(1);
    const auto start = std::chrono::steady_clock::now();
    for (int b = 0; b < kBlocks; ++b)
        for (int v = 0; v < kVoices; ++v)
        {
            auto& proc = *voices[static_cast<std::size_t>(v)];
            if (pluckReset && b % 100 == 0)
                proc.reset();
            auto& x = buf[static_cast<std::size_t>(v)];
            for (auto& s : x)
                s = 0.1f * r.nextBipolar();
            ControlContext ctx;
            ctx.f0 = f0Base * static_cast<float>(v + 1) * (1.0f + 0.001f * static_cast<float>(b % 7));
            for (int t = 0; t < kBlock; t += kControlBlock)
            {
                proc.setParams(p.data(), ctx);
                float* ch[1] = {x.data() + t};
                proc.process(ch, kControlBlock);
            }
        }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    const double perBlock = us / kBlocks;
    std::printf("%s: %.2f us per 128-sample block for 16 mono voices at 48 kHz (%.2f%% of one core)\n", name,
                perBlock, 100.0 * perBlock / (1.0e6 * kBlock / 48000.0));
}
} // namespace

TEST_CASE("WAVEGUIDE cost, 16 voices, dispersion + tension + pluck", "[.bench][dsp][physical]")
{
    using W = WaveguideProcessor;
    auto p = pluckedString(5.0f, 0.6f, 1.0f);
    p[W::kTension] = 0.5f;
    p[W::kInputGain] = 0.3f;
    bench<W>("WAVEGUIDE", p, true, 55.0f);
}

TEST_CASE("RESONATOR cost, 16 voices x 32 modes", "[.bench][dsp][physical]")
{
    using R = ResonatorProcessor;
    auto p = defaults<R>();
    p[R::kStructure] = 0.1f; // string / stiff string morph keeps all 32 modes below Nyquist
    set<R>(p, R::kSpread, 10.0f);
    set<R>(p, R::kDecay, 3.0f);
    bench<R>("RESONATOR", p, false, 25.0f);
}
