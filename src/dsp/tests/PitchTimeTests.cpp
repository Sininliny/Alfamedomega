#include "TestHelpers.h"
#include "alfa/dsp/PitchTime.h"
#include "alfa/dsp/detail/FastMath.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace alfa::dsp;
using alfa::dsp::test::AllocationGuard;
using alfa::dsp::test::rms;
using alfa::dsp::test::toneAmplitude;
using alfa::dsp::test::zeroCrossingHz;

namespace
{

constexpr double kPiD = 3.14159265358979323846;
constexpr std::array<double, 3> kRates{44100.0, 48000.0, 96000.0};

std::size_t u(long long i)
{
    return static_cast<std::size_t>(i);
}

std::vector<float> sine(double hz, double sr, long long n, float amp = 0.5f)
{
    std::vector<float> x(u(n));
    for (long long i = 0; i < n; ++i)
        x[u(i)] = amp * static_cast<float>(std::sin(2.0 * kPiD * hz * static_cast<double>(i) / sr));
    return x;
}

std::vector<float> noise(long long n, std::uint32_t seed = 1, float amp = 0.5f)
{
    detail::Random r(seed);
    std::vector<float> x(u(n));
    for (auto& v : x)
        v = amp * r.nextBipolar();
    return x;
}

using Params = std::array<float, kSlotParams>;

Params defaults()
{
    Params p{};
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = PitchShiftProcessor::params()[i].defaultNorm;
    return p;
}

float shiftNorm(double semitones)
{
    return static_cast<float>(0.5 + semitones / 48.0);
}

ProcessSpec spec(double sr, int channels = 2)
{
    ProcessSpec s;
    s.sampleRate = sr;
    s.channels = channels;
    return s;
}

std::vector<std::vector<float>> run(Processor& p, const Params& params, std::vector<std::vector<float>> x)
{
    detail::Random r(17);
    const auto n = static_cast<long long>(x[0].size());
    long long pos = 0;
    std::array<float*, 2> ptrs{};
    while (pos < n)
    {
        const int len = static_cast<int>(std::min<long long>(n - pos, 1 + (r.nextU32() % kControlBlock)));
        p.setParams(params.data(), {});
        for (std::size_t c = 0; c < x.size(); ++c)
            ptrs[c] = x[c].data() + pos;
        p.process(ptrs.data(), len);
        pos += len;
    }
    return x;
}

double cents(double hz, double reference)
{
    return 1200.0 * std::log2(hz / reference);
}

/// Frequency of the strongest spectral line near guess: coarse grid, then golden-section on the
/// Hann-windowed DFT magnitude.
double peakHz(const std::vector<float>& x, double sr, double guess, double span)
{
    double best = guess;
    double bestA = -1.0;
    for (double f = guess - span; f <= guess + span; f += 0.5)
    {
        const double a = toneAmplitude(x, f, sr);
        if (a > bestA)
        {
            bestA = a;
            best = f;
        }
    }
    double lo = best - 0.5;
    double hi = best + 0.5;
    for (int i = 0; i < 40; ++i)
    {
        const double m1 = lo + (hi - lo) * 0.382;
        const double m2 = lo + (hi - lo) * 0.618;
        if (toneAmplitude(x, m1, sr) > toneAmplitude(x, m2, sr))
            hi = m2;
        else
            lo = m1;
    }
    return 0.5 * (lo + hi);
}

} // namespace

TEST_CASE("PITCH SHIFT param table", "[dsp][pitchtime]")
{
    const auto& t = PitchShiftProcessor::params();
    REQUIRE(mapParam(t[0], 0.5f) == 0.0f);
    REQUIRE(mapParam(t[1], 0.5f) == 0.0f);
    REQUIRE(std::fabs(mapParam(t[2], t[2].defaultNorm) - 50.0f) < 0.1f);
    for (int i = 9; i < kSlotParams; ++i)
        REQUIRE(t[static_cast<std::size_t>(i)].name.empty());
}

TEST_CASE("PITCH SHIFT moves a 440 Hz tone up 7 semitones within 5 cents", "[dsp][pitchtime]")
{
    for (const double sr : kRates)
    {
        PitchShiftProcessor p;
        p.prepare(spec(sr, 1));
        auto params = defaults();
        params[0] = shiftNorm(7.0);
        const auto y = run(p, params, {sine(440.0, sr, static_cast<long long>(1.5 * sr))})[0];
        const std::vector<float> steady(y.begin() + static_cast<long>(0.5 * sr), y.end());
        const double target = 440.0 * std::exp2(7.0 / 12.0);
        const double zc = zeroCrossingHz(steady, sr);
        const double peak = peakHz(steady, sr, target, 20.0);
        INFO("sr " << sr << " zero-crossing " << zc << " peak " << peak);
        REQUIRE(std::fabs(cents(zc, target)) < 5.0);
        REQUIRE(std::fabs(cents(peak, target)) < 5.0);
        REQUIRE(toneAmplitude(steady, target, sr) > 0.35);
    }
}

TEST_CASE("PITCH SHIFT shifts other intervals and grain sizes accurately", "[dsp][pitchtime]")
{
    constexpr double sr = 48000.0;
    struct Case
    {
        double semitones;
        double hz;
        float grain;
    };
    for (const auto c : {Case{-12.0, 440.0, 0.7f}, Case{12.0, 220.0, 0.3f}, Case{-5.0, 330.0, 1.0f},
                         Case{3.0, 523.25, 0.0f}, Case{24.0, 110.0, 0.7f}})
    {
        PitchShiftProcessor p;
        p.prepare(spec(sr, 1));
        auto params = defaults();
        params[0] = shiftNorm(c.semitones);
        params[2] = c.grain;
        const auto y = run(p, params, {sine(c.hz, sr, 72000)})[0];
        const std::vector<float> steady(y.begin() + 24000, y.end());
        const double target = c.hz * std::exp2(c.semitones / 12.0);
        const double zc = zeroCrossingHz(steady, sr);
        INFO("shift " << c.semitones << " from " << c.hz << " measured " << zc);
        REQUIRE(std::fabs(cents(zc, target)) < 5.0);
    }
}

TEST_CASE("PITCH SHIFT reports its latency and fine tune, spread work", "[dsp][pitchtime]")
{
    constexpr double sr = 48000.0;
    PitchShiftProcessor p;
    p.prepare(spec(sr));
    auto params = defaults();
    p.setParams(params.data(), {});
    const int l0 = p.latencySamples();
    REQUIRE(l0 > 0);
    REQUIRE(l0 < static_cast<int>(0.01 * sr)); // low latency at unity
    params[0] = shiftNorm(7.0);
    p.setParams(params.data(), {});
    REQUIRE(p.latencySamples() > l0);

    // Unity shift: an impulse comes out close to the reported latency.
    params[0] = 0.5f;
    p.setParams(params.data(), {});
    p.reset();
    std::vector<float> imp(u(8192), 0.0f);
    imp[2000] = 1.0f;
    const auto y = run(p, params, {imp, imp})[0];
    const auto peak =
        std::max_element(y.begin(), y.end(), [](float a, float b) { return std::fabs(a) < std::fabs(b); }) -
        y.begin();
    INFO("peak at " << peak - 2000 << " reported " << p.latencySamples());
    REQUIRE(std::abs(static_cast<int>(peak - 2000) - p.latencySamples()) < static_cast<int>(0.008 * sr));

    // Spread detunes left down and right up.
    params[5] = 1.0f; // 50 cents
    p.reset();
    const auto s = run(p, params, {sine(440.0, sr, 48000), sine(440.0, sr, 48000)});
    const std::vector<float> left(s[0].begin() + 12000, s[0].end());
    const std::vector<float> right(s[1].begin() + 12000, s[1].end());
    REQUIRE(std::fabs(cents(zeroCrossingHz(left, sr), 440.0) + 50.0) < 5.0);
    REQUIRE(std::fabs(cents(zeroCrossingHz(right, sr), 440.0) - 50.0) < 5.0);
}

TEST_CASE("PITCH SHIFT silence in gives silence out", "[dsp][pitchtime]")
{
    PitchShiftProcessor p;
    p.prepare(spec(48000.0));
    auto params = defaults();
    params[0] = 0.9f;
    params[3] = 1.0f;
    std::vector<float> z(u(8192), 0.0f);
    const auto y = run(p, params, {z, z});
    REQUIRE(rms(y[0]) == 0.0);
    REQUIRE(rms(y[1]) == 0.0);
}

TEST_CASE("PITCH SHIFT survives NaN and Inf input and NaN params", "[dsp][pitchtime]")
{
    PitchShiftProcessor p;
    p.prepare(spec(48000.0));
    auto params = defaults();
    params[0] = 0.7f;
    params[3] = 0.8f;
    auto x = noise(8192);
    x[500] = std::numeric_limits<float>::quiet_NaN();
    x[4000] = -std::numeric_limits<float>::infinity();
    auto y = run(p, params, {x, x});
    for (const auto& ch : y)
        for (const float v : ch)
            REQUIRE(std::isfinite(v));
    Params nanParams;
    nanParams.fill(std::numeric_limits<float>::quiet_NaN());
    y = run(p, nanParams, {noise(8192, 2), noise(8192, 3)});
    for (const auto& ch : y)
        for (const float v : ch)
            REQUIRE(std::isfinite(v));
    // Recovers: a tone still comes out at the right pitch.
    p.reset();
    const auto t = run(p, defaults(), {sine(440.0, 48000.0, 24000), sine(440.0, 48000.0, 24000)})[0];
    REQUIRE(std::fabs(zeroCrossingHz(std::vector<float>(t.begin() + 4000, t.end()), 48000.0) - 440.0) < 1.0);
}

TEST_CASE("PITCH SHIFT stays bounded at maximum feedback and under extreme modulation", "[dsp][pitchtime]")
{
    for (const double sr : kRates)
    {
        PitchShiftProcessor p;
        p.prepare(spec(sr));
        detail::Random r(7);
        const auto n = static_cast<long long>(2.0 * sr);
        auto l = noise(n, 1, 1.0f);
        auto rr = sine(3000.0, sr, n, 1.0f);
        std::array<float*, 2> ptr{};
        Params params{};
        for (long long pos = 0; pos < n; pos += 32)
        {
            const auto block = pos / 32;
            for (auto& v : params)
                v = block < 1000 ? 1.0f : (block < 2000 ? 0.0f : 0.5f + 0.5f * r.nextBipolar());
            params[3] = 1.0f; // feedback at its maximum throughout
            p.setParams(params.data(), {});
            ptr[0] = l.data() + pos;
            ptr[1] = rr.data() + pos;
            p.process(ptr.data(), static_cast<int>(std::min<long long>(32, n - pos)));
        }
        for (const auto* ch : {&l, &rr})
            for (const float v : *ch)
            {
                REQUIRE(std::isfinite(v));
                REQUIRE(std::fabs(v) < 64.0f);
            }
    }
}

TEST_CASE("PITCH SHIFT feedback rings out and the tail covers it", "[dsp][pitchtime]")
{
    constexpr double sr = 48000.0;
    PitchShiftProcessor p;
    p.prepare(spec(sr, 1));
    auto params = defaults();
    params[0] = shiftNorm(12.0);
    params[3] = 0.7f / 0.9f;
    auto x = sine(220.0, sr, 4800);
    x.resize(u(96000), 0.0f);
    const auto y = run(p, params, {x})[0];
    REQUIRE(rms(y.data() + 9600, 4800) > 1e-3); // still ringing after the input stopped
    const int tail = p.tailSamples();
    REQUIRE(tail > 4800);
    REQUIRE(rms(y.data() + 4800 + tail, 4800) < 2e-3);
}

TEST_CASE("PITCH SHIFT does not allocate in setParams or process", "[dsp][pitchtime]")
{
    PitchShiftProcessor p;
    p.prepare(spec(48000.0));
    auto params = defaults();
    params[3] = 0.5f;
    params[5] = 0.5f;
    auto l = noise(8192, 1);
    auto r = noise(8192, 2);
    std::array<float*, 2> ptr{};
    detail::Random rnd(3);
    AllocationGuard guard;
    for (std::size_t pos = 0; pos < l.size(); pos += 32)
    {
        params[0] = 0.5f + 0.5f * rnd.nextBipolar();
        params[2] = 0.5f + 0.5f * rnd.nextBipolar();
        p.setParams(params.data(), {});
        ptr[0] = l.data() + pos;
        ptr[1] = r.data() + pos;
        p.process(ptr.data(), 32);
    }
    REQUIRE(guard.count() == 0);
}

// Offline -------------------------------------------------------------------------------------

TEST_CASE("Offline time stretch x2 doubles the duration and keeps the pitch", "[dsp][pitchtime]")
{
    for (const double sr : kRates)
    {
        const auto n = static_cast<long long>(sr);
        const auto x = sine(440.0, sr, n);
        const auto y = offline::timeStretch(x.data(), x.size(), 2.0, 2048);
        REQUIRE(y.size() == 2 * x.size());
        const std::vector<float> mid(y.begin() + n / 2, y.end() - n / 2);
        const double zc = zeroCrossingHz(mid, sr);
        INFO("sr " << sr << " measured " << zc);
        REQUIRE(std::fabs(cents(zc, 440.0)) < 5.0);
        REQUIRE(std::fabs(rms(mid) - rms(x)) < 0.05 * rms(x));
    }
    // Compression halves it.
    const auto x = sine(440.0, 48000.0, 48000);
    const auto y = offline::timeStretch(x.data(), x.size(), 0.5, 2048);
    REQUIRE(y.size() == x.size() / 2);
    const std::vector<float> mid(y.begin() + 4000, y.end() - 4000);
    REQUIRE(std::fabs(cents(zeroCrossingHz(mid, 48000.0), 440.0)) < 5.0);
}

TEST_CASE("Offline resampler and pitch shift", "[dsp][pitchtime]")
{
    constexpr double sr = 48000.0;
    const auto x = sine(1000.0, sr, 24000);
    // Reading at step 1.5 raises the pitch by 1.5.
    const auto y = offline::resample(x.data(), x.size(), 1.5, 12000);
    const std::vector<float> mid(y.begin() + 1000, y.end() - 1000);
    REQUIRE(std::fabs(zeroCrossingHz(mid, sr) - 1500.0) < 0.2);
    REQUIRE(std::fabs(rms(mid) - rms(x)) < 0.01);

    const auto a = sine(440.0, sr, 48000);
    const auto b = offline::pitchShiftOffline(a.data(), a.size(), 7.0, 2048);
    REQUIRE(b.size() == a.size());
    const std::vector<float> bm(b.begin() + 6000, b.end() - 6000);
    REQUIRE(std::fabs(cents(zeroCrossingHz(bm, sr), 440.0 * std::exp2(7.0 / 12.0))) < 5.0);
}

TEST_CASE("PITCH SHIFT cost, 1 stereo instance", "[.bench][dsp][pitchtime]")
{
    PitchShiftProcessor p;
    p.prepare(spec(48000.0));
    auto params = defaults();
    params[0] = shiftNorm(7.0);
    params[2] = 0.0f; // 10 ms grains: most correlation searches per second
    params[3] = 0.5f;
    params[5] = 0.5f;
    params[7] = 0.8f;
    auto l = noise(48000, 1);
    auto r = noise(48000, 2);
    std::array<float*, 2> ptr{};
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t pos = 0; pos < l.size(); pos += 32)
    {
        p.setParams(params.data(), {});
        ptr[0] = l.data() + pos;
        ptr[1] = r.data() + pos;
        p.process(ptr.data(), 32);
    }
    const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("PITCH SHIFT: %.2f us per 128-sample stereo block (sink %g)\n", us / (48000.0 / 128.0),
                static_cast<double>(l[100] + r[200]));
}
