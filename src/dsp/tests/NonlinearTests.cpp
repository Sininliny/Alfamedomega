#include "TestHelpers.h"
#include "alfa/dsp/Nonlinear.h"
#include "alfa/dsp/detail/FastMath.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{
using NP = NonlinearProcessor;
using Params = std::array<float, kSlotParams>;
constexpr double kSr = 48000.0;
constexpr double kPi = 3.14159265358979323846;

std::size_t u(long long i)
{
    return static_cast<std::size_t>(i);
}

Params defaults()
{
    Params p{};
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = NP::params()[i].defaultNorm;
    return p;
}

float norm(int param, float value)
{
    const auto& info = NP::params()[u(param)];
    if (info.curve == Curve::Log)
        return std::log(value / info.min) / std::log(info.max / info.min);
    return (value - info.min) / (info.max - info.min);
}

/// Params with oversampling 1x, auto gain off, everything else neutral.
Params plain()
{
    Params p = defaults();
    p[NP::kOversampling] = 0.0f;
    p[NP::kAutoGain] = 0.0f;
    return p;
}

float oversampling(int factor)
{
    return factor == 4 ? 1.0f : factor == 2 ? 0.5f : 0.0f;
}

std::vector<float> sine(double hz, double sr, long long n, float amp)
{
    std::vector<float> x(u(n));
    for (long long i = 0; i < n; ++i)
        x[u(i)] = amp * static_cast<float>(std::sin(2.0 * kPi * hz * static_cast<double>(i) / sr));
    return x;
}

/// Runs mono x through p in control blocks, calling setParams before every block.
void run(NP& p, std::vector<float>& x, const Params& params, int block = kControlBlock)
{
    for (std::size_t pos = 0; pos < x.size(); pos += u(block))
    {
        const int n = static_cast<int>(std::min(u(block), x.size() - pos));
        p.setParams(params.data(), {});
        float* ch[1] = {x.data() + pos};
        p.process(ch, n);
    }
}

NP& prepared(NP& p, double sr = kSr, int channels = 1)
{
    ProcessSpec s;
    s.sampleRate = sr;
    s.channels = channels;
    p.prepare(s);
    return p;
}

std::vector<float> tail(const std::vector<float>& x, std::size_t n)
{
    return {x.end() - static_cast<std::ptrdiff_t>(n), x.end()};
}

double maxAbs(const std::vector<float>& x)
{
    double m = 0.0;
    for (float v : x)
        m = std::max(m, static_cast<double>(std::fabs(v)));
    return m;
}

bool allFinite(const std::vector<float>& x)
{
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
} // namespace

TEST_CASE("Nonlinear param table is well formed", "[dsp][nonlinear]")
{
    const auto& t = NP::params();
    for (int i = 0; i < kSlotParams; ++i)
    {
        const auto& info = t[u(i)];
        if (i < NP::kParamCount)
        {
            REQUIRE(!info.name.empty());
            REQUIRE(info.defaultNorm >= 0.0f);
            REQUIRE(info.defaultNorm <= 1.0f);
        }
        else
            REQUIRE(info.name.empty());
    }
    REQUIRE(mapParam(t[NP::kBias], t[NP::kBias].defaultNorm) == 0.0f);
    REQUIRE(mapParam(t[NP::kTrim], t[NP::kTrim].defaultNorm) == 0.0f);
    REQUIRE(mapParam(t[NP::kBits], t[NP::kBits].defaultNorm) == 24.0f);
    REQUIRE(mapParam(t[NP::kDownsample], t[NP::kDownsample].defaultNorm) == 1.0f);
    REQUIRE(mapParam(t[NP::kTone], t[NP::kTone].defaultNorm) == 0.0f);
}

TEST_CASE("Nonlinear shapes are bounded and continuous across the morph", "[dsp][nonlinear]")
{
    for (float type = 0.0f; type <= 7.0f; type += 0.125f)
        for (float character : {0.0f, 0.5f, 1.0f})
        {
            float prev = NP::shape(-300.0f, type, character);
            for (float x = -300.0f; x <= 300.0f; x += 0.01f)
            {
                const float y = NP::shape(x, type, character);
                REQUIRE(std::fabs(y) <= 1.0f + 1e-5f);
                // The steepest slope is the folder's (pi/2 * 4); 0.01 steps move at most ~0.07.
                REQUIRE(std::fabs(y - prev) < 0.08f);
                prev = y;
            }
            REQUIRE(NP::shape(0.0f, type, character) == 0.0f);
        }
    // Triangle folder with whole cells: an exact triangle.
    REQUIRE(std::fabs(NP::shape(2.0f, 5.0f, 0.0f)) < 1e-6f);
    REQUIRE(std::fabs(NP::shape(3.0f, 5.0f, 0.0f) + 1.0f) < 1e-6f);
    REQUIRE(std::fabs(NP::shape(0.5f, 5.0f, 0.0f) - 0.5f) < 1e-6f);
    // Tube is asymmetric.
    REQUIRE(NP::shape(2.0f, 1.0f, 0.5f) > -NP::shape(-2.0f, 1.0f, 0.5f) + 0.1f);
}

TEST_CASE("Nonlinear transfer is near linear at low drive", "[dsp][nonlinear]")
{
    for (int factor : {1, 2, 4})
        for (float type : {0.0f, 3.0f})
        {
            NP p;
            prepared(p);
            Params params = plain();
            params[NP::kOversampling] = oversampling(factor);
            params[NP::kType] = norm(NP::kType, type);
            auto x = sine(440.0, kSr, 48000, 0.05f);
            run(p, x, params);
            const auto out = tail(x, 24000);
            const double f1 = toneAmplitude(out, 440.0, kSr);
            REQUIRE(std::fabs(f1 - 0.05) < 0.0005);
            REQUIRE(toneAmplitude(out, 1320.0, kSr) < 1e-4);
            REQUIRE(toneAmplitude(out, 880.0, kSr) < 1e-4);
        }
}

TEST_CASE("Nonlinear hard clip bounds the output", "[dsp][nonlinear]")
{
    NP p;
    prepared(p);
    p.setDcBlockerEnabled(false);
    Params params = plain();
    params[NP::kType] = norm(NP::kType, 3.0f);
    params[NP::kDrive] = norm(NP::kDrive, 30.0f);
    auto x = sine(220.0, kSr, 48000, 0.9f);
    run(p, x, params);
    const auto out = tail(x, 24000);
    REQUIRE(maxAbs(out) <= 1.0);
    REQUIRE(maxAbs(out) > 0.95);
    // Nearly square: RMS close to the peak.
    REQUIRE(rms(out) > 0.9);

    // The 10 Hz DC blocker tilts the flat tops of a 220 Hz square only slightly.
    NP b;
    prepared(b);
    auto z = sine(220.0, kSr, 48000, 0.9f);
    run(b, z, params);
    REQUIRE(maxAbs(tail(z, 24000)) < 1.1);

    // Oversampled, the band-limited square overshoots only a little (Gibbs).
    NP q;
    prepared(q);
    params[NP::kOversampling] = 1.0f;
    auto y = sine(220.0, kSr, 48000, 0.9f);
    run(q, y, params);
    REQUIRE(maxAbs(tail(y, 24000)) < 1.25);
}

TEST_CASE("Nonlinear folders at high drive stay bounded with rich harmonics", "[dsp][nonlinear]")
{
    for (float type : {4.0f, 5.0f})
    {
        NP p;
        prepared(p);
        Params params = plain();
        params[NP::kOversampling] = 1.0f;
        params[NP::kType] = norm(NP::kType, type);
        params[NP::kDrive] = norm(NP::kDrive, 24.0f);
        params[NP::kCharacter] = 0.5f;
        auto x = sine(200.0, kSr, 48000, 0.8f);
        run(p, x, params);
        const auto out = tail(x, 24000);
        REQUIRE(allFinite(out));
        REQUIRE(maxAbs(out) < 1.3);
        int strong = 0;
        for (int h = 2; h <= 40; ++h)
            if (toneAmplitude(out, 200.0 * h, kSr) > 0.01)
                ++strong;
        REQUIRE(strong >= 8);
    }
}

TEST_CASE("Nonlinear rectifiers produce the expected DC and 2f content", "[dsp][nonlinear]")
{
    // Input 0.2: the soft saturation ahead of the rectifier is nearly linear there.
    const float a = 0.2f;
    for (int type : {6, 7})
    {
        NP p;
        prepared(p);
        p.setDcBlockerEnabled(false);
        Params params = plain();
        params[NP::kType] = norm(NP::kType, static_cast<float>(type));
        auto x = sine(250.0, kSr, 48000, a);
        run(p, x, params);
        const auto out = tail(x, 24000);
        double mean = 0.0;
        for (float v : out)
            mean += v;
        mean /= static_cast<double>(out.size());
        const double f1 = toneAmplitude(out, 250.0, kSr);
        const double f2 = toneAmplitude(out, 500.0, kSr);
        const double full = type == 7 ? 2.0 : 1.0;
        REQUIRE(std::fabs(mean - full * a / kPi) < 0.03 * full * a / kPi);
        REQUIRE(std::fabs(f2 - full * 2.0 * a / (3.0 * kPi)) < 0.05 * full * 2.0 * a / (3.0 * kPi));
        if (type == 6)
            REQUIRE(std::fabs(f1 - a / 2.0) < 0.03 * a / 2.0);
        else
            REQUIRE(f1 < 1e-3);

        // With the blocker on, the DC is gone and 2f stays.
        NP q;
        prepared(q);
        auto y = sine(250.0, kSr, 48000, a);
        run(q, y, params);
        const auto outB = tail(y, 24000);
        double meanB = 0.0;
        for (float v : outB)
            meanB += v;
        REQUIRE(std::fabs(meanB / static_cast<double>(outB.size())) < 1e-3);
        REQUIRE(toneAmplitude(outB, 500.0, kSr) > 0.9 * f2);
    }
}

TEST_CASE("Nonlinear bias DC is removed and does not step the output", "[dsp][nonlinear]")
{
    NP p;
    prepared(p);
    Params params = plain();
    params[NP::kType] = norm(NP::kType, 1.0f);
    params[NP::kBias] = norm(NP::kBias, 0.6f);
    params[NP::kDrive] = norm(NP::kDrive, 12.0f);
    std::vector<float> silence(4800, 0.0f);
    run(p, silence, params);
    REQUIRE(maxAbs(silence) < 1e-6);
    auto x = sine(300.0, kSr, 96000, 0.5f);
    run(p, x, params);
    double mean = 0.0;
    const auto out = tail(x, 48000);
    for (float v : out)
        mean += v;
    REQUIRE(std::fabs(mean / static_cast<double>(out.size())) < 2e-3);
    REQUIRE(toneAmplitude(out, 600.0, kSr) > 0.01); // asymmetry makes even harmonics
}

TEST_CASE("Nonlinear bit reduction to 4 bits gives 16 levels", "[dsp][nonlinear]")
{
    NP p;
    prepared(p);
    Params params = plain();
    params[NP::kType] = norm(NP::kType, 3.0f);
    params[NP::kBits] = norm(NP::kBits, 4.0f);
    auto x = sine(97.0, kSr, 48000, 0.99f);
    run(p, x, params);
    const auto out = tail(x, 24000);
    std::set<float> levels(out.begin(), out.end());
    REQUIRE(levels.size() == 16);
    for (float v : levels)
        REQUIRE(std::fabs(v * 8.0f - std::round(v * 8.0f)) < 1e-6f);
}

TEST_CASE("Nonlinear rate reduction by 4 holds 4 samples", "[dsp][nonlinear]")
{
    NP p;
    prepared(p);
    Params params = plain();
    params[NP::kType] = norm(NP::kType, 3.0f);
    params[NP::kDownsample] = norm(NP::kDownsample, 4.0f);
    REQUIRE(mapParam(NP::params()[NP::kDownsample], params[NP::kDownsample]) == 4.0f);
    auto x = sine(313.0, kSr, 4800, 0.5f);
    run(p, x, params);
    for (std::size_t i = 0; i + 4 <= x.size(); i += 4)
    {
        REQUIRE(x[i + 1] == x[i]);
        REQUIRE(x[i + 2] == x[i]);
        REQUIRE(x[i + 3] == x[i]);
        if (i >= 4)
            REQUIRE(x[i] != x[i - 1]);
    }

    // Fractional factor 2.5 alternates holds of 2 and 3.
    NP q;
    prepared(q);
    params[NP::kDownsample] = norm(NP::kDownsample, 2.5f);
    auto y = sine(313.0, kSr, 4800, 0.5f);
    run(q, y, params);
    int changes = 0;
    for (std::size_t i = 1; i < y.size(); ++i)
        if (y[i] != y[i - 1])
            ++changes;
    REQUIRE(std::abs(changes - 4800 * 2 / 5) <= 3);
}

TEST_CASE("Nonlinear 4x oversampling lowers aliasing of a driven 5 kHz sine", "[dsp][nonlinear]")
{
    // Harmonics of 5 kHz land on multiples of 5 kHz; everything that folds back at 48 kHz
    // lands on the other multiples of 1 kHz.
    const auto aliasEnergy = [](int factor)
    {
        NP p;
        prepared(p);
        Params params = plain();
        params[NP::kType] = norm(NP::kType, 3.0f);
        params[NP::kDrive] = norm(NP::kDrive, 24.0f);
        params[NP::kOversampling] = oversampling(factor);
        auto x = sine(5000.0, kSr, 48000, 0.5f);
        run(p, x, params);
        const auto out = tail(x, 24000);
        double e = 0.0;
        for (int k = 1; k <= 22; ++k)
            if (k % 5 != 0)
            {
                const double a = toneAmplitude(out, 1000.0 * k, kSr);
                e += a * a;
            }
        return e;
    };
    const double e1 = aliasEnergy(1);
    const double e2 = aliasEnergy(2);
    const double e4 = aliasEnergy(4);
    const double db1 = 10.0 * std::log10(e1);
    const double db2 = 10.0 * std::log10(e2);
    const double db4 = 10.0 * std::log10(e4);
    INFO("alias energy 1x " << db1 << " dB, 2x " << db2 << " dB, 4x " << db4 << " dB");
    REQUIRE(db4 < db1 - 15.0);
    REQUIRE(db2 < db1 - 6.0);
}

TEST_CASE("Nonlinear reports the oversampling latency and aligns the dry path", "[dsp][nonlinear]")
{
    for (int factor : {1, 2, 4})
    {
        NP p;
        prepared(p);
        Params params = plain();
        params[NP::kOversampling] = oversampling(factor);
        std::vector<float> x(512, 0.0f);
        x[100] = 0.01f;
        run(p, x, params);
        const int expected = factor == 4 ? 29 : factor == 2 ? 23 : 0;
        REQUIRE(p.latencySamples() == expected);
        const auto peak = std::max_element(x.begin(), x.end(),
                                           [](float a, float b) { return std::fabs(a) < std::fabs(b); });
        REQUIRE(peak - x.begin() == 100 + expected);
        REQUIRE(std::fabs(*peak - 0.01f) < 0.001f);

        // A 50 / 50 mix of the dry and the (linear) wet signal is the input, delayed: no comb.
        NP q;
        prepared(q);
        params[NP::kMix] = 0.5f;
        auto y = sine(9000.0, kSr, 9600, 0.05f);
        run(q, y, params);
        REQUIRE(std::fabs(toneAmplitude(tail(y, 4800), 9000.0, kSr) - 0.05) < 0.001);
    }
}

TEST_CASE("Nonlinear switching oversampling does not click", "[dsp][nonlinear]")
{
    NP p;
    prepared(p);
    Params params = plain();
    auto x = sine(100.0, kSr, 48000, 0.3f);
    std::vector<float> out = x;
    for (std::size_t pos = 0; pos < out.size(); pos += kControlBlock)
    {
        params[NP::kOversampling] = oversampling(1 << ((pos / 3200) % 3));
        p.setParams(params.data(), {});
        float* ch[1] = {out.data() + pos};
        p.process(ch, kControlBlock);
    }
    double maxStep = 0.0;
    for (std::size_t i = 1; i < out.size(); ++i)
        maxStep = std::max(maxStep, static_cast<double>(std::fabs(out[i] - out[i - 1])));
    // A 100 Hz sine at 0.3 moves at most 0.004 per sample; a hard latency switch would jump.
    REQUIRE(maxStep < 0.02);
}

TEST_CASE("Nonlinear silence in gives silence out", "[dsp][nonlinear]")
{
    for (float type = 0.0f; type <= 7.0f; type += 0.5f)
        for (int factor : {1, 2, 4})
        {
            NP p;
            prepared(p);
            Params params = defaults();
            params[NP::kType] = norm(NP::kType, type);
            params[NP::kDrive] = 1.0f;
            params[NP::kBits] = norm(NP::kBits, 6.0f);
            params[NP::kDownsample] = 0.5f;
            params[NP::kTone] = 0.2f;
            params[NP::kOversampling] = oversampling(factor);
            std::vector<float> x(4096, 0.0f);
            run(p, x, params);
            REQUIRE(maxAbs(x) == 0.0);
        }
}

TEST_CASE("Nonlinear recovers from NaN and Inf input and NaN params", "[dsp][nonlinear]")
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (float bad : {nan, inf, -inf})
    {
        NP p;
        prepared(p, kSr, 2);
        Params params = defaults();
        params[NP::kOversampling] = 1.0f;
        params[NP::kDrive] = 0.5f;
        std::vector<float> l = sine(440.0, kSr, 4096, 0.5f), r = l;
        l[1000] = bad;
        for (std::size_t pos = 0; pos < l.size(); pos += kControlBlock)
        {
            p.setParams(params.data(), {});
            float* ch[2] = {l.data() + pos, r.data() + pos};
            p.process(ch, kControlBlock);
        }
        REQUIRE(allFinite(l));
        REQUIRE(allFinite(r));
        REQUIRE(rms(tail(l, 1024)) > 0.05); // recovered
    }

    NP p;
    prepared(p);
    Params params;
    params.fill(nan);
    auto x = sine(440.0, kSr, 4096, 0.5f);
    run(p, x, params);
    REQUIRE(allFinite(x));
    REQUIRE(rms(tail(x, 1024)) > 0.05);
}

TEST_CASE("Nonlinear extreme and random params stay finite and bounded", "[dsp][nonlinear]")
{
    for (float v : {0.0f, 1.0f})
    {
        NP p;
        prepared(p, kSr, 2);
        Params params;
        params.fill(v);
        std::vector<float> l = sine(1000.0, kSr, 9600, 1.0f), r = sine(1500.0, kSr, 9600, 1.0f);
        for (std::size_t pos = 0; pos < l.size(); pos += kControlBlock)
        {
            p.setParams(params.data(), {});
            float* ch[2] = {l.data() + pos, r.data() + pos};
            p.process(ch, kControlBlock);
        }
        REQUIRE(allFinite(l));
        REQUIRE(maxAbs(l) < 8.0);
        REQUIRE(maxAbs(r) < 8.0);
    }

    // Params jumping every block, loud input.
    alfa::dsp::detail::Random rng(1234);
    NP p;
    prepared(p, kSr, 2);
    std::vector<float> l(48000), r(48000);
    for (std::size_t i = 0; i < l.size(); ++i)
    {
        l[i] = 2.0f * rng.nextBipolar();
        r[i] = 2.0f * rng.nextBipolar();
    }
    Params params{};
    for (std::size_t pos = 0; pos < l.size(); pos += kControlBlock)
    {
        for (auto& v : params)
            v = 0.5f + 0.5f * rng.nextBipolar();
        p.setParams(params.data(), {});
        float* ch[2] = {l.data() + pos, r.data() + pos};
        p.process(ch, kControlBlock);
    }
    REQUIRE(allFinite(l));
    REQUIRE(allFinite(r));
    // Worst case: dry at 2 plus trim +12 dB on a wet signal of about 1.
    REQUIRE(maxAbs(l) < 12.0);
    REQUIRE(maxAbs(r) < 12.0);
}

TEST_CASE("Nonlinear works at 44.1, 48 and 96 kHz", "[dsp][nonlinear]")
{
    for (double sr : {44100.0, 48000.0, 96000.0})
    {
        NP p;
        prepared(p, sr);
        Params params = plain();
        params[NP::kOversampling] = 1.0f;
        params[NP::kType] = norm(NP::kType, 3.0f);
        params[NP::kDrive] = norm(NP::kDrive, 20.0f);
        const auto n = static_cast<long long>(sr);
        auto x = sine(500.0, sr, n, 0.5f);
        run(p, x, params);
        const auto out = tail(x, u(n / 2));
        // A square of amplitude 1: fundamental 4/pi, third harmonic 4/(3 pi).
        REQUIRE(std::fabs(toneAmplitude(out, 500.0, sr) - 4.0 / kPi) < 0.08);
        REQUIRE(std::fabs(toneAmplitude(out, 1500.0, sr) - 4.0 / (3.0 * kPi)) < 0.05);
        REQUIRE(toneAmplitude(out, 1000.0, sr) < 0.01);
        REQUIRE(p.tailSamples() > static_cast<int>(0.05 * sr));
    }
}

TEST_CASE("Nonlinear tone tilts the spectrum", "[dsp][nonlinear]")
{
    const auto level = [](float tone, double hz)
    {
        NP p;
        prepared(p);
        Params params = plain();
        params[NP::kTone] = tone;
        auto x = sine(hz, kSr, 24000, 0.05f);
        run(p, x, params);
        return toneAmplitude(tail(x, 12000), hz, kSr) / 0.05;
    };
    REQUIRE(level(1.0f, 10000.0) > 1.8); // ~ +6 dB
    REQUIRE(level(1.0f, 50.0) < 0.55);   // ~ -6 dB
    REQUIRE(level(0.0f, 10000.0) < 0.7); // the one-pole split is gentle
    REQUIRE(level(0.0f, 50.0) > 1.8);
}

TEST_CASE("Nonlinear does not allocate in setParams or process", "[dsp][nonlinear]")
{
    NP p;
    prepared(p, kSr, 2);
    std::vector<float> l = sine(440.0, kSr, 4096, 0.5f), r = l;
    alfa::dsp::detail::Random rng(7);
    Params params{};
    AllocationGuard guard;
    for (std::size_t pos = 0; pos < l.size(); pos += kControlBlock)
    {
        for (auto& v : params)
            v = 0.5f + 0.5f * rng.nextBipolar();
        p.setParams(params.data(), {});
        float* ch[2] = {l.data() + pos, r.data() + pos};
        p.process(ch, kControlBlock);
    }
    p.reset();
    REQUIRE(guard.count() == 0);
}

TEST_CASE("Nonlinear cost, 16 mono voices, 4x, all stages on", "[.bench][dsp][nonlinear]")
{
    constexpr int kVoices = 16;
    constexpr int kBlock = 128;
    constexpr int kBlocks = 2000;
    std::vector<NP> voices(kVoices);
    Params params = defaults();
    params[NP::kOversampling] = 1.0f;
    params[NP::kType] = norm(NP::kType, 4.5f);
    params[NP::kDrive] = 0.5f;
    params[NP::kCharacter] = 0.6f;
    params[NP::kBias] = 0.6f;
    params[NP::kBits] = norm(NP::kBits, 8.0f);
    params[NP::kDownsample] = norm(NP::kDownsample, 2.5f);
    params[NP::kTone] = 0.3f;
    params[NP::kMix] = 0.7f;
    for (auto& v : voices)
        prepared(v);
    auto src = sine(220.0, kSr, kBlock, 0.5f);
    std::vector<float> buf(kBlock);
    float sink = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < kBlocks; ++b)
        for (auto& v : voices)
        {
            buf = src;
            for (int pos = 0; pos < kBlock; pos += kControlBlock)
            {
                v.setParams(params.data(), {});
                float* ch[1] = {buf.data() + pos};
                v.process(ch, kControlBlock);
            }
            sink += buf[5];
        }
    const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("Nonlinear: %.2f us per 128-sample block for 16 mono voices at 4x (sink %g)\n", us / kBlocks,
                static_cast<double>(sink));
    REQUIRE(std::isfinite(sink));
}
