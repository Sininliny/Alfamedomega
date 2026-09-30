#include "TestHelpers.h"
#include "alfa/dsp/Filters.h"
#include "alfa/dsp/detail/FastMath.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{

constexpr double kPiD = 3.14159265358979323846;
using Params = std::array<float, kSlotParams>;

inline std::size_t u(int i)
{
    return static_cast<std::size_t>(i);
}

template <typename P>
Params defaults()
{
    Params p{};
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = P::params()[i].defaultNorm;
    return p;
}

/// Inverse of a Log param mapping.
float logNorm(float value, float min, float max)
{
    return static_cast<float>(std::log(static_cast<double>(value) / min) /
                              std::log(static_cast<double>(max) / min));
}

template <typename P>
void prepare(P& proc, double sr, int channels = 1)
{
    ProcessSpec spec;
    spec.sampleRate = sr;
    spec.channels = channels;
    proc.prepare(spec);
    proc.reset();
}

/// Runs x through proc (mono) in control blocks, calling setParams before each block.
template <typename P>
std::vector<float> run(P& proc, const Params& p, std::vector<float> x, float f0 = 0.0f)
{
    ControlContext ctx;
    ctx.f0 = f0;
    for (std::size_t pos = 0; pos < x.size(); pos += kControlBlock)
    {
        const int n = static_cast<int>(std::min<std::size_t>(kControlBlock, x.size() - pos));
        proc.setParams(p.data(), ctx);
        float* ch[1] = {x.data() + pos};
        proc.process(ch, n);
    }
    return x;
}

std::vector<float> sine(double hz, double sr, std::size_t n, float amp = 0.1f)
{
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = amp * static_cast<float>(std::sin(2.0 * kPiD * hz * static_cast<double>(i) / sr));
    return x;
}

std::vector<float> noise(std::size_t n, float amp = 1.0f, std::uint32_t seed = 1)
{
    alfa::dsp::detail::Random r(seed);
    std::vector<float> x(n);
    for (auto& v : x)
        v = amp * r.nextBipolar();
    return x;
}

/// Pink noise (Paul Kellet's economy filter), roughly the spectral balance of music.
std::vector<float> pinkNoise(std::size_t n, float amp, std::uint32_t seed)
{
    auto w = noise(n, 1.0f, seed);
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
    for (auto& v : w)
    {
        b0 = 0.99765f * b0 + v * 0.0990460f;
        b1 = 0.96300f * b1 + v * 0.2965164f;
        b2 = 0.57000f * b2 + v * 1.0526913f;
        v = amp * 0.25f * (b0 + b1 + b2 + v * 0.1848f);
    }
    return w;
}

std::vector<float> impulse(std::size_t n)
{
    std::vector<float> x(n, 0.0f);
    x[0] = 1.0f;
    return x;
}

/// Steady-state gain (dB) of a linear processor for a sine at hz.
template <typename P>
double gainDb(P& proc, const Params& p, double hz, double sr, float f0 = 0.0f)
{
    proc.reset();
    const auto n = static_cast<std::size_t>(sr * 0.75);
    const auto y = run(proc, p, sine(hz, sr, n), f0);
    const std::vector<float> tail(y.end() - 16384, y.end());
    return 20.0 * std::log10(toneAmplitude(tail, hz, sr) / 0.1);
}

/// |DTFT| of x at hz.
double dtft(const std::vector<float>& x, double hz, double sr)
{
    const std::complex<double> rot = std::polar(1.0, -2.0 * kPiD * hz / sr);
    std::complex<double> z = 1.0, acc{};
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        acc += static_cast<double>(x[i]) * z;
        z *= rot;
        if ((i & 1023u) == 0)
            z /= std::abs(z);
    }
    return std::abs(acc);
}

/// Frequency of the |DTFT| maximum in [lo, hi] (golden-section search; assumes one peak).
double peakHz(const std::vector<float>& x, double lo, double hi, double sr)
{
    const double gr = 0.6180339887498949;
    double a = lo, b = hi;
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = dtft(x, c, sr), fd = dtft(x, d, sr);
    while (b - a > 1e-4)
    {
        if (fc > fd)
        {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = dtft(x, c, sr);
        }
        else
        {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = dtft(x, d, sr);
        }
    }
    return 0.5 * (a + b);
}

/// Grid search for the |DTFT| maximum in [lo, hi].
double gridPeakHz(const std::vector<float>& x, double lo, double hi, double sr, int steps = 200)
{
    double best = lo, bestMag = -1.0;
    for (int i = 0; i <= steps; ++i)
    {
        const double f = lo * std::pow(hi / lo, i / static_cast<double>(steps));
        const double m = dtft(x, f, sr);
        if (m > bestMag)
        {
            bestMag = m;
            best = f;
        }
    }
    return best;
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

double cents(double hz, double ref)
{
    return 1200.0 * std::log2(hz / ref);
}

// Parameter index helpers for readability.
namespace fp
{
constexpr std::size_t cutoff = 0, res = 1, mode = 2, slope = 3, drive = 4, key = 5, type = 6, gain = 7,
                      mix = 8;
}
namespace cp
{
constexpr std::size_t freq = 0, key = 1, fb = 2, damp = 3, balance = 4;
}

Params filterParams(float cutoffHz, float res = 0.0f, float mode = 0.0f, bool slope24 = false)
{
    Params p = defaults<FilterProcessor>();
    p[fp::cutoff] = logNorm(cutoffHz, 20.0f, 20000.0f);
    p[fp::res] = res;
    p[fp::mode] = mode;
    p[fp::slope] = slope24 ? 1.0f : 0.0f;
    return p;
}

Params combParams(float hz, float fb, float damping = 0.0f, float balance = 1.0f)
{
    Params p = defaults<CombProcessor>();
    p[cp::freq] = logNorm(hz, 20.0f, 5000.0f);
    p[cp::key] = 0.0f;
    p[cp::fb] = (fb / 0.99f + 1.0f) * 0.5f;
    p[cp::damp] = damping;
    p[cp::balance] = balance;
    return p;
}

template <typename P>
const char* nameOf()
{
    if constexpr (std::is_same_v<P, FilterProcessor>)
        return "FILTER (24 dB, drive)";
    else if constexpr (std::is_same_v<P, CombProcessor>)
        return "COMB";
    else if constexpr (std::is_same_v<P, FormantProcessor>)
        return "FORMANT";
    else
        return "BODY";
}

/// Params with every feature engaged, for the generic tests and the benchmark.
template <typename P>
Params busyParams()
{
    Params p = defaults<P>();
    for (std::size_t i = 0; i < p.size(); ++i)
        if (!P::params()[i].name.empty())
            p[i] = 0.3f + 0.05f * static_cast<float>(i);
    return p;
}

} // namespace

// Generic contract tests (all four families) -----------------------------------------------

#define ALFA_FILTER_TYPES FilterProcessor, CombProcessor, FormantProcessor, BodyProcessor

TEMPLATE_TEST_CASE("filters: param tables are well formed", "[dsp][filters]", ALFA_FILTER_TYPES)
{
    const ParamTable& t = TestType::params();
    bool ended = false;
    int used = 0;
    for (const ParamInfo& info : t)
    {
        if (info.name.empty())
        {
            ended = true;
            continue;
        }
        REQUIRE_FALSE(ended); // names are contiguous from 0
        ++used;
        REQUIRE(info.defaultNorm >= 0.0f);
        REQUIRE(info.defaultNorm <= 1.0f);
        REQUIRE(info.min < info.max);
        if (info.curve == Curve::Log)
            REQUIRE(info.min > 0.0f);
        if (info.curve == Curve::Choice)
            REQUIRE(info.min == 0.0f);
    }
    REQUIRE(used >= 4);
    REQUIRE(allowedInVoice(TestType::kFamily));
}

TEMPLATE_TEST_CASE("filters: silence in gives silence out", "[dsp][filters]", ALFA_FILTER_TYPES)
{
    for (const Params& p : {defaults<TestType>(), busyParams<TestType>()})
    {
        TestType proc;
        prepare(proc, 48000.0);
        const auto y = run(proc, p, std::vector<float>(4800, 0.0f), 220.0f);
        REQUIRE(peakAbs(y) == 0.0f);
    }
}

TEMPLATE_TEST_CASE("filters: NaN and Inf input give finite output and recover", "[dsp][filters]",
                   ALFA_FILTER_TYPES)
{
    TestType proc;
    prepare(proc, 48000.0);
    const Params p = busyParams<TestType>();
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(), 1e30f})
    {
        auto x = sine(300.0, 48000.0, 9600, 0.5f);
        x[1000] = bad;
        x[1001] = bad;
        const auto y = run(proc, p, x, 220.0f);
        REQUIRE(allFinite(y));
        // Recovers: the last part still carries signal.
        REQUIRE(rms(y.data() + 7200, 2400) > 1e-4);
    }
}

TEMPLATE_TEST_CASE("filters: NaN params give finite output", "[dsp][filters]", ALFA_FILTER_TYPES)
{
    TestType proc;
    prepare(proc, 48000.0);
    Params p{};
    p.fill(std::numeric_limits<float>::quiet_NaN());
    auto y = run(proc, p, noise(4800, 0.5f), std::numeric_limits<float>::quiet_NaN());
    REQUIRE(allFinite(y));
    p.fill(std::numeric_limits<float>::infinity());
    y = run(proc, p, noise(4800, 0.5f), std::numeric_limits<float>::infinity());
    REQUIRE(allFinite(y));
}

TEMPLATE_TEST_CASE("filters: extreme and randomly jumping params stay bounded", "[dsp][filters]",
                   ALFA_FILTER_TYPES)
{
    for (double sr : {44100.0, 48000.0, 96000.0})
    {
        TestType proc;
        prepare(proc, sr);
        Params zeros{}, ones{};
        ones.fill(1.0f);
        for (const Params& p : {zeros, ones})
        {
            proc.reset();
            const auto y = run(proc, p, noise(u(static_cast<int>(sr / 2))), 440.0f);
            REQUIRE(allFinite(y));
            REQUIRE(peakAbs(y) < 200.0f);
        }
        // Every param jumps to a random value every control block.
        alfa::dsp::detail::Random r(7);
        auto x = noise(u(static_cast<int>(sr) / kControlBlock * kControlBlock));
        float worst = 0.0f;
        for (std::size_t pos = 0; pos < x.size(); pos += kControlBlock)
        {
            Params p{};
            for (auto& v : p)
                v = 0.5f + 0.5f * r.nextBipolar();
            ControlContext ctx;
            ctx.f0 = 30.0f + 3000.0f * (0.5f + 0.5f * r.nextBipolar());
            proc.setParams(p.data(), ctx);
            float* ch[1] = {x.data() + pos};
            proc.process(ch, kControlBlock);
            for (int i = 0; i < kControlBlock; ++i)
                worst = std::max(worst, std::fabs(x[pos + u(i)]));
        }
        REQUIRE(allFinite(x));
        REQUIRE(worst < 200.0f);
    }
}

TEMPLATE_TEST_CASE("filters: runs at 44.1, 48 and 96 kHz", "[dsp][filters]", ALFA_FILTER_TYPES)
{
    for (double sr : {44100.0, 48000.0, 96000.0})
    {
        TestType proc;
        prepare(proc, sr);
        const auto y =
            run(proc, busyParams<TestType>(), sine(220.0, sr, u(static_cast<int>(sr / 2)), 0.5f), 220.0f);
        REQUIRE(allFinite(y));
        REQUIRE(rms(y) > 1e-3);
        REQUIRE(proc.tailSamples() > 0);
    }
}

TEMPLATE_TEST_CASE("filters: stereo channels have independent state", "[dsp][filters]", ALFA_FILTER_TYPES)
{
    TestType proc;
    prepare(proc, 48000.0, 2);
    const Params p = busyParams<TestType>();
    auto l = noise(4800, 0.5f, 3);
    std::vector<float> r(l.size(), 0.0f);
    for (std::size_t pos = 0; pos < l.size(); pos += kControlBlock)
    {
        proc.setParams(p.data(), {});
        float* ch[2] = {l.data() + pos, r.data() + pos};
        proc.process(ch, kControlBlock);
    }
    REQUIRE(rms(l) > 1e-3);
    REQUIRE(peakAbs(r) == 0.0f);

    TestType mono;
    prepare(mono, 48000.0);
    const auto m = run(mono, p, noise(4800, 0.5f, 3));
    for (std::size_t i = 0; i < m.size(); ++i)
        REQUIRE(m[i] == l[i]);
}

TEMPLATE_TEST_CASE("filters: no allocation in setParams and process", "[dsp][filters]", ALFA_FILTER_TYPES)
{
    TestType proc;
    prepare(proc, 48000.0, 2);
    std::vector<float> l = noise(512), r = noise(512, 1.0f, 9);
    Params p = busyParams<TestType>();
    AllocationGuard guard;
    for (int pos = 0; pos < 512; pos += kControlBlock)
    {
        p[0] = static_cast<float>(pos) / 512.0f;
        ControlContext ctx;
        ctx.f0 = 110.0f;
        proc.setParams(p.data(), ctx);
        float* ch[2] = {l.data() + pos, r.data() + pos};
        proc.process(ch, kControlBlock);
    }
    proc.reset();
    REQUIRE(guard.count() == 0);
}

TEMPLATE_TEST_CASE("filters: cost, 16 mono voices at 48 kHz", "[.bench][dsp][filters]", ALFA_FILTER_TYPES)
{
    constexpr int kVoices = 16;
    constexpr int kBlocks = 48000 / kControlBlock; // 1 s
    std::vector<TestType> voices(kVoices);
    Params p = busyParams<TestType>();
    if constexpr (std::is_same_v<TestType, FilterProcessor>)
    {
        p[fp::slope] = 1.0f;
        p[fp::drive] = 0.5f;
        p[fp::res] = 0.8f;
    }
    for (auto& v : voices)
        prepare(v, 48000.0);
    const auto input = noise(kControlBlock);
    std::vector<float> x(input);
    float sink = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < kBlocks; ++b)
        for (int v = 0; v < kVoices; ++v)
        {
            p[0] = 0.4f + 0.2f * static_cast<float>((b + v) % 16) / 16.0f; // moving first param
            ControlContext ctx;
            ctx.f0 = 110.0f + static_cast<float>(v);
            voices[u(v)].setParams(p.data(), ctx);
            float* ch[1] = {x.data()};
            voices[u(v)].process(ch, kControlBlock);
            sink += x[0];
            x = input;
        }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    const double perBlock = us / (kBlocks * kControlBlock / 128.0);
    std::printf("%s: %.2f us per 128-sample block for %d voices, %.3f us per voice (sink %g)\n",
                nameOf<TestType>(), perBlock, kVoices, perBlock / kVoices, static_cast<double>(sink));
}

// FILTER --------------------------------------------------------------------------------------

TEST_CASE("FILTER 12 dB low-pass is -3 dB at the cutoff", "[dsp][filters]")
{
    for (double sr : {44100.0, 48000.0, 96000.0})
    {
        FilterProcessor f;
        prepare(f, sr);
        const Params p = filterParams(1000.0f);
        INFO("sr " << sr);
        REQUIRE(std::abs(gainDb(f, p, 1000.0, sr) + 3.01) < 0.25);
        REQUIRE(std::abs(gainDb(f, p, 100.0, sr)) < 0.2);
        // 12 dB/oct: about -40 dB at 10x the cutoff (bilinear warping makes it steeper).
        REQUIRE(gainDb(f, p, 10000.0, sr) < -38.0);
    }
}

TEST_CASE("FILTER 24 dB low-pass is -3 dB at the cutoff and twice as steep", "[dsp][filters]")
{
    FilterProcessor f;
    prepare(f, 48000.0);
    const Params p = filterParams(1000.0f, 0.0f, 0.0f, true);
    REQUIRE(std::abs(gainDb(f, p, 1000.0, 48000.0) + 3.01) < 0.25);
    REQUIRE(std::abs(gainDb(f, p, 100.0, 48000.0)) < 0.2);
    REQUIRE(gainDb(f, p, 4000.0, 48000.0) < -47.0);
}

TEST_CASE("FILTER high-pass, band-pass and notch responses", "[dsp][filters]")
{
    FilterProcessor f;
    prepare(f, 48000.0);
    SECTION("high-pass")
    {
        const Params p = filterParams(1000.0f, 0.0f, 0.5f);
        REQUIRE(std::abs(gainDb(f, p, 1000.0, 48000.0) + 3.01) < 0.25);
        REQUIRE(gainDb(f, p, 100.0, 48000.0) < -38.0);
        REQUIRE(std::abs(gainDb(f, p, 10000.0, 48000.0)) < 0.2);
    }
    SECTION("band-pass")
    {
        const Params p = filterParams(1000.0f, 0.0f, 0.25f);
        REQUIRE(std::abs(gainDb(f, p, 1000.0, 48000.0)) < 0.2);
        REQUIRE(gainDb(f, p, 100.0, 48000.0) < -16.0);
        REQUIRE(gainDb(f, p, 10000.0, 48000.0) < -16.0);
    }
    SECTION("notch")
    {
        const Params p = filterParams(1000.0f, 0.5f, 0.75f);
        REQUIRE(gainDb(f, p, 1000.0, 48000.0) < -30.0);
        REQUIRE(std::abs(gainDb(f, p, 100.0, 48000.0)) < 0.5);
        REQUIRE(std::abs(gainDb(f, p, 10000.0, 48000.0)) < 0.5);
    }
    SECTION("resonance raises the peak at the cutoff")
    {
        const Params p = filterParams(1000.0f, 0.9f);
        REQUIRE(gainDb(f, p, 1000.0, 48000.0) > 15.0);
    }
    SECTION("mix 0 is dry")
    {
        Params p = filterParams(200.0f, 0.7f);
        p[fp::mix] = 0.0f;
        const auto x = noise(4800);
        f.reset();
        REQUIRE(run(f, p, x) == x);
    }
}

TEST_CASE("FILTER bell and shelves", "[dsp][filters]")
{
    FilterProcessor f;
    prepare(f, 48000.0);
    Params p = filterParams(1000.0f, 0.3f);
    p[fp::gain] = 0.75f; // +12 dB
    SECTION("bell")
    {
        p[fp::type] = 1.0f / 3.0f;
        REQUIRE(std::abs(gainDb(f, p, 1000.0, 48000.0) - 12.0) < 0.3);
        REQUIRE(std::abs(gainDb(f, p, 50.0, 48000.0)) < 0.5);
        REQUIRE(std::abs(gainDb(f, p, 18000.0, 48000.0)) < 0.5);
    }
    SECTION("low shelf")
    {
        p[fp::type] = 2.0f / 3.0f;
        REQUIRE(std::abs(gainDb(f, p, 30.0, 48000.0) - 12.0) < 0.3);
        REQUIRE(std::abs(gainDb(f, p, 1000.0, 48000.0) - 6.0) < 0.5);
        REQUIRE(std::abs(gainDb(f, p, 15000.0, 48000.0)) < 0.3);
    }
    SECTION("high shelf")
    {
        p[fp::type] = 1.0f;
        REQUIRE(std::abs(gainDb(f, p, 15000.0, 48000.0) - 12.0) < 0.4);
        REQUIRE(std::abs(gainDb(f, p, 1000.0, 48000.0) - 6.0) < 0.5);
        REQUIRE(std::abs(gainDb(f, p, 30.0, 48000.0)) < 0.3);
    }
}

TEST_CASE("FILTER key tracking scales the cutoff by f0 / middle C", "[dsp][filters]")
{
    FilterProcessor f;
    prepare(f, 48000.0);
    Params p = filterParams(1000.0f);
    p[fp::key] = 1.0f;
    // f0 one octave above middle C doubles the cutoff to 2 kHz.
    REQUIRE(std::abs(gainDb(f, p, 2000.0, 48000.0, 523.2511f) + 3.01) < 0.25);
    // No pitch (instance slot): the absolute cutoff.
    REQUIRE(std::abs(gainDb(f, p, 1000.0, 48000.0, 0.0f) + 3.01) < 0.25);
}

TEST_CASE("FILTER at full resonance rings but stays bounded", "[dsp][filters]")
{
    for (bool slope24 : {false, true})
        for (double cutoff : {20.0, 440.0, 5000.0, 20000.0})
        {
            FilterProcessor f;
            prepare(f, 48000.0);
            INFO("slope24 " << slope24 << " cutoff " << cutoff);
            const Params p = filterParams(static_cast<float>(cutoff), 1.0f, 0.0f, slope24);
            // Impulse: a long ring that decays.
            const auto ir = run(f, p, impulse(96000));
            REQUIRE(allFinite(ir));
            REQUIRE(peakAbs(ir) < 10.0f);
            // Sustained full-scale sine at the resonance: nonlinear damping keeps it a few units.
            f.reset();
            const auto y = run(f, p, sine(std::min(cutoff, 20000.0), 48000.0, 48000, 1.0f));
            REQUIRE(allFinite(y));
            REQUIRE(peakAbs(y) < 20.0f);
            // Loud noise.
            f.reset();
            const auto z = run(f, p, noise(48000));
            REQUIRE(allFinite(z));
            REQUIRE(peakAbs(z) < 20.0f);
        }
}

TEST_CASE("FILTER cutoff swept every block from 20 Hz to 20 kHz at max resonance stays finite",
          "[dsp][filters]")
{
    for (double sr : {44100.0, 48000.0, 96000.0})
        for (bool slope24 : {false, true})
            for (float drive : {0.0f, 1.0f})
            {
                FilterProcessor f;
                prepare(f, sr);
                Params p = filterParams(20.0f, 1.0f, 0.0f, slope24);
                p[fp::drive] = drive;
                auto x = noise(u(static_cast<int>(sr) / kControlBlock * kControlBlock));
                const std::size_t blocks = x.size() / kControlBlock;
                float worst = 0.0f;
                for (std::size_t b = 0; b < blocks; ++b)
                {
                    // Up and down the full range 8 times, plus a mode sweep.
                    const double ph = static_cast<double>(b) / static_cast<double>(blocks) * 8.0;
                    p[fp::cutoff] = static_cast<float>(1.0 - std::fabs(2.0 * (ph - std::floor(ph)) - 1.0));
                    p[fp::mode] = static_cast<float>(ph / 8.0);
                    f.setParams(p.data(), {});
                    float* ch[1] = {x.data() + b * kControlBlock};
                    f.process(ch, kControlBlock);
                    for (int i = 0; i < kControlBlock; ++i)
                        worst = std::max(worst, std::fabs(x[b * kControlBlock + u(i)]));
                }
                INFO("sr " << sr << " slope24 " << slope24 << " drive " << drive);
                REQUIRE(allFinite(x));
                REQUIRE(worst < 50.0f);
                // Jumping between the extremes every block.
                for (std::size_t b = 0; b < blocks; ++b)
                {
                    p[fp::cutoff] = (b & 1u) != 0 ? 1.0f : 0.0f;
                    f.setParams(p.data(), {});
                    float* ch[1] = {x.data() + b * kControlBlock};
                    f.process(ch, kControlBlock);
                }
                REQUIRE(allFinite(x));
                REQUIRE(peakAbs(x) < 50.0f);
            }
}

TEST_CASE("FILTER drive saturates", "[dsp][filters]")
{
    FilterProcessor f;
    prepare(f, 48000.0);
    Params p = filterParams(20000.0f);
    p[fp::drive] = 1.0f;
    const auto y = run(f, p, sine(200.0, 48000.0, 24000, 1.0f));
    REQUIRE(peakAbs(y) < 0.2f); // tanh limits to 1, then -18 dB makeup
    const std::vector<float> tail(y.end() - 8192, y.end());
    // Odd harmonics appear.
    REQUIRE(toneAmplitude(tail, 600.0, 48000.0) > 0.1 * toneAmplitude(tail, 200.0, 48000.0));
}

TEST_CASE("FILTER tail follows resonance", "[dsp][filters]")
{
    FilterProcessor f;
    prepare(f, 48000.0);
    f.setParams(filterParams(1000.0f).data(), {});
    const int calm = f.tailSamples();
    f.setParams(filterParams(1000.0f, 1.0f).data(), {});
    REQUIRE(f.tailSamples() > 10 * calm);
    // The ring after an impulse is below -60 dB after the reported tail.
    f.reset();
    const Params p = filterParams(1000.0f, 0.95f);
    f.setParams(p.data(), {});
    const int tail = f.tailSamples();
    const auto ir = run(f, p, impulse(u(tail + 4800)));
    REQUIRE(peakAbs(std::vector<float>(ir.begin() + tail, ir.end())) < 1e-3f * peakAbs(ir));
}

// COMB ----------------------------------------------------------------------------------------

TEST_CASE("COMB feedback peak is within 1 cent of the comb frequency", "[dsp][filters]")
{
    for (double sr : {44100.0, 48000.0, 96000.0})
        for (double hz : {55.0, 440.0, 1234.5, 4000.0})
            for (float damping : {0.0f, 0.4f})
            {
                // Heavy loop damping near the comb frequency tilts (and so skews) the peak.
                if (damping > 0.0f && hz > 2000.0)
                    continue;
                CombProcessor comb;
                prepare(comb, sr);
                const auto ir =
                    run(comb, combParams(static_cast<float>(hz), 0.99f, damping), impulse(1u << 17));
                const double lo = hz * std::exp2(-40.0 / 1200.0), hi = hz * std::exp2(40.0 / 1200.0);
                const double peak = peakHz(ir, lo, hi, sr);
                INFO("sr " << sr << " hz " << hz << " damping " << damping << " peak " << peak);
                REQUIRE(std::abs(cents(peak, hz)) < 1.0);
            }
}

TEST_CASE("COMB key tracking tunes to f0", "[dsp][filters]")
{
    CombProcessor comb;
    prepare(comb, 48000.0);
    Params p = combParams(261.6256f, 0.99f);
    p[cp::key] = 1.0f;
    const auto ir = run(comb, p, impulse(1u << 17), 440.0f);
    REQUIRE(std::abs(cents(peakHz(ir, 430.0, 450.0, 48000.0), 440.0)) < 1.0);
}

TEST_CASE("COMB feedforward notches half-way between harmonics", "[dsp][filters]")
{
    CombProcessor comb;
    prepare(comb, 48000.0);
    const Params p = combParams(500.0f, 0.99f, 0.0f, 0.0f);
    REQUIRE(gainDb(comb, p, 250.0, 48000.0) < -30.0);
    REQUIRE(gainDb(comb, p, 750.0, 48000.0) < -30.0);
    REQUIRE(gainDb(comb, p, 500.0, 48000.0) > 2.5);
}

TEST_CASE("COMB is stable at maximum feedback under extreme modulation", "[dsp][filters]")
{
    for (double sr : {44100.0, 96000.0})
        for (float fb : {-0.99f, 0.99f})
        {
            CombProcessor comb;
            prepare(comb, sr);
            auto x = noise(u(static_cast<int>(sr) / kControlBlock * kControlBlock));
            for (std::size_t b = 0; b * kControlBlock < x.size(); ++b)
            {
                Params p = combParams((b & 1u) != 0 ? 20.0f : 5000.0f, fb, static_cast<float>(b % 3) * 0.5f);
                ControlContext ctx;
                p[cp::key] = 1.0f;
                ctx.f0 = (b & 2u) != 0 ? 8000.0f : 20.0f; // tracked far outside the range
                comb.setParams(p.data(), ctx);
                float* ch[1] = {x.data() + b * kControlBlock};
                comb.process(ch, kControlBlock);
            }
            REQUIRE(allFinite(x));
            REQUIRE(peakAbs(x) < 50.0f);
        }
}

TEST_CASE("COMB rings out within its reported tail", "[dsp][filters]")
{
    CombProcessor comb;
    prepare(comb, 48000.0);
    const Params p = combParams(100.0f, 0.95f);
    comb.setParams(p.data(), {});
    const int tail = comb.tailSamples();
    REQUIRE(tail > 480 * 100); // 100 Hz, 0.95: about 135 periods
    const auto ir = run(comb, p, impulse(u(tail + 4800)));
    REQUIRE(peakAbs(std::vector<float>(ir.begin() + tail, ir.end())) < 2e-3f * peakAbs(ir));
}

// FORMANT -------------------------------------------------------------------------------------

TEST_CASE("FORMANT peaks sit at the table frequencies", "[dsp][filters]")
{
    struct Case
    {
        float vowel, voice;
        std::array<double, 2> hz; // F1, F2
    };
    const Case cases[] = {
        {0.0f, 0.0f, {600.0, 1040.0}},  // male A
        {0.5f, 0.0f, {250.0, 1750.0}},  // male I
        {0.5f, 1.0f, {270.0, 2140.0}},  // female I
        {1.0f, 1.0f, {325.0, 700.0}},   // female U
        {0.25f, 1.0f, {350.0, 2000.0}}, // female E
    };
    for (double sr : {44100.0, 48000.0, 96000.0})
        for (const Case& c : cases)
        {
            FormantProcessor fmt;
            prepare(fmt, sr);
            Params p = defaults<FormantProcessor>();
            p[0] = c.vowel;
            p[1] = c.voice;
            const auto ir = run(fmt, p, impulse(1u << 15));
            for (std::size_t i = 0; i < 2; ++i)
            {
                REQUIRE(std::abs(fmt.formantHz(static_cast<int>(i)) - c.hz[i]) < 0.01 * c.hz[i]);
                const double peak = gridPeakHz(ir, c.hz[i] * 0.85, c.hz[i] * 1.15, sr);
                INFO("sr " << sr << " vowel " << c.vowel << " voice " << c.voice << " F" << i + 1 << " "
                           << c.hz[i] << " peak " << peak);
                REQUIRE(std::abs(peak / c.hz[i] - 1.0) < 0.04);
            }
        }
}

TEST_CASE("FORMANT shift moves the formants", "[dsp][filters]")
{
    FormantProcessor fmt;
    prepare(fmt, 48000.0);
    Params p = defaults<FormantProcessor>();
    p[2] = 1.0f; // +12 st
    const auto ir = run(fmt, p, impulse(1u << 15));
    REQUIRE(std::abs(fmt.formantHz(0) - 1200.0f) < 1.0f);
    REQUIRE(std::abs(gridPeakHz(ir, 1000.0, 1400.0, 48000.0) / 1200.0 - 1.0) < 0.04);
    // First formant peak is unity gain.
    REQUIRE(std::abs(gainDb(fmt, p, 1200.0, 48000.0)) < 1.5);
}

TEST_CASE("FORMANT vowel morph is smooth", "[dsp][filters]")
{
    FormantProcessor fmt;
    prepare(fmt, 48000.0);
    Params p = defaults<FormantProcessor>();
    float prev = 600.0f;
    for (int i = 1; i <= 100; ++i)
    {
        p[0] = 0.25f * static_cast<float>(i) / 100.0f; // A to E
        fmt.setParams(p.data(), {});
        const float f1 = fmt.formantHz(0);
        REQUIRE(std::abs(f1 - prev) < 5.0f);
        prev = f1;
    }
    REQUIRE(std::abs(prev - 400.0f) < 0.5f);
}

// BODY ----------------------------------------------------------------------------------------

TEST_CASE("BODY resonates at its mode frequencies and scales with size", "[dsp][filters]")
{
    for (double sr : {44100.0, 48000.0, 96000.0})
    {
        BodyProcessor body;
        prepare(body, sr);
        Params p = defaults<BodyProcessor>(); // guitar, size 1
        // Guitar top-plate mode at 204 Hz stands out over the valley at 300 Hz.
        REQUIRE(gainDb(body, p, 204.0, sr) - gainDb(body, p, 300.0, sr) > 6.0);
        const auto ir = run(body, p, impulse(1u << 15));
        REQUIRE(std::abs(gridPeakHz(ir, 180.0, 240.0, sr) / 204.0 - 1.0) < 0.03);
        // Size x2 halves the mode frequencies.
        p[1] = 1.0f;
        body.reset();
        const auto big = run(body, p, impulse(1u << 15));
        REQUIRE(std::abs(gridPeakHz(big, 90.0, 120.0, sr) / 102.0 - 1.0) < 0.03);
    }
}

TEST_CASE("BODY types keep roughly unity broadband gain", "[dsp][filters]")
{
    for (int type = 0; type < 4; ++type)
    {
        BodyProcessor body;
        prepare(body, 48000.0);
        Params p = defaults<BodyProcessor>();
        p[0] = static_cast<float>(type) / 3.0f;
        const auto x = pinkNoise(96000, 0.3f, 11);
        const auto y = run(body, p, x);
        const double db = 20.0 * std::log10(rms(y) / rms(x));
        INFO("type " << type << " gain " << db << " dB");
        REQUIRE(std::abs(db) < 2.0);
    }
}

TEST_CASE("BODY damping shortens the ring", "[dsp][filters]")
{
    BodyProcessor body;
    prepare(body, 48000.0);
    Params p = defaults<BodyProcessor>();
    p[3] = 0.0f;
    const auto live = run(body, p, impulse(48000));
    body.reset();
    p[3] = 1.0f;
    const auto dead = run(body, p, impulse(48000));
    REQUIRE(rms(live.data() + 4800, 9600) > 3.0 * rms(dead.data() + 4800, 9600));
}
