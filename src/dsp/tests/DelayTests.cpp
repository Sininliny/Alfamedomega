#include "TestHelpers.h"
#include "alfa/dsp/Delay.h"
#include "alfa/dsp/detail/DelayAllpass.h"
#include "alfa/dsp/detail/DelayFdn.h"
#include "alfa/dsp/detail/DelayLine.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;
namespace dd = alfa::dsp::detail;

namespace
{
constexpr double kPi = 3.14159265358979323846;
using Params = std::array<float, kSlotParams>;
using Buffers = std::vector<std::vector<float>>;

std::size_t u(long long i)
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

/// Inverse of mapParam for building test patches from physical values.
float toNorm(const ParamInfo& info, float v)
{
    switch (info.curve)
    {
    case Curve::Linear:
        return (v - info.min) / (info.max - info.min);
    case Curve::Log:
        return std::log(v / info.min) / std::log(info.max / info.min);
    case Curve::Choice:
        return v / info.max;
    case Curve::Toggle:
        return v >= 0.5f ? 1.0f : 0.0f;
    }
    return 0.0f;
}

template <typename P>
void set(Params& p, int index, float physical)
{
    p[u(index)] = toNorm(P::params()[u(index)], physical);
}

template <typename P>
std::unique_ptr<P> make(double sr, int channels)
{
    auto p = std::make_unique<P>();
    ProcessSpec spec;
    spec.sampleRate = sr;
    spec.channels = channels;
    p->prepare(spec);
    p->reset();
    return p;
}

struct NoChange
{
    void operator()(int, Params&) const {}
};

/// Processes io in place in control blocks, calling perBlock(blockIndex, params) before each
/// block's setParams.
template <typename P, typename F = NoChange>
void run(P& proc, Params& prm, Buffers& io, F&& perBlock = {}, int block = kControlBlock)
{
    const auto n = static_cast<int>(io[0].size());
    std::array<float*, 2> ptr{};
    for (int start = 0, b = 0; start < n; start += block, ++b)
    {
        perBlock(b, prm);
        proc.setParams(prm.data(), {});
        for (std::size_t c = 0; c < io.size(); ++c)
            ptr[c] = io[c].data() + start;
        proc.process(ptr.data(), std::min(block, n - start));
    }
}

Buffers silence(int channels, long long n)
{
    return Buffers(u(channels), std::vector<float>(u(n), 0.0f));
}

Buffers noise(int channels, long long n, float amp = 0.5f, std::uint32_t seed = 1)
{
    dd::Random r(seed);
    Buffers b = silence(channels, n);
    for (auto& ch : b)
        for (auto& x : ch)
            x = amp * r.nextBipolar();
    return b;
}

Buffers sine(int channels, long long n, double hz, double sr, float amp = 0.5f)
{
    Buffers b = silence(channels, n);
    for (auto& ch : b)
        for (long long i = 0; i < n; ++i)
            ch[u(i)] = amp * static_cast<float>(std::sin(2.0 * kPi * hz * static_cast<double>(i) / sr));
    return b;
}

bool finiteAndBounded(const Buffers& b, float bound)
{
    for (const auto& ch : b)
        for (float x : ch)
            if (!std::isfinite(x) || std::fabs(x) > bound)
                return false;
    return true;
}

float peak(const std::vector<float>& x, std::size_t from = 0,
           std::size_t to = std::numeric_limits<std::size_t>::max())
{
    float m = 0.0f;
    for (std::size_t i = from; i < std::min(to, x.size()); ++i)
        m = std::max(m, std::fabs(x[i]));
    return m;
}

long long argmaxAbs(const std::vector<float>& x)
{
    return static_cast<long long>(
        std::max_element(x.begin(), x.end(), [](float a, float b) { return std::fabs(a) < std::fabs(b); }) -
        x.begin());
}

double windowRms(const std::vector<float>& x, double fromSec, double toSec, double sr)
{
    const auto a = static_cast<std::size_t>(fromSec * sr);
    const auto b = std::min(x.size(), static_cast<std::size_t>(toSec * sr));
    return rms(x.data() + a, b - a);
}

/// Amplitude of the steady-state response at hz (least-squares sine/cosine fit over [from, end)).
double fitAmplitude(const std::vector<float>& x, std::size_t from, double hz, double sr)
{
    double ss = 0, cc = 0, sc = 0, xs = 0, xc = 0;
    for (std::size_t i = from; i < x.size(); ++i)
    {
        const double w = 2.0 * kPi * hz * static_cast<double>(i) / sr;
        const double s = std::sin(w), c = std::cos(w);
        ss += s * s;
        cc += c * c;
        sc += s * c;
        xs += x[i] * s;
        xc += x[i] * c;
    }
    const double det = ss * cc - sc * sc;
    const double a = (xs * cc - xc * sc) / det;
    const double b = (xc * ss - xs * sc) / det;
    return std::sqrt(a * a + b * b);
}

} // namespace

// Building blocks -------------------------------------------------------------------------------

TEST_CASE("DelayLine integer delays are exact", "[dsp][delay]")
{
    dd::DelayLine line;
    line.prepare(1000);
    for (int d : {0, 1, 7, 64, 999})
    {
        line.reset();
        std::vector<float> out(1100);
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            line.write(i == 0 ? 1.0f : 0.0f);
            out[i] = line.readNone(static_cast<float>(d));
        }
        REQUIRE(argmaxAbs(out) == d);
        REQUIRE(out[u(d)] == 1.0f);
        REQUIRE(rms(out) == Catch::Approx(1.0 / std::sqrt(1100.0)).epsilon(1e-6));
    }
}

TEST_CASE("DelayLine fractional delay accuracy per interpolator", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    constexpr double hz = 1000.0;
    struct Case
    {
        dd::Interpolation mode;
        double tolerance; // max abs error on a unit sine
    };
    // Linear low-passes by cos(w/2) at f = 0.5 (2e-3 at 1 kHz); the others are near exact.
    const Case cases[] = {
        {dd::Interpolation::Linear, 4e-3},
        {dd::Interpolation::Cubic, 5e-4},
        {dd::Interpolation::Allpass, 1e-3},
        {dd::Interpolation::Sinc, 5e-4},
    };
    for (const auto& c : cases)
        for (double d : {10.0, 10.25, 10.5, 10.75, 37.3})
        {
            dd::DelayLine line;
            line.prepare(100);
            dd::AllpassReadState st;
            double maxErr = 0.0;
            for (int n = 0; n < 4000; ++n)
            {
                line.write(static_cast<float>(std::sin(2.0 * kPi * hz * n / sr)));
                const float y = line.read(static_cast<float>(d), c.mode, st);
                if (n > 500)
                    maxErr = std::max(maxErr, std::fabs(y - std::sin(2.0 * kPi * hz * (n - d) / sr)));
            }
            INFO("mode " << static_cast<int>(c.mode) << " delay " << d);
            REQUIRE(maxErr < c.tolerance);
        }
}

TEST_CASE("DelayLine sinc read is accurate high in the band", "[dsp][delay]")
{
    // 8 kHz at 48 kHz: linear would be off by ~13%, the windowed sinc stays within 1%.
    dd::DelayLine line;
    line.prepare(64);
    double maxErr = 0.0;
    for (int n = 0; n < 2000; ++n)
    {
        line.write(static_cast<float>(std::sin(2.0 * kPi * 8000.0 * n / 48000.0)));
        const float y = line.readSinc(12.5f);
        if (n > 100)
            maxErr = std::max(maxErr, std::fabs(y - std::sin(2.0 * kPi * 8000.0 * (n - 12.5) / 48000.0)));
    }
    REQUIRE(maxErr < 0.01);
}

TEST_CASE("Schroeder and nested allpasses preserve energy", "[dsp][delay]")
{
    dd::SchroederAllpass ap;
    ap.prepare(200);
    ap.setDelay(113.0f);
    ap.setGain(0.7f);
    dd::NestedAllpass nested;
    nested.prepare(300, 100);
    nested.setOuter(257.0f, 0.6f);
    nested.setInner(61.0f, -0.5f);
    double e1 = 0.0, e2 = 0.0;
    for (int n = 0; n < 200000; ++n)
    {
        const float x = n == 0 ? 1.0f : 0.0f;
        const double a = ap.process(x);
        const double b = nested.process(x);
        e1 += a * a;
        e2 += b * b;
    }
    REQUIRE(e1 == Catch::Approx(1.0).epsilon(1e-4));
    REQUIRE(e2 == Catch::Approx(1.0).epsilon(1e-4));
}

TEST_CASE("Allpass diffuser preserves energy and spreads an impulse", "[dsp][delay]")
{
    dd::AllpassDiffuser diff;
    const float delays[] = {81.0f, 139.0f, 206.0f, 293.0f};
    const float gains[] = {0.75f, 0.75f, 0.625f, 0.625f};
    diff.prepare(delays, 4);
    diff.setGains(gains);
    double e = 0.0;
    float maxAbs = 0.0f;
    for (int n = 0; n < 100000; ++n)
    {
        const float y = diff.process(n == 0 ? 1.0f : 0.0f);
        e += static_cast<double>(y) * y;
        maxAbs = std::max(maxAbs, std::fabs(y));
    }
    REQUIRE(e == Catch::Approx(1.0).epsilon(1e-4));
    REQUIRE(maxAbs < 0.8f); // the impulse has been smeared
}

TEST_CASE("Damped comb decays by its gain per period; DC blocker removes DC", "[dsp][delay]")
{
    dd::DampedComb comb;
    comb.prepare(200);
    comb.setDelay(100.0f);
    comb.setFeedback(0.8f);
    comb.setDamping(0.0f);
    std::vector<float> out(1000);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = comb.process(i == 0 ? 1.0f : 0.0f);
    for (int k = 1; k < 8; ++k)
        REQUIRE(out[u(100 * (k + 1))] == Catch::Approx(out[u(100 * k)] * 0.8f).epsilon(1e-5));

    dd::DcBlocker dc;
    dc.setCorner(10.0, 48000.0);
    float y = 0.0f;
    for (int n = 0; n < 48000; ++n)
        y = dc.process(1.0f);
    REQUIRE(std::fabs(y) < 1e-3f);
}

TEST_CASE("Hadamard mixing is orthogonal and the frozen FDN holds its energy", "[dsp][delay]")
{
    std::array<float, 8> v{1, 2, 3, 4, 5, 6, 7, 8};
    double e0 = 0.0, e1 = 0.0;
    for (float x : v)
        e0 += x * x;
    dd::DelayFdn::hadamard(v);
    for (float x : v)
        e1 += x * x;
    REQUIRE(e1 == Catch::Approx(e0).epsilon(1e-6));

    dd::DelayFdn fdn;
    fdn.prepare(48000.0);
    fdn.setParams(0.5f, 1e6f, 1.0f, 1.0f, 0.0f, true);
    std::array<float, 8> in{};
    std::array<float, 8> out{};
    dd::Random r(3);
    for (int n = 0; n < 480; ++n)
    {
        for (auto& x : in)
            x = 0.1f * r.nextBipolar();
        fdn.tick(in.data(), out.data());
    }
    in.fill(0.0f);
    const auto energy = [&](int samples)
    {
        double e = 0.0;
        for (int n = 0; n < samples; ++n)
        {
            fdn.tick(in.data(), out.data());
            for (float x : out)
                e += static_cast<double>(x) * x;
        }
        return e;
    };
    energy(48000);
    const double a = energy(48000);
    energy(48000 * 5);
    const double b = energy(48000);
    REQUIRE(b / a > 0.99);
    REQUIRE(b / a < 1.001);
}

// DelayProcessor --------------------------------------------------------------------------------

TEST_CASE("Delay echoes land on the sample at every rate", "[dsp][delay]")
{
    for (double sr : {44100.0, 48000.0, 96000.0})
    {
        auto d = make<DelayProcessor>(sr, 2);
        auto prm = defaults<DelayProcessor>();
        set<DelayProcessor>(prm, DelayProcessor::kTimeL, 100.0f);
        set<DelayProcessor>(prm, DelayProcessor::kTimeR, 40.0f);
        set<DelayProcessor>(prm, DelayProcessor::kFeedback, 0.0f);
        auto io = silence(2, static_cast<long long>(0.3 * sr));
        io[0][0] = 1.0f;
        io[1][0] = 1.0f;
        run(*d, prm, io);
        const float tL =
            mapParam(DelayProcessor::params()[DelayProcessor::kTimeL], prm[DelayProcessor::kTimeL]);
        const float tR =
            mapParam(DelayProcessor::params()[DelayProcessor::kTimeR], prm[DelayProcessor::kTimeR]);
        INFO("sr " << sr);
        REQUIRE(argmaxAbs(io[0]) == std::llround(tL * sr / 1000.0));
        REQUIRE(argmaxAbs(io[1]) == std::llround(tR * sr / 1000.0));
        REQUIRE(peak(io[0]) > 0.98f);
        REQUIRE(peak(io[1]) > 0.98f);
    }
}

TEST_CASE("Delay feedback repeats decay by the feedback gain", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    for (float g : {0.3f, 0.6f, 0.9f})
    {
        auto d = make<DelayProcessor>(sr, 1);
        auto prm = defaults<DelayProcessor>();
        set<DelayProcessor>(prm, DelayProcessor::kTimeL, 100.0f);
        set<DelayProcessor>(prm, DelayProcessor::kFeedback, g);
        auto io = silence(1, static_cast<long long>(0.75 * sr));
        // 20 ms burst of 1 kHz, low level so the soft limiter stays out of the way.
        for (int i = 0; i < 960; ++i)
            io[0][u(i)] = 0.1f * static_cast<float>(std::sin(2.0 * kPi * 1000.0 * i / sr));
        run(*d, prm, io);
        std::vector<double> echo;
        for (int k = 1; k <= 5; ++k)
            echo.push_back(windowRms(io[0], 0.1 * k + 0.002, 0.1 * k + 0.018, sr));
        INFO("feedback " << g);
        for (std::size_t k = 1; k < echo.size(); ++k)
            REQUIRE(echo[k] / echo[k - 1] == Catch::Approx(g).epsilon(0.02));
    }
}

TEST_CASE("Delay ping-pong alternates channels", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    auto d = make<DelayProcessor>(sr, 2);
    auto prm = defaults<DelayProcessor>();
    set<DelayProcessor>(prm, DelayProcessor::kTimeL, 50.0f);
    set<DelayProcessor>(prm, DelayProcessor::kLink, 1.0f);
    set<DelayProcessor>(prm, DelayProcessor::kFeedback, 0.5f);
    set<DelayProcessor>(prm, DelayProcessor::kPingPong, 1.0f);
    auto io = silence(2, static_cast<long long>(0.2 * sr));
    io[0][0] = 1.0f; // left only
    run(*d, prm, io);
    const auto at = [&](int ch, double sec)
    {
        return peak(io[u(ch)], static_cast<std::size_t>(sec * sr) - 20,
                    static_cast<std::size_t>(sec * sr) + 20);
    };
    // Mono input sum (0.5) goes to the left line; repeats then cross sides.
    REQUIRE(at(0, 0.05) > 0.45f);
    REQUIRE(at(1, 0.05) < 1e-3f);
    REQUIRE(at(1, 0.10) > 0.2f);
    REQUIRE(at(0, 0.10) < 1e-3f);
    REQUIRE(at(0, 0.15) > 0.1f);
}

TEST_CASE("Delay stays bounded at maximum feedback over 60 s of noise", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    auto d = make<DelayProcessor>(sr, 2);
    Params prm{};
    prm.fill(1.0f); // max feedback, ping-pong, diffusion, wow, 2 s delays
    prm[DelayProcessor::kLowCut] = 0.0f;
    prm[DelayProcessor::kTimeL] = 0.4f;
    prm[DelayProcessor::kTimeR] = 0.45f;
    prm[DelayProcessor::kLink] = 0.0f;
    auto io = noise(2, static_cast<long long>(60.0 * sr), 0.5f);
    run(*d, prm, io);
    REQUIRE(finiteAndBounded(io, 8.0f));
    // The level saturates instead of growing: the last seconds are no louder than the middle.
    REQUIRE(windowRms(io[0], 55.0, 60.0, sr) < 1.5 * windowRms(io[0], 25.0, 30.0, sr) + 1e-3);
}

TEST_CASE("Delay time sweeps and jumps are click-free", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    auto d = make<DelayProcessor>(sr, 1);
    auto prm = defaults<DelayProcessor>();
    set<DelayProcessor>(prm, DelayProcessor::kFeedback, 0.0f);
    auto io = sine(1, static_cast<long long>(4.0 * sr), 440.0, sr, 0.5f);
    dd::Random r(9);
    const int blocks = static_cast<int>(io[0].size()) / kControlBlock;
    run(*d, prm, io,
        [&](int b, Params& p)
        {
            if (b < blocks / 2)
                p[DelayProcessor::kTimeL] =
                    0.3f + 0.6f * static_cast<float>(b) / static_cast<float>(blocks / 2); // sweep
            else if (b % 150 == 0)
                p[DelayProcessor::kTimeL] = 0.5f + 0.4f * r.nextBipolar(); // jumps every 100 ms
        });
    // Input slope is 0.5 * 2 pi 440 / 48000 = 0.029 per sample; allow for the tape pitch bend.
    float maxStep = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(0.6 * sr); i < io[0].size(); ++i)
        maxStep = std::max(maxStep, std::fabs(io[0][i] - io[0][i - 1]));
    REQUIRE(maxStep < 0.06f);
}

// ModDelayProcessor -----------------------------------------------------------------------------

TEST_CASE("ModDelay without depth is a plain delay at the base time", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    auto m = make<ModDelayProcessor>(sr, 1);
    auto prm = defaults<ModDelayProcessor>();
    set<ModDelayProcessor>(prm, ModDelayProcessor::kBase, 10.0f);
    set<ModDelayProcessor>(prm, ModDelayProcessor::kDepth, 0.0f);
    set<ModDelayProcessor>(prm, ModDelayProcessor::kFeedback, 0.0f);
    set<ModDelayProcessor>(prm, ModDelayProcessor::kMix, 1.0f);
    auto io = silence(1, 2000);
    io[0][0] = 1.0f;
    run(*m, prm, io);
    const float base =
        mapParam(ModDelayProcessor::params()[ModDelayProcessor::kBase], prm[ModDelayProcessor::kBase]);
    REQUIRE(argmaxAbs(io[0]) == std::llround(base * sr / 1000.0));
}

TEST_CASE("Chorus pitch modulation has the expected depth", "[dsp][delay]")
{
    // d(t) = base + depth/2 (1 + sin(2 pi r t)) => f_inst = f (1 - d'(t)), peak deviation
    // f * depth * pi * r.
    constexpr double sr = 48000.0;
    constexpr double f = 1000.0, depthMs = 4.0, rate = 2.0;
    auto m = make<ModDelayProcessor>(sr, 1);
    auto prm = defaults<ModDelayProcessor>();
    set<ModDelayProcessor>(prm, ModDelayProcessor::kBase, 5.0f);
    set<ModDelayProcessor>(prm, ModDelayProcessor::kDepth, static_cast<float>(depthMs));
    set<ModDelayProcessor>(prm, ModDelayProcessor::kRate, static_cast<float>(rate));
    set<ModDelayProcessor>(prm, ModDelayProcessor::kFeedback, 0.0f);
    set<ModDelayProcessor>(prm, ModDelayProcessor::kMix, 1.0f);
    auto io = sine(1, static_cast<long long>(2.0 * sr), f, sr, 0.5f);
    run(*m, prm, io);

    std::vector<double> crossings;
    const auto& y = io[0];
    for (std::size_t n = static_cast<std::size_t>(0.1 * sr); n < y.size(); ++n)
        if (y[n - 1] < 0.0f && y[n] >= 0.0f)
            crossings.push_back(static_cast<double>(n - 1) + y[n - 1] / (y[n - 1] - y[n]));
    double lo = 1e9, hi = 0.0;
    for (std::size_t k = 1; k < crossings.size(); ++k)
    {
        const double hz = sr / (crossings[k] - crossings[k - 1]);
        lo = std::min(lo, hz);
        hi = std::max(hi, hz);
    }
    const double expected = f * depthMs * 1e-3 * kPi * rate; // 25.1 Hz
    REQUIRE((hi - f) == Catch::Approx(expected).epsilon(0.1));
    REQUIRE((f - lo) == Catch::Approx(expected).epsilon(0.1));
}

TEST_CASE("Flanger is stable at extreme feedback and the shortest delay", "[dsp][delay]")
{
    for (float fb : {-0.95f, 0.95f})
    {
        auto m = make<ModDelayProcessor>(48000.0, 2);
        auto prm = defaults<ModDelayProcessor>();
        prm[ModDelayProcessor::kBase] = 0.0f;
        prm[ModDelayProcessor::kDepth] = 1.0f;
        prm[ModDelayProcessor::kRate] = 1.0f;
        set<ModDelayProcessor>(prm, ModDelayProcessor::kFeedback, fb);
        auto io = noise(2, 48000 * 5);
        run(*m, prm, io);
        REQUIRE(finiteAndBounded(io, 30.0f));
    }
}

// PhaserProcessor -------------------------------------------------------------------------------

TEST_CASE("Phaser notches sit where the allpass chain reaches odd multiples of 180 degrees", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    constexpr double fc = 1000.0;
    for (int stages : {4, 8})
    {
        auto ph = make<PhaserProcessor>(sr, 1);
        auto prm = defaults<PhaserProcessor>();
        set<PhaserProcessor>(prm, PhaserProcessor::kStages, static_cast<float>(stages / 2 - 1));
        set<PhaserProcessor>(prm, PhaserProcessor::kCentre, static_cast<float>(fc));
        set<PhaserProcessor>(prm, PhaserProcessor::kDepth, 0.0f);
        set<PhaserProcessor>(prm, PhaserProcessor::kFeedback, 0.0f);
        set<PhaserProcessor>(prm, PhaserProcessor::kMix, 0.5f);
        const auto gainAt = [&](double hz)
        {
            ph->reset();
            auto io = sine(1, 24000, hz, sr, 1.0f);
            run(*ph, prm, io);
            return fitAmplitude(io[0], 12000, hz, sr);
        };
        const double tc = std::tan(kPi * fc / sr);
        for (int k = 0; k < stages / 2; ++k)
        {
            const double notch = std::atan(tc * std::tan(kPi * (2 * k + 1) / (2.0 * stages))) * sr / kPi;
            INFO("stages " << stages << " notch " << notch);
            REQUIRE(gainAt(notch) < 0.01); // below -40 dB
            REQUIRE(gainAt(notch * 1.3) > gainAt(notch) * 10.0);
        }
    }
}

TEST_CASE("Phaser is stable at maximum feedback with a fast deep sweep", "[dsp][delay]")
{
    for (float fb : {-0.95f, 0.95f})
    {
        auto ph = make<PhaserProcessor>(48000.0, 2);
        Params prm{};
        prm.fill(1.0f);
        set<PhaserProcessor>(prm, PhaserProcessor::kFeedback, fb);
        auto io = noise(2, 48000 * 5);
        run(*ph, prm, io);
        REQUIRE(finiteAndBounded(io, 30.0f));
    }
}

// ReverbProcessor -------------------------------------------------------------------------------

namespace
{
/// RT60 from the Schroeder energy decay curve, fitted between -5 and -35 dB.
double measureRt60(const Buffers& io, double sr)
{
    const std::size_t n = io[0].size();
    std::vector<double> edc(n);
    double acc = 0.0;
    for (std::size_t i = n; i-- > 0;)
    {
        for (const auto& ch : io)
            acc += static_cast<double>(ch[i]) * ch[i];
        edc[i] = acc;
    }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int count = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double db = 10.0 * std::log10(edc[i] / edc[0] + 1e-30);
        if (db <= -5.0 && db >= -35.0)
        {
            const double t = static_cast<double>(i) / sr;
            sx += t;
            sy += db;
            sxx += t * t;
            sxy += t * db;
            ++count;
        }
    }
    const double denom = count * sxx - sx * sx;
    if (count < 2 || denom == 0.0)
        return 0.0; // no usable decay: callers' RT60 checks fail rather than divide by zero
    const double slope = (count * sxy - sx * sy) / denom;
    return slope < 0.0 ? -60.0 / slope : 0.0;
}
} // namespace

TEST_CASE("Reverb RT60 is within 10% of the target", "[dsp][delay]")
{
    for (double sr : {44100.0, 48000.0, 96000.0})
        for (float rt : {0.5f, 2.0f, 6.0f})
        {
            if (sr > 50000.0 && rt > 3.0f)
                continue; // keeps the debug run short; 96 kHz is covered by the shorter decays
            auto rv = make<ReverbProcessor>(sr, 2);
            auto prm = defaults<ReverbProcessor>();
            set<ReverbProcessor>(prm, ReverbProcessor::kDecay, rt);
            set<ReverbProcessor>(prm, ReverbProcessor::kDamping, 0.0f);
            set<ReverbProcessor>(prm, ReverbProcessor::kLowDamping, 0.0f);
            set<ReverbProcessor>(prm, ReverbProcessor::kEarlyLate, 1.0f);
            set<ReverbProcessor>(prm, ReverbProcessor::kPreDelay, 0.0f);
            auto io = silence(2, static_cast<long long>((2.0 * rt + 0.3) * sr));
            io[0][0] = 1.0f;
            io[1][0] = 1.0f;
            run(*rv, prm, io);
            INFO("sr " << sr << " rt60 " << rt);
            REQUIRE(measureRt60(io, sr) == Catch::Approx(rt).epsilon(0.05)); // spec: 10%
        }
}

TEST_CASE("Reverb damping shortens the high-frequency decay", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    const auto energyAfter = [&](float damping, double hz)
    {
        auto rv = make<ReverbProcessor>(sr, 1);
        auto prm = defaults<ReverbProcessor>();
        set<ReverbProcessor>(prm, ReverbProcessor::kDecay, 2.0f);
        set<ReverbProcessor>(prm, ReverbProcessor::kDamping, damping);
        auto io = sine(1, static_cast<long long>(1.5 * sr), hz, sr, 0.5f);
        std::fill(io[0].begin() + static_cast<std::ptrdiff_t>(0.5 * sr), io[0].end(), 0.0f);
        run(*rv, prm, io);
        return windowRms(io[0], 1.2, 1.5, sr);
    };
    REQUIRE(energyAfter(1.0f, 6000.0) < 0.2 * energyAfter(0.0f, 6000.0));
    REQUIRE(energyAfter(1.0f, 200.0) > 0.5 * energyAfter(0.0f, 200.0));
}

TEST_CASE("Reverb freeze sustains without growing and mutes the input", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    auto rv = make<ReverbProcessor>(sr, 2);
    auto prm = defaults<ReverbProcessor>();
    set<ReverbProcessor>(prm, ReverbProcessor::kDecay, 5.0f);
    auto io = noise(2, static_cast<long long>(16.0 * sr), 0.3f);
    run(*rv, prm, io, [&](int b, Params& p)
        { p[ReverbProcessor::kFreeze] = b * kControlBlock >= static_cast<int>(sr) ? 1.0f : 0.0f; });
    // Input noise keeps running after the freeze at 1 s; it must not reach the tail.
    const double early = windowRms(io[0], 4.0, 5.0, sr) + windowRms(io[1], 4.0, 5.0, sr);
    const double late = windowRms(io[0], 15.0, 16.0, sr) + windowRms(io[1], 15.0, 16.0, sr);
    REQUIRE(early > 0.01);
    REQUIRE(late / early > 0.9);
    REQUIRE(late / early < 1.02);
}

TEST_CASE("Reverb stereo outputs are decorrelated", "[dsp][delay]")
{
    constexpr double sr = 48000.0;
    auto rv = make<ReverbProcessor>(sr, 2);
    auto prm = defaults<ReverbProcessor>();
    auto io = silence(2, static_cast<long long>(sr));
    io[0][0] = io[1][0] = 1.0f;
    run(*rv, prm, io);
    double lr = 0, ll = 0, rr = 0;
    for (std::size_t i = static_cast<std::size_t>(0.1 * sr); i < io[0].size(); ++i)
    {
        lr += static_cast<double>(io[0][i]) * io[1][i];
        ll += static_cast<double>(io[0][i]) * io[0][i];
        rr += static_cast<double>(io[1][i]) * io[1][i];
    }
    REQUIRE(std::fabs(lr / std::sqrt(ll * rr)) < 0.3);
}

// Generic checks for every family ---------------------------------------------------------------

TEMPLATE_TEST_CASE("Delay families: silence in gives silence out", "[dsp][delay]", DelayProcessor,
                   ModDelayProcessor, PhaserProcessor, ReverbProcessor)
{
    for (int channels : {1, 2})
    {
        auto p = make<TestType>(48000.0, channels);
        auto prm = defaults<TestType>();
        auto io = silence(channels, 48000);
        run(*p, prm, io);
        for (const auto& ch : io)
            REQUIRE(peak(ch) == 0.0f);
    }
}

TEMPLATE_TEST_CASE("Delay families: NaN and Inf inputs and params stay finite and recover", "[dsp][delay]",
                   DelayProcessor, ModDelayProcessor, PhaserProcessor, ReverbProcessor)
{
    for (int channels : {1, 2})
    {
        auto p = make<TestType>(48000.0, channels);
        auto prm = defaults<TestType>();
        auto io = noise(channels, 48000);
        io[0][1000] = std::numeric_limits<float>::quiet_NaN();
        io[u(channels - 1)][5000] = std::numeric_limits<float>::infinity();
        io[0][9000] = -std::numeric_limits<float>::infinity();
        run(*p, prm, io);
        REQUIRE(finiteAndBounded(io, 100.0f));
        REQUIRE(rms(io[0].data() + 30000, 18000) > 1e-3); // recovered

        Params bad = prm;
        bad.fill(std::numeric_limits<float>::quiet_NaN());
        auto io2 = noise(channels, 24000);
        run(*p, bad, io2);
        REQUIRE(finiteAndBounded(io2, 100.0f));
    }
}

TEMPLATE_TEST_CASE("Delay families: extreme and jumping params stay finite and bounded", "[dsp][delay]",
                   DelayProcessor, ModDelayProcessor, PhaserProcessor, ReverbProcessor)
{
    for (int channels : {1, 2})
        for (float fill : {0.0f, 1.0f})
        {
            auto p = make<TestType>(48000.0, channels);
            Params prm{};
            prm.fill(fill);
            prm[TestType::kMix] = 1.0f;
            auto io = noise(channels, 48000 * 3);
            run(*p, prm, io);
            INFO("channels " << channels << " fill " << fill);
            REQUIRE(finiteAndBounded(io, 30.0f));
        }

    auto p = make<TestType>(48000.0, 2);
    Params prm{};
    dd::Random r(42);
    auto io = noise(2, 48000 * 4);
    run(*p, prm, io,
        [&](int, Params& q)
        {
            for (auto& x : q)
                x = 0.5f + 0.5f * r.nextBipolar(); // every param jumps every block
        });
    REQUIRE(finiteAndBounded(io, 30.0f));
}

TEMPLATE_TEST_CASE("Delay families: run at 44.1, 48 and 96 kHz", "[dsp][delay]", DelayProcessor,
                   ModDelayProcessor, PhaserProcessor, ReverbProcessor)
{
    for (double sr : {44100.0, 48000.0, 96000.0})
    {
        auto p = make<TestType>(sr, 2);
        auto prm = defaults<TestType>();
        auto io = sine(2, static_cast<long long>(sr), 440.0, sr, 0.5f);
        run(*p, prm, io);
        INFO("sr " << sr);
        REQUIRE(finiteAndBounded(io, 10.0f));
        REQUIRE(rms(io[0].data() + io[0].size() / 2, io[0].size() / 2) > 0.01);
        REQUIRE(p->tailSamples() >= 0);
    }
}

TEMPLATE_TEST_CASE("Delay families: setParams and process do not allocate", "[dsp][delay]", DelayProcessor,
                   ModDelayProcessor, PhaserProcessor, ReverbProcessor)
{
    auto p = make<TestType>(48000.0, 2);
    Params prm{};
    dd::Random r(7);
    auto io = noise(2, 48000);
    std::array<float*, 2> ptr{};
    AllocationGuard guard;
    for (std::size_t start = 0; start < io[0].size(); start += kControlBlock)
    {
        for (auto& x : prm)
            x = 0.5f + 0.5f * r.nextBipolar();
        p->setParams(prm.data(), {});
        ptr[0] = io[0].data() + start;
        ptr[1] = io[1].data() + start;
        p->process(ptr.data(), kControlBlock);
    }
    p->reset();
    REQUIRE(guard.count() == 0);
}

TEMPLATE_TEST_CASE("Delay families: parameter tables are well formed", "[dsp][delay]", DelayProcessor,
                   ModDelayProcessor, PhaserProcessor, ReverbProcessor)
{
    for (const auto& info : TestType::params())
    {
        if (info.name.empty())
            continue;
        REQUIRE(info.defaultNorm >= 0.0f);
        REQUIRE(info.defaultNorm <= 1.0f);
        REQUIRE(info.max > info.min);
        if (info.curve == Curve::Log)
            REQUIRE(info.min > 0.0f);
    }
}

// Benchmarks ------------------------------------------------------------------------------------

namespace
{
template <typename P>
void bench(const char* name, int instances, int channels, Params prm)
{
    constexpr double sr = 48000.0;
    std::vector<std::unique_ptr<P>> procs;
    for (int i = 0; i < instances; ++i)
        procs.push_back(make<P>(sr, channels));
    const auto src = noise(channels, 128);
    auto io = src;
    std::array<float*, 2> ptr{};
    const int blocks = static_cast<int>(sr) / 128;
    float sink = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < blocks; ++b)
        for (auto& p : procs)
            for (int s = 0; s < 128; s += kControlBlock)
            {
                p->setParams(prm.data(), {});
                for (int c = 0; c < channels; ++c)
                {
                    std::copy_n(src[u(c)].data() + s, kControlBlock, io[u(c)].data() + s); // fresh input
                    ptr[u(c)] = io[u(c)].data() + s;
                }
                p->process(ptr.data(), kControlBlock);
                sink += io[0][u(s)];
            }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %.2f us per 128-sample block (%d x %d ch)  [%g]\n", name, us / blocks, instances,
                channels, static_cast<double>(sink));
}
} // namespace

TEST_CASE("Delay family cost", "[.bench][dsp][delay]")
{
    {
        auto prm = defaults<DelayProcessor>();
        prm[DelayProcessor::kFeedback] = 0.9f;
        prm[DelayProcessor::kPingPong] = 0.5f;
        prm[DelayProcessor::kModDepth] = 0.5f;
        prm[DelayProcessor::kDiffusion] = 0.8f;
        prm[DelayProcessor::kHighCut] = 0.6f;
        prm[DelayProcessor::kLowCut] = 0.3f;
        bench<DelayProcessor>("Delay", 1, 2, prm);
    }
    {
        auto prm = defaults<ModDelayProcessor>();
        prm[ModDelayProcessor::kVoices] = 1.0f;
        prm[ModDelayProcessor::kFeedback] = 0.8f;
        bench<ModDelayProcessor>("ModDelay (4 voices)", 16, 1, prm);
        bench<ModDelayProcessor>("ModDelay stereo (4 voices)", 1, 2, prm);
    }
    {
        auto prm = defaults<PhaserProcessor>();
        prm[PhaserProcessor::kStages] = 1.0f;
        prm[PhaserProcessor::kFeedback] = 0.8f;
        bench<PhaserProcessor>("Phaser (12 stages)", 16, 1, prm);
        bench<PhaserProcessor>("Phaser stereo (12 stages)", 1, 2, prm);
    }
    {
        auto prm = defaults<ReverbProcessor>();
        prm[ReverbProcessor::kLowDamping] = 0.5f;
        bench<ReverbProcessor>("Reverb", 1, 2, prm);
    }
}
