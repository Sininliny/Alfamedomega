#include "TestHelpers.h"
#include "alfa/dsp/Spatial.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{
constexpr double kPi = 3.14159265358979323846;

std::vector<float> randomSignal(std::size_t n, std::uint32_t seed, float scale = 0.5f)
{
    detail::Random rng(seed);
    std::vector<float> x(n);
    for (auto& v : x)
        v = scale * rng.nextBipolar();
    return x;
}

std::vector<float> sine(double hz, double rate, std::size_t n, float amp = 0.5f)
{
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = amp * static_cast<float>(std::sin(2.0 * kPi * hz * static_cast<double>(i) / rate));
    return x;
}

std::array<float, kSlotParams> defaults()
{
    std::array<float, kSlotParams> p{};
    const auto& t = SpatialProcessor::params();
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = t[i].defaultNorm;
    return p;
}

float norm(int param, float value)
{
    const auto& info = SpatialProcessor::params()[static_cast<std::size_t>(param)];
    return (value - info.min) / (info.max - info.min);
}

ProcessSpec stereoSpec(double rate = 48000.0)
{
    ProcessSpec s;
    s.sampleRate = rate;
    s.channels = 2;
    return s;
}

void run(SpatialProcessor& p, const std::array<float, kSlotParams>& params, std::vector<float>& l,
         std::vector<float>& r)
{
    ControlContext ctx;
    for (std::size_t pos = 0; pos < l.size(); pos += kControlBlock)
    {
        const int n = static_cast<int>(std::min<std::size_t>(kControlBlock, l.size() - pos));
        p.setParams(params.data(), ctx);
        float* ch[2] = {l.data() + pos, r.data() + pos};
        p.process(ch, n);
    }
}

std::vector<float> slice(const std::vector<float>& x, std::size_t from)
{
    return {x.begin() + static_cast<std::ptrdiff_t>(from), x.end()};
}

bool finite(const std::vector<float>& x)
{
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
} // namespace

// Helpers --------------------------------------------------------------------------------------

TEST_CASE("Pan laws: equal power keeps L^2 + R^2, linear keeps L + R, -4.5 dB at centre",
          "[dsp][convspatial]")
{
    for (int i = 0; i <= 100; ++i)
    {
        const float pan = -1.0f + 0.02f * static_cast<float>(i);
        const StereoGains ep = panGains(pan, PanLaw::EqualPower);
        REQUIRE(std::abs(ep.left * ep.left + ep.right * ep.right - 1.0f) < 1e-6f);
        const StereoGains lin = panGains(pan, PanLaw::Linear);
        REQUIRE(std::abs(lin.left + lin.right - 1.0f) < 1e-6f);
        const StereoGains mid = panGains(pan, PanLaw::Minus4_5dB);
        REQUIRE(mid.left >= std::min(lin.left, ep.left) - 1e-6f);
        REQUIRE(mid.left <= std::max(lin.left, ep.left) + 1e-6f);
    }
    REQUIRE(std::abs(20.0 * std::log10(panGains(0.0f, PanLaw::EqualPower).left) + 3.0103) < 0.01);
    REQUIRE(std::abs(20.0 * std::log10(panGains(0.0f, PanLaw::Minus4_5dB).left) + 4.515) < 0.01);
    REQUIRE(std::abs(20.0 * std::log10(panGains(0.0f, PanLaw::Linear).left) + 6.0206) < 0.01);
    for (auto law : {PanLaw::EqualPower, PanLaw::Minus4_5dB, PanLaw::Linear})
    {
        REQUIRE(panGains(-1.0f, law).left == 1.0f);
        REQUIRE(panGains(-1.0f, law).right == 0.0f);
        REQUIRE(panGains(1.0f, law).left == 0.0f);
        REQUIRE(panGains(1.0f, law).right == 1.0f);
    }
}

TEST_CASE("Mid/side round trip, width, balance, crossfade and rotation helpers", "[dsp][convspatial]")
{
    detail::Random rng(5);
    for (int i = 0; i < 10000; ++i)
    {
        const float l = rng.nextBipolar();
        const float r = rng.nextBipolar();
        const MidSide ms = midSideEncode(l, r);
        const StereoGains back = midSideDecode(ms.mid, ms.side);
        // One rounding of the sum and one of the difference: within an ulp of the inputs.
        REQUIRE(std::abs(back.left - l) <= 1.2e-7f);
        REQUIRE(std::abs(back.right - r) <= 1.2e-7f);

        float a = l, b = r;
        applyWidth(a, b, 0.0f);
        REQUIRE(a == b);
        a = l;
        b = r;
        applyWidth(a, b, 2.0f);
        REQUIRE(std::abs((a - b) - 2.0f * (l - r)) < 1e-5f);
        REQUIRE(std::abs((a + b) - (l + r)) < 1e-5f);

        const float t = 0.5f + 0.5f * rng.nextBipolar();
        const StereoGains x = crossfadeGains(t);
        REQUIRE(std::abs(x.left * x.left + x.right * x.right - 1.0f) < 1e-6f);

        const StereoMatrix rot = StereoMatrix::rotation(rng.nextBipolar());
        const float ol = rot.ll * l + rot.rl * r;
        const float orr = rot.lr * l + rot.rr * r;
        REQUIRE(std::abs(ol * ol + orr * orr - (l * l + r * r)) < 1e-5f);
    }
    REQUIRE(crossfadeGains(0.0f).left == 1.0f);
    REQUIRE(crossfadeGains(0.0f).right == 0.0f);
    REQUIRE(crossfadeGains(1.0f).right == 1.0f);
    REQUIRE(balanceGains(0.0f).left == 1.0f);
    REQUIRE(balanceGains(0.0f).right == 1.0f);
    REQUIRE(balanceGains(1.0f).left == 0.0f);
    REQUIRE(balanceGains(-0.5f).right == 0.5f);
    const StereoMatrix quarter = StereoMatrix::rotation(0.25f * static_cast<float>(kPi));
    // A centred source turns hard right at +45 degrees.
    REQUIRE(std::abs(quarter.ll + quarter.rl) < 1e-6f);
}

// Processor ------------------------------------------------------------------------------------

TEST_CASE("SPATIAL neutral defaults are bit-exact identity", "[dsp][convspatial]")
{
    for (double rate : {44100.0, 48000.0, 96000.0})
    {
        SpatialProcessor p;
        p.prepare(stereoSpec(rate));
        auto l = randomSignal(10000, 1);
        auto r = randomSignal(10000, 2);
        const auto l0 = l, r0 = r;
        run(p, defaults(), l, r);
        INFO("rate " << rate);
        REQUIRE(l == l0);
        REQUIRE(r == r0);
    }
}

TEST_CASE("SPATIAL width 0 gives L = R and width 2 doubles the side", "[dsp][convspatial]")
{
    SpatialProcessor p;
    p.prepare(stereoSpec());
    auto params = defaults();
    params[2] = 0.0f;
    auto l = randomSignal(4800, 1);
    auto r = randomSignal(4800, 2);
    run(p, params, l, r);
    REQUIRE(l == r);

    p.reset();
    params[2] = 1.0f;
    l = randomSignal(4800, 3);
    r = randomSignal(4800, 4);
    const auto l0 = l, r0 = r;
    run(p, params, l, r);
    for (std::size_t i = 0; i < l.size(); ++i)
    {
        REQUIRE(std::abs((l[i] - r[i]) - 2.0f * (l0[i] - r0[i])) < 1e-5f);
        REQUIRE(std::abs((l[i] + r[i]) - (l0[i] + r0[i])) < 1e-5f);
    }
}

TEST_CASE("SPATIAL pan, balance and rotation move the image", "[dsp][convspatial]")
{
    SpatialProcessor p;
    p.prepare(stereoSpec());
    auto params = defaults();

    SECTION("pan hard right sums both channels into R")
    {
        params[0] = 1.0f;
        auto l = randomSignal(4800, 1);
        auto r = randomSignal(4800, 2);
        const auto l0 = l, r0 = r;
        run(p, params, l, r);
        for (std::size_t i = 0; i < l.size(); ++i)
        {
            REQUIRE(l[i] == 0.0f);
            REQUIRE(std::abs(r[i] - (l0[i] + r0[i])) < 1e-6f);
        }
    }
    SECTION("M/S balance -1 keeps only the mid")
    {
        params[3] = 0.0f;
        auto l = randomSignal(4800, 1);
        auto r = randomSignal(4800, 2);
        run(p, params, l, r);
        REQUIRE(l == r);
    }
    SECTION("rotation +45 degrees sends a centred source hard right")
    {
        params[6] = 1.0f;
        auto l = sine(440.0, 48000.0, 4800);
        auto r = l;
        run(p, params, l, r);
        for (std::size_t i = 0; i < l.size(); ++i)
            REQUIRE(std::abs(l[i]) < 1e-6f);
        REQUIRE(std::abs(toneAmplitude(r, 440.0, 48000.0) - 0.5 * std::sqrt(2.0)) < 0.01);
    }
}

TEST_CASE("SPATIAL Haas delay is sample accurate on the selected side", "[dsp][convspatial]")
{
    for (double rate : {44100.0, 48000.0, 96000.0})
    {
        for (bool right : {false, true})
        {
            SpatialProcessor p;
            p.prepare(stereoSpec(rate));
            auto params = defaults();
            params[4] = norm(4, 10.0f);
            params[5] = right ? 1.0f : 0.0f;
            std::vector<float> l(static_cast<std::size_t>(0.1 * rate), 0.0f), r = l;
            l[100] = 1.0f;
            r[100] = 1.0f;
            run(p, params, l, r);
            const auto expected = 100 + static_cast<std::size_t>(std::lround(0.01 * rate));
            const auto& delayed = right ? r : l;
            const auto& direct = right ? l : r;
            INFO("rate " << rate << " right " << right);
            REQUIRE(direct[100] == 1.0f);
            REQUIRE(std::abs(delayed[expected] - 1.0f) < 1e-3f);
            REQUIRE(std::abs(delayed[expected - 1]) < 1e-3f);
            REQUIRE(std::abs(delayed[expected + 1]) < 1e-3f);
        }
    }
}

TEST_CASE("SPATIAL mono-maker sums the lows only", "[dsp][convspatial]")
{
    SpatialProcessor p;
    p.prepare(stereoSpec());
    auto params = defaults();
    params[7] = norm(7, 200.0f);

    auto measure = [&](double sideHz)
    {
        p.reset();
        // Mid at 1 kHz plus a side component at sideHz.
        const auto mid = sine(1000.0, 48000.0, 48000, 0.3f);
        const auto side = sine(sideHz, 48000.0, 48000, 0.3f);
        std::vector<float> l(mid.size()), r(mid.size());
        for (std::size_t i = 0; i < l.size(); ++i)
        {
            l[i] = mid[i] + side[i];
            r[i] = mid[i] - side[i];
        }
        run(p, params, l, r);
        std::vector<float> s(l.size()), m(l.size());
        for (std::size_t i = 0; i < l.size(); ++i)
        {
            s[i] = 0.5f * (l[i] - r[i]);
            m[i] = 0.5f * (l[i] + r[i]);
        }
        return std::pair{toneAmplitude(slice(s, 9600), sideHz, 48000.0),
                         toneAmplitude(slice(m, 9600), 1000.0, 48000.0)};
    };
    const auto [lowSide, mid1] = measure(40.0);
    const auto [highSide, mid2] = measure(3000.0);
    INFO("low side " << lowSide << " high side " << highSide);
    REQUIRE(lowSide < 0.3 * 0.01);                  // 40 Hz sides removed (> 40 dB)
    REQUIRE(std::abs(highSide / 0.3 - 1.0) < 0.02); // 3 kHz sides kept
    REQUIRE(std::abs(mid1 / 0.3 - 1.0) < 0.02);     // mid only phase-shifted
    REQUIRE(std::abs(mid2 / 0.3 - 1.0) < 0.02);

    // Turning it off again returns to the exact identity once the fade is over.
    auto off = defaults();
    auto l = randomSignal(4800, 1), r = randomSignal(4800, 2);
    run(p, off, l, r);
    l = randomSignal(4800, 3);
    r = randomSignal(4800, 4);
    const auto l0 = l, r0 = r;
    run(p, off, l, r);
    REQUIRE(l == l0);
    REQUIRE(r == r0);
}

TEST_CASE("SPATIAL distance lowers the level and rolls off the highs", "[dsp][convspatial]")
{
    SpatialProcessor p;
    p.prepare(stereoSpec());
    auto params = defaults();
    params[8] = 1.0f;
    auto level = [&](double hz)
    {
        p.reset();
        auto l = sine(hz, 48000.0, 24000);
        auto r = l;
        run(p, params, l, r);
        return 20.0 * std::log10(toneAmplitude(slice(l, 4800), hz, 48000.0) / 0.5);
    };
    const double low = level(100.0);
    const double high = level(16000.0);
    INFO("low " << low << " dB, high " << high << " dB");
    REQUIRE(std::abs(low + 18.0) < 0.5);
    REQUIRE(high < -27.0);
}

TEST_CASE("SPATIAL mix blends dry and processed", "[dsp][convspatial]")
{
    SpatialProcessor p;
    p.prepare(stereoSpec());
    auto params = defaults();
    params[2] = 0.0f; // processed = mono
    params[9] = 0.0f; // fully dry
    auto l = randomSignal(4800, 1);
    auto r = randomSignal(4800, 2);
    const auto l0 = l, r0 = r;
    run(p, params, l, r);
    REQUIRE(l == l0);
    REQUIRE(r == r0);
}

TEST_CASE("SPATIAL silence in gives silence out", "[dsp][convspatial]")
{
    SpatialProcessor p;
    p.prepare(stereoSpec());
    std::array<float, kSlotParams> params{};
    params.fill(0.7f);
    std::vector<float> l(48000, 0.0f), r = l;
    run(p, params, l, r);
    REQUIRE(*std::max_element(l.begin(), l.end()) == 0.0f);
    REQUIRE(*std::min_element(r.begin(), r.end()) == 0.0f);
    REQUIRE(p.tailSamples() > static_cast<int>(0.7 * 30e-3 * 48000.0) - 2);
}

TEST_CASE("SPATIAL survives NaN and Inf input and NaN params", "[dsp][convspatial]")
{
    SpatialProcessor p;
    p.prepare(stereoSpec());
    std::array<float, kSlotParams> params{};
    params.fill(0.6f);
    auto l = randomSignal(9600, 1);
    auto r = randomSignal(9600, 2);
    l[500] = std::nanf("");
    r[3000] = -INFINITY;
    run(p, params, l, r);
    REQUIRE(finite(l));
    REQUIRE(finite(r));
    std::array<float, kSlotParams> bad;
    bad.fill(std::nanf(""));
    l = randomSignal(9600, 3);
    r = randomSignal(9600, 4);
    run(p, bad, l, r);
    REQUIRE(finite(l));
    REQUIRE(rms(l) > 0.05);
}

TEST_CASE("SPATIAL extreme and randomly jumping params stay bounded without allocating", "[dsp][convspatial]")
{
    for (double rate : {44100.0, 48000.0, 96000.0})
    {
        for (int channels : {1, 2})
        {
            SpatialProcessor p;
            ProcessSpec spec = stereoSpec(rate);
            spec.channels = channels;
            p.prepare(spec);
            std::array<float, kSlotParams> params{};
            detail::Random rng(9);
            ControlContext ctx;
            float peak = 0.0f;
            long allocations = 0;
            for (int b = 0; b < 3000; ++b)
            {
                const int mode = b / 1000;
                for (auto& v : params)
                    v = mode == 0 ? 0.0f : mode == 1 ? 1.0f : 0.5f + 0.5f * rng.nextBipolar();
                std::array<float, kControlBlock> l{}, r{};
                for (std::size_t i = 0; i < l.size(); ++i)
                {
                    // Very low and very high frequencies mixed with noise.
                    const double t = static_cast<double>(b * kControlBlock + static_cast<int>(i)) / rate;
                    l[i] =
                        0.3f * static_cast<float>(std::sin(2.0 * kPi * 25.0 * t)) + 0.2f * rng.nextBipolar();
                    r[i] = 0.3f * static_cast<float>(std::sin(2.0 * kPi * 0.45 * rate * t)) +
                           0.2f * rng.nextBipolar();
                }
                float* ch[2] = {l.data(), r.data()};
                AllocationGuard guard;
                p.setParams(params.data(), ctx);
                p.process(ch, kControlBlock);
                allocations += guard.count();
                for (std::size_t i = 0; i < l.size(); ++i)
                {
                    REQUIRE(std::isfinite(l[i]));
                    REQUIRE(std::isfinite(r[i]));
                    peak = std::max({peak, std::abs(l[i]), std::abs(r[i])});
                }
            }
            INFO("rate " << rate << " channels " << channels << " peak " << peak);
            REQUIRE(allocations == 0);
            REQUIRE(peak < 4.0f);
        }
    }
}

TEST_CASE("SPATIAL cost, 1 stereo instance, 128-sample blocks", "[.bench][dsp][convspatial]")
{
    SpatialProcessor p;
    p.prepare(stereoSpec());
    auto params = defaults();
    params[0] = 0.3f;
    params[2] = 0.7f;
    params[3] = 0.6f;
    params[4] = 0.3f;
    params[6] = 0.6f;
    params[7] = 0.3f;
    params[8] = 0.4f;
    params[9] = 0.8f;
    auto l = randomSignal(128, 1), r = randomSignal(128, 2);
    ControlContext ctx;
    const int blocks = 48000 * 20 / 128;
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < blocks; ++b)
    {
        for (int s = 0; s < 128; s += kControlBlock)
        {
            p.setParams(params.data(), ctx);
            float* ch[2] = {l.data() + s, r.data() + s};
            p.process(ch, kControlBlock);
        }
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("SpatialProcessor, all features on: %.3f us per 128-sample block\n", us / blocks);
}
