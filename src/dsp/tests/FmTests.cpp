#include "TestHelpers.h"
#include "alfa/dsp/Fm.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace alfa::dsp;
using alfa::dsp::test::AllocationGuard;
using alfa::dsp::test::toneAmplitude;
using alfa::dsp::test::zeroCrossingHz;
using Catch::Matchers::WithinAbs;

namespace
{
// Bessel function of the first kind, J_n(x), by its power series. Apple's libc++ has no
// std::cyl_bessel_j; 40 terms are accurate far beyond the FM indices tested here.
double besselJ(int n, double x)
{
    if (n < 0)
        return (n % 2 == 0 ? 1.0 : -1.0) * besselJ(-n, x);
    const double half = 0.5 * x;
    double term = std::pow(half, n);
    for (int i = 1; i <= n; ++i)
        term /= i; // (x/2)^n / n!
    double sum = term;
    for (int m = 1; m < 40; ++m)
    {
        term *= -half * half / (static_cast<double>(m) * static_cast<double>(m + n));
        sum += term;
    }
    return sum;
}

constexpr double kSr = 48000.0;
constexpr int kHop = 192;
constexpr double kPiD = 3.14159265358979323846;

std::size_t u(int i)
{
    return static_cast<std::size_t>(i);
}

PrepareSpec spec(Tier tier = Tier::Studio)
{
    PrepareSpec s;
    s.sampleRate = kSr;
    s.hop = kHop;
    s.limits = limitsFor(tier);
    return s;
}

/// Renders `hops` hops with constant controls, in chunks of `chunk` samples.
std::vector<float> renderConstant(FmPrimitive& fm, const VoiceFrame& f, float f0, int hops, int chunk = kHop)
{
    std::vector<float> out(u(hops * kHop));
    fm.reset();
    for (int h = 0; h < hops; ++h)
    {
        HopContext ctx{&f, &f, f0, f0, h == 0};
        fm.beginHop(ctx);
        for (int done = 0; done < kHop;)
        {
            const int n = std::min(chunk, kHop - done);
            fm.render(out.data() + h * kHop + done, n);
            done += n;
        }
    }
    return out;
}

std::unique_ptr<VoiceFrame> carrier()
{
    auto f = std::make_unique<VoiceFrame>();
    f->fm.ratio[0] = 1.0f;
    f->fm.level[0] = 1.0f;
    return f;
}

} // namespace

TEST_CASE("FM single carrier is a unit sine at f0", "[dsp][fm]")
{
    FmPrimitive fm;
    fm.prepare(spec());
    auto f = carrier();
    const auto out = renderConstant(fm, *f, 440.0f, 250);

    double maxErr = 0.0;
    for (int n = 0; n < 4800; ++n)
        maxErr = std::max(maxErr, std::abs(out[u(n)] - std::sin(2.0 * kPiD * 440.0 * n / kSr)));
    REQUIRE(maxErr < 1e-4);
    REQUIRE_THAT(toneAmplitude(out, 440.0, kSr), WithinAbs(1.0, 0.01));
    REQUIRE_THAT(zeroCrossingHz(out, kSr), WithinAbs(440.0, 0.1));
    REQUIRE(fm.activeOperators() == 1);
}

TEST_CASE("FM 1:1 carrier:modulator matches the Bessel sideband amplitudes", "[dsp][fm]")
{
    // sin(wt + I sin(wt)) = sum_n J_n(I) sin((1 + n) wt). Harmonic k >= 1 collects n = k - 1 and
    // n = -(k + 1): amplitude J_{k-1}(I) + (-1)^k J_{k+1}(I). No DC.
    FmPrimitive fm;
    fm.prepare(spec());
    for (const double index : {0.5, 1.5, 3.0})
    {
        auto f = carrier();
        f->fm.ratio[1] = 1.0f;
        f->fm.index[0 * 8 + 1] = static_cast<float>(index);
        const float f0 = 300.0f;
        const auto out = renderConstant(fm, *f, f0, 250);
        REQUIRE(fm.activeOperators() == 2);
        for (int k = 1; k <= 5; ++k)
        {
            const double expected =
                std::abs(besselJ(k - 1, index) + ((k % 2 == 0) ? 1.0 : -1.0) * besselJ(k + 1, index));
            INFO("I = " << index << ", harmonic " << k);
            REQUIRE_THAT(toneAmplitude(out, static_cast<double>(f0) * k, kSr), WithinAbs(expected, 0.01));
        }
    }
}

TEST_CASE("FM ignores index entries on and below the diagonal", "[dsp][fm]")
{
    FmPrimitive fm;
    fm.prepare(spec());
    auto plain = carrier();
    plain->fm.ratio[1] = 2.0f; // present but silent and not routed
    const auto ref = renderConstant(fm, *plain, 440.0f, 20);

    auto f = carrier();
    f->fm.ratio[1] = 2.0f;
    f->fm.index[0 * 8 + 0] = 3.0f; // self
    f->fm.index[1 * 8 + 0] = 5.0f; // op1 modulated by op0: lower triangle
    f->fm.index[1 * 8 + 1] = 2.0f;
    const auto out = renderConstant(fm, *f, 440.0f, 20);
    REQUIRE(fm.activeOperators() == 1);
    REQUIRE(out == ref);
}

TEST_CASE("FM honours the tier operator limit", "[dsp][fm]")
{
    auto f = carrier();
    f->fm.ratio[2] = 3.0f;
    f->fm.ratio[3] = 2.0f;
    f->fm.level[3] = 1.0f;         // an audible operator beyond Lite's two
    f->fm.index[0 * 8 + 2] = 2.0f; // and a modulator beyond it

    FmPrimitive lite;
    lite.prepare(spec(Tier::Lite));
    const auto out = renderConstant(lite, *f, 440.0f, 20);

    FmPrimitive ref;
    ref.prepare(spec(Tier::Lite));
    auto plain = carrier();
    REQUIRE(out == renderConstant(ref, *plain, 440.0f, 20));

    FmPrimitive studio;
    studio.prepare(spec(Tier::Studio));
    const auto full = renderConstant(studio, *f, 440.0f, 20);
    REQUIRE(studio.activeOperators() == 3);
    REQUIRE(full != out);
}

namespace
{

/// Double-precision reference of the FM contract with ramps (k / H interpolation).
struct FmReference
{
    double phase[8] = {};

    void hop(const VoiceFrame& prev, const VoiceFrame& cur, double f0s, double f0e, int ops, float* out)
    {
        for (int k = 0; k < kHop; ++k)
        {
            const double t = static_cast<double>(k) / kHop;
            auto lerp = [t](float a, float b)
            {
                return a + (b - a) * t;
            };
            double y[8] = {};
            double acc = 0.0;
            for (int i = ops - 1; i >= 0; --i)
            {
                double pm = 0.0;
                for (int j = i + 1; j < ops; ++j)
                    pm += lerp(prev.fm.index[u(i * 8 + j)], cur.fm.index[u(i * 8 + j)]) * y[j];
                y[i] = std::sin(2.0 * kPiD * phase[i] + pm);
                acc += lerp(prev.fm.level[u(i)], cur.fm.level[u(i)]) * y[i];
            }
            out[k] = static_cast<float>(acc);
            const double f0 = f0s + (f0e - f0s) * t;
            for (int i = 0; i < ops; ++i)
            {
                phase[i] += f0 * lerp(prev.fm.ratio[u(i)], cur.fm.ratio[u(i)]) / kSr;
                phase[i] -= std::floor(phase[i]);
            }
        }
    }
};

std::vector<std::unique_ptr<VoiceFrame>> rampFrames(int count)
{
    std::vector<std::unique_ptr<VoiceFrame>> frames;
    for (int j = 0; j < count; ++j)
    {
        auto f = std::make_unique<VoiceFrame>();
        const auto x = static_cast<float>(j);
        f->fm.ratio[0] = 1.0f;
        f->fm.ratio[1] = 2.0f + 0.1f * std::sin(x);
        f->fm.ratio[2] = 3.5f;
        f->fm.level[0] = 0.5f + 0.4f * std::sin(0.7f * x);
        f->fm.level[2] = (j % 3 == 0) ? 0.3f : 0.0f;
        f->fm.index[0 * 8 + 1] = 1.0f + std::cos(0.3f * x);
        f->fm.index[1 * 8 + 2] = 0.5f * static_cast<float>(j % 4);
        f->fm.index[0 * 8 + 2] = 0.2f;
        frames.push_back(std::move(f));
    }
    return frames;
}

float f0At(int j)
{
    return 220.0f * std::pow(2.0f, static_cast<float>(j % 7) / 12.0f);
}

} // namespace

TEST_CASE("FM parameter ramps follow the contract and are continuous at hop boundaries", "[dsp][fm]")
{
    const int hops = 40;
    const auto frames = rampFrames(hops);
    FmPrimitive fm;
    fm.prepare(spec());
    fm.reset();
    FmReference ref;

    std::vector<float> out(u(hops * kHop)), expected(u(hops * kHop));
    for (int j = 0; j < hops; ++j)
    {
        const VoiceFrame& prev = *frames[u(j > 0 ? j - 1 : 0)];
        const VoiceFrame& cur = *frames[u(j)];
        const float fs = f0At(j > 0 ? j - 1 : 0), fe = f0At(j);
        fm.beginHop({&prev, &cur, fs, fe, j == 0});
        // Uneven chunks to check that render() continues within the hop.
        fm.render(out.data() + j * kHop, 37);
        fm.render(out.data() + j * kHop + 37, 100);
        fm.render(out.data() + j * kHop + 137, kHop - 137);
        ref.hop(prev, cur, fs, fe, 8, expected.data() + j * kHop);
    }

    double maxErr = 0.0, maxStep = 0.0;
    for (std::size_t n = 0; n < out.size(); ++n)
    {
        maxErr = std::max(maxErr, static_cast<double>(std::abs(out[n] - expected[n])));
        if (n > 0)
            maxStep = std::max(maxStep, static_cast<double>(std::abs(out[n] - out[n - 1])));
    }
    REQUIRE(maxErr < 2e-3);
    // Largest per-sample step of the reference bounds the rendered one: no jumps at boundaries.
    double refStep = 0.0;
    for (std::size_t n = 1; n < expected.size(); ++n)
        refStep = std::max(refStep, static_cast<double>(std::abs(expected[n] - expected[n - 1])));
    REQUIRE(maxStep < refStep + 4e-3);
}

TEST_CASE("FM skipHop keeps phases in step with rendering", "[dsp][fm]")
{
    const int hops = 12;
    const auto frames = rampFrames(hops);
    FmPrimitive a, b;
    a.prepare(spec());
    b.prepare(spec());
    a.reset();
    b.reset();
    std::vector<float> outA(u(kHop)), outB(u(kHop));
    for (int j = 0; j < hops; ++j)
    {
        const HopContext ctx{frames[u(j > 0 ? j - 1 : 0)].get(), frames[u(j)].get(), f0At(j > 0 ? j - 1 : 0),
                             f0At(j), j == 0};
        a.beginHop(ctx);
        a.render(outA.data(), kHop);
        if (j >= 3 && j <= 6)
        {
            b.skipHop(ctx);
            continue;
        }
        b.beginHop(ctx);
        b.render(outB.data(), kHop);
        if (j > 6)
            for (int n = 0; n < kHop; ++n)
                REQUIRE_THAT(outB[u(n)], WithinAbs(outA[u(n)], 1e-3));
    }
}

TEST_CASE("FM render does not allocate", "[dsp][fm]")
{
    const auto frames = rampFrames(8);
    FmPrimitive fm;
    fm.prepare(spec());
    std::vector<float> out(u(kHop));
    AllocationGuard guard;
    fm.reset();
    for (int j = 1; j < 8; ++j)
    {
        const HopContext ctx{frames[u(j - 1)].get(), frames[u(j)].get(), 200.0f, 210.0f, false};
        if (j == 4)
        {
            fm.skipHop(ctx);
            continue;
        }
        fm.beginHop(ctx);
        fm.render(out.data(), 64);
        fm.render(out.data() + 64, kHop - 64);
    }
    REQUIRE(guard.count() == 0);
}

TEST_CASE("FM cost, 16 voices x 1 s, all Studio operators", "[.bench][dsp][fm]")
{
    constexpr int kVoices = 16;
    const int hops = static_cast<int>(kSr) / kHop;
    auto frame = std::make_unique<VoiceFrame>();
    for (int i = 0; i < kMaxFmOperators; ++i)
    {
        frame->fm.ratio[u(i)] = static_cast<float>(i + 1);
        frame->fm.level[u(i)] = 0.1f;
        for (int j = i + 1; j < kMaxFmOperators; ++j)
            frame->fm.index[u(i * 8 + j)] = 0.3f;
    }
    std::vector<FmPrimitive> voices(kVoices);
    for (auto& v : voices)
    {
        v.prepare(spec(Tier::Studio));
        v.reset();
    }
    std::vector<float> out(u(kHop));
    float sink = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    for (int h = 0; h < hops; ++h)
        for (int v = 0; v < kVoices; ++v)
        {
            voices[u(v)].beginHop({frame.get(), frame.get(), 110.0f + static_cast<float>(v),
                                   110.0f + static_cast<float>(v), h == 0});
            voices[u(v)].render(out.data(), kHop);
            sink += out[0];
        }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    const double samples = static_cast<double>(hops) * kHop;
    const double perBlockAll = us / (samples / 128.0);
    std::printf("FM (8 ops): %.2f us per 128-sample block for %d voices, %.3f us per voice (sink %g)\n",
                perBlockAll, kVoices, perBlockAll / kVoices, static_cast<double>(sink));
    REQUIRE(voices[0].activeOperators() == 8);
}
