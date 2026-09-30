#include "TestHelpers.h"
#include "alfa/dsp/Osc.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace alfa::dsp;
using alfa::dsp::test::AllocationGuard;
using alfa::dsp::test::rms;
using alfa::dsp::test::toneAmplitude;
using alfa::dsp::test::zeroCrossingHz;

namespace
{
constexpr double kSr = 48000.0;
constexpr int kHop = 192;
constexpr double kTwoPiD = 6.283185307179586;

PrepareSpec spec()
{
    PrepareSpec s;
    s.sampleRate = kSr;
    s.hop = kHop;
    return s;
}

/// Renders `hops` hops of a steady frame (prev == cur after the first) at a fixed f0.
std::vector<float> renderSteady(OscPrimitive& osc, const VoiceFrame& f, float f0, int hops)
{
    std::vector<float> out(static_cast<std::size_t>(hops * kHop));
    osc.reset();
    for (int h = 0; h < hops; ++h)
    {
        HopContext ctx{&f, &f, f0, f0, h == 0};
        osc.beginHop(ctx);
        osc.render(out.data() + h * kHop, kHop);
    }
    return out;
}
} // namespace

TEST_CASE("osc: harmonic 1 cosine part renders a unit cosine at f0", "[dsp][osc]")
{
    auto frame = std::make_unique<VoiceFrame>();
    frame->osc.re[0] = 1.0f;
    OscPrimitive osc;
    osc.prepare(spec());
    const auto y = renderSteady(osc, *frame, 440.0f, 250);

    REQUIRE(std::abs(y[0] - 1.0f) < 1e-4f);
    REQUIRE(std::abs(toneAmplitude(y, 440.0, kSr) - 1.0) < 2e-3);
    REQUIRE(std::abs(zeroCrossingHz(y, kSr) - 440.0) < 0.05);
    REQUIRE(std::abs(rms(y) - std::sqrt(0.5)) < 1e-3);
    // Sample-exact against the ideal cosine (linear table interpolation error is ~1e-6).
    double worst = 0.0;
    for (std::size_t n = 0; n < y.size(); ++n)
        worst = std::max(worst, std::abs(static_cast<double>(y[n]) -
                                         std::cos(kTwoPiD * 440.0 * static_cast<double>(n) / kSr)));
    REQUIRE(worst < 1e-4);
}

TEST_CASE("osc: im = -1 gives a sine starting at zero", "[dsp][osc]")
{
    auto frame = std::make_unique<VoiceFrame>();
    frame->osc.im[0] = -1.0f;
    OscPrimitive osc;
    osc.prepare(spec());
    const auto y = renderSteady(osc, *frame, 440.0f, 100);
    REQUIRE(std::abs(y[0]) < 1e-5f);
    REQUIRE(y[1] > 0.0f);
    REQUIRE(std::abs(static_cast<double>(y[1]) - std::sin(kTwoPiD * 440.0 / kSr)) < 1e-4);
    REQUIRE(std::abs(toneAmplitude(y, 440.0, kSr) - 1.0) < 2e-3);
}

TEST_CASE("osc: a saw spectrum renders its harmonic amplitudes", "[dsp][osc]")
{
    auto frame = std::make_unique<VoiceFrame>();
    for (int k = 0; k < kOscHarmonics; ++k)
        frame->osc.im[static_cast<std::size_t>(k)] =
            -1.0f / static_cast<float>(k + 1); // sum sin(2 pi h phi) / h
    OscPrimitive osc;
    osc.prepare(spec());
    const float f0 = 220.0f;
    const auto y = renderSteady(osc, *frame, f0, 250);
    for (int h = 1; h <= 5; ++h)
    {
        const double a = toneAmplitude(y, h * static_cast<double>(f0), kSr);
        INFO("harmonic " << h << " amplitude " << a);
        REQUIRE(std::abs(a - 1.0 / h) < 0.01 / h);
    }
    REQUIRE(std::abs(zeroCrossingHz(y, kSr) - 220.0) < 0.5);
}

TEST_CASE("osc: harmonics at or above Nyquist are removed", "[dsp][osc]")
{
    OscPrimitive osc;
    osc.prepare(spec());
    auto frame = std::make_unique<VoiceFrame>();
    frame->osc.re[4] = 1.0f; // harmonic 5 at 25 kHz
    auto y = renderSteady(osc, *frame, 5000.0f, 50);
    REQUIRE(rms(y) < 1e-6);

    // Exactly at Nyquist is removed too: harmonic 4 of 6 kHz.
    y = renderSteady(osc, *frame, 6000.0f, 10);
    REQUIRE(rms(y) < 1e-6);

    // Harmonic 4 at 20 kHz stays.
    auto below = std::make_unique<VoiceFrame>();
    below->osc.re[3] = 1.0f;
    y = renderSteady(osc, *below, 5000.0f, 50);
    REQUIRE(std::abs(toneAmplitude(y, 20000.0, kSr) - 1.0) < 0.01);

    // Harmonic 1024 sits on the Nyquist bin of the table and is kept when it is below Nyquist.
    auto top = std::make_unique<VoiceFrame>();
    top->osc.re[kOscHarmonics - 1] = 1.0f;
    // f0 = sr / 4096 steps half a table sample per output sample, where linear interpolation of
    // the table's Nyquist bin is exact: harmonic 1024 at 12 kHz reads cos(pi n / 2).
    y = renderSteady(osc, *top, static_cast<float>(kSr / 4096.0), 10);
    double worst = 0.0;
    for (std::size_t n = 0; n < y.size(); ++n)
        worst = std::max(worst, std::abs(static_cast<double>(y[n]) -
                                         std::cos(0.5 * 3.141592653589793 * static_cast<double>(n))));
    REQUIRE(worst < 1e-4);
}

TEST_CASE("osc: the crossfade between frames is continuous", "[dsp][osc]")
{
    auto a = std::make_unique<VoiceFrame>();
    auto b = std::make_unique<VoiceFrame>();
    a->osc.re[0] = 1.0f;
    a->osc.im[1] = -0.5f;
    b->osc.im[2] = 0.8f;
    b->osc.re[4] = -0.3f;
    const float f0 = 300.0f;
    // Largest slope of either wave, plus the crossfade's own step.
    const double slope = kTwoPiD * f0 / kSr * std::max(1.0 + 2 * 0.5, 3 * 0.8 + 5 * 0.3);
    const double bound = slope + (1.5 + 1.1) / kHop;

    OscPrimitive osc;
    osc.prepare(spec());
    osc.reset();
    std::vector<float> y(static_cast<std::size_t>(40 * kHop));
    const VoiceFrame* prev = a.get();
    for (int h = 0; h < 40; ++h)
    {
        const VoiceFrame* cur = (h % 2 == 0) ? a.get() : b.get();
        osc.beginHop({h == 0 ? cur : prev, cur, f0, f0, h == 0});
        // Uneven render calls inside the hop.
        osc.render(y.data() + h * kHop, 50);
        osc.render(y.data() + h * kHop + 50, kHop - 50);
        prev = cur;
    }
    double worst = 0.0;
    for (std::size_t n = 1; n < y.size(); ++n)
        worst = std::max(worst, static_cast<double>(std::abs(y[n] - y[n - 1])));
    INFO("worst step " << worst << " bound " << bound);
    REQUIRE(worst < bound);

    // At the end of hop 1 (a -> b) the output is almost b; at the start of hop 2 it is exactly b.
    auto ref = std::make_unique<VoiceFrame>(*b);
    OscPrimitive steady;
    steady.prepare(spec());
    const auto yb = renderSteady(steady, *ref, f0, 3);
    REQUIRE(std::abs(y[2 * kHop] - yb[2 * kHop]) < 1e-5f);
}

TEST_CASE("osc: a skipped hop keeps the phase and the next crossfade", "[dsp][osc]")
{
    auto a = std::make_unique<VoiceFrame>();
    auto b = std::make_unique<VoiceFrame>();
    a->osc.re[0] = 1.0f;
    b->osc.im[1] = 1.0f;
    OscPrimitive full, skipping;
    full.prepare(spec());
    skipping.prepare(spec());
    full.reset();
    skipping.reset();
    std::vector<float> y1(kHop), y2(kHop);

    const HopContext h0{a.get(), a.get(), 200.0f, 230.0f, true};
    const HopContext h1{a.get(), b.get(), 230.0f, 250.0f, false};
    const HopContext h2{b.get(), a.get(), 250.0f, 250.0f, false};
    full.beginHop(h0);
    full.render(y1.data(), kHop);
    full.beginHop(h1);
    full.render(y1.data(), kHop);
    full.beginHop(h2);
    full.render(y1.data(), kHop);

    skipping.skipHop(h0);
    skipping.skipHop(h1);
    skipping.beginHop(h2);
    skipping.render(y2.data(), kHop);

    double worst = 0.0;
    for (std::size_t n = 0; n < y1.size(); ++n)
        worst = std::max(worst, static_cast<double>(std::abs(y1[n] - y2[n])));
    REQUIRE(worst < 1e-4);
}

TEST_CASE("osc: beginHop and render do not allocate", "[dsp][osc]")
{
    auto frame = std::make_unique<VoiceFrame>();
    for (int k = 0; k < kOscHarmonics; ++k)
        frame->osc.re[static_cast<std::size_t>(k)] = 1.0f / static_cast<float>(k + 1);
    OscPrimitive osc;
    osc.prepare(spec());
    std::vector<float> y(kHop);
    long count = 0;
    {
        AllocationGuard guard;
        osc.reset();
        for (int h = 0; h < 20; ++h)
        {
            const HopContext ctx{frame.get(), frame.get(), 100.0f, 110.0f, h == 0};
            if (h % 5 == 3)
            {
                osc.skipHop(ctx);
                continue;
            }
            osc.beginHop(ctx);
            osc.render(y.data(), 64);
            osc.render(y.data() + 64, kHop - 64);
        }
        count = guard.count();
    }
    REQUIRE(count == 0);
}

TEST_CASE("osc: benchmark 16 voices", "[dsp][osc][.bench]")
{
    constexpr int kVoices = 16;
    constexpr int kBlock = 128;
    constexpr int kSeconds = 1;
    const int total = static_cast<int>(kSr) * kSeconds;

    std::vector<std::unique_ptr<VoiceFrame>> frames;
    for (int i = 0; i < 2; ++i)
    {
        frames.push_back(std::make_unique<VoiceFrame>());
        for (int k = 0; k < kOscHarmonics; ++k)
        {
            frames.back()->osc.re[static_cast<std::size_t>(k)] = 0.5f / static_cast<float>(k + 1 + i);
            frames.back()->osc.im[static_cast<std::size_t>(k)] = -0.3f / static_cast<float>(k + 2);
        }
    }
    std::vector<OscPrimitive> voices(kVoices);
    std::vector<int> pos(kVoices, 0);
    std::vector<int> hopIndex(kVoices, 0);
    for (auto& v : voices)
    {
        v.prepare(spec());
        v.reset();
    }
    std::vector<float> buf(kBlock);
    float sink = 0.0f;

    const auto t0 = std::chrono::steady_clock::now();
    int blocks = 0;
    for (int start = 0; start < total; start += kBlock, ++blocks)
    {
        for (int v = 0; v < kVoices; ++v)
        {
            const float f0 = 20.0f + 7.0f * static_cast<float>(v); // low f0: nearly all 1024 harmonics
            int done = 0;
            while (done < kBlock)
            {
                auto& p = pos[static_cast<std::size_t>(v)];
                if (p == 0)
                {
                    auto& j = hopIndex[static_cast<std::size_t>(v)];
                    const VoiceFrame* cur = frames[static_cast<std::size_t>(j % 2)].get();
                    const VoiceFrame* prv = frames[static_cast<std::size_t>((j + 1) % 2)].get();
                    voices[static_cast<std::size_t>(v)].beginHop({prv, cur, f0, f0, j == 0});
                    ++j;
                }
                const int n = std::min(kBlock - done, kHop - p);
                voices[static_cast<std::size_t>(v)].render(buf.data() + done, n);
                done += n;
                p = (p + n) % kHop;
            }
            sink += buf[5];
        }
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    const int hops = total / kHop;
    std::printf("OSC: %d voices x %d s: %.1f us total, %.2f us per %d-sample block (all voices), "
                "%.2f us per voice per hop, %.2f%% of one core per voice (sink %g)\n",
                kVoices, kSeconds, us, us / blocks, kBlock, us / (kVoices * hops),
                100.0 * us / (1e6 * kSeconds) / kVoices, static_cast<double>(sink));
    SUCCEED();
}
