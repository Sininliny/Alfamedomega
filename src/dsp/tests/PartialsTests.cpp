#include "TestHelpers.h"
#include "alfa/dsp/Partials.h"

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

namespace
{
constexpr double kSr = 48000.0;
constexpr int kHop = 192;

PrepareSpec spec(Tier tier = Tier::Studio)
{
    PrepareSpec s;
    s.sampleRate = kSr;
    s.hop = kHop;
    s.limits = limitsFor(tier);
    return s;
}

std::vector<float> renderSteady(PartialsPrimitive& p, const VoiceFrame& f, float f0, int hops)
{
    std::vector<float> out(static_cast<std::size_t>(hops * kHop));
    p.reset();
    for (int h = 0; h < hops; ++h)
    {
        p.beginHop({&f, &f, f0, f0, h == 0});
        p.render(out.data() + h * kHop, kHop);
    }
    return out;
}

void setPartial(VoiceFrame& f, int m, float ratio, float amp)
{
    f.partials.ratio[static_cast<std::size_t>(m)] = ratio;
    f.partials.amp[static_cast<std::size_t>(m)] = amp;
}
} // namespace

TEST_CASE("partials: ratios and amplitudes render as tones", "[dsp][partials]")
{
    auto f = std::make_unique<VoiceFrame>();
    setPartial(*f, 0, 1.0f, 0.5f);
    setPartial(*f, 1, 2.5f, 0.25f);
    PartialsPrimitive p;
    p.prepare(spec());
    const auto y = renderSteady(p, *f, 440.0f, 250);
    REQUIRE(p.activeCount() == 2);
    REQUIRE(std::abs(y[0]) < 1e-6f); // phases start at 0, sin(0) = 0
    REQUIRE(std::abs(toneAmplitude(y, 440.0, kSr) - 0.5) < 2e-3);
    REQUIRE(std::abs(toneAmplitude(y, 1100.0, kSr) - 0.25) < 2e-3);
    REQUIRE(toneAmplitude(y, 880.0, kSr) < 1e-3);
    // Sample-exact against the ideal sum.
    double worst = 0.0;
    for (std::size_t n = 0; n < y.size(); ++n)
    {
        const double t = static_cast<double>(n) / kSr;
        const double ideal =
            0.5 * std::sin(6.283185307179586 * 440.0 * t) + 0.25 * std::sin(6.283185307179586 * 1100.0 * t);
        worst = std::max(worst, std::abs(static_cast<double>(y[n]) - ideal));
    }
    REQUIRE(worst < 1e-4);
}

TEST_CASE("partials: the tier limit silences partials beyond it", "[dsp][partials]")
{
    auto f = std::make_unique<VoiceFrame>();
    setPartial(*f, 20, 1.0f, 1.0f); // beyond Lite's 16
    PartialsPrimitive p;
    p.prepare(spec(Tier::Lite));
    auto y = renderSteady(p, *f, 440.0f, 20);
    REQUIRE(p.activeCount() == 0);
    REQUIRE(rms(y) == 0.0);

    p.prepare(spec(Tier::Standard)); // 32
    y = renderSteady(p, *f, 440.0f, 20);
    REQUIRE(p.activeCount() == 1);
    REQUIRE(rms(y) > 0.5);
}

TEST_CASE("partials: a partial at or above Nyquist is silent", "[dsp][partials]")
{
    PartialsPrimitive p;
    p.prepare(spec());
    auto f = std::make_unique<VoiceFrame>();
    setPartial(*f, 0, 3.0f, 1.0f); // 30 kHz
    setPartial(*f, 1, 2.4f, 1.0f); // exactly 24 kHz
    auto y = renderSteady(p, *f, 10000.0f, 20);
    REQUIRE(rms(y) == 0.0);

    setPartial(*f, 2, 2.0f, 1.0f); // 20 kHz stays
    y = renderSteady(p, *f, 10000.0f, 50);
    REQUIRE(std::abs(toneAmplitude(y, 20000.0, kSr) - 1.0) < 5e-3);
}

TEST_CASE("partials: a partial gliding through Nyquist goes silent from that sample", "[dsp][partials]")
{
    auto f = std::make_unique<VoiceFrame>();
    setPartial(*f, 0, 1.0f, 1.0f);
    setPartial(*f, 1, 0.1f, 0.0f); // silent, only checks that other partials don't interfere
    PartialsPrimitive p;
    p.prepare(spec());
    p.reset();
    std::vector<float> y(kHop);
    // f0 glides 20 kHz -> 30 kHz: Nyquist is reached at n = 0.4 * 192 = 76.8.
    p.beginHop({f.get(), f.get(), 20000.0f, 30000.0f, true});
    p.render(y.data(), kHop);
    REQUIRE(p.activeCount() == 1);
    REQUIRE(rms(y.data(), 76) > 0.3);
    for (std::size_t n = 77; n < y.size(); ++n)
        REQUIRE(y[n] == 0.0f);
    // The next hop stays above Nyquist throughout, so the partial is dropped from the bank.
    p.beginHop({f.get(), f.get(), 30000.0f, 30000.0f, false});
    REQUIRE(p.activeCount() == 0);
}

TEST_CASE("partials: amplitude ramps linearly across a hop", "[dsp][partials]")
{
    // f0 * ratio = sr / 4: the sinusoid reads 0, 1, 0, -1, so sample 4q + 1 is the amplitude.
    auto silent = std::make_unique<VoiceFrame>();
    setPartial(*silent, 0, 1.0f, 0.0f);
    auto loud = std::make_unique<VoiceFrame>(*silent);
    setPartial(*loud, 0, 1.0f, 1.0f);

    PartialsPrimitive p;
    p.prepare(spec());
    p.reset();
    std::vector<float> y(kHop);
    p.beginHop({silent.get(), silent.get(), 12000.0f, 12000.0f, true});
    p.render(y.data(), kHop);
    REQUIRE(rms(y) == 0.0);
    p.beginHop({silent.get(), loud.get(), 12000.0f, 12000.0f, false});
    p.render(y.data(), 100);
    p.render(y.data() + 100, kHop - 100);
    for (int n = 1; n < kHop; n += 4)
    {
        INFO("sample " << n);
        REQUIRE(std::abs(y[static_cast<std::size_t>(n)] - static_cast<float>(n) / kHop) < 1e-5f);
    }
}

TEST_CASE("partials: skipHop advances phases like a rendered hop", "[dsp][partials]")
{
    auto a = std::make_unique<VoiceFrame>();
    auto b = std::make_unique<VoiceFrame>();
    setPartial(*a, 0, 1.0f, 0.5f);
    setPartial(*a, 3, 3.7f, 0.2f);
    setPartial(*b, 0, 1.1f, 0.5f);
    setPartial(*b, 3, 3.1f, 0.3f);
    PartialsPrimitive full, skipping;
    full.prepare(spec());
    skipping.prepare(spec());
    full.reset();
    skipping.reset();
    std::vector<float> y1(kHop), y2(kHop);
    const HopContext hops[] = {{a.get(), a.get(), 200.0f, 260.0f, true},
                               {a.get(), b.get(), 260.0f, 240.0f, false},
                               {b.get(), a.get(), 240.0f, 300.0f, false}};
    for (const auto& h : hops)
    {
        full.beginHop(h);
        full.render(y1.data(), kHop);
    }
    skipping.skipHop(hops[0]);
    skipping.skipHop(hops[1]);
    skipping.beginHop(hops[2]);
    skipping.render(y2.data(), kHop);
    double worst = 0.0;
    for (std::size_t n = 0; n < y1.size(); ++n)
        worst = std::max(worst, static_cast<double>(std::abs(y1[n] - y2[n])));
    REQUIRE(worst < 1e-5);
    REQUIRE(std::abs(full.phase(3) - skipping.phase(3)) < 1e-12);
}

TEST_CASE("partials: beginHop and render do not allocate", "[dsp][partials]")
{
    auto f = std::make_unique<VoiceFrame>();
    for (int m = 0; m < kMaxPartials; ++m)
        setPartial(*f, m, 1.0f + 0.37f * static_cast<float>(m), 0.01f);
    PartialsPrimitive p;
    p.prepare(spec());
    std::vector<float> y(kHop);
    long count = 0;
    {
        AllocationGuard guard;
        p.reset();
        for (int h = 0; h < 20; ++h)
        {
            const HopContext ctx{f.get(), f.get(), 100.0f, 120.0f, h == 0};
            if (h % 5 == 3)
            {
                p.skipHop(ctx);
                continue;
            }
            p.beginHop(ctx);
            p.render(y.data(), 70);
            p.render(y.data() + 70, kHop - 70);
        }
        count = guard.count();
    }
    REQUIRE(count == 0);
}

TEST_CASE("partials: benchmark 16 voices", "[dsp][partials][.bench]")
{
    constexpr int kVoices = 16;
    constexpr int kBlock = 128;
    const int total = static_cast<int>(kSr);

    auto fa = std::make_unique<VoiceFrame>();
    auto fb = std::make_unique<VoiceFrame>();
    for (int m = 0; m < kMaxPartials; ++m)
    {
        setPartial(*fa, m, 1.0f + 0.51f * static_cast<float>(m), 0.02f);
        setPartial(*fb, m, 1.0f + 0.53f * static_cast<float>(m), 0.01f);
    }
    std::vector<PartialsPrimitive> voices(kVoices);
    std::vector<int> pos(kVoices, 0), hopIndex(kVoices, 0);
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
        for (std::size_t v = 0; v < kVoices; ++v)
        {
            const float f0 = 110.0f + 13.0f * static_cast<float>(v);
            int done = 0;
            while (done < kBlock)
            {
                if (pos[v] == 0)
                {
                    const int j = hopIndex[v]++;
                    voices[v].beginHop(
                        {j % 2 ? fa.get() : fb.get(), j % 2 ? fb.get() : fa.get(), f0, f0 * 1.001f, j == 0});
                }
                const int n = std::min(kBlock - done, kHop - pos[v]);
                voices[v].render(buf.data() + done, n);
                done += n;
                pos[v] = (pos[v] + n) % kHop;
            }
            sink += buf[7];
        }
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    const int hops = total / kHop;
    std::printf(
        "PARTIALS: %d voices x 64 partials x 1 s: %.1f us total, %.2f us per %d-sample block (all voices), "
        "%.2f us per voice per hop, %.2f%% of one core per voice (sink %g)\n",
        kVoices, us, us / blocks, kBlock, us / (kVoices * hops), 100.0 * us / 1e6 / kVoices,
        static_cast<double>(sink));
    SUCCEED();
}
