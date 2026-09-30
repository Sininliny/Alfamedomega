#include "TestHelpers.h"
#include "alfa/dsp/Osc.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{
std::vector<float> renderOsc(OscPrimitive& osc, const VoiceFrame& frame, int hops, float f0)
{
    constexpr int kHop = 192;
    std::vector<float> out(static_cast<std::size_t>(hops * kHop));
    for (int j = 0; j < hops; ++j)
    {
        HopContext ctx;
        ctx.prev = &frame;
        ctx.cur = &frame;
        ctx.f0Start = f0;
        ctx.f0End = f0;
        ctx.firstHop = j == 0;
        osc.beginHop(ctx);
        osc.render(out.data() + j * kHop, kHop);
    }
    return out;
}
} // namespace

TEST_CASE("OSC detuned read adds a second copy at f0 * ratio", "[dsp][osc]")
{
    OscPrimitive osc;
    osc.prepare(PrepareSpec{});
    osc.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->osc.re[0] = 1.0f;
    frame->oscDetuneRatio[0] = 1.5f;
    frame->oscDetuneLevel[0] = 0.5f;
    const auto out = renderOsc(osc, *frame, 250, 440.0f);
    REQUIRE(std::abs(toneAmplitude(out, 440.0, 48000.0) - 1.0) < 0.01);
    REQUIRE(std::abs(toneAmplitude(out, 660.0, 48000.0) - 0.5) < 0.01);
}

TEST_CASE("OSC detune at level 0 changes nothing", "[dsp][osc]")
{
    OscPrimitive a, b;
    a.prepare(PrepareSpec{});
    b.prepare(PrepareSpec{});
    a.reset();
    b.reset();
    auto plain = std::make_unique<VoiceFrame>();
    plain->osc.re[0] = 1.0f;
    plain->osc.re[3] = 0.25f;
    auto detuned = std::make_unique<VoiceFrame>(*plain);
    detuned->oscDetuneRatio[0] = 3.0f; // must not tighten the Nyquist cut while silent
    const auto x = renderOsc(a, *plain, 20, 5500.0f);
    const auto y = renderOsc(b, *detuned, 20, 5500.0f);
    REQUIRE(x == y);
}

TEST_CASE("OSC detune tightens the Nyquist cut to the higher read", "[dsp][osc]")
{
    OscPrimitive osc;
    osc.prepare(PrepareSpec{});
    osc.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->osc.re[2] = 1.0f; // harmonic 3 at 5 kHz: 15 kHz, detuned x2 would be 30 kHz
    frame->oscDetuneRatio[0] = 2.0f;
    frame->oscDetuneLevel[0] = 1.0f;
    const auto out = renderOsc(osc, *frame, 50, 5000.0f);
    REQUIRE(rms(out) < 1e-6);
}

TEST_CASE("OSC unison reads are limited by the tier", "[dsp][osc]")
{
    auto frame = std::make_unique<VoiceFrame>();
    frame->osc.re[0] = 1.0f;
    for (int r = 0; r < kMaxOscDetune; ++r)
    {
        frame->oscDetuneRatio[static_cast<std::size_t>(r)] = 1.0f + 0.25f * static_cast<float>(r + 1);
        frame->oscDetuneLevel[static_cast<std::size_t>(r)] = 0.5f;
    }
    for (const Tier tier : {Tier::Lite, Tier::Standard, Tier::Pro, Tier::Studio})
    {
        PrepareSpec spec;
        spec.limits = limitsFor(tier);
        OscPrimitive osc;
        osc.prepare(spec);
        osc.reset();
        const auto out = renderOsc(osc, *frame, 250, 440.0f);
        for (int r = 0; r < kMaxOscDetune; ++r)
        {
            const double hz = 440.0 * (1.0 + 0.25 * (r + 1));
            const double expected = r < spec.limits.oscUnison - 1 ? 0.5 : 0.0;
            INFO("tier " << static_cast<int>(tier) << " read " << r);
            REQUIRE(std::abs(toneAmplitude(out, hz, 48000.0) - expected) < 0.02);
        }
    }
}
