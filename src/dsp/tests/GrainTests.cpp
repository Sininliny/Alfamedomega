#include "TestHelpers.h"
#include "alfa/dsp/Grain.h"

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

constexpr double kSr = 48000.0;
constexpr int kHop = 192;
constexpr int kLen = kGrainSourceLength;
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

/// Bank: 0 = DC 1.0, 1 = sine of 64 cycles per buffer (375 Hz at rate 1), 2 = sine of 128 cycles
/// (750 Hz), 3 = ramp -1..1. With H = 192, sine 1 at rate 2 and sine 2 at rate 1 advance a whole
/// number of cycles per hop, so successive grains are in phase and overlap-add coherently.
struct TestBank
{
    std::vector<float> data;
    GrainBank bank;

    TestBank()
        : data(u(4 * kLen))
    {
        for (int n = 0; n < kLen; ++n)
        {
            data[u(n)] = 1.0f;
            data[u(kLen + n)] = static_cast<float>(std::sin(2.0 * kPiD * 64.0 * n / kLen));
            data[u(2 * kLen + n)] = static_cast<float>(std::sin(2.0 * kPiD * 128.0 * n / kLen));
            data[u(3 * kLen + n)] = -1.0f + 2.0f * static_cast<float>(n) / kLen;
        }
        bank.data = data.data();
        bank.count = 4;
        bank.length = kLen;
    }
};

std::unique_ptr<VoiceFrame> layerFrame(int layer, float source, float rate, float gain = 1.0f,
                                       float position = 0.0f)
{
    auto f = std::make_unique<VoiceFrame>();
    f->grain.source[u(2 * layer)] = source;
    f->grain.weight[u(2 * layer)] = 1.0f;
    f->grain.rate[u(layer)] = rate;
    f->grain.gain[u(layer)] = gain;
    f->grain.position[u(layer)] = position;
    return f;
}

std::vector<float> renderConstant(GrainPrimitive& gr, const VoiceFrame& f, int hops, int chunk = kHop)
{
    std::vector<float> out(u(hops * kHop));
    gr.reset();
    for (int h = 0; h < hops; ++h)
    {
        gr.beginHop({&f, &f, 440.0f, 440.0f, h == 0});
        for (int done = 0; done < kHop;)
        {
            const int n = std::min(chunk, kHop - done);
            gr.render(out.data() + h * kHop + done, n);
            done += n;
        }
    }
    return out;
}

double maxAbsFrom(const std::vector<float>& x, std::size_t from)
{
    double m = 0.0;
    for (std::size_t n = from; n < x.size(); ++n)
        m = std::max(m, static_cast<double>(std::abs(x[n])));
    return m;
}

} // namespace

TEST_CASE("GRAIN on a DC source overlap-adds to exactly 1", "[dsp][grain]")
{
    TestBank tb;
    GrainPrimitive gr;
    gr.prepare(spec());
    gr.setBank(&tb.bank);
    auto f = layerFrame(0, 0.0f, 1.0f);
    const auto out = renderConstant(gr, *f, 20, 50);
    // The first hop fades in with the first grain's rising half-window.
    REQUIRE(out[0] == 0.0f);
    for (int n = kHop; n < 20 * kHop; ++n)
        REQUIRE_THAT(out[u(n)], WithinAbs(1.0, 1e-5));
    REQUIRE(gr.liveGrains() == 1); // the older grain ended with the hop
}

TEST_CASE("GRAIN rate 2 doubles the source frequency", "[dsp][grain]")
{
    TestBank tb;
    GrainPrimitive gr;
    gr.prepare(spec());
    gr.setBank(&tb.bank);

    auto f1 = layerFrame(0, 1.0f, 2.0f);
    auto out = renderConstant(gr, *f1, 250);
    out.erase(out.begin(), out.begin() + kHop);
    REQUIRE_THAT(toneAmplitude(out, 750.0, kSr), WithinAbs(1.0, 0.01));
    REQUIRE(toneAmplitude(out, 375.0, kSr) < 0.01);
    REQUIRE_THAT(zeroCrossingHz(out, kSr), WithinAbs(750.0, 0.5));

    // And rate 1 on the 750 Hz source gives 750 Hz.
    auto f2 = layerFrame(0, 2.0f, 1.0f);
    auto ref = renderConstant(gr, *f2, 250);
    ref.erase(ref.begin(), ref.begin() + kHop);
    REQUIRE_THAT(toneAmplitude(ref, 750.0, kSr), WithinAbs(1.0, 0.01));
}

TEST_CASE("GRAIN plays backwards with a negative rate and wraps", "[dsp][grain]")
{
    TestBank tb;
    GrainPrimitive gr;
    gr.prepare(spec());
    gr.setBank(&tb.bank);
    // Ramp source read backwards from position 0: first read is -1 at index 0, then wraps to the
    // end of the buffer (about +1) and descends.
    auto f = layerFrame(0, 3.0f, -1.0f);
    const auto out = renderConstant(gr, *f, 1);
    const auto& w = tb.data;
    for (int n = 1; n < kHop; ++n)
    {
        const double win = 0.5 - 0.5 * std::cos(2.0 * kPiD * n / (2.0 * kHop));
        REQUIRE_THAT(out[u(n)], WithinAbs(win * w[u(3 * kLen + kLen - n)], 1e-5));
    }
}

TEST_CASE("GRAIN mixes two sources linearly by weight", "[dsp][grain]")
{
    TestBank tb;
    GrainPrimitive gr;
    gr.prepare(spec());
    gr.setBank(&tb.bank);

    auto mixFrame = [](float w0, float w1)
    {
        auto f = std::make_unique<VoiceFrame>();
        f->grain.source[2] = 1.0f; // layer 1
        f->grain.source[3] = 2.6f; // rounds to 3
        f->grain.weight[2] = w0;
        f->grain.weight[3] = w1;
        f->grain.rate[1] = 1.3f;
        f->grain.gain[1] = 0.8f;
        f->grain.position[1] = 0.25f;
        return f;
    };
    const auto a = renderConstant(gr, *mixFrame(1.0f, 0.0f), 10);
    const auto b = renderConstant(gr, *mixFrame(0.0f, 1.0f), 10);
    const auto m = renderConstant(gr, *mixFrame(0.3f, 0.7f), 10);
    REQUIRE(maxAbsFrom(a, 0) > 0.1);
    REQUIRE(maxAbsFrom(b, 0) > 0.1);
    for (std::size_t n = 0; n < m.size(); ++n)
        REQUIRE_THAT(m[n], WithinAbs(0.3 * a[n] + 0.7 * b[n], 1e-5));
}

TEST_CASE("GRAIN is silent for out-of-range sources and a null bank", "[dsp][grain]")
{
    TestBank tb;
    GrainPrimitive gr;
    gr.prepare(spec());

    SECTION("out-of-range index")
    {
        gr.setBank(&tb.bank);
        for (const float s : {4.0f, 3.6f, -1.0f, 100.0f})
        {
            const auto out = renderConstant(gr, *layerFrame(0, s, 1.0f), 4);
            REQUIRE(maxAbsFrom(out, 0) == 0.0);
        }
        // -0.4 rounds to 0, which is valid.
        REQUIRE(maxAbsFrom(renderConstant(gr, *layerFrame(0, -0.4f, 1.0f), 4), 0) > 0.5);
    }
    SECTION("null bank")
    {
        gr.setBank(nullptr);
        const auto out = renderConstant(gr, *layerFrame(0, 0.0f, 1.0f), 4);
        REQUIRE(maxAbsFrom(out, 0) == 0.0);
        REQUIRE(gr.liveGrains() == 0);
    }
    SECTION("zero weight or gain")
    {
        gr.setBank(&tb.bank);
        REQUIRE(maxAbsFrom(renderConstant(gr, *layerFrame(0, 0.0f, 1.0f, 0.0f), 4), 0) == 0.0);
        auto f = layerFrame(0, 0.0f, 1.0f);
        f->grain.weight[0] = 0.0f;
        REQUIRE(maxAbsFrom(renderConstant(gr, *f, 4), 0) == 0.0);
    }
}

TEST_CASE("GRAIN honours the tier layer limit", "[dsp][grain]")
{
    TestBank tb;
    auto f = layerFrame(2, 0.0f, 1.0f); // only layer 2 is active

    GrainPrimitive lite;
    lite.prepare(spec(Tier::Lite));
    lite.setBank(&tb.bank);
    REQUIRE(maxAbsFrom(renderConstant(lite, *f, 4), 0) == 0.0);

    GrainPrimitive standard;
    standard.prepare(spec(Tier::Standard));
    standard.setBank(&tb.bank);
    const auto out = renderConstant(standard, *f, 4);
    REQUIRE_THAT(out[u(3 * kHop)], WithinAbs(1.0, 1e-5));
}

TEST_CASE("GRAIN holds a grain's values for its whole life", "[dsp][grain]")
{
    TestBank tb;
    GrainPrimitive gr;
    gr.prepare(spec());
    gr.setBank(&tb.bank);
    auto loud = layerFrame(0, 0.0f, 1.0f, 1.0f);
    auto quiet = layerFrame(0, 0.0f, 1.0f, 0.5f);
    std::vector<float> out(u(kHop));
    gr.reset();
    gr.beginHop({loud.get(), loud.get(), 440.0f, 440.0f, true});
    gr.render(out.data(), kHop);
    gr.beginHop({loud.get(), quiet.get(), 440.0f, 440.0f, false});
    gr.render(out.data(), kHop);
    // Old grain (gain 1) falling half + new grain (gain 0.5) rising half.
    for (int n = 0; n < kHop; ++n)
    {
        const double rise = 0.5 - 0.5 * std::cos(kPiD * n / kHop);
        REQUIRE_THAT(out[u(n)], WithinAbs((1.0 - rise) + 0.5 * rise, 1e-5));
    }
}

TEST_CASE("GRAIN skipHop drops grains and does not allocate", "[dsp][grain]")
{
    TestBank tb;
    GrainPrimitive gr;
    gr.prepare(spec());
    gr.setBank(&tb.bank);
    auto f = std::make_unique<VoiceFrame>();
    for (int l = 0; l < kMaxGrainLayers; ++l)
    {
        f->grain.source[u(2 * l)] = static_cast<float>(l % 4);
        f->grain.source[u(2 * l + 1)] = static_cast<float>((l + 1) % 4);
        f->grain.weight[u(2 * l)] = 0.6f;
        f->grain.weight[u(2 * l + 1)] = 0.4f;
        f->grain.rate[u(l)] = 0.5f + 0.25f * static_cast<float>(l);
        f->grain.gain[u(l)] = 0.1f;
        f->grain.position[u(l)] = 0.1f * static_cast<float>(l);
    }
    std::vector<float> out(u(kHop));
    AllocationGuard guard;
    gr.reset();
    for (int h = 0; h < 10; ++h)
    {
        const HopContext ctx{f.get(), f.get(), 440.0f, 440.0f, h == 0};
        if (h == 5)
        {
            gr.skipHop(ctx);
            REQUIRE(gr.liveGrains() == 0);
            continue;
        }
        gr.beginHop(ctx);
        gr.render(out.data(), 100);
        gr.render(out.data() + 100, kHop - 100);
    }
    REQUIRE(guard.count() == 0);
    REQUIRE(gr.liveGrains() == kMaxGrainLayers); // older grains ended with the hop
}

TEST_CASE("GRAIN cost, 16 voices x 1 s, all Studio layers", "[.bench][dsp][grain]")
{
    constexpr int kVoices = 16;
    const int hops = static_cast<int>(kSr) / kHop;
    TestBank tb;
    auto f = std::make_unique<VoiceFrame>();
    for (int l = 0; l < kMaxGrainLayers; ++l)
    {
        f->grain.source[u(2 * l)] = static_cast<float>(l % 4);
        f->grain.source[u(2 * l + 1)] = static_cast<float>((l + 1) % 4);
        f->grain.weight[u(2 * l)] = 0.6f;
        f->grain.weight[u(2 * l + 1)] = 0.4f;
        f->grain.rate[u(l)] = 0.5f + 0.37f * static_cast<float>(l);
        f->grain.gain[u(l)] = 0.1f;
        f->grain.position[u(l)] = 0.1f * static_cast<float>(l);
    }
    std::vector<GrainPrimitive> voices(kVoices);
    for (auto& v : voices)
    {
        v.prepare(spec(Tier::Studio));
        v.setBank(&tb.bank);
        v.reset();
    }
    std::vector<float> out(u(kHop));
    float sink = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    for (int h = 0; h < hops; ++h)
        for (auto& v : voices)
        {
            v.beginHop({f.get(), f.get(), 440.0f, 440.0f, h == 0});
            v.render(out.data(), kHop);
            sink += out[0];
        }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    const double perBlockAll = us / (static_cast<double>(hops) * kHop / 128.0);
    std::printf("GRAIN (8 layers x 2 sources): %.2f us per 128-sample block for %d voices, %.3f us per voice "
                "(sink %g)\n",
                perBlockAll, kVoices, perBlockAll / kVoices, static_cast<double>(sink));
    REQUIRE(voices[0].liveGrains() == kMaxGrainLayers);
}
