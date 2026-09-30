#include "TestHelpers.h"
#include "alfa/dsp/Noise.h"
#include "alfa/dsp/Sampler.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{
constexpr double kSr = 48000.0;
constexpr int kHop = 192;

PrepareSpec spec48()
{
    PrepareSpec s;
    s.sampleRate = kSr;
    s.hop = hopSamples(kSr);
    return s;
}

template <typename Prim>
std::vector<float> renderHops(Prim& p, VoiceFrame& prev, VoiceFrame& cur, int hops, float f0 = 440.0f,
                              int chunk = 64)
{
    std::vector<float> out(static_cast<std::size_t>(hops * kHop));
    for (int j = 0; j < hops; ++j)
    {
        HopContext ctx;
        ctx.prev = j == 0 ? &cur : &prev;
        ctx.cur = &cur;
        ctx.f0Start = f0;
        ctx.f0End = f0;
        ctx.firstHop = j == 0;
        p.beginHop(ctx);
        for (int t = 0; t < kHop; t += chunk)
            p.render(out.data() + j * kHop + t, std::min(chunk, kHop - t));
    }
    return out;
}

std::vector<float> sine(double hz, double sr, long long n, float amp = 0.5f)
{
    std::vector<float> x(static_cast<std::size_t>(n));
    for (long long i = 0; i < n; ++i)
        x[static_cast<std::size_t>(i)] =
            amp *
            static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * hz * static_cast<double>(i) / sr));
    return x;
}
} // namespace

TEST_CASE("NOISE with flat bands has the expected level", "[dsp][noise]")
{
    NoisePrimitive noise;
    noise.prepare(spec48());
    noise.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->noise.band.fill(1.0f);
    const auto out = renderHops(noise, *frame, *frame, 500);
    // Uniform noise (variance 1/3) under two Hann windows at 50%: mean of w^2 + w'^2 is 3/4.
    REQUIRE(std::abs(rms(out) - 0.5) < 0.03);
}

TEST_CASE("NOISE silent bands are silent", "[dsp][noise]")
{
    NoisePrimitive noise;
    noise.prepare(spec48());
    noise.reset();
    auto frame = std::make_unique<VoiceFrame>();
    const auto out = renderHops(noise, *frame, *frame, 20);
    REQUIRE(rms(out) == 0.0);
}

TEST_CASE("NOISE follows the band curve", "[dsp][noise]")
{
    NoisePrimitive noise;
    noise.prepare(spec48());
    noise.reset();
    auto frame = std::make_unique<VoiceFrame>();
    // Only the upper third of the bands.
    for (int b = 44; b < kBands; ++b)
        frame->noise.band[static_cast<std::size_t>(b)] = 1.0f;
    const auto out = renderHops(noise, *frame, *frame, 400);
    double low = 0.0, high = 0.0;
    for (int k = 0; k < 20; ++k)
    {
        low += toneAmplitude(out, 200.0 + 37.0 * k, kSr);
        high += toneAmplitude(out, 12000.0 + 97.0 * k, kSr);
    }
    REQUIRE(high > 20.0 * low);
}

TEST_CASE("NOISE render is allocation-free", "[dsp][noise][realtime]")
{
    NoisePrimitive noise;
    noise.prepare(spec48());
    noise.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->noise.band.fill(0.5f);
    AllocationGuard guard;
    (void)renderHops(noise, *frame, *frame, 10);
    REQUIRE(guard.count() <= 1); // the output vector in renderHops
}

TEST_CASE("SAMPLER with a flat EQ reproduces the sample from the first sample", "[dsp][sampler]")
{
    const auto data = sine(440.0, kSr, 48000);
    SampleData sample{data.data(), static_cast<long long>(data.size()), kSr, 440.0, -1, -1};
    SamplerPrimitive sampler;
    sampler.prepare(spec48());
    sampler.setSample(&sample);
    sampler.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->sampler.rate = 1.0f;
    frame->sampler.gain = 1.0f;
    frame->sampler.eq.fill(1.0f);
    const auto out = renderHops(sampler, *frame, *frame, 50, 440.0f, 37);
    double worst = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::abs(out[i] - data[i])));
    REQUIRE(worst < 1e-5);
}

TEST_CASE("SAMPLER STFT path is transparent at unity gain and keeps the onset", "[dsp][sampler]")
{
    const auto data = sine(1000.0, kSr, 48000);
    SampleData sample{data.data(), static_cast<long long>(data.size()), kSr, 1000.0, -1, -1};
    SamplerPrimitive sampler;
    sampler.prepare(spec48());
    sampler.setSample(&sample);
    sampler.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->sampler.rate = 1.0f;
    frame->sampler.gain = 1.0f;
    frame->sampler.eq.fill(1.0f);
    frame->sampler.eq[0] = 0.9999f; // forces the FFT path
    const auto out = renderHops(sampler, *frame, *frame, 50, 1000.0f);
    double worst = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::abs(out[i] - data[i])));
    std::size_t at = 0;
    for (std::size_t i = 0; i < out.size(); ++i)
        if (std::abs(out[i] - data[i]) >= worst)
            at = i;
    INFO("worst at " << at);
    REQUIRE(worst < 2e-3);
}

TEST_CASE("SAMPLER EQ attenuates the bands it is told to", "[dsp][sampler]")
{
    const auto data = sine(1000.0, kSr, 48000);
    SampleData sample{data.data(), static_cast<long long>(data.size()), kSr, 1000.0, -1, -1};
    SamplerPrimitive sampler;
    sampler.prepare(spec48());
    sampler.setSample(&sample);
    sampler.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->sampler.rate = 1.0f;
    frame->sampler.gain = 1.0f;
    frame->sampler.eq.fill(0.1f);
    const auto out = renderHops(sampler, *frame, *frame, 100, 1000.0f);
    const std::vector<float> steady(out.begin() + 4 * kHop, out.end());
    REQUIRE(std::abs(toneAmplitude(steady, 1000.0, kSr) - 0.05) < 0.005);
}

TEST_CASE("SAMPLER transposes by f0 / root and rate", "[dsp][sampler]")
{
    const auto data = sine(440.0, kSr, 96000);
    SampleData sample{data.data(), static_cast<long long>(data.size()), kSr, 440.0, -1, -1};
    SamplerPrimitive sampler;
    sampler.prepare(spec48());
    sampler.setSample(&sample);
    sampler.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->sampler.rate = 1.5f;
    frame->sampler.gain = 1.0f;
    frame->sampler.eq.fill(1.0f);
    const auto out = renderHops(sampler, *frame, *frame, 100, 220.0f);
    REQUIRE(std::abs(zeroCrossingHz(out, kSr) - 330.0) < 0.5);
}

TEST_CASE("SAMPLER loops and stops at the end without a loop", "[dsp][sampler]")
{
    const auto data = sine(480.0, kSr, 4800); // 100 ms, whole cycles
    SampleData looped{data.data(), 4800, kSr, 480.0, 0, 4800};
    SampleData oneShot{data.data(), 4800, kSr, 480.0, -1, -1};
    auto frame = std::make_unique<VoiceFrame>();
    frame->sampler.rate = 1.0f;
    frame->sampler.gain = 1.0f;
    frame->sampler.eq.fill(1.0f);

    SamplerPrimitive sampler;
    sampler.prepare(spec48());
    sampler.setSample(&looped);
    sampler.reset();
    auto out = renderHops(sampler, *frame, *frame, 100, 480.0f);
    const std::vector<float> late(out.end() - 4800, out.end());
    REQUIRE(std::abs(rms(late) - 0.5 / std::sqrt(2.0)) < 0.01);

    sampler.setSample(&oneShot);
    sampler.reset();
    out = renderHops(sampler, *frame, *frame, 100, 480.0f);
    const std::vector<float> after(out.begin() + 5000, out.end());
    REQUIRE(rms(after) == 0.0);
}

TEST_CASE("SAMPLER start offset comes from the note-on frame", "[dsp][sampler]")
{
    std::vector<float> ramp(48000);
    for (std::size_t i = 0; i < ramp.size(); ++i)
        ramp[i] = static_cast<float>(i) / 48000.0f;
    SampleData sample{ramp.data(), 48000, kSr, 440.0, -1, -1};
    SamplerPrimitive sampler;
    sampler.prepare(spec48());
    sampler.setSample(&sample);
    sampler.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->sampler.rate = 1.0f;
    frame->sampler.gain = 1.0f;
    frame->sampler.start = 0.5f;
    frame->sampler.eq.fill(1.0f);
    const auto out = renderHops(sampler, *frame, *frame, 2);
    REQUIRE(std::abs(out[0] - 0.5f) < 1e-3f);
}

TEST_CASE("SAMPLER without a sample is silent and render is allocation-free", "[dsp][sampler][realtime]")
{
    SamplerPrimitive sampler;
    sampler.prepare(spec48());
    sampler.setSample(nullptr);
    sampler.reset();
    auto frame = std::make_unique<VoiceFrame>();
    frame->sampler.gain = 1.0f;
    frame->sampler.eq.fill(0.5f);
    AllocationGuard guard;
    const auto out = renderHops(sampler, *frame, *frame, 10);
    REQUIRE(guard.count() <= 1); // the output vector in renderHops
    REQUIRE(rms(out) == 0.0);
}
