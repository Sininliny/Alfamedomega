#include "TestHelpers.h"
#include "alfa/dsp/Synth.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{
constexpr double kSr = 48000.0;

std::unique_ptr<VoiceFrame> sineFrame(float tau = 1.0e-4f)
{
    auto f = std::make_unique<VoiceFrame>();
    f->osc.re[0] = 1.0f; // harmonic 1, cosine
    f->setMix(Primitive::Osc, 1.0f);
    f->attackTau = tau;
    return f;
}

std::vector<float> run(Synth& synth, int samples, int block = 128)
{
    std::vector<float> out(static_cast<std::size_t>(samples));
    std::vector<float> right(static_cast<std::size_t>(block));
    for (int i = 0; i < samples; i += block)
    {
        const int n = std::min(block, samples - i);
        synth.process(out.data() + i, right.data(), n);
    }
    return out;
}

/// A frame with every primitive active at the tier's full size.
std::unique_ptr<VoiceFrame> everythingFrame()
{
    auto f = std::make_unique<VoiceFrame>();
    for (int k = 0; k < 64; ++k)
    {
        f->osc.re[static_cast<std::size_t>(k)] = 1.0f / static_cast<float>(k + 1);
        f->osc.im[static_cast<std::size_t>(k)] = 0.3f / static_cast<float>(k + 1);
    }
    for (int m = 0; m < kMaxPartials; ++m)
    {
        f->partials.ratio[static_cast<std::size_t>(m)] = 1.0f + 1.07f * static_cast<float>(m);
        f->partials.amp[static_cast<std::size_t>(m)] = 0.02f;
    }
    for (int i = 0; i < kMaxFmOperators; ++i)
    {
        f->fm.ratio[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        f->fm.level[static_cast<std::size_t>(i)] = 0.1f;
        for (int j = i + 1; j < kMaxFmOperators; ++j)
            f->fm.index[static_cast<std::size_t>(i * kMaxFmOperators + j)] = 0.5f;
    }
    f->noise.band.fill(0.1f);
    for (int l = 0; l < kMaxGrainLayers; ++l)
    {
        const auto i = static_cast<std::size_t>(l);
        f->grain.source[2 * i] = static_cast<float>(l % 4);
        f->grain.source[2 * i + 1] = static_cast<float>((l + 1) % 4);
        f->grain.weight[2 * i] = 0.7f;
        f->grain.weight[2 * i + 1] = 0.3f;
        f->grain.position[i] = 0.1f * static_cast<float>(l);
        f->grain.rate[i] = 1.0f;
        f->grain.gain[i] = 0.1f;
    }
    f->sampler.rate = 1.0f;
    f->sampler.gain = 0.5f;
    f->sampler.eq.fill(0.8f);
    f->mix.fill(0.2f);
    f->attackTau = 0.002f;
    return f;
}
} // namespace

TEST_CASE("a note renders the OSC fundamental at the note's pitch", "[dsp][synth]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = sineFrame();
    REQUIRE(synth.noteOn(69, 1.0f, frame.get()) >= 0);
    const auto out = run(synth, 24000);
    REQUIRE(std::abs(zeroCrossingHz(out, kSr) - 440.0) < 0.1);
    REQUIRE(std::abs(toneAmplitude(out, 440.0, kSr) - 1.0) < 0.01);
}

TEST_CASE("the attack envelope is 1 - exp(-t / tau) from the note-on sample", "[dsp][synth]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = std::make_unique<VoiceFrame>();
    // DC-free constant is awkward with OSC; use a high partial and read the envelope from peaks.
    frame->osc.re[0] = 1.0f;
    frame->setMix(Primitive::Osc, 1.0f);
    frame->attackTau = 0.01f;
    synth.noteOn(69, 1.0f, frame.get());
    const auto out = run(synth, 4800);
    // Peak within one period around t = tau (480 samples) and t = 3 tau.
    const auto peakAround = [&](int centre)
    {
        float p = 0.0f;
        for (int i = centre - 55; i < centre + 55; ++i)
            p = std::max(p, std::abs(out[static_cast<std::size_t>(i)]));
        return p;
    };
    REQUIRE(std::abs(peakAround(480) - (1.0 - std::exp(-1.0))) < 0.05);
    REQUIRE(std::abs(peakAround(1440) - (1.0 - std::exp(-3.0))) < 0.05);
    REQUIRE(std::abs(out[0]) < 1e-3f); // starts from silence
}

TEST_CASE("a voice waits for its first frame", "[dsp][synth]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Lite);
    const int v = synth.noteOn(60, 1.0f, nullptr);
    REQUIRE(rms(run(synth, 1024)) == 0.0);
    auto frame = sineFrame();
    synth.setFrame(v, *frame);
    REQUIRE(rms(run(synth, 4096)) > 0.5);
}

TEST_CASE("a released voice ends once the network's mix gains reach zero", "[dsp][synth]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = sineFrame();
    const int v = synth.noteOn(60, 1.0f, frame.get());
    (void)run(synth, 4800);
    synth.noteOff(60);
    REQUIRE(synth.voiceInfo(v).releasing);
    (void)run(synth, 4800);
    REQUIRE(synth.voiceInfo(v).active); // the network is still sounding it
    frame->setMix(Primitive::Osc, 0.0f);
    synth.setFrame(v, *frame);
    const auto tail = run(synth, 4800);
    REQUIRE_FALSE(synth.voiceInfo(v).active);
    // The last hop ramps the gain down rather than cutting.
    float maxStep = 0.0f;
    for (std::size_t i = 1; i < tail.size(); ++i)
        maxStep = std::max(maxStep, std::abs(tail[i] - tail[i - 1]));
    REQUIRE(maxStep < 0.1f);
}

TEST_CASE("pitch bend moves every voice", "[dsp][synth]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = sineFrame();
    synth.noteOn(69, 1.0f, frame.get());
    synth.setPitchBend(12.0f);
    const auto out = run(synth, 24000);
    const std::vector<float> late(out.begin() + 4800, out.end());
    REQUIRE(std::abs(zeroCrossingHz(late, kSr) - 880.0) < 0.2);
}

TEST_CASE("voice stealing takes the oldest voice when the pool is full", "[dsp][synth]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Studio); // 8 voices
    auto frame = sineFrame();
    for (int n = 0; n < 8; ++n)
        synth.noteOn(60 + n, 1.0f, frame.get());
    REQUIRE(synth.voiceInfo(0).note == 60);
    const int v = synth.noteOn(80, 1.0f, frame.get());
    REQUIRE(v == 0);
    synth.noteOff(61);
    const int w = synth.noteOn(81, 1.0f, frame.get());
    REQUIRE(synth.voiceInfo(w).note == 81);
    REQUIRE(w == 1); // released voices go first
}

TEST_CASE("the full audio path is allocation-free with every primitive active", "[dsp][synth][realtime]")
{
    for (const Tier tier : {Tier::Lite, Tier::Standard, Tier::Pro, Tier::Studio})
    {
        Synth synth;
        synth.prepare(kSr, 512, tier);

        std::vector<float> bankData(4 * kGrainSourceLength);
        for (std::size_t i = 0; i < bankData.size(); ++i)
            bankData[i] = std::sin(0.01f * static_cast<float>(i));
        GrainBank bank{bankData.data(), 4, kGrainSourceLength};
        std::vector<float> sampleData(48000);
        for (std::size_t i = 0; i < sampleData.size(); ++i)
            sampleData[i] = std::sin(0.05f * static_cast<float>(i));
        SampleData sample{sampleData.data(), 48000, kSr, 261.63, 1000, 40000};
        synth.setGrainBank(&bank);
        synth.setSample(&sample);

        auto frame = everythingFrame();
        std::vector<float> left(512), right(512);
        {
            AllocationGuard guard;
            for (int v = 0; v < synth.voiceCount(); ++v)
                synth.noteOn(48 + 3 * v, 0.8f, frame.get());
            for (int block = 0; block < 400; ++block)
            {
                const int n = 1 + (block * 37) % 512; // odd sizes cross hop boundaries anywhere
                for (int v = 0; v < synth.voiceCount(); ++v)
                    synth.setFrame(v, *frame);
                synth.process(left.data(), right.data(), n);
                if (block == 200)
                    synth.allNotesOff();
            }
            REQUIRE(guard.count() == 0);
        }
        REQUIRE(std::isfinite(left[0]));
    }
}
