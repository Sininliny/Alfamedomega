// Whole-path tests: primitives, voice slots, modulation, pan and the instance chain together.

#include "TestHelpers.h"
#include "alfa/dsp/Family.h"
#include "alfa/dsp/Synth.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using namespace alfa::dsp;
using alfa::dsp::test::AllocationGuard;
using alfa::dsp::test::rms;

namespace
{

constexpr double kSr = 48000.0;

std::unique_ptr<VoiceFrame> sawFrame()
{
    auto f = std::make_unique<VoiceFrame>();
    for (int h = 0; h < 16; ++h)
        f->osc.re[static_cast<std::size_t>(h)] = 0.5f / static_cast<float>(h + 1);
    f->setMix(Primitive::Osc, 1.0f);
    f->attackTau = 1.0e-4f;
    return f;
}

struct Stereo
{
    std::vector<float> left, right;
};

Stereo run(Synth& synth, int samples, int block = 128)
{
    Stereo out{std::vector<float>(static_cast<std::size_t>(samples)),
               std::vector<float>(static_cast<std::size_t>(samples))};
    for (int i = 0; i < samples; i += block)
    {
        const int n = std::min(block, samples - i);
        synth.process(out.left.data() + i, out.right.data() + i, n);
    }
    return out;
}

bool finiteAndBounded(const std::vector<float>& x, float bound)
{
    return std::all_of(x.begin(), x.end(),
                       [bound](float v) { return std::isfinite(v) && std::fabs(v) <= bound; });
}

bool available(Family f)
{
    return makeProcessor(f) != nullptr;
}

} // namespace

TEST_CASE("a voice slot processes the primitives it is routed", "[dsp][integration]")
{
    if (!available(Family::Filter))
        SKIP("FILTER family not merged");
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = sawFrame();
    frame->dryGain = 0.0f;
    auto& slot = frame->slots[0];
    slot.family = static_cast<float>(Family::Filter);
    slot.outGain = 1.0f;
    slot.input[static_cast<std::size_t>(Primitive::Osc)] = 1.0f;
    for (int i = 0; i < kSlotParams; ++i)
        slot.params[static_cast<std::size_t>(i)] =
            paramTable(Family::Filter)[static_cast<std::size_t>(i)].defaultNorm;

    const int v = synth.noteOn(57, 1.0f, frame.get());
    const auto open = run(synth, 9600);
    REQUIRE(rms(open.left) > 0.05);

    // Unrouted slot: dry is 0 and the slot hears nothing, so the voice goes silent.
    slot.input.fill(0.0f);
    synth.setFrame(v, *frame);
    (void)run(synth, 9600); // the filter rings out
    const auto closed = run(synth, 4800);
    REQUIRE(rms(closed.left) < 1.0e-4);
}

TEST_CASE("pan and instance width shape the stereo image", "[dsp][integration]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = sawFrame();
    frame->pan = -1.0f;
    synth.noteOn(60, 1.0f, frame.get());
    const auto hardLeft = run(synth, 9600);
    REQUIRE(rms(hardLeft.left) > 0.1);
    REQUIRE(rms(hardLeft.right) < 1.0e-4);

    InstanceFrame inst;
    inst.width = 0.0f; // mono
    synth.setInstanceFrame(inst);
    const auto mono = run(synth, 9600);
    const std::vector<float> l(mono.left.begin() + 480, mono.left.end());
    const std::vector<float> r(mono.right.begin() + 480, mono.right.end());
    float maxDiff = 0.0f;
    for (std::size_t i = 0; i < l.size(); ++i)
        maxDiff = std::max(maxDiff, std::fabs(l[i] - r[i]));
    REQUIRE(maxDiff < 1.0e-6f);
}

TEST_CASE("an LFO route moves the pan of a voice", "[dsp][integration]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = sawFrame();
    // LFO 1 at about 2 Hz (rates are log 0.01..100 Hz), sine, onto pan at full depth.
    frame->mod.sources[12] = static_cast<float>(std::log(2.0 / 0.01) / std::log(100.0 / 0.01));
    frame->mod.sources[13] = 0.0f;
    frame->mod.routeSource[0] = static_cast<float>(ModSource::Lfo1);
    frame->mod.routeDestination[0] = static_cast<float>(moddest::pan);
    frame->mod.routeDepth[0] = 1.0f;
    synth.noteOn(60, 1.0f, frame.get());
    const auto out = run(synth, 48000);

    // Balance in 20 ms windows swings both ways.
    double minBal = 1.0, maxBal = -1.0;
    for (std::size_t w = 0; w + 960 <= out.left.size(); w += 960)
    {
        const double l = rms(out.left.data() + w, 960);
        const double r = rms(out.right.data() + w, 960);
        const double bal = (r - l) / (r + l + 1.0e-12);
        minBal = std::min(minBal, bal);
        maxBal = std::max(maxBal, bal);
    }
    REQUIRE(minBal < -0.8);
    REQUIRE(maxBal > 0.8);
}

TEST_CASE("an instance slot processes the voice bus", "[dsp][integration]")
{
    if (!available(Family::Nonlinear))
        SKIP("NONLINEAR family not merged");
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = sawFrame();
    synth.noteOn(60, 1.0f, frame.get());

    InstanceFrame inst;
    inst.dryGain = 0.0f;
    auto& slot = inst.slots[0];
    slot.family = static_cast<float>(Family::Nonlinear);
    slot.outGain = 1.0f;
    slot.input[0] = 1.0f;
    for (int i = 0; i < kSlotParams; ++i)
        slot.params[static_cast<std::size_t>(i)] =
            paramTable(Family::Nonlinear)[static_cast<std::size_t>(i)].defaultNorm;
    synth.setInstanceFrame(inst);
    const auto wet = run(synth, 9600);
    REQUIRE(rms(wet.left) > 0.05);

    slot.outGain = 0.0f;
    synth.setInstanceFrame(inst);
    (void)run(synth, 960);
    const auto muted = run(synth, 4800);
    REQUIRE(rms(muted.left) < 1.0e-6);
}

TEST_CASE("random family switching and params on every slot stay finite and allocation-free",
          "[dsp][integration]")
{
    for (const double sr : {44100.0, 48000.0, 96000.0})
    {
        Synth synth;
        synth.prepare(sr, 256, Tier::Pro);
        auto frame = sawFrame();
        frame->setMix(Primitive::Noise, 0.2f);
        InstanceFrame inst;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::uniform_int_distribution<int> voiceFamily(0, static_cast<int>(Family::Body));
        std::uniform_int_distribution<int> anyFamily(0, kFamilyCount - 1);

        const auto randomise = [&](SlotControls& s, int family)
        {
            s.family = static_cast<float>(family);
            s.outGain = u(rng);
            for (auto& p : s.params)
                p = u(rng);
            for (auto& w : s.input)
                w = u(rng);
        };

        int voices[3];
        for (int n = 0; n < 3; ++n)
            voices[n] = synth.noteOn(48 + 7 * n, 1.0f, frame.get());

        std::vector<float> l(256), r(256);
        bool ok = true;
        long allocations = 0;
        float peak = 0.0f;
        for (int b = 0; b < 400; ++b)
        {
            for (auto& s : frame->slots)
                randomise(s, voiceFamily(rng));
            for (auto& s : inst.slots)
                randomise(s, anyFamily(rng));
            frame->mod.routeSource[0] = static_cast<float>(ModSource::Lfo1);
            frame->mod.routeDestination[0] = u(rng) * 63.0f;
            frame->mod.routeDepth[0] = 1.0f;
            // Every 50 blocks, poison the frame.
            if (b % 50 == 49)
                frame->slots[1].params[3] = std::numeric_limits<float>::quiet_NaN();
            {
                AllocationGuard guard;
                for (const int v : voices)
                    synth.setFrame(v, *frame);
                synth.setInstanceFrame(inst);
                synth.process(l.data(), r.data(), 256);
                allocations += guard.count();
            }
            for (std::size_t i = 0; i < l.size(); ++i)
                peak = std::max({peak, std::isfinite(l[i]) ? std::fabs(l[i]) : 1e30f,
                                 std::isfinite(r[i]) ? std::fabs(r[i]) : 1e30f});
            ok = ok && finiteAndBounded(l, 4.0f) && finiteAndBounded(r, 4.0f);
        }
        INFO("sample rate " << sr << " peak " << peak);
        REQUIRE(ok);
        REQUIRE(allocations == 0);
    }
}

TEST_CASE("NaN primitive mix and pan do not reach the output", "[dsp][integration]")
{
    Synth synth;
    synth.prepare(kSr, 128, Tier::Standard);
    auto frame = sawFrame();
    const int v = synth.noteOn(60, 1.0f, frame.get());
    (void)run(synth, 1920);
    frame->pan = std::numeric_limits<float>::quiet_NaN();
    frame->dryGain = std::numeric_limits<float>::infinity();
    synth.setFrame(v, *frame);
    const auto out = run(synth, 1920);
    REQUIRE(finiteAndBounded(out.left, 100.0f));
    REQUIRE(finiteAndBounded(out.right, 100.0f));
}
