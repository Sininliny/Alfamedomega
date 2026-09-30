// The audio budget guard: estimates, victim order, bypass, determinism and real time safety.

#include "TestHelpers.h"
#include "alfa/dsp/BudgetGuard.h"
#include "alfa/dsp/Family.h"
#include "alfa/dsp/Synth.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using namespace alfa::dsp;
using alfa::dsp::test::AllocationGuard;

namespace
{

std::unique_ptr<VoiceFrame> heavyFrame(Family slotFamily, int slots)
{
    auto f = std::make_unique<VoiceFrame>();
    for (int k = 0; k < kOscHarmonics; ++k)
        f->osc.re[static_cast<std::size_t>(k)] = 0.3f / static_cast<float>(k + 1);
    for (int m = 0; m < kMaxPartials; ++m)
    {
        f->partials.ratio[static_cast<std::size_t>(m)] = 1.0f + static_cast<float>(m);
        f->partials.amp[static_cast<std::size_t>(m)] = 0.01f;
    }
    f->noise.band.fill(0.01f);
    for (int p : {0, 1, 2, 3})
        f->setMix(static_cast<Primitive>(p), 0.2f);
    f->attackTau = 1.0e-3f;
    for (int k = 0; k < slots; ++k)
    {
        auto& s = f->slots[static_cast<std::size_t>(k)];
        s.family = static_cast<float>(slotFamily);
        s.outGain = 0.3f;
        s.input[0] = 1.0f;
        const auto& t = paramTable(slotFamily);
        for (int i = 0; i < kSlotParams; ++i)
            s.params[static_cast<std::size_t>(i)] = t[static_cast<std::size_t>(i)].defaultNorm;
    }
    return f;
}

void play(Synth& synth, const VoiceFrame& frame, int notes)
{
    for (int n = 0; n < notes; ++n)
        synth.noteOn(36 + 3 * n, 0.8f, &frame);
}

void run(Synth& synth, int samples)
{
    std::vector<float> l(256), r(256);
    for (int i = 0; i < samples; i += 256)
        synth.process(l.data(), r.data(), 256);
}

} // namespace

TEST_CASE("the guard reads the tier spec's budget and stays out of light patches", "[dsp][guard]")
{
    Synth synth;
    synth.prepare(48000.0, 256, Tier::Standard);
    auto frame = heavyFrame(Family::Filter, 1);
    play(synth, *frame, 2);
    run(synth, 4800);
    const auto& r = synth.guardReport();
    REQUIRE(r.budgetShare == budgetGuardFor(Tier::Standard).maxShare);
    REQUIRE(r.estimatedShare > 0.0f);
    REQUIRE(r.estimatedShare < r.budgetShare);
    REQUIRE(r.downgraded == 0);
    REQUIRE(r.bypassed == 0);
}

TEST_CASE("a Pro worst case is brought under budget, heaviest slots first", "[dsp][guard]")
{
    for (const double sr : {48000.0, 96000.0})
    {
        Synth synth;
        synth.prepare(sr, 256, Tier::Pro);
        auto frame = heavyFrame(Family::Nonlinear, 4);
        frame->slots[3].family = static_cast<float>(Family::Filter); // lighter, should survive longest
        play(synth, *frame, limitsFor(Tier::Pro).polyphony);
        run(synth, static_cast<int>(sr / 4));
        long allocations = 0;
        {
            std::vector<float> l(256), r(256);
            AllocationGuard guard;
            for (int b = 0; b < 100; ++b)
                synth.process(l.data(), r.data(), 256);
            allocations = guard.count();
        }
        const auto& r = synth.guardReport();
        INFO("sr " << sr << " estimated " << r.estimatedShare << " final " << r.finalShare);
        REQUIRE(allocations == 0);
        REQUIRE(r.estimatedShare > r.budgetShare);
        REQUIRE(r.downgraded + r.bypassed > 0);
        // Only voice slots are the guard's to degrade: when the primitives alone are over
        // budget (16 four-primitive voices at 96 kHz), it bypasses every slot and stops there.
        const int slots = limitsFor(Tier::Pro).polyphony * limitsFor(Tier::Pro).voiceSlots;
        REQUIRE((r.finalShare <= r.budgetShare || r.bypassed == slots));
    }
}

TEST_CASE("the guard reduces quality before it bypasses, and bypasses the heaviest first", "[dsp][guard]")
{
    BudgetGuard g;
    g.prepare(48000.0, Tier::Pro);
    REQUIRE(g.voiceSlotShare(Family::Nonlinear, 1) < g.voiceSlotShare(Family::Nonlinear, 0));
    REQUIRE(g.voiceSlotShare(Family::Filter, 1) == g.voiceSlotShare(Family::Filter, 0));

    Synth synth;
    synth.prepare(48000.0, 256, Tier::Pro);
    auto frame = heavyFrame(Family::Nonlinear, 4);
    frame->slots[3].family = static_cast<float>(Family::Filter);
    // Ten voices: over budget, but degrading the three nonlinear slots of each is enough.
    play(synth, *frame, 10);
    run(synth, 4800);
    const auto& r = synth.guardReport();
    INFO("estimated " << r.estimatedShare << " final " << r.finalShare);
    REQUIRE(r.estimatedShare > r.budgetShare);
    // Nonlinear slots (the heaviest) go before any filter slot is touched.
    REQUIRE(r.downgraded + r.bypassed <= 3 * 10);
    REQUIRE(r.downgraded > 0);
}

TEST_CASE("with the guard off nothing is degraded, and its decisions are deterministic", "[dsp][guard]")
{
    auto frame = heavyFrame(Family::Body, 4);
    GuardReport reports[2];
    for (int i = 0; i < 2; ++i)
    {
        Synth synth;
        synth.prepare(96000.0, 256, Tier::Pro);
        play(synth, *frame, limitsFor(Tier::Pro).polyphony);
        run(synth, 9600);
        reports[i] = synth.guardReport();
    }
    REQUIRE(reports[0].downgraded == reports[1].downgraded);
    REQUIRE(reports[0].bypassed == reports[1].bypassed);
    REQUIRE(reports[0].finalShare == reports[1].finalShare);

    Synth off;
    off.prepare(96000.0, 256, Tier::Pro);
    off.setBudgetGuardEnabled(false);
    play(off, *frame, limitsFor(Tier::Pro).polyphony);
    run(off, 9600);
    REQUIRE(off.guardReport().downgraded == 0);
    REQUIRE(off.guardReport().bypassed == 0);
    REQUIRE(off.guardReport().finalShare == off.guardReport().estimatedShare);
}

TEST_CASE("cost scale moves the guard's threshold", "[dsp][guard]")
{
    Synth synth;
    synth.prepare(48000.0, 256, Tier::Standard);
    synth.setCostScale(50.0f); // a machine 50x slower than the calibration one
    auto frame = heavyFrame(Family::Filter, 2);
    play(synth, *frame, 2);
    run(synth, 4800);
    REQUIRE(synth.guardReport().bypassed > 0);
}
