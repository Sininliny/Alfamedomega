#include "TestHelpers.h"
#include "alfa/dsp/Modulation.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace alfa::dsp;
using alfa::dsp::test::AllocationGuard;
using alfa::dsp::test::zeroCrossingHz;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

std::size_t u(int i)
{
    return static_cast<std::size_t>(i);
}

std::vector<float> renderEnv(DahdsrEnvelope& env, int n)
{
    std::vector<float> out(u(n));
    env.render(out.data(), n);
    return out;
}

/// First index at or after `from` where pred(out[i]) holds, or -1.
template <typename Pred>
int firstIndex(const std::vector<float>& out, int from, Pred pred)
{
    for (int i = from; i < static_cast<int>(out.size()); ++i)
        if (pred(out[u(i)]))
            return i;
    return -1;
}

double maxStep(const std::vector<float>& x)
{
    double m = 0.0;
    for (std::size_t i = 1; i < x.size(); ++i)
        m = std::max(m, static_cast<double>(std::fabs(x[i] - x[i - 1])));
    return m;
}

void requireWithinOnePercent(int measuredSamples, double seconds, double sr)
{
    const double expected = seconds * sr;
    INFO("measured " << measuredSamples << " expected " << expected);
    REQUIRE(std::fabs(static_cast<double>(measuredSamples) - expected) <= std::max(1.0, 0.01 * expected));
}

} // namespace

// Maps --------------------------------------------------------------------------------------

TEST_CASE("Modulation time and rate maps are log over the documented ranges", "[dsp][mod]")
{
    REQUIRE_THAT(modTimeSeconds(0.0f), WithinRel(1.0e-4f, 1e-4f));
    REQUIRE_THAT(modTimeSeconds(1.0f), WithinRel(10.0f, 1e-4f));
    REQUIRE_THAT(modTimeSeconds(0.6f), WithinRel(0.1f, 1e-4f)); // 1e-4 * 1e5^0.6 = 0.1
    REQUIRE_THAT(modRateHz(0.0f), WithinRel(0.01f, 1e-4f));
    REQUIRE_THAT(modRateHz(0.5f), WithinRel(1.0f, 1e-4f));
    REQUIRE_THAT(modRateHz(1.0f), WithinRel(100.0f, 1e-4f));
    REQUIRE(modTimeSeconds(kNaN) == 0.0f);
    REQUIRE(modRateHz(kNaN) == 0.0f);
    REQUIRE(modTimeSeconds(2.0f) == modTimeSeconds(1.0f));
}

// Envelopes -----------------------------------------------------------------------------------

TEST_CASE("DAHDSR stage timings are within 1% at 44.1, 48 and 96 kHz", "[dsp][mod]")
{
    const double sr = GENERATE(44100.0, 48000.0, 96000.0);
    CAPTURE(sr);
    constexpr float delay = 0.01f, attack = 0.05f, hold = 0.02f, decay = 0.1f, sustain = 0.5f, release = 0.2f;
    DahdsrEnvelope env;
    env.prepare(sr);
    env.setTimes(delay, attack, hold, decay, sustain, release);
    env.noteOn();
    const auto gate = renderEnv(env, static_cast<int>(0.5 * sr));

    // out[i] is the level after i + 1 samples, so "first index where" + 1 is the elapsed time.
    const int attackStart = firstIndex(gate, 0, [](float v) { return v > 0.0f; });
    requireWithinOnePercent(attackStart, delay, sr); // the first rising sample ends the delay
    const int peak = firstIndex(gate, 0, [](float v) { return v >= 1.0f - 1e-6f; });
    requireWithinOnePercent(peak + 1, delay + attack, sr);
    const int decayStart = firstIndex(gate, peak, [](float v) { return v < 1.0f - 1e-6f; });
    requireWithinOnePercent(decayStart, delay + attack + hold, sr);
    const int atSustain =
        firstIndex(gate, decayStart, [&](float v) { return std::fabs(v - sustain) < 1e-6f; });
    requireWithinOnePercent(atSustain + 1, delay + attack + hold + decay, sr);
    REQUIRE_THAT(gate.back(), WithinAbs(sustain, 1e-6));

    env.noteOff();
    const auto rel = renderEnv(env, static_cast<int>(0.4 * sr));
    const int zero = firstIndex(rel, 0, [](float v) { return v <= 0.0f; });
    requireWithinOnePercent(zero + 1, release, sr);
    REQUIRE(env.idle());
    REQUIRE(rel.back() == 0.0f);
}

TEST_CASE("DAHDSR normalised times follow the log map", "[dsp][mod]")
{
    constexpr double sr = 48000.0;
    DahdsrEnvelope env;
    env.prepare(sr);
    // delay, attack 0.6 (0.1 s), hold, decay, sustain 1, release
    const std::array<float, 6> p{0.0f, 0.6f, 0.0f, 0.0f, 1.0f, 0.0f};
    env.setNormalised(p.data());
    env.noteOn();
    const auto out = renderEnv(env, 10000);
    const int peak = firstIndex(out, 0, [](float v) { return v >= 1.0f - 1e-6f; });
    requireWithinOnePercent(peak + 1, 1.0e-4 + 0.1, sr); // plus the 0.1 ms minimum delay
}

TEST_CASE("DAHDSR sustain holds its level and follows changes smoothly", "[dsp][mod]")
{
    DahdsrEnvelope env;
    env.prepare(48000.0);
    env.setTimes(0.0f, 0.001f, 0.0f, 0.01f, 0.3f, 0.1f);
    env.noteOn();
    env.advance(4800);
    REQUIRE(env.level() == 0.3f);
    env.advance(48000);
    REQUIRE(env.level() == 0.3f);
    env.setTimes(0.0f, 0.001f, 0.0f, 0.01f, 0.8f, 0.1f);
    const auto out = renderEnv(env, 4800);
    REQUIRE(maxStep(out) < 0.01); // slewed, not stepped
    REQUIRE_THAT(out.back(), WithinAbs(0.8, 1e-5));
}

TEST_CASE("DAHDSR curves: attack slightly convex, decay and release exponential", "[dsp][mod]")
{
    DahdsrEnvelope env;
    env.prepare(48000.0);
    env.setTimes(0.0f, 0.1f, 0.0f, 0.1f, 0.0f, 0.1f);
    env.noteOn();
    const auto out = renderEnv(env, 9600);
    const float midAttack = out[2399];
    REQUIRE(midAttack > 0.55f);               // convex: above the linear 0.5
    REQUIRE(midAttack < 0.8f);                // but only slightly
    const float fifthDecay = out[4800 + 959]; // 20% into the decay
    REQUIRE(fifthDecay < 0.4f);               // more than 60% of the way down
    for (std::size_t i = 1; i < 4800; ++i)
        REQUIRE(out[i] >= out[i - 1]);
}

TEST_CASE("DAHDSR release from mid-attack is continuous", "[dsp][mod]")
{
    const double sr = GENERATE(44100.0, 96000.0);
    DahdsrEnvelope env;
    env.prepare(sr);
    env.setTimes(0.0f, 0.1f, 0.0f, 0.1f, 0.5f, 0.05f);
    env.noteOn();
    auto out = renderEnv(env, static_cast<int>(0.04 * sr));
    const float before = out.back();
    REQUIRE(before > 0.2f);
    REQUIRE(before < 0.9f);
    env.noteOff();
    const auto rel = renderEnv(env, static_cast<int>(0.06 * sr));
    // The step into the release is the release's own initial slope: k / (1 - e^-k) x level
    // per release length for k = 5, and no bigger than the steps that follow.
    const double firstStep = std::fabs(rel[0] - before);
    REQUIRE(firstStep <= 1.01 * maxStep(rel));
    REQUIRE(firstStep < 1.01 * 5.0 / (1.0 - std::exp(-5.0)) * before / (0.05 * sr));
    for (std::size_t i = 1; i < rel.size(); ++i)
        REQUIRE(rel[i] <= rel[i - 1]);
    REQUIRE(rel.back() == 0.0f);
    const int zero = firstIndex(rel, 0, [](float v) { return v <= 0.0f; });
    requireWithinOnePercent(zero + 1, 0.05, sr);
}

TEST_CASE("DAHDSR retrigger restarts from the current level", "[dsp][mod]")
{
    DahdsrEnvelope env;
    env.prepare(48000.0);
    env.setTimes(0.0f, 0.02f, 0.0f, 0.05f, 0.7f, 0.2f);
    env.noteOn();
    env.advance(9600);
    env.noteOff();
    env.advance(2400);
    const float mid = env.level();
    REQUIRE(mid > 0.1f);
    REQUIRE(mid < 0.7f);
    env.noteOn();
    const auto out = renderEnv(env, 960);
    REQUIRE(std::fabs(out[0] - mid) < 0.01f);
    REQUIRE(out[0] >= mid);
    REQUIRE(out.back() == 1.0f); // attack still takes its full time (960 samples)
}

TEST_CASE("DAHDSR with zero times does not divide by zero", "[dsp][mod]")
{
    DahdsrEnvelope env;
    env.prepare(48000.0);
    env.setTimes(0.0f, 0.0f, 0.0f, 0.0f, 0.4f, 0.0f);
    REQUIRE(env.advance(1) == 0.0f);
    env.noteOn();
    REQUIRE(env.advance(0) == 0.4f); // instant stages resolve even without samples
    REQUIRE(env.advance(1) == 0.4f);
    env.noteOff();
    REQUIRE(env.advance(1) == 0.0f);
    REQUIRE(env.idle());

    env.setTimes(kNaN, -1.0f, kInf, kNaN, kNaN, -kInf);
    env.noteOn();
    for (int i = 0; i < 10; ++i)
        REQUIRE(std::isfinite(env.advance(32)));
    env.noteOff();
    REQUIRE(std::isfinite(env.advance(32)));
}

TEST_CASE("Envelope time changes mid-stage do not jump", "[dsp][mod]")
{
    DahdsrEnvelope env;
    env.prepare(48000.0);
    env.setTimes(0.0f, 0.1f, 0.0f, 0.1f, 0.5f, 0.1f);
    env.noteOn();
    std::vector<float> out;
    for (int b = 0; b < 300; ++b)
    {
        // Attack time jumps between 50 ms and 200 ms every block.
        env.setTimes(0.0f, (b % 2) != 0 ? 0.05f : 0.2f, 0.0f, 0.1f, 0.5f, 0.1f);
        out.push_back(env.advance(32));
    }
    REQUIRE(maxStep(out) < 0.05);
    REQUIRE(out.back() == 0.5f);
}

TEST_CASE("SegmentEnvelope runs arbitrary stage lists", "[dsp][mod]")
{
    using Kind = EnvelopeStage::Kind;
    SegmentEnvelope env;
    env.prepare(1000.0);
    // Five-stage one-shot: up, down, up, hold, down, with no sustain and no release stage.
    env.configure(5, 5);
    env.setStage(0, {Kind::Ramp, 0.010f, 1.0f, 0.0f});
    env.setStage(1, {Kind::Ramp, 0.010f, 0.2f, 0.0f});
    env.setStage(2, {Kind::Ramp, 0.010f, 0.6f, -2.0f});
    env.setStage(3, {Kind::Hold, 0.005f, 0.0f, 0.0f});
    env.setStage(4, {Kind::Ramp, 0.010f, 0.0f, 3.0f});
    env.noteOn();
    std::vector<float> out(60);
    env.render(out.data(), 60);
    REQUIRE(out[9] == 1.0f);
    REQUIRE_THAT(out[4], WithinAbs(0.5, 1e-6)); // linear
    REQUIRE(out[19] == 0.2f);
    REQUIRE(out[24] < 0.4f); // curve < 0 starts slowly
    REQUIRE(out[29] == 0.6f);
    REQUIRE(out[34] == 0.6f);
    REQUIRE(out[44] == 0.0f);
    REQUIRE(env.idle());
    env.noteOff(); // no release stage: nothing happens
    REQUIRE(env.advance(10) == 0.0f);
}

// LFO ----------------------------------------------------------------------------------------

TEST_CASE("LFO frequency and phase are accurate", "[dsp][mod]")
{
    const double sr = GENERATE(44100.0, 48000.0, 96000.0);
    CAPTURE(sr);
    Lfo lfo;
    lfo.prepare(sr);
    lfo.setRateHz(7.0f);
    lfo.setShape(0.0f);
    lfo.setStartPhase(0.25f);
    lfo.reset();
    REQUIRE_THAT(lfo.value(), WithinAbs(1.0, 1e-5)); // sine starts at its peak
    std::vector<float> out(u(static_cast<int>(3.0 * sr)));
    lfo.render(out.data(), static_cast<int>(out.size()));
    REQUIRE_THAT(zeroCrossingHz(out, sr), WithinRel(7.0, 1e-4));
    for (std::size_t i : {std::size_t{0}, std::size_t{1234}, out.size() - 1})
    {
        const double phase = 0.25 + 7.0 * static_cast<double>(i + 1) / sr;
        REQUIRE_THAT(out[i], WithinAbs(std::sin(2.0 * 3.14159265358979323846 * phase), 1e-4));
    }
    // Block-rate advance lands on the same phase as the per-sample render.
    Lfo blocky;
    blocky.prepare(sr);
    blocky.setRateHz(7.0f);
    blocky.setStartPhase(0.25f);
    blocky.reset();
    for (std::size_t i = 0; i < out.size() / 32; ++i)
        blocky.advance(32);
    REQUIRE_THAT(blocky.value(), WithinAbs(out[out.size() / 32 * 32 - 1], 1e-4));
}

TEST_CASE("LFO normalised params: rate map, retrigger and free-running", "[dsp][mod]")
{
    Lfo lfo;
    lfo.prepare(48000.0);
    const std::array<float, 4> p{0.5f, 0.0f, 0.5f, 1.0f}; // 1 Hz, sine, start 0.5, retrigger
    lfo.setNormalised(p.data());
    REQUIRE_THAT(lfo.rateHz(), WithinRel(1.0f, 1e-4f));
    lfo.advance(10000);
    lfo.noteOn();
    REQUIRE_THAT(lfo.phase(), WithinAbs(0.5, 1e-12));
    lfo.setRetrigger(false);
    lfo.advance(1000);
    const double ph = lfo.phase();
    lfo.noteOn();
    REQUIRE(lfo.phase() == ph);
}

TEST_CASE("LFO shape morph endpoints are the five shapes", "[dsp][mod]")
{
    for (int k = 0; k < 64; ++k)
    {
        const double p = (static_cast<double>(k) + 0.37) / 64.0;
        const auto pf = static_cast<float>(p);
        const float sine = static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * p));
        const float tri = pf < 0.25f ? 4.0f * pf : (pf < 0.75f ? 2.0f - 4.0f * pf : 4.0f * pf - 4.0f);
        const float saw = pf < 0.5f ? 2.0f * pf : 2.0f * pf - 2.0f;
        const float square = pf < 0.5f ? 1.0f : -1.0f;
        REQUIRE_THAT(Lfo::shapeAt(0.0f, p, 0.0f, 0.1f, 0.3f), WithinAbs(sine, 1e-5));
        REQUIRE_THAT(Lfo::shapeAt(0.25f, p, 0.0f, 0.1f, 0.3f), WithinAbs(tri, 1e-5));
        REQUIRE_THAT(Lfo::shapeAt(0.5f, p, 0.0f, 0.1f, 0.3f), WithinAbs(saw, 1e-5));
        REQUIRE_THAT(Lfo::shapeAt(0.75f, p, 0.0f, 0.1f, 0.3f), WithinAbs(square, 1e-6));
        REQUIRE_THAT(Lfo::shapeAt(1.0f, p, 0.0f, 0.1f, 0.3f), WithinAbs(0.3, 1e-6));
        // Halfway between two shapes is their average.
        REQUIRE_THAT(Lfo::shapeAt(0.125f, p, 0.0f, 0.1f, 0.3f), WithinAbs(0.5f * (sine + tri), 1e-5));
        REQUIRE_THAT(Lfo::shapeAt(0.875f, p, 0.0f, 0.1f, 0.3f), WithinAbs(0.5f * (square + 0.3f), 1e-5));
        // Soft edges keep every shape inside -1..1.
        for (float s = 0.0f; s <= 1.0f; s += 0.05f)
            REQUIRE(std::fabs(Lfo::shapeAt(s, p, 0.05f, -1.0f, 1.0f)) <= 1.0f);
    }
}

TEST_CASE("LFO square and saw edges do not step at the control rate", "[dsp][mod]")
{
    const float shape = GENERATE(0.5f, 0.75f, 1.0f);
    CAPTURE(shape);
    Lfo lfo;
    lfo.prepare(48000.0);
    lfo.setSeed(7);
    lfo.setRateHz(20.0f);
    lfo.setShape(shape);
    lfo.reset();
    std::vector<float> blocks;
    for (int b = 0; b < 1500; ++b)
        blocks.push_back(lfo.advance(kControlBlock));
    // A 2 ms edge spans 3 control blocks, so a full -1..1 swing moves at most 2/3 per block.
    REQUIRE(maxStep(blocks) < 0.7);
    // The shape itself is intact: it still reaches both extremes (S&H: varies).
    const auto [lo, hi] = std::minmax_element(blocks.begin(), blocks.end());
    REQUIRE(*hi - *lo > 1.0f);
}

TEST_CASE("LFO shape changes are smoothed", "[dsp][mod]")
{
    Lfo lfo;
    lfo.prepare(48000.0);
    lfo.setRateHz(0.01f);
    lfo.setStartPhase(0.2f);
    lfo.setShape(0.0f);
    lfo.reset();
    lfo.setShape(0.75f); // sine 0.95 -> square 1: small; go to saw instead for a large jump
    lfo.setShape(0.5f);
    std::vector<float> blocks;
    for (int b = 0; b < 100; ++b)
        blocks.push_back(lfo.advance(kControlBlock));
    REQUIRE(maxStep(blocks) < 0.2);
    REQUIRE_THAT(blocks.back(), WithinAbs(Lfo::shapeAt(0.5f, lfo.phase(), 0.0f, 0.0f, 0.0f), 1e-3));
}

// Random sources ---------------------------------------------------------------------------------

TEST_CASE("Sample-and-hold holds between ticks and is seeded", "[dsp][mod]")
{
    const double sr = GENERATE(44100.0, 48000.0, 96000.0);
    SampleHold sh;
    sh.prepare(sr);
    sh.setSeed(1234);
    sh.setRateHz(10.0f);
    std::vector<float> out(u(static_cast<int>(sr)));
    sh.render(out.data(), static_cast<int>(out.size()));
    const auto period = static_cast<int>(sr / 10.0);
    const auto edge = static_cast<int>(std::ceil(0.002 * sr)) + 2;
    std::vector<float> held;
    for (int t = 0; t + period <= static_cast<int>(out.size()); t += period)
    {
        // Constant from just after the edge ramp to just before the next tick.
        const float v = out[u(t + edge)];
        for (int i = t + edge; i < t + period - 2; ++i)
            REQUIRE(out[u(i)] == v);
        REQUIRE(std::fabs(v) <= 1.0f);
        held.push_back(v);
    }
    REQUIRE(held.size() >= 9);
    std::sort(held.begin(), held.end());
    REQUIRE(std::unique(held.begin(), held.end()) - held.begin() >= 8);

    SampleHold again;
    again.prepare(sr);
    again.setSeed(1234);
    again.setRateHz(10.0f);
    std::vector<float> out2(out.size());
    again.render(out2.data(), static_cast<int>(out2.size()));
    REQUIRE(out2 == out);
    again.setSeed(99);
    again.render(out2.data(), static_cast<int>(out2.size()));
    REQUIRE(out2 != out);
}

TEST_CASE("Smooth random is continuous and bounded", "[dsp][mod]")
{
    SmoothRandom sr;
    sr.prepare(48000.0);
    sr.setSeed(42);
    std::vector<float> out(96000);
    sr.setRateHz(50.0f);
    sr.render(out.data(), 48000);
    sr.setRateHz(100.0f); // rate change mid-segment keeps the output continuous
    sr.render(out.data() + 48000, 48000);
    for (float v : out)
        REQUIRE(std::fabs(v) <= 1.0f);
    // smoothstep's slope is at most 1.5 x (segment span 2) x rate / sr.
    REQUIRE(maxStep(out) <= 1.5 * 2.0 * 100.0 / 48000.0 + 1e-6);
    double lo = 1.0, hi = -1.0;
    for (float v : out)
    {
        lo = std::min(lo, static_cast<double>(v));
        hi = std::max(hi, static_cast<double>(v));
    }
    REQUIRE(hi - lo > 1.0); // it actually moves
}

// Follower ----------------------------------------------------------------------------------

TEST_CASE("Envelope follower tracks a step with its time constants", "[dsp][mod]")
{
    const double sr = GENERATE(44100.0, 48000.0, 96000.0);
    CAPTURE(sr);
    EnvelopeFollower f;
    f.prepare(sr);
    f.setTimes(0.01f, 0.1f);
    const auto tauA = static_cast<int>(std::lround(0.01 * sr));
    const auto tauR = static_cast<int>(std::lround(0.1 * sr));
    const std::vector<float> ones(u(tauA), -1.0f); // peak mode rectifies
    f.process(ones.data(), tauA);
    REQUIRE_THAT(f.value(), WithinRel(1.0 - std::exp(-1.0), 0.01));
    std::vector<float> longOnes(u(tauA * 10), 1.0f);
    f.process(longOnes.data(), tauA * 10);
    REQUIRE(f.value() > 0.9999f);
    const std::vector<float> zeros(u(tauR), 0.0f);
    f.process(zeros.data(), tauR);
    REQUIRE_THAT(f.value(), WithinRel(std::exp(-1.0), 0.01));
    f.process(nullptr, tauR); // nullptr is silence
    REQUIRE_THAT(f.value(), WithinRel(std::exp(-2.0), 0.01));
}

TEST_CASE("Envelope follower RMS mode, clamping and NaN input", "[dsp][mod]")
{
    EnvelopeFollower f;
    f.prepare(48000.0);
    f.setMode(EnvelopeFollower::Mode::Rms);
    f.setTimes(0.05f, 0.05f);
    std::vector<float> x(48000);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = 0.8f * static_cast<float>(
                          std::sin(2.0 * 3.14159265358979323846 * 200.0 * static_cast<double>(i) / 48000.0));
    f.process(x.data(), 48000);
    REQUIRE_THAT(f.value(), WithinAbs(0.8 / std::sqrt(2.0), 0.02));

    std::fill(x.begin(), x.end(), 10.0f);
    f.process(x.data(), 4800);
    REQUIRE(f.value() == 1.0f); // clamped 0..1
    x[3] = kNaN;
    x[7] = kInf;
    f.process(x.data(), 32);
    REQUIRE(std::isfinite(f.value()));
    REQUIRE(std::isfinite(f.raw()));
    std::fill(x.begin(), x.end(), 0.0f);
    f.process(x.data(), 48000);
    REQUIRE(f.value() < 1.0e-4f); // power 100 released over 20 time constants
}

// Bank -------------------------------------------------------------------------------------

namespace
{

ModControls controlsForBank()
{
    ModControls c{};
    // env1: fast ADSR; env2: slow attack
    const std::array<float, 6> e1{0.0f, 0.3f, 0.0f, 0.5f, 0.6f, 0.5f};
    const std::array<float, 6> e2{0.0f, 0.8f, 0.0f, 0.5f, 1.0f, 0.5f};
    std::copy(e1.begin(), e1.end(), c.sources.begin());
    std::copy(e2.begin(), e2.end(), c.sources.begin() + 6);
    c.sources[12] = 0.6f; // lfo1 ~2.5 Hz
    c.sources[13] = 0.0f; // sine
    c.sources[14] = 0.0f;
    c.sources[15] = 1.0f;  // retrigger
    c.sources[16] = 0.7f;  // lfo2 ~6.3 Hz
    c.sources[17] = 0.6f;  // saw..square
    c.sources[20] = 0.75f; // S&H 10 Hz
    c.sources[21] = 0.75f; // smooth random 10 Hz
    c.sources[22] = 0.4f;  // follower attack 10 ms
    c.sources[23] = 0.6f;  // follower release 100 ms
    return c;
}

} // namespace

TEST_CASE("ModBank exposes every source in its range", "[dsp][mod]")
{
    ModBank bank;
    bank.prepare(48000.0);
    bank.setSeed(5);
    bank.update(controlsForBank());
    bank.noteOn(0.75f, 64);
    REQUIRE(bank.value(ModSource::Velocity) == 0.75f);
    REQUIRE_THAT(bank.value(ModSource::Key), WithinAbs(64.0 / 127.0, 1e-6));
    REQUIRE(bank.value(ModSource::None) == 0.0f);
    std::vector<float> audio(kControlBlock, 0.5f);
    float envPeak = 0.0f, followerPeak = 0.0f;
    for (int b = 0; b < 1500; ++b)
    {
        bank.advance(audio.data(), kControlBlock);
        for (int s = 0; s < kModSourceCount; ++s)
        {
            const float v = bank.value(static_cast<ModSource>(s));
            REQUIRE(std::isfinite(v));
            REQUIRE(v >= -1.0f);
            REQUIRE(v <= 1.0f);
        }
        envPeak = std::max(envPeak, bank.value(ModSource::Env1));
        followerPeak = std::max(followerPeak, bank.value(ModSource::Follower));
    }
    REQUIRE(envPeak == 1.0f);
    REQUIRE_THAT(bank.value(ModSource::Env1), WithinAbs(0.6, 1e-5));
    REQUIRE_THAT(followerPeak, WithinAbs(0.5, 1e-3));
    bank.noteOff();
    for (int b = 0; b < 3000; ++b)
        bank.advance(nullptr, kControlBlock);
    REQUIRE(bank.value(ModSource::Env1) == 0.0f);
    REQUIRE(bank.env1().idle());
    REQUIRE(bank.value(ModSource::Follower) < 1.0e-3f);
}

TEST_CASE("ModBank is deterministic for a seed", "[dsp][mod]")
{
    const auto run = [](std::uint32_t seed)
    {
        ModBank bank;
        bank.prepare(48000.0);
        bank.setSeed(seed);
        bank.update(controlsForBank());
        bank.noteOn(1.0f, 60);
        std::vector<float> trace;
        for (int b = 0; b < 400; ++b)
        {
            bank.advance(nullptr, kControlBlock);
            for (auto s : {ModSource::Lfo2, ModSource::SampleHold, ModSource::SmoothRandom})
                trace.push_back(bank.value(s));
        }
        return trace;
    };
    REQUIRE(run(11) == run(11));
    REQUIRE(run(11) != run(12));

    // reset() replays the same stream.
    ModBank bank;
    bank.prepare(48000.0);
    bank.setSeed(3);
    bank.update(controlsForBank());
    bank.advance(nullptr, 32);
    const float first = bank.value(ModSource::SmoothRandom);
    for (int b = 0; b < 100; ++b)
        bank.advance(nullptr, 32);
    bank.reset();
    bank.advance(nullptr, 32);
    REQUIRE(bank.value(ModSource::SmoothRandom) == first);
}

TEST_CASE("Instance ModBank runs only the LFOs and random sources", "[dsp][mod]")
{
    ModBank bank(ModScope::Instance);
    bank.prepare(48000.0);
    bank.update(controlsForBank());
    bank.noteOn(1.0f, 100);
    std::vector<float> audio(kControlBlock, 1.0f);
    bool lfoMoved = false;
    for (int b = 0; b < 500; ++b)
    {
        bank.advance(audio.data(), kControlBlock);
        for (auto s :
             {ModSource::Env1, ModSource::Env2, ModSource::Follower, ModSource::Velocity, ModSource::Key})
            REQUIRE(bank.value(s) == 0.0f);
        lfoMoved = lfoMoved || std::fabs(bank.value(ModSource::Lfo1)) > 0.5f;
    }
    REQUIRE(lfoMoved);
}

TEST_CASE("ModBank survives NaN and extreme source params", "[dsp][mod]")
{
    ModBank bank;
    bank.prepare(96000.0);
    ModControls c{};
    auto seed = static_cast<std::uint32_t>(GENERATE(1, 2, 3));
    alfa::dsp::detail::Random rng(seed);
    std::vector<float> audio(kControlBlock);
    bank.noteOn(kNaN, -5);
    REQUIRE(bank.value(ModSource::Velocity) == 0.0f);
    for (int b = 0; b < 2000; ++b)
    {
        for (auto& s : c.sources)
        {
            const auto r = rng.nextU32() % 8u;
            s = r == 0   ? kNaN
                : r == 1 ? kInf
                : r == 2 ? 0.0f
                : r == 3 ? 1.0f
                         : 0.5f + 0.5f * rng.nextBipolar();
        }
        for (auto& a : audio)
            a = (rng.nextU32() % 50u) == 0 ? kNaN : 4.0f * rng.nextBipolar();
        bank.update(c);
        if (b % 97 == 0)
            bank.noteOn(1.0f, 200);
        if (b % 131 == 0)
            bank.noteOff();
        bank.advance(audio.data(), kControlBlock);
        for (int s = 0; s < kModSourceCount; ++s)
        {
            const float v = bank.value(static_cast<ModSource>(s));
            REQUIRE(std::isfinite(v));
            REQUIRE(std::fabs(v) <= 1.0f);
        }
    }
}

// Matrix ----------------------------------------------------------------------------------

namespace
{

/// A bank whose velocity is 0.5 and key 1 (note 127), with everything else at rest.
ModBank velocityBank(ModScope scope = ModScope::Voice)
{
    ModBank bank(scope);
    bank.prepare(48000.0);
    bank.noteOn(0.5f, 127);
    return bank;
}

std::array<float, kModDestinations> baseDestinations()
{
    std::array<float, kModDestinations> d{};
    d.fill(0.5f);
    d[moddest::pitch] = 0.0f;
    d[moddest::pan] = 0.0f;
    return d;
}

void setRoute(ModControls& c, int r, float source, float dest, float depth)
{
    c.routeSource[u(r)] = source;
    c.routeDestination[u(r)] = dest;
    c.routeDepth[u(r)] = depth;
}

} // namespace

TEST_CASE("ModMatrix adds depth x source and clamps per destination", "[dsp][mod]")
{
    const ModBank bank = velocityBank();
    ModControls c{};
    setRoute(c, 0, 8.0f, 3.0f, 0.4f);                      // slot 0 param 3 += 0.2
    setRoute(c, 1, 9.0f, 17.0f, 1.0f);                     // slot 1 param 1 += 1 -> clamp 1
    setRoute(c, 2, 9.0f, 18.0f, -1.0f);                    // slot 1 param 2 -= 1 -> clamp 0
    setRoute(c, 3, 9.0f, moddest::slotOutGain + 2, -1.0f); // gain -> clamp 0
    setRoute(c, 4, 8.0f, moddest::pitch, 0.5f);            // 24 * 0.5 * 0.5 = 6 semitones
    setRoute(c, 5, 9.0f, moddest::pan, 1.0f);              // 0 + 1 -> 1
    setRoute(c, 6, 9.0f, moddest::pan, 1.0f);              // 2 -> clamp 1
    setRoute(c, 7, 8.0f, moddest::mix + 1, 3.0f);          // depth clamps to 1: += 0.5
    auto d = baseDestinations();
    ModMatrix::apply(c, kMaxModRoutes, bank, d.data());
    REQUIRE_THAT(d[3], WithinAbs(0.7, 1e-6));
    REQUIRE(d[17] == 1.0f);
    REQUIRE(d[18] == 0.0f);
    REQUIRE(d[u(moddest::slotOutGain + 2)] == 0.0f);
    REQUIRE_THAT(d[u(moddest::pitch)], WithinAbs(6.0, 1e-6));
    REQUIRE(d[u(moddest::pan)] == 1.0f);
    REQUIRE_THAT(d[u(moddest::mix + 1)], WithinAbs(1.0, 1e-6));
    // Untouched destinations keep their base.
    REQUIRE(d[0] == 0.5f);
    REQUIRE(d[u(moddest::dryGain)] == 0.5f);
    // Gains may exceed 1.
    ModControls g{};
    setRoute(g, 0, 9.0f, moddest::dryGain, 1.0f);
    auto d2 = baseDestinations();
    ModMatrix::apply(g, 1, bank, d2.data());
    REQUIRE(d2[u(moddest::dryGain)] == 1.5f);
}

TEST_CASE("ModMatrix rounds, ignores NaN and out-of-range routes and honours the limit", "[dsp][mod]")
{
    const ModBank bank = velocityBank();
    const auto base = baseDestinations();

    SECTION("rounding")
    {
        ModControls c{};
        setRoute(c, 0, 7.6f, 2.4f, 0.2f);  // velocity -> dest 2
        setRoute(c, 1, 8.5f, 4.51f, 0.2f); // 8.5 rounds up to key -> dest 5
        auto d = base;
        ModMatrix::apply(c, 2, bank, d.data());
        REQUIRE_THAT(d[2], WithinAbs(0.6, 1e-6));
        REQUIRE_THAT(d[5], WithinAbs(0.7, 1e-6));
    }
    SECTION("NaN and Inf routes are ignored")
    {
        ModControls c{};
        setRoute(c, 0, kNaN, 1.0f, 1.0f);
        setRoute(c, 1, 8.0f, kNaN, 1.0f);
        setRoute(c, 2, 8.0f, 1.0f, kNaN);
        setRoute(c, 3, kInf, 1.0f, 1.0f);
        setRoute(c, 4, 8.0f, -kInf, 1.0f);
        setRoute(c, 5, 8.0f, 1.0f, kInf);
        auto d = base;
        ModMatrix::apply(c, kMaxModRoutes, bank, d.data());
        REQUIRE(d == base);
    }
    SECTION("out-of-range indices and source None are ignored")
    {
        ModControls c{};
        setRoute(c, 0, 10.0f, 1.0f, 1.0f);
        setRoute(c, 1, -1.0f, 1.0f, 1.0f);
        setRoute(c, 2, 8.0f, 77.0f, 1.0f);
        setRoute(c, 3, 8.0f, -1.0f, 1.0f);
        setRoute(c, 4, 0.0f, 1.0f, 1.0f);
        setRoute(c, 5, 8.0f, 1e30f, 1.0f);
        auto d = base;
        ModMatrix::apply(c, kMaxModRoutes, bank, d.data());
        REQUIRE(d == base);
    }
    SECTION("only the first routeLimit routes run")
    {
        ModControls c{};
        for (int r = 0; r < kMaxModRoutes; ++r)
            setRoute(c, r, 8.0f, static_cast<float>(r), 0.2f);
        for (int limit : {-3, 0, 4, 6, 8, 20})
        {
            auto d = base;
            ModMatrix::apply(c, limit, bank, d.data());
            const int active = std::clamp(limit, 0, kMaxModRoutes);
            for (int r = 0; r < kMaxModRoutes; ++r)
                REQUIRE_THAT(d[u(r)], WithinAbs(r < active ? 0.6 : 0.5, 1e-6));
        }
    }
    SECTION("routes sum in order and the result is deterministic")
    {
        ModControls c{};
        for (int r = 0; r < kMaxModRoutes; ++r)
            setRoute(c, r, 8.0f, 10.0f, 0.1f * static_cast<float>(r % 3) - 0.1f);
        auto a = base;
        auto b = base;
        ModMatrix::apply(c, kMaxModRoutes, bank, a.data());
        ModMatrix::apply(c, kMaxModRoutes, bank, b.data());
        REQUIRE(a == b);
        REQUIRE_THAT(a[10], WithinAbs(0.5 - 0.5 * 0.1, 1e-6)); // depths -0.1, 0, 0.1, ... sum to -0.1
    }
    SECTION("non-finite base values are sanitised")
    {
        ModControls c{};
        auto d = base;
        d[0] = kNaN;
        d[u(moddest::pitch)] = kInf;
        d[u(moddest::pan)] = -5.0f;
        ModMatrix::apply(c, 0, bank, d.data());
        REQUIRE(d[0] == 0.0f);
        REQUIRE(d[u(moddest::pitch)] == 0.0f);
        REQUIRE(d[u(moddest::pan)] == -1.0f);
    }
}

TEST_CASE("Instance ModMatrix applies only instance destinations", "[dsp][mod]")
{
    ModBank bank(ModScope::Instance);
    bank.prepare(48000.0);
    ModControls c{};
    c.sources[12] = 0.5f;
    c.sources[14] = 0.25f; // lfo1 starts at +1
    bank.update(c);
    bank.reset();
    REQUIRE_THAT(bank.value(ModSource::Lfo1), WithinAbs(1.0, 1e-5));
    setRoute(c, 0, 3.0f, 20.0f, 0.25f);
    setRoute(c, 1, 3.0f, moddest::slotOutGain + 3, 0.25f);
    setRoute(c, 2, 3.0f, moddest::dryGain, 0.25f);
    setRoute(c, 3, 3.0f, moddest::mix, 0.25f);
    setRoute(c, 4, 3.0f, moddest::pitch, 0.25f);
    setRoute(c, 5, 3.0f, moddest::pan, 0.25f);
    setRoute(c, 6, 8.0f, 21.0f, 1.0f); // velocity reads 0 in an instance bank
    auto d = baseDestinations();
    ModMatrix::apply(c, kMaxModRoutes, bank, d.data());
    REQUIRE_THAT(d[20], WithinAbs(0.75, 1e-5));
    REQUIRE_THAT(d[u(moddest::slotOutGain + 3)], WithinAbs(0.75, 1e-5));
    REQUIRE_THAT(d[u(moddest::dryGain)], WithinAbs(0.75, 1e-5));
    REQUIRE(d[u(moddest::mix)] == 0.5f);
    REQUIRE(d[u(moddest::pitch)] == 0.0f);
    REQUIRE(d[u(moddest::pan)] == 0.0f);
    REQUIRE(d[21] == 0.5f);
}

// Smoother ---------------------------------------------------------------------------------

TEST_CASE("ParamSmoother ramps linearly and lands exactly", "[dsp][mod]")
{
    ParamSmoother s;
    s.reset(0.0f);
    s.setTarget(1.0f, 4);
    std::array<float, 6> out{};
    s.fill(out.data(), 6);
    REQUIRE(out == std::array<float, 6>{0.25f, 0.5f, 0.75f, 1.0f, 1.0f, 1.0f});
    REQUIRE(!s.smoothing());
    s.setTarget(0.0f, 32);
    REQUIRE_THAT(s.skip(16), WithinAbs(0.5, 1e-6));
    REQUIRE(s.skip(100) == 0.0f);
    s.setTarget(kNaN, 8); // ignored
    REQUIRE(s.target() == 0.0f);
    s.setTarget(2.0f, 0); // jump
    REQUIRE(s.current() == 2.0f);
    s.reset(kInf);
    REQUIRE(s.current() == 0.0f);
    // Retargeting mid-ramp continues from where it is.
    s.setTarget(1.0f, 10);
    s.skip(5);
    s.setTarget(0.0f, 10);
    REQUIRE_THAT(s.next(), WithinAbs(0.45, 1e-6));
}

// Real-time safety ---------------------------------------------------------------------------

TEST_CASE("Modulation is allocation-free after prepare", "[dsp][mod]")
{
    ModBank bank;
    bank.prepare(48000.0);
    ModBank inst(ModScope::Instance);
    inst.prepare(48000.0);
    const ModControls c = controlsForBank();
    ModControls routed = c;
    for (int r = 0; r < kMaxModRoutes; ++r)
        setRoute(routed, r, static_cast<float>(1 + r), static_cast<float>(r * 9), 0.5f);
    std::vector<float> audio(kControlBlock, 0.25f);
    std::vector<float> out(kControlBlock);
    std::array<float, kModDestinations> d{};
    DahdsrEnvelope env;
    env.prepare(48000.0);
    Lfo lfo;
    lfo.prepare(48000.0);
    ParamSmoother sm;

    AllocationGuard guard;
    bank.setSeed(9);
    bank.noteOn(0.8f, 60);
    for (int b = 0; b < 200; ++b)
    {
        bank.update(c);
        inst.update(c);
        bank.advance(audio.data(), kControlBlock);
        inst.advance(nullptr, kControlBlock);
        d.fill(0.5f);
        ModMatrix::apply(routed, kMaxModRoutes, bank, d.data());
        ModMatrix::apply(routed, kMaxModRoutes, inst, d.data());
        env.setNormalised(c.sources.data());
        env.render(out.data(), kControlBlock);
        lfo.setNormalised(c.sources.data() + 12);
        lfo.render(out.data(), kControlBlock);
        sm.setTarget(d[3], kControlBlock);
        sm.fill(out.data(), kControlBlock);
    }
    bank.noteOff();
    bank.reset();
    REQUIRE(guard.count() == 0);
}

// Benchmark --------------------------------------------------------------------------------

TEST_CASE("Modulation cost, 16 voices of ModBank + ModMatrix with 8 routes", "[.bench][dsp][mod]")
{
    constexpr int kVoices = 16;
    constexpr double kSr = 48000.0;
    constexpr int kBlock = 128;
    std::vector<ModBank> banks(kVoices);
    ModControls c = controlsForBank();
    for (int r = 0; r < kMaxModRoutes; ++r)
        setRoute(c, r, static_cast<float>(1 + r % 9), static_cast<float>((r * 11) % kModDestinations), 0.3f);
    for (int v = 0; v < kVoices; ++v)
    {
        banks[u(v)].prepare(kSr);
        banks[u(v)].setSeed(static_cast<std::uint32_t>(v + 1));
        banks[u(v)].noteOn(0.8f, 40 + v);
    }
    std::vector<float> audio(kBlock, 0.3f);
    std::array<float, kModDestinations> d{};
    const int blocks = static_cast<int>(kSr) / kBlock * 10; // 10 s
    float sink = 0.0f;
    const int hop = hopSamples(kSr);
    int sinceHop = hop;
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < blocks; ++b)
    {
        for (int sub = 0; sub < kBlock; sub += kControlBlock)
        {
            const bool newHop = sinceHop >= hop;
            if (newHop)
                sinceHop -= hop;
            sinceHop += kControlBlock;
            for (auto& bank : banks)
            {
                if (newHop)
                    bank.update(c);
                bank.advance(audio.data() + sub, kControlBlock);
                d.fill(0.5f);
                ModMatrix::apply(c, kMaxModRoutes, bank, d.data());
                sink += d[11];
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / blocks;
    std::printf("mod: %.3f us per 128-sample block (16 voices, 8 routes, 48 kHz; budget %.0f us)  [%g]\n", us,
                1e6 * kBlock / kSr, static_cast<double>(sink));
    REQUIRE(std::isfinite(sink));
}
