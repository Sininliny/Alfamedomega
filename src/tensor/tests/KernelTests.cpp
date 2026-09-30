#include "alfa/tensor/Kernels.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace
{
using namespace alfa::tensor;
using namespace alfa::tensor::kernels;

// Exercises callbacks with a newly bound parameter slice each sample, as a lowering
// implementation can do for audio-rate modulation. No tensor routing is inferred.
struct KernelHarness
{
    Primitive primitive;
    std::vector<float> state;
    std::vector<float> next;
    std::array<float, 1> output{};
    double rate;
    explicit KernelHarness(Primitive p, double sampleRate)
        : primitive(std::move(p))
        , state(primitive.stateElements, 0)
        , next(primitive.stateElements, 0)
        , rate(sampleRate)
    {
    }
    float tick(std::span<const float> parameters, float audio = 0, std::uint64_t period = 1)
    {
        std::array<float, 1> input{audio};
        Step s;
        s.sampleRate = rate;
        s.clock = {period, 0};
        s.activation = 1;
        s.localState = state;
        s.parameters = parameters;
        s.inputs.audio.values = input;
        primitive.readout(s, output);
        primitive.transition(s, next);
        state.swap(next);
        return output[0];
    }
};
} // namespace

TEST_CASE("Tensor sine preserves physical frequency at different sample rates", "[tensor][kernel]")
{
    for (double rate : {32000.0, 48000.0, 96000.0})
    {
        KernelHarness h(sine(), rate);
        const std::array<float, 2> p{400, 0.75f};
        std::size_t crossings = 0;
        double power = 0;
        float previous = 0;
        for (std::size_t n = 0; n < static_cast<std::size_t>(rate); ++n)
        {
            const auto value = h.tick(p);
            if (previous <= 0 && value > 0)
                ++crossings;
            previous = value;
            power += value * value;
        }
        REQUIRE(crossings >= 399);
        REQUIRE(crossings <= 401);
        REQUIRE(power / rate == Catch::Approx(0.75 * 0.75 / 2).margin(0.0002));
    }
}

TEST_CASE("Tensor sine follows sample-rate frequency modulation and explicit clocks", "[tensor][kernel]")
{
    KernelHarness h(sine(), 48000);
    double accumulated = 0;
    for (int n = 0; n < 512; ++n)
    {
        const std::array<float, 2> p{n % 2 == 0 ? 300.0f : -1200.0f, 1};
        REQUIRE(h.tick(p) == Catch::Approx(std::sin(2 * std::numbers::pi * accumulated)).margin(0.0001));
        accumulated += p[0] / 48000.0;
        accumulated -= std::floor(accumulated);
    }
    KernelHarness slow(sine(), 48000);
    const std::array<float, 2> p{1000, 1};
    slow.tick(p, 0, 12);
    REQUIRE(slow.tick(p, 0, 12) == Catch::Approx(1).margin(0.00001));
}

TEST_CASE("Tensor one-pole has a decaying causal impulse and normalized DC gain", "[tensor][kernel]")
{
    for (double rate : {44100.0, 96000.0})
    {
        KernelHarness h(onePole(), rate);
        const std::array<float, 1> p{2000};
        REQUIRE(h.tick(p, 1) == 0); // Readout is G(S_t); impulse enters S_(t+1).
        double sum = 0;
        float previous = 1;
        for (int n = 0; n < 2000; ++n)
        {
            const float value = h.tick(p);
            REQUIRE(value >= 0);
            REQUIRE(value <= previous);
            previous = value;
            sum += value;
        }
        REQUIRE(sum == Catch::Approx(1).margin(0.00001));
        for (int n = 0; n < 2000; ++n)
            h.tick(p, 1);
        REQUIRE(h.tick(p, 1) == Catch::Approx(1).margin(0.00001));
    }
}

TEST_CASE("Tensor one-pole accepts time-varying cutoff without resetting memory", "[tensor][kernel]")
{
    KernelHarness h(onePole(), 48000);
    const std::array<float, 1> open{48000}, frozen{0};
    h.tick(open, 0.5f);
    for (int n = 0; n < 256; ++n)
        REQUIRE(h.tick(frozen, -1) == Catch::Approx(0.5).margin(0.002));
    h.tick(open, -0.25f);
    REQUIRE(h.tick(frozen) == Catch::Approx(-0.25).margin(0.002));
}

TEST_CASE("Tensor feedback cell is the specified one-update recurrence", "[tensor][kernel]")
{
    KernelHarness h(feedbackCell(), 48000);
    const std::array<float, 1> p{0.5f};
    REQUIRE(h.tick(p, 1) == 0);
    REQUIRE(h.tick(p) == 1);
    REQUIRE(h.tick(p) == 0.5f);
    REQUIRE(h.tick(p) == 0.25f);
    const std::array<float, 1> invert{-0.5f};
    REQUIRE(h.tick(invert) == 0.125f);
    REQUIRE(h.tick(invert) == -0.0625f);
}

TEST_CASE("Tensor delay length is allocated state, with delayed feedback echoes", "[tensor][kernel]")
{
    for (const std::size_t length : {1u, 7u, 513u})
    {
        KernelHarness h(delay(length), 48000);
        const std::array<float, 1> p{0.25f};
        for (std::size_t n = 0; n <= length * 3; ++n)
        {
            const auto value = h.tick(p, n == 0 ? 1.0f : 0.0f);
            const float expected = n == length       ? 1.0f
                                   : n == length * 2 ? 0.25f
                                   : n == length * 3 ? 0.0625f
                                                     : 0;
            REQUIRE(value == expected);
        }
    }
}

TEST_CASE("Tensor kernels safely silence invalid parameter bindings and hold valid state", "[tensor][kernel]")
{
    KernelHarness oscillator(sine(), 48000);
    const std::array<float, 2> p{1000, 1};
    oscillator.tick(p);
    const auto saved = oscillator.state;
    REQUIRE(oscillator.tick({}) == 0);
    REQUIRE(oscillator.state == saved);
    const std::array<float, 2> invalid{std::numeric_limits<float>::infinity(), 1};
    REQUIRE(oscillator.tick(invalid) == 0);
    REQUIRE(oscillator.state == saved);
    KernelHarness cell(feedbackCell(), 48000);
    const std::array<float, 1> stable{0.5f}, unstable{1.5f};
    cell.tick(stable, 1);
    REQUIRE(cell.tick(unstable) == 0);
    REQUIRE(cell.state.front() == 1);
    REQUIRE(cell.tick(stable, std::numeric_limits<float>::quiet_NaN()) == 1);
    REQUIRE(cell.state.front() == 0.5f);
}

TEST_CASE("Tensor primitive metadata exposes each parameter without tier-specific kinds", "[tensor][kernel]")
{
    for (const Metadata* m : {&sineMetadata(), &onePoleMetadata(), &feedbackCellMetadata(), &delayMetadata()})
    {
        REQUIRE_FALSE(m->id.empty());
        REQUIRE_FALSE(m->inputRepresentation.empty());
        REQUIRE_FALSE(m->outputRepresentation.empty());
        REQUIRE_FALSE(m->stateRepresentation.empty());
        REQUIRE_FALSE(m->costPerUpdate.empty());
        REQUIRE_FALSE(m->supportedRates.empty());
        REQUIRE_FALSE(m->reusableCount.empty());
        for (const auto& p : m->parameters)
        {
            REQUIRE(p.trainable);
            REQUIRE(p.controllable);
            REQUIRE(p.modulationTarget);
            REQUIRE_FALSE(p.units.empty());
            REQUIRE_FALSE(p.validRange.empty());
        }
    }
}

TEST_CASE("Runtime executes sine modulation and delayed feedback with shared sample input",
          "[tensor][kernel][integration]")
{
    auto oscillator = sine();
    auto echo = delay(7);
    Configuration c;
    c.activation = {{"primitive"}, {2}, {1.0f, 1.0f}, {}};
    c.parameters = {{"parameter"}, {3}, {400.0f, 1.0f, 0.5f}, {}};
    c.routing = {{"weight"}, {1}, {1200.0f}, {}};
    c.instances = {{&oscillator, 0, {0, 2}, {1, 0}, OutputBetweenUpdates::Zero},
                   {&echo, 1, {2, 1}, {1, 0}, OutputBetweenUpdates::Zero}};
    c.projections = {{SourceKind::Gesture, 0, 0, WeightTensor::Routing, 0, MissingSource::Reject}};
    c.initialState = {{0.0f}, std::vector<float>(7, 0.0f)};
    Runtime runtime;
    std::string error;
    const Budget budget{2, 8, 3, 1, 0, 2, 6, 1};
    REQUIRE(runtime.prepare(c, budget, 48000, error));
    const std::array<std::string, 1> axes{"scalar"};
    const std::array<std::size_t, 1> shape{1};
    std::array<float, 1> audio{}, gesture{};
    Inputs input;
    input.audio = {axes, shape, audio, {}};
    input.gesture = {axes, shape, gesture, {}};
    double phase = 0;
    for (int n = 0; n < 256; ++n)
    {
        audio[0] = n == 0 ? 1.0f : 0.0f;
        gesture[0] = n % 2 == 0 ? 0.1f : -0.1f;
        REQUIRE(runtime.tick(input));
        REQUIRE(runtime.output()[0] == Catch::Approx(std::sin(2 * std::numbers::pi * phase)).margin(0.0002));
        const double frequency = 400.0 + 1200.0 * gesture[0];
        phase += frequency / 48000.0;
        phase -= std::floor(phase);
        const double expectedEcho = n > 0 && n % 7 == 0 ? std::pow(0.5, n / 7 - 1) : 0.0;
        REQUIRE(runtime.output()[1] == Catch::Approx(expectedEcho).margin(0.000001));
    }
    runtime.reset();
    REQUIRE(runtime.sample() == 0);
    REQUIRE(runtime.output()[0] == 0.0f);
    REQUIRE(runtime.output()[1] == 0.0f);
}
