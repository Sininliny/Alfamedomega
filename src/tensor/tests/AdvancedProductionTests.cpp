#include "alfa/tensor/AdvancedProduction.h"
#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Kernels.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <nlohmann/json.hpp>
#include <numbers>

namespace
{
using namespace alfa::tensor;
template <class S>
ConfigurationT<S> config(const PrimitiveT<S>& primitive, std::vector<S> parameters)
{
    ConfigurationT<S> c;
    c.activation = {{"instance"}, {1}, {1}, {}};
    const auto count = parameters.size();
    c.parameters = {{"parameter"}, {count}, std::move(parameters), {}};
    c.instances = {{&primitive, 0, {0, count}, {1, 0}, OutputBetweenUpdates::Zero}};
    c.initialState = {std::vector<S>(primitive.stateElements)};
    return c;
}
template <class S>
bool tick(RuntimeT<S>& runtime, S sample)
{
    const std::array<std::string, 1> axes{"channel"};
    const std::array<size_t, 1> shape{1};
    const std::array<S, 1> audio{sample};
    InputsT<S> in;
    in.audio = {axes, shape, audio, {}};
    return runtime.tick(in);
}
template <class S>
void convolutionTest()
{
    const size_t b = 16;
    std::vector<S> ir(257);
    for (size_t i = 0; i < ir.size(); ++i)
        ir[i] =
            static_cast<S>(std::cos(.13 * static_cast<double>(i)) * std::exp(-.03 * static_cast<double>(i)));
    auto primitive = advanced::partitionedPrimitive<S>(b, (ir.size() - 1) / b + 1);
    auto c = config(primitive, {static_cast<S>(b)});
    c.instances[0].assets = {advanced::partitionIR(TensorT<S>{{"sample"}, {ir.size()}, ir, {}}, b)};
    Budget budget{1,
                  primitive.stateElements,
                  1,
                  0,
                  0,
                  1,
                  c.instances[0].assets[0].values.size() + 2,
                  0,
                  primitive.sparseWriteCapacity,
                  primitive.scratchElements};
    RuntimeT<S> runtime;
    std::string error;
    REQUIRE(runtime.prepare(c, budget, 48000, error));
    std::vector<S> input(1000);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<S>(std::sin(.071 * static_cast<double>(i)) +
                                  .2 * std::cos(.3 * static_cast<double>(i)));
    for (size_t sample = 0; sample < input.size(); ++sample)
    {
        REQUIRE(tick(runtime, input[sample]));
        double expected = 0;
        if (sample >= b)
            for (size_t tap = 0; tap < ir.size() && tap <= sample - b; ++tap)
                expected += static_cast<double>(ir[tap]) * input[sample - b - tap];
        CHECK(runtime.output()[0] == Catch::Approx(expected).margin(std::is_same_v<S, float> ? 2e-5 : 2e-12));
    }
    runtime.reset();
    for (size_t sample = 0; sample < 400; ++sample)
    {
        REQUIRE(tick(runtime, sample == 0 ? S(1) : S(0)));
        const S expected = sample >= b && sample - b < ir.size() ? ir[sample - b] : S(0);
        CHECK(runtime.output()[0] == Catch::Approx(expected).margin(std::is_same_v<S, float> ? 2e-6 : 1e-12));
    }
    // Compiler accepts authored time-domain IR and allocates only the exact frame workspaces.
    c.instances[0].assets = {{{"sample"}, {ir.size()}, ir, {}}};
    auto document = nlohmann::json::parse(serializeJSON(c));
    document["budget"] = {{"active_primitives", 1},
                          {"state_elements", primitive.stateElements},
                          {"parameter_elements", 1},
                          {"routing_elements", 0},
                          {"modulation_elements", 0},
                          {"output_elements", 1},
                          {"stored_tensor_elements", budget.storedTensorElements},
                          {"projection_terms", 0},
                          {"sparse_state_writes", primitive.sparseWriteCapacity},
                          {"scratch_elements", primitive.scratchElements}};
    auto text = document.dump();
    RuntimeT<S> compiled;
    REQUIRE(compileJSON(text, 48000, compiled, error));
    REQUIRE(tick(compiled, S(1)));
    CHECK(compiled.output()[0] == 0);
}
template <class S>
void aliasTest()
{
    PrimitiveT<S> primitive;
    std::string error;
    REQUIRE(advanced::createPrimitive("pitchtime.asset_resample_antialias.v1", 1, primitive, error));
    auto c = config(primitive, {S(96000), S(1), S(32)});
    c.initialState = {{S(1000)}};
    std::vector<S> samples(10000);
    const auto run = [&](double cycles)
    {
        for (size_t i = 0; i < samples.size(); ++i)
            samples[i] = static_cast<S>(std::sin(2 * std::numbers::pi * cycles * static_cast<double>(i)));
        c.instances[0].assets = {{{"sample"}, {samples.size()}, samples, {}}};
        RuntimeT<S> runtime;
        Budget budget{1, 1, 3, 0, 0, 1, samples.size() + 4, 0};
        REQUIRE(runtime.prepare(c, budget, 48000, error));
        double energy = 0;
        for (int i = 0; i < 2048; ++i)
        {
            REQUIRE(tick(runtime, S(0)));
            energy += static_cast<double>(runtime.output()[0]) * runtime.output()[0];
        }
        return std::sqrt(energy / 2048);
    };
    CHECK(run(.4) < .001);
    CHECK(run(.025) > .69);
}
} // namespace
TEST_CASE("Partitioned convolution matches independent direct convolution across partition boundaries",
          "[tensor][advanced]")
{
    convolutionTest<float>();
    convolutionTest<double>();
}
TEST_CASE("Ratio-dependent sinc playback suppresses folded aliases and retains passband",
          "[tensor][advanced]")
{
    aliasTest<float>();
    aliasTest<double>();
}
TEST_CASE("Modal rotation preserves long double decay and explicit mode count", "[tensor][advanced]")
{
    Primitive64 primitive;
    std::string error;
    REQUIRE(advanced::createPrimitive("physical.modal_rotation.v1", 2, primitive, error));
    auto c = config(primitive, {1000., 1., 1.});
    c.initialState = {{1., 0.}};
    Runtime64 runtime;
    REQUIRE(runtime.prepare(c, {1, 2, 3, 0, 0, 1, 4, 0}, 96000, error));
    for (int i = 0; i < 96000; ++i)
        REQUIRE(tick(runtime, 0.));
    CHECK(std::hypot(runtime.state()[0], runtime.state()[1]) == Catch::Approx(.001).epsilon(1e-8));
    c.parameters.values[1] = -1;
    Runtime64 invalid;
    REQUIRE(invalid.prepare(c, {1, 2, 3, 0, 0, 1, 4, 0}, 96000, error));
    CHECK_FALSE(tick(invalid, 0.));
    CHECK(invalid.state()[0] == 1.);
}
TEST_CASE("Live state takeover swaps buffers and refuses identity clock asset or operand changes",
          "[tensor][state-transfer]")
{
    const auto primitive = kernels::sine();
    auto c = config(primitive, {440.f, .5f});
    c.instances[0].instanceId = "voice/A";
    Runtime old;
    std::string error;
    Budget budget{1, 1, 2, 0, 0, 1, 3, 0};
    REQUIRE(old.prepare(c, budget, 48000, error));
    for (int i = 0; i < 100; ++i)
        REQUIRE(tick(old, 0.f));
    const auto* state = old.state().data();
    const auto phase = old.state()[0];
    c.parameters.values[1] = .2f;
    Runtime changed;
    REQUIRE(changed.prepare(c, budget, 48000, error));
    REQUIRE(changed.takeStateFrom(old));
    CHECK(changed.state().data() == state);
    CHECK(changed.state()[0] == phase);
    CHECK(changed.sample() == 100);
    for (int kind = 0; kind < 3; ++kind)
    {
        auto structural = c;
        if (kind == 0)
            structural.instances[0].instanceId = "voice/B";
        if (kind == 1)
            structural.instances[0].clock = {2, 0};
        if (kind == 2)
            structural.initialState = {{.3f}};
        Runtime next;
        REQUIRE(next.prepare(structural, budget, 48000, error));
        CHECK_FALSE(next.takeStateFrom(changed));
        CHECK(changed.state().data() == state);
    }
    auto switched = c;
    switched.activation = {{"instance"}, {2}, {1.f, 0.f}, {}};
    switched.instances.push_back(switched.instances[0]);
    switched.instances[1].activationIndex = 1;
    switched.instances[1].instanceId = "voice/B";
    switched.initialState.push_back(switched.initialState[0]);
    budget.storedTensorElements = 4;
    Runtime first, second;
    REQUIRE(first.prepare(switched, budget, 48000, error));
    switched.activation.values = {0.f, 1.f};
    REQUIRE(second.prepare(switched, budget, 48000, error));
    CHECK_FALSE(second.takeStateFrom(first));
}

TEST_CASE("Tuned waveguide compensates loop filter and dispersion phase at the requested pitch",
          "[tensor][advanced]")
{
    Primitive64 primitive;
    std::string error;
    REQUIRE(advanced::createPrimitive("physical.tuned_waveguide.v1", 263, primitive, error));
    auto c = config(primitive, {480., .9999, .15, .2});
    Runtime64 runtime;
    REQUIRE(runtime.prepare(c, {1, 263, 4, 0, 0, 1, 5, 0, 8, 0}, 48000, error));
    std::vector<double> samples(8192);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        REQUIRE(tick(runtime, i == 0 ? 1. : 0.));
        samples[i] = runtime.output()[0];
    }
    double best = -1;
    size_t period = 0;
    for (size_t lag = 90; lag <= 110; ++lag)
    {
        double sum = 0;
        for (size_t i = 4096; i < samples.size(); ++i)
            sum += samples[i] * samples[i - lag];
        if (sum > best)
        {
            best = sum;
            period = lag;
        }
    }
    CHECK(period == 100);
    CHECK(best > 0);
    const auto state = runtime.state()[2];
    c.parameters.values[0] = 1.;
    Runtime64 invalid;
    REQUIRE(invalid.prepare(c, {1, 263, 4, 0, 0, 1, 5, 0, 8, 0}, 48000, error));
    CHECK_FALSE(tick(invalid, 0.));
    CHECK(std::isfinite(state));
    c.parameters.values = {480., 0., 0., 0.}; // Integer roundtrip; no excitation/feedback.
    c.initialState[0][3] = .25;               // Residue from a preceding fractional-delay setting.
    Runtime64 integerDelay;
    REQUIRE(integerDelay.prepare(c, {1, 263, 4, 0, 0, 1, 5, 0, 8, 0}, 48000, error));
    for (int i = 0; i < 8; ++i)
    {
        REQUIRE(tick(integerDelay, 0.));
        CHECK(integerDelay.output()[0] == 0.);
    }
    c.parameters.values[1] = -.9;
    c.initialState[0][3] = 0.;
    Runtime64 inverted;
    REQUIRE(inverted.prepare(c, {1, 263, 4, 0, 0, 1, 5, 0, 8, 0}, 48000, error));
    for (int i = 0; i <= 50; ++i)
    {
        REQUIRE(tick(inverted, i == 0 ? 1. : 0.));
        CHECK(inverted.output()[0] == Catch::Approx(i == 50 ? 1. : 0.).margin(1e-12));
    }
}

TEST_CASE("Phase-locked frame processor preserves Hermitian identity and emits exactly one delayed frame",
          "[tensor][advanced]")
{
    constexpr size_t n = 32;
    Primitive64 primitive;
    std::string error;
    REQUIRE(advanced::createPrimitive("pitchtime.phase_vocoder_locked.v1", 4 * n + 2, primitive, error));
    auto c = config(primitive, {8., 8., 1.});
    Runtime64 runtime;
    REQUIRE(runtime.prepare(c, {1, 4 * n + 2, 3, 0, 0, 2 * n + 1, 4, 0, 4 * n + 2, 6 * n}, 48000, error));
    std::vector<double> frame(2 * n + 1);
    frame[4] = .5;
    frame[5] = .25;
    frame[2 * (n - 2)] = .5;
    frame[2 * (n - 2) + 1] = -.25;
    frame[2 * n] = 1;
    const std::array<std::string, 1> axes{"channel"};
    const std::array<size_t, 1> shape{frame.size()};
    Inputs64 input;
    input.audio = {axes, shape, frame, {}};
    REQUIRE(runtime.tick(input));
    CHECK(runtime.output()[2 * n] == 0);
    frame[2 * n] = 0;
    REQUIRE(runtime.tick(input));
    CHECK(runtime.output()[2 * n] == 1);
    for (size_t k = 0; k < 2 * n; ++k)
        CHECK(runtime.output()[k] == Catch::Approx(frame[k]).margin(1e-12));
    REQUIRE(runtime.tick(input));
    CHECK(runtime.output()[2 * n] == 0);
    for (auto value : runtime.output())
        CHECK(value == 0);
    c.parameters.values[2] = 2.;
    Runtime64 pitched;
    REQUIRE(pitched.prepare(c, {1, 4 * n + 2, 3, 0, 0, 2 * n + 1, 4, 0, 4 * n + 2, 6 * n}, 48000, error));
    frame[2 * n] = 1;
    REQUIRE(pitched.tick(input));
    frame[2 * n] = 0;
    REQUIRE(pitched.tick(input));
    CHECK(std::hypot(pitched.output()[8], pitched.output()[9]) ==
          Catch::Approx(std::hypot(.5, .25)).margin(1e-12));
    CHECK(pitched.output()[8] == pitched.output()[2 * (n - 4)]);
    CHECK(pitched.output()[9] == -pitched.output()[2 * (n - 4) + 1]);
}
