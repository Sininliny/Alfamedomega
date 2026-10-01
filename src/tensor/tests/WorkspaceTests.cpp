#include "alfa/tensor/Runtime.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>

namespace
{
template <class S>
void transition(const alfa::tensor::StepT<S>& s, std::span<S> next) noexcept
{
    std::fill(s.scratch.begin(), s.scratch.end(), s.localState[0]);
    next[0] = s.scratch[0] + static_cast<S>(s.scratch.size());
}
template <class S>
void readout(const alfa::tensor::StepT<S>& s, std::span<S> output) noexcept
{
    std::fill(s.scratch.begin(), s.scratch.end(), s.localState[0] + S(0.125));
    output[0] = s.scratch.back();
}
template <class S>
void verify()
{
    using namespace alfa::tensor;
    PrimitiveT<S> first{"workspace.first", 1, 1, transition<S>, readout<S>, 0, nullptr, 3};
    PrimitiveT<S> second{"workspace.second", 1, 1, transition<S>, readout<S>, 0, nullptr, 5};
    ConfigurationT<S> configuration;
    configuration.activation = {{"instance"}, {2}, {S(1), S(1)}, {}};
    for (auto* tensor : {&configuration.parameters, &configuration.routing, &configuration.modulation})
        *tensor = {{"coordinate"}, {0}, {}, {}};
    configuration.instances = {{&first, 0, {0, 0}, {1, 0}}, {&second, 1, {0, 0}, {1, 0}}};
    configuration.initialState = {{S(0)}, {S(0)}};
    Budget budget{2, 2, 0, 0, 0, 2, 2, 0, 0, 5};
    RuntimeT<S> runtime;
    std::string error;
    REQUIRE(runtime.prepare(configuration, budget, 48000, error));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[0] == S(3));
    REQUIRE(runtime.state()[1] == S(5));
    REQUIRE(runtime.output()[0] == S(.125));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.output()[0] == S(3.125));
    REQUIRE(runtime.output()[1] == S(5.125));
    const auto sample = runtime.sample();
    const auto bytes = runtime.numericStorageBytes();
    budget.scratchElements = 4;
    REQUIRE_FALSE(runtime.prepare(configuration, budget, 48000, error));
    REQUIRE(runtime.sample() == sample);
    REQUIRE(runtime.state()[0] == S(6));
    runtime.reset();
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.output()[0] == S(.125));
    // Capacity is one reused workspace, not a sum or extra recurrent state.
    first.scratchElements = second.scratchElements = 0;
    first.transition = second.transition = [](const StepT<S>& s, std::span<S> next) noexcept
    {
        next[0] = s.localState[0];
    };
    first.readout = second.readout = [](const StepT<S>& s, std::span<S> output) noexcept
    {
        output[0] = s.localState[0];
    };
    budget.scratchElements = 0;
    RuntimeT<S> baseline;
    REQUIRE(baseline.prepare(configuration, budget, 48000, error));
    REQUIRE(bytes - baseline.numericStorageBytes() == 5 * sizeof(S));
}
} // namespace
TEST_CASE("Typed numerical workspace is reused without changing shared recurrent state or rollback",
          "[tensor][workspace]")
{
    verify<float>();
    verify<double>();
}
