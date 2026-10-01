#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <array>

TEST_CASE("Audio operand tensors read previous outputs simultaneously and retain state on edits",
          "[tensor][audio-routing]")
{
    using namespace alfa::tensor;
    const auto cell = kernels::feedbackCell();
    Configuration config;
    config.activation = {{"primitive"}, {2}, {1, 1}, {}};
    config.parameters = {{"parameter"}, {2}, {0, 0}, {}};
    config.routing = {{"weight"}, {1}, {1}, {}};
    config.modulation = {};
    config.instances = {{&cell, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero},
                        {&cell, 1, {1, 1}, {1, 0}, OutputBetweenUpdates::Zero}};
    config.instances[1].inputChannels = 1;
    config.instances[1].audioBindings = {
        {SourceKind::OutputHistory, 0, 0, WeightTensor::Routing, 0, MissingSource::Reject}};
    config.initialState = {{0}, {0}};
    Budget budget{2, 2, 2, 1, 0, 2, 6, 1};
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(config, budget, 48000, error));
    std::array<std::string, 1> axes{"channel"};
    std::array<size_t, 1> shape{1};
    std::array<float, 1> audio{1};
    Inputs input;
    input.audio = {axes, shape, audio, {}};
    REQUIRE(runtime.tick(input));
    CHECK(runtime.output()[1] == 0);
    audio[0] = 0;
    REQUIRE(runtime.tick(input));
    CHECK(runtime.output()[0] == 1);
    CHECK(runtime.output()[1] == 0);
    REQUIRE(runtime.tick(input));
    CHECK(runtime.output()[1] == 0);
    REQUIRE(runtime.tick(input));
    CHECK(runtime.output()[1] == 1);
    Configuration reparsed;
    REQUIRE(parseJSON(serializeJSON(config), reparsed, error));
    CHECK(reparsed.instances[1].audioBindings[0].source == SourceKind::OutputHistory);
    Runtime changed;
    REQUIRE(changed.prepare(reparsed, budget, 48000, error));
    REQUIRE(changed.adoptStateFrom(runtime));
    CHECK(changed.sample() == runtime.sample());
    CHECK(changed.output()[1] == runtime.output()[1]);
    budget.projectionTerms = 0;
    REQUIRE_FALSE(changed.prepare(config, budget, 48000, error));
}
