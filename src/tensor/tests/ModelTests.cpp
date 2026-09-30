#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Model.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace
{
std::string fileContents(const char* path)
{
    std::ifstream file(std::string(ALFA_SOURCE_DIR) + "/" + path);
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}
} // namespace

TEST_CASE("Native compiler GRU matches PyTorch golden predictions and recurrent context", "[tensor][model]")
{
    using namespace alfa::tensor;
    const auto document = fileContents("training/aom/testdata/runtime_gru.json");
    const auto fixture = nlohmann::json::parse(document);
    Configuration config;
    std::string error;
    REQUIRE(parseJSON(fileContents("programs/instrument.aom.json"), config, error));
    TensorModel model;
    REQUIRE(model.prepare(document, config, error));
    std::array<std::string, 1> axes{"feature"};
    for (const auto& golden : fixture.at("golden"))
    {
        const auto midi = golden.at("inputs").at("midi").get<std::vector<float>>();
        std::array<std::size_t, 1> shape{midi.size()};
        Inputs inputs;
        inputs.midi = {axes, shape, midi, {}};
        REQUIRE(model.infer(inputs));
        for (const auto& pair :
             {std::pair{model.values(), golden.at("values").get<std::vector<float>>()},
              std::pair{model.presenceLogits(), golden.at("presence_logits").get<std::vector<float>>()},
              std::pair{model.state(), golden.at("state").get<std::vector<float>>()}})
        {
            REQUIRE(pair.first.size() == pair.second.size());
            for (std::size_t index = 0; index < pair.first.size(); ++index)
                CHECK(pair.first[index] == Catch::Approx(pair.second[index]).margin(0.000001));
        }
    }
    model.reset();
    CHECK(model.state()[0] == 0.0f);
}

TEST_CASE("Compiler model execution uses its own event clock and independent memory budget",
          "[tensor][model]")
{
    using namespace alfa::tensor;
    const auto text = fileContents("training/aom/testdata/runtime_gru.json");
    Configuration config;
    Budget budget;
    std::string error;
    const auto program = fileContents("programs/instrument.aom.json");
    REQUIRE(parseJSON(program, config, error));
    REQUIRE(readBudgetJSON(program, budget, error));
    Runtime runtime;
    REQUIRE(runtime.prepare(config, budget, 48000, error));
    TensorModel model;
    REQUIRE(model.prepare(text, config, error));
    REQUIRE(model.apply({}, runtime)); // Sample 0 is not this model's phase2 event.
    CHECK(model.state()[0] == 0.0f);
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.tick({}));
    REQUIRE_FALSE(model.apply({}, runtime)); // Due event requires declared MIDI features.
    CHECK(model.state()[0] == 0.0f);
    auto changedMask = nlohmann::json::parse(text);
    changedMask["presence_threshold"] = 0.999f;
    TensorModel structural;
    REQUIRE(structural.prepare(changedMask.dump(), config, error));
    const std::array<std::string, 1> axes{"feature"};
    const std::array<std::size_t, 1> shape{2};
    const std::array<float, 2> midi{440, 0.7f};
    Inputs inputs;
    inputs.midi = {axes, shape, midi, {}};
    REQUIRE_FALSE(structural.apply(inputs, runtime));
    CHECK(structural.state()[0] == 0.0f); // Rejected publication also retains neural state.
    auto refused = nlohmann::json::parse(text);
    refused["budget"]["available_ram_bytes"] = 1;
    REQUIRE_FALSE(model.prepare(refused.dump(), config, error));
    CHECK_FALSE(error.empty());
}
