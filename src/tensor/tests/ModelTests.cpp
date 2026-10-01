#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Model.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <type_traits>

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

TEST_CASE("Double compiler model keeps typed precision recurrent reset and scalar memory accounting",
          "[tensor][model][precision]")
{
    using namespace alfa::tensor;
    auto document = nlohmann::json::parse(fileContents("training/aom/testdata/runtime_gru.json"));
    auto program = nlohmann::json::parse(fileContents("programs/instrument.aom.json"));
    program["tier"] = "Pro";
    program["scalar_type"] = "float64";
    Configuration64 config;
    std::string error;
    REQUIRE(parseJSON(program.dump(), config, error));
    const auto hidden = document["hidden_width"].get<std::size_t>();
    const auto inputsCount = document["input_widths"]["midi"].get<std::size_t>();
    std::size_t parameterCount = 0;
    for (auto& [name, weights] : document["weights"].items())
    {
        parameterCount += weights.size();
        for (auto& weight : weights)
            weight = 0.;
    }
    constexpr double preciseWeight = .123456789011223;
    constexpr double preciseBias = .234567890112233;
    constexpr double preciseInput = .345678901122334;
    document["weights"]["cell.weight_ih"][2 * hidden * inputsCount] = preciseWeight;
    document["weights"]["values.weight"][0] = 1.;
    document["weights"]["values.bias"][0] = preciseBias;
    document["initial_state"] = std::vector<double>(hidden, 0.);
    const auto outputs = document["weights"]["values.bias"].size();
    const auto numericElements =
        parameterCount + inputsCount + 9 * hidden + 6 * outputs + config.parameters.values.size();
    document["budget"]["estimated_peak_inference_bytes"] = 1;
    document["budget"]["available_ram_bytes"] = numericElements * sizeof(float);
    // The same producer allowance can fit f32 but must reject the f64 storage.
    auto floatProgram = program;
    floatProgram["tier"] = "Standard";
    floatProgram["scalar_type"] = "float32";
    Configuration floatConfig;
    REQUIRE(parseJSON(floatProgram.dump(), floatConfig, error));
    TensorModel floatModel;
    REQUIRE(floatModel.prepare(document.dump(), floatConfig, error));
    TensorModel64 model;
    CHECK_FALSE(model.prepare(document.dump(), config, error));
    CHECK(error.find("RAM") != std::string::npos);
    document["budget"]["available_ram_bytes"] = numericElements * sizeof(double);
    REQUIRE(model.prepare(document.dump(), config, error));
    static_assert(std::is_same_v<decltype(model.state()), std::span<const double>>);
    static_assert(std::is_same_v<decltype(model.values()), std::span<const double>>);
    const std::array<std::string, 1> axes{"feature"};
    const std::array<std::size_t, 1> shape{2};
    const std::array<double, 2> midi{preciseInput, 0.};
    Inputs64 input;
    input.midi = {axes, shape, midi, {}};
    const auto excitation = std::tanh(preciseWeight * preciseInput);
    REQUIRE(model.infer(input));
    const auto first = model.values()[0];
    CHECK(model.state()[0] == Catch::Approx(.5 * excitation).margin(1e-15).epsilon(1e-14));
    CHECK(first == Catch::Approx(preciseBias + .5 * excitation).margin(1e-15).epsilon(1e-14));
    REQUIRE(model.infer(input));
    CHECK(model.state()[0] == Catch::Approx(.75 * excitation).margin(1e-15).epsilon(1e-14));
    CHECK(model.values()[0] != first);
    model.reset();
    CHECK(model.state()[0] == 0.);
    REQUIRE(model.infer(input));
    CHECK(model.values()[0] == first);
    Inputs floatInput;
    const std::array<float, 2> floatMidi{static_cast<float>(preciseInput), 0.f};
    floatInput.midi = {axes, shape, floatMidi, {}};
    REQUIRE(floatModel.infer(floatInput));
    CHECK(static_cast<double>(floatModel.values()[0]) != first);
}
