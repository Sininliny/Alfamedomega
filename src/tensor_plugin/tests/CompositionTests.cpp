#include "TensorComposition.h"
#include "TensorExecution.h"
#include "alfa/tensor/Compiler.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <nlohmann/json.hpp>

namespace
{
using JSON = nlohmann::json;
using alfa::tensorhost::TensorComposition;
juce::String budget()
{
    // Explicit test capacity, not a named product tier policy.
    return JSON{{"active_primitives", 16},
                {"state_elements", 4096},
                {"parameter_elements", 256},
                {"routing_elements", 256},
                {"modulation_elements", 256},
                {"output_elements", 64},
                {"stored_tensor_elements", 16384},
                {"projection_terms", 256},
                {"boundary_conversion_bytes", 16384},
                {"host_alignment_bytes", 32768},
                {"compiler_model_bytes", 1000000}}
        .dump();
}
JSON reference(const char* name)
{
    return JSON::parse(
        juce::File(juce::String(ALFA_SOURCE_DIR)).getChildFile(name).loadFileAsString().toStdString());
}
JSON cell(double initial)
{
    auto document = reference("programs/fx.aom.json");
    document["instances"][0]["primitive"] = "math.feedback_cell";
    document["instances"][0]["initial_state"] = {initial};
    document["instances"][0]["input_channels"] = 1;
    document["instances"][0]["audio_bindings"] = JSON::array();
    document["tensors"]["parameters"]["values"] = {0.};
    document["budget"]["stored_tensor_elements"] = 3;
    return document;
}
TensorComposition::Binding link(const char* id, const char* from, const char* to, double weight)
{
    TensorComposition::Binding result;
    result.id = id;
    result.sourceFragment = from;
    result.destinationFragment = to;
    result.weight = weight;
    return result;
}
} // namespace

TEST_CASE("Composed fragment compiler models retain disjoint predictions recurrence and precision",
          "[tensor-host][composition][model]")
{
    auto instrument = reference("programs/instrument.aom.json");
    instrument["runtimeModel"] = reference("training/aom/testdata/runtime_gru.json");
    TensorComposition composition(48000, budget(), "Standard");
    juce::String error;
    REQUIRE(composition.add("A", instrument.dump(), error));
    REQUIRE(composition.clone("A", "B", error));
    const auto document = JSON::parse(composition.document().toStdString());
    REQUIRE(document["runtimeModels"].size() == 2);
    CHECK(document["runtimeModels"][1]["offsets"][1] == 2);
    std::string diagnostic;
    auto combined = alfa::tensorhost::prepareExecution(document.dump(), 48000, diagnostic);
    auto single = alfa::tensorhost::prepareExecution(instrument.dump(), 48000, diagnostic);
    REQUIRE(combined != nullptr);
    REQUIRE(single != nullptr);
    const std::array<std::string, 1> axes{"feature"};
    const std::array<size_t, 1> shape{2};
    const std::array<float, 2> midi{440.f, .8f};
    alfa::tensor::Inputs inputs;
    inputs.midi = {axes, shape, midi, {}};
    for (int i = 0; i < 1000; ++i)
    {
        double milliseconds = 0;
        const auto a = combined->tick(inputs, milliseconds), b = single->tick(inputs, milliseconds);
        REQUIRE(a.audioRendered);
        REQUIRE(a.modelAccepted);
        REQUIRE(b.modelAccepted);
        CHECK(combined->mixedOutput(0) == Catch::Approx(2 * single->mixedOutput(0)).margin(1e-6));
    }
    auto changed = document;
    changed["tensors"]["parameters"]["values"][1] = .2;
    auto edited = alfa::tensorhost::prepareExecution(changed.dump(), 48000, diagnostic);
    REQUIRE(edited != nullptr);
    REQUIRE(edited->adopt(*combined));
    auto rejected = document;
    rejected["budget"]["compiler_model_bytes"] = 1;
    CHECK(alfa::tensorhost::prepareExecution(rejected.dump(), 48000, diagnostic) == nullptr);
    rejected = document;
    rejected["runtimeModels"][1]["offsets"] = rejected["runtimeModels"][0]["offsets"];
    CHECK(alfa::tensorhost::prepareExecution(rejected.dump(), 48000, diagnostic) == nullptr);
    auto doubleDocument = document;
    doubleDocument["tier"] = "Pro";
    doubleDocument["scalar_type"] = "float64";
    REQUIRE(alfa::tensorhost::prepareExecution(doubleDocument.dump(), 48000, diagnostic) != nullptr);
}

TEST_CASE("Composition clones remap tensor state and output identities without losing original metadata",
          "[tensor-host][composition]")
{
    TensorComposition composition(48000, budget());
    auto fragment = cell(1);
    fragment["shared_initial_state"] = {7.};
    fragment["budget"]["state_elements"] = 2;
    fragment["parameter_locks"] = {1};
    fragment["metadata"] = {{"user", "retained"}};
    fragment["instances"][0]["clock"] = {{"period_samples", 2}, {"phase_samples", 1}};
    fragment["tensors"]["routing"] = {{"axes", {"condition", "time"}}, {"shape", {1, 1}}, {"values", {0.}}};
    fragment["instances"][0]["audio_bindings"] = JSON::array({{{"source", "state"},
                                                               {"source_element", 1},
                                                               {"destination_element", 0},
                                                               {"weights", "routing"},
                                                               {"weight_element", 0},
                                                               {"missing_source", "reject"}}});
    fragment["budget"]["routing_elements"] = 1;
    fragment["budget"]["projection_terms"] = 1;
    fragment["budget"]["stored_tensor_elements"] = 4;
    juce::String error;
    REQUIRE(composition.add("first", fragment.dump(), error));
    REQUIRE(composition.clone("first", "second", error));
    const auto combined = JSON::parse(composition.document().toStdString());
    REQUIRE(combined["instances"].size() == 2);
    CHECK(combined["instances"][0]["instance_id"] != combined["instances"][1]["instance_id"]);
    CHECK(combined["instances"][1]["activation"] == 1);
    CHECK(combined["instances"][1]["parameters"]["offset"] == 1);
    CHECK(combined["instances"][0]["audio_bindings"][0]["source_element"] == 2);
    CHECK(combined["instances"][1]["audio_bindings"][0]["source_element"] == 3);
    CHECK(combined["instances"][1]["audio_bindings"][0]["weight_element"] == 1);
    CHECK(combined["parameter_locks"] == JSON::array({1, 1}));
    CHECK(combined["composition"]["fragments"][1]["document"] == fragment);
    CHECK(combined["instances"][1]["clock"] == fragment["instances"][0]["clock"]);
    CHECK(combined["outputMix"][2][0] == 1);
    TensorComposition restored(48000, budget());
    REQUIRE(restored.restoreDocument(composition.document(), error));
    CHECK(restored.document() == composition.document());
    REQUIRE(restored.remove("first", error));
    const auto single = JSON::parse(restored.document().toStdString());
    CHECK(single["instances"][0]["fragment_id"] == "second");
    CHECK(single["instances"][0]["audio_bindings"][0]["source_element"] == 1);
    CHECK(single["instances"][0]["audio_bindings"][0]["weight_element"] == 0);
}

TEST_CASE("Explicit inter-fragment feedback executes committed history without graph ordering",
          "[tensor-host][composition]")
{
    TensorComposition composition(48000, budget());
    juce::String error;
    REQUIRE(composition.add("a", cell(1).dump(), error));
    REQUIRE(composition.add("b", cell(0).dump(), error));
    REQUIRE(composition.bind(link("a-to-b", "a", "b", 1), error));
    REQUIRE(composition.bind(link("b-to-a", "b", "a", .5), error));
    alfa::tensor::Runtime runtime;
    std::string nativeError;
    REQUIRE(alfa::tensor::compileJSON(composition.document().toStdString(), 48000, runtime, nativeError));
    const std::array<std::array<float, 2>, 5> expected{{{1, 0}, {0, 0}, {0, 1}, {0, 0}, {.5f, 0}}};
    for (const auto& frame : expected)
    {
        REQUIRE(runtime.tick({}));
        REQUIRE(runtime.output().size() == 2);
        CHECK(runtime.output()[0] == frame[0]);
        CHECK(runtime.output()[1] == frame[1]);
    }
    const auto retained = composition.document();
    CHECK_FALSE(composition.remove("a", error));
    CHECK(composition.document() == retained);
    REQUIRE(composition.unbind("a-to-b", error));
    REQUIRE(composition.unbind("b-to-a", error));
    REQUIRE(composition.remove("a", error));
    CHECK(composition.fragmentIds() == juce::StringArray{"b"});
}

TEST_CASE("Instrument and FX fragments share explicit recurrent information and survive precision conversion",
          "[tensor-host][composition][precision]")
{
    auto instrument = reference("programs/instrument.aom.json");
    auto fx = cell(0);
    TensorComposition composition(48000, budget(), "Pro");
    juce::String error;
    REQUIRE(composition.add("instrument", instrument.dump(), error));
    REQUIRE(composition.add("effect", fx.dump(), error));
    REQUIRE(composition.bind(link("excitation", "instrument", "effect", .7), error));
    auto feedback = link("return", "effect", "instrument", .01);
    feedback.destination = "parameter";
    feedback.destinationElement = 1;
    REQUIRE(composition.bind(feedback, error));
    alfa::tensor::Runtime64 runtime;
    std::string nativeError;
    REQUIRE(alfa::tensor::compileJSON(composition.document().toStdString(), 48000, runtime, nativeError));
    const std::array<std::string, 1> axes{"feature"};
    const std::array<std::size_t, 1> shape{2};
    const std::array<double, 2> midi{440., .8};
    alfa::tensor::Inputs64 inputs;
    inputs.midi = {axes, shape, midi, {}};
    double instrumentPeak = 0, effectPeak = 0;
    for (int sample = 0; sample < 512; ++sample)
    {
        REQUIRE(runtime.tick(inputs));
        CHECK(std::isfinite(runtime.output()[0]));
        CHECK(std::isfinite(runtime.output()[1]));
        instrumentPeak = std::max(instrumentPeak, std::abs(runtime.output()[0]));
        effectPeak = std::max(effectPeak, std::abs(runtime.output()[1]));
    }
    CHECK(instrumentPeak > .1);
    CHECK(effectPeak > .05);
    const auto lowered = JSON::parse(composition.document().toStdString());
    CHECK(lowered["scalar_type"] == "float64");
    CHECK(lowered["tier"] == "Pro");
    CHECK(lowered["instances"][1]["audio_bindings"][0]["source"] == "output_history");
}

TEST_CASE(
    "Composition preserves independent unequal-depth control matrices and rejects stale editing recipes",
    "[tensor-host][composition]")
{
    auto first = cell(0), second = cell(0);
    const auto layer = [](double value, double weight)
    {
        return JSON{
            {"values", {value}},
            {"ranges", {{-1., 1.}}},
            {"matrix", {{"axes", {"destination", "source"}}, {"shape", {1, 1}}, {"values", {weight}}}}};
    };
    first["control_layers"] = JSON::array({layer(.1, .5), layer(.2, .25)});
    second["control_layers"] = JSON::array({layer(.3, .75)});
    second["control_layers"][0]["labels"] = {"Bite"};
    second["control_layers"][0]["ranking"] = {{"model", "fragment-only"}, {"importance", {1.}}};
    first["budget"]["stored_tensor_elements"] = 7;
    second["budget"]["stored_tensor_elements"] = 5;
    TensorComposition composition(48000, budget());
    juce::String error;
    REQUIRE(composition.add("deep", first.dump(), error));
    REQUIRE(composition.add("shallow", second.dump(), error));
    const auto combined = JSON::parse(composition.document().toStdString());
    CHECK(combined["control_layers"][0]["matrix"]["shape"] == JSON::array({2, 1}));
    CHECK(combined["control_layers"][0]["matrix"]["values"] == JSON::array({.5, 0.}));
    CHECK(combined["control_layers"][1]["values"] == JSON::array({.2, .3}));
    CHECK(combined["control_layers"][1]["matrix"]["values"] == JSON::array({.25, 0., 0., .75}));
    // User names follow their controls; a fragment's learned ranking does not describe the composite.
    CHECK(combined["control_layers"][1]["labels"] == JSON::array({"", "Bite"}));
    CHECK_FALSE(combined["control_layers"][0].contains("labels"));
    CHECK_FALSE(combined["control_layers"][1].contains("ranking"));
    std::string nativeError;
    auto execution =
        alfa::tensorhost::prepareExecution(composition.document().toStdString(), 48000, nativeError);
    REQUIRE(execution != nullptr);
    REQUIRE(execution->updateControls({}));
    double modelMilliseconds = 0;
    REQUIRE(execution->tick({}, modelMilliseconds).audioRendered);
    auto edited = combined;
    edited["tensors"]["parameters"]["values"][0] = .125;
    REQUIRE(composition.restoreDocument(edited.dump(), error));
    CHECK(JSON::parse(composition.document().toStdString())["tensors"]["parameters"]["values"][0] == .125);
    CHECK(JSON::parse(composition.recipe().toStdString())["fragments"][0]["id"] == "deep");
    auto renamed = JSON::parse(composition.document().toStdString());
    renamed["control_layers"][1]["labels"][0] = "Body";
    REQUIRE(composition.restoreDocument(renamed.dump(), error));
    CHECK(JSON::parse(composition.document().toStdString())["control_layers"][1]["labels"] ==
          JSON::array({"Body", "Bite"}));
    auto cross = JSON::parse(composition.document().toStdString());
    cross["control_layers"][1]["matrix"]["values"][1] = .4;
    REQUIRE(composition.restoreDocument(cross.dump(), error));
    CHECK(JSON::parse(composition.recipe().toStdString())["control_bindings"].size() == 1);
    REQUIRE(composition.clone("shallow", "third", error));
    const auto expanded = JSON::parse(composition.document().toStdString());
    CHECK(expanded["control_layers"][1]["matrix"]["values"][1] == .4);
    const auto retained = composition.document();
    edited["instances"][0]["clock"]["period_samples"] = 2;
    CHECK_FALSE(composition.restoreDocument(edited.dump(), error));
    CHECK(composition.document() == retained);
}

TEST_CASE("Composition carries authored latency without inferring paths and permits explicit override",
          "[tensor-host][composition][latency]")
{
    TensorComposition composition(48000, budget());
    juce::String error;
    auto first = cell(0);
    first["host_latency_samples"] = 11;
    first["budget"]["host_alignment_bytes"] = 11 * 8;
    REQUIRE(composition.add("first", first.dump(), error));
    CHECK(JSON::parse(composition.recipe().toStdString())["host_latency_samples"] == 11);
    CHECK(JSON::parse(composition.document().toStdString())["host_latency_samples"] == 11);

    auto second = cell(0);
    second["host_latency_samples"] = 97;
    second["budget"]["host_alignment_bytes"] = 97 * 8;
    second["instances"][0]["clock"] = {{"period_samples", 8}, {"phase_samples", 3}};
    REQUIRE(composition.add("second", second.dump(), error));
    REQUIRE(composition.clone("second", "third", error));
    REQUIRE(composition.bind(link("recurrent", "first", "second", .5), error));
    auto recipe = JSON::parse(composition.recipe().toStdString());
    CHECK(recipe["host_latency_samples"] == 11);
    CHECK(recipe["fragments"][1]["document"]["host_latency_samples"] == 97);
    CHECK(recipe["fragments"][2]["document"]["host_latency_samples"] == 97);
    CHECK(JSON::parse(composition.document().toStdString())["host_latency_samples"] == 11);

    recipe["host_latency_samples"] = 23;
    REQUIRE(composition.restore(recipe.dump(), error));
    CHECK(JSON::parse(composition.recipe().toStdString())["host_latency_samples"] == 23);
    CHECK(JSON::parse(composition.document().toStdString())["host_latency_samples"] == 23);
    recipe.erase("host_latency_samples");
    REQUIRE(composition.restore(recipe.dump(), error));
    CHECK_FALSE(JSON::parse(composition.document().toStdString()).contains("host_latency_samples"));
}

TEST_CASE("Composition refuses invalid identities precision relations and budgets transactionally",
          "[tensor-host][composition]")
{
    TensorComposition composition(48000, budget());
    juce::String error;
    REQUIRE(composition.add("first", cell(0).dump(), error));
    const auto retained = composition.document();
    CHECK_FALSE(composition.clone("first", "first", error));
    auto badLink = link("invalid", "first", "absent", 1);
    CHECK_FALSE(composition.bind(badLink, error));
    CHECK(composition.document() == retained);
    auto doubleFragment = cell(0);
    doubleFragment["tier"] = "Pro";
    doubleFragment["scalar_type"] = "float64";
    doubleFragment["budget"]["boundary_conversion_bytes"] = 16384;
    CHECK_FALSE(composition.add("double", doubleFragment.dump(), error));
    CHECK(error.contains("precision"));
    CHECK(composition.document() == retained);
    auto recipe = JSON::parse(composition.recipe().toStdString());
    recipe["budget"]["state_elements"] = 0;
    CHECK_FALSE(composition.restore(recipe.dump(), error));
    CHECK(composition.document() == retained);
    auto modelFragment = cell(0);
    modelFragment["runtimeModel"] = JSON::object();
    CHECK_FALSE(composition.add("model", modelFragment.dump(), error));
    CHECK(error.isNotEmpty());
    CHECK(composition.document() == retained);
}

TEST_CASE("Model predictions retain user control authority across layers clocks locks and composition",
          "[tensor-host][composition][model][controls]")
{
    const auto layer = [](double value, double weight, size_t destinations)
    {
        std::vector<double> matrix(destinations, 0.);
        matrix.back() = weight;
        return JSON{
            {"values", {value}},
            {"ranges", {{-1., 1.}}},
            {"matrix",
             {{"axes", {"destination", "source"}}, {"shape", {destinations, 1}}, {"values", matrix}}}};
    };
    for (const auto* tier : {"Standard", "Pro"})
        for (const bool locked : {false, true})
            for (const bool composed : {false, true})
            {
                auto authored = reference("programs/instrument.aom.json");
                authored["tier"] = tier;
                authored["scalar_type"] = std::string(tier) == "Pro" ? "float64" : "float32";
                authored["budget"] = JSON::parse(budget().toStdString());
                authored["projections"] = JSON::array();
                authored["tensors"]["parameters"]["values"] = {400., .12};
                authored["tensors"]["routing"]["values"] = {0., 0.};
                authored["parameter_locks"] = {0, locked ? 1 : 0};
                authored["control_layers"] =
                    JSON::array({layer(0., 2., 1), layer(.3, 3., 1), layer(.1, .25, 2)});
                auto model = reference("training/aom/testdata/runtime_gru.json");
                for (auto& weights : model["weights"])
                    for (auto& value : weights)
                        value = 0.;
                model["weights"]["values.bias"] = {1., 400., .2, 0., 0.};
                model["weights"]["presence.bias"] = {10., 10., 10., 10., 10.};
                model["output_bounds"] = {{0., 1.}, {0., 1000.}, {0., 1.}, {-1., 1.}, {-1., 1.}};
                model["clock"] = {{"period_samples", 7}, {"phase_samples", 0}};
                authored["runtimeModel"] = model;
                auto expected = authored;
                expected.erase("runtimeModel");
                expected["tensors"]["parameters"]["values"][1] = locked ? .12 : .2;
                juce::String error;
                if (composed)
                {
                    TensorComposition actualComposition(48000, budget(), tier);
                    TensorComposition expectedComposition(48000, budget(), tier);
                    REQUIRE(actualComposition.add("deep", authored.dump(), error));
                    REQUIRE(expectedComposition.add("deep", expected.dump(), error));
                    authored["runtimeModel"]["clock"]["period_samples"] = 11;
                    authored["control_layers"].erase(authored["control_layers"].begin());
                    expected["control_layers"].erase(expected["control_layers"].begin());
                    REQUIRE(actualComposition.add("shallow", authored.dump(), error));
                    REQUIRE(expectedComposition.add("shallow", expected.dump(), error));
                    authored = JSON::parse(actualComposition.document().toStdString());
                    expected = JSON::parse(expectedComposition.document().toStdString());
                }
                std::string diagnostic;
                auto actual = alfa::tensorhost::prepareExecution(authored.dump(), 48000, diagnostic);
                auto oracle = alfa::tensorhost::prepareExecution(expected.dump(), 48000, diagnostic);
                REQUIRE(actual != nullptr);
                REQUIRE(oracle != nullptr);
                const std::array<std::string, 1> inputAxes{"feature"};
                const std::array<size_t, 1> inputShape{2};
                const std::array<float, 2> midiValues{440.f, .8f};
                alfa::tensor::Inputs inputs;
                inputs.midi = {inputAxes, inputShape, midiValues, {}};
                for (int sample = 0; sample < 512; ++sample)
                {
                    if (sample % 13 == 0)
                    {
                        const std::array<float, 1> x{sample % 26 == 0 ? .7f : -.8f};
                        REQUIRE(actual->updateControls(x));
                        REQUIRE(oracle->updateControls(x));
                    }
                    double milliseconds = 0;
                    const auto result = actual->tick(inputs, milliseconds);
                    REQUIRE(result.audioRendered);
                    REQUIRE(result.modelAccepted);
                    REQUIRE(oracle->tick(inputs, milliseconds).audioRendered);
                    CHECK(actual->mixedOutput(0) == Catch::Approx(oracle->mixedOutput(0)).margin(1e-6));
                }
            }
}

TEST_CASE("Scoped compiler loading validates containers identities and latent storage",
          "[tensor-host][composition][model][validation]")
{
    auto source = reference("programs/instrument.aom.json");
    source["runtimeModel"] = reference("training/aom/testdata/runtime_gru.json");
    TensorComposition composition(48000, budget());
    juce::String error;
    REQUIRE(composition.add("first", source.dump(), error));
    REQUIRE(composition.clone("first", "second", error));
    const auto valid = JSON::parse(composition.document().toStdString());
    std::string diagnostic;
    auto malformed = valid;
    malformed["runtimeModels"] = {{"first", valid["runtimeModels"][0]}};
    CHECK(alfa::tensorhost::prepareExecution(malformed.dump(), 48000, diagnostic) == nullptr);
    malformed = valid;
    malformed["runtimeModels"][1]["id"] = "first";
    CHECK(alfa::tensorhost::prepareExecution(malformed.dump(), 48000, diagnostic) == nullptr);
    malformed["runtimeModels"][1]["id"] = "";
    CHECK(alfa::tensorhost::prepareExecution(malformed.dump(), 48000, diagnostic) == nullptr);
    for (const auto& offsets :
         {JSON::array({0., 0, 0, 0}), JSON::array({-1, 0, 0, 0}), JSON::array({0, 0, 0, 0, 0})})
    {
        malformed = valid;
        malformed["runtimeModels"][0]["offsets"] = offsets;
        CHECK(alfa::tensorhost::prepareExecution(malformed.dump(), 48000, diagnostic) == nullptr);
    }
    for (const auto* capacity : {"compiler_model_bytes", "boundary_conversion_bytes", "host_alignment_bytes"})
    {
        malformed = valid;
        malformed["budget"][capacity] = -1;
        CHECK(alfa::tensorhost::prepareExecution(malformed.dump(), 48000, diagnostic) == nullptr);
        malformed["budget"][capacity] = .5;
        CHECK(alfa::tensorhost::prepareExecution(malformed.dump(), 48000, diagnostic) == nullptr);
    }
    auto global = source;
    global["budget"]["compiler_model_bytes"] = 1;
    CHECK(alfa::tensorhost::prepareExecution(global.dump(), 48000, diagnostic) == nullptr);
    malformed = valid;
    malformed["budget"]["stored_tensor_elements"] = 10;
    malformed["runtimeModels"][0]["controls"]["latentSound"] = {
        {"axes", {"latent"}}, {"shape", {1}}, {"values", {1.}}};
    CHECK(alfa::tensorhost::prepareExecution(malformed.dump(), 48000, diagnostic) == nullptr);
    malformed["budget"]["stored_tensor_elements"] = 11;
    REQUIRE(alfa::tensorhost::prepareExecution(malformed.dump(), 48000, diagnostic) != nullptr);
}

TEST_CASE("Execution reset restores authored tensors before the first delayed model event",
          "[tensor-host][model][reset][controls]")
{
    for (const auto* tier : {"Standard", "Pro"})
        for (const bool scoped : {false, true})
        {
            auto document = reference("programs/instrument.aom.json");
            document["tier"] = tier;
            document["budget"] = JSON::parse(budget().toStdString());
            auto model = reference("training/aom/testdata/runtime_gru.json");
            model["clock"] = {{"period_samples", 97}, {"phase_samples", 11}};
            document["runtimeModel"] = model;
            document["control_layers"] = JSON::array(
                {{{"values", {0.}},
                  {"ranges", {{-1., 1.}}},
                  {"matrix",
                   {{"axes", {"destination", "source"}}, {"shape", {2, 1}}, {"values", {0., .1}}}}}});
            if (scoped)
            {
                TensorComposition composition(48000, budget(), tier);
                juce::String error;
                REQUIRE(composition.add("model", document.dump(), error));
                document = JSON::parse(composition.document().toStdString());
            }
            std::string error;
            auto execution = alfa::tensorhost::prepareExecution(document.dump(), 48000, error);
            auto fresh = alfa::tensorhost::prepareExecution(document.dump(), 48000, error);
            REQUIRE(execution != nullptr);
            REQUIRE(fresh != nullptr);
            const std::array<std::string, 1> axes{"feature"};
            const std::array<size_t, 1> shape{2};
            const std::array<float, 2> midi{440.f, .8f};
            alfa::tensor::Inputs inputs;
            inputs.midi = {axes, shape, midi, {}};
            double milliseconds = 0;
            for (int sample = 0; sample < 256; ++sample)
                REQUIRE(execution->tick(inputs, milliseconds).audioRendered);
            execution->reset();
            for (int sample = 0; sample < 128; ++sample)
            {
                if (sample % 8 == 0)
                {
                    const std::array<float, 1> x{.3f};
                    REQUIRE(execution->updateControls(x));
                    REQUIRE(fresh->updateControls(x));
                }
                const auto a = execution->tick(inputs, milliseconds);
                const auto b = fresh->tick(inputs, milliseconds);
                REQUIRE(a.audioRendered);
                REQUIRE(b.audioRendered);
                CHECK(a.modelAccepted == b.modelAccepted);
                CHECK(execution->mixedOutput(0) == Catch::Approx(fresh->mixedOutput(0)).margin(1e-6));
            }
        }
}

TEST_CASE("Compiler imports reject feature widths unavailable at the native host boundary",
          "[tensor-host][model][validation]")
{
    auto document = reference("programs/instrument.aom.json");
    auto model = reference("training/aom/testdata/runtime_gru.json");
    model["input_widths"]["audio"] = 3;
    model["input_widths"]["midi"] = 0;
    model["weights"]["cell.weight_ih"] = std::vector<double>(27);
    model["budget"]["parameter_elements"] = 112;
    document["runtimeModel"] = model;
    std::string error;
    CHECK(alfa::tensorhost::prepareExecution(document.dump(), 48000, error) == nullptr);
    CHECK(error.find("source capacity") != std::string::npos);
    model["input_widths"]["audio"] = 0;
    model["input_widths"]["latent"] = 3;
    document["runtimeModel"] = model;
    CHECK(alfa::tensorhost::prepareExecution(document.dump(), 48000, error) == nullptr);
    CHECK(error.find("source capacity") != std::string::npos);
}
