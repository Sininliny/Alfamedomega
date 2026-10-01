#include "PrimitiveFragmentPanel.h"
#include "alfa/tensor/Compiler.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace
{
using JSON = nlohmann::json;
using alfa::tensorhost::PrimitiveFragmentPanel;
template <typename Control>
Control& find(juce::Component& panel, const juce::String& id)
{
    for (auto* child : panel.getChildren())
        if (child->getComponentID() == id)
        {
            if (auto* typed = dynamic_cast<Control*>(child))
                return *typed;
        }
    throw std::runtime_error("Primitive input unavailable: " + id.toStdString());
}
void text(juce::Component& panel, const char* id, const juce::String& value)
{
    find<juce::TextEditor>(panel, id).setText(value);
}
void select(juce::Component& panel, const char* id, const juce::String& value)
{
    auto& combo = find<juce::ComboBox>(panel, id);
    for (int item = 0; item < combo.getNumItems(); ++item)
        if (combo.getItemText(item) == value)
        {
            combo.setSelectedId(combo.getItemId(item), juce::sendNotificationSync);
            return;
        }
    throw std::runtime_error("Primitive choice unavailable: " + value.toStdString());
}
void create(juce::Component& panel)
{
    for (auto* child : panel.getChildren())
        if (auto* button = dynamic_cast<juce::Button*>(child);
            button != nullptr && button->getButtonText() == "Create fragment")
        {
            button->triggerClick();
            juce::MessageManager::getInstance()->runDispatchLoopUntil(25);
            return;
        }
    throw std::runtime_error("Create primitive button unavailable");
}
void inputs(juce::Component& panel)
{
    text(panel, "primitive.state_size", "1");
    text(panel, "primitive.input_channels", "1");
    text(panel, "primitive.period", "3");
    text(panel, "primitive.phase", "1");
    text(panel, "primitive.parameters", "[0.5]");
    text(panel, "primitive.routing", "[1]");
    text(
        panel, "primitive.audio_bindings",
        R"([{"source":"audio","source_element":0,"destination_element":0,"weights":"routing","weight_element":0,"missing_source":"reject"}])");
    select(panel, "primitive.between", "hold");
    text(panel, "primitive.budget",
         JSON{{"active_primitives", 1},
              {"state_elements", 32},
              {"parameter_elements", 8},
              {"routing_elements", 1},
              {"modulation_elements", 0},
              {"output_elements", 8},
              {"stored_tensor_elements", 32},
              {"projection_terms", 1},
              {"boundary_conversion_bytes", 16384}}
             .dump());
}
} // namespace

TEST_CASE("Primitive authoring exposes the full catalog and creates explicit feedback clocks and operands",
          "[tensor-host][primitive-fragment]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    int count = 0;
    juce::String published;
    PrimitiveFragmentPanel panel(48000, "Standard",
                                 [&](const juce::String& sound, juce::String&)
                                 {
                                     ++count;
                                     published = sound;
                                     return true;
                                 });
    const auto& operations = find<juce::ComboBox>(panel, "primitive.operation");
    CHECK(operations.getNumItems() > 50);
    select(panel, "primitive.operation", "math.feedback_cell");
    CHECK(find<juce::TextEditor>(panel, "primitive.metadata").getText().contains("feedback"));
    CHECK(find<juce::TextEditor>(panel, "primitive.metadata").getText().contains("Cost per update"));
    inputs(panel);
    create(panel);
    REQUIRE(count == 1);
    const auto sound = JSON::parse(published.toStdString());
    CHECK(sound["scalar_type"] == "float32");
    CHECK(sound["instances"][0]["clock"]["period_samples"] == 3);
    CHECK(sound["instances"][0]["clock"]["phase_samples"] == 1);
    CHECK(sound["instances"][0]["initial_state"] == JSON::array({0.}));
    CHECK(sound["outputMix"].empty()); // Creation never invents an audible path.
    CHECK(sound["instances"][0]["audio_bindings"][0]["source"] == "audio");
    alfa::tensor::Runtime runtime;
    std::string error;
    REQUIRE(alfa::tensor::compileJSON(published.toStdString(), 48000, runtime, error));
    const std::array<std::string, 1> axes{"channel"};
    const std::array<std::size_t, 1> shape{1};
    const std::array<float, 1> audio{1};
    alfa::tensor::Inputs input;
    input.audio = {axes, shape, audio, {}};
    REQUIRE(runtime.tick(input));
    REQUIRE(runtime.tick(input));
    for (int sample = 2; sample <= 4; ++sample)
        REQUIRE(runtime.tick(input));
    CHECK(runtime.output()[0] == 1);
    panel.setVisible(true);
    const auto snapshotFile = juce::File(juce::String(ALFA_SOURCE_DIR))
                                  .getChildFile("build")
                                  .getChildFile("primitive-panel-" + juce::Uuid().toString() + ".png");
    auto snapshotStream = snapshotFile.createOutputStream();
    REQUIRE(snapshotStream != nullptr);
    REQUIRE(juce::PNGImageFormat{}.writeImageToStream(panel.createComponentSnapshot(panel.getLocalBounds()),
                                                      *snapshotStream));
}

TEST_CASE("Primitive authoring creates uniformly double production ring state without fixed execution rate",
          "[tensor-host][primitive-fragment][precision]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::String published;
    PrimitiveFragmentPanel panel(48000, "Pro",
                                 [&](const juce::String& sound, juce::String&)
                                 {
                                     published = sound;
                                     return true;
                                 });
    inputs(panel);
    select(panel, "primitive.operation", "time.ring_delay.v1");
    text(panel, "primitive.state_size", "6");
    text(panel, "primitive.parameters", "[2,0.25]");
    text(panel, "primitive.initial_state", "[0,0,0.125,0,0,0]");
    text(panel, "primitive.period", "7");
    text(panel, "primitive.phase", "2");
    text(panel, "primitive.output_mix", "[[0,1,0.75]]");
    auto budget = JSON::parse(find<juce::TextEditor>(panel, "primitive.budget").getText().toStdString());
    budget["sparse_state_writes"] = 3;
    text(panel, "primitive.budget", budget.dump());
    create(panel);
    INFO(find<juce::Label>(panel, "primitive.status").getText().toStdString());
    REQUIRE(published.isNotEmpty());
    const auto sound = JSON::parse(published.toStdString());
    CHECK(sound["tier"] == "Pro");
    CHECK(sound["scalar_type"] == "float64");
    CHECK(sound["instances"][0]["clock"] == JSON({{"period_samples", 7}, {"phase_samples", 2}}));
    CHECK(sound["outputMix"] == JSON::array({JSON::array({0, 1, .75})}));
    alfa::tensor::Runtime64 runtime;
    std::string error;
    REQUIRE(alfa::tensor::compileJSON(published.toStdString(), 48000, runtime, error));
    CHECK(runtime.state().size() == 6);
    CHECK(runtime.numericStorageBytes() >= 6 * sizeof(double));
}

TEST_CASE("Primitive authoring refuses malformed clocks state bindings and callback publication",
          "[tensor-host][primitive-fragment]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    int count = 0;
    PrimitiveFragmentPanel panel(48000, "Standard",
                                 [&](const juce::String&, juce::String& error)
                                 {
                                     ++count;
                                     error = "Explicit receiver refusal";
                                     return false;
                                 });
    select(panel, "primitive.operation", "math.feedback_cell");
    inputs(panel);
    text(panel, "primitive.parameters", "[]");
    create(panel);
    CHECK(count == 0);
    CHECK(find<juce::Label>(panel, "primitive.status").getText().contains("documented minimum"));
    text(panel, "primitive.parameters", "[0.5]");
    text(panel, "primitive.phase", "3");
    create(panel);
    CHECK(count == 0);
    text(panel, "primitive.phase", "1");
    text(panel, "primitive.state_size", "999999999");
    create(panel);
    CHECK(count == 0);
    CHECK(find<juce::Label>(panel, "primitive.status").getText().contains("before allocation"));
    text(panel, "primitive.state_size", "1");
    text(panel, "primitive.initial_state", "[0,0]");
    create(panel);
    CHECK(count == 0);
    text(panel, "primitive.initial_state", "");
    text(
        panel, "primitive.audio_bindings",
        R"([{"source":"audio","source_element":0,"destination_element":2,"weights":"routing","weight_element":0,"missing_source":"reject"}])");
    create(panel);
    CHECK(count == 0);
    inputs(panel);
    create(panel);
    CHECK(count == 1);
    CHECK(find<juce::Label>(panel, "primitive.status").getText() == "Explicit receiver refusal");
}

TEST_CASE("WAV primitive assets preserve the selected channel and source rate with bounded decoding",
          "[tensor-host][primitive-fragment]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    const auto file = juce::File(juce::String(ALFA_SOURCE_DIR))
                          .getChildFile("build")
                          .getChildFile("primitive-wav-" + juce::Uuid().toString() + ".wav");
    REQUIRE(file.getParentDirectory().createDirectory().wasOk());
    {
        auto stream = file.createOutputStream();
        REQUIRE(stream != nullptr);
        stream->write("RIFF", 4);
        stream->writeInt(52);
        stream->write("WAVEfmt ", 8);
        stream->writeInt(16);
        stream->writeShort(1);
        stream->writeShort(2);
        stream->writeInt(22050);
        stream->writeInt(22050 * 4);
        stream->writeShort(4);
        stream->writeShort(16);
        stream->write("data", 4);
        stream->writeInt(16);
        for (const auto value : std::array<short, 8>{8192, -16384, 4096, -8192, 0, 16384, -4096, 8192})
            stream->writeShort(value);
    }
    juce::String published;
    PrimitiveFragmentPanel panel(48000, "Standard",
                                 [&](const juce::String& sound, juce::String&)
                                 {
                                     published = sound;
                                     return true;
                                 });
    inputs(panel);
    select(panel, "primitive.operation", "math.feedback_cell");
    juce::String error;
    REQUIRE(panel.loadWavAsset(file, 1, error));
    const auto assets =
        JSON::parse(find<juce::TextEditor>(panel, "primitive.assets").getText().toStdString());
    REQUIRE(assets.size() == 1);
    CHECK(assets[0]["values"] == JSON::array({-.5, -.25, .5, .25}));
    CHECK(assets[0]["shape"] == JSON::array({4}));
    create(panel);
    REQUIRE(published.isNotEmpty());
    const auto sound = JSON::parse(published.toStdString());
    CHECK(sound["asset_sources"][0]["sample_rate"] == 22050);
    CHECK(sound["asset_sources"][0]["source_channel"] == 1);
    CHECK(sound["instances"][0]["clock"]["period_samples"] == 3);
    const auto retained = find<juce::TextEditor>(panel, "primitive.assets").getText();
    CHECK_FALSE(panel.loadWavAsset(file, 2, error));
    CHECK(find<juce::TextEditor>(panel, "primitive.assets").getText() == retained);
    auto budget = JSON::parse(find<juce::TextEditor>(panel, "primitive.budget").getText().toStdString());
    budget["stored_tensor_elements"] = 5;
    text(panel, "primitive.budget", budget.dump());
    CHECK_FALSE(panel.loadWavAsset(file, 0, error));
    CHECK(error.contains("before WAV decoding"));
    CHECK(find<juce::TextEditor>(panel, "primitive.assets").getText() == retained);
    CHECK(file.deleteFile());
}
