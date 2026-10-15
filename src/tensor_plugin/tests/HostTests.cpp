#include "TensorProcessor.h"
#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Kernels.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

namespace
{
juce::String document(bool midiProjection = false)
{
    using namespace alfa::tensor;
    const auto cell = kernels::feedbackCell();
    Configuration c;
    c.activation = {{"primitive"}, {1}, {1}, {}};
    c.parameters = {{"parameter"}, {1}, {0}, {}};
    c.routing = {{"weight"}, {1}, {0.25f}, {}};
    c.modulation = {{}, {}, {}, {}};
    c.instances = {{&cell, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero}};
    c.initialState = {{0}};
    if (midiProjection)
        c.projections = {{SourceKind::Midi, 1, 0, WeightTensor::Routing, 0, MissingSource::Zero}};
    auto j = nlohmann::json::parse(serializeJSON(c));
    j["budget"] = {{"active_primitives", 1},      {"state_elements", 1},
                   {"parameter_elements", 1},     {"routing_elements", 1},
                   {"modulation_elements", 0},    {"output_elements", 1},
                   {"stored_tensor_elements", 3}, {"projection_terms", midiProjection ? 1 : 0}};
    j["outputMix"] = {{0, 0, 1}, {0, 1, 1}};
    return j.dump();
}
} // namespace
TEST_CASE("Built VST3 loads and renders in an independent plugin host", "[.vst3-binary]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    const auto path = juce::SystemStats::getEnvironmentVariable("ALFA_VST3_BINARY", {});
    REQUIRE(path.isNotEmpty());
    juce::VST3PluginFormat format;
    juce::OwnedArray<juce::PluginDescription> types;
    format.findAllTypesForFile(types, path);
    REQUIRE(types.size() == 1);
    juce::String error;
    auto plugin = format.createInstanceFromDescription(*types[0], 48000, 128, error);
    INFO(error.toStdString());
    REQUIRE(plugin != nullptr);
    plugin->prepareToPlay(48000, 128);
    juce::AudioBuffer<float> audio(2, 128);
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.75f), 0);
    double energy = 0;
    for (int block = 0; block < 20; ++block)
    {
        audio.clear();
        plugin->processBlock(audio, midi);
        midi.clear();
        for (int channel = 0; channel < audio.getNumChannels(); ++channel)
            for (int sample = 0; sample < audio.getNumSamples(); ++sample)
            {
                const auto value = audio.getSample(channel, sample);
                REQUIRE(std::isfinite(value));
                energy += value * value;
            }
    }
    CHECK(energy > 0.01);
    juce::MemoryBlock state;
    plugin->getStateInformation(state);
    REQUIRE(state.getSize() > 0);
    plugin->setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    std::unique_ptr<juce::AudioProcessorEditor> editor(plugin->createEditorIfNeeded());
    REQUIRE(editor != nullptr);
    CHECK(editor->getWidth() >= 1040);
    editor.reset();
    plugin->releaseResources();
}
TEST_CASE("Native FX host preserves audio input for recurrent tensor processing", "[tensor-host]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    alfa::tensorhost::Processor p;
    p.prepareToPlay(48000, 32);
    p.mode.store(1);
    juce::String error;
    REQUIRE(p.loadJSON(document(), error));
    juce::AudioBuffer<float> audio(2, 4);
    audio.clear();
    audio.setSample(0, 0, 0.4f);
    juce::MidiBuffer midi;
    p.processBlock(audio, midi);
    CHECK(audio.getSample(0, 0) == 0);
    CHECK(audio.getSample(0, 1) == 0.4f);
    CHECK(audio.getSample(1, 1) == 0.4f);
    CHECK(p.failures.load() == 0);
}
TEST_CASE("Hybrid host routes MIDI velocity and audio through same compiled state", "[tensor-host]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    alfa::tensorhost::Processor p;
    p.prepareToPlay(48000, 32);
    p.mode.store(2);
    juce::String error;
    REQUIRE(p.loadJSON(document(true), error));
    juce::AudioBuffer<float> audio(2, 4);
    audio.clear();
    audio.setSample(0, 0, 0.4f);
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(1, 60, 1.0f), 0);
    p.processBlock(audio, midi);
    CHECK(audio.getSample(0, 1) == 0.4f);
    CHECK(audio.getSample(0, 2) == 0.1f);
    CHECK(p.failures.load() == 0);
}
TEST_CASE("Tensor host configuration and xControls survive project persistence", "[tensor-host]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    alfa::tensorhost::Processor original;
    juce::String error;
    REQUIRE(original.loadJSON(document(), error));
    original.mode.store(1);
    original.xControls[2].store(0.5f);
    juce::MemoryBlock bytes;
    original.getStateInformation(bytes);
    alfa::tensorhost::Processor restored;
    restored.setStateInformation(bytes.getData(), static_cast<int>(bytes.getSize()));
    CHECK(restored.document() == original.document());
    CHECK(restored.mode.load() == 1);
    CHECK(restored.xControls[2].load() == Catch::Approx(0.5f).margin(0.000001f));
    REQUIRE_FALSE(restored.loadJSON("{}", error));
    CHECK(restored.document() == original.document());
}
TEST_CASE("Embedded instrument executes MIDI projection and note-off without audio bleed", "[tensor-host]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    alfa::tensorhost::Processor p;
    juce::String error;
    REQUIRE(p.loadJSON(alfa::tensorhost::Processor::referenceDocument(0), error));
    p.prepareToPlay(48000, 64);
    p.mode.store(0);
    juce::AudioBuffer<float> audio(2, 64);
    juce::MidiBuffer midi;
    for (int channel = 0; channel < 2; ++channel)
        for (int sample = 0; sample < 64; ++sample)
            audio.setSample(channel, sample, 0.7f);
    p.processBlock(audio, midi);
    CHECK(audio.getMagnitude(0, 64) == 0);
    audio.clear();
    midi.addEvent(juce::MidiMessage::noteOn(3, 60, 0.5f), 0);
    p.processBlock(audio, midi);
    CHECK(audio.getMagnitude(0, 64) > 0.01f);
    audio.clear();
    midi.clear();
    midi.addEvent(juce::MidiMessage::noteOff(3, 60), 0);
    p.processBlock(audio, midi);
    CHECK(audio.getMagnitude(0, 64) == 0);
    CHECK(p.failures.load() == 0);
}
TEST_CASE("Host MPE poly pressure is retained as an independently routable input", "[tensor-host]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    alfa::tensorhost::Processor p;
    p.prepareToPlay(48000, 32);
    p.mode.store(2);
    auto data = nlohmann::json::parse(document(true).toStdString());
    data["projections"][0]["source_element"] = 4;
    juce::String error;
    REQUIRE(p.loadJSON(data.dump(), error));
    juce::AudioBuffer<float> audio(2, 4);
    audio.clear();
    audio.setSample(0, 0, 0.4f);
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(3, 60, 0.2f), 0);
    midi.addEvent(juce::MidiMessage::aftertouchChange(3, 60, 127), 0);
    p.processBlock(audio, midi);
    CHECK(audio.getSample(0, 2) == 0.1f);
    CHECK(p.failures.load() == 0);
}
TEST_CASE("Host imports a genuine runtime GRU compiler and refuses incompatible replacement", "[tensor-host]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    alfa::tensorhost::Processor p;
    p.prepareToPlay(48000, 64);
    auto data = nlohmann::json::parse(alfa::tensorhost::Processor::referenceDocument(0).toStdString());
    const auto model = juce::File(juce::String(ALFA_SOURCE_DIR) + "/training/aom/testdata/runtime_gru.json")
                           .loadFileAsString();
    data["runtimeModel"] = nlohmann::json::parse(model.toStdString());
    data["performance"] =
        nlohmann::json{{"midi_layout", "voice_indexed"}, {"voice_capacity", 1}, {"release_seconds", 0.0}};
    juce::String error;
    REQUIRE(p.loadJSON(data.dump(), error));
    CHECK(p.inspection().contains("GRU compiler model loaded"));
    juce::AudioBuffer<float> audio(2, 64);
    audio.clear();
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.7f), 0);
    p.processBlock(audio, midi);
    CHECK(p.failures.load() == 0);
    const auto retained = p.document();
    data["runtimeModel"]["budget"]["available_ram_bytes"] = 1;
    REQUIRE_FALSE(p.loadJSON(data.dump(), error));
    CHECK(p.document() == retained);
}
TEST_CASE("Host borrows immutable latent input and rejects malformed replacements", "[tensor-host]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    alfa::tensorhost::Processor p;
    p.prepareToPlay(48000, 32);
    p.mode.store(2);
    auto data = nlohmann::json::parse(document(true).toStdString());
    data["projections"][0]["source"] = "latent_sound";
    data["projections"][0]["source_element"] = 0;
    data["latentSound"] = {{"axes", {"latent"}}, {"shape", {1}}, {"values", {1.0f}}, {"presence", {1}}};
    data["budget"]["stored_tensor_elements"] = 4;
    juce::String error;
    REQUIRE(p.loadJSON(data.dump(), error));
    juce::AudioBuffer<float> audio(2, 4);
    audio.clear();
    audio.setSample(0, 0, 0.4f);
    juce::MidiBuffer midi;
    p.processBlock(audio, midi);
    CHECK(audio.getSample(0, 2) == 0.1f);
    CHECK(p.failures.load() == 0);
    const auto retained = p.document();
    auto invalid = data;
    invalid["latentSound"]["shape"] = {2};
    REQUIRE_FALSE(p.loadJSON(invalid.dump(), error));
    CHECK(p.document() == retained);
    invalid = data;
    invalid["latentSound"]["presence"] = {0};
    REQUIRE_FALSE(p.loadJSON(invalid.dump(), error));
    CHECK(p.document() == retained);
    invalid = data;
    invalid["latentSound"]["presence"] = {2};
    REQUIRE_FALSE(p.loadJSON(invalid.dump(), error));
    CHECK(p.document() == retained);
    juce::MemoryBlock state;
    p.getStateInformation(state);
    alfa::tensorhost::Processor restored;
    restored.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    CHECK(restored.document() == retained);
}
