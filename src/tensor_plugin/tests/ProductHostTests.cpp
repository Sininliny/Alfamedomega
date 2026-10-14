#include "FactoryPrograms.h"
#include "TensorEditor.h"
#include "TensorProcessor.h"
#include "alfa/ui/Theme.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <nlohmann/json.hpp>
#include <numbers>
#include <stdexcept>

namespace
{
using JSON = nlohmann::json;
using alfa::tensorhost::Processor;

juce::String factory(const char* id)
{
    for (const auto& entry : alfa::tensorhost::factoryPrograms())
        if (entry.id == id)
            return juce::String(entry.document);
    throw std::runtime_error("Required authored fixture is unavailable");
}

JSON sineFixture()
{
    auto document = JSON::parse(Processor::referenceDocument(0).toStdString());
    document["tensors"]["parameters"]["values"] = {400., .2};
    document["tensors"]["routing"] = {{"axes", {"coefficient"}}, {"shape", {0}}, {"values", JSON::array()}};
    document["projections"] = JSON::array();
    document["budget"]["routing_elements"] = 0;
    document["budget"]["projection_terms"] = 0;
    document["budget"]["stored_tensor_elements"] = 3;
    return document;
}

JSON fanoutFixture()
{
    auto document = sineFixture();
    constexpr std::size_t middle = 128;
    document["tensors"]["parameters"]["values"][1] = 0;
    std::vector<double> first(middle * 8, .001), second(2 * middle, 0.);
    for (std::size_t source = 0; source < middle; ++source)
        second[middle + source] = 1.;
    const auto layer = [](std::size_t sources, std::size_t destinations, std::vector<double> weights)
    {
        return JSON{{"values", std::vector<double>(sources, 0.)},
                    {"ranges", std::vector<std::array<double, 2>>(sources, {-1., 1.})},
                    {"matrix",
                     {{"axes", {"destination", "source"}},
                      {"shape", {destinations, sources}},
                      {"values", std::move(weights)}}}};
    };
    document["control_layers"] =
        JSON::array({layer(8, middle, std::move(first)), layer(middle, 2, std::move(second))});
    document["budget"]["stored_tensor_elements"] = 3 + 8 + middle * 8 + middle + 2 * middle;
    return document;
}

void render(Processor& processor, juce::AudioBuffer<float>& audio, juce::MidiBuffer& midi)
{
    processor.processBlock(audio, midi);
    midi.clear();
    for (int channel = 0; channel < audio.getNumChannels(); ++channel)
        for (int sample = 0; sample < audio.getNumSamples(); ++sample)
            REQUIRE(std::isfinite(audio.getSample(channel, sample)));
}

void fill(juce::AudioBuffer<float>& audio, float value)
{
    for (int channel = 0; channel < audio.getNumChannels(); ++channel)
        for (int sample = 0; sample < audio.getNumSamples(); ++sample)
            audio.setSample(channel, sample, value);
}
} // namespace

TEST_CASE("Product host exposes stable X parameter IDs with functional large tensor fanout",
          "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 256);
    juce::String error;
    REQUIRE(processor.loadJSON(juce::String(fanoutFixture().dump()), error));
    juce::AudioBuffer<float> audio(2, 256);
    juce::MidiBuffer midi;
    audio.clear();
    render(processor, audio, midi);
    CHECK(audio.getMagnitude(0, 256) == 0);
    for (int coordinate = 0; coordinate < 8; ++coordinate)
    {
        const auto id = "x" + juce::String(coordinate);
        REQUIRE(processor.parameters().getParameter(id) != nullptr);
        const auto range = processor.parameters().getParameter(id)->getNormalisableRange();
        CHECK(range.start == -1);
        CHECK(range.end == 1);
        for (int previous = 0; previous < 8; ++previous)
            processor.setParameterValue("x" + juce::String(previous), 0);
        processor.setParameterValue(id, 1);
        audio.clear();
        render(processor, audio, midi);
        CHECK(audio.getMagnitude(0, 256) > .1f);
        CHECK(processor.xControls[static_cast<std::size_t>(coordinate)].load() == 1);
        const auto snapshot = JSON::parse(processor.document().toStdString());
        CHECK(snapshot["control_layers"][0]["values"][static_cast<size_t>(coordinate)] == 1);
        CHECK(snapshot["control_layers"][1]["values"].size() == 128);
    }
    CHECK(processor.failures.load() == 0);
}

TEST_CASE("Product parameter edits preserve running phase and locks refuse unauthorized replacement",
          "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor edited, reference;
    edited.prepareToPlay(48000, 32);
    reference.prepareToPlay(48000, 32);
    const auto fixture = juce::String(sineFixture().dump());
    juce::String error;
    REQUIRE(edited.loadJSON(fixture, error));
    REQUIRE(reference.loadJSON(fixture, error));
    juce::AudioBuffer<float> actual(2, 32), expected(2, 32);
    juce::MidiBuffer midi;
    actual.clear();
    expected.clear();
    render(edited, actual, midi);
    render(reference, expected, midi);
    REQUIRE(edited.editTensorParameter(1, .3f, error));
    actual.clear();
    expected.clear();
    render(edited, actual, midi);
    render(reference, expected, midi);
    for (int sample = 0; sample < 32; ++sample)
        CHECK(actual.getSample(0, sample) ==
              Catch::Approx(expected.getSample(0, sample) * 1.5f).margin(.000001));
    REQUIRE(edited.setParameterLock(1, true, error));
    const auto locked = edited.document();
    CHECK_FALSE(edited.editTensorParameter(1, .7f, error));
    CHECK(edited.document() == locked);
    auto replacement = sineFixture();
    replacement["tensors"]["parameters"]["values"][1] = .8;
    REQUIRE(edited.loadPresetJSON(juce::String(replacement.dump()), error));
    CHECK(JSON::parse(edited.document().toStdString())["tensors"]["parameters"]["values"][1].get<float>() ==
          Catch::Approx(.3));
    replacement["instances"][0]["clock"]["period_samples"] = 2;
    CHECK_FALSE(edited.loadPresetJSON(juce::String(replacement.dump()), error));
    CHECK(edited.failures.load() == 0);
}

TEST_CASE("Product MIDI plays two actual voices and respects sustain release and panic",
          "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 256);
    juce::String error;
    REQUIRE(processor.loadPresetJSON(factory("harmonic-field"), error));
    juce::AudioBuffer<float> audio(2, 256);
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(1, 69, .8f), 0);
    midi.addEvent(juce::MidiMessage::noteOn(1, 72, .6f), 0);
    std::array<std::complex<double>, 2> tones{};
    const std::array<double, 2> frequencies{440., juce::MidiMessage::getMidiNoteInHertz(72)};
    constexpr int measuredFrames = 8192;
    int measured = 0;
    for (int block = 0; block < 48; ++block)
    {
        audio.clear();
        render(processor, audio, midi);
        if (block < 16)
            continue;
        for (int sample = 0; sample < 256; ++sample, ++measured)
            for (std::size_t tone = 0; tone < tones.size(); ++tone)
                tones[tone] += static_cast<double>(audio.getSample(0, sample)) *
                               std::exp(std::complex<double>(0, -2 * std::numbers::pi * frequencies[tone] *
                                                                    measured / 48000));
    }
    REQUIRE(measured == measuredFrames);
    for (const auto& tone : tones)
        CHECK(std::abs(tone) / measuredFrames > .0001);
    CHECK(processor.activeNotes.load() == 2);
    midi.addEvent(juce::MidiMessage::controllerEvent(1, 64, 127), 0);
    midi.addEvent(juce::MidiMessage::noteOff(1, 69), 1);
    audio.clear();
    render(processor, audio, midi);
    CHECK(processor.activeNotes.load() == 2);
    midi.addEvent(juce::MidiMessage::controllerEvent(1, 64, 0), 0);
    audio.clear();
    render(processor, audio, midi);
    CHECK(processor.activeNotes.load() == 1);
    midi.addEvent(juce::MidiMessage::noteOff(1, 72), 0);
    audio.clear();
    render(processor, audio, midi);
    CHECK(processor.activeNotes.load() == 0);
    CHECK(processor.getTailLengthSeconds() > 0);
    processor.requestPanic();
    audio.clear();
    render(processor, audio, midi);
    CHECK(audio.getMagnitude(0, 256) == 0);
    CHECK(processor.activeNotes.load() == 0);
    CHECK(processor.failures.load() == 0);
}

TEST_CASE("Product FX gain mix and bypass act on native audio without losing host state",
          "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 4096);
    juce::String error;
    REQUIRE(processor.loadPresetJSON(factory("stereo-field"), error));
    juce::AudioBuffer<float> audio(2, 4096);
    juce::MidiBuffer midi;
    processor.setParameterValue("input_gain", -6);
    fill(audio, .4f);
    render(processor, audio, midi);
    CHECK(audio.getSample(0, 4095) ==
          Catch::Approx(.4f * juce::Decibels::decibelsToGain(-6.f)).margin(.00001));
    processor.setParameterValue("mix", 0);
    fill(audio, .4f);
    render(processor, audio, midi);
    CHECK(audio.getSample(0, 4095) == Catch::Approx(.4).margin(.00001));
    processor.setParameterValue("output_gain", -6);
    fill(audio, .4f);
    render(processor, audio, midi);
    CHECK(audio.getSample(0, 4095) ==
          Catch::Approx(.4f * juce::Decibels::decibelsToGain(-6.f)).margin(.00001));
    processor.setParameterValue("bypass", 1);
    fill(audio, .4f);
    render(processor, audio, midi);
    CHECK(audio.getSample(0, 4095) == Catch::Approx(.4).margin(.00001));
    processor.setParameterValue("x7", .375f);
    processor.uiSettings().setProperty("scale", 1.4, nullptr);
    processor.uiSettings().setProperty("presetFolder", "fixture-user-library", nullptr);
    juce::MemoryBlock state;
    processor.getStateInformation(state);
    Processor restored;
    restored.prepareToPlay(48000, 4096);
    restored.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    for (const auto* id : {"input_gain", "output_gain", "mix", "bypass", "mode", "x7"})
        CHECK(restored.parameters().getRawParameterValue(id)->load() ==
              Catch::Approx(processor.parameters().getRawParameterValue(id)->load()).margin(.00001));
    CHECK(static_cast<double>(restored.uiSettings().getProperty("scale")) == Catch::Approx(1.4));
    CHECK(restored.uiSettings().getProperty("presetFolder").toString() == "fixture-user-library");
    fill(audio, .4f);
    render(restored, audio, midi);
    CHECK(audio.getSample(0, 4095) == Catch::Approx(.4).margin(.00001));
    CHECK(restored.failures.load() == 0);
}

TEST_CASE("Malformed product control layers refuse replacement and retain active sound",
          "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 256);
    const auto good = fanoutFixture();
    juce::String error;
    REQUIRE(processor.loadJSON(juce::String(good.dump()), error));
    processor.setParameterValue("x0", 1);
    juce::AudioBuffer<float> audio(2, 256);
    juce::MidiBuffer midi;
    audio.clear();
    render(processor, audio, midi);
    const auto retained = processor.document();
    const auto revision = processor.programRevision.load();
    std::vector<JSON> malformed;
    auto changed = good;
    changed["control_layers"][0]["matrix"]["shape"] = {127, 8};
    malformed.push_back(changed);
    changed = good;
    changed["control_layers"][0]["matrix"]["axes"] = {"source", "destination"};
    malformed.push_back(changed);
    changed = good;
    changed["control_layers"][0]["ranges"][0] = {1., -1.};
    malformed.push_back(changed);
    changed = good;
    changed["control_layers"][0]["values"][0] = 2.;
    malformed.push_back(changed);
    changed = good;
    changed["budget"]["stored_tensor_elements"] = 3;
    malformed.push_back(changed);
    changed = good;
    changed["control_layers"][0]["labels"] = {"Only one"};
    malformed.push_back(changed);
    changed = good;
    changed["control_layers"][0]["labels"] = std::vector<int>(8, 1);
    malformed.push_back(changed);
    changed = good;
    changed["control_layers"][0]["ranking"] = {{"model", "m"}, {"importance", std::vector<double>(7, 1.)}};
    malformed.push_back(changed);
    changed = good;
    changed["control_layers"] = JSON::object();
    changed["control_layers"]["invalid"] = true;
    malformed.push_back(changed);
    for (const auto& candidate : malformed)
    {
        CHECK_FALSE(processor.loadJSON(juce::String(candidate.dump()), error));
        CHECK(error.isNotEmpty());
        CHECK(processor.document() == retained);
        CHECK(processor.programRevision.load() == revision);
        audio.clear();
        render(processor, audio, midi);
        CHECK(audio.getMagnitude(0, 256) > .1f);
    }
    CHECK(processor.failures.load() == 0);
}

TEST_CASE("Configuration undo and redo retain held notes and continuous oscillator phase",
          "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor edited, reference;
    edited.prepareToPlay(48000, 32);
    reference.prepareToPlay(48000, 32);
    juce::String error;
    const auto fixture = juce::String(sineFixture().dump());
    REQUIRE(edited.loadJSON(fixture, error));
    REQUIRE(reference.loadJSON(fixture, error));
    juce::AudioBuffer<float> actual(2, 32), expected(2, 32);
    const auto compare = [&](float ratio, bool noteOn)
    {
        juce::MidiBuffer first, second;
        if (noteOn)
        {
            first.addEvent(juce::MidiMessage::noteOn(1, 69, .8f), 0);
            second.addEvent(juce::MidiMessage::noteOn(1, 69, .8f), 0);
        }
        actual.clear();
        expected.clear();
        render(edited, actual, first);
        render(reference, expected, second);
        for (int sample = 0; sample < 32; ++sample)
            CHECK(actual.getSample(0, sample) ==
                  Catch::Approx(expected.getSample(0, sample) * ratio).margin(.000001));
        CHECK(edited.activeNotes.load() == 1);
    };
    compare(1, true);
    REQUIRE(edited.editTensorParameter(1, .3f, error));
    compare(1.5f, false);
    REQUIRE(edited.undoManager().canUndo());
    REQUIRE(edited.undoManager().undo());
    compare(1, false);
    REQUIRE(edited.undoManager().canRedo());
    REQUIRE(edited.undoManager().redo());
    compare(1.5f, false);
    const auto retained = edited.document();
    const auto revision = edited.programRevision.load();
    CHECK_FALSE(edited.editTensorParameter(9999, .7f, error));
    CHECK_FALSE(edited.editConfigurationJSON("{invalid", error));
    CHECK(edited.document() == retained);
    CHECK(edited.programRevision.load() == revision);
    REQUIRE(edited.undoManager().undo());
    compare(1, false);
    CHECK(edited.failures.load() == 0);
}

TEST_CASE("Custom interface theme survives editor construction reopen and host restore",
          "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    auto custom = alfa::ui::Theme::builtIn("graphite");
    custom.name = "User fixture theme";
    custom.accent = juce::Colour(0xff12abcd);
    const auto serialized = custom.toJson();
    processor.uiSettings().setProperty("theme", "graphite", nullptr);
    processor.uiSettings().setProperty("custom_theme", serialized, nullptr);
    const auto reopen = [&](Processor& owner)
    {
        std::unique_ptr<juce::AudioProcessorEditor> editor(owner.createEditor());
        REQUIRE(editor != nullptr);
        juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
        CHECK(owner.uiSettings().getProperty("custom_theme").toString() == serialized);
        CHECK(alfa::ui::Theme::fromJson(owner.uiSettings().getProperty("custom_theme").toString()).accent ==
              custom.accent);
    };
    reopen(processor);
    reopen(processor);
    juce::MemoryBlock state;
    processor.getStateInformation(state);
    Processor restored;
    restored.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    reopen(restored);
}

TEST_CASE("Mix, input gain and bypass changes ramp instead of stepping", "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 4096);
    juce::String error;
    REQUIRE(processor.loadPresetJSON(factory("stereo-field"), error));
    juce::AudioBuffer<float> audio(2, 4096);
    juce::MidiBuffer midi;
    // A nearly silent wet path against a constant dry input: every switch is a 0.4 jump
    // unless the processor ramps it.
    processor.setParameterValue("input_gain", -60);
    const auto largestStep = [&](float& previous)
    {
        fill(audio, .4f);
        render(processor, audio, midi);
        float step = 0;
        for (int i = 0; i < audio.getNumSamples(); ++i)
        {
            step = std::max(step, std::abs(audio.getSample(0, i) - previous));
            previous = audio.getSample(0, i);
        }
        return step;
    };
    float previous = 0;
    largestStep(previous);
    REQUIRE(previous == Catch::Approx(.0004).margin(.00001));
    processor.setParameterValue("mix", 0);
    CHECK(largestStep(previous) < .01f);
    CHECK(previous == Catch::Approx(.4).margin(.00001));
    processor.setParameterValue("mix", 1);
    largestStep(previous);
    CHECK(previous == Catch::Approx(.0004).margin(.00001));
    processor.setParameterValue("bypass", 1);
    CHECK(largestStep(previous) < .01f);
    CHECK(previous == .4f);
    processor.setParameterValue("bypass", 0);
    CHECK(largestStep(previous) < .01f);
    processor.setParameterValue("input_gain", 0);
    CHECK(largestStep(previous) < .01f);
    CHECK(previous == Catch::Approx(.4).margin(.00001));
    CHECK(processor.failures.load() == 0);
}

TEST_CASE("Host bypass parameter is the functional APVTS bypass object", "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    REQUIRE(processor.getBypassParameter() != nullptr);
    CHECK(processor.getBypassParameter() == processor.parameters().getParameter("bypass"));
    processor.getBypassParameter()->setValueNotifyingHost(1);
    CHECK(processor.parameters().getRawParameterValue("bypass")->load() == 1);
    processor.setParameterValue("bypass", 0);
    CHECK(processor.getBypassParameter()->getValue() == 0);
}

TEST_CASE("Indexed X host controls never implicitly populate the preference tensor", "[tensor-host][product]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    auto fixture = sineFixture();
    fixture["tensors"]["routing"] = {
        {"axes", {"coefficient"}}, {"shape", {5}}, {"values", {1., 1., 1., 1., 1.}}};
    for (int source = 0; source < 5; ++source)
        fixture["projections"].push_back({{"source", "user_preference"},
                                          {"source_element", source},
                                          {"parameter_element", 1},
                                          {"weights", "routing"},
                                          {"weight_element", source},
                                          {"missing_source", "reject"}});
    fixture["budget"]["routing_elements"] = 5;
    fixture["budget"]["projection_terms"] = 5;
    fixture["budget"]["stored_tensor_elements"] = 8;
    Processor changed, reference;
    changed.prepareToPlay(48000, 128);
    reference.prepareToPlay(48000, 128);
    juce::String error;
    REQUIRE(changed.loadJSON(juce::String(fixture.dump()), error));
    REQUIRE(reference.loadJSON(juce::String(fixture.dump()), error));
    juce::AudioBuffer<float> actual(2, 128), expected(2, 128);
    juce::MidiBuffer midi;
    for (int coordinate = 0; coordinate < 8; ++coordinate)
    {
        changed.setParameterValue("x" + juce::String(coordinate), 1);
        actual.clear();
        expected.clear();
        render(changed, actual, midi);
        render(reference, expected, midi);
        for (int sample = 0; sample < 128; ++sample)
            CHECK(actual.getSample(0, sample) ==
                  Catch::Approx(expected.getSample(0, sample)).margin(.000001));
    }
    CHECK(changed.failures.load() == 0);
}

TEST_CASE("Precision tier switches execute real notes and restore the same tensor coordinates",
          "[tensor-host][product][precision]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 256);
    juce::String error;
    REQUIRE(processor.loadPresetJSON(factory("harmonic-field"), error));
    const auto original = JSON::parse(processor.document().toStdString());
    REQUIRE(processor.setTier("Pro", error));
    juce::AudioBuffer<float> audio(2, 256);
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(1, 69, .8f), 0);
    float peak = 0;
    for (int block = 0; block < 12; ++block)
    {
        audio.clear();
        render(processor, audio, midi);
        peak = std::max(peak, audio.getMagnitude(0, 256));
    }
    CHECK(peak > .0001f);
    CHECK(processor.scalarBytes() == 8);
    CHECK(processor.inspection().contains("float64"));
    // The sound's own capacity budget is retained and reported next to what the sound uses.
    CHECK(processor.inspection().contains("Used / budget: primitives"));
    CHECK(processor.inspection().contains("Budget source: the loaded sound"));
    const auto pro = JSON::parse(processor.document().toStdString());
    CHECK(pro["budget"]["state_elements"] == original["budget"]["state_elements"]);
    CHECK(pro["budget"]["active_primitives"] == original["budget"]["active_primitives"]);
    CHECK(pro["tier"] == "Pro");
    CHECK(pro["scalar_type"] == "float64");
    for (const auto* name : {"activation", "parameters", "routing", "modulation"})
        CHECK(pro["tensors"][name]["shape"] == original["tensors"][name]["shape"]);
    juce::MemoryBlock state;
    processor.getStateInformation(state);
    Processor restored;
    restored.prepareToPlay(48000, 256);
    restored.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    audio.clear();
    render(restored, audio, midi);
    CHECK(restored.scalarBytes() == 8);
    CHECK(JSON::parse(restored.document().toStdString())["tier"] == "Pro");
    REQUIRE(processor.setTier("Lite", error));
    audio.clear();
    render(processor, audio, midi);
    CHECK(processor.scalarBytes() == 4);
    const auto lite = JSON::parse(processor.document().toStdString());
    for (const auto* name : {"activation", "parameters", "routing", "modulation"})
        CHECK(lite["tensors"][name]["shape"] == pro["tensors"][name]["shape"]);
    const auto retained = processor.document();
    CHECK_FALSE(processor.setTier("Unspecified", error));
    CHECK(processor.document() == retained);
    CHECK(processor.scalarBytes() == 4);
    CHECK(processor.failures.load() == 0);
}

TEST_CASE("Double raw parameter edits retain precision and inadequate boundary memory refuses replacement",
          "[tensor-host][product][precision]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 128);
    juce::String error;
    REQUIRE(processor.loadJSON(juce::String(sineFixture().dump()), error));
    REQUIRE(processor.setTier("Pro", error));
    constexpr double frequency = 400.123456789011;
    REQUIRE(processor.editTensorParameter(0, frequency, error));
    auto document = JSON::parse(processor.document().toStdString());
    CHECK(document["tensors"]["parameters"]["values"][0].get<double>() == frequency);
    CHECK(document["tensors"]["parameters"]["values"][0].get<double>() !=
          static_cast<double>(static_cast<float>(frequency)));
    juce::AudioBuffer<float> audio(2, 128);
    juce::MidiBuffer midi;
    audio.clear();
    render(processor, audio, midi);
    CHECK(processor.scalarBytes() == 8);
    CHECK(audio.getMagnitude(0, 128) > .1f);
    const auto retained = processor.document();
    document["budget"]["boundary_conversion_bytes"] = 0;
    CHECK_FALSE(processor.loadJSON(juce::String(document.dump()), error));
    CHECK(error.contains("boundary_conversion_bytes"));
    CHECK(processor.document() == retained);
    audio.clear();
    render(processor, audio, midi);
    CHECK(processor.scalarBytes() == 8);
    CHECK(audio.getMagnitude(0, 128) > .1f);
    CHECK(processor.failures.load() == 0);
}

TEST_CASE("Editor has no tier selector and precision follows the loaded sound",
          "[tensor-host][product][precision]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::String error;
    Processor source;
    source.prepareToPlay(48000, 128);
    REQUIRE(source.loadJSON(juce::String(sineFixture().dump()), error));
    REQUIRE(source.setTier("Pro", error));
    const auto float64Sound = source.document();

    Processor processor;
    processor.prepareToPlay(48000, 128);
    std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
    for (int i = 0; i < editor->getNumChildComponents(); ++i)
        if (auto* combo = dynamic_cast<juce::ComboBox*>(editor->getChildComponent(i)))
            CHECK_FALSE(combo->getItemText(0) == "Lite");
    CHECK(processor.scalarBytes() == 4);
    REQUIRE(processor.loadJSON(float64Sound, error));
    juce::AudioBuffer<float> audio(2, 128);
    juce::MidiBuffer midi;
    audio.clear();
    render(processor, audio, midi);
    CHECK(processor.scalarBytes() == 8);
    CHECK(processor.inspection().contains("float64 (from the loaded sound)"));
    CHECK(processor.failures.load() == 0);
}

TEST_CASE("Editor instance and tensor inspection leave executable sound unchanged",
          "[tensor-host][product][ui-loop]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 128);
    std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
    auto* console = dynamic_cast<alfa::tensorhost::Editor*>(editor.get());
    REQUIRE(console != nullptr);
    const auto initial = processor.programRevision.load();
    const auto document = processor.document();
    juce::MessageManager::getInstance()->runDispatchLoopUntil(200);
    REQUIRE(processor.programRevision.load() == initial);
    console->snapshotView(1);
    juce::ListBox* instances = nullptr;
    for (auto* child : console->getChildren())
        if (auto* list = dynamic_cast<juce::ListBox*>(child))
            instances = list;
    REQUIRE(instances != nullptr);
    instances->selectRow(1);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(100);
    CHECK(processor.programRevision.load() == initial);
    CHECK(processor.document() == document);
    console->snapshotView(2);
    juce::ComboBox* tensors = nullptr;
    for (auto* child : console->getChildren())
        if (auto* combo = dynamic_cast<juce::ComboBox*>(child);
            combo && combo->getItemText(0) == "Routing tensor")
            tensors = combo;
    REQUIRE(tensors != nullptr);
    tensors->setSelectedId(3, juce::sendNotificationAsync);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(100);
    CHECK(processor.programRevision.load() == initial);
    CHECK(processor.document() == document);
    for (auto* child : console->getChildren())
        if (auto* combo = dynamic_cast<juce::ComboBox*>(child))
            CHECK_FALSE(combo->getItemText(0).contains("Harmonic Field"));
}

TEST_CASE("Play view names controls, follows a learned ranking and renames into the sound",
          "[tensor-host][product][ui-loop]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 128);
    auto document = fanoutFixture();
    document["control_layers"][0]["labels"] = {"Bite", "", "", "", "", "", "", ""};
    document["control_layers"][0]["ranking"] = {
        {"model", std::string(64, 'a')}, {"name", "importance"}, {"importance", {.5, 0, 0, 0, 0, .9, 0, 0}}};
    juce::String error;
    REQUIRE(processor.loadJSON(juce::String(document.dump()), error));
    std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
    auto* console = dynamic_cast<alfa::tensorhost::Editor*>(editor.get());
    REQUIRE(console != nullptr);
    console->snapshotView(0);
    const auto titles = [&]
    {
        std::vector<juce::Label*> found;
        std::function<void(juce::Component&)> visit = [&](juce::Component& parent)
        {
            for (auto* child : parent.getChildren())
            {
                if (!child->isVisible())
                    continue;
                if (auto* label = dynamic_cast<juce::Label*>(child);
                    label && label->isEditableOnDoubleClick() && parent.isVisible())
                    found.push_back(label);
                visit(*child);
            }
        };
        visit(*console);
        std::sort(found.begin(), found.end(),
                  [](auto* a, auto* b) { return a->getScreenX() < b->getScreenX(); });
        return found;
    };
    auto labels = titles();
    REQUIRE(labels.size() == 8);
    CHECK(labels[0]->getText() == "1  X[0,5]");
    CHECK(labels[1]->getText() == "2  Bite");
    juce::TextButton* order = nullptr;
    for (auto* child : console->getChildren())
        if (auto* button = dynamic_cast<juce::TextButton*>(child);
            button && button->getButtonText() == "Learned order")
            order = button;
    REQUIRE(order != nullptr);
    CHECK(order->getToggleState());
    order->triggerClick();
    juce::MessageManager::getInstance()->runDispatchLoopUntil(100);
    labels = titles();
    REQUIRE(labels.size() == 8);
    CHECK(labels[0]->getText() == "Bite");
    CHECK(labels[1]->getText() == "X[0,1]");
    const auto revision = processor.programRevision.load();
    labels[1]->setText("Air", juce::sendNotificationSync);
    juce::MessageManager::getInstance()->runDispatchLoopUntil(300);
    CHECK(processor.programRevision.load() != revision);
    const auto renamed = JSON::parse(processor.document().toStdString());
    CHECK(renamed["control_layers"][0]["labels"][1] == "Air");
    CHECK(renamed["control_layers"][0]["values"] == document["control_layers"][0]["values"]);
    REQUIRE(processor.undoManager().undo());
    CHECK(JSON::parse(processor.document().toStdString())["control_layers"][0]["labels"][1] == "");
}

TEST_CASE("Tensor view shows the matrix and capacity first and opens expert tools on request",
          "[tensor-host][product][ui-loop]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 128);
    std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
    auto* console = dynamic_cast<alfa::tensorhost::Editor*>(editor.get());
    REQUIRE(console != nullptr);
    console->snapshotView(2);
    juce::TextEditor* json = nullptr;
    alfa::tensorhost::CapacityView* capacity = nullptr;
    juce::Button* expert = nullptr;
    for (auto* child : console->getChildren())
    {
        if (auto* text = dynamic_cast<juce::TextEditor*>(child); text && text->isMultiLine() && !json)
            json = text;
        if (auto* view = dynamic_cast<alfa::tensorhost::CapacityView*>(child))
            capacity = view;
        if (child->getComponentID() == "view.expert")
            expert = dynamic_cast<juce::Button*>(child);
    }
    REQUIRE(json != nullptr);
    REQUIRE(capacity != nullptr);
    REQUIRE(expert != nullptr);
    CHECK(capacity->isVisible());
    CHECK_FALSE(json->isVisible());
    const auto revision = processor.programRevision.load();
    expert->triggerClick();
    juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    CHECK(json->isVisible());
    CHECK_FALSE(capacity->isVisible());
    CHECK(processor.programRevision.load() == revision);
    juce::String source;
    const auto usage = alfa::tensorhost::CapacityView::parse(processor.inspection(), source);
    REQUIRE(usage.size() == 5);
    CHECK(usage[0].name == "Primitives");
    CHECK(usage[0].budget >= usage[0].used);
    CHECK(source.isNotEmpty());
}
