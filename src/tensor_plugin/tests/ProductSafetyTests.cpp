#include "CompositionPanel.h"
#include "PrimitiveFragmentPanel.h"
#include "TensorProcessor.h"
#include "TierConversion.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace
{
using JSON = nlohmann::json;
using alfa::tensorhost::Processor;
JSON sine()
{
    auto sound = JSON::parse(Processor::referenceDocument(0).toStdString());
    sound["projections"] = JSON::array();
    sound["tensors"]["routing"] = {{"axes", {"coefficient"}}, {"shape", {0}}, {"values", JSON::array()}};
    sound["tensors"]["parameters"]["values"] = {100., .2};
    sound["budget"]["routing_elements"] = 0;
    sound["budget"]["projection_terms"] = 0;
    sound["budget"]["stored_tensor_elements"] = 3;
    return sound;
}
void pump()
{
    juce::MessageManager::getInstance()->runDispatchLoopUntil(30);
}
juce::Button* button(juce::Component& parent, const juce::String& text)
{
    if (auto* candidate = dynamic_cast<juce::Button*>(&parent);
        candidate && candidate->getButtonText() == text)
        return candidate;
    for (auto* child : parent.getChildren())
        if (auto* found = button(*child, text))
            return found;
    return nullptr;
}
void click(juce::Component& parent, const juce::String& text)
{
    auto* found = button(parent, text);
    REQUIRE(found != nullptr);
    INFO(text.toStdString());
    REQUIRE(found->isVisible());
    found->triggerClick();
    pump();
}
template <typename Content>
juce::DialogWindow* dialog()
{
    for (int index = 0; index < juce::Component::getNumCurrentlyModalComponents(); ++index)
        if (auto* window =
                dynamic_cast<juce::DialogWindow*>(juce::Component::getCurrentlyModalComponent(index));
            window && dynamic_cast<Content*>(window->getContentComponent()))
            return window;
    return nullptr;
}
} // namespace

TEST_CASE("Global FX mix and bypass align dry impulse to declared algorithm latency",
          "[tensor-host][latency]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    for (const float mix : {0.f, .5f, 1.f})
        for (const int bypass : {0, 1, 2})
        {
            Processor p;
            p.prepareToPlay(48000, 4);
            auto sound = sine();
            sound["instances"][0]["primitive"] = "time.ring_delay.v1";
            sound["instances"][0]["state_size"] = 7;
            sound["instances"][0]["initial_state"] = std::vector<double>(7);
            sound["tensors"]["parameters"]["values"] = {5., 0.};
            sound["budget"]["state_elements"] = 7;
            sound["budget"]["sparse_state_writes"] = 3;
            sound["budget"]["host_alignment_bytes"] = 40;
            sound["host_latency_samples"] = 5;
            juce::String error;
            REQUIRE(p.loadJSON(sound.dump(), error));
            p.setParameterValue("mode", 1);
            p.setParameterValue("mix", mix);
            p.setParameterValue("bypass", bypass == 1 ? 1.f : 0.f);
            juce::MidiBuffer midi;
            for (int block = 0; block < 4; ++block)
            {
                juce::AudioBuffer<float> audio(2, 4);
                audio.clear();
                if (block == 0)
                {
                    audio.setSample(0, 0, 1);
                    audio.setSample(1, 0, 1);
                }
                if (bypass == 2)
                    p.processBlockBypassed(audio, midi);
                else
                    p.processBlock(audio, midi);
                for (int sample = 0; sample < 4; ++sample)
                    CHECK(audio.getSample(0, sample) ==
                          Catch::Approx(block * 4 + sample == 5 ? 1.f : 0.f).margin(1e-6));
            }
        }
}

TEST_CASE("Tier switching enforces authored capacity preserves locks and measures explicit candidates",
          "[tensor-host][tier-conversion]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor p;
    p.prepareToPlay(48000, 32);
    auto source = sine();
    auto capacity = source["budget"];
    capacity["boundary_conversion_bytes"] = 20000;
    for (const auto* tier : {"Lite", "Standard", "Pro", "Studio"})
        source["tier_profiles"][tier] = {{"budget", capacity}};
    source["tier_profiles"]["Lite"]["budget"]["active_primitives"] = 0;
    juce::String error;
    REQUIRE(p.loadJSON(source.dump(), error));
    const auto retained = p.document();
    CHECK_FALSE(p.setTier("Lite", error));
    CHECK(p.document() == retained);
    REQUIRE(p.setTier("Pro", error));
    juce::AudioBuffer<float> audio(2, 32);
    audio.clear();
    juce::MidiBuffer midi;
    p.processBlock(audio, midi);
    CHECK(p.scalarBytes() == 8);
    CHECK(JSON::parse(p.document().toStdString())["tier_conversion"]["candidate"] == 0);
    auto reference = sine();
    reference["parameter_locks"] = {1, 0};
    auto limited = capacity;
    limited["state_elements"] = 0;
    auto candidate = reference;
    candidate["tensors"]["parameters"]["values"][0] = .1;
    reference["tier_profiles"] = {{"Lite", {{"budget", limited}}}};
    reference["tier_candidates"] = {{"Lite", JSON::array({candidate})}};
    std::string result, diagnostic;
    CHECK_FALSE(alfa::tensorhost::convertTier(reference.dump(), "Lite", 48000, result, diagnostic));
    CHECK(diagnostic.find("locked") != std::string::npos);
    auto measured = sine();
    const JSON validation{
        {"evaluation_bytes", 1000000},
        {"evaluation_ticks", 20000},
        {"maximum_errors",
         {{"relative_error", .0001}, {"pitch_error_cents", .1}, {"timing_error_samples", 0}}},
        {"cases", JSON::array({{{"frames", 4096}, {"fundamental_hz", 100.}, {"timing_search_samples", 8}}})}};
    measured["tier_profiles"] = {{"Standard", {{"budget", capacity}, {"validation", validation}}}};
    auto bad = measured;
    bad["tensors"]["parameters"]["values"][0] = 110.;
    bad.erase("tier_profiles");
    measured["tier_candidates"] = {{"Standard", JSON::array({bad})}};
    REQUIRE(alfa::tensorhost::convertTier(measured.dump(), "Standard", 48000, result, diagnostic));
    const auto accepted = JSON::parse(result);
    CHECK(accepted["tier_conversion"]["candidate"] == 0);
    REQUIRE(accepted["tier_conversion"]["rejected"].size() == 1);
    CHECK(accepted["tier_conversion"]["metrics"][0]["relative_error"].get<double>() < 1e-6);
}

TEST_CASE("Tier optimizer merges equivalent recurrences under a smaller active capacity",
          "[tensor-host][tier-conversion]")
{
    auto source = sine();
    source["instances"].push_back(source["instances"][0]);
    source["instances"][1]["activation"] = 1;
    source["instances"][1]["parameters"]["offset"] = 2;
    source["tensors"]["activation"]["shape"] = {2};
    source["tensors"]["activation"]["values"] = {1., 1.};
    source["tensors"]["parameters"]["shape"] = {4};
    source["tensors"]["parameters"]["values"] = {100., .2, 100., .2};
    source["parameter_locks"] = {0, 0, 0, 0};
    source["outputMix"] = {{0, 0, 1.}, {0, 1, 1.}, {1, 0, 1.}, {1, 1, 1.}};
    source["budget"]["active_primitives"] = 2;
    source["budget"]["state_elements"] = 2;
    source["budget"]["output_elements"] = 2;
    source["budget"]["parameter_elements"] = 4;
    source["budget"]["activation_elements"] = 2;
    source["budget"]["stored_tensor_elements"] = 6;
    auto budget = source["budget"];
    budget["active_primitives"] = 1;
    source["tier_profiles"]["Lite"] = {
        {"budget", budget},
        {"strategies", {"merge_equivalent_instances"}},
        {"validation",
         {{"evaluation_bytes", 200000},
          {"evaluation_ticks", 10000},
          {"maximum_errors",
           {{"relative_error", .0001}, {"pitch_error_cents", .1}, {"timing_error_samples", 0}}},
          {"cases",
           JSON::array({{{"frames", 4096}, {"fundamental_hz", 100.}, {"timing_search_samples", 8}}})}}}};
    std::string result, error;
    REQUIRE(alfa::tensorhost::convertTier(source.dump(), "Lite", 48000, result, error));
    const auto converted = JSON::parse(result);
    CHECK(converted["tensors"]["activation"]["values"] == JSON::array({1., 0.}));
    CHECK(converted["tier_conversion"]["candidate"] == 1);
    CHECK(converted["tier_conversion"]["metrics"][0]["relative_error"].get<double>() < 1e-6);
    source["parameter_locks"][0] = 1;
    CHECK_FALSE(alfa::tensorhost::convertTier(source.dump(), "Lite", 48000, result, error));
}

TEST_CASE("Malformed and unsupported host sessions cannot partially replace a running sound",
          "[tensor-host][product-safety]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 32);
    juce::String error;
    REQUIRE(processor.loadJSON(sine().dump(), error));
    juce::MemoryBlock saved;
    processor.getStateInformation(saved);
    const auto good = JSON::parse(std::string(static_cast<const char*>(saved.getData()), saved.getSize()));
    const auto retained = processor.document();
    const auto revision = processor.programRevision.load();
    auto alternative = sine();
    alternative["tensors"]["parameters"]["values"] = {800., .8};
    for (int failure = 0; failure < 6; ++failure)
    {
        auto corrupt = good;
        corrupt["configuration"] = alternative.dump();
        if (failure == 0)
            corrupt["state_schema"] = "aom.host/999";
        if (failure == 1)
            corrupt["mode"] = "invalid";
        if (failure == 2)
            corrupt["x_controls"] = JSON::object();
        if (failure == 3)
            corrupt["parameters"] = "<WrongParameterTree/>";
        if (failure == 4)
        {
            auto xml = juce::parseXML(juce::String(good.at("parameters").get<std::string>()));
            REQUIRE(xml != nullptr);
            auto* mix = xml->getChildByAttribute("id", "mix");
            REQUIRE(mix != nullptr);
            mix->setAttribute("value", 42.);
            corrupt["parameters"] = xml->toString().toStdString();
        }
        if (failure == 5)
            corrupt["ui_settings"] = "<WrongUITree/>";
        const auto serialized = corrupt.dump();
        processor.setStateInformation(serialized.data(), static_cast<int>(serialized.size()));
        CHECK(processor.document() == retained);
        CHECK(processor.programRevision.load() == revision);
    }
}

TEST_CASE("Invalid host audio is silenced with finite output meters and valid dry audio recovers",
          "[tensor-host][product-safety]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 8);
    processor.setParameterValue("mode", 1);
    processor.setParameterValue("mix", 0);
    processor.setParameterValue("bypass", 1);
    juce::AudioBuffer<float> audio(2, 8);
    juce::MidiBuffer midi;
    audio.clear();
    audio.setSample(0, 0, std::numeric_limits<float>::infinity());
    audio.setSample(1, 0, std::numeric_limits<float>::quiet_NaN());
    processor.processBlock(audio, midi);
    for (int channel = 0; channel < 2; ++channel)
    {
        for (int sample = 0; sample < 8; ++sample)
            CHECK(std::isfinite(audio.getSample(channel, sample)));
        CHECK(audio.getSample(channel, 0) == 0);
        CHECK(std::isfinite(processor.inputPeak[static_cast<std::size_t>(channel)].load()));
        CHECK(std::isfinite(processor.outputPeak[static_cast<std::size_t>(channel)].load()));
    }
    for (int channel = 0; channel < 2; ++channel)
        for (int sample = 0; sample < 8; ++sample)
            audio.setSample(channel, sample, .25f);
    processor.processBlock(audio, midi);
    CHECK(audio.getSample(0, 7) == .25f);
    CHECK(audio.getSample(1, 7) == .25f);
}

TEST_CASE("Host accepts declared mono stereo and disabled input layouts and rejects mismatched buses",
          "[tensor-host][product-safety]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    auto layout = processor.getBusesLayout();
    CHECK(processor.isBusesLayoutSupported(layout));
    layout.inputBuses.set(0, juce::AudioChannelSet::mono());
    layout.outputBuses.set(0, juce::AudioChannelSet::mono());
    CHECK(processor.isBusesLayoutSupported(layout));
    layout.inputBuses.set(0, juce::AudioChannelSet::disabled());
    CHECK(processor.isBusesLayoutSupported(layout));
    layout.outputBuses.set(0, juce::AudioChannelSet::create5point1());
    CHECK_FALSE(processor.isBusesLayoutSupported(layout));
    layout.outputBuses.set(0, juce::AudioChannelSet::stereo());
    layout.inputBuses.set(0, juce::AudioChannelSet::mono());
    CHECK_FALSE(processor.isBusesLayoutSupported(layout));
    layout.outputBuses.set(0, juce::AudioChannelSet::disabled());
    CHECK_FALSE(processor.isBusesLayoutSupported(layout));
}

TEST_CASE("Latest prepared publication executes after repeated edits before a callback",
          "[tensor-host][product-safety]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 32);
    juce::String error;
    auto document = sine();
    for (int publication = 0; publication < 40; ++publication)
    {
        document["tensors"]["parameters"]["values"][1] = .01 * (publication + 1);
        REQUIRE(processor.loadJSON(document.dump(), error));
    }
    juce::AudioBuffer<float> audio(2, 32);
    juce::MidiBuffer midi;
    audio.clear();
    processor.processBlock(audio, midi);
    CHECK(audio.getSample(0, 0) == 0);
    CHECK(audio.getSample(0, 1) ==
          Catch::Approx(.4 * std::sin(2 * juce::MathConstants<double>::pi * 100 / 48000)).margin(1e-6));
    CHECK(processor.failures.load() == 0);
    CHECK(JSON::parse(processor.document().toStdString())["tensors"]["parameters"]["values"][1] == .4);
}

TEST_CASE("Compatible parameter edits with a runtime compiler retain held notes and recurrent audio",
          "[tensor-host][product-safety][model]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    auto sound = JSON::parse(Processor::referenceDocument(0).toStdString());
    sound["runtimeModel"] = JSON::parse(juce::File(juce::String(ALFA_SOURCE_DIR))
                                            .getChildFile("training/aom/testdata/runtime_gru.json")
                                            .loadFileAsString()
                                            .toStdString());
    sound["performance"] =
        nlohmann::json{{"midi_layout", "voice_indexed"}, {"voice_capacity", 1}, {"release_seconds", 0.0}};
    Processor edited, reference;
    edited.prepareToPlay(48000, 32);
    reference.prepareToPlay(48000, 32);
    juce::String error;
    REQUIRE(edited.loadJSON(sound.dump(), error));
    REQUIRE(reference.loadJSON(sound.dump(), error));
    juce::AudioBuffer<float> actual(2, 32), expected(2, 32);
    juce::MidiBuffer first, second;
    first.addEvent(juce::MidiMessage::noteOn(1, 69, .7f), 0);
    second.addEvent(juce::MidiMessage::noteOn(1, 69, .7f), 0);
    actual.clear();
    expected.clear();
    edited.processBlock(actual, first);
    reference.processBlock(expected, second);
    REQUIRE(edited.activeNotes.load() == 1);
    REQUIRE(edited.editTensorParameter(1, .1, error));
    first.clear();
    second.clear();
    actual.clear();
    expected.clear();
    edited.processBlock(actual, first);
    reference.processBlock(expected, second);
    CHECK(edited.activeNotes.load() == 1);
    // Any base-parameter change before the next compiler event is deliberate;
    // after due events, identical retained MIDI/context should predict identical audio.
    for (int sample = 16; sample < 32; ++sample)
        CHECK(actual.getSample(0, sample) == Catch::Approx(expected.getSample(0, sample)).margin(1e-6));
    CHECK(edited.failures.load() == 0);
    REQUIRE(edited.setTier("Pro", error));
    actual.clear();
    edited.processBlock(actual, first);
    CHECK(edited.scalarBytes() == 8);
    CHECK(edited.activeNotes.load() == 1);
    CHECK(edited.failures.load() == 0);
    first.addEvent(juce::MidiMessage::noteOff(1, 69), 0);
    edited.processBlock(actual, first);
    CHECK(edited.activeNotes.load() == 0);
}

TEST_CASE("Authored host latency validates transactionally and absence clears the declaration",
          "[tensor-host][product-safety][latency]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 32);
    juce::String error;
    auto sound = sine();
    sound["budget"]["host_alignment_bytes"] = 127 * 8;
    for (const int latency : {0, 127})
    {
        sound["host_latency_samples"] = latency;
        REQUIRE(processor.loadJSON(sound.dump(), error));
        CHECK(processor.getLatencySamples() == latency);
        CHECK(JSON::parse(processor.document().toStdString())["host_latency_samples"] == latency);
    }
    const auto retained = processor.document();
    const auto revision = processor.programRevision.load();
    const auto invalid =
        JSON::array({-1, 1.5, "32", true, nullptr,
                     static_cast<unsigned long long>(std::numeric_limits<int>::max()) + 1ULL});
    for (const auto& value : invalid)
    {
        INFO(value.dump());
        sound["host_latency_samples"] = value;
        sound["tensors"]["parameters"]["values"][1] = .7;
        CHECK_FALSE(processor.loadJSON(sound.dump(), error));
        CHECK(error.isNotEmpty());
        CHECK(processor.getLatencySamples() == 127);
        CHECK(processor.document() == retained);
        CHECK(processor.programRevision.load() == revision);
    }
    sound["host_latency_samples"] = std::numeric_limits<int>::max();
    CHECK_FALSE(processor.loadJSON(sound.dump(), error));
    CHECK(processor.document() == retained);
    sound.erase("host_latency_samples");
    REQUIRE(processor.loadJSON(sound.dump(), error));
    CHECK(processor.getLatencySamples() == 0);
    CHECK_FALSE(JSON::parse(processor.document().toStdString()).contains("host_latency_samples"));
}

TEST_CASE("Closing the actual editor destroys nested composition and primitive dialogs safely",
          "[tensor-host][product-safety][gui]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 32);
    std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
    REQUIRE(editor != nullptr);
    editor->setVisible(true);
    click(*editor, "Tensor");
    click(*editor, "Expert tools");
    click(*editor, "Compose fragments");
    auto* composition = dialog<alfa::tensorhost::CompositionPanel>();
    REQUIRE(composition != nullptr);
    juce::Component::SafePointer<juce::DialogWindow> compositionSafe(composition);
    click(*composition->getContentComponent(), "Create primitive");
    auto* primitive = dialog<alfa::tensorhost::PrimitiveFragmentPanel>();
    REQUIRE(primitive != nullptr);
    juce::Component::SafePointer<juce::DialogWindow> primitiveSafe(primitive);
    editor.reset();
    pump();
    CHECK(compositionSafe == nullptr);
    CHECK(primitiveSafe == nullptr);
    CHECK(dialog<alfa::tensorhost::CompositionPanel>() == nullptr);
    CHECK(dialog<alfa::tensorhost::PrimitiveFragmentPanel>() == nullptr);
    std::unique_ptr<juce::AudioProcessorEditor> reopened(processor.createEditor());
    REQUIRE(reopened != nullptr);
    reopened->setVisible(true);
    click(*reopened, "Compose fragments"); // Tensor view and open expert tools were persisted.
    auto* next = dialog<alfa::tensorhost::CompositionPanel>();
    REQUIRE(next != nullptr);
    juce::Component::SafePointer<juce::DialogWindow> nextSafe(next);
    next->closeButtonPressed();
    pump();
    CHECK(nextSafe == nullptr);
}

TEST_CASE("Tier measurements support right-channel pitched signals and explicit unpitched FX",
          "[tensor-host][tier-conversion]")
{
    auto source = sine();
    source["outputMix"] = {{0, 1, 1.}};
    const JSON validation{
        {"evaluation_bytes", 1000000},
        {"evaluation_ticks", 20000},
        {"maximum_errors",
         {{"relative_error", .0001}, {"pitch_error_cents", .1}, {"timing_error_samples", 0}}},
        {"cases", JSON::array({{{"frames", 4096}, {"fundamental_hz", 100.}, {"timing_search_samples", 8}}})}};
    source["tier_profiles"]["Standard"] = {{"budget", source["budget"]}, {"validation", validation}};
    std::string result, error;
    REQUIRE(alfa::tensorhost::convertTier(source.dump(), "Standard", 48000, result, error));
    CHECK(JSON::parse(result)["tier_conversion"]["metrics"][0]["pitch_measured"] == true);
    source["instances"][0]["primitive"] = "math.feedback_cell";
    source["instances"][0]["parameters"]["count"] = 1;
    source["tensors"]["parameters"] = {{"axes", {"parameter"}}, {"shape", {1}}, {"values", {.1}}};
    source["budget"]["parameter_elements"] = 1;
    source["budget"]["stored_tensor_elements"] = 2;
    JSON samples = JSON::array();
    uint32_t rng = 123;
    for (int i = 0; i < 4096; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        samples.push_back({static_cast<double>(rng % 1000) / 1000. - .5, 0.});
    }
    auto unpitched = validation;
    unpitched["cases"][0].erase("fundamental_hz");
    unpitched["cases"][0]["signal_type"] = "unpitched";
    unpitched["cases"][0]["audio_frames"] = samples;
    unpitched["maximum_errors"].erase("pitch_error_cents");
    source["tier_profiles"]["Standard"] = {{"budget", source["budget"]}, {"validation", unpitched}};
    auto bad = source;
    bad["tensors"]["parameters"]["values"][0] = .9;
    bad.erase("tier_profiles");
    source["tier_candidates"]["Standard"] = JSON::array({bad});
    REQUIRE(alfa::tensorhost::convertTier(source.dump(), "Standard", 48000, result, error));
    const auto converted = JSON::parse(result);
    CHECK(converted["tier_conversion"]["candidate"] == 0);
    REQUIRE(converted["tier_conversion"]["rejected"].size() == 1);
    CHECK(converted["tier_conversion"]["metrics"][0]["pitch_measured"] == false);
    CHECK(converted["tier_conversion"]["metrics"][0]["pitch_error_cents"].is_null());
    CHECK(converted["tier_conversion"]["metrics"][0]["timing_error_samples"] == 0);
    source["tier_profiles"]["Standard"]["validation"]["cases"][0].erase("signal_type");
    CHECK_FALSE(alfa::tensorhost::convertTier(source.dump(), "Standard", 48000, result, error));
}
