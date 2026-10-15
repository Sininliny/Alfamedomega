// Host compatibility and robustness: bus layouts, block sizes, sample rates, MIDI timing,
// session migration and sessions this build cannot restore.
#include "FactoryPrograms.h"
#include "TensorProcessor.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <locale>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <vector>

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
JSON session(Processor& processor)
{
    juce::MemoryBlock saved;
    processor.getStateInformation(saved);
    return JSON::parse(std::string(static_cast<const char*>(saved.getData()), saved.getSize()));
}
void restore(Processor& processor, const std::string& bytes)
{
    processor.setStateInformation(bytes.data(), static_cast<int>(bytes.size()));
}
float actual(Processor& processor, const char* id)
{
    auto* parameter = processor.parameters().getParameter(id);
    REQUIRE(parameter != nullptr);
    return parameter->convertFrom0to1(parameter->getValue());
}
struct GestureCounter final : juce::AudioProcessorListener
{
    int gestures = 0, changes = 0;
    void audioProcessorParameterChanged(juce::AudioProcessor*, int, float) override { ++changes; }
    void audioProcessorChanged(juce::AudioProcessor*, const ChangeDetails&) override {}
    void audioProcessorParameterChangeGestureBegin(juce::AudioProcessor*, int) override { ++gestures; }
};
// Renders `total` samples of a 220 Hz input with a note in the middle, cut into blocks of the
// given sizes (cycled), and returns the interleaved output.
std::vector<float> renderInBlocks(const juce::String& sound, int mode, const std::vector<int>& sizes,
                                  int total)
{
    Processor processor;
    processor.prepareToPlay(48000, 4096);
    juce::String error;
    REQUIRE(processor.loadPresetJSON(sound, error));
    processor.setParameterValue("mode", static_cast<float>(mode));
    std::vector<float> output;
    int done = 0;
    size_t next = 0;
    while (done < total)
    {
        const auto size = std::min(sizes[next++ % sizes.size()], total - done);
        juce::AudioBuffer<float> audio(2, size);
        juce::MidiBuffer midi;
        for (int i = 0; i < size; ++i)
            for (int channel = 0; channel < 2; ++channel)
                audio.setSample(channel, i,
                                static_cast<float>(.3 * std::sin(2 * juce::MathConstants<double>::pi * 220 *
                                                                 (done + i) / 48000.0)));
        for (const auto& [at, on] : {std::pair{1000, true}, std::pair{3000, false}})
            if (at >= done && at < done + size)
                midi.addEvent(on ? juce::MidiMessage::noteOn(1, 60, .8f) : juce::MidiMessage::noteOff(1, 60),
                              at - done);
        processor.processBlock(audio, midi);
        for (int i = 0; i < size; ++i)
            for (int channel = 0; channel < 2; ++channel)
                output.push_back(audio.getSample(channel, i));
        done += size;
    }
    return output;
}
} // namespace

TEST_CASE("Output does not depend on how the host cuts blocks", "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    // FL Studio and offline renders pass variable and odd block sizes, including empty blocks.
    for (const auto& [id, mode] : {std::pair{"glass-partials", 2}, std::pair{"moving-delay", 1}})
    {
        INFO(id);
        const auto reference = renderInBlocks(factory(id), mode, {4096}, 4096);
        double energy = 0;
        for (const auto value : reference)
            energy += std::abs(value);
        REQUIRE(energy > 0);
        for (const auto& sizes : std::vector<std::vector<int>>{{1}, {7, 0, 13}, {64, 513, 3}, {4095, 1}})
            CHECK(renderInBlocks(factory(id), mode, sizes, 4096) == reference);
    }
}

TEST_CASE("Mono input on a stereo output feeds both channels", "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    auto layout = processor.getBusesLayout();
    layout.inputBuses.set(0, juce::AudioChannelSet::mono());
    layout.outputBuses.set(0, juce::AudioChannelSet::stereo());
    REQUIRE(processor.isBusesLayoutSupported(layout));
    REQUIRE(processor.setBusesLayout(layout));
    REQUIRE(processor.getTotalNumInputChannels() == 1);
    processor.prepareToPlay(48000, 64);
    juce::String error;
    REQUIRE(processor.loadPresetJSON(factory("stereo-field"), error));
    processor.setParameterValue("mode", 1);
    juce::AudioBuffer<float> audio(2, 64);
    juce::MidiBuffer midi;
    // Dry path: the mono input appears on both outputs; whatever the host left in channel 1 is ignored.
    processor.setParameterValue("mix", 0);
    for (int block = 0; block < 8; ++block)
    {
        for (int i = 0; i < 64; ++i)
        {
            audio.setSample(0, i, .25f);
            audio.setSample(1, i, .9f);
        }
        processor.processBlock(audio, midi);
    }
    CHECK(audio.getSample(0, 63) == Catch::Approx(.25f));
    CHECK(audio.getSample(1, 63) == Catch::Approx(.25f));
    // Wet path: a sound with a separate right-channel path still hears the mono input there.
    processor.setParameterValue("mix", 1);
    double left = 0, right = 0;
    for (int block = 0; block < 32; ++block)
    {
        for (int i = 0; i < 64; ++i)
        {
            audio.setSample(0, i,
                            static_cast<float>(.3 * std::sin(2 * juce::MathConstants<double>::pi * 220 *
                                                             (block * 64 + i) / 48000.0)));
            audio.setSample(1, i, 0);
        }
        processor.processBlock(audio, midi);
        for (int i = 0; i < 64; ++i)
        {
            left += std::abs(audio.getSample(0, i));
            right += std::abs(audio.getSample(1, i));
        }
    }
    CHECK(left > 0);
    CHECK(right > .1 * left);
    CHECK(processor.failures.load() == 0);
}

TEST_CASE("Mono output carries both channels of a stereo sound", "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor stereo, mono;
    auto layout = mono.getBusesLayout();
    layout.inputBuses.set(0, juce::AudioChannelSet::mono());
    layout.outputBuses.set(0, juce::AudioChannelSet::mono());
    REQUIRE(mono.setBusesLayout(layout));
    juce::String error;
    for (auto* processor : {&stereo, &mono})
    {
        processor->prepareToPlay(48000, 32);
        REQUIRE(processor->loadJSON(sine().dump(), error));
    }
    juce::AudioBuffer<float> two(2, 32), one(1, 32);
    juce::MidiBuffer midi;
    two.clear();
    one.clear();
    stereo.processBlock(two, midi);
    mono.processBlock(one, midi);
    for (int i = 0; i < 32; ++i)
        CHECK(one.getSample(0, i) ==
              Catch::Approx(.5f * (two.getSample(0, i) + two.getSample(1, i))).margin(1e-6));
    CHECK(std::abs(one.getSample(0, 31)) > 0);
}

TEST_CASE("Host buffers with fewer channels than the layout are never over-read",
          "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 16);
    processor.setParameterValue("mode", 1);
    juce::AudioBuffer<float> audio(1, 16);
    juce::MidiBuffer midi;
    audio.clear();
    processor.processBlock(audio, midi);
    for (int i = 0; i < 16; ++i)
        CHECK(std::isfinite(audio.getSample(0, i)));
}

TEST_CASE("MIDI stamped outside the block or behind large SysEx still plays", "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 64);
    juce::String error;
    REQUIRE(processor.loadPresetJSON(factory("glass-partials"), error));
    juce::AudioBuffer<float> audio(2, 64);
    juce::MidiBuffer midi;
    // A note-on stamped exactly at numSamples used to be dropped, and its note-off then had nothing
    // to release; a note-off stamped there left the note stuck.
    audio.clear();
    midi.addEvent(juce::MidiMessage::noteOn(1, 60, .8f), 64);
    processor.processBlock(audio, midi);
    CHECK(processor.activeNotes.load() == 1);
    midi.clear();
    audio.clear();
    midi.addEvent(juce::MidiMessage::noteOff(1, 60), 200);
    processor.processBlock(audio, midi);
    CHECK(processor.activeNotes.load() == 0);
    // A 70 KB SysEx dump (sample libraries, controller setup) used to silence the whole block.
    std::vector<juce::uint8> dump(70000, 0x11);
    dump.front() = 0xf0;
    dump.back() = 0xf7;
    midi.clear();
    midi.addEvent(juce::MidiMessage(dump.data(), static_cast<int>(dump.size())), 0);
    midi.addEvent(juce::MidiMessage::noteOn(1, 64, .8f), 10);
    const auto failures = processor.failures.load();
    double energy = 0;
    for (int block = 0; block < 4; ++block)
    {
        audio.clear();
        processor.processBlock(audio, midi);
        midi.clear();
        for (int i = 0; i < 64; ++i)
            energy += std::abs(audio.getSample(0, i));
    }
    CHECK(processor.activeNotes.load() == 1);
    CHECK(energy > 0);
    CHECK(processor.failures.load() == failures);
}

TEST_CASE("Unusual host sample rates compile and render finite audio", "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    for (const double rate : {22050.0, 32000.0, 88200.0, 176400.0, 352800.0, 384000.0})
    {
        INFO(rate);
        std::string error;
        CHECK(alfa::tensorhost::validateFactoryPrograms(rate, error));
        INFO(error);
        Processor processor;
        processor.prepareToPlay(rate, 256);
        juce::String loadError;
        REQUIRE(processor.loadPresetJSON(factory("glass-partials"), loadError));
        juce::AudioBuffer<float> audio(2, 256);
        juce::MidiBuffer midi;
        midi.addEvent(juce::MidiMessage::noteOn(1, 72, .8f), 0);
        double energy = 0;
        for (int block = 0; block < 8; ++block)
        {
            audio.clear();
            processor.processBlock(audio, midi);
            midi.clear();
            for (int i = 0; i < 256; ++i)
            {
                REQUIRE(std::isfinite(audio.getSample(0, i)));
                energy += std::abs(audio.getSample(0, i));
            }
        }
        CHECK(energy > 0);
        CHECK(processor.hostWarning().isEmpty());
    }
}

TEST_CASE("A sample-rate change recompiles without sending host gestures", "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(44100, 512);
    juce::String error;
    REQUIRE(processor.loadPresetJSON(factory("glass-partials"), error));
    processor.setParameterValue("x0", .37f);
    const auto revision = processor.programRevision.load();
    GestureCounter counter;
    processor.addListener(&counter);
    processor.prepareToPlay(96000, 128);
    processor.prepareToPlay(48000, 1024);
    processor.removeListener(&counter);
    CHECK(counter.gestures == 0);
    CHECK(counter.changes == 0);
    CHECK(processor.programRevision.load() > revision);
    CHECK(processor.xControls[0].load() == Catch::Approx(.37f));
}

TEST_CASE("Sessions with unknown or out-of-range host parameters still restore",
          "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 32);
    juce::String error;
    REQUIRE(processor.loadJSON(sine().dump(), error));
    processor.setParameterValue("output_gain", -6);
    auto saved = session(processor);
    auto alternative = sine();
    alternative["tensors"]["parameters"]["values"] = {300., .3};
    saved["configuration"] = alternative.dump();
    auto xml = juce::parseXML(juce::String(saved.at("parameters").get<std::string>()));
    REQUIRE(xml != nullptr);
    // A parameter a later version added or this one removed, and a value outside today's range.
    auto* future = xml->createNewChildElement("PARAM");
    future->setAttribute("id", "future_parameter");
    future->setAttribute("value", 3.0);
    xml->getChildByAttribute("id", "mix")->setAttribute("value", 1.5);
    xml->getChildByAttribute("id", "input_gain")->setAttribute("value", "-12.5");
    saved["parameters"] = xml->toString().toStdString();
    saved["x_controls"] = {0.25, 1.5};
    saved["mode"] = 7;

    Processor restored;
    restored.prepareToPlay(48000, 32);
    // Number parsing must not follow the process locale (a comma decimal separator here).
    struct Comma final : std::numpunct<char>
    {
        char do_decimal_point() const override { return ','; }
    };
    const auto previous = std::locale::global(std::locale(std::locale::classic(), new Comma));
    restore(restored, saved.dump());
    std::locale::global(previous);
    CHECK(restored.hostWarning().isEmpty());
    CHECK_FALSE(restored.keepsUnrestoredSession());
    CHECK(JSON::parse(restored.document().toStdString())["tensors"]["parameters"]["values"][0] == 300.);
    CHECK(actual(restored, "mix") == 1.0f);
    CHECK(actual(restored, "input_gain") == Catch::Approx(-12.5f));
    CHECK(actual(restored, "output_gain") == Catch::Approx(-6.0f));
    CHECK(restored.xControls[0].load() == Catch::Approx(.25f));
    CHECK(restored.xControls[1].load() == 1.0f);
    CHECK(restored.mode.load() == 2);
}

TEST_CASE("A session this build cannot restore is kept verbatim and reported", "[tensor-host][compatibility]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    Processor processor;
    processor.prepareToPlay(48000, 32);
    juce::String error;
    REQUIRE(processor.loadJSON(sine().dump(), error));
    const auto running = processor.document();
    auto newer = session(processor);
    newer["state_schema"] = "aom.host/3";
    newer["configuration"] = "{\"schema\": \"aom.tensor/2\"}";
    const auto bytes = newer.dump();
    restore(processor, bytes);
    // The running sound stays, the user is told, and saving returns the session untouched.
    CHECK(processor.document() == running);
    CHECK(processor.keepsUnrestoredSession());
    CHECK(processor.hostWarning().contains("could not be restored"));
    CHECK(processor.hostWarningRevision.load() > 0);
    juce::MemoryBlock saved;
    processor.getStateInformation(saved);
    CHECK(saved.toString().toStdString() == bytes);
    // A sample-rate change or panic does not discard it; loading a sound does.
    processor.prepareToPlay(96000, 64);
    processor.requestPanic();
    CHECK(processor.keepsUnrestoredSession());
    REQUIRE(processor.loadJSON(sine().dump(), error));
    CHECK_FALSE(processor.keepsUnrestoredSession());
    CHECK(session(processor).at("state_schema") == "aom.host/2");
    // A good session after a bad one restores normally and clears the kept bytes.
    restore(processor, bytes);
    CHECK(processor.keepsUnrestoredSession());
    const auto kept = session(processor);
    Processor other;
    other.prepareToPlay(48000, 32);
    REQUIRE(other.loadJSON(sine().dump(), error));
    const auto fresh = session(other).dump();
    restore(processor, fresh);
    CHECK_FALSE(processor.keepsUnrestoredSession());
    CHECK(kept.dump() == JSON::parse(bytes).dump());
}
