#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "alfa/plugin/ParameterLayout.h"
#include "alfa/ui/Theme.h"
#include "alfa/ui/UiState.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <set>

using namespace alfa;

namespace
{
/// JUCE needs a message manager for processors and components.
struct JuceFixture
{
    juce::ScopedJuceInitialiser_GUI init;
};

float realValue(juce::AudioProcessorValueTreeState& state, const juce::String& id)
{
    auto* p = state.getParameter(id);
    return p->convertFrom0to1(p->getValue());
}

void setReal(juce::AudioProcessorValueTreeState& state, const juce::String& id, float v)
{
    auto* p = state.getParameter(id);
    p->setValueNotifyingHost(p->convertTo0to1(v));
}
} // namespace

TEST_CASE("Parameter IDs are unique and every spec is in the layout", "[params]")
{
    std::set<juce::String> ids;
    for (const auto& spec : params::allSpecs())
    {
        INFO(spec.id);
        CHECK(ids.insert(spec.id).second);
        CHECK(spec.min < spec.max);
        CHECK(spec.defaultValue >= spec.min);
        CHECK(spec.defaultValue <= spec.max);
        CHECK(params::findSpec(spec.id) == &spec);
    }

    CHECK(ids.size() == 1 + params::kNumMethods + params::kNumMacros + 2 + 7 + 3 + 1 +
                            params::kNumMethods * params::kNumEngineSlots);

    for (int m = 0; m < params::kNumMethods; ++m)
        CHECK(params::engineSpecs(m).size() == static_cast<size_t>(params::kNumEngineSlots));
}

TEST_CASE("Capacity shares add up to one", "[params]")
{
    float total = 0.0f;
    for (const auto& m : params::kMethods)
        total += m.capacityShare;
    CHECK(total == Catch::Approx(1.0f).margin(0.01f));
}

TEST_CASE("Values format in real units and parse back", "[params]")
{
    const auto* cutoff = params::findSpec(params::id::engine(0, 0));
    REQUIRE(cutoff != nullptr);
    CHECK(params::formatValue(*cutoff, 1200.0f) == "1.20 kHz");
    CHECK(params::parseValue(*cutoff, "1.2 kHz") == Catch::Approx(1200.0f));
    CHECK(params::parseValue(*cutoff, "440") == Catch::Approx(440.0f));

    const auto* attack = params::findSpec(params::id::attack);
    CHECK(params::formatValue(*attack, 35.0f) == "35 ms");
    CHECK(params::parseValue(*attack, "0.5 s") == Catch::Approx(500.0f));

    const auto* sustain = params::findSpec(params::id::sustain);
    CHECK(params::formatValue(*sustain, 0.7f) == "70 %");
    CHECK(params::parseValue(*sustain, "25%") == Catch::Approx(0.25f));

    const auto* gain = params::findSpec(params::id::outputGain);
    CHECK(params::formatValue(*gain, -6.0f) == "-6.0 dB");
    CHECK(params::parseValue(*gain, "100") == Catch::Approx(gain->max)); // clamped
}

TEST_CASE("Theme token files round-trip and fall back on bad input", "[ui]")
{
    const auto paper = ui::Theme::paper();
    const auto back = ui::Theme::fromJson(paper.toJson());
    CHECK(back.name == paper.name);
    CHECK(back.accent == paper.accent);
    CHECK(back.bg == paper.bg);

    const auto broken = ui::Theme::fromJson("not json");
    CHECK(broken.bg == ui::Theme::graphite().bg);

    const auto partial = ui::Theme::fromJson(R"({"name":"Mine","accent":"#112233"})");
    CHECK(partial.accent == juce::Colour(0xff112233));
    CHECK(partial.panel == ui::Theme::graphite().panel);
}

TEST_CASE("UI state keeps hidden steps and locks as lists", "[ui]")
{
    ui::UiState state;
    state.setStepHidden("model", true);
    state.setStepHidden("blend", true);
    state.setStepHidden("model", false);
    CHECK_FALSE(state.isStepHidden("model"));
    CHECK(state.isStepHidden("blend"));

    state.setLocked("amp_attack", true);
    CHECK(state.isLocked("amp_attack"));
    CHECK(state.lockedParameters().size() == 1);

    // Restoring an older tree keeps defaults for properties it doesn't have.
    juce::ValueTree old{ui::UiState::type};
    old.setProperty(ui::UiState::Ids::theme, "paper", nullptr);
    state.restore(old);
    CHECK(state.getString(ui::UiState::Ids::theme) == "paper");
    CHECK(state.getInt(ui::UiState::Ids::macrosPerPage) == 8);
}

TEST_CASE("Processor state restores parameters and UI settings", "[processor]")
{
    JuceFixture juce;
    juce::MemoryBlock saved;
    {
        AlfamedomegaProcessor p;
        setReal(p.parameters(), params::id::attack, 250.0f);
        setReal(p.parameters(), params::id::modelTier, 2.0f);
        p.uiState().set(ui::UiState::Ids::theme, "contrast");
        p.getStateInformation(saved);
    }

    AlfamedomegaProcessor restored;
    restored.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
    CHECK(realValue(restored.parameters(), params::id::attack) == Catch::Approx(250.0f).margin(0.5f));
    CHECK(juce::roundToInt(realValue(restored.parameters(), params::id::modelTier)) == 2);
    CHECK(restored.uiState().getString(ui::UiState::Ids::theme) == "contrast");
}

TEST_CASE("Loading a preset never changes UI settings, model size or locked "
          "parameters",
          "[processor]")
{
    JuceFixture juce;
    AlfamedomegaProcessor p;
    auto& state = p.parameters();

    p.uiState().set(ui::UiState::Ids::theme, "paper");
    setReal(state, params::id::modelTier, 3.0f);
    setReal(state, params::id::release, 1234.0f);
    p.uiState().setLocked(params::id::release, true);

    for (int i = 0; i < p.numPresets(); ++i)
    {
        p.loadPreset(i);
        INFO(p.presetName(i));
        CHECK(p.uiState().getString(ui::UiState::Ids::theme) == "paper");
        CHECK(juce::roundToInt(realValue(state, params::id::modelTier)) == 3);
        CHECK(realValue(state, params::id::release) == Catch::Approx(1234.0f).margin(1.0f));
    }
}

TEST_CASE("Processor renders sound for a note and reports telemetry", "[processor]")
{
    JuceFixture juce;
    AlfamedomegaProcessor p;
    p.setPlayConfigDetails(0, 2, 48000.0, 256);
    p.prepareToPlay(48000.0, 256);

    juce::AudioBuffer<float> buffer(2, 256);
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.8f), 0);

    float peak = 0.0f;
    for (int block = 0; block < 20; ++block)
    {
        p.processBlock(buffer, midi);
        midi.clear();
        peak = juce::jmax(peak, buffer.getMagnitude(0, 256));
    }

    CHECK(peak > 0.01f);
    CHECK(p.telemetry().activeVoices.load() == 1);
    CHECK(p.telemetry().noteVelocity[60].load() > 0.0f);
    CHECK(p.telemetry().engineLevel[0].load() > 0.0f); // default blend is mostly subtractive

    midi.addEvent(juce::MidiMessage::noteOff(1, 60), 0);
    p.processBlock(buffer, midi);
    CHECK_FALSE(p.telemetry().noteVelocity[60].load() > 0.0f);
}

TEST_CASE("Editor opens in every view, theme and scale", "[editor]")
{
#if JUCE_LINUX
    if (std::getenv("DISPLAY") == nullptr)
        SKIP("No display; run under xvfb-run to include editor tests");
#endif
    JuceFixture juce;
    AlfamedomegaProcessor p;
    std::unique_ptr<juce::AudioProcessorEditor> editor(p.createEditor());
    REQUIRE(editor != nullptr);
    CHECK(editor->getWidth() == 1280);

    for (const auto* view : {"perform", "signal", "sound", "customize"})
        for (const auto* theme : {"graphite", "paper", "contrast"})
        {
            p.uiState().set(ui::UiState::Ids::view, view);
            p.uiState().set(ui::UiState::Ids::theme, theme);
            auto image = editor->createComponentSnapshot(editor->getLocalBounds());
            CHECK(image.isValid());
        }

    p.uiState().set(ui::UiState::Ids::scale, 1.5);
    CHECK(editor->getWidth() == 1920);
    CHECK(editor->getHeight() == 1200);

    // Every signal-flow step can be hidden and the view still lays out.
    p.uiState().set(ui::UiState::Ids::view, "signal");
    for (const auto& step : ui::SignalFlowView::steps())
        p.uiState().setStepHidden(step.id, true);
    CHECK(editor->createComponentSnapshot(editor->getLocalBounds()).isValid());
}
