#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "alfa/plugin/ParameterLayout.h"
#include "alfa/plugin/PreferenceLog.h"
#include "alfa/ui/Theme.h"
#include "alfa/ui/UiState.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <set>

using namespace alfa;

namespace
{
/// A temporary folder removed at the end of the test.
struct TempFolder
{
    juce::File dir = juce::File::getSpecialLocation(juce::File::tempDirectory)
                         .getChildFile("alfa-tests-" + juce::Uuid().toString());
    TempFolder() { dir.createDirectory(); }
    ~TempFolder() { dir.deleteRecursively(); }
};

/// Points the control-center home at a scratch folder so no test reads or writes
/// the real user's preference data.
void isolateHome()
{
    static TempFolder home;
#if JUCE_WINDOWS
    _putenv_s("ALFA_CC_HOME", home.dir.getFullPathName().toRawUTF8());
#else
    setenv("ALFA_CC_HOME", home.dir.getFullPathName().toRawUTF8(), 1);
#endif
}

/// JUCE needs a message manager for processors and components.
struct JuceFixture
{
    JuceFixture() { isolateHome(); }
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

    // Model size, method, two hybrid components, 32 knobs, 3 voice settings, output.
    CHECK(ids.size() == 1 + 3 + params::kNumKnobs + 3 + 1);
    CHECK(params::id::knob(0) == "knob_01");
    CHECK(params::id::knob(31) == "knob_32");
}

TEST_CASE("Method order matches the model contract", "[params]")
{
    // src/inference/MODEL_CONTRACT.md: the one-hot and voice routing order.
    const char* contract[] = {"subtractive", "wavetable", "additive", "spectral", "fm",
                              "physical",    "granular",  "sampling", "hybrid"};
    for (int i = 0; i < params::kNumMethods; ++i)
        CHECK(juce::String(params::kMethods[static_cast<size_t>(i)].id) == contract[i]);

    float total = 0.0f;
    for (const auto& m : params::kMethods)
        total += m.capacityShare;
    CHECK(total == Catch::Approx(1.0f).margin(0.01f));
}

TEST_CASE("Knob meanings follow the teacher programs", "[params]")
{
    using params::Method;
    for (int m = 0; m < params::kNumMethods - 1; ++m)
    {
        int used = 0;
        for (int k = 0; k < params::kNumKnobs; ++k)
        {
            const auto info = params::knobInfo(static_cast<Method>(m), k);
            used += info.used ? 1 : 0;
            if (info.used)
                CHECK(info.name.isNotEmpty());
        }
        INFO(params::kMethods[static_cast<size_t>(m)].name);
        CHECK(used >= 6);
    }

    // Knobs 19-22 are the amp envelope for every method but physical.
    CHECK(params::knobInfo(Method::subtractive, 18).name == "Amp attack");
    CHECK(params::knobInfo(Method::granular, 21).name == "Amp release");
    CHECK_FALSE(params::knobInfo(Method::physical, 18).used);

    // Real units from the teacher curves.
    const auto cutoff = params::knobInfo(Method::subtractive, 6);
    CHECK(params::formatKnob(cutoff, 0.5f) == "807 Hz");
    CHECK(params::formatKnob(cutoff, 1.0f) == "20.00 kHz");
    CHECK(params::formatKnob(params::knobInfo(Method::subtractive, 18), 0.0f) == "1.0 ms");
    CHECK(params::formatKnob(params::knobInfo(Method::subtractive, 22), 0.3f) == "off");
    CHECK(params::formatKnob(params::knobInfo(Method::fm, 2), 0.43f).endsWith("4"));
    CHECK(params::formatKnob(params::knobInfo(Method::physical, 0), 0.0f) == "String");

    // Operators beyond Lite's two need a bigger model.
    CHECK(params::knobInfo(Method::fm, 2).minimumTier == params::Tier::lite);
    CHECK(params::knobInfo(Method::fm, 8).minimumTier == params::Tier::studio);
}

TEST_CASE("Hybrids rotate component B's knobs by 11", "[params]")
{
    using params::Method;
    CHECK(params::isValidHybridPair(Method::fm, Method::physical));
    CHECK_FALSE(params::isValidHybridPair(Method::additive, Method::physical));
    CHECK_FALSE(params::isValidHybridPair(Method::fm, Method::fm));
    CHECK_FALSE(params::isValidHybridPair(Method::sampling, Method::fm));

    // Knob 22 (index 21) is FM's amp release and physical's knob 1 (resonator).
    const auto both = params::hybridKnobInfo(Method::fm, Method::physical, 21);
    CHECK(both.name == "A Amp release / B Resonator");
    const auto crossfade = params::hybridKnobInfo(Method::fm, Method::physical, 31);
    CHECK(crossfade.name == "A to B crossfade");
    CHECK(crossfade.used);
}

TEST_CASE("Values format in real units and parse back", "[params]")
{
    const auto* glide = params::findSpec(params::id::glide);
    REQUIRE(glide != nullptr);
    CHECK(params::formatValue(*glide, 35.0f) == "35 ms");
    CHECK(params::parseValue(*glide, "0.5 s") == Catch::Approx(500.0f));

    const auto* knob = params::findSpec(params::id::knob(3));
    CHECK(params::formatValue(*knob, 0.7f) == "70 %");
    CHECK(params::parseValue(*knob, "25%") == Catch::Approx(0.25f));

    const auto* gain = params::findSpec(params::id::outputGain);
    CHECK(params::formatValue(*gain, -6.0f) == "-6.0 dB");
    CHECK(params::parseValue(*gain, "100") == Catch::Approx(gain->max)); // clamped

    const auto* method = params::findSpec(params::id::method);
    CHECK(params::formatValue(*method, 4.0f) == "FM");
    CHECK(params::parseValue(*method, "granular") == Catch::Approx(6.0f));
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

    state.setLocked("knob_19", true);
    CHECK(state.isLocked("knob_19"));
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
        setReal(p.parameters(), params::id::knob(18), 0.25f);
        setReal(p.parameters(), params::id::method, 4.0f);
        setReal(p.parameters(), params::id::modelTier, 2.0f);
        p.uiState().set(ui::UiState::Ids::theme, "contrast");
        p.getStateInformation(saved);
    }

    AlfamedomegaProcessor restored;
    restored.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
    CHECK(realValue(restored.parameters(), params::id::knob(18)) == Catch::Approx(0.25f).margin(0.001f));
    CHECK(params::readShape(restored.parameters()).method == params::Method::fm);
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
    setReal(state, params::id::knob(21), 0.42f);
    p.uiState().setLocked(params::id::knob(21), true);

    for (int i = 0; i < p.numPresets(); ++i)
    {
        p.loadPreset(i, false);
        INFO(p.presetName(i));
        CHECK(p.uiState().getString(ui::UiState::Ids::theme) == "paper");
        CHECK(juce::roundToInt(realValue(state, params::id::modelTier)) == 3);
        CHECK(realValue(state, params::id::knob(21)) == Catch::Approx(0.42f).margin(0.001f));
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
    CHECK(p.telemetry().engineLevel[0].load() > 0.0f); // the default sound is subtractive

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

TEST_CASE("An invalid hybrid is silent, as the model contract says", "[processor]")
{
    JuceFixture juce;
    AlfamedomegaProcessor p;
    setReal(p.parameters(), params::id::method, 8.0f);  // hybrid
    setReal(p.parameters(), params::id::hybridA, 2.0f); // additive
    setReal(p.parameters(), params::id::hybridB, 4.0f); // physical: not allowed with additive
    p.setPlayConfigDetails(0, 2, 48000.0, 256);
    p.prepareToPlay(48000.0, 256);

    juce::AudioBuffer<float> buffer(2, 256);
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.8f), 0);
    float peak = 0.0f;
    for (int block = 0; block < 10; ++block)
    {
        p.processBlock(buffer, midi);
        midi.clear();
        peak = juce::jmax(peak, buffer.getMagnitude(0, 256));
    }
    CHECK(peak < 1.0e-6f);
}

// Preference learning event log
// ---------------------------------------------------------------------------

namespace
{
std::vector<juce::var> readEvents(const juce::File& folder)
{
    std::vector<juce::var> out;
    for (const auto& f : folder.findChildFiles(juce::File::findFiles, false, "*.jsonl"))
    {
        juce::StringArray lines;
        f.readLines(lines);
        for (const auto& line : lines)
            if (line.trim().isNotEmpty())
                out.push_back(juce::JSON::parse(line));
    }
    std::stable_sort(out.begin(), out.end(), [](const juce::var& a, const juce::var& b)
                     { return static_cast<double>(a["t"]) < static_cast<double>(b["t"]); });
    return out;
}

juce::StringArray types(const std::vector<juce::var>& events)
{
    juce::StringArray out;
    for (const auto& e : events)
        out.add(e["type"].toString());
    return out;
}

juce::MidiBuffer notes(std::initializer_list<juce::MidiMessage> messages)
{
    juce::MidiBuffer b;
    for (const auto& m : messages)
        b.addEvent(m, static_cast<int>(m.getTimeStamp()));
    return b;
}
} // namespace

TEST_CASE("Play counter measures held time, notes, pitch and note length", "[preference]")
{
    preference::PlayCounter counter;
    auto on = juce::MidiMessage::noteOn(1, 60, 0.8f);
    auto on2 = juce::MidiMessage::noteOn(1, 64, 0.8f);
    on2.setTimeStamp(100);
    counter.process(notes({on, on2}), 200); // 60 for 200 samples, 64 for 100
    auto off = juce::MidiMessage::noteOff(1, 60);
    off.setTimeStamp(50);
    counter.process(notes({off}), 200); // 60 for 50 more, 64 for 200 more

    const auto t = counter.take();
    CHECK(t.notes == 2);
    CHECK(t.pitchSum == 124);
    CHECK(t.heldSamples == 200 + 100 + 50 + 200);
    CHECK(t.endedNotes == 1);
    CHECK(t.lengthSamples == 250);
    CHECK(counter.take().notes == 0);
}

TEST_CASE("Event log follows the learner's schema and only writes while learning is on", "[preference]")
{
    JuceFixture juce;
    TempFolder temp;
    preference::LearningSwitch learning(temp.dir);
    juce::AudioProcessorValueTreeState::ParameterLayout layout = params::createLayout();
    juce::AudioProcessorGraph dummy; // any processor can own the parameters
    juce::AudioProcessorValueTreeState state(dummy, nullptr, "Parameters", std::move(layout));
    preference::PlayCounter counter;

    double clock = 1'800'000'000.0;
    preference::Recorder::Options options;
    options.learning = &learning;
    options.now = [&clock]
    {
        return clock;
    };
    options.startTimer = false;

    {
        preference::Recorder recorder(state, counter, options);
        recorder.tick();
        clock += 10.0;
        recorder.tick();
        juce::SharedResourcePointer<preference::EventWriter>()->flush();
        CHECK(readEvents(learning.eventsFolder()).empty()); // learning is off by default

        learning.setOn(true);
        recorder.tick(); // a fresh instance's sound, once the restore window has passed

        auto on = juce::MidiMessage::noteOn(1, 48, 0.8f);
        auto off = juce::MidiMessage::noteOff(1, 48);
        off.setTimeStamp(24000);
        counter.process(notes({on}), 24000);
        counter.process(notes({off}), 48000);
        clock += 1.0;
        recorder.tick(); // play

        setReal(state, params::id::knob(6), 0.9f);
        clock += 1.0;
        recorder.tick(); // edit seen, not settled
        clock += 3.0;
        recorder.tick(); // edit logged

        recorder.stateSaved();
        clock += 1.0;
        recorder.tick(); // too soon after loading to count as kept
        clock += 400.0;
        recorder.stateSaved();
        recorder.tick(); // kept
        recorder.stateSaved();
        clock += 1.0;
        recorder.tick(); // only once per exposure

        recorder.rate(1);
        setReal(state, params::id::method, 8.0f); // hybrid, subtractive + wavetable by default
        recorder.loaded(preference::Recorder::Source::browser);
    } // close
    juce::SharedResourcePointer<preference::EventWriter>()->flush();

    const auto events = readEvents(learning.eventsFolder());
    CHECK(types(events).joinIntoString(",") == "load,play,edit,project_save,rate,load,close");
    REQUIRE(events.size() == 7);

    const auto& first = events[0];
    CHECK(first["source"].toString() == "init");
    CHECK(first["inst"].toString().length() == 16);
    CHECK(first["patch"]["method"].toString() == "subtractive");
    CHECK(first["patch"]["pair"].isVoid());
    CHECK(first["patch"]["knobs"].size() == params::kNumKnobs);

    const auto& play = events[1];
    CHECK(static_cast<double>(play["seconds"]) == Catch::Approx(1.0));
    CHECK(static_cast<int>(play["notes"]) == 1);
    CHECK(static_cast<double>(play["pitch_mean"]) == Catch::Approx(48.0));
    CHECK(static_cast<double>(play["len_mean"]) == Catch::Approx(1.0));

    CHECK(static_cast<double>(events[2]["patch"]["knobs"][6]) == Catch::Approx(0.9));
    CHECK(static_cast<int>(events[4]["value"]) == 1);

    const auto& hybrid = events[5];
    CHECK(hybrid["source"].toString() == "browser");
    CHECK(hybrid["patch"]["method"].toString() == "hybrid");
    CHECK(hybrid["patch"]["pair"][0].toString() == "subtractive");
    CHECK(hybrid["patch"]["pair"][1].toString() == "wavetable");
}

TEST_CASE("A restored project's sound is logged as a project load under its saved instance", "[preference]")
{
    JuceFixture juce;
    TempFolder temp;
    preference::LearningSwitch learning(temp.dir);
    learning.setOn(true);
    juce::AudioProcessorGraph dummy;
    juce::AudioProcessorValueTreeState state(dummy, nullptr, "Parameters", params::createLayout());
    preference::PlayCounter counter;

    double clock = 1'800'000'000.0;
    preference::Recorder::Options options;
    options.learning = &learning;
    options.now = [&clock]
    {
        return clock;
    };
    options.startTimer = false;

    {
        preference::Recorder recorder(state, counter, options);
        recorder.projectRestored("0123456789abcdef");
        clock += 0.5;
        recorder.tick();
        clock += 5.0;
        recorder.tick(); // no "init" after a restore
        CHECK(recorder.instanceId() == "0123456789abcdef");
    }
    juce::SharedResourcePointer<preference::EventWriter>()->flush();

    const auto events = readEvents(learning.eventsFolder());
    CHECK(types(events).joinIntoString(",") == "load,close");
    REQUIRE_FALSE(events.empty());
    CHECK(events[0]["source"].toString() == "project");
    CHECK(events[0]["inst"].toString() == "0123456789abcdef");
}
