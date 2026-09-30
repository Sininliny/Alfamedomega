#pragma once

#include "alfa/plugin/ParameterLayout.h"

#include <juce_audio_processors/juce_audio_processors.h>

#include <atomic>
#include <functional>
#include <optional>

// The behaviour event log for preference learning (docs/preference/design.md,
// sections 3.1 and 8.1). Events go to <home>/preference/events/<date>.jsonl in
// the schema of preference/alfapref/events.py, and only while "Learn from how
// I play" is on: the learner treats a present log as consent.
//
// Nothing here runs on the audio thread except PlayCounter::process, which only
// touches atomics.
namespace alfa::preference
{

/// The control center's per-user folder (apps/control-center settings.py,
/// default_home): ALFA_CC_HOME if set, otherwise the platform's app data folder.
juce::File controlCenterHome();

/// The learning switch, shared by every instance and by the control center:
/// <home>/preference/plugin.json, {"learn": true|false}. Off unless the file
/// says on.
class LearningSwitch
{
public:
    explicit LearningSwitch(juce::File preferenceFolder = controlCenterHome().getChildFile("preference"));

    bool isOn() const;
    void setOn(bool on);

    juce::File eventsFolder() const { return folder.getChildFile("events"); }

private:
    juce::File folder;
    mutable juce::Time lastChecked;
    mutable juce::Time lastModified;
    mutable bool cached = false;
};

/// A sound in the learner's terms: {"method", "pair", "knobs"}.
juce::var patchToVar(const params::SoundShape& shape, const std::array<float, params::kNumKnobs>& knobs);
juce::var capturePatch(const juce::AudioProcessorValueTreeState& state);

/// Counts what is played. The audio thread calls process() once per block;
/// the message thread takes the totals once a second.
class PlayCounter
{
public:
    void process(const juce::MidiBuffer& midi, int numSamples) noexcept;
    void reset() noexcept;

    struct Totals
    {
        int64_t heldSamples = 0; // summed over held notes
        int notes = 0;
        int64_t pitchSum = 0;
        int64_t lengthSamples = 0; // of notes that ended
        int endedNotes = 0;
    };
    Totals take() noexcept;

private:
    std::array<int64_t, 128> onAt{}; // audio thread only; -1 when not held
    std::array<bool, 128> held{};
    int heldCount = 0;
    int64_t clock = 0;

    std::atomic<int64_t> heldSamples{0}, pitchSum{0}, lengthSamples{0};
    std::atomic<int> notes{0}, endedNotes{0};
};

/// Appends lines from any instance on one background thread, so no instance
/// blocks on the disk. Shared through juce::SharedResourcePointer.
class EventWriter : private juce::Thread
{
public:
    EventWriter();
    ~EventWriter() override;

    void append(const juce::File& file, const juce::String& line);

    /// Blocks until everything queued so far is on disk (tests).
    void flush();

private:
    void run() override;
    void drain();

    juce::CriticalSection lock;
    std::vector<std::pair<juce::File, juce::String>> queue;
    juce::WaitableEvent wake;
};

/// Turns one instance's activity into events. Lives on the message thread.
class Recorder : private juce::Timer
{
public:
    enum class Source
    {
        browser,  // stepped to with previous/next
        project,  // restored with the host project
        generate, // proposed by Generate
        init,     // a fresh instance's default sound
        user      // picked deliberately
    };

    struct Options
    {
        LearningSwitch* learning = nullptr;
        std::function<double()> now;     // unix seconds; wall clock if empty
        double initDelaySeconds = 2.0;   // wait this long for a project restore
        double keptAfterSeconds = 300.0; // design.md 8.1: project_save counts after 5 minutes
        double editSettleSeconds = 2.0;  // an edit is logged once the sound stops changing
        double sampleRate = 48000.0;
        bool startTimer = true;
    };

    Recorder(juce::AudioProcessorValueTreeState& state, PlayCounter& counter, Options options);
    ~Recorder() override;

    /// A new sound was loaded. Message thread.
    void loaded(Source source);

    /// The host restored a project. Any thread; handled on the next tick.
    void projectRestored(const juce::String& instanceId);

    /// The host asked for state (a save, autosave or undo point). Any thread.
    void stateSaved() noexcept { saveRequested.store(true); }

    /// Like (+1) or dislike (-1) the current sound; 0 clears. Message thread.
    void rate(int value);
    int rating() const noexcept { return currentRating; }

    void setSampleRate(double sr) noexcept { sampleRate.store(sr); }

    juce::String instanceId() const { return inst; }

    /// One tick of the 1 Hz timer; public so tests can drive it.
    void tick();

private:
    void timerCallback() override { tick(); }
    double now() const;
    bool learning() const;
    void emit(const juce::String& type, juce::DynamicObject::Ptr fields = {});
    void startExposure(Source source);
    void endExposure();
    void flushPlay();

    juce::AudioProcessorValueTreeState& state;
    PlayCounter& counter;
    Options options;
    juce::SharedResourcePointer<EventWriter> writer;

    juce::String inst;
    double createdAt = 0.0;
    bool hasExposure = false;
    double exposureStart = 0.0;
    bool kept = false;
    int currentRating = 0;
    juce::var lastPatch;
    juce::var pendingPatch;
    double pendingSince = 0.0;
    std::optional<Source> pendingLoad;

    std::atomic<bool> saveRequested{false};
    std::atomic<bool> restoreRequested{false};
    juce::CriticalSection restoreLock;
    juce::String restoredId;
    std::atomic<double> sampleRate{48000.0};
};

juce::String sourceName(Recorder::Source source);

} // namespace alfa::preference
