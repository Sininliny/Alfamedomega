#pragma once
#include "TensorExecution.h"
#include "alfa/tensor/Model.h"
#include "alfa/tensor/Runtime.h"

#include <juce_audio_utils/juce_audio_utils.h>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace alfa::tensorhost
{
class Processor final : public juce::AudioProcessor,
                        private juce::Timer,
                        private juce::AudioProcessorValueTreeState::Listener,
                        private juce::MidiKeyboardState::Listener
{
public:
    Processor();
    ~Processor() override;
    void prepareToPlay(double, int) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout&) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlockBypassed(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    juce::AudioProcessorParameter* getBypassParameter() const override
    {
        return parameters_.getParameter("bypass");
    }
    const juce::String getName() const override { return "Alfamedomega"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return tailSeconds_.load(); }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Tensor configuration"; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;
    bool loadJSON(const juce::String&, juce::String& error);
    bool editConfigurationJSON(const juce::String&, juce::String& error);
    bool loadPresetJSON(const juce::String&, juce::String& error);
    bool setTier(const juce::String&, juce::String& error);
    int scalarBytes() const noexcept { return scalarBytes_.load(); }
    juce::AudioProcessorValueTreeState& parameters() noexcept { return parameters_; }
    juce::ValueTree& uiSettings() noexcept { return uiSettings_; }
    juce::UndoManager& undoManager() noexcept { return undo_; }
    void setParameterValue(const juce::String& id, float actual);
    bool editTensorParameter(size_t index, double value, juce::String& error);
    bool setParameterLock(size_t index, bool locked, juce::String& error);
    bool editLayerValue(size_t layer, size_t index, double value, juce::String& error);
    // UI/message-thread command: prepare reset storage before publication.
    void requestPanic() noexcept;
    std::array<std::atomic<float>, 2048> waveformSnapshot{};
    std::atomic<size_t> waveformCursor{0};
    std::array<std::atomic<float>, 2> inputPeak{}, outputPeak{};
    std::atomic<int> activeNotes{0};
    std::atomic<uint64_t> programRevision{0};
    static juce::String referenceDocument(int index);
    juce::String document() const;
    juce::String inspection() const
    {
        std::lock_guard lock(configurationMutex_);
        return inspection_;
    }
    std::atomic<int> mode{0}; // instrument, FX, hybrid; same executable tensor space.
    std::array<std::atomic<float>, 8> xControls{};
    std::atomic<uint64_t> samples{0}, failures{0};
    std::atomic<double> blockMilliseconds{0}, modelMilliseconds{0};
    std::array<std::atomic<float>, 16> stateSnapshot{};
    // Editor telemetry, written once per block: instance output peaks, the parameters the runtime used
    // (base plus projections) and control layer values. Sized for display; larger sounds are truncated.
    std::array<std::atomic<float>, 128> instanceLevels{};
    std::array<std::atomic<float>, 1024> liveParameters{};
    std::array<std::atomic<float>, 256> liveLayerValues{};
    std::atomic<uint64_t> telemetryBlocks{0};
    // Output and state slices per instance of the current sound, for mapping telemetry to instances.
    TensorExecution::Layout layout() const
    {
        std::lock_guard lock(configurationMutex_);
        return layout_;
    }
    juce::MidiKeyboardState keyboard;

private:
    struct Program
    {
        std::unique_ptr<TensorExecution> execution;
        size_t voiceCapacity = 0;
        double releaseSeconds = 0;
        float bendSemitones = 0;
        bool preserveState = false;
        int latencySamples = 0;
        std::vector<std::array<float, 2>> dryHistory;
        size_t dryCursor = 0;
    };
    std::atomic<int> scalarBytes_{4};
    std::vector<std::unique_ptr<Program>> owned_;
    juce::UndoManager undo_;
    juce::AudioProcessorValueTreeState parameters_;
    juce::ValueTree uiSettings_{"TensorUI"};
    void parameterChanged(const juce::String&, float) override;
    std::atomic<bool> panic_{false};
    std::atomic<double> tailSeconds_{0};
    std::array<bool, 128> voiceUsed_{}, deferredRelease_{};
    std::array<bool, 16> sustain_{};
    std::array<uint64_t, 128> releaseSamples_{};
    void clearPerformance() noexcept;
    bool indexedEvent(const juce::MidiMessage&) noexcept;
    // Per-sample smoothed copies of the host controls, so automation and bypass do not click.
    // The first block after prepareToPlay snaps them to their targets instead of ramping.
    float gainCurrent_ = 1, inputGainCurrent_ = 1, mixCurrent_ = 1, bypassCurrent_ = 0;
    bool snapControls_ = true;
    std::atomic<float>* inputGain_ = nullptr;
    std::atomic<float>* outputGain_ = nullptr;
    std::atomic<float>* mix_ = nullptr;
    std::atomic<float>* bypass_ = nullptr;
    size_t waveformWrite_ = 0;
    class ConfigurationEdit;
    bool loadJSONImpl(const juce::String&, juce::String& error, bool preserveState);
    void processBlockImpl(juce::AudioBuffer<float>&, juce::MidiBuffer&, bool forceBypass);
    juce::MidiBuffer mergedMidi_;
    struct KeyEvent
    {
        int channel = 1, note = 60;
        float velocity = 0;
        bool on = false;
    };
    std::array<KeyEvent, 256> keyEvents_{};
    std::atomic<size_t> keyRead_{0}, keyWrite_{0};
    void queueKey(int channel, int note, float velocity, bool on) noexcept;
    void handleNoteOn(juce::MidiKeyboardState*, int, int, float) override;
    void handleNoteOff(juce::MidiKeyboardState*, int, int, float) override;
    std::atomic<Program*> pending_{nullptr};
    Program* active_ = nullptr;
    juce::String json_, inspection_;
    TensorExecution::Layout layout_;
    mutable std::mutex configurationMutex_;
    std::atomic<double> rate_{48000};
    uint64_t clock_ = 0;
    std::array<float, 128 * 10> notes_{};
    size_t noteCount_ = 0;
    std::array<Program*, 64> retired_{};
    std::atomic<size_t> retireRead_{0}, retireWrite_{0};
    std::array<float, 16 * 3> gestures_{};
    std::array<float, 8> context_{};
    std::array<float, 5> preferences_{};
    std::array<std::string, 2> midiAxes_{"note", "feature"};
    std::array<size_t, 2> midiShape_{0, 10};
    std::array<std::string, 2> gestureAxes_{"channel", "feature"};
    std::array<size_t, 2> gestureShape_{16, 3};
    std::array<std::string, 1> audioAxes_{"channel"}, contextAxes_{"feature"}, preferenceAxes_{"feature"};
    std::array<size_t, 1> stereoShape_{2}, contextShape_{8}, preferenceShape_{5};
    void event(const juce::MidiMessage&) noexcept;
    void timerCallback() override;
};
} // namespace alfa::tensorhost
