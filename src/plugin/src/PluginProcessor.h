#pragma once

#include "SynthBackend.h"
#include "alfa/plugin/ParameterLayout.h"
#include "alfa/plugin/PreferenceLog.h"
#include "alfa/plugin/PresetLibrary.h"
#include "alfa/plugin/Telemetry.h"
#include "alfa/ui/EditorContext.h"
#include "alfa/ui/UiState.h"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>

namespace alfa
{

/// The plugin shell: owns parameters, presets, per-instance UI state and the
/// synth backend, and feeds telemetry to the UI.
class AlfamedomegaProcessor final : public juce::AudioProcessor,
                                    public ui::HostBridge,
                                    private juce::AudioProcessorValueTreeState::Listener,
                                    private juce::Timer
{
public:
    AlfamedomegaProcessor();
    ~AlfamedomegaProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void setNonRealtime(bool isNonRealtime) noexcept override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return presets.size(); }
    int getCurrentProgram() override { return currentPresetIndex; }
    void setCurrentProgram(int index) override { loadPreset(index, false); }
    const juce::String getProgramName(int index) override { return presets.get(index).name; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    // HostBridge
    int numPresets() const override { return presets.size(); }
    int currentPreset() const override { return currentPresetIndex; }
    juce::String presetName(int i) const override { return presets.get(i).name; }
    juce::String presetCategory(int i) const override { return presets.get(i).category; }
    params::Tier presetMadeOn(int i) const override { return presets.get(i).madeOn; }
    void loadPreset(int index, bool browsing) override;
    bool learningEnabled() const override { return learning.isOn(); }
    void setLearningEnabled(bool on) override;
    void rateSound(int value) override { recorder->rate(value); }
    int soundRating() const override { return recorder->rating(); }
    params::Tier loadedTier() const override;
    int latencySamples() const override { return getLatencySamples(); }
    juce::String backendDescription() const override { return backend->description(); }

    juce::AudioProcessorValueTreeState& parameters() noexcept { return state; }
    ui::UiState& uiState() noexcept { return ui; }
    Telemetry& telemetry() noexcept { return telem; }
    juce::MidiKeyboardState& keyboardState() noexcept { return keyboard; }

    static inline const juce::Identifier stateType{"AlfamedomegaState"};

private:
    void parameterChanged(const juce::String& parameterID, float newValue) override;
    void timerCallback() override;

    // Declared before `state`: the layout's knob text reads it.
    std::atomic<int> packedShape{0};
    juce::AudioProcessorValueTreeState state;
    ui::UiState ui;
    PresetLibrary presets;
    int currentPresetIndex = 0;

    std::unique_ptr<SynthBackend> backend;
    Telemetry telem;
    juce::MidiKeyboardState keyboard;
    juce::AudioProcessLoadMeasurer loadMeasurer;
    juce::SmoothedValue<float> outputGain;
    std::atomic<float>* outputGainParam = nullptr;
    std::atomic<bool> transportPlaying{false};
    std::atomic<int> requestedTier{1};

    preference::LearningSwitch learning;
    preference::PlayCounter playCounter;
    std::unique_ptr<preference::Recorder> recorder;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AlfamedomegaProcessor)
};

} // namespace alfa
