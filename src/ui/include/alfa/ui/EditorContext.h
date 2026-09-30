#pragma once

#include "alfa/plugin/ParameterLayout.h"
#include "alfa/plugin/Telemetry.h"
#include "alfa/ui/AlfaLookAndFeel.h"
#include "alfa/ui/UiState.h"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>

namespace alfa::ui
{

/// What the UI asks of the processor beyond parameters and telemetry. Keeping
/// it an interface lets the UI build and be tested without the processor.
class HostBridge
{
public:
    virtual ~HostBridge() = default;

    virtual int numPresets() const = 0;
    virtual int currentPreset() const = 0;
    virtual juce::String presetName(int index) const = 0;
    virtual juce::String presetCategory(int index) const = 0;
    virtual params::Tier presetMadeOn(int index) const = 0;
    /// `browsing` is true when stepping with previous/next, false for a
    /// deliberate pick; preference learning weighs the two differently.
    virtual void loadPreset(int index, bool browsing) = 0;

    /// "Learn from how I play" (preference learning), shared by all instances.
    virtual bool learningEnabled() const = 0;
    virtual void setLearningEnabled(bool on) = 0;

    /// Like (+1), dislike (-1) or clear (0) the current sound.
    virtual void rateSound(int value) = 0;
    virtual int soundRating() const = 0;

    /// Model size actually playing (switches complete in the background).
    virtual params::Tier loadedTier() const = 0;
    virtual int latencySamples() const = 0;
    virtual juce::String backendDescription() const = 0;
};

/// Everything a view needs, passed by reference from the editor.
struct EditorContext
{
    juce::AudioProcessorValueTreeState& params;
    const Telemetry& telemetry;
    UiState& ui;
    juce::MidiKeyboardState& keyboard;
    HostBridge& host;
    AlfaLookAndFeel& lnf;

    const Theme& theme() const noexcept { return lnf.theme(); }

    params::Tier selectedTier() const
    {
        return static_cast<params::Tier>(
            juce::roundToInt(params.getRawParameterValue(params::id::modelTier)->load()));
    }
};

} // namespace alfa::ui
