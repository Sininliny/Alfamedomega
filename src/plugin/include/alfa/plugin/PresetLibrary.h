#pragma once

#include "alfa/plugin/ParameterLayout.h"

#include <juce_audio_processors/juce_audio_processors.h>

namespace alfa
{
/// A preset stores parameter values and the model size it was made on, never UI
/// layout (ui-design.md principle 5). Factory presets are defined in code until
/// a preset file format and a sound-design pass exist.
struct Preset
{
    juce::String name;
    juce::String category;
    params::Tier madeOn = params::Tier::standard;
    std::vector<std::pair<juce::String, float>> values; // parameter ID, real value; others at default
};

class PresetLibrary
{
public:
    PresetLibrary();

    [[nodiscard]] int size() const noexcept { return static_cast<int>(presets.size()); }
    [[nodiscard]] const Preset& get(int index) const { return presets[static_cast<size_t>(index)]; }

    /// Applies a preset: every unlocked parameter goes to the preset's value or
    /// its default. Locked parameters keep their current value.
    static void apply(const Preset& preset, juce::AudioProcessorValueTreeState& state,
                      const juce::StringArray& lockedParamIds);

    /// Captures the current parameter values as a preset.
    static Preset capture(const juce::String& name, const juce::String& category, params::Tier madeOn,
                          juce::AudioProcessorValueTreeState& state);

private:
    std::vector<Preset> presets;
};

} // namespace alfa
