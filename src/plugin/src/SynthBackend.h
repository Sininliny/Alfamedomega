#pragma once

#include "alfa/plugin/ParameterLayout.h"
#include "alfa/plugin/Telemetry.h"

#include <juce_audio_processors/juce_audio_processors.h>

#include <memory>

namespace alfa
{
/// What the processor needs from the sound-producing side: the inference host
/// (Alfa::inference) turning notes and controls into control frames, and the
/// DSP engines (Alfa::dsp) rendering them. Until those components land, a stub
/// backend plays simple stand-in engines so the shell and UI can be exercised.
class SynthBackend
{
public:
    virtual ~SynthBackend() = default;

    /// Called from prepareToPlay.
    virtual void prepare(double sampleRate, int maxBlockSize) = 0;

    /// Audio thread. Renders into `buffer` (already cleared) and updates
    /// telemetry.
    virtual void render(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi,
                        Telemetry& telemetry) = 0;

    /// Message thread. Starts loading a model size; the old one keeps playing
    /// until ready.
    virtual void requestTier(params::Tier tier) = 0;

    /// Latency the current model adds, in samples.
    [[nodiscard]] virtual int latencySamples(double sampleRate) const = 0;

    /// Shown in the Developer detail level of the Signal flow view.
    [[nodiscard]] virtual juce::String description() const = 0;
};

std::unique_ptr<SynthBackend> createStubBackend(juce::AudioProcessorValueTreeState& state);

} // namespace alfa
