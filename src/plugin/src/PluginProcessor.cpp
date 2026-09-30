#include "PluginProcessor.h"

#include "PluginEditor.h"

namespace alfa
{

AlfamedomegaProcessor::AlfamedomegaProcessor()
    : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true))
    , state(*this, nullptr, "Parameters", params::createLayout())
{
    backend = createStubBackend(state);
    outputGainParam = state.getRawParameterValue(params::id::outputGain);
    state.addParameterListener(params::id::modelTier, this);

    ui.loadUserDefault();

    const auto tier = juce::roundToInt(state.getRawParameterValue(params::id::modelTier)->load());
    requestedTier.store(tier);
    backend->requestTier(static_cast<params::Tier>(tier));
    setLatencySamples(backend->latencySamples(48000.0));

    // Latency changes are applied from the message thread, and only while the
    // transport is stopped (hosts handle a mid-playback latency change badly).
    startTimerHz(10);
}

AlfamedomegaProcessor::~AlfamedomegaProcessor()
{
    stopTimer();
    state.removeParameterListener(params::id::modelTier, this);
}

void AlfamedomegaProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    backend->prepare(sampleRate, samplesPerBlock);
    loadMeasurer.reset(sampleRate, samplesPerBlock);
    outputGain.reset(sampleRate, 0.02);
    outputGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain(outputGainParam->load()));
    telem.sampleRate.store(sampleRate);
    telem.blockSize.store(samplesPerBlock);
    keyboard.reset();
    setLatencySamples(backend->latencySamples(sampleRate));
}

bool AlfamedomegaProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
}

void AlfamedomegaProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    juce::AudioProcessLoadMeasurer::ScopedTimer timer(loadMeasurer, buffer.getNumSamples());

    bool playing = false;
    if (auto* head = getPlayHead())
        if (auto pos = head->getPosition())
            playing = pos->getIsPlaying();
    transportPlaying.store(playing);

    const auto numSamples = buffer.getNumSamples();
    keyboard.processNextMidiBuffer(midi, 0, numSamples, true);

    buffer.clear();
    backend->render(buffer, midi, telem);

    outputGain.setTargetValue(juce::Decibels::decibelsToGain(outputGainParam->load()));
    for (int i = 0; i < numSamples; ++i)
    {
        const auto gain = outputGain.getNextValue();
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            buffer.getWritePointer(ch)[i] *= gain;
    }

    telem.peakL.store(buffer.getMagnitude(0, 0, numSamples));
    telem.peakR.store(buffer.getMagnitude(buffer.getNumChannels() > 1 ? 1 : 0, 0, numSamples));
    telem.pushScope(buffer.getReadPointer(0), numSamples);
    telem.cpuLoad.store(static_cast<float>(loadMeasurer.getLoadAsProportion()));
}

void AlfamedomegaProcessor::parameterChanged(const juce::String& parameterID, float newValue)
{
    if (parameterID == params::id::modelTier)
    {
        requestedTier.store(juce::roundToInt(newValue));
        backend->requestTier(static_cast<params::Tier>(juce::roundToInt(newValue)));
    }
}

void AlfamedomegaProcessor::timerCallback()
{
    if (transportPlaying.load())
        return;

    const auto latency = backend->latencySamples(telem.sampleRate.load());
    if (latency != getLatencySamples())
        setLatencySamples(latency);
}

params::Tier AlfamedomegaProcessor::loadedTier() const
{
    return static_cast<params::Tier>(telem.loadedTier.load());
}

void AlfamedomegaProcessor::loadPreset(int index)
{
    if (index < 0 || index >= presets.size())
        return;
    currentPresetIndex = index;
    PresetLibrary::apply(presets.get(index), state, ui.lockedParameters());
    updateHostDisplay(ChangeDetails().withProgramChanged(true));
}

void AlfamedomegaProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    juce::ValueTree root{stateType};
    root.setProperty("kLayoutVersion", params::kLayoutVersion, nullptr);
    root.setProperty("preset", currentPresetIndex, nullptr);
    root.appendChild(state.copyState(), nullptr);
    root.appendChild(ui.tree().createCopy(), nullptr);

    if (auto xml = root.createXml())
        copyXmlToBinary(*xml, destData);
}

void AlfamedomegaProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    auto xml = getXmlFromBinary(data, sizeInBytes);
    if (xml == nullptr)
        return;

    const auto root = juce::ValueTree::fromXml(*xml);
    if (!root.hasType(stateType))
        return;

    currentPresetIndex = juce::jlimit(0, presets.size() - 1, static_cast<int>(root.getProperty("preset", 0)));

    const auto saved = root.getChildWithName(state.state.getType());
    if (saved.isValid())
        state.replaceState(saved);

    const auto savedUi = root.getChildWithName(ui::UiState::type);
    if (savedUi.isValid())
        ui.restore(savedUi); // hosts restore state on the message thread, where the
                             // UI listens
}

juce::AudioProcessorEditor* AlfamedomegaProcessor::createEditor()
{
    return new AlfamedomegaEditor(*this);
}

} // namespace alfa

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new alfa::AlfamedomegaProcessor();
}
