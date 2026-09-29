#include "PluginProcessor.h"

#include "PluginEditor.h"

namespace alfa
{

AlfamedomegaProcessor::AlfamedomegaProcessor()
    : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

void AlfamedomegaProcessor::prepareToPlay(double, int)
{
}

bool AlfamedomegaProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
}

void AlfamedomegaProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    buffer.clear();
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
