#include "TensorProcessor.h"

#include <juce_audio_plugin_client/Standalone/juce_StandaloneFilterWindow.h>

#include <cmath>

namespace alfa::tensorhost
{
class TensorApplication final : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override { return "Alfamedomega Tensor"; }
    const juce::String getApplicationVersion() override { return "0.1.0"; }
    bool moreThanOneInstanceAllowed() override { return true; }
    void anotherInstanceStarted(const juce::String&) override {}
    void initialise(const juce::String& arguments) override
    {
        if (arguments.contains("--tensor-smoke-test"))
        {
            try
            {
                Processor processor;
                processor.prepareToPlay(48000, 64);
                std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
                juce::AudioBuffer<float> audio(2, 64);
                audio.clear();
                juce::MidiBuffer midi;
                midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.5f), 0);
                processor.processBlock(audio, midi);
                const auto instrumentOK = audio.getMagnitude(0, 64) > 0 && processor.failures.load() == 0;
                juce::String error;
                const auto compiled = processor.loadJSON(Processor::referenceDocument(1), error);
                processor.mode.store(1);
                audio.clear();
                midi.clear();
                audio.setSample(0, 0, 0.4f);
                processor.processBlock(audio, midi);
                const auto fxOK = audio.getMagnitude(0, 64) > 0 && processor.failures.load() == 0;
                juce::MemoryBlock state;
                processor.getStateInformation(state);
                Processor restored;
                restored.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
                const auto stateOK = restored.document() == processor.document();
                setApplicationReturnValue(instrumentOK && compiled && fxOK && stateOK ? 0 : 1);
            }
            catch (const std::exception&)
            {
                setApplicationReturnValue(2);
            }
            quit();
            return;
        }
        window_ = std::make_unique<juce::StandaloneFilterWindow>(
            getApplicationName(), juce::Colour(0xff20242b), nullptr, false, juce::String{}, nullptr,
            juce::Array<juce::StandalonePluginHolder::PluginInOuts>{}, true);
        window_->setUsingNativeTitleBar(true);
        window_->setVisible(true);
    }
    void shutdown() override { window_.reset(); }
    void systemRequestedQuit() override { quit(); }

private:
    std::unique_ptr<juce::StandaloneFilterWindow> window_;
};
} // namespace alfa::tensorhost
JUCE_CREATE_APPLICATION_DEFINE(alfa::tensorhost::TensorApplication)
