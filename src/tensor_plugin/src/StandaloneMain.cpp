#include "SoundLibraryPanel.h"
#include "TensorEditor.h"
#include "TensorProcessor.h"

#include <juce_audio_plugin_client/Standalone/juce_StandaloneFilterWindow.h>

#include <cmath>

namespace alfa::tensorhost
{
class TensorApplication final : public juce::JUCEApplication, private juce::Timer
{
public:
    const juce::String getApplicationName() override { return "Alfamedomega"; }
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
                processor.setPlayConfigDetails(2, 2, 48000, 64);
                processor.prepareToPlay(48000, 64);
                // Optional sound for UI review snapshots.
                if (const auto sound = juce::SystemStats::getEnvironmentVariable("ALFA_UI_DOCUMENT", {});
                    sound.isNotEmpty())
                {
                    juce::String error;
                    if (!processor.loadJSON(juce::File(sound).loadFileAsString(), error))
                        throw std::runtime_error(error.toStdString());
                }
                if (const auto theme = juce::SystemStats::getEnvironmentVariable("ALFA_UI_THEME", {});
                    theme.isNotEmpty())
                    processor.uiSettings().setProperty("theme", theme, nullptr);
                std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
                juce::AudioBuffer<float> audio(2, 64);
                audio.clear();
                juce::MidiBuffer midi;
                midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.5f), 0);
                for (int block = 0; block < 64; ++block)
                {
                    audio.clear();
                    processor.processBlock(audio, midi);
                    midi.clear();
                }
                const auto screenshot = juce::SystemStats::getEnvironmentVariable("ALFA_UI_SNAPSHOT", {});
                if (screenshot.isNotEmpty())
                {
                    if (auto* console = dynamic_cast<Editor*>(editor.get()))
                        console->snapshotView(
                            juce::SystemStats::getEnvironmentVariable("ALFA_UI_VIEW", "0").getIntValue());
                    if (juce::SystemStats::getEnvironmentVariable("ALFA_UI_SMALL", "0") == "1")
                        editor->setSize(1040, 700);
                    else if (const auto size = juce::SystemStats::getEnvironmentVariable("ALFA_UI_SIZE", {});
                             size.containsChar('x'))
                        editor->setSize(size.upToFirstOccurrenceOf("x", false, false).getIntValue(),
                                        size.fromFirstOccurrenceOf("x", false, false).getIntValue());
                    const auto image = editor->createComponentSnapshot(editor->getLocalBounds());
                    juce::FileOutputStream output{juce::File(screenshot)};
                    juce::PNGImageFormat png;
                    if (!output.openedOk() || !png.writeImageToStream(image, output))
                        throw std::runtime_error("Could not save editor snapshot");
                }
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
        auto& devices = window_->getDeviceManager();
        if (devices.getCurrentAudioDevice() == nullptr)
        {
            // A missing/unavailable capture device must not prevent instrument
            // playback. The device settings still allow enabling audio inputs.
            const auto error = devices.initialise(0, 2, nullptr, true);
            if (auto* processor = window_->getPluginHolder()->processor.get())
                if (auto* editor = dynamic_cast<Editor*>(processor->getActiveEditor()))
                    editor->reportDeviceStatus(error.isEmpty() ? "Audio input unavailable; output playback "
                                                                 "enabled. Configure inputs in Settings..."
                                                               : "Audio device unavailable: " + error +
                                                                     ". Open Settings...",
                                               error.isNotEmpty());
        }
        window_->setUsingNativeTitleBar(true);
        window_->setVisible(true);
        if (arguments.contains("--standalone-interaction-test"))
            startTimer(500);
    }
    void shutdown() override
    {
        stopTimer();
        window_.reset();
    }
    void systemRequestedQuit() override { quit(); }

private:
    void timerCallback() override
    {
        try
        {
            auto* processor = dynamic_cast<Processor*>(window_->getPluginHolder()->processor.get());
            auto* editor = processor ? processor->getActiveEditor() : nullptr;
            if (!processor || !editor)
                throw std::runtime_error("Standalone processor/editor unavailable");
            const auto require = [](bool value, const char* reason)
            {
                if (!value)
                    throw std::runtime_error(reason);
            };
            const auto button = [editor](const char* name)
            {
                for (int i = 0; i < editor->getNumChildComponents(); ++i)
                    if (auto* control = dynamic_cast<juce::Button*>(editor->getChildComponent(i));
                        control && control->getComponentID() == name)
                    {
                        control->triggerClick();
                        return;
                    }
                throw std::runtime_error("Interaction button unavailable");
            };
            if (interactionStep_ == 0)
            {
                interactionRevision_ = processor->programRevision.load();
                interactionSamples_ = processor->samples.load();
                interactionDocument_ = processor->document();
            }
            else if (interactionStep_ == 1)
            {
                require(processor->programRevision.load() == interactionRevision_,
                        "Idle UI changes executable sound");
                require(processor->samples.load() > interactionSamples_,
                        "Audio device callback is not advancing");
                button("view.structure");
                for (auto* child : editor->getChildren())
                    if (auto* list = dynamic_cast<juce::ListBox*>(child))
                        list->selectRow(1);
                button("header.library");
            }
            else if (interactionStep_ == 2)
            {
                require(processor->programRevision.load() == interactionRevision_,
                        "Instance inspection changes sound");
                auto& desktop = juce::Desktop::getInstance();
                for (int i = 0; i < desktop.getNumComponents(); ++i)
                    if (auto* dialog = dynamic_cast<juce::DialogWindow*>(desktop.getComponent(i));
                        dialog && dynamic_cast<SoundLibraryPanel*>(dialog->getContentComponent()))
                        interactionLibrary_ = dialog;
                require(interactionLibrary_ != nullptr, "Sound library window unavailable");
                auto* list = dynamic_cast<juce::ListBox*>(
                    interactionLibrary_->getContentComponent()->findChildWithID("library.list"));
                require(list != nullptr, "Sound library list unavailable");
                list->selectRow(1);
            }
            else if (interactionStep_ == 3)
            {
                require(processor->programRevision.load() == interactionRevision_,
                        "Browsing library changes sound");
                require(processor->document() == interactionDocument_, "Browsing library modifies tensors");
                require(interactionLibrary_ != nullptr, "Library closed before explicit load");
                auto* load = dynamic_cast<juce::Button*>(
                    interactionLibrary_->getContentComponent()->findChildWithID("library.load"));
                require(load != nullptr, "Explicit library load unavailable");
                load->triggerClick();
            }
            else if (interactionStep_ == 4)
            {
                require(processor->programRevision.load() == interactionRevision_ + 1,
                        "Explicit load did not publish exactly once");
                if (interactionLibrary_)
                    delete interactionLibrary_.getComponent();
                processor->keyboard.noteOn(1, 69, .8f);
                processor->setParameterValue("x0", .25f);
            }
            else if (interactionStep_ == 5)
            {
                require(processor->activeNotes.load() == 1, "Keyboard note did not reach device callback");
                require(processor->outputPeak[0].load() > 0, "Keyboard note produced no device output");
                require(processor->programRevision.load() == interactionRevision_ + 1,
                        "Idle UI repeats sound publication");
                processor->keyboard.noteOff(1, 69, 0);
                button("header.panic");
                button("view.settings");
            }
            else if (interactionStep_ == 6)
            {
                require(processor->activeNotes.load() == 0, "Panic did not release notes");
                require(processor->programRevision.load() == interactionRevision_ + 2,
                        "Panic did not publish exactly one prepared reset");
                juce::ComboBox* selector = nullptr;
                for (auto* child : editor->getChildren())
                    if (auto* control = dynamic_cast<juce::ComboBox*>(child);
                        control && control->getNumItems() == 4 && control->getItemText(0) == "Lite")
                        selector = control;
                require(selector != nullptr, "Tier selector unavailable");
                selector->setSelectedId(3, juce::sendNotificationAsync);
            }
            else if (interactionStep_ == 7)
            {
                require(processor->scalarBytes() == 8, "Tier selection did not publish float64 execution");
                require(processor->programRevision.load() == interactionRevision_ + 3,
                        "Tier selection repeated compilation");
                require(processor->failures.load() == 0, "Runtime rejected device execution");
                finishInteraction(true, "idle, instance/library inspection without mutation, explicit load, "
                                        "keyboard/audio, X controls, panic and Pro tier passed");
                return;
            }
            ++interactionStep_;
        }
        catch (const std::exception& failure)
        {
            auto diagnostic = juce::String(failure.what());
            if (auto* device = window_->getDeviceManager().getCurrentAudioDevice())
                diagnostic += " | device=" + device->getName() +
                              " open=" + juce::String(static_cast<int>(device->isOpen())) +
                              " playing=" + juce::String(static_cast<int>(device->isPlaying())) +
                              " rate=" + juce::String(device->getCurrentSampleRate()) +
                              " error=" + device->getLastError();
            else
                diagnostic += " | no current audio device";
            finishInteraction(false, diagnostic);
        }
    }
    void finishInteraction(bool success, const juce::String& result)
    {
        stopTimer();
        const auto path = juce::SystemStats::getEnvironmentVariable("ALFA_INTERACTION_RESULT", {});
        if (path.isNotEmpty())
            juce::File(path).replaceWithText((success ? "PASS: " : "FAIL: ") + result);
        setApplicationReturnValue(success ? 0 : 1);
        quit();
    }
    juce::String interactionDocument_;
    juce::Component::SafePointer<juce::DialogWindow> interactionLibrary_;
    int interactionStep_ = 0;
    uint64_t interactionRevision_ = 0, interactionSamples_ = 0;
    std::unique_ptr<juce::StandaloneFilterWindow> window_;
};
} // namespace alfa::tensorhost
JUCE_CREATE_APPLICATION_DEFINE(alfa::tensorhost::TensorApplication)
