// Renders each view of the editor to PNG, with a note held so the
// visualisations have something to show. Usage: alfa_ui_snapshot <output-dir>
// [theme]

#include "PluginEditor.h"
#include "PluginProcessor.h"

int main(int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI init;

    const juce::File outDir = argc > 1 ? juce::File::getCurrentWorkingDirectory().getChildFile(argv[1])
                                       : juce::File::getCurrentWorkingDirectory();
    const juce::String theme = argc > 2 ? argv[2] : "graphite";
    outDir.createDirectory();

    alfa::AlfamedomegaProcessor processor;
    processor.setPlayConfigDetails(0, 2, 48000.0, 512);
    processor.prepareToPlay(48000.0, 512);
    processor.loadPreset(1);

    std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
    auto& main = dynamic_cast<alfa::AlfamedomegaEditor&>(*editor).mainComponent();
    processor.uiState().set(alfa::ui::UiState::Ids::theme, theme);

    // Play a chord through the processor so notes, engines, scope and spectrum
    // are live.
    juce::AudioBuffer<float> buffer(2, 512);
    juce::MidiBuffer midi;
    for (auto note : {48, 55, 60, 64})
        midi.addEvent(juce::MidiMessage::noteOn(1, note, 0.8f), 0);
    for (int block = 0; block < 120; ++block)
    {
        processor.processBlock(buffer, midi);
        midi.clear();
        if (block % 3 == 0)
            main.liveData().update();
    }

    auto save = [&](const juce::String& name)
    {
        main.liveData().update();
        const auto image = editor->createComponentSnapshot(editor->getLocalBounds(), true, 1.0f);
        const auto file = outDir.getChildFile(name + ".png");
        file.deleteFile();
        juce::FileOutputStream out(file);
        juce::PNGImageFormat().writeImageToStream(image, out);
        std::cout << file.getFullPathName() << std::endl;
    };

    using Ids = alfa::ui::UiState::Ids;
    processor.uiState().set(Ids::view, "perform");
    save("perform-" + theme);

    processor.uiState().set(Ids::showFlowStrip, true);
    save("perform-flowstrip-" + theme);
    processor.uiState().set(Ids::showFlowStrip, false);

    processor.uiState().set(Ids::view, "signal");
    for (const auto* level : {"musician", "student", "developer"})
    {
        processor.uiState().set(Ids::flowDetail, level);
        save("signal-" + juce::String(level) + "-" + theme);
    }
    processor.uiState().setStepHidden("controls", true);
    processor.uiState().setStepHidden("envfx", true);
    processor.uiState().set(Ids::flowSelected, 4);
    save("signal-hidden-steps-" + theme);
    processor.uiState().setStepHidden("controls", false);
    processor.uiState().setStepHidden("envfx", false);

    processor.uiState().set(Ids::view, "sound");
    processor.uiState().set(Ids::soundLayer, 6);
    save("sound-" + theme);

    processor.uiState().set(Ids::view, "customize");
    save("customize-" + theme);
    return 0;
}
