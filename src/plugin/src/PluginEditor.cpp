#include "PluginEditor.h"

namespace alfa
{

AlfamedomegaEditor::AlfamedomegaEditor(AlfamedomegaProcessor& p)
    : AudioProcessorEditor(p)
{
    setLookAndFeel(&lookAndFeel);
    setSize(1280, 800);
}

AlfamedomegaEditor::~AlfamedomegaEditor()
{
    setLookAndFeel(nullptr);
}

void AlfamedomegaEditor::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    g.setColour(juce::Colour(0xffece8e1));
    g.setFont(20.0f);
    g.drawFittedText("Alfamedomega", getLocalBounds(), juce::Justification::centred, 1);
}

} // namespace alfa
