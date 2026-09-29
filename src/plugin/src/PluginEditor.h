#pragma once

#include "PluginProcessor.h"
#include "alfa/ui/AlfaLookAndFeel.h"

namespace alfa
{

/// Placeholder editor. The UI owner replaces it.
class AlfamedomegaEditor final : public juce::AudioProcessorEditor
{
public:
    explicit AlfamedomegaEditor(AlfamedomegaProcessor&);
    ~AlfamedomegaEditor() override;

    void paint(juce::Graphics&) override;

private:
    ui::AlfaLookAndFeel lookAndFeel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AlfamedomegaEditor)
};

} // namespace alfa
