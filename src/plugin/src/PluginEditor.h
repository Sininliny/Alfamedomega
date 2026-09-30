#pragma once

#include "PluginProcessor.h"
#include "alfa/ui/AlfaLookAndFeel.h"
#include "alfa/ui/Views.h"

namespace alfa
{

/// Hosts the UI at its design size and scales it uniformly (75% to 200%), with
/// the window's aspect ratio fixed.
class AlfamedomegaEditor final : public juce::AudioProcessorEditor, private juce::ValueTree::Listener
{
public:
    explicit AlfamedomegaEditor(AlfamedomegaProcessor&);
    ~AlfamedomegaEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    ui::MainComponent& mainComponent() noexcept { return *content; }

private:
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void applyScale();

    AlfamedomegaProcessor& processor;
    ui::AlfaLookAndFeel lookAndFeel;
    ui::EditorContext context;
    std::unique_ptr<ui::MainComponent> content;
    bool applyingScale = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AlfamedomegaEditor)
};

} // namespace alfa
