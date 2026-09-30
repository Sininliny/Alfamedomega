#pragma once
#include "TensorProcessor.h"
namespace alfa::tensorhost
{
class Editor final : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit Editor(Processor&);
    void resized() override;

private:
    Processor& processor_;
    juce::MidiKeyboardComponent keyboard_;
    juce::Label title_, status_, hint_;
    juce::ComboBox mode_, layer_, references_;
    juce::TextButton importModel_{"Import runtime compiler model"};
    juce::TextButton loadReference_{"Load explicit demonstration"};
    juce::TextEditor document_, inspection_;
    juce::TextButton apply_{"Compile tensor JSON"}, open_{"Open configuration"}, save_{"Save configuration"};
    std::array<juce::Slider, 5> sliders_;
    std::array<juce::Label, 5> labels_;
    std::unique_ptr<juce::FileChooser> chooser_;
    void timerCallback() override;
};
} // namespace alfa::tensorhost
