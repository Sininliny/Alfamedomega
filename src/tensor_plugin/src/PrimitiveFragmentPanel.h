#pragma once

#include "alfa/tensor/Kernels.h"
#include "alfa/ui/AlfaLookAndFeel.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <vector>

namespace alfa::tensorhost
{
class PrimitiveFragmentPanel final : public juce::Component
{
public:
    using Publish = std::function<bool(const juce::String&, juce::String&)>;
    PrimitiveFragmentPanel(double sampleRate, juce::String tier, Publish,
                           const alfa::ui::Theme* theme = nullptr);
    ~PrimitiveFragmentPanel() override;
    void paint(juce::Graphics&) override;
    void resized() override;
    // Same explicit action used by the file chooser. No resampling or mixing.
    bool loadWavAsset(const juce::File&, std::size_t sourceChannel, juce::String& error);

private:
    double sampleRate_;
    juce::String tier_;
    Publish publish_;
    alfa::ui::AlfaLookAndFeel look_;
    std::vector<alfa::tensor::kernels::Metadata> catalog_;
    juce::ComboBox operation_, between_;
    juce::TextEditor state_, channels_, period_, phase_, parameters_, initial_, assets_, bindings_, routing_,
        modulation_, outputMix_, budget_, metadata_, wavChannel_;
    juce::Label heading_, status_;
    juce::TextButton publishButton_{"Create fragment"};
    juce::TextButton wavButton_{"Import WAV asset"};
    std::unique_ptr<juce::FileChooser> chooser_;
    juce::String assetSources_ = "[]";
    void selectionChanged();
    void create();
};
} // namespace alfa::tensorhost
