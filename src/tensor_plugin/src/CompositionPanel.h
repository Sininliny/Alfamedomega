#pragma once
#include "TensorComposition.h"
#include "alfa/ui/AlfaLookAndFeel.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <memory>

namespace alfa::tensorhost
{
class CompositionPanel final : public juce::Component
{
public:
    using Publish = std::function<bool(const juce::String&, juce::String&)>;
    CompositionPanel(const juce::String& sound, double sampleRate, Publish,
                     const alfa::ui::Theme* theme = nullptr);
    ~CompositionPanel() override;
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    double sampleRate_;
    Publish publish_;
    juce::String currentSound_;
    bool needsBase_ = false;
    alfa::ui::AlfaLookAndFeel look_;
    std::unique_ptr<TensorComposition> composer_;
    juce::ComboBox fragment_, source_, destination_, sourceKind_, targetKind_, weights_;
    juce::TextEditor newId_, bindingId_, sourceElement_, instance_, targetElement_, weight_, budget_, recipe_;
    juce::Label help_, status_;
    juce::TextButton import_{"Insert fragment"}, clone_{"Clone"}, remove_{"Remove"}, bind_{"Add binding"},
        unbind_{"Remove binding"}, prepare_{"Prepare recipe"}, apply_{"Apply to sound"},
        base_{"Use edited sound as base"};
    juce::TextButton create_{"Create primitive"};
    juce::TextButton policy_{"Use current input policy"};
    juce::Component::SafePointer<juce::DialogWindow> primitiveDialog_;
    std::unique_ptr<juce::FileChooser> chooser_;
    void refresh();
    void report(const juce::String&, bool error = false);
    bool prepareRecipe();
    void importFragment();
    void createFragment();
    void addBinding();
};
} // namespace alfa::tensorhost
