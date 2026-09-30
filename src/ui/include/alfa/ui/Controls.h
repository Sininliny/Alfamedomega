#pragma once

#include "alfa/ui/EditorContext.h"

namespace alfa::ui
{

/// Draws a titled panel surface. Views use it for every grouped area so spacing
/// and borders stay consistent.
void drawPanel(juce::Graphics& g, const Theme& theme, juce::Rectangle<int> area, const juce::String& title,
               const juce::String& subtitle = {});

/// Height taken by the title row drawn by drawPanel.
inline constexpr int kPanelTitleHeight = 26;

/// A parameter control: name, value in real units, and the badges from
/// ui-design.md (tier requirement, lock). Right-click offers the per-parameter
/// menu. Binds through an APVTS attachment so automation, undo and presets
/// share one path.
class ParamControl : public juce::Component,
                     private juce::AudioProcessorValueTreeState::Listener,
                     private juce::ValueTree::Listener
{
public:
    enum class Style
    {
        knob,
        smallKnob,
        verticalBar,
        horizontalBar
    };

    ParamControl(EditorContext& context, const juce::String& paramId, Style style = Style::knob);
    ~ParamControl() override;

    void paint(juce::Graphics&) override;
    void paintOverChildren(juce::Graphics&) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent&) override;
    void mouseEnter(const juce::MouseEvent&) override { repaint(); }
    void mouseExit(const juce::MouseEvent&) override { repaint(); }

    /// Overrides the name shown under the control (macro pages rename macros).
    void setDisplayName(const juce::String& name);

    /// Width of the name column for horizontal bars (default: a third of the
    /// width, at most 110).
    void setLabelWidth(int width);

    juce::Slider& slider() noexcept { return control; }
    const params::ParamSpec& spec() const noexcept { return *paramSpec; }

    bool isDimmed() const;

    /// For a knob parameter: what it means for the current method.
    const params::KnobInfo& knobInfo() const noexcept { return knob; }

private:
    void parameterChanged(const juce::String& parameterID, float newValue) override;
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void applyGesture();
    void showMenu();
    void typeValue();
    juce::String valueText() const;
    juce::Rectangle<int> valueArea() const;
    int labelWidth() const;
    void refreshKnob();
    params::Tier requiredTier() const;

    EditorContext& ctx;
    const params::ParamSpec* paramSpec;
    Style style;
    juce::String displayName;
    bool nameOverridden = false;
    params::KnobInfo knob;
    int fixedLabelWidth = 0;
    juce::Slider control;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ParamControl)
};

/// A row of mutually exclusive text buttons (view switch, pages, detail
/// levels).
class Segmented : public juce::Component
{
public:
    Segmented(juce::StringArray labels, std::function<void(int)> onSelect);
    void setSelected(int index);
    int selected() const noexcept { return current; }
    void resized() override;

private:
    juce::OwnedArray<juce::TextButton> buttons;
    std::function<void(int)> callback;
    int current = 0;
};

/// Short text badge ("PRO+", "M3", "Locked").
void drawBadge(juce::Graphics& g, juce::Rectangle<float> area, const juce::String& text, juce::Colour colour,
               const Theme& theme);

} // namespace alfa::ui
