#pragma once

#include "alfa/ui/Theme.h"

#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_basics/juce_gui_basics.h>

namespace alfa::ui
{

/// Draws every control from the current Theme's tokens. Switching theme is a
/// setTheme call followed by a repaint; no component stores colours itself.
class AlfaLookAndFeel : public juce::LookAndFeel_V4
{
public:
    AlfaLookAndFeel();

    void setTheme(const Theme& newTheme);
    const Theme& theme() const noexcept { return current; }

    /// Proportional font for labels, monospaced font for every number so values
    /// don't jitter while they change.
    static juce::Font labelFont(float height, bool bold = false);
    static juce::Font monoFont(float height);

    // Component properties the drawing code reads:
    //   "modDepth" (double, -1..1): modulation range drawn as a blue ring
    //   "dimmed" (bool): control needs a bigger model than the one loaded
    static inline const juce::Identifier modDepthProperty{"modDepth"};
    static inline const juce::Identifier dimmedProperty{"dimmed"};
    //   "look" (String): "tab", "segment" or empty for a raised button
    static inline const juce::Identifier lookProperty{"look"};

    void drawRotarySlider(juce::Graphics&, int x, int y, int width, int height, float sliderPos,
                          float rotaryStartAngle, float rotaryEndAngle, juce::Slider&) override;
    void drawLinearSlider(juce::Graphics&, int x, int y, int width, int height, float sliderPos,
                          float minSliderPos, float maxSliderPos, juce::Slider::SliderStyle,
                          juce::Slider&) override;
    juce::Label* createSliderTextBox(juce::Slider&) override;
    void drawButtonBackground(juce::Graphics&, juce::Button&, const juce::Colour& backgroundColour,
                              bool isMouseOverButton, bool isButtonDown) override;
    void drawButtonText(juce::Graphics&, juce::TextButton&, bool isMouseOverButton,
                        bool isButtonDown) override;
    void drawToggleButton(juce::Graphics&, juce::ToggleButton&, bool isMouseOverButton,
                          bool isButtonDown) override;
    void drawComboBox(juce::Graphics&, int width, int height, bool isButtonDown, int buttonX, int buttonY,
                      int buttonW, int buttonH, juce::ComboBox&) override;
    juce::Font getComboBoxFont(juce::ComboBox&) override;
    juce::Font getTextButtonFont(juce::TextButton&, int buttonHeight) override;
    juce::Font getLabelFont(juce::Label&) override;
    juce::Font getPopupMenuFont() override;
    void drawFocusOutline(juce::Graphics&, juce::Rectangle<float> bounds, float cornerSize) const;

private:
    void applyColours();
    Theme current;
};

} // namespace alfa::ui
