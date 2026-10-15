#include "alfa/ui/AlfaLookAndFeel.h"

namespace alfa::ui
{

AlfaLookAndFeel::AlfaLookAndFeel()
{
    setTheme(Theme::graphite());
}

void AlfaLookAndFeel::setTheme(const Theme& newTheme)
{
    current = newTheme;
    applyColours();
}

juce::Font AlfaLookAndFeel::labelFont(float height, bool bold)
{
    return juce::Font(juce::FontOptions(height, bold ? juce::Font::bold : juce::Font::plain));
}

juce::Font AlfaLookAndFeel::monoFont(float height)
{
    return juce::Font(
        juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), height, juce::Font::plain));
}

void AlfaLookAndFeel::applyColours()
{
    const auto& t = current;
    setColour(juce::ResizableWindow::backgroundColourId, t.bg);
    setColour(juce::DocumentWindow::backgroundColourId, t.bg);
    setColour(juce::Label::textColourId, t.text);
    setColour(juce::Slider::textBoxTextColourId, t.text2);
    setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    setColour(juce::Slider::textBoxHighlightColourId, t.accent.withAlpha(0.35f));
    setColour(juce::Slider::thumbColourId, t.accent);
    setColour(juce::Slider::trackColourId, t.accent);
    setColour(juce::Slider::backgroundColourId, t.line);
    setColour(juce::Slider::rotarySliderFillColourId, t.accent);
    setColour(juce::Slider::rotarySliderOutlineColourId, t.line);
    setColour(juce::TextButton::buttonColourId, t.panel);
    setColour(juce::TextButton::buttonOnColourId, t.accent);
    setColour(juce::TextButton::textColourOffId, t.text2);
    setColour(juce::TextButton::textColourOnId, t.bg);
    setColour(juce::ToggleButton::textColourId, t.text);
    setColour(juce::ToggleButton::tickColourId, t.accent);
    setColour(juce::ToggleButton::tickDisabledColourId, t.line);
    setColour(juce::ComboBox::backgroundColourId, t.bg);
    setColour(juce::ComboBox::textColourId, t.text);
    setColour(juce::ComboBox::outlineColourId, t.line);
    setColour(juce::ComboBox::arrowColourId, t.text2);
    setColour(juce::ComboBox::focusedOutlineColourId, t.accent);
    setColour(juce::PopupMenu::backgroundColourId, t.panel);
    setColour(juce::PopupMenu::textColourId, t.text);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, t.accent.withAlpha(0.25f));
    setColour(juce::PopupMenu::highlightedTextColourId, t.text);
    setColour(juce::TextEditor::backgroundColourId, t.bg.interpolatedWith(t.panel, 0.5f));
    setColour(juce::TextEditor::textColourId, t.text);
    setColour(juce::TextEditor::outlineColourId, t.line.withAlpha(0.6f));
    setColour(juce::TextEditor::focusedOutlineColourId, t.accent);
    setColour(juce::TextEditor::highlightColourId, t.accent.withAlpha(0.35f));
    setColour(juce::CaretComponent::caretColourId, t.accent);
    // Keys keep their real-world polarity in every theme: light white keys, dark
    // black keys.
    const bool lightTheme = t.panel.getPerceivedBrightness() > 0.5f;
    setColour(juce::MidiKeyboardComponent::whiteNoteColourId,
              lightTheme ? juce::Colours::white : t.text.interpolatedWith(t.panel, 0.08f));
    setColour(juce::MidiKeyboardComponent::blackNoteColourId, lightTheme ? t.text : t.bg);
    setColour(juce::MidiKeyboardComponent::keySeparatorLineColourId, t.line);
    setColour(juce::MidiKeyboardComponent::keyDownOverlayColourId, t.accent.withAlpha(0.85f));
    setColour(juce::MidiKeyboardComponent::mouseOverKeyOverlayColourId, t.accent.withAlpha(0.3f));
    setColour(juce::MidiKeyboardComponent::shadowColourId, juce::Colours::transparentBlack);
    setColour(juce::MidiKeyboardComponent::upDownButtonBackgroundColourId, t.panel);
    setColour(juce::MidiKeyboardComponent::upDownButtonArrowColourId, t.text2);
    setColour(juce::ScrollBar::thumbColourId, t.line);
    setColour(juce::TooltipWindow::backgroundColourId, t.panel);
    setColour(juce::TooltipWindow::textColourId, t.text);
    setColour(juce::TooltipWindow::outlineColourId, t.line);
    setColour(juce::ListBox::backgroundColourId, t.panel);
    setColour(juce::AlertWindow::backgroundColourId, t.panel);
    setColour(juce::AlertWindow::textColourId, t.text);
    setColour(juce::AlertWindow::outlineColourId, t.line);
}

void AlfaLookAndFeel::drawFocusOutline(juce::Graphics& g, juce::Rectangle<float> bounds,
                                       float cornerSize) const
{
    g.setColour(current.accent);
    g.drawRoundedRectangle(bounds.reduced(0.5f), cornerSize, 1.5f);
}

void AlfaLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                       float sliderPos, float startAngle, float endAngle,
                                       juce::Slider& slider)
{
    // A raised cap sitting in a recessed well, with a thin value arc around it. Bipolar ranges grow from
    // the centre: amber above it, blue below, as every signed value in the product.
    const auto& t = current;
    const bool dimmed = slider.getProperties()[dimmedProperty];
    const bool light = t.bg.getPerceivedBrightness() > 0.5f;
    const auto area = juce::Rectangle<int>(x, y, width, height).toFloat().reduced(2.0f);
    const auto diameter = juce::jmin(area.getWidth(), area.getHeight(), 64.0f);
    const auto centre = area.getCentre();
    const auto radius = diameter * 0.5f;
    const auto trackWidth = juce::jlimit(2.0f, 3.2f, radius * 0.11f);
    const auto arcRadius = radius - trackWidth * 0.5f - 0.5f;

    auto arc = [&](float from, float to, float r)
    {
        juce::Path p;
        p.addCentredArc(centre.x, centre.y, r, r, 0.0f, from, to, true);
        return p;
    };
    const auto rounded = [](float w)
    {
        return juce::PathStrokeType(w, juce::PathStrokeType::curved, juce::PathStrokeType::rounded);
    };

    const auto track = light ? t.line : t.panel.brighter(0.12f);
    g.setColour(track);
    g.strokePath(arc(startAngle, endAngle, arcRadius), rounded(trackWidth));

    const auto angle = startAngle + sliderPos * (endAngle - startAngle);
    const bool bipolar = slider.getMinimum() < 0.0 && slider.getMaximum() > 0.0;
    const auto origin = bipolar ? startAngle + static_cast<float>(slider.valueToProportionOfLength(0.0)) *
                                                   (endAngle - startAngle)
                                : startAngle;
    if (bipolar)
    {
        g.setColour(t.text2.withAlpha(0.6f));
        g.fillEllipse(juce::Rectangle<float>(2.6f, 2.6f).withCentre(centre.getPointOnCircumference(arcRadius, origin)));
    }
    const auto valueColour = dimmed ? t.text2.withAlpha(0.5f) : (angle < origin ? t.mod : t.accent);
    if (std::abs(angle - origin) > 0.01f)
    {
        g.setColour(valueColour);
        g.strokePath(arc(juce::jmin(origin, angle), juce::jmax(origin, angle), arcRadius), rounded(trackWidth));
    }

    // Modulation ring just outside the value arc, covering the range the source moves it through.
    const double modDepth = slider.getProperties()[modDepthProperty];
    if (std::abs(modDepth) > 0.001)
    {
        const auto span = static_cast<float>(modDepth) * (endAngle - startAngle);
        const auto modEnd = juce::jlimit(startAngle, endAngle, angle + span);
        g.setColour(t.mod.withAlpha(0.8f));
        g.strokePath(arc(juce::jmin(angle, modEnd), juce::jmax(angle, modEnd), arcRadius + trackWidth + 1.0f),
                     rounded(1.5f));
    }

    // Well, then the cap with a soft shadow and a hairline of light on its upper rim.
    const auto wellRadius = arcRadius - trackWidth - 2.5f;
    const auto well = juce::Rectangle<float>(wellRadius * 2, wellRadius * 2).withCentre(centre);
    g.setColour(light ? t.bg.darker(0.03f) : t.bg.darker(0.12f));
    g.fillEllipse(well);
    const auto capRadius = wellRadius - 3.0f;
    const auto cap = juce::Rectangle<float>(capRadius * 2, capRadius * 2).withCentre(centre);
    g.setColour(juce::Colours::black.withAlpha(light ? 0.16f : 0.45f));
    g.fillEllipse(cap.translated(0, 1.6f).expanded(0.6f));
    const auto top = light ? juce::Colours::white : t.panel.brighter(0.55f);
    const auto bottom = light ? t.panel.darker(0.03f) : t.panel.brighter(0.2f);
    g.setGradientFill(juce::ColourGradient(top, centre.x, cap.getY(), bottom, centre.x, cap.getBottom(), false));
    g.fillEllipse(cap);
    g.setColour(juce::Colours::white.withAlpha(light ? 1.0f : 0.09f));
    g.strokePath(arc(juce::MathConstants<float>::pi * 1.62f, juce::MathConstants<float>::pi * 2.38f, capRadius - 0.6f),
                 juce::PathStrokeType(1.0f));

    const auto pointerStart = centre.getPointOnCircumference(capRadius * 0.3f, angle);
    const auto pointerEnd = centre.getPointOnCircumference(capRadius * 0.78f, angle);
    g.setColour(dimmed ? t.text2 : (slider.isMouseOverOrDragging() ? valueColour : t.text));
    g.drawLine({pointerStart, pointerEnd}, 2.0f);

    if (slider.hasKeyboardFocus(true))
        drawFocusOutline(g, juce::Rectangle<float>(diameter, diameter).withCentre(centre).expanded(1.0f), diameter * 0.5f);
}

void AlfaLookAndFeel::drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                                       float sliderPos, float, float, juce::Slider::SliderStyle style,
                                       juce::Slider& slider)
{
    const auto& t = current;
    const bool dimmed = slider.getProperties()[dimmedProperty];
    const bool light = t.bg.getPerceivedBrightness() > 0.5f;
    const auto bounds = juce::Rectangle<int>(x, y, width, height).toFloat();
    const bool vertical = style == juce::Slider::LinearVertical || style == juce::Slider::LinearBarVertical;
    const auto thickness = 4.0f;

    // A recessed groove, a neutral fill and a small raised cap.
    auto track = vertical ? bounds.withSizeKeepingCentre(thickness, bounds.getHeight())
                          : bounds.withSizeKeepingCentre(bounds.getWidth(), thickness);
    g.setColour(light ? t.bg.darker(0.04f) : t.bg.darker(0.15f));
    g.fillRoundedRectangle(track, thickness * 0.5f);
    g.setColour(juce::Colours::black.withAlpha(light ? 0.08f : 0.3f));
    g.drawHorizontalLine(juce::roundToInt(track.getY()), track.getX() + 2, track.getRight() - 2);

    const bool bipolar = slider.getMinimum() < 0.0 && slider.getMaximum() > 0.0;
    const auto zero = bipolar ? static_cast<float>(slider.getPositionOfValue(0.0))
                              : (vertical ? track.getBottom() : track.getX());
    const auto from = juce::jmin(zero, sliderPos), to = juce::jmax(zero, sliderPos);
    auto filled = vertical ? track.withTop(from).withBottom(to) : track.withLeft(from).withRight(to);
    const auto fill = dimmed ? t.text2.withAlpha(0.4f)
                      : bipolar ? (sliderPos < zero ? t.mod : t.accent).withAlpha(0.85f)
                                : t.text2.withAlpha(0.8f);
    g.setColour(fill);
    g.fillRoundedRectangle(filled, thickness * 0.5f);

    const auto thumbSize = 12.0f;
    const auto thumb = juce::Rectangle<float>(thumbSize, thumbSize)
                           .withCentre(vertical ? juce::Point<float>{track.getCentreX(), sliderPos}
                                                : juce::Point<float>{sliderPos, track.getCentreY()});
    g.setColour(juce::Colours::black.withAlpha(light ? 0.18f : 0.5f));
    g.fillEllipse(thumb.translated(0.0f, 1.2f).expanded(0.5f));
    g.setGradientFill(juce::ColourGradient(light ? juce::Colours::white : t.panel.brighter(0.6f),
                                           thumb.getCentreX(), thumb.getY(),
                                           light ? t.panel.darker(0.03f) : t.panel.brighter(0.25f),
                                           thumb.getCentreX(), thumb.getBottom(), false));
    g.fillEllipse(thumb);
    if (slider.isMouseOverOrDragging())
    {
        g.setColour(t.accent.withAlpha(0.8f));
        g.drawEllipse(thumb.reduced(0.5f), 1.2f);
    }

    if (slider.hasKeyboardFocus(true))
        drawFocusOutline(g, bounds, 4.0f);
}

juce::Label* AlfaLookAndFeel::createSliderTextBox(juce::Slider& slider)
{
    // Values read as quiet text under the control, not as boxed fields; double-click to type one.
    auto* label = LookAndFeel_V4::createSliderTextBox(slider);
    label->setFont(labelFont(11.0f));
    label->setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    label->setColour(juce::Label::outlineColourId, juce::Colours::transparentBlack);
    label->setColour(juce::Label::textColourId, current.text2);
    label->setColour(juce::Label::outlineWhenEditingColourId, current.accent);
    label->setColour(juce::TextEditor::backgroundColourId, current.bg);
    return label;
}

void AlfaLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button, const juce::Colour&,
                                           bool isMouseOverButton, bool isButtonDown)
{
    // Three button looks, set with the "look" property: "tab" (text with an underline when on),
    // "segment" (one choice inside a recessed group the parent draws) and the default raised button.
    const auto& t = current;
    const bool light = t.bg.getPerceivedBrightness() > 0.5f;
    const auto look = button.getProperties()[lookProperty].toString();
    const auto bounds = button.getLocalBounds().toFloat();
    const bool on = button.getToggleState();
    if (look == "tab")
    {
        if (on)
        {
            g.setColour(t.text);
            g.fillRoundedRectangle(bounds.withTop(bounds.getBottom() - 2).reduced(10, 0), 1);
        }
        else if (isMouseOverButton)
        {
            g.setColour(t.text2.withAlpha(0.35f));
            g.fillRoundedRectangle(bounds.withTop(bounds.getBottom() - 2).reduced(10, 0), 1);
        }
    }
    else if (look == "segment")
    {
        if (on)
        {
            const auto r = bounds.reduced(0.5f, 0.5f);
            g.setColour(juce::Colours::black.withAlpha(light ? 0.12f : 0.45f));
            g.fillRoundedRectangle(r.translated(0, 1), 5);
            g.setColour(light ? juce::Colours::white : t.panel.brighter(0.14f));
            g.fillRoundedRectangle(r, 5);
            g.setColour(juce::Colours::white.withAlpha(light ? 1.0f : 0.06f));
            g.drawHorizontalLine(juce::roundToInt(r.getY()), r.getX() + 4, r.getRight() - 4);
        }
        else if (isMouseOverButton)
        {
            g.setColour(t.text2.withAlpha(0.1f));
            g.fillRoundedRectangle(bounds.reduced(0.5f), 5);
        }
    }
    else
    {
        auto fill = on ? (light ? juce::Colours::white : t.panel.brighter(0.16f))
                       : (light ? t.panel.brighter(0.4f) : t.panel.brighter(0.07f));
        if (isButtonDown)
            fill = fill.darker(0.08f);
        else if (isMouseOverButton)
            fill = fill.brighter(light ? 0.0f : 0.08f).darker(light ? 0.02f : 0.0f);
        const auto r = bounds.reduced(0.5f, 1.0f);
        g.setColour(juce::Colours::black.withAlpha(light ? 0.1f : 0.45f));
        g.fillRoundedRectangle(r.translated(0, 1), 7);
        g.setColour(fill);
        g.fillRoundedRectangle(r, 7);
        g.setColour(juce::Colours::white.withAlpha(light ? 1.0f : 0.06f));
        g.drawHorizontalLine(juce::roundToInt(r.getY()), r.getX() + 5, r.getRight() - 5);
        if (on)
        {
            g.setColour(t.accent);
            g.fillRoundedRectangle(r.withTop(r.getBottom() - 2).reduced(8, 0), 1);
        }
    }
    if (button.hasKeyboardFocus(false))
        drawFocusOutline(g, bounds.reduced(0.5f), 7.0f);
}

void AlfaLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button, bool isMouseOverButton, bool)
{
    const auto& t = current;
    const auto look = button.getProperties()[lookProperty].toString();
    const bool on = button.getToggleState();
    const bool prominent = look == "tab" || look == "segment" ? on : true;
    g.setFont(labelFont(look == "tab" ? 12.5f : 12.0f, on && look != ""));
    auto colour = prominent || isMouseOverButton ? t.text : t.text2;
    if (look.isEmpty() && !on && button.getProperties()["quiet"])
        colour = isMouseOverButton ? t.text : t.text2;
    if (!button.isEnabled())
        colour = t.text2.withAlpha(0.45f);
    g.setColour(colour);
    g.drawFittedText(button.getButtonText(), button.getLocalBounds().reduced(6, 2),
                     juce::Justification::centred, 1);
}

void AlfaLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& button, bool isMouseOverButton,
                                       bool)
{
    const auto& t = current;
    const bool light = t.bg.getPerceivedBrightness() > 0.5f;
    auto bounds = button.getLocalBounds().toFloat();
    const auto switchArea = bounds.removeFromLeft(30.0f).withSizeKeepingCentre(26.0f, 14.0f);
    const bool on = button.getToggleState();

    g.setColour(on ? t.accent : (light ? t.bg.darker(0.05f) : t.bg.darker(0.15f)));
    g.fillRoundedRectangle(switchArea, 7.0f);
    if (!on)
    {
        g.setColour(juce::Colours::black.withAlpha(light ? 0.08f : 0.3f));
        g.drawHorizontalLine(juce::roundToInt(switchArea.getY()), switchArea.getX() + 4, switchArea.getRight() - 4);
    }
    const auto knob = juce::Rectangle<float>(10.0f, 10.0f)
                          .withCentre({on ? switchArea.getRight() - 7.0f : switchArea.getX() + 7.0f,
                                       switchArea.getCentreY()});
    g.setColour(juce::Colours::black.withAlpha(0.25f));
    g.fillEllipse(knob.translated(0.0f, 1.0f));
    g.setColour(on ? juce::Colours::white : (isMouseOverButton ? t.text : (light ? juce::Colours::white : t.text2)));
    g.fillEllipse(knob);

    g.setColour(button.isEnabled() ? (on || isMouseOverButton ? t.text : t.text2) : t.text2.withAlpha(0.5f));
    g.setFont(labelFont(12.0f));
    g.drawFittedText(button.getButtonText(), bounds.withTrimmedLeft(6.0f).toNearestInt(),
                     juce::Justification::centredLeft, 1);

    if (button.hasKeyboardFocus(false))
        drawFocusOutline(g, switchArea.expanded(2.0f), 9.0f);
}

void AlfaLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool, int, int, int, int,
                                   juce::ComboBox& box)
{
    const auto& t = current;
    const bool light = t.bg.getPerceivedBrightness() > 0.5f;
    const auto bounds =
        juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)).reduced(0.5f);
    g.setColour(light ? t.bg.interpolatedWith(t.panel, 0.3f) : t.bg.darker(0.12f));
    g.fillRoundedRectangle(bounds, 7.0f);
    g.setColour(juce::Colours::black.withAlpha(light ? 0.08f : 0.35f));
    g.drawHorizontalLine(juce::roundToInt(bounds.getY()) + 1, bounds.getX() + 5, bounds.getRight() - 5);
    if (box.hasKeyboardFocus(true))
    {
        g.setColour(t.accent);
        g.drawRoundedRectangle(bounds, 7.0f, 1.0f);
    }

    juce::Path arrow;
    const auto cx = bounds.getRight() - 12.0f;
    const auto cy = bounds.getCentreY();
    arrow.startNewSubPath(cx - 3.5f, cy - 1.5f);
    arrow.lineTo(cx, cy + 2.0f);
    arrow.lineTo(cx + 3.5f, cy - 1.5f);
    g.setColour(t.text2);
    g.strokePath(arrow, juce::PathStrokeType(1.4f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
}

juce::Font AlfaLookAndFeel::getComboBoxFont(juce::ComboBox& box)
{
    return labelFont(juce::jmin(13.0f, static_cast<float>(box.getHeight()) * 0.48f));
}

juce::Font AlfaLookAndFeel::getTextButtonFont(juce::TextButton&, int buttonHeight)
{
    return labelFont(juce::jmin(12.5f, static_cast<float>(buttonHeight) * 0.46f));
}

juce::Font AlfaLookAndFeel::getLabelFont(juce::Label& label)
{
    return label.getFont();
}

juce::Font AlfaLookAndFeel::getPopupMenuFont()
{
    return labelFont(14.0f);
}

} // namespace alfa::ui
