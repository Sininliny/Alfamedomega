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
    setColour(juce::ComboBox::backgroundColourId, t.panel);
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
    const auto& t = current;
    const bool dimmed = slider.getProperties()[dimmedProperty];
    const bool light = t.bg.getPerceivedBrightness() > 0.5f;
    const auto area = juce::Rectangle<int>(x, y, width, height).toFloat().reduced(3.0f);
    const auto diameter = juce::jmin(area.getWidth(), area.getHeight());
    const auto centre = area.getCentre();
    const auto radius = diameter * 0.5f;
    const auto trackWidth = juce::jlimit(2.0f, 4.0f, radius * 0.09f);
    const auto arcRadius = radius - trackWidth - 1.5f;

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

    // Track
    g.setColour(t.line);
    g.strokePath(arc(startAngle, endAngle, arcRadius), rounded(trackWidth));

    // Value arc with a soft glow; bipolar ranges grow from the centre, others from the start.
    const auto angle = startAngle + sliderPos * (endAngle - startAngle);
    const bool bipolar = slider.getMinimum() < 0.0 && slider.getMaximum() > 0.0;
    const auto origin = bipolar ? (startAngle + endAngle) * 0.5f : startAngle;
    const auto valueColour = dimmed ? t.text2.withAlpha(0.5f) : t.accent;
    const auto valueArc = arc(juce::jmin(origin, angle), juce::jmax(origin, angle), arcRadius);
    if (!dimmed)
    {
        g.setColour(valueColour.withAlpha(0.14f));
        g.strokePath(valueArc, rounded(trackWidth * 3.2f));
    }
    g.setColour(valueColour);
    g.strokePath(valueArc, rounded(trackWidth));

    // Modulation ring outside the value arc, covering the range the source moves it through.
    const double modDepth = slider.getProperties()[modDepthProperty];
    if (std::abs(modDepth) > 0.001)
    {
        const auto span = static_cast<float>(modDepth) * (endAngle - startAngle);
        const auto modEnd = juce::jlimit(startAngle, endAngle, angle + span);
        g.setColour(t.mod);
        g.strokePath(arc(juce::jmin(angle, modEnd), juce::jmax(angle, modEnd), radius - trackWidth * 0.4f),
                     rounded(trackWidth * 0.7f));
    }

    // Body: a soft shadow, then a lit dome.
    const auto bodyRadius = arcRadius - trackWidth * 2.2f;
    const auto body = juce::Rectangle<float>(bodyRadius * 2.0f, bodyRadius * 2.0f).withCentre(centre);
    g.setColour(juce::Colours::black.withAlpha(light ? 0.10f : 0.32f));
    g.fillEllipse(body.translated(0.0f, bodyRadius * 0.12f).expanded(1.5f));
    const auto top = light ? t.panel.brighter(0.4f) : t.panel.brighter(0.22f);
    const auto bottom = light ? t.panel.darker(0.06f) : t.panel.darker(0.12f);
    g.setGradientFill(
        juce::ColourGradient(top, centre.x, body.getY(), bottom, centre.x, body.getBottom(), false));
    g.fillEllipse(body);
    g.setColour(juce::Colours::white.withAlpha(light ? 0.6f : 0.07f));
    g.drawEllipse(body.reduced(0.75f), 1.0f);

    // Pointer: a short line near the rim.
    const auto pointerStart = centre.getPointOnCircumference(bodyRadius * 0.45f, angle);
    const auto pointerEnd = centre.getPointOnCircumference(bodyRadius * 0.82f, angle);
    g.setColour(dimmed ? t.text2 : (slider.isMouseOverOrDragging() ? t.accent : t.text));
    g.drawLine({pointerStart, pointerEnd}, juce::jmax(1.8f, trackWidth * 0.75f));

    if (slider.hasKeyboardFocus(true))
        drawFocusOutline(g, area.withSizeKeepingCentre(diameter, diameter).expanded(1.0f), diameter * 0.5f);
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

    auto track = vertical ? bounds.withSizeKeepingCentre(thickness, bounds.getHeight())
                          : bounds.withSizeKeepingCentre(bounds.getWidth(), thickness);
    g.setColour(t.line);
    g.fillRoundedRectangle(track, thickness * 0.5f);

    // Fill from zero when the range spans it, else from the minimum.
    const bool bipolar = slider.getMinimum() < 0.0 && slider.getMaximum() > 0.0;
    const auto zero = bipolar ? static_cast<float>(slider.getPositionOfValue(0.0))
                              : (vertical ? track.getBottom() : track.getX());
    const auto from = juce::jmin(zero, sliderPos), to = juce::jmax(zero, sliderPos);
    auto filled = vertical ? track.withTop(from).withBottom(to) : track.withLeft(from).withRight(to);
    const auto fill = dimmed ? t.text2.withAlpha(0.5f) : t.accent;
    g.setColour(fill.withAlpha(0.18f));
    g.fillRoundedRectangle(filled.expanded(vertical ? 3.0f : 0.0f, vertical ? 0.0f : 3.0f), 4.0f);
    g.setColour(fill);
    g.fillRoundedRectangle(filled, thickness * 0.5f);

    // Thumb: a lit disc with an accent rim.
    const auto thumbSize = 14.0f;
    const auto thumb = juce::Rectangle<float>(thumbSize, thumbSize)
                           .withCentre(vertical ? juce::Point<float>{track.getCentreX(), sliderPos}
                                                : juce::Point<float>{sliderPos, track.getCentreY()});
    g.setColour(juce::Colours::black.withAlpha(light ? 0.12f : 0.35f));
    g.fillEllipse(thumb.translated(0.0f, 1.5f));
    g.setGradientFill(juce::ColourGradient(light ? juce::Colours::white : t.panel.brighter(0.35f),
                                           thumb.getCentreX(), thumb.getY(), t.panel.brighter(0.05f),
                                           thumb.getCentreX(), thumb.getBottom(), false));
    g.fillEllipse(thumb);
    g.setColour(slider.isMouseOverOrDragging() ? fill : fill.withAlpha(0.75f));
    g.drawEllipse(thumb.reduced(1.0f), 1.6f);

    if (slider.hasKeyboardFocus(true))
        drawFocusOutline(g, bounds, 4.0f);
}

juce::Label* AlfaLookAndFeel::createSliderTextBox(juce::Slider& slider)
{
    // Values read as quiet text under the control, not as boxed fields; double-click to type one.
    auto* label = LookAndFeel_V4::createSliderTextBox(slider);
    label->setFont(monoFont(12.0f));
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
    const auto& t = current;
    const auto bounds = button.getLocalBounds().toFloat().reduced(0.5f);
    const bool on = button.getToggleState();
    const auto radius = 7.0f;

    auto fill = on ? t.accent : t.panel.brighter(0.03f);
    if (isButtonDown)
        fill = fill.darker(0.12f);
    else if (isMouseOverButton)
        fill = fill.brighter(on ? 0.08f : 0.06f);

    g.setGradientFill(juce::ColourGradient(fill.brighter(0.05f), 0.0f, bounds.getY(), fill.darker(0.05f),
                                           0.0f, bounds.getBottom(), false));
    g.fillRoundedRectangle(bounds, radius);
    g.setColour(on ? t.accent.darker(0.1f) : t.line);
    g.drawRoundedRectangle(bounds, radius, 1.0f);

    if (button.hasKeyboardFocus(false))
        drawFocusOutline(g, bounds, radius);
}

void AlfaLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button, bool, bool)
{
    const auto& t = current;
    g.setFont(getTextButtonFont(button, button.getHeight()));
    auto colour = button.getToggleState() ? t.bg : t.text;
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
    auto bounds = button.getLocalBounds().toFloat();
    const auto switchArea = bounds.removeFromLeft(34.0f).withSizeKeepingCentre(30.0f, 16.0f);
    const bool on = button.getToggleState();

    g.setColour(on ? t.accent : t.line);
    g.fillRoundedRectangle(switchArea, 8.0f);
    const auto knob = juce::Rectangle<float>(12.0f, 12.0f)
                          .withCentre({on ? switchArea.getRight() - 8.0f : switchArea.getX() + 8.0f,
                                       switchArea.getCentreY()});
    g.setColour(juce::Colours::black.withAlpha(0.2f));
    g.fillEllipse(knob.translated(0.0f, 1.0f));
    g.setColour(on ? t.bg : (isMouseOverButton ? t.text : t.text2));
    g.fillEllipse(knob);

    g.setColour(button.isEnabled() ? t.text : t.text2.withAlpha(0.5f));
    g.setFont(labelFont(13.0f));
    g.drawFittedText(button.getButtonText(), bounds.withTrimmedLeft(8.0f).toNearestInt(),
                     juce::Justification::centredLeft, 1);

    if (button.hasKeyboardFocus(false))
        drawFocusOutline(g, switchArea.expanded(2.0f), 9.0f);
}

void AlfaLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool, int, int, int, int,
                                   juce::ComboBox& box)
{
    const auto& t = current;
    const auto bounds =
        juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height))
            .reduced(0.5f);
    g.setGradientFill(juce::ColourGradient(t.panel.brighter(0.05f), 0.0f, bounds.getY(),
                                           t.panel.darker(0.03f), 0.0f, bounds.getBottom(), false));
    g.fillRoundedRectangle(bounds, 7.0f);
    g.setColour(box.hasKeyboardFocus(true) ? t.accent : t.line);
    g.drawRoundedRectangle(bounds, 7.0f, 1.0f);

    juce::Path arrow;
    const auto cx = bounds.getRight() - 12.0f;
    const auto cy = bounds.getCentreY();
    arrow.startNewSubPath(cx - 4.0f, cy - 2.0f);
    arrow.lineTo(cx, cy + 2.0f);
    arrow.lineTo(cx + 4.0f, cy - 2.0f);
    g.setColour(t.text2);
    g.strokePath(arrow, juce::PathStrokeType(1.5f));
}

juce::Font AlfaLookAndFeel::getComboBoxFont(juce::ComboBox& box)
{
    return labelFont(juce::jmin(14.0f, static_cast<float>(box.getHeight()) * 0.5f));
}

juce::Font AlfaLookAndFeel::getTextButtonFont(juce::TextButton&, int buttonHeight)
{
    return labelFont(juce::jmin(14.0f, static_cast<float>(buttonHeight) * 0.5f));
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
