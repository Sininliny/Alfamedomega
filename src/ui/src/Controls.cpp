#include "alfa/ui/Controls.h"

namespace alfa::ui
{

void drawPanel(juce::Graphics& g, const Theme& theme, juce::Rectangle<int> area, const juce::String& title,
               const juce::String& subtitle)
{
    const auto bounds = area.toFloat().reduced(0.5f);
    g.setColour(theme.panel);
    g.fillRoundedRectangle(bounds, 8.0f);
    g.setColour(theme.line);
    g.drawRoundedRectangle(bounds, 8.0f, 1.0f);

    if (title.isEmpty())
        return;

    auto titleRow = area.reduced(12, 0).withHeight(kPanelTitleHeight).withTrimmedTop(6);
    g.setColour(theme.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.5f, true));
    const auto upper = title.toUpperCase();
    g.drawText(upper, titleRow, juce::Justification::centredLeft, false);

    if (subtitle.isNotEmpty())
    {
        g.setFont(AlfaLookAndFeel::labelFont(11.5f));
        g.drawText(subtitle, titleRow, juce::Justification::centredRight, true);
    }
}

void drawBadge(juce::Graphics& g, juce::Rectangle<float> area, const juce::String& text, juce::Colour colour,
               const Theme& theme)
{
    g.setColour(colour.withAlpha(0.18f));
    g.fillRoundedRectangle(area, 3.0f);
    g.setColour(colour);
    g.drawRoundedRectangle(area.reduced(0.5f), 3.0f, 1.0f);
    g.setFont(AlfaLookAndFeel::labelFont(10.0f, true));
    g.drawText(text, area, juce::Justification::centred, false);
    juce::ignoreUnused(theme);
}

// ParamControl
// -----------------------------------------------------------------

ParamControl::ParamControl(EditorContext& context, const juce::String& paramId, Style s)
    : ctx(context)
    , paramSpec(params::findSpec(paramId))
    , style(s)
{
    jassert(paramSpec != nullptr);
    displayName = paramSpec->shortName;

    switch (style)
    {
    case Style::knob:
    case Style::smallKnob:
        control.setSliderStyle(juce::Slider::RotaryVerticalDrag);
        break;
    case Style::verticalBar:
        control.setSliderStyle(juce::Slider::LinearVertical);
        break;
    case Style::horizontalBar:
        control.setSliderStyle(juce::Slider::LinearHorizontal);
        break;
    }

    control.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    control.setDoubleClickReturnValue(true, static_cast<double>(paramSpec->defaultValue));
    control.setScrollWheelEnabled(true);
    control.setWantsKeyboardFocus(true);
    control.setTitle(paramSpec->name);
    control.addMouseListener(this, false);
    control.onValueChange = [this]
    {
        repaint();
    };
    addAndMakeVisible(control);

    attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        ctx.params, paramSpec->id, control);
    // The attachment sets the default double-click value in normalised terms;
    // restore real units.
    control.setDoubleClickReturnValue(true, static_cast<double>(paramSpec->defaultValue));

    ctx.params.addParameterListener(params::id::modelTier, this);
    ctx.ui.tree().addListener(this);
    applyGesture();
    control.getProperties().set(AlfaLookAndFeel::dimmedProperty, isDimmed());
    control.setTooltip(paramSpec->name);
}

ParamControl::~ParamControl()
{
    ctx.params.removeParameterListener(params::id::modelTier, this);
    ctx.ui.tree().removeListener(this);
}

void ParamControl::setDisplayName(const juce::String& name)
{
    displayName = name;
    resized();
    repaint();
}

int ParamControl::labelWidth() const
{
    return fixedLabelWidth > 0 ? fixedLabelWidth : juce::jmin(110, getWidth() / 3);
}

void ParamControl::setLabelWidth(int width)
{
    fixedLabelWidth = width;
    resized();
    repaint();
}

bool ParamControl::isDimmed() const
{
    return static_cast<int>(ctx.selectedTier()) < static_cast<int>(paramSpec->minimumTier);
}

void ParamControl::parameterChanged(const juce::String&, float)
{
    juce::MessageManager::callAsync(
        [safe = juce::Component::SafePointer<ParamControl>(this)]
        {
            if (safe != nullptr)
            {
                safe->control.getProperties().set(AlfaLookAndFeel::dimmedProperty, safe->isDimmed());
                safe->repaint();
                safe->control.repaint();
            }
        });
}

void ParamControl::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    if (id == UiState::Ids::knobGesture)
        applyGesture();
    if (id == UiState::Ids::valueDisplay || id == UiState::Ids::locks)
        repaint();
}

void ParamControl::applyGesture()
{
    if (style != Style::knob && style != Style::smallKnob)
        return;

    const auto gesture = ctx.ui.getString(UiState::Ids::knobGesture);
    if (gesture == "horizontal")
        control.setSliderStyle(juce::Slider::RotaryHorizontalDrag);
    else if (gesture == "circular")
        control.setSliderStyle(juce::Slider::Rotary);
    else
        control.setSliderStyle(juce::Slider::RotaryVerticalDrag);

    // Shift switches to fine steps.
    control.setVelocityBasedMode(false);
    control.setVelocityModeParameters(0.15, 1, 0.0, true, juce::ModifierKeys::shiftModifier);
    control.setMouseDragSensitivity(200);
}

juce::String ParamControl::valueText() const
{
    const auto v = static_cast<float>(control.getValue());
    if (ctx.ui.getString(UiState::Ids::valueDisplay) == "percent" && paramSpec->choices.isEmpty())
    {
        const auto norm = (v - paramSpec->min) / juce::jmax(1.0e-6f, paramSpec->max - paramSpec->min);
        return juce::String(juce::roundToInt(norm * 100.0f)) + " %";
    }
    return params::formatValue(*paramSpec, v);
}

juce::Rectangle<int> ParamControl::valueArea() const
{
    auto b = getLocalBounds();
    switch (style)
    {
    case Style::knob:
    case Style::smallKnob:
        return b.removeFromBottom(16);
    case Style::verticalBar:
        return b.removeFromTop(16);
    case Style::horizontalBar:
        return b.removeFromRight(72);
    }
    return {};
}

void ParamControl::resized()
{
    auto b = getLocalBounds();
    switch (style)
    {
    case Style::knob:
    case Style::smallKnob:
    {
        b.removeFromTop(14);    // badges
        b.removeFromBottom(32); // name and value
        const auto size = juce::jmin(b.getWidth(), b.getHeight());
        control.setBounds(b.withSizeKeepingCentre(size, size));
        break;
    }
    case Style::verticalBar:
        b.removeFromTop(18);
        b.removeFromBottom(20);
        control.setBounds(b.withSizeKeepingCentre(28, b.getHeight()));
        break;
    case Style::horizontalBar:
        b.removeFromLeft(labelWidth());
        b.removeFromRight(72);
        control.setBounds(b);
        break;
    }
}

void ParamControl::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const bool dimmed = isDimmed();
    const bool showValue =
        ctx.ui.getString(UiState::Ids::valueDisplay) != "hover" || isMouseOverOrDragging(true);
    auto nameColour = dimmed ? t.text2.withAlpha(0.6f) : t.text2;
    auto valueColour = dimmed ? t.text2.withAlpha(0.6f) : t.text;

    auto b = getLocalBounds();
    switch (style)
    {
    case Style::knob:
    case Style::smallKnob:
    {
        auto bottom = b.removeFromBottom(32);
        g.setColour(nameColour);
        g.setFont(AlfaLookAndFeel::labelFont(style == Style::knob ? 12.5f : 11.5f));
        g.drawText(displayName, bottom.removeFromTop(16), juce::Justification::centred, true);
        if (showValue)
        {
            g.setColour(valueColour);
            g.setFont(AlfaLookAndFeel::monoFont(style == Style::knob ? 12.5f : 11.5f));
            g.drawText(valueText(), bottom, juce::Justification::centred, true);
        }
        break;
    }
    case Style::verticalBar:
    {
        if (showValue)
        {
            g.setColour(valueColour);
            g.setFont(AlfaLookAndFeel::monoFont(11.5f));
            g.drawText(valueText(), b.removeFromTop(16), juce::Justification::centred, true);
        }
        g.setColour(nameColour);
        g.setFont(AlfaLookAndFeel::labelFont(11.5f));
        g.drawText(displayName, getLocalBounds().removeFromBottom(18), juce::Justification::centred, true);
        break;
    }
    case Style::horizontalBar:
    {
        g.setColour(nameColour);
        g.setFont(AlfaLookAndFeel::labelFont(12.5f));
        g.drawText(displayName, b.removeFromLeft(labelWidth()), juce::Justification::centredLeft, true);
        if (showValue)
        {
            g.setColour(valueColour);
            g.setFont(AlfaLookAndFeel::monoFont(12.0f));
            g.drawText(valueText(), b.removeFromRight(72), juce::Justification::centredRight, true);
        }
        break;
    }
    }
}

void ParamControl::paintOverChildren(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    auto badgeRow = getLocalBounds().removeFromTop(14).toFloat();
    if (style == Style::horizontalBar)
        badgeRow = getLocalBounds().toFloat().withWidth(0.0f);

    if (isDimmed())
    {
        const auto label =
            juce::String(params::kTiers[static_cast<size_t>(paramSpec->minimumTier)].shortName) + "+";
        drawBadge(g, badgeRow.removeFromRight(38.0f).withTrimmedBottom(1.0f), label, t.warn, t);
        badgeRow.removeFromRight(3.0f);
    }

    if (ctx.ui.isLocked(paramSpec->id))
    {
        auto area = style == Style::horizontalBar
                        ? getLocalBounds().toFloat().removeFromLeft(34.0f).withHeight(14.0f)
                        : badgeRow.removeFromLeft(34.0f).withTrimmedBottom(1.0f);
        drawBadge(g, area, "LOCK", t.text2, t);
    }
}

void ParamControl::mouseDown(const juce::MouseEvent& e)
{
    if (e.mods.isPopupMenu())
    {
        showMenu();
        return;
    }

    if (e.eventComponent == this && valueArea().contains(e.getPosition()))
        typeValue();
}

void ParamControl::showMenu()
{
    juce::PopupMenu menu;
    menu.addSectionHeader(paramSpec->name);
    menu.addItem("Type a value...", [this] { typeValue(); });
    menu.addItem("Reset to default",
                 [this] { control.setValue(static_cast<double>(paramSpec->defaultValue)); });
    menu.addSeparator();
    const bool locked = ctx.ui.isLocked(paramSpec->id);
    menu.addItem("Lock when browsing presets", true, locked,
                 [this, locked] { ctx.ui.setLocked(paramSpec->id, !locked); });
    // These arrive with the modulation matrix and MIDI learn; shown so people
    // know they're coming.
    menu.addItem("Map to macro (coming)", false, false, nullptr);
    menu.addItem("Add modulation (coming)", false, false, nullptr);
    menu.addItem("MIDI learn (coming)", false, false, nullptr);
    if (isDimmed())
    {
        menu.addSeparator();
        menu.addItem("Needs the " +
                         juce::String(params::kTiers[static_cast<size_t>(paramSpec->minimumTier)].name) +
                         " model or larger",
                     false, false, nullptr);
    }
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(this));
}

void ParamControl::typeValue()
{
    auto* editor = new juce::TextEditor();
    editor->setText(params::formatValue(*paramSpec, static_cast<float>(control.getValue())), false);
    editor->setFont(AlfaLookAndFeel::monoFont(13.0f));
    editor->setJustification(juce::Justification::centred);
    editor->selectAll();
    editor->setSize(110, 26);

    juce::Component::SafePointer<ParamControl> safe(this);
    editor->onReturnKey = [safe, editor]
    {
        if (safe != nullptr)
            safe->control.setValue(
                static_cast<double>(params::parseValue(*safe->paramSpec, editor->getText())));
        if (auto* box = editor->findParentComponentOfClass<juce::CallOutBox>())
            box->dismiss();
    };

    auto& box = juce::CallOutBox::launchAsynchronously(std::unique_ptr<juce::Component>(editor),
                                                       getScreenBounds(), nullptr);
    juce::ignoreUnused(box);
    editor->grabKeyboardFocus();
}

// Segmented
// -----------------------------------------------------------------------

Segmented::Segmented(juce::StringArray labels, std::function<void(int)> onSelect)
    : callback(std::move(onSelect))
{
    for (int i = 0; i < labels.size(); ++i)
    {
        auto* b = buttons.add(new juce::TextButton(labels[i]));
        b->setClickingTogglesState(false);
        b->onClick = [this, i]
        {
            setSelected(i);
            if (callback)
                callback(i);
        };
        addAndMakeVisible(b);
    }
    setSelected(0);
}

void Segmented::setSelected(int index)
{
    current = index;
    for (int i = 0; i < buttons.size(); ++i)
        buttons[i]->setToggleState(i == index, juce::dontSendNotification);
}

void Segmented::resized()
{
    auto b = getLocalBounds();
    const auto w = b.getWidth() / juce::jmax(1, buttons.size());
    for (auto* button : buttons)
        button->setBounds(b.removeFromLeft(w).reduced(2, 0));
}

} // namespace alfa::ui
