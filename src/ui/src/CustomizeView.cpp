#include "alfa/ui/Views.h"

namespace alfa::ui
{
namespace
{
using I = UiState::Ids;

struct AccentChoice
{
    const char* name;
    const char* hex; // empty: the theme's own accent
};

constexpr AccentChoice accents[] = {
    {"Theme default", ""}, {"Amber", "E9A23B"}, {"Teal", "3BB8A6"},
    {"Violet", "A78BE8"},  {"Lime", "A6CE3F"},  {"Rose", "E8829E"},
};

const juce::StringArray densityIds{"comfortable", "compact"};
const juce::StringArray gestureIds{"vertical", "horizontal", "circular"};
const juce::StringArray valueIds{"units", "percent", "hover"};
const juce::StringArray macroCounts{"4", "8", "12"};

struct PanelToggle
{
    const juce::Identifier* id;
    const char* label;
};

const PanelToggle panelToggleList[] = {
    {&I::showBlend, "Method blend"}, {&I::showMacros, "Macros"},     {&I::showTimbre, "Timbre space"},
    {&I::showVoice, "Voice strip"},  {&I::showKeyboard, "Keyboard"}, {&I::showFlowStrip, "Signal flow strip"},
};
} // namespace

CustomizeView::CustomizeView(EditorContext& c)
    : ctx(c)
    , themeChoice(Theme::builtInNames(), [this](int i) { ctx.ui.set(I::theme, Theme::builtInIds()[i]); })
    , density({"Comfortable", "Compact"}, [this](int i) { ctx.ui.set(I::density, densityIds[i]); })
    , gesture({"Vertical", "Horizontal", "Circular"},
              [this](int i) { ctx.ui.set(I::knobGesture, gestureIds[i]); })
    , valueDisplay({"Units", "Percent", "On hover"},
                   [this](int i) { ctx.ui.set(I::valueDisplay, valueIds[i]); })
    , macrosPerPage(macroCounts,
                    [this](int i)
                    {
                        ctx.ui.set(I::macrosPerPage, macroCounts[i].getIntValue());
                        ctx.ui.set(I::macroPage, 0);
                    })
{
    for (auto* comp : std::initializer_list<juce::Component*>{&themeChoice, &density, &gesture, &valueDisplay,
                                                              &macrosPerPage, &accent, &scale, &useForNew,
                                                              &exportButton, &importButton})
        addAndMakeVisible(comp);

    for (int i = 0; i < static_cast<int>(std::size(accents)); ++i)
        accent.addItem(accents[i].name, i + 1);
    accent.onChange = [this]
    {
        const auto i = accent.getSelectedItemIndex();
        if (i >= 0)
            ctx.ui.set(I::accent, accents[i].hex);
    };

    scale.setSliderStyle(juce::Slider::LinearHorizontal);
    scale.setRange(75.0, 200.0, 5.0);
    scale.setTextBoxStyle(juce::Slider::TextBoxRight, false, 64, 24);
    scale.setTextValueSuffix(" %");
    scale.setTitle("Interface size");
    // Apply on release: resizing the window mid-drag would move the slider under
    // the mouse.
    scale.onDragEnd = [this]
    {
        ctx.ui.set(I::scale, scale.getValue() / 100.0);
    };
    scale.onValueChange = [this]
    {
        if (!scale.isMouseButtonDown())
            ctx.ui.set(I::scale, scale.getValue() / 100.0);
    };

    for (const auto& p : panelToggleList)
    {
        auto* toggle = panelToggles.add(new juce::ToggleButton(p.label));
        toggle->onClick = [this, toggle, id = p.id]
        {
            ctx.ui.set(*id, toggle->getToggleState());
        };
        addAndMakeVisible(toggle);
    }

    useForNew.onClick = [this]
    {
        ctx.ui.saveAsUserDefault();
        statusText = "Saved. New instances will open with this setup.";
        repaint();
    };
    exportButton.onClick = [this]
    {
        exportTheme();
    };
    importButton.onClick = [this]
    {
        importTheme();
    };

    syncFromState();
    ctx.ui.tree().addListener(this);
}

CustomizeView::~CustomizeView()
{
    ctx.ui.tree().removeListener(this);
}

void CustomizeView::syncFromState()
{
    themeChoice.setSelected(juce::jmax(0, Theme::builtInIds().indexOf(ctx.ui.getString(I::theme))));
    density.setSelected(juce::jmax(0, densityIds.indexOf(ctx.ui.getString(I::density))));
    gesture.setSelected(juce::jmax(0, gestureIds.indexOf(ctx.ui.getString(I::knobGesture))));
    valueDisplay.setSelected(juce::jmax(0, valueIds.indexOf(ctx.ui.getString(I::valueDisplay))));
    macrosPerPage.setSelected(
        juce::jmax(0, macroCounts.indexOf(juce::String(ctx.ui.getInt(I::macrosPerPage)))));

    const auto hex = ctx.ui.getString(I::accent);
    for (int i = 0; i < static_cast<int>(std::size(accents)); ++i)
        if (hex.equalsIgnoreCase(accents[i].hex))
            accent.setSelectedItemIndex(i, juce::dontSendNotification);

    if (!scale.isMouseButtonDown())
        scale.setValue(ctx.ui.getDouble(I::scale) * 100.0, juce::dontSendNotification);

    for (size_t i = 0; i < std::size(panelToggleList); ++i)
        panelToggles[static_cast<int>(i)]->setToggleState(ctx.ui.getBool(*panelToggleList[i].id),
                                                          juce::dontSendNotification);
}

void CustomizeView::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    syncFromState();
    if (id == I::density)
        resized();
    repaint();
}

void CustomizeView::exportTheme()
{
    chooser =
        std::make_unique<juce::FileChooser>("Export theme file",
                                            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                                                .getChildFile(ctx.theme().name + ".alfatheme.json"),
                                            "*.json");
    chooser->launchAsync(juce::FileBrowserComponent::saveMode |
                             juce::FileBrowserComponent::warnAboutOverwriting,
                         [this](const juce::FileChooser& fc)
                         {
                             const auto file = fc.getResult();
                             if (file == juce::File())
                                 return;
                             statusText = file.replaceWithText(ctx.theme().toJson())
                                              ? "Theme written to " + file.getFileName()
                                              : "Could not write " + file.getFullPathName();
                             repaint();
                         });
}

void CustomizeView::importTheme()
{
    chooser = std::make_unique<juce::FileChooser>(
        "Import theme file", juce::File::getSpecialLocation(juce::File::userDocumentsDirectory), "*.json");
    chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                         [this](const juce::FileChooser& fc)
                         {
                             const auto file = fc.getResult();
                             if (!file.existsAsFile())
                                 return;
                             ctx.lnf.setTheme(Theme::fromJson(file.loadFileAsString()));
                             if (auto* top = getTopLevelComponent())
                             {
                                 top->sendLookAndFeelChange();
                                 top->repaint();
                             }
                             statusText = "Loaded " + file.getFileName() + " for this session.";
                             repaint();
                         });
}

void CustomizeView::resized()
{
    const auto gap = gapFor(ctx.ui);
    auto b = getLocalBounds();
    previewArea = b.removeFromRight(420);
    b.removeFromRight(gap);
    settingsArea = b;

    rowLabels.clear();
    auto rows = settingsArea.withTrimmedTop(kPanelTitleHeight + 10).reduced(18, 6);
    auto row = [&](const juce::String& label, juce::Component& comp, int width)
    {
        auto r = rows.removeFromTop(40);
        rowLabels.emplace_back(label, r.removeFromLeft(170));
        comp.setBounds(r.removeFromLeft(width).reduced(0, 5));
    };

    row("Theme", themeChoice, 380);
    row("Accent colour", accent, 200);
    row("Interface size", scale, 380);
    row("Density", density, 260);
    row("Knob drag", gesture, 380);
    row("Value display", valueDisplay, 380);
    row("Macros per page", macrosPerPage, 200);

    rows.removeFromTop(12);
    auto panelsRow = rows.removeFromTop(24);
    rowLabels.emplace_back("Perform shows", panelsRow.removeFromLeft(170));
    auto grid = panelsRow.withHeight(96);
    rows.removeFromTop(80);
    for (int i = 0; i < panelToggles.size(); ++i)
    {
        const auto col = i % 2, line = i / 2;
        panelToggles[i]->setBounds(grid.getX() + col * 200, grid.getY() + line * 30 - 4, 190, 28);
    }

    rows.removeFromTop(16);
    auto buttons = rows.removeFromTop(34);
    useForNew.setBounds(buttons.removeFromLeft(190));
    buttons.removeFromLeft(10);
    exportButton.setBounds(buttons.removeFromLeft(170));
    buttons.removeFromLeft(10);
    importButton.setBounds(buttons.removeFromLeft(170));
}

void CustomizeView::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    drawPanel(g, t, settingsArea, "Customize", "saved with this instance, never in presets");

    g.setFont(AlfaLookAndFeel::labelFont(13.5f));
    g.setColour(t.text);
    for (const auto& [label, area] : rowLabels)
        g.drawText(label, area, juce::Justification::centredLeft, false);

    if (statusText.isNotEmpty())
    {
        g.setColour(t.text2);
        g.setFont(AlfaLookAndFeel::labelFont(12.5f));
        g.drawText(statusText, settingsArea.reduced(18, 14).removeFromBottom(20),
                   juce::Justification::centredLeft, true);
    }

    // Preview: the theme's tokens with what each one means.
    drawPanel(g, t, previewArea, "Preview", t.name);
    auto p = previewArea.withTrimmedTop(kPanelTitleHeight + 12).reduced(18, 6);
    const std::pair<const char*, juce::Colour> swatches[] = {
        {"Window", t.bg},    {"Panel", t.panel},          {"Lines", t.line},     {"Text", t.text},
        {"Labels", t.text2}, {"Value you set", t.accent}, {"Modulation", t.mod}, {"Warning", t.warn},
    };
    for (const auto& [name, colour] : swatches)
    {
        auto r = p.removeFromTop(30);
        auto sw = r.removeFromLeft(44).reduced(0, 4).toFloat();
        g.setColour(colour);
        g.fillRoundedRectangle(sw, 4.0f);
        g.setColour(t.line);
        g.drawRoundedRectangle(sw, 4.0f, 1.0f);
        r.removeFromLeft(12);
        g.setColour(t.text);
        g.setFont(AlfaLookAndFeel::labelFont(13.0f));
        g.drawText(name, r, juce::Justification::centredLeft, false);
        g.setColour(t.text2);
        g.setFont(AlfaLookAndFeel::monoFont(12.0f));
        g.drawText("#" + colour.toDisplayString(false), r, juce::Justification::centredRight, false);
    }

    // A sample knob with value and modulation ring, drawn by the same
    // look-and-feel.
    p.removeFromTop(16);
    auto sample = p.removeFromTop(120);
    juce::Slider demo(juce::Slider::RotaryVerticalDrag, juce::Slider::NoTextBox);
    demo.setRange(0.0, 1.0);
    demo.setValue(0.62);
    demo.getProperties().set(AlfaLookAndFeel::modDepthProperty, 0.2);
    const auto knob = sample.removeFromLeft(110).reduced(4);
    const auto params = demo.getRotaryParameters();
    ctx.lnf.drawRotarySlider(g, knob.getX(), knob.getY(), knob.getWidth(), knob.getHeight(), 0.62f,
                             params.startAngleRadians, params.endAngleRadians, demo);
    sample.removeFromLeft(12);
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(12.5f));
    g.drawFittedText("Amber arc: the value you set.\nBlue ring: how far modulation moves "
                     "it.\nDrag, Shift-drag for "
                     "fine steps, double-click to reset, click the value to type one.",
                     sample, juce::Justification::centredLeft, 5);
}

} // namespace alfa::ui
