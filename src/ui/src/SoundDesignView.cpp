#include "alfa/ui/Views.h"

namespace alfa::ui
{

SoundDesignView::SoundDesignView(EditorContext& c, LiveData& l)
    : ctx(c)
    , live(l)
    , methods(c, l, "Method")
    , watcher(c.params, [this] { rebuild(); })
{
    methods.setShowSummaries(true);
    addAndMakeVisible(methods);

    showUnused.setTooltip("Knobs this method ignores are hidden; show them dimmed instead");
    showUnused.onClick = [this]
    {
        rebuild();
    };
    addAndMakeVisible(showUnused);

    addRoute.setEnabled(false);
    addRoute.setTooltip("Modulation routing arrives with the DSP engines");
    addAndMakeVisible(addRoute);

    rebuild();
    live.addListener(this);
    ctx.ui.tree().addListener(this);
}

SoundDesignView::~SoundDesignView()
{
    live.removeListener(this);
    ctx.ui.tree().removeListener(this);
}

void SoundDesignView::rebuild()
{
    knobControls.clear();
    const auto used = usedKnobs(ctx.params);
    for (int i = 0; i < params::kNumKnobs; ++i)
    {
        const bool isUsed = std::find(used.begin(), used.end(), i) != used.end();
        if (isUsed || showUnused.getToggleState())
            addAndMakeVisible(
                knobControls.add(new ParamControl(ctx, params::id::knob(i), ParamControl::Style::smallKnob)));
    }
    resized();
    repaint();
}

void SoundDesignView::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    if (id == UiState::Ids::density)
        resized();
}

void SoundDesignView::liveDataUpdated()
{
    if (isShowing())
        repaint(displayArea);
}

void SoundDesignView::resized()
{
    const auto gap = gapFor(ctx.ui);
    auto b = getLocalBounds();
    methodArea = b.removeFromLeft(300);
    methods.setBounds(methodArea);
    b.removeFromLeft(gap);
    modArea = b.removeFromRight(250);
    b.removeFromRight(gap);
    layerArea = b;

    auto inner = layerArea.withTrimmedTop(kPanelTitleHeight + 6).reduced(16, 8);
    displayArea = inner.removeFromTop(150);
    inner.removeFromTop(8);
    showUnused.setBounds(inner.removeFromTop(24).removeFromRight(190));
    inner.removeFromTop(4);
    knobGrid = inner;

    // Up to eight knobs per row; 32 knobs fit in four rows.
    const auto count = knobControls.size();
    const auto perRow = 8;
    const auto rows = juce::jmax(1, (count + perRow - 1) / perRow);
    const auto rowH = juce::jmin(110, knobGrid.getHeight() / juce::jmax(3, rows));
    const auto w = knobGrid.getWidth() / perRow;
    for (int i = 0; i < count; ++i)
        knobControls[i]->setBounds(knobGrid.getX() + (i % perRow) * w, knobGrid.getY() + (i / perRow) * rowH,
                                   w, rowH);

    addRoute.setBounds(modArea.getRight() - 12 - 100, modArea.getY() + 4, 100, 22);
}

void SoundDesignView::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const auto& s = live.snapshot();
    const auto shape = params::readShape(ctx.params);
    const auto& info = params::kMethods[static_cast<size_t>(shape.method)];

    // Selected method
    auto subtitle = juce::String(juce::roundToInt(info.capacityShare * 100.0f)) + "% of model capacity";
    if (shape.method == params::Method::hybrid)
        subtitle = juce::String(params::kMethods[static_cast<size_t>(shape.hybridA)].name) + " + " +
                   params::kMethods[static_cast<size_t>(shape.hybridB)].name;
    drawPanel(g, t, layerArea, info.name, subtitle);
    g.setColour(t.bg);
    g.fillRoundedRectangle(displayArea.toFloat(), 6.0f);
    viz::waveform(g, displayArea.toFloat().reduced(10.0f), t, s, t.accent);
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.0f));
    g.drawText(juce::String(info.summary) + " (output, live)", displayArea.reduced(10, 6),
               juce::Justification::topLeft, false);

    g.setFont(AlfaLookAndFeel::labelFont(11.5f, true));
    g.drawText("KNOBS", showUnused.getBounds().withX(knobGrid.getX()).withWidth(200),
               juce::Justification::centredLeft, false);

    // Modulation
    drawPanel(g, t, modArea, "Modulation");
    auto m = modArea.withTrimmedTop(kPanelTitleHeight + 12).reduced(14, 8);
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.0f, true));
    auto header = m.removeFromTop(18);
    g.drawText("SOURCE", header.removeFromLeft(70), juce::Justification::centredLeft, false);
    g.drawText("TARGET", header.removeFromLeft(80), juce::Justification::centredLeft, false);
    g.drawText("AMOUNT", header, juce::Justification::centredLeft, false);
    g.setColour(t.line);
    g.fillRect(m.removeFromTop(1));
    m.removeFromTop(16);
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(13.0f));
    g.drawFittedText(
        "No routes yet.\n\nModulation routing (LFOs, envelopes and MPE driving any knob) arrives "
        "with the DSP engines. The mod wheel already moves the knobs each method assigns to it. "
        "Modulated controls will show a blue ring around their value.",
        m.removeFromTop(160), juce::Justification::topLeft, 8);
}

} // namespace alfa::ui
