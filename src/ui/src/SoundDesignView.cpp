#include "alfa/ui/Views.h"

namespace alfa::ui
{

SoundDesignView::SoundDesignView(EditorContext& c, LiveData& l)
    : ctx(c)
    , live(l)
{
    for (const auto& id : {params::id::latentDrift, params::id::seed, params::id::followModel})
        addAndMakeVisible(neuralControls.add(new ParamControl(ctx, id, ParamControl::Style::smallKnob)));

    addRoute.setEnabled(false);
    addRoute.setTooltip("Modulation routing arrives with the DSP engines");
    addAndMakeVisible(addRoute);

    selectLayer(layer());
    live.addListener(this);
    ctx.ui.tree().addListener(this);
}

SoundDesignView::~SoundDesignView()
{
    live.removeListener(this);
    ctx.ui.tree().removeListener(this);
}

int SoundDesignView::layer() const
{
    return juce::jlimit(0, params::kNumMethods - 1, ctx.ui.getInt(UiState::Ids::soundLayer));
}

void SoundDesignView::selectLayer(int index)
{
    engineControls.clear();
    for (const auto* spec : params::engineSpecs(index))
        addAndMakeVisible(engineControls.add(new ParamControl(ctx, spec->id)));

    layerBlend =
        std::make_unique<ParamControl>(ctx, params::id::blend(index), ParamControl::Style::horizontalBar);
    layerBlend->setDisplayName("Blend");
    addAndMakeVisible(*layerBlend);

    resized();
    repaint();
}

void SoundDesignView::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    if (id == UiState::Ids::soundLayer)
        selectLayer(layer());
    else if (id == UiState::Ids::density)
        resized();
}

void SoundDesignView::liveDataUpdated()
{
    if (isShowing())
    {
        repaint(displayArea);
        repaint(layersArea);
    }
}

void SoundDesignView::mouseUp(const juce::MouseEvent& e)
{
    for (size_t i = 0; i < layerRows.size(); ++i)
        if (layerRows[i].contains(e.getPosition()))
            ctx.ui.set(UiState::Ids::soundLayer, static_cast<int>(i));
}

void SoundDesignView::resized()
{
    const auto gap = gapFor(ctx.ui);
    auto b = getLocalBounds();
    layersArea = b.removeFromLeft(230);
    b.removeFromLeft(gap);
    modArea = b.removeFromRight(330);
    b.removeFromRight(gap);
    layerArea = b;

    layerRows.clear();
    auto rows = layersArea.withTrimmedTop(kPanelTitleHeight + 6).reduced(8, 4);
    for (int i = 0; i < params::kNumMethods; ++i)
        layerRows.push_back(rows.removeFromTop(46).reduced(0, 2));

    auto inner = layerArea.withTrimmedTop(kPanelTitleHeight + 6).reduced(16, 8);
    if (layerBlend != nullptr)
        layerBlend->setBounds(inner.removeFromTop(28).removeFromRight(360));
    inner.removeFromTop(8);
    displayArea = inner.removeFromTop(200);
    inner.removeFromTop(16);

    auto knobs = inner.removeFromTop(150);
    const auto w = knobs.getWidth() / juce::jmax(1, engineControls.size());
    for (auto* k : engineControls)
        k->setBounds(knobs.removeFromLeft(w).reduced(8, 0));

    inner.removeFromTop(30); // "Neural" heading
    auto neural = inner.removeFromTop(110).removeFromLeft(3 * 110);
    for (auto* k : neuralControls)
        k->setBounds(neural.removeFromLeft(110).reduced(8, 0));

    addRoute.setBounds(modArea.getRight() - 12 - 100, modArea.getY() + 4, 100, 22);
}

void SoundDesignView::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const auto& s = live.snapshot();
    const auto selected = layer();

    // Layers
    drawPanel(g, t, layersArea, "Layers");
    for (size_t i = 0; i < layerRows.size(); ++i)
    {
        const auto row = layerRows[i];
        const bool isSelected = static_cast<int>(i) == selected;
        const auto amount = s.blend[i];
        if (isSelected)
        {
            g.setColour(t.accent.withAlpha(0.14f));
            g.fillRoundedRectangle(row.toFloat(), 5.0f);
            g.setColour(t.accent);
            g.fillRect(row.withWidth(3));
        }
        auto r = row.reduced(12, 6);
        g.setColour(amount > 0.001f ? t.text : t.text2.withAlpha(0.6f));
        g.setFont(AlfaLookAndFeel::labelFont(13.0f, isSelected));
        g.drawText(params::kMethods[i].name, r.removeFromTop(18), juce::Justification::centredLeft, false);
        g.setFont(AlfaLookAndFeel::monoFont(11.0f));
        g.setColour(t.text2);
        g.drawText(juce::String(juce::roundToInt(amount * 100.0f)) + "%",
                   row.reduced(12, 6).removeFromTop(18), juce::Justification::centredRight, false);
        auto bar = r.toFloat().withSizeKeepingCentre(static_cast<float>(r.getWidth()), 4.0f);
        g.setColour(t.line);
        g.fillRoundedRectangle(bar, 2.0f);
        g.setColour(amount > 0.001f ? t.accent : t.line);
        g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * amount), 2.0f);
    }

    // Selected layer
    drawPanel(g, t, layerArea, params::kMethods[static_cast<size_t>(selected)].name,
              juce::String(
                  juce::roundToInt(params::kMethods[static_cast<size_t>(selected)].capacityShare * 100.0f)) +
                  "% of model capacity");
    g.setColour(t.bg);
    g.fillRoundedRectangle(displayArea.toFloat(), 6.0f);
    viz::waveform(g, displayArea.toFloat().reduced(10.0f), t, s, t.accent);
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.0f));
    g.drawText(s.engines[static_cast<size_t>(selected)] > 0.0f ? "output (all layers, live)"
                                                               : "output (layer silent)",
               displayArea.reduced(10, 6), juce::Justification::topLeft, false);

    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.5f, true));
    if (!neuralControls.isEmpty())
        g.drawText("NEURAL",
                   neuralControls[0]
                       ->getBounds()
                       .withY(neuralControls[0]->getY() - 26)
                       .withHeight(18)
                       .withWidth(200),
                   juce::Justification::centredLeft, false);

    // Modulation
    drawPanel(g, t, modArea, "Modulation");
    auto m = modArea.withTrimmedTop(kPanelTitleHeight + 12).reduced(14, 8);
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.0f, true));
    auto header = m.removeFromTop(18);
    g.drawText("SOURCE", header.removeFromLeft(90), juce::Justification::centredLeft, false);
    g.drawText("TARGET", header.removeFromLeft(120), juce::Justification::centredLeft, false);
    g.drawText("AMOUNT", header, juce::Justification::centredLeft, false);
    g.setColour(t.line);
    g.fillRect(m.removeFromTop(1));
    m.removeFromTop(16);
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(13.0f));
    g.drawFittedText("No routes yet.\n\nModulation routing (LFOs, envelopes and "
                     "MPE driving any control) arrives "
                     "with the DSP engines. Modulated controls will show a blue "
                     "ring around their value.",
                     m.removeFromTop(120), juce::Justification::topLeft, 6);
}

} // namespace alfa::ui
