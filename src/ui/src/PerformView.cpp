#include "alfa/ui/Views.h"

namespace alfa::ui
{

// TimbrePad
// ------------------------------------------------------------------------

TimbrePad::TimbrePad(EditorContext& c)
    : ctx(c)
    , xParam(c.params.getParameter(params::id::timbreX))
    , yParam(c.params.getParameter(params::id::timbreY))
{
    ctx.params.addParameterListener(params::id::timbreX, this);
    ctx.params.addParameterListener(params::id::timbreY, this);

    snapA.setTooltip("Click to recall snapshot A, Shift-click to store the current position");
    snapB.setTooltip("Click to recall snapshot B, Shift-click to store the current position");
    snapA.onClick = [this]
    {
        juce::ModifierKeys::currentModifiers.isShiftDown() || snapshots[0].x < 0.0f ? storeSnapshot(0)
                                                                                    : recallSnapshot(0);
    };
    snapB.onClick = [this]
    {
        juce::ModifierKeys::currentModifiers.isShiftDown() || snapshots[1].x < 0.0f ? storeSnapshot(1)
                                                                                    : recallSnapshot(1);
    };
    randomize.onClick = [this]
    {
        randomizeNearby();
    };
    addAndMakeVisible(snapA);
    addAndMakeVisible(snapB);
    addAndMakeVisible(randomize);
    setWantsKeyboardFocus(true);
    setTitle("Timbre space");
}

TimbrePad::~TimbrePad()
{
    ctx.params.removeParameterListener(params::id::timbreX, this);
    ctx.params.removeParameterListener(params::id::timbreY, this);
}

void TimbrePad::parameterChanged(const juce::String&, float)
{
    juce::MessageManager::callAsync(
        [safe = juce::Component::SafePointer<TimbrePad>(this)]
        {
            if (safe != nullptr)
                safe->repaint();
        });
}

juce::Rectangle<float> TimbrePad::padArea() const
{
    return getLocalBounds()
        .withTrimmedTop(kPanelTitleHeight + 4)
        .withTrimmedBottom(40)
        .reduced(14, 4)
        .toFloat();
}

void TimbrePad::setFromPosition(juce::Point<float> p)
{
    const auto area = padArea();
    xParam->setValueNotifyingHost(juce::jlimit(0.0f, 1.0f, (p.x - area.getX()) / area.getWidth()));
    yParam->setValueNotifyingHost(juce::jlimit(0.0f, 1.0f, (area.getBottom() - p.y) / area.getHeight()));
}

void TimbrePad::mouseDown(const juce::MouseEvent& e)
{
    if (!padArea().contains(e.position))
        return;
    xParam->beginChangeGesture();
    yParam->beginChangeGesture();
    setFromPosition(e.position);
}

void TimbrePad::mouseDrag(const juce::MouseEvent& e)
{
    setFromPosition(e.position);
}

void TimbrePad::mouseUp(const juce::MouseEvent&)
{
    xParam->endChangeGesture();
    yParam->endChangeGesture();
}

void TimbrePad::storeSnapshot(int slot)
{
    snapshots[static_cast<size_t>(slot)] = {xParam->getValue(), yParam->getValue()};
    repaint();
}

void TimbrePad::recallSnapshot(int slot)
{
    const auto p = snapshots[static_cast<size_t>(slot)];
    xParam->setValueNotifyingHost(p.x);
    yParam->setValueNotifyingHost(p.y);
}

void TimbrePad::randomizeNearby()
{
    auto& rng = juce::Random::getSystemRandom();
    auto nudge = [&rng](float v)
    {
        return juce::jlimit(0.0f, 1.0f, v + (rng.nextFloat() - 0.5f) * 0.2f);
    };
    xParam->setValueNotifyingHost(nudge(xParam->getValue()));
    yParam->setValueNotifyingHost(nudge(yParam->getValue()));
}

void TimbrePad::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    drawPanel(g, t, getLocalBounds(), "Timbre space");

    const auto area = padArea();
    g.setColour(t.bg);
    g.fillRoundedRectangle(area, 6.0f);
    g.setColour(t.line);
    for (int i = 1; i < 4; ++i)
    {
        const auto fx = area.getX() + area.getWidth() * static_cast<float>(i) / 4.0f;
        const auto fy = area.getY() + area.getHeight() * static_cast<float>(i) / 4.0f;
        g.drawVerticalLine(juce::roundToInt(fx), area.getY(), area.getBottom());
        g.drawHorizontalLine(juce::roundToInt(fy), area.getX(), area.getRight());
    }
    g.drawRoundedRectangle(area, 6.0f, 1.0f);

    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.0f));
    g.drawText("warm", area.reduced(6.0f), juce::Justification::bottomLeft, false);
    g.drawText("bright", area.reduced(6.0f), juce::Justification::bottomRight, false);
    g.drawText("evolving", area.reduced(6.0f), juce::Justification::topLeft, false);

    auto toPoint = [&area](float x, float y)
    {
        return juce::Point<float>(area.getX() + x * area.getWidth(), area.getBottom() - y * area.getHeight());
    };

    for (size_t i = 0; i < snapshots.size(); ++i)
    {
        if (snapshots[i].x < 0.0f)
            continue;
        const auto p = toPoint(snapshots[i].x, snapshots[i].y);
        g.setColour(t.text2);
        g.drawEllipse(juce::Rectangle<float>(12.0f, 12.0f).withCentre(p), 1.2f);
        g.drawText(i == 0 ? "A" : "B",
                   juce::Rectangle<float>(12.0f, 12.0f).withCentre(p.translated(12.0f, -10.0f)),
                   juce::Justification::centred, false);
    }

    const auto p = toPoint(xParam->getValue(), yParam->getValue());
    g.setColour(t.accent.withAlpha(0.25f));
    g.fillEllipse(juce::Rectangle<float>(34.0f, 34.0f).withCentre(p));
    g.setColour(t.accent);
    g.fillEllipse(juce::Rectangle<float>(14.0f, 14.0f).withCentre(p));

    g.setColour(t.text);
    g.setFont(AlfaLookAndFeel::monoFont(11.5f));
    const auto readout = "bright " + juce::String(juce::roundToInt(xParam->getValue() * 100.0f)) +
                         "   evolve " + juce::String(juce::roundToInt(yParam->getValue() * 100.0f));
    g.drawText(readout, getLocalBounds().reduced(12, 6).removeFromTop(kPanelTitleHeight - 6),
               juce::Justification::centredRight, false);
}

void TimbrePad::resized()
{
    auto row = getLocalBounds().reduced(14, 8).removeFromBottom(28);
    snapA.setBounds(row.removeFromLeft(36));
    row.removeFromLeft(6);
    snapB.setBounds(row.removeFromLeft(36));
    row.removeFromLeft(6);
    randomize.setBounds(row.removeFromRight(juce::jmin(150, row.getWidth())));
}

// FlowStrip
// ------------------------------------------------------------------------

FlowStrip::FlowStrip(EditorContext& c, LiveData& l)
    : ctx(c)
    , live(l)
{
    live.addListener(this);
    setTooltip("Signal flow: click to open the full view");
}

FlowStrip::~FlowStrip()
{
    live.removeListener(this);
}

void FlowStrip::mouseUp(const juce::MouseEvent&)
{
    ctx.ui.set(UiState::Ids::view, "signal");
}

// PerformView
// -------------------------------------------------------------------------

PerformView::PerformView(EditorContext& c, LiveData& l)
    : ctx(c)
    , live(l)
    , macroPages({"A", "B", "C"}, [this](int i) { ctx.ui.set(UiState::Ids::macroPage, i); })
    , timbre(c)
    , keyboard(c.keyboard, juce::MidiKeyboardComponent::horizontalKeyboard)
    , flowStrip(c, l)
{
    for (int m = 0; m < params::kNumMethods; ++m)
        addAndMakeVisible(blendControls.add(
            new ParamControl(ctx, params::id::blend(m), ParamControl::Style::horizontalBar)));

    addAndMakeVisible(macroPages);
    addAndMakeVisible(timbre);

    for (const auto& id :
         {params::id::attack, params::id::decay, params::id::sustain, params::id::release, params::id::voices,
          params::id::glide, params::id::bendRange, params::id::outputGain})
        addAndMakeVisible(voiceControls.add(new ParamControl(ctx, id, ParamControl::Style::smallKnob)));

    keyboard.setAvailableRange(24, 108);
    keyboard.setLowestVisibleKey(36);
    keyboard.setScrollButtonsVisible(false);
    keyboard.setOctaveForMiddleC(4);
    keyboard.setWantsKeyboardFocus(false);
    addAndMakeVisible(keyboard);
    addChildComponent(flowStrip);

    rebuildMacros();
    live.addListener(this);
    ctx.ui.tree().addListener(this);
}

PerformView::~PerformView()
{
    live.removeListener(this);
    ctx.ui.tree().removeListener(this);
}

void PerformView::rebuildMacros()
{
    macroControls.clear();
    const auto perPage = juce::jlimit(4, 12, ctx.ui.getInt(UiState::Ids::macrosPerPage));
    const auto pages = params::kNumMacros / perPage;
    const auto page = juce::jlimit(0, juce::jmin(2, pages - 1), ctx.ui.getInt(UiState::Ids::macroPage));
    macroPages.setSelected(page);

    for (int i = 0; i < perPage; ++i)
    {
        const auto index = page * perPage + i;
        addAndMakeVisible(macroControls.add(new ParamControl(ctx, params::id::macro(index))));
    }
    resized();
}

void PerformView::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    using I = UiState::Ids;
    if (id == I::macrosPerPage || id == I::macroPage)
        rebuildMacros();
    else if (id == I::showBlend || id == I::showMacros || id == I::showTimbre || id == I::showVoice ||
             id == I::showKeyboard || id == I::showFlowStrip || id == I::density)
        resized();
}

void PerformView::liveDataUpdated()
{
    if (isShowing())
    {
        repaint(voiceArea);
        repaint(meterArea);
        repaint(blendArea);
    }
}

void PerformView::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const auto& s = live.snapshot();

    if (!blendArea.isEmpty())
    {
        drawPanel(g, t, blendArea, "Method blend", "capacity");
        // Capacity share of each engine, drawn right of its slider, plus a live
        // activity dot.
        for (int m = 0; m < params::kNumMethods; ++m)
        {
            const auto* control = blendControls[m];
            auto row = control->getBounds().withX(blendArea.getRight() - 58).withWidth(48);
            const auto share = params::kMethods[static_cast<size_t>(m)].capacityShare;
            auto bar = row.toFloat().withSizeKeepingCentre(static_cast<float>(row.getWidth()), 4.0f);
            g.setColour(t.line);
            g.fillRoundedRectangle(bar, 2.0f);
            g.setColour(t.text2);
            g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * share / 0.3f), 2.0f);
            g.setFont(AlfaLookAndFeel::monoFont(10.5f));
            g.drawText(juce::String(juce::roundToInt(share * 100.0f)) + "%",
                       row.withTrimmedBottom(14).translated(0, -4), juce::Justification::centredRight, false);

            const auto activity = s.engines[static_cast<size_t>(m)];
            g.setColour(activity > 0.0f ? t.accent.withAlpha(0.4f + 0.6f * activity) : t.line);
            g.fillEllipse(juce::Rectangle<float>(6.0f, 6.0f)
                              .withCentre({static_cast<float>(blendArea.getX() + 10),
                                           static_cast<float>(control->getBounds().getCentreY())}));
        }
    }

    if (!macroArea.isEmpty())
        drawPanel(g, t, macroArea, "Macros");

    if (!voiceArea.isEmpty())
    {
        drawPanel(g, t, voiceArea, "Voice", juce::String(s.voices) + " playing");
        viz::envelope(g, envelopeArea.toFloat(), t, s);
        viz::meter(g, meterArea.toFloat(), t, s.peakL, s.peakR);
    }

    if (!stripArea.isEmpty())
        drawPanel(g, t, stripArea, {});
}

void FlowStrip::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const auto& s = live.snapshot();
    drawPanel(g, t, getLocalBounds(), {});

    const juce::StringArray names{"Notes", "Controls", "Model", "Blend", "Engines", "Env + FX", "Output"};
    const auto& steps = SignalFlowView::steps();
    std::vector<int> shown;
    for (int i = 0; i < names.size(); ++i)
        if (!ctx.ui.isStepHidden(steps[static_cast<size_t>(i)].id))
            shown.push_back(i);
    if (shown.empty())
        return;

    auto b = getLocalBounds().reduced(10, 8);
    const auto w = b.getWidth() / static_cast<int>(shown.size());
    for (size_t k = 0; k < shown.size(); ++k)
    {
        const auto i = shown[k];
        auto cell = b.removeFromLeft(w).reduced(4, 0);
        g.setColour(t.text2);
        g.setFont(AlfaLookAndFeel::labelFont(11.0f));
        g.drawText(names[i], cell.removeFromTop(14), juce::Justification::centredLeft, false);
        auto area = cell.toFloat().reduced(0.0f, 2.0f);
        switch (i)
        {
        case 0:
            viz::pianoRoll(g, area, t, live);
            break;
        case 1:
            viz::controlVector(g, area, t, s);
            break;
        case 2:
            viz::modelGraph(g, area, t, s, false);
            break;
        case 3:
            viz::blendBars(g, area, t, s, false);
            break;
        case 4:
            viz::waveform(g, area, t, s, t.accent);
            break;
        case 5:
            viz::envelope(g, area, t, s);
            break;
        default:
            viz::spectrum(g, area, t, s);
            break;
        }
        if (k + 1 < shown.size())
        {
            g.setColour(t.line);
            g.drawVerticalLine(cell.getRight() + 4, static_cast<float>(cell.getY() - 14),
                               static_cast<float>(cell.getBottom()));
        }
    }
}

void PerformView::resized()
{
    using I = UiState::Ids;
    const auto gap = gapFor(ctx.ui);
    auto b = getLocalBounds();

    const bool showKeys = ctx.ui.getBool(I::showKeyboard);
    const bool showVoice = ctx.ui.getBool(I::showVoice);
    const bool showStrip = ctx.ui.getBool(I::showFlowStrip);
    const bool showBlend = ctx.ui.getBool(I::showBlend);
    const bool showMacros = ctx.ui.getBool(I::showMacros);
    const bool showTimbre = ctx.ui.getBool(I::showTimbre);

    keyboardArea = showKeys ? b.removeFromBottom(84) : juce::Rectangle<int>();
    if (showKeys)
        b.removeFromBottom(gap);
    keyboard.setVisible(showKeys);
    keyboard.setBounds(keyboardArea);
    keyboard.setKeyWidth(static_cast<float>(keyboardArea.getWidth()) / 49.0f);

    stripArea = showStrip ? b.removeFromBottom(92) : juce::Rectangle<int>();
    if (showStrip)
        b.removeFromBottom(gap);
    flowStrip.setVisible(showStrip);
    flowStrip.setBounds(stripArea);

    voiceArea = showVoice ? b.removeFromBottom(150) : juce::Rectangle<int>();
    if (showVoice)
        b.removeFromBottom(gap);
    for (auto* v : voiceControls)
        v->setVisible(showVoice);
    if (showVoice)
    {
        auto inner = voiceArea.withTrimmedTop(kPanelTitleHeight).reduced(12, 6);
        meterArea = inner.removeFromRight(22);
        inner.removeFromRight(12);
        envelopeArea = inner.removeFromLeft(220).reduced(0, 14);
        inner.removeFromLeft(16);
        const auto w = inner.getWidth() / voiceControls.size();
        for (auto* v : voiceControls)
            v->setBounds(inner.removeFromLeft(w).reduced(4, 0));
    }
    else
    {
        meterArea = envelopeArea = {};
    }

    blendArea = showBlend ? b.removeFromLeft(360) : juce::Rectangle<int>();
    if (showBlend)
        b.removeFromLeft(gap);
    for (auto* c : blendControls)
        c->setVisible(showBlend);
    if (showBlend)
    {
        auto inner = blendArea.withTrimmedTop(kPanelTitleHeight + 4).reduced(20, 6);
        inner.removeFromRight(56);
        const auto rowH = inner.getHeight() / params::kNumMethods;
        for (auto* c : blendControls)
            c->setBounds(inner.removeFromTop(rowH).reduced(0, 3));
    }

    timbreArea = showTimbre
                     ? (showMacros ? b.removeFromRight(juce::jmin(b.getHeight() + 20, b.getWidth() / 2)) : b)
                     : juce::Rectangle<int>();
    if (showTimbre && showMacros)
        b.removeFromRight(gap);
    timbre.setVisible(showTimbre);
    timbre.setBounds(timbreArea);

    macroArea = showMacros ? b : juce::Rectangle<int>();
    macroPages.setVisible(showMacros);
    for (auto* m : macroControls)
        m->setVisible(showMacros);
    if (showMacros)
    {
        macroPages.setBounds(macroArea.getRight() - 12 - 108, macroArea.getY() + 5, 108, 22);
        auto inner = macroArea.withTrimmedTop(kPanelTitleHeight + 8).reduced(12, 8);
        const auto count = macroControls.size();
        const auto rows = count > 4 ? 2 : 1;
        const auto perRow = (count + rows - 1) / rows;
        const auto rowH = inner.getHeight() / rows;
        for (int r = 0; r < rows; ++r)
        {
            auto row = inner.removeFromTop(rowH);
            const auto w = row.getWidth() / perRow;
            for (int i = 0; i < perRow && r * perRow + i < count; ++i)
                macroControls[r * perRow + i]->setBounds(row.removeFromLeft(w).reduced(6, 4));
        }
    }
}

} // namespace alfa::ui
