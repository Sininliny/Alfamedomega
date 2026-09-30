#include "alfa/ui/Views.h"

namespace alfa::ui
{

std::vector<int> usedKnobs(const juce::AudioProcessorValueTreeState& state)
{
    const auto shape = params::readShape(state);
    std::vector<int> out;
    for (int i = 0; i < params::kNumKnobs; ++i)
        if (params::knobInfoFor(shape.method, shape.hybridA, shape.hybridB, i).used)
            out.push_back(i);
    return out;
}

std::array<int, 4> envelopeKnobs(const juce::AudioProcessorValueTreeState& state)
{
    const auto shape = params::readShape(state);
    const auto m = shape.method == params::Method::hybrid ? shape.hybridA : shape.method;
    if (m == params::Method::physical)
        return {2, 3, 4, 7}; // decay, high damping, release damping, bowed
    return {18, 19, 20, 21};
}

// ShapeWatcher
// ------------------------------------------------------------------------

namespace
{
const juce::StringArray nearbyModeIds{"my_taste", "model", "community", "wild"};

const juce::String* shapeIds()
{
    static const juce::String ids[] = {params::id::method, params::id::hybridA, params::id::hybridB};
    return ids;
}
} // namespace

ShapeWatcher::ShapeWatcher(juce::AudioProcessorValueTreeState& p, std::function<void()> onChange)
    : params(p)
    , callback(std::move(onChange))
{
    for (int i = 0; i < 3; ++i)
        params.addParameterListener(shapeIds()[i], this);
}

ShapeWatcher::~ShapeWatcher()
{
    *alive = false;
    for (int i = 0; i < 3; ++i)
        params.removeParameterListener(shapeIds()[i], this);
}

void ShapeWatcher::parameterChanged(const juce::String&, float)
{
    // Parameter callbacks can arrive on the audio thread (automation); rebuild on
    // the message thread, and only while this watcher still exists.
    juce::MessageManager::callAsync(
        [this, weak = std::weak_ptr<bool>(alive)]
        {
            if (auto a = weak.lock(); a && *a && callback)
                callback();
        });
}

// MethodPicker
// ------------------------------------------------------------------------

MethodPicker::MethodPicker(EditorContext& c, LiveData& l, juce::String panelTitle)
    : ctx(c)
    , live(l)
    , title(std::move(panelTitle))
    , watcher(c.params,
              [this]
              {
                  resized();
                  repaint();
              })
{
    for (auto* box : {&hybridA, &hybridB})
    {
        for (int i = 0; i < static_cast<int>(params::kHybridComponents.size()); ++i)
            box->addItem(
                params::kMethods[static_cast<size_t>(params::kHybridComponents[static_cast<size_t>(i)])].name,
                i + 1);
        addChildComponent(box);
    }
    hybridA.setTooltip("Hybrid component A: plays on the knobs as they are");
    hybridB.setTooltip("Hybrid component B: plays on the knobs shifted by 11; knob 32 crossfades A to B");
    attachA = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        ctx.params, params::id::hybridA, hybridA);
    attachB = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        ctx.params, params::id::hybridB, hybridB);
    setTitle(title);
    setTooltip("Pick the sound's synthesis method. The bar is the method's share of model capacity.");
    live.addListener(this);
}

MethodPicker::~MethodPicker()
{
    live.removeListener(this);
}

void MethodPicker::setShowSummaries(bool show)
{
    summaries = show;
    resized();
}

void MethodPicker::liveDataUpdated()
{
    if (isShowing())
        repaint();
}

void MethodPicker::select(int method)
{
    if (auto* p = ctx.params.getParameter(params::id::method))
    {
        p->beginChangeGesture();
        p->setValueNotifyingHost(p->convertTo0to1(static_cast<float>(method)));
        p->endChangeGesture();
    }
}

int MethodPicker::rowAt(juce::Point<int> pos) const
{
    for (size_t i = 0; i < rows.size(); ++i)
        if (rows[i].contains(pos))
            return static_cast<int>(i);
    return -1;
}

void MethodPicker::mouseUp(const juce::MouseEvent& e)
{
    const auto row = rowAt(e.getPosition());
    if (row >= 0)
        select(row);
}

void MethodPicker::mouseMove(const juce::MouseEvent& e)
{
    const auto row = rowAt(e.getPosition());
    if (row != hover)
    {
        hover = row;
        repaint();
    }
}

void MethodPicker::mouseExit(const juce::MouseEvent&)
{
    hover = -1;
    repaint();
}

void MethodPicker::resized()
{
    const bool hybrid = params::readShape(ctx.params).method == params::Method::hybrid;
    auto inner = getLocalBounds().withTrimmedTop(kPanelTitleHeight + 4).reduced(8, 6);

    hybridArea = hybrid ? inner.removeFromBottom(58) : juce::Rectangle<int>();
    hybridA.setVisible(hybrid);
    hybridB.setVisible(hybrid);
    if (hybrid)
    {
        auto boxes = hybridArea.withTrimmedTop(6).removeFromTop(26);
        const auto w = (boxes.getWidth() - 8) / 2;
        hybridA.setBounds(boxes.removeFromLeft(w).withTrimmedLeft(20));
        boxes.removeFromLeft(8);
        hybridB.setBounds(boxes.withTrimmedLeft(20));
    }

    rows.clear();
    const auto rowH = inner.getHeight() / params::kNumMethods;
    for (int i = 0; i < params::kNumMethods; ++i)
        rows.push_back(inner.removeFromTop(rowH).reduced(0, 1));
}

void MethodPicker::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const auto& s = live.snapshot();
    const auto shape = params::readShape(ctx.params);
    drawPanel(g, t, getLocalBounds(), title, "capacity");

    for (size_t i = 0; i < rows.size(); ++i)
    {
        const auto m = static_cast<params::Method>(i);
        const auto& info = params::kMethods[i];
        const bool selected = m == shape.method;
        const bool component =
            shape.method == params::Method::hybrid && (m == shape.hybridA || m == shape.hybridB);
        auto row = rows[i];

        if (selected || static_cast<int>(i) == hover)
        {
            g.setColour(t.accent.withAlpha(selected ? 0.16f : 0.07f));
            g.fillRoundedRectangle(row.toFloat(), 5.0f);
        }
        if (selected)
        {
            g.setColour(t.accent);
            g.fillRect(row.withWidth(3));
        }

        auto r = row.reduced(12, 0);
        // Live activity: the method's expert is running.
        const auto activity = s.experts[i];
        g.setColour(activity > 0.0f ? t.accent.withAlpha(0.4f + 0.6f * juce::jmin(1.0f, activity)) : t.line);
        g.fillEllipse(
            juce::Rectangle<float>(6.0f, 6.0f)
                .withCentre({static_cast<float>(r.getX() + 3), static_cast<float>(r.getCentreY())}));
        r.removeFromLeft(14);

        const bool twoLines = summaries && row.getHeight() >= 34;
        auto nameRow = twoLines ? r.removeFromTop(r.getHeight() / 2).removeFromBottom(18) : r;

        auto cap = nameRow.removeFromRight(64);
        const auto share = info.capacityShare;
        auto bar = cap.withTrimmedRight(30).toFloat().withSizeKeepingCentre(
            static_cast<float>(cap.getWidth() - 30), 4.0f);
        g.setColour(t.line);
        g.fillRoundedRectangle(bar, 2.0f);
        g.setColour(t.text2);
        g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * share / 0.3f), 2.0f);
        g.setFont(AlfaLookAndFeel::monoFont(10.5f));
        g.drawText(juce::String(juce::roundToInt(share * 100.0f)) + "%", cap.removeFromRight(28),
                   juce::Justification::centredRight, false);

        g.setColour(selected || component ? t.text : t.text2);
        g.setFont(AlfaLookAndFeel::labelFont(13.0f, selected));
        auto label = juce::String(info.name);
        if (component)
            label << (m == shape.hybridA ? "  (A)" : "  (B)");
        g.drawText(label, nameRow,
                   twoLines ? juce::Justification::bottomLeft : juce::Justification::centredLeft, true);
        if (twoLines)
        {
            g.setColour(t.text2);
            g.setFont(AlfaLookAndFeel::labelFont(11.0f));
            g.drawText(info.summary, r, juce::Justification::topLeft, true);
        }
    }

    if (!hybridArea.isEmpty())
    {
        g.setColour(t.text2);
        g.setFont(AlfaLookAndFeel::labelFont(11.5f, true));
        auto boxes = hybridArea.withTrimmedTop(6).removeFromTop(26);
        const auto w = (boxes.getWidth() - 8) / 2;
        g.drawText("A", boxes.removeFromLeft(w).withWidth(18), juce::Justification::centredLeft, false);
        boxes.removeFromLeft(8);
        g.drawText("B", boxes.withWidth(18), juce::Justification::centredLeft, false);

        const bool ok = params::isValidHybridPair(shape.hybridA, shape.hybridB);
        g.setColour(ok ? t.text2 : t.warn);
        g.setFont(AlfaLookAndFeel::labelFont(11.5f));
        g.drawText(ok ? "Knob 32 crossfades A to B"
                      : (shape.hybridA == shape.hybridB ? "Pick two different methods: the hybrid is silent"
                                                        : "Additive and physical can't be combined: silent"),
                   hybridArea.withTrimmedTop(36), juce::Justification::centredLeft, true);
    }
}

// TimbrePad
// ------------------------------------------------------------------------

TimbrePad::TimbrePad(EditorContext& c)
    : ctx(c)
    , watcher(c.params,
              [this]
              {
                  bind();
                  repaint();
              })
{
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
    randomize.setTooltip("Nudge every knob this method uses a little, to find a neighbouring sound");
    randomize.onClick = [this]
    {
        randomizeNearby();
    };
    // design.md section 6.5: My Taste is the default.
    nearbyMode.addItemList({"My Taste", "Model Space", "Community Space", "Wild"}, 1);
    nearbyMode.setTooltip("My Taste leans toward sounds you have liked. Model Space steps evenly around this "
                          "sound. Community Space follows a shared taste profile. Wild takes bigger steps.");
    nearbyMode.setSelectedItemIndex(
        juce::jmax(0, nearbyModeIds.indexOf(ctx.ui.getString(UiState::Ids::nearbyMode))),
        juce::dontSendNotification);
    nearbyMode.onChange = [this]
    {
        ctx.ui.set(UiState::Ids::nearbyMode, nearbyModeIds[juce::jmax(0, nearbyMode.getSelectedItemIndex())]);
        nearbyNote = {};
        repaint();
    };
    addAndMakeVisible(nearbyMode);
    addAndMakeVisible(snapA);
    addAndMakeVisible(snapB);
    addAndMakeVisible(randomize);
    setWantsKeyboardFocus(true);
    setTitle("Timbre space");
    bind();
}

TimbrePad::~TimbrePad()
{
    if (xParam != nullptr)
    {
        ctx.params.removeParameterListener(params::id::knob(xKnob), this);
        ctx.params.removeParameterListener(params::id::knob(yKnob), this);
    }
}

void TimbrePad::bind()
{
    // The pad drives the two knobs that change this method's character most.
    if (xParam != nullptr)
    {
        ctx.params.removeParameterListener(params::id::knob(xKnob), this);
        ctx.params.removeParameterListener(params::id::knob(yKnob), this);
    }
    const auto shape = params::readShape(ctx.params);
    std::tie(xKnob, yKnob) = params::timbreKnobsFor(shape.method, shape.hybridA);
    xParam = ctx.params.getParameter(params::id::knob(xKnob));
    yParam = ctx.params.getParameter(params::id::knob(yKnob));
    xInfo = params::knobInfoFor(shape.method, shape.hybridA, shape.hybridB, xKnob);
    yInfo = params::knobInfoFor(shape.method, shape.hybridA, shape.hybridB, yKnob);
    ctx.params.addParameterListener(params::id::knob(xKnob), this);
    ctx.params.addParameterListener(params::id::knob(yKnob), this);
    snapshots = {juce::Point<float>{-1.0f, -1.0f}, {-1.0f, -1.0f}};
    setTooltip("Drag to set " + xInfo.name.toLowerCase() + " (across) and " + yInfo.name.toLowerCase() +
               " (up)");
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
        .withTrimmedBottom(58)
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
    // A Gaussian step on every knob the method uses, as the preference
    // learner's "randomize nearby" does (preference/alfapref/generate.py,
    // design.md section 6.5). Scoring neighbours by taste needs the learner's
    // scorer, which the plugin does not have yet, so My Taste and Community
    // Space take Model Space's unscored step and say so.
    const auto mode = nearbyModeIds[juce::jmax(0, nearbyMode.getSelectedItemIndex())];
    const float sigma = mode == "wild" ? 0.25f : 0.08f;
    if (mode == "my_taste")
        nearbyNote = "My Taste isn't in the plugin yet: used Model Space";
    else if (mode == "community")
        nearbyNote = "No community profile yet: used Model Space";
    else
        nearbyNote = {};
    repaint();

    auto& rng = juce::Random::getSystemRandom();
    for (auto i : usedKnobs(ctx.params))
    {
        auto* p = ctx.params.getParameter(params::id::knob(i));
        // Sum of three uniforms: roughly Gaussian with standard deviation 0.5.
        const auto gauss =
            static_cast<float>(rng.nextDouble() + rng.nextDouble() + rng.nextDouble() - 1.5) * 2.0f;
        auto v = p->getValue() + sigma * gauss;
        v = v < 0.0f ? -v : (v > 1.0f ? 2.0f - v : v); // reflect into 0..1
        p->beginChangeGesture();
        p->setValueNotifyingHost(juce::jlimit(0.0f, 1.0f, v));
        p->endChangeGesture();
    }
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
    g.drawText(xInfo.name + "  " + juce::String::charToString(0x2192), area.reduced(6.0f),
               juce::Justification::bottomRight, false);
    g.drawText(juce::String::charToString(0x2191) + "  " + yInfo.name, area.reduced(6.0f),
               juce::Justification::topLeft, false);

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
    const auto readout = xInfo.shortName + " " + params::formatKnob(xInfo, xParam->getValue()) + "   " +
                         yInfo.shortName + " " + params::formatKnob(yInfo, yParam->getValue());
    g.drawText(readout, getLocalBounds().reduced(12, 6).removeFromTop(kPanelTitleHeight - 6),
               juce::Justification::centredRight, false);

    if (nearbyNote.isNotEmpty())
    {
        g.setColour(t.text2);
        g.setFont(AlfaLookAndFeel::labelFont(11.5f));
        g.drawText(nearbyNote, getLocalBounds().reduced(14, 8).withTrimmedBottom(32).removeFromBottom(16),
                   juce::Justification::centredRight, true);
    }
}

void TimbrePad::resized()
{
    auto row = getLocalBounds().reduced(14, 8).removeFromBottom(28);
    snapA.setBounds(row.removeFromLeft(36));
    row.removeFromLeft(6);
    snapB.setBounds(row.removeFromLeft(36));
    row.removeFromLeft(6);
    randomize.setBounds(row.removeFromRight(juce::jmin(140, row.getWidth())));
    row.removeFromRight(6);
    nearbyMode.setBounds(row.removeFromRight(juce::jmin(140, row.getWidth())));
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
    , methods(c, l, "Method")
    , timbre(c)
    , keyboard(c.keyboard, juce::MidiKeyboardComponent::horizontalKeyboard)
    , flowStrip(c, l)
    , watcher(c.params,
              [this]
              {
                  ctx.ui.set(UiState::Ids::macroPage, 0);
                  rebuildKnobs();
                  rebuildVoice();
              })
{
    addAndMakeVisible(methods);
    addAndMakeVisible(timbre);

    keyboard.setAvailableRange(24, 108);
    keyboard.setLowestVisibleKey(36);
    keyboard.setScrollButtonsVisible(false);
    keyboard.setOctaveForMiddleC(4);
    keyboard.setWantsKeyboardFocus(false);
    addAndMakeVisible(keyboard);
    addChildComponent(flowStrip);

    rebuildVoice();
    rebuildKnobs();
    live.addListener(this);
    ctx.ui.tree().addListener(this);
}

PerformView::~PerformView()
{
    live.removeListener(this);
    ctx.ui.tree().removeListener(this);
}

void PerformView::rebuildKnobs()
{
    // Only the knobs the current method uses, paged by the "per page" setting.
    knobControls.clear();
    const auto used = usedKnobs(ctx.params);
    const auto perPage = juce::jlimit(4, 12, ctx.ui.getInt(UiState::Ids::macrosPerPage));
    const auto pages = juce::jmax(1, (static_cast<int>(used.size()) + perPage - 1) / perPage);
    const auto page = juce::jlimit(0, pages - 1, ctx.ui.getInt(UiState::Ids::macroPage));

    juce::StringArray labels;
    for (int i = 0; i < pages; ++i)
        labels.add(juce::String(i + 1));
    // Deferred: changing the page rebuilds this Segmented, which must not happen
    // inside its own click handler.
    knobPages = std::make_unique<Segmented>(labels,
                                            [safe = juce::Component::SafePointer<PerformView>(this)](int i)
                                            {
                                                juce::MessageManager::callAsync(
                                                    [safe, i]
                                                    {
                                                        if (safe != nullptr)
                                                            safe->ctx.ui.set(UiState::Ids::macroPage, i);
                                                    });
                                            });
    knobPages->setSelected(page);
    addChildComponent(*knobPages);

    for (int i = page * perPage; i < juce::jmin(static_cast<int>(used.size()), (page + 1) * perPage); ++i)
        addChildComponent(
            knobControls.add(new ParamControl(ctx, params::id::knob(used[static_cast<size_t>(i)]))));

    const auto shape = params::readShape(ctx.params);
    knobsSubtitle = juce::String(params::kMethods[static_cast<size_t>(shape.method)].name) + ", " +
                    juce::String(used.size()) + " knobs";
    resized();
    repaint();
}

void PerformView::rebuildVoice()
{
    voiceControls.clear();
    const auto env = envelopeKnobs(ctx.params);
    for (auto k : env)
        addChildComponent(
            voiceControls.add(new ParamControl(ctx, params::id::knob(k), ParamControl::Style::smallKnob)));
    for (const auto& id :
         {params::id::voices, params::id::glide, params::id::bendRange, params::id::outputGain})
        addChildComponent(voiceControls.add(new ParamControl(ctx, id, ParamControl::Style::smallKnob)));
    resized();
}

void PerformView::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    using I = UiState::Ids;
    if (id == I::macrosPerPage || id == I::macroPage)
        rebuildKnobs();
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
    }
}

void PerformView::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const auto& s = live.snapshot();

    if (!knobArea.isEmpty())
        drawPanel(g, t, knobArea, "Knobs", {});

    if (!knobArea.isEmpty() && knobPages != nullptr)
    {
        g.setColour(t.text2);
        g.setFont(AlfaLookAndFeel::labelFont(11.5f));
        g.drawText(knobsSubtitle, knobArea.reduced(12, 0).withHeight(kPanelTitleHeight).withTrimmedTop(6),
                   juce::Justification::centred, false);
    }

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

    const juce::StringArray names{"Notes", "Controls", "Model", "Method", "Engines", "Env + FX", "Output"};
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
            viz::methodBars(g, area, t, s, false);
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
    const bool showMethod = ctx.ui.getBool(I::showBlend);
    const bool showKnobs = ctx.ui.getBool(I::showMacros);
    const bool showTimbre = ctx.ui.getBool(I::showTimbre);

    keyboardArea = showKeys ? b.removeFromBottom(84) : juce::Rectangle<int>();
    if (showKeys)
        b.removeFromBottom(gap);
    keyboard.setVisible(showKeys);
    keyboard.setBounds(keyboardArea);
    if (keyboardArea.getWidth() > 0)
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
    if (showVoice && !voiceControls.isEmpty())
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

    methodArea = showMethod ? b.removeFromLeft(300) : juce::Rectangle<int>();
    if (showMethod)
        b.removeFromLeft(gap);
    methods.setVisible(showMethod);
    methods.setBounds(methodArea);

    timbreArea = showTimbre
                     ? (showKnobs ? b.removeFromRight(juce::jmin(b.getHeight() + 20, b.getWidth() / 2)) : b)
                     : juce::Rectangle<int>();
    if (showTimbre && showKnobs)
        b.removeFromRight(gap);
    timbre.setVisible(showTimbre);
    timbre.setBounds(timbreArea);

    knobArea = showKnobs ? b : juce::Rectangle<int>();
    if (knobPages != nullptr)
        knobPages->setVisible(showKnobs);
    for (auto* m : knobControls)
        m->setVisible(showKnobs);
    if (showKnobs)
    {
        if (knobPages != nullptr)
        {
            const auto w = juce::jmin(160, 28 * knobPages->getNumChildComponents());
            knobPages->setBounds(knobArea.getRight() - 12 - w, knobArea.getY() + 5, w, 22);
        }
        auto inner = knobArea.withTrimmedTop(kPanelTitleHeight + 8).reduced(12, 8);
        const auto perPage = juce::jlimit(4, 12, ctx.ui.getInt(UiState::Ids::macrosPerPage));
        const auto rows = perPage > 4 ? 2 : 1;
        const auto perRow = (perPage + rows - 1) / rows;
        const auto rowH = inner.getHeight() / rows;
        const auto count = knobControls.size();
        for (int r = 0; r < rows; ++r)
        {
            auto row = inner.removeFromTop(rowH);
            const auto w = row.getWidth() / perRow;
            for (int i = 0; i < perRow && r * perRow + i < count; ++i)
                knobControls[r * perRow + i]->setBounds(row.removeFromLeft(w).reduced(6, 4));
        }
    }
}

} // namespace alfa::ui
