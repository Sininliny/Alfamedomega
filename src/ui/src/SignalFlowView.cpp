#include "alfa/ui/Views.h"

namespace alfa::ui
{

const std::vector<SignalFlowView::Step>& SignalFlowView::steps()
{
    using namespace params;
    static const std::vector<Step> all{
        {"notes",
         "Notes in",
         "MIDI from the host or the on-screen keys",
         "note events: pitch, velocity, time",
         "audio thread",
         "Every key you press becomes a note event with a pitch and a velocity. "
         "Each note gets its own voice, up to "
         "the voice limit; when voices run out, the oldest one is reused.",
         {id::voices, id::glide, id::bendRange}},
        {"controls",
         "Controls",
         "knob, macro and timbre-space values",
         "a 16-number control vector per voice",
         "audio thread",
         "Everything you can turn is gathered into one short list of numbers, "
         "the control vector. This is the "
         "model's whole view of what you want: the knobs are its input, not the "
         "sound itself.",
         {id::timbreX, id::timbreY, id::macro(0), id::macro(1)}},
        {"model",
         "Neural model",
         "control vector + note",
         "control frames for each engine",
         "inference thread pool (anira)",
         "A shared trunk reads the control vector, then a few specialist "
         "experts, one per synthesis method, turn it "
         "into settings for the engines. Only the experts you are using run, "
         "which is how a large model stays "
         "affordable. Bigger model sizes have larger experts.",
         {id::modelTier, id::latentDrift, id::followModel}},
        {"blend",
         "Method blend",
         "per-engine control frames",
         "a mix weight for each of the nine engines",
         "audio thread",
         "The Method blend decides how much of each engine you hear. It is the "
         "same set of sliders as on the "
         "Perform view. An engine at zero costs nothing.",
         {id::blend(0), id::blend(3), id::blend(2), id::blend(6), id::blend(5)}},
        {"engines",
         "Sound engines",
         "control frames at the model rate",
         "audio samples",
         "audio thread",
         "Ordinary DSP engines (oscillators, filters, grains, resonators) make "
         "the actual sound, steered by the "
         "model's frames. The network never generates samples directly, which "
         "keeps CPU cost predictable.",
         {id::engine(0, 0), id::engine(0, 1), id::engine(3, 0), id::engine(2, 1)}},
        {"envfx",
         "Envelope + FX",
         "engine audio per voice",
         "shaped voice audio",
         "audio thread",
         "The amp envelope shapes each note over time: how fast it starts "
         "(attack), falls (decay), holds (sustain) "
         "and fades after you let go (release).",
         {id::attack, id::decay, id::sustain, id::release}},
        {"output",
         "Output",
         "all voices summed",
         "stereo audio to the host",
         "audio thread",
         "All voices are summed and sent to your DAW. The spectrum shows which "
         "frequencies are loudest right now.",
         {id::outputGain}},
    };
    return all;
}

namespace
{
juce::String summaryFor(const juce::String& stepId, const Snapshot& s, const EditorContext& ctx)
{
    using namespace params;
    const auto& tier = kTiers[static_cast<size_t>(ctx.selectedTier())];

    if (stepId == "notes")
        return s.heldNotes == 0
                   ? juce::String("No notes held")
                   : juce::String(s.heldNotes) + " held, " + juce::String(s.voices) + " voices sounding";
    if (stepId == "controls")
        return "Bright " + juce::String(juce::roundToInt(s.controlVector[0] * 100.0f)) + ", evolve " +
               juce::String(juce::roundToInt(s.controlVector[1] * 100.0f));
    if (stepId == "model")
    {
        int active = 0;
        for (auto e : s.experts)
            active += e > 0.0f ? 1 : 0;
        return juce::String(tier.name) + ", " + juce::String(active) + " of 9 experts active" +
               (s.modelIsStub ? " (stub)" : "");
    }
    if (stepId == "blend")
    {
        size_t top = 0;
        for (size_t m = 1; m < s.blend.size(); ++m)
            if (s.blend[m] > s.blend[top])
                top = m;
        return juce::String("Mostly ") + kMethods[top].name;
    }
    if (stepId == "engines")
    {
        int on = 0;
        for (auto b : s.blend)
            on += b > 0.001f ? 1 : 0;
        return juce::String(on) + " engines running";
    }
    if (stepId == "envfx")
        return "A " + juce::String(juce::roundToInt(s.attackMs)) + " ms, R " +
               juce::String(juce::roundToInt(s.releaseMs)) + " ms";
    const auto db = juce::Decibels::gainToDecibels(juce::jmax(s.peakL, s.peakR), -90.0f);
    return db <= -89.0f ? juce::String("Silent") : "Peak " + juce::String(db, 1) + " dB";
}

juce::String rateFor(const juce::String& stepId, const Snapshot& s)
{
    const auto khz = juce::String(s.sampleRate / 1000.0, 1) + " kHz";
    if (stepId == "notes")
        return "per event";
    if (stepId == "controls" || stepId == "model" || stepId == "blend")
        return juce::String(juce::roundToInt(params::kControlRateHz)) + " frames/s";
    return khz + ", " + juce::String(s.blockSize) + "-sample blocks";
}

juce::String costFor(const juce::String& stepId, const EditorContext& ctx)
{
    const auto& tier = params::kTiers[static_cast<size_t>(ctx.selectedTier())];
    if (stepId == "model")
        return juce::String(tier.latencyMs, 1) + " ms latency, " + juce::String(tier.budgetVoices) +
               " voices per reference core";
    if (stepId == "engines")
        return "stub engines until src/dsp lands";
    return "negligible";
}

void drawStepViz(juce::Graphics& g, const juce::String& id, juce::Rectangle<float> area, const Theme& t,
                 const LiveData& live, bool labels)
{
    const auto& s = live.snapshot();
    if (id == "notes")
        viz::pianoRoll(g, area, t, live);
    else if (id == "controls")
        viz::controlVector(g, area, t, s);
    else if (id == "model")
        viz::modelGraph(g, area, t, s, labels);
    else if (id == "blend")
        viz::blendBars(g, area, t, s, labels);
    else if (id == "engines")
        viz::waveform(g, area, t, s, t.accent);
    else if (id == "envfx")
        viz::envelope(g, area, t, s);
    else
        viz::spectrum(g, area, t, s);
}
} // namespace

SignalFlowView::SignalFlowView(EditorContext& c, LiveData& l)
    : ctx(c)
    , live(l)
    , detailLevel({"Musician", "Student", "Developer"},
                  [this](int i)
                  {
                      static const juce::StringArray ids{"musician", "student", "developer"};
                      ctx.ui.set(UiState::Ids::flowDetail, ids[i]);
                  })
{
    addAndMakeVisible(detailLevel);
    detailLevel.setSelected(
        juce::jmax(0, juce::StringArray{"musician", "student", "developer"}.indexOf(detail())));

    explain.setToggleState(ctx.ui.getBool(UiState::Ids::flowExplain), juce::dontSendNotification);
    explain.onClick = [this]
    {
        ctx.ui.set(UiState::Ids::flowExplain, explain.getToggleState());
    };
    addAndMakeVisible(explain);

    for (const auto& step : steps())
    {
        auto* chip = stepChips.add(new juce::TextButton(step.title));
        chip->setTooltip("Show or hide this step. Hidden steps still process the sound.");
        chip->setToggleState(!ctx.ui.isStepHidden(step.id), juce::dontSendNotification);
        chip->onClick = [this, chip, id = step.id]
        {
            ctx.ui.setStepHidden(id, !ctx.ui.isStepHidden(id));
        };
        addAndMakeVisible(chip);
    }

    rebuildInspector();
    live.addListener(this);
    ctx.ui.tree().addListener(this);
}

SignalFlowView::~SignalFlowView()
{
    live.removeListener(this);
    ctx.ui.tree().removeListener(this);
}

juce::String SignalFlowView::detail() const
{
    return ctx.ui.getString(UiState::Ids::flowDetail);
}

void SignalFlowView::liveDataUpdated()
{
    if (isShowing())
    {
        repaint(pipelineArea);
        repaint(inspectorViz);
    }
}

void SignalFlowView::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    using I = UiState::Ids;
    if (id == I::flowSelected)
        rebuildInspector();
    if (id == I::flowHidden || id == I::flowDetail || id == I::flowExplain || id == I::density)
    {
        explain.setToggleState(ctx.ui.getBool(I::flowExplain), juce::dontSendNotification);
        detailLevel.setSelected(
            juce::jmax(0, juce::StringArray{"musician", "student", "developer"}.indexOf(detail())));
        for (size_t i = 0; i < steps().size(); ++i)
            stepChips[static_cast<int>(i)]->setToggleState(!ctx.ui.isStepHidden(steps()[i].id),
                                                           juce::dontSendNotification);
        resized();
        repaint();
    }
}

void SignalFlowView::rebuildInspector()
{
    inspectorControls.clear();
    const auto selected =
        juce::jlimit(0, static_cast<int>(steps().size()) - 1, ctx.ui.getInt(UiState::Ids::flowSelected));
    for (const auto& id : steps()[static_cast<size_t>(selected)].paramIds)
    {
        // Full names here: "Subtractive Cutoff", not "Cutoff", since steps mix
        // parameters from several places.
        auto* control = inspectorControls.add(new ParamControl(ctx, id, ParamControl::Style::horizontalBar));
        control->setDisplayName(control->spec().name);
        control->setLabelWidth(150);
        addAndMakeVisible(control);
    }
    resized();
    repaint();
}

void SignalFlowView::mouseUp(const juce::MouseEvent& e)
{
    for (const auto& [index, area] : cardAreas)
        if (area.contains(e.getPosition()))
            ctx.ui.set(UiState::Ids::flowSelected, index);
}

void SignalFlowView::resized()
{
    const auto gap = gapFor(ctx.ui);
    auto b = getLocalBounds();

    toolbarArea = b.removeFromTop(34);
    {
        auto tb = toolbarArea;
        detailLevel.setBounds(tb.removeFromLeft(300).reduced(0, 3));
        tb.removeFromLeft(12);
        explain.setBounds(tb.removeFromLeft(170));
        tb.removeFromLeft(8);
        const auto chipW = tb.getWidth() / static_cast<int>(stepChips.size());
        for (auto* chip : stepChips)
            chip->setBounds(tb.removeFromLeft(chipW).reduced(2, 4));
    }
    b.removeFromTop(gap);

    const auto& all = steps();
    const bool student = detail() != "musician";
    const bool developer = detail() == "developer";
    pipelineArea = b.removeFromTop(developer ? 250 : (student ? 222 : 196));
    b.removeFromTop(gap);
    inspectorArea = b;

    cardAreas.clear();
    int hidden = 0;
    std::vector<int> visible;
    for (size_t i = 0; i < all.size(); ++i)
    {
        if (ctx.ui.isStepHidden(all[i].id))
            ++hidden;
        else
            visible.push_back(static_cast<int>(i));
    }
    hiddenText = hidden == 0 ? juce::String()
                             : juce::String(hidden) + (hidden == 1 ? " step hidden" : " steps hidden") +
                                   ". The sound still passes through " + (hidden == 1 ? "it." : "them.");

    auto row = pipelineArea.withTrimmedBottom(hiddenText.isEmpty() ? 0 : 18);
    const auto arrowW = 16;
    const auto n = static_cast<int>(visible.size());
    if (n > 0)
    {
        const auto cardW = (row.getWidth() - arrowW * (n - 1)) / n;
        for (int i = 0; i < n; ++i)
        {
            cardAreas.emplace_back(visible[static_cast<size_t>(i)], row.removeFromLeft(cardW));
            row.removeFromLeft(arrowW);
        }
    }

    auto inner = inspectorArea.withTrimmedTop(kPanelTitleHeight + 6).reduced(14, 8);
    inspectorViz = inner.removeFromLeft(inner.getWidth() * 2 / 5);
    inner.removeFromLeft(18);
    const bool showNotes = ctx.ui.getBool(UiState::Ids::flowExplain);
    inspectorNotes = showNotes ? inner.removeFromRight(inner.getWidth() * 9 / 20) : juce::Rectangle<int>();
    if (showNotes)
        inner.removeFromRight(18);
    inspectorParams = inner;

    auto p = inspectorParams.withTrimmedTop(20);
    for (auto* c : inspectorControls)
        c->setBounds(p.removeFromTop(30).reduced(0, 3));
}

void SignalFlowView::paintCard(juce::Graphics& g, const Step& step, juce::Rectangle<int> area, bool selected)
{
    const auto& t = ctx.theme();
    const auto& s = live.snapshot();
    const auto bounds = area.toFloat().reduced(0.5f);
    g.setColour(t.panel);
    g.fillRoundedRectangle(bounds, 8.0f);
    g.setColour(selected ? t.accent : t.line);
    g.drawRoundedRectangle(bounds, 8.0f, selected ? 2.0f : 1.0f);

    auto b = area.reduced(10, 8);
    const auto index = static_cast<int>(&step - steps().data()) + 1;
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::monoFont(11.0f));
    g.drawText(juce::String(index), b.removeFromTop(14), juce::Justification::centredLeft, false);
    g.setColour(t.text);
    g.setFont(AlfaLookAndFeel::labelFont(14.0f, true));
    g.drawText(step.title, b.removeFromTop(20), juce::Justification::centredLeft, true);

    const bool student = detail() != "musician";
    const bool developer = detail() == "developer";

    auto textRows = b.removeFromBottom(developer ? 88 : (student ? 60 : 32));
    b.removeFromBottom(6);
    drawStepViz(g, step.id, b.toFloat().reduced(0.0f, 4.0f), t, live, false);

    g.setColour(t.text);
    g.setFont(AlfaLookAndFeel::labelFont(12.0f));
    g.drawFittedText(summaryFor(step.id, s, ctx), textRows.removeFromTop(30), juce::Justification::topLeft,
                     2);

    g.setFont(AlfaLookAndFeel::labelFont(11.0f));
    g.setColour(t.text2);
    if (student)
    {
        g.drawFittedText("in: " + step.input, textRows.removeFromTop(14), juce::Justification::centredLeft,
                         1);
        g.drawFittedText("out: " + step.output, textRows.removeFromTop(14), juce::Justification::centredLeft,
                         1);
    }
    if (developer)
    {
        g.setFont(AlfaLookAndFeel::monoFont(10.5f));
        g.drawFittedText(step.thread, textRows.removeFromTop(14), juce::Justification::centredLeft, 1);
        g.drawFittedText(rateFor(step.id, s), textRows.removeFromTop(14), juce::Justification::centredLeft,
                         1);
    }
}

void SignalFlowView::paintInspector(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const auto& s = live.snapshot();
    const auto selected =
        juce::jlimit(0, static_cast<int>(steps().size()) - 1, ctx.ui.getInt(UiState::Ids::flowSelected));
    const auto& step = steps()[static_cast<size_t>(selected)];

    drawPanel(g, t, inspectorArea, "Step " + juce::String(selected + 1) + ": " + step.title,
              ctx.ui.isStepHidden(step.id) ? "hidden from the strip" : "");

    g.setColour(t.bg);
    g.fillRoundedRectangle(inspectorViz.toFloat(), 6.0f);
    drawStepViz(g, step.id, inspectorViz.toFloat().reduced(12.0f), t, live, true);

    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.5f, true));
    g.drawText("KEY PARAMETERS", inspectorParams.withHeight(16), juce::Justification::centredLeft, false);

    // Data flow and cost lines under the parameters, by detail level.
    auto info = inspectorParams.withTrimmedTop(24 + 30 * static_cast<int>(step.paramIds.size()) + 10);
    g.setFont(AlfaLookAndFeel::labelFont(12.0f));
    if (detail() != "musician")
    {
        g.setColour(t.text2);
        g.drawFittedText("Takes " + step.input + ".", info.removeFromTop(18),
                         juce::Justification::centredLeft, 1);
        g.drawFittedText("Produces " + step.output + ".", info.removeFromTop(18),
                         juce::Justification::centredLeft, 1);
    }
    if (detail() == "developer")
    {
        g.setFont(AlfaLookAndFeel::monoFont(11.5f));
        g.setColour(t.text2);
        g.drawFittedText("thread: " + step.thread, info.removeFromTop(18), juce::Justification::centredLeft,
                         1);
        g.drawFittedText("rate:   " + rateFor(step.id, s), info.removeFromTop(18),
                         juce::Justification::centredLeft, 1);
        g.drawFittedText("cost:   " + costFor(step.id, ctx), info.removeFromTop(18),
                         juce::Justification::centredLeft, 1);
        if (step.id == "engines" || step.id == "model")
            g.drawFittedText("impl:   " + ctx.host.backendDescription(), info.removeFromTop(18),
                             juce::Justification::centredLeft, 1);
    }

    if (!inspectorNotes.isEmpty())
    {
        // The explanatory note, set as a quotation: an accent rule, open and close
        // quote marks, and the text in between.
        auto q = inspectorNotes;
        g.setColour(t.accent);
        g.fillRect(q.removeFromLeft(3));
        q.removeFromLeft(12);
        g.setFont(AlfaLookAndFeel::labelFont(34.0f, true));
        g.drawText(juce::CharPointer_UTF8("\xe2\x80\x9c"), q.removeFromTop(30), juce::Justification::topLeft,
                   false);
        g.setColour(t.text);
        g.setFont(AlfaLookAndFeel::labelFont(14.0f));
        juce::AttributedString text;
        text.append(step.note, AlfaLookAndFeel::labelFont(14.0f), t.text);
        text.setLineSpacing(4.0f);
        juce::TextLayout layout;
        layout.createLayout(text, static_cast<float>(q.getWidth()));
        const auto textArea = q.removeFromTop(juce::roundToInt(layout.getHeight()) + 4);
        layout.draw(g, textArea.toFloat());
        g.setColour(t.accent);
        g.setFont(AlfaLookAndFeel::labelFont(34.0f, true));
        g.drawText(juce::CharPointer_UTF8("\xe2\x80\x9d"), q.removeFromTop(30), juce::Justification::topRight,
                   false);
    }
}

void SignalFlowView::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    const auto selected = ctx.ui.getInt(UiState::Ids::flowSelected);

    for (size_t i = 0; i < cardAreas.size(); ++i)
    {
        const auto& [index, area] = cardAreas[i];
        paintCard(g, steps()[static_cast<size_t>(index)], area, index == selected);

        if (i + 1 < cardAreas.size())
        {
            const auto y = static_cast<float>(area.getCentreY());
            const auto x0 = static_cast<float>(area.getRight()) + 3.0f;
            const auto x1 = static_cast<float>(cardAreas[i + 1].second.getX()) - 3.0f;
            juce::Path arrow;
            arrow.addArrow({x0, y, x1, y}, 1.5f, 8.0f, 6.0f);
            g.setColour(t.text2);
            g.fillPath(arrow);
        }
    }

    if (hiddenText.isNotEmpty())
    {
        g.setColour(t.text2);
        g.setFont(AlfaLookAndFeel::labelFont(12.0f));
        g.drawText(hiddenText, pipelineArea.removeFromBottom(16), juce::Justification::centredLeft, false);
    }

    paintInspector(g);
}

} // namespace alfa::ui
