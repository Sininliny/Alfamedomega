#include "alfa/ui/LiveData.h"

namespace alfa::ui
{

LiveData::LiveData(EditorContext& context)
    : ctx(context)
    , history(kHistoryLength)
{
    update();
    startTimerHz(30);
}

LiveData::~LiveData()
{
    stopTimer();
}

void LiveData::update()
{
    const auto& t = ctx.telemetry;
    auto raw = [this](const juce::String& id)
    {
        return ctx.params.getRawParameterValue(id)->load();
    };

    int held = 0;
    std::array<bool, 128> frame{};
    for (size_t n = 0; n < snap.notes.size(); ++n)
    {
        snap.notes[n] = t.noteVelocity[n].load(std::memory_order_relaxed);
        // Notes played on the on-screen keyboard count too.
        if (snap.notes[n] <= 0.0f && ctx.keyboard.isNoteOn(1, static_cast<int>(n)))
            snap.notes[n] = 0.8f;
        frame[n] = snap.notes[n] > 0.0f;
        held += frame[n] ? 1 : 0;
    }
    snap.heldNotes = held;
    std::rotate(history.begin(), history.begin() + 1, history.end());
    history.back() = frame;

    for (size_t i = 0; i < snap.controlVector.size(); ++i)
        snap.controlVector[i] = t.controlVector[i].load(std::memory_order_relaxed);

    for (size_t m = 0; m < static_cast<size_t>(params::kNumMethods); ++m)
    {
        snap.experts[m] = t.expertActivity[m].load(std::memory_order_relaxed);
        snap.engines[m] = t.engineLevel[m].load(std::memory_order_relaxed);
        snap.blend[m] = raw(params::id::blend(static_cast<int>(m)));
    }

    snap.attackMs = raw(params::id::attack);
    snap.decayMs = raw(params::id::decay);
    snap.sustain = raw(params::id::sustain);
    snap.releaseMs = raw(params::id::release);

    // Meters fall back smoothly rather than jumping to zero between blocks.
    snap.peakL = juce::jmax(t.peakL.load(), snap.peakL * 0.8f);
    snap.peakR = juce::jmax(t.peakR.load(), snap.peakR * 0.8f);
    snap.cpu = t.cpuLoad.load();
    snap.voices = t.activeVoices.load();
    snap.loadedTier = t.loadedTier.load();
    snap.modelIsStub = t.modelIsStub.load();
    snap.sampleRate = t.sampleRate.load();
    snap.blockSize = t.blockSize.load();

    std::array<float, 2048> samples{};
    t.readScope(samples.data(), static_cast<int>(samples.size()));

    // Scope: the last few cycles, decimated to the point count.
    const auto step = 2;
    const auto start = samples.size() - static_cast<size_t>(Snapshot::kScopePoints * step);
    for (size_t i = 0; i < snap.scope.size(); ++i)
        snap.scope[i] = samples[start + i * step];

    computeSpectrum(samples);

    listeners.call([](Listener& l) { l.liveDataUpdated(); });
}

void LiveData::computeSpectrum(const std::array<float, 2048>& samples)
{
    std::array<float, 4096> data{};
    std::copy(samples.begin(), samples.end(), data.begin());
    window.multiplyWithWindowingTable(data.data(), 2048);
    fft.performFrequencyOnlyForwardTransform(data.data());

    // Log-spaced bands from 30 Hz to 20 kHz.
    const auto sr = static_cast<float>(snap.sampleRate);
    for (int b = 0; b < Snapshot::kSpectrumBands; ++b)
    {
        const auto f0 = 30.0f * std::pow(20000.0f / 30.0f, static_cast<float>(b) / Snapshot::kSpectrumBands);
        const auto f1 =
            30.0f * std::pow(20000.0f / 30.0f, static_cast<float>(b + 1) / Snapshot::kSpectrumBands);
        const auto i0 = juce::jlimit(1, 1023, static_cast<int>(f0 / sr * 2048.0f));
        const auto i1 = juce::jlimit(i0 + 1, 1024, static_cast<int>(f1 / sr * 2048.0f) + 1);
        float peak = 0.0f;
        for (int i = i0; i < i1; ++i)
            peak = juce::jmax(peak, data[static_cast<size_t>(i)]);
        const auto db = juce::Decibels::gainToDecibels(peak / 512.0f, -90.0f);
        auto& slot = snap.spectrum[static_cast<size_t>(b)];
        slot = juce::jmax(db, slot - 3.0f); // fall at 90 dB/s
    }
}

// Visualisations
// -----------------------------------------------------------------

namespace viz
{
void pianoRoll(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, const LiveData& data)
{
    const auto& hist = data.noteHistory();
    int lo = 127, hi = 0;
    for (const auto& frame : hist)
        for (int n = 0; n < 128; ++n)
            if (frame[static_cast<size_t>(n)])
            {
                lo = juce::jmin(lo, n);
                hi = juce::jmax(hi, n);
            }

    if (lo > hi)
    {
        lo = 48;
        hi = 72;
    }
    lo = juce::jmax(0, lo - 4);
    hi = juce::jmin(127, juce::jmax(hi + 4, lo + 16));

    const auto rows = static_cast<float>(hi - lo + 1);
    const auto colW = area.getWidth() / static_cast<float>(hist.size());
    const auto rowH = area.getHeight() / rows;

    g.setColour(t.line.withAlpha(0.5f));
    for (int n = lo; n <= hi; ++n)
        if (n % 12 == 0)
            g.fillRect(area.getX(), area.getBottom() - static_cast<float>(n - lo + 1) * rowH, area.getWidth(),
                       1.0f);

    g.setColour(t.accent);
    for (size_t x = 0; x < hist.size(); ++x)
        for (int n = lo; n <= hi; ++n)
            if (hist[x][static_cast<size_t>(n)])
                g.fillRect(area.getX() + static_cast<float>(x) * colW,
                           area.getBottom() - static_cast<float>(n - lo + 1) * rowH, colW + 0.5f,
                           juce::jmax(2.0f, rowH - 1.0f));
}

void controlVector(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, const Snapshot& s)
{
    const auto n = static_cast<float>(s.controlVector.size());
    const auto w = area.getWidth() / n;
    for (size_t i = 0; i < s.controlVector.size(); ++i)
    {
        auto bar =
            juce::Rectangle<float>(area.getX() + static_cast<float>(i) * w, area.getY(), w, area.getHeight())
                .reduced(w * 0.15f, 0.0f);
        g.setColour(t.line);
        g.fillRoundedRectangle(bar, 1.5f);
        g.setColour(t.accent.withAlpha(0.85f));
        g.fillRoundedRectangle(bar.withTop(bar.getBottom() - bar.getHeight() * s.controlVector[i]), 1.5f);
    }
}

void modelGraph(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, const Snapshot& s,
                bool showLabels)
{
    // Trunk (shared by every engine) on the left, one expert per engine on the
    // right, each drawn with height proportional to its share of capacity.
    auto trunk = area.removeFromLeft(area.getWidth() * 0.28f).reduced(2.0f);
    g.setColour(t.text2.withAlpha(0.25f));
    g.fillRoundedRectangle(trunk, 3.0f);
    g.setColour(t.text2);
    g.drawRoundedRectangle(trunk, 3.0f, 1.0f);
    if (showLabels)
    {
        g.setFont(AlfaLookAndFeel::labelFont(10.5f));
        g.drawText("trunk", trunk, juce::Justification::centred, false);
    }

    area.removeFromLeft(8.0f);
    auto y = area.getY();
    const auto total = area.getHeight() - 2.0f * static_cast<float>(params::kNumMethods - 1);
    for (size_t m = 0; m < static_cast<size_t>(params::kNumMethods); ++m)
    {
        const auto h = total * params::kMethods[m].capacityShare;
        auto box = juce::Rectangle<float>(area.getX(), y, area.getWidth(), h);
        y += h + 2.0f;

        const auto activity = s.experts[m];
        g.setColour(activity > 0.0f ? t.accent.withAlpha(0.25f + 0.75f * activity) : t.line);
        g.fillRoundedRectangle(box, 2.0f);

        // Wires from trunk to the active experts
        if (activity > 0.0f)
        {
            g.setColour(t.accent.withAlpha(0.5f));
            g.drawLine(trunk.getRight(), trunk.getCentreY(), box.getX(), box.getCentreY(), 1.0f);
        }

        if (showLabels && h >= 10.0f)
        {
            g.setColour(activity > 0.0f ? t.bg : t.text2);
            g.setFont(AlfaLookAndFeel::labelFont(juce::jmin(11.0f, h - 1.0f)));
            g.drawText(params::kMethods[m].name, box.reduced(4.0f, 0.0f), juce::Justification::centredLeft,
                       false);
        }
    }
}

void blendBars(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, const Snapshot& s,
               bool showLabels)
{
    const auto rowH = area.getHeight() / static_cast<float>(params::kNumMethods);
    for (size_t m = 0; m < static_cast<size_t>(params::kNumMethods); ++m)
    {
        auto row = juce::Rectangle<float>(area.getX(), area.getY() + static_cast<float>(m) * rowH,
                                          area.getWidth(), rowH)
                       .reduced(0.0f, rowH * 0.15f);
        if (showLabels)
        {
            g.setColour(t.text2);
            g.setFont(AlfaLookAndFeel::labelFont(juce::jmin(11.0f, rowH)));
            g.drawText(params::kMethods[m].name, row.removeFromLeft(70.0f), juce::Justification::centredLeft,
                       false);
        }
        g.setColour(t.line);
        g.fillRoundedRectangle(row, 2.0f);
        g.setColour(t.accent);
        g.fillRoundedRectangle(row.withWidth(row.getWidth() * s.blend[m]), 2.0f);
    }
}

void waveform(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, const Snapshot& s,
              juce::Colour colour)
{
    g.setColour(t.line);
    g.drawHorizontalLine(juce::roundToInt(area.getCentreY()), area.getX(), area.getRight());

    float peak = 0.05f;
    for (auto v : s.scope)
        peak = juce::jmax(peak, std::abs(v));

    juce::Path p;
    const auto n = s.scope.size();
    for (size_t i = 0; i < n; ++i)
    {
        const auto x = area.getX() + area.getWidth() * static_cast<float>(i) / static_cast<float>(n - 1);
        const auto y = area.getCentreY() - s.scope[i] / peak * area.getHeight() * 0.45f;
        if (i == 0)
            p.startNewSubPath(x, y);
        else
            p.lineTo(x, y);
    }
    g.setColour(colour);
    g.strokePath(p, juce::PathStrokeType(1.5f));
}

void envelope(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, const Snapshot& s)
{
    // Segment widths on a log-ish scale so a 5 ms attack is still visible next to
    // a 5 s release.
    auto seg = [](float ms)
    {
        return std::log1p(ms / 10.0f) + 0.3f;
    };
    const auto a = seg(s.attackMs), d = seg(s.decayMs), hold = 1.2f, r = seg(s.releaseMs);
    const auto scale = area.getWidth() / (a + d + hold + r);

    const auto x0 = area.getX();
    const auto top = area.getY() + 2.0f;
    const auto bottom = area.getBottom();
    const auto sustainY = bottom - (bottom - top) * s.sustain;

    juce::Path p;
    p.startNewSubPath(x0, bottom);
    p.lineTo(x0 + a * scale, top);
    p.lineTo(x0 + (a + d) * scale, sustainY);
    p.lineTo(x0 + (a + d + hold) * scale, sustainY);
    p.lineTo(x0 + (a + d + hold + r) * scale, bottom);

    auto fill = p;
    fill.closeSubPath();
    g.setColour(t.accent.withAlpha(0.15f));
    g.fillPath(fill);
    g.setColour(t.accent);
    g.strokePath(p, juce::PathStrokeType(1.5f));
}

void spectrum(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, const Snapshot& s)
{
    const auto n = static_cast<float>(s.spectrum.size());
    const auto w = area.getWidth() / n;
    for (size_t i = 0; i < s.spectrum.size(); ++i)
    {
        const auto norm = juce::jlimit(0.0f, 1.0f, (s.spectrum[i] + 90.0f) / 90.0f);
        auto bar =
            juce::Rectangle<float>(area.getX() + static_cast<float>(i) * w, area.getY(), w, area.getHeight());
        bar = bar.withTop(bar.getBottom() - bar.getHeight() * norm).reduced(w * 0.1f, 0.0f);
        g.setColour(t.accent.withAlpha(0.35f + 0.65f * norm));
        g.fillRect(bar);
    }
}

void meter(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, float peakL, float peakR)
{
    auto draw = [&](juce::Rectangle<float> r, float peak)
    {
        g.setColour(t.line);
        g.fillRoundedRectangle(r, 2.0f);
        const auto db = juce::Decibels::gainToDecibels(peak, -60.0f);
        const auto norm = (db + 60.0f) / 60.0f;
        g.setColour(db > -1.0f ? t.warn : t.accent);
        g.fillRoundedRectangle(r.withTop(r.getBottom() - r.getHeight() * norm), 2.0f);
    };
    const auto half = area.getWidth() * 0.5f;
    draw(area.withWidth(half - 1.0f), peakL);
    draw(area.withTrimmedLeft(half + 1.0f), peakR);
}
} // namespace viz

} // namespace alfa::ui
