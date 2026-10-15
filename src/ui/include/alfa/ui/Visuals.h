#pragma once
#include "alfa/ui/AlfaLookAndFeel.h"

#include <cmath>
#include <optional>
#include <span>
#include <vector>

// Shared presentation primitives. No method, tier, tensor or host contract.
namespace alfa::ui::visuals
{
inline void rotaryGesture(juce::Slider& slider, const juce::String& gesture)
{
    slider.setSliderStyle(gesture == "horizontal" ? juce::Slider::RotaryHorizontalDrag
                          : gesture == "circular" ? juce::Slider::Rotary
                                                  : juce::Slider::RotaryVerticalDrag);
    slider.setVelocityBasedMode(false);
    slider.setVelocityModeParameters(0.15, 1, 0, true, juce::ModifierKeys::shiftModifier);
    slider.setMouseDragSensitivity(200);
}
inline bool isLight(const Theme& t)
{
    return t.bg.getPerceivedBrightness() > 0.5f;
}
// The recessed tone a curve, scope or field draws on: below the background in dark themes, between
// background and panel in light ones.
inline juce::Colour screenColour(const Theme& t)
{
    return isLight(t) ? t.bg.interpolatedWith(t.panel, 0.3f) : t.bg.darker(0.18f);
}
// "The sound now" in every curve: a neutral, slightly warm tone. Amber and blue are kept for change.
inline juce::Colour nowColour(const Theme& t)
{
    return t.text.interpolatedWith(t.text2, isLight(t) ? 0.1f : 0.12f);
}
inline juce::Colour edgeLight(const Theme& t)
{
    return juce::Colours::white.withAlpha(isLight(t) ? 0.95f : 0.055f);
}
inline juce::Colour edgeShadow(const Theme& t)
{
    return juce::Colours::black.withAlpha(isLight(t) ? 0.13f : 0.5f);
}
// A raised surface: flat fill, a hairline of light on the top edge, a soft shadow under the bottom edge.
inline void raised(juce::Graphics& g, const Theme& t, juce::Rectangle<float> r, float radius,
                   std::optional<juce::Colour> fill = std::nullopt)
{
    juce::Path shape;
    shape.addRoundedRectangle(r, radius);
    juce::DropShadow(juce::Colours::black.withAlpha(isLight(t) ? 0.07f : 0.28f), 10, {0, 3})
        .drawForPath(g, shape);
    g.setColour(edgeShadow(t));
    g.fillRoundedRectangle(r.translated(0, 1), radius);
    g.setColour(fill.value_or(t.panel));
    g.fillRoundedRectangle(r, radius);
    const juce::Graphics::ScopedSaveState saved(g);
    g.reduceClipRegion(shape);
    g.setColour(edgeLight(t));
    g.drawRoundedRectangle(r.translated(0, 1).withTrimmedBottom(-2), radius, 1.0f);
}
// A recessed well: darker flat fill with a short inner shadow from the top edge and light on the bottom.
inline void recessed(juce::Graphics& g, const Theme& t, juce::Rectangle<float> r, float radius,
                     std::optional<juce::Colour> fill = std::nullopt)
{
    juce::Path shape;
    shape.addRoundedRectangle(r, radius);
    g.setColour(fill.value_or(screenColour(t)));
    g.fillPath(shape);
    const juce::Graphics::ScopedSaveState saved(g);
    g.reduceClipRegion(shape);
    const auto depth = juce::jmin(5.0f, r.getHeight() * 0.25f);
    g.setGradientFill(juce::ColourGradient(juce::Colours::black.withAlpha(isLight(t) ? 0.10f : 0.42f),
                                           r.getX(), r.getY(), juce::Colours::transparentBlack, r.getX(),
                                           r.getY() + depth, false));
    g.fillRect(r.withHeight(depth));
    g.setColour(edgeLight(t));
    g.drawRoundedRectangle(r.translated(0, -1).withTrimmedTop(-2), radius, 1.0f);
}
// Panel with an optional title row: a small identity dot, the title in mixed case, a quiet subtitle.
inline void panel(juce::Graphics& g, const Theme& t, juce::Rectangle<int> area, const juce::String& title,
                  const juce::String& subtitle = {}, std::optional<juce::Colour> dot = std::nullopt)
{
    raised(g, t, area.toFloat(), 12);
    if (title.isEmpty())
        return;
    auto row = area.reduced(14, 0).withHeight(34);
    auto titleArea = row;
    if (dot)
    {
        g.setColour(*dot);
        g.fillEllipse(juce::Rectangle<float>(5, 5).withCentre(
            {static_cast<float>(row.getX()) + 2.5f, static_cast<float>(row.getCentreY())}));
        titleArea.removeFromLeft(13);
    }
    const auto titleFont = AlfaLookAndFeel::labelFont(12.5f, true);
    g.setColour(t.text);
    g.setFont(titleFont);
    g.drawText(title, titleArea, juce::Justification::centredLeft, false);
    if (subtitle.isNotEmpty())
    {
        auto free = titleArea.withTrimmedLeft(
            static_cast<int>(juce::GlyphArrangement::getStringWidth(titleFont, title)) + 16);
        g.setColour(t.text2);
        g.setFont(AlfaLookAndFeel::labelFont(11.0f));
        g.drawFittedText(subtitle, free, juce::Justification::centredRight, 1, 0.9f);
    }
}
// Now against where the sound is heading, on the same x samples (screen coordinates). The band between
// the two is amber where the heading is higher and blue where it is lower; the heading is dashed.
inline void deltaCurves(juce::Graphics& g, const Theme& t, const std::vector<juce::Point<float>>& now,
                        const std::vector<juce::Point<float>>& next, float baseline, bool fillNow = true)
{
    if (now.size() < 2)
        return;
    const auto line = [](const std::vector<juce::Point<float>>& points)
    {
        juce::Path p;
        p.startNewSubPath(points.front());
        for (size_t i = 1; i < points.size(); ++i)
            p.lineTo(points[i]);
        return p;
    };
    const auto curve = line(now);
    const auto left = now.front().x, right = now.back().x;
    if (fillNow)
    {
        auto area = curve;
        area.lineTo(right, baseline);
        area.lineTo(left, baseline);
        area.closeSubPath();
        float top = baseline;
        for (const auto& p : now)
            top = juce::jmin(top, p.y);
        g.setGradientFill(juce::ColourGradient(nowColour(t).withAlpha(isLight(t) ? 0.13f : 0.11f), left, top,
                                               nowColour(t).withAlpha(0.01f), left, baseline, false));
        g.fillPath(area);
    }
    if (next.size() == now.size())
    {
        juce::Path between = line(now);
        for (size_t i = next.size(); i-- > 0;)
            between.lineTo(next[i]);
        between.closeSubPath();
        for (const bool up : {true, false})
        {
            juce::Path half = curve;
            half.lineTo(right, up ? -1e5f : 1e5f);
            half.lineTo(left, up ? -1e5f : 1e5f);
            half.closeSubPath();
            const juce::Graphics::ScopedSaveState saved(g);
            g.reduceClipRegion(half);
            g.setColour((up ? t.accent : t.mod).withAlpha(isLight(t) ? 0.4f : 0.5f));
            g.fillPath(between);
        }
    }
    g.setColour(nowColour(t));
    g.strokePath(curve, juce::PathStrokeType(1.6f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    if (next.size() == now.size())
    {
        juce::Path dashed;
        const float pattern[] = {3.0f, 3.0f};
        juce::PathStrokeType(1.2f).createDashedStroke(dashed, line(next), pattern, 2);
        g.setColour(t.text2);
        g.fillPath(dashed);
    }
}
// A thin grid line in the screen tone.
inline juce::Colour gridColour(const Theme& t)
{
    return (isLight(t) ? juce::Colours::black : juce::Colours::white).withAlpha(isLight(t) ? 0.065f : 0.05f);
}
// A curve handle: hollow ring, or filled with a halo while it is the one being dragged.
inline void handle(juce::Graphics& g, const Theme& t, juce::Point<float> p, bool active)
{
    if (active)
    {
        g.setColour(nowColour(t).withAlpha(0.12f));
        g.fillEllipse(juce::Rectangle<float>(20, 20).withCentre(p));
        g.setColour(nowColour(t));
        g.fillEllipse(juce::Rectangle<float>(9, 9).withCentre(p));
        g.setColour(screenColour(t));
        g.drawEllipse(juce::Rectangle<float>(9, 9).withCentre(p), 1.5f);
        return;
    }
    g.setColour(screenColour(t));
    g.fillEllipse(juce::Rectangle<float>(9, 9).withCentre(p));
    g.setColour(nowColour(t));
    g.drawEllipse(juce::Rectangle<float>(9, 9).withCentre(p), 1.6f);
}
inline void badge(juce::Graphics& g, juce::Rectangle<float> area, const juce::String& text,
                  juce::Colour colour)
{
    g.setColour(colour.withAlpha(0.18f));
    g.fillRoundedRectangle(area, 3);
    g.setColour(colour);
    g.drawRoundedRectangle(area.reduced(0.5f), 3, 1);
    g.setFont(AlfaLookAndFeel::labelFont(10, true));
    g.drawText(text, area, juce::Justification::centred, false);
}
inline void waveform(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t,
                     std::span<const float> samples, juce::Colour colour, bool normalize = true)
{
    if (samples.size() < 2 || area.isEmpty())
        return;
    const juce::Graphics::ScopedSaveState saved(g);
    g.reduceClipRegion(area.toNearestInt());
    g.setColour(t.line);
    g.drawHorizontalLine(juce::roundToInt(area.getCentreY()), area.getX(), area.getRight());
    float peak = normalize ? 0.05f : 1.0f;
    if (normalize)
        for (const auto value : samples)
            peak = juce::jmax(peak, std::abs(value));
    juce::Path path;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const auto x =
            area.getX() + area.getWidth() * static_cast<float>(i) / static_cast<float>(samples.size() - 1);
        const auto y = area.getCentreY() - samples[i] / peak * area.getHeight() * 0.45f;
        if (i == 0)
            path.startNewSubPath(x, y);
        else
            path.lineTo(x, y);
    }
    g.setColour(colour);
    g.strokePath(path, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
}
inline void spectrum(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t,
                     std::span<const float> bands, std::optional<juce::Colour> colour = std::nullopt)
{
    if (bands.empty())
        return;
    const auto width = area.getWidth() / static_cast<float>(bands.size());
    for (size_t i = 0; i < bands.size(); ++i)
    {
        const auto norm = juce::jlimit(0.0f, 1.0f, (bands[i] + 90) / 90);
        auto bar = juce::Rectangle<float>(area.getX() + static_cast<float>(i) * width, area.getY(), width,
                                          area.getHeight());
        bar = bar.withTop(bar.getBottom() - bar.getHeight() * norm).reduced(width * 0.15f, 0);
        g.setColour(colour.value_or(t.accent).withAlpha(0.3f + 0.6f * norm));
        g.fillRoundedRectangle(bar, juce::jmin(1.5f, bar.getWidth() * 0.5f));
    }
}
inline void meter(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, float left, float right,
                  juce::Colour colour)
{
    auto draw = [&](juce::Rectangle<float> r, float peak)
    {
        g.setColour(t.line);
        g.fillRoundedRectangle(r, juce::jmin(4.0f, r.getWidth() * 0.5f));
        const auto db = juce::Decibels::gainToDecibels(juce::jmax(peak, 0.0f), -60.0f);
        const auto norm = juce::jlimit(0.0f, 1.0f, (db + 60) / 60);
        g.setColour(db > -1 ? t.warn : colour);
        g.fillRoundedRectangle(r.withTop(r.getBottom() - r.getHeight() * norm), juce::jmin(4.0f, r.getWidth() * 0.5f));
    };
    const auto half = area.getWidth() * 0.5f;
    draw(area.withWidth(half - 1), left);
    draw(area.withTrimmedLeft(half + 1), right);
}
inline void logSpectrum(std::span<const float> magnitudes, size_t fftSize, float sampleRate,
                        std::span<float> bands)
{
    if (magnitudes.size() < fftSize / 2 + 1 || fftSize < 4 || sampleRate <= 0 || bands.empty())
        return;
    const auto half = static_cast<int>(fftSize / 2);
    for (size_t band = 0; band < bands.size(); ++band)
    {
        const auto low =
            30.0f * std::pow(20000.0f / 30.0f, static_cast<float>(band) / static_cast<float>(bands.size()));
        const auto high = 30.0f * std::pow(20000.0f / 30.0f,
                                           static_cast<float>(band + 1) / static_cast<float>(bands.size()));
        const auto first =
            juce::jlimit(1, half - 1, static_cast<int>(low / sampleRate * static_cast<float>(fftSize)));
        const auto last = juce::jlimit(first + 1, half,
                                       static_cast<int>(high / sampleRate * static_cast<float>(fftSize)) + 1);
        float peak = 0;
        for (int bin = first; bin < last; ++bin)
            peak = juce::jmax(peak, magnitudes[static_cast<size_t>(bin)]);
        const auto db = juce::Decibels::gainToDecibels(peak / (static_cast<float>(fftSize) * 0.25f), -90.0f);
        bands[band] = juce::jmax(db, bands[band] - 3.0f);
    }
}
} // namespace alfa::ui::visuals
