#pragma once
#include "alfa/ui/AlfaLookAndFeel.h"

#include <cmath>
#include <span>

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
inline void panel(juce::Graphics& g, const Theme& t, juce::Rectangle<int> area, const juce::String& title,
                  const juce::String& subtitle = {})
{
    // Soft vertical gradient, hairline border and a faint top highlight: surfaces read as material.
    const auto bounds = area.toFloat().reduced(0.5f);
    g.setGradientFill(juce::ColourGradient(t.panel.brighter(0.035f), bounds.getX(), bounds.getY(),
                                           t.panel.darker(0.05f), bounds.getX(), bounds.getBottom(), false));
    g.fillRoundedRectangle(bounds, 10);
    g.setColour(t.line);
    g.drawRoundedRectangle(bounds, 10, 1);
    g.setColour(juce::Colours::white.withAlpha(t.panel.getPerceivedBrightness() > 0.5f ? 0.5f : 0.035f));
    g.drawHorizontalLine(juce::roundToInt(bounds.getY()) + 1, bounds.getX() + 10, bounds.getRight() - 10);
    if (title.isEmpty())
        return;
    auto row = area.reduced(12, 0).withHeight(26).withTrimmedTop(6);
    g.setColour(t.text2);
    g.setFont(AlfaLookAndFeel::labelFont(11.5f, true));
    g.drawText(title.toUpperCase(), row, juce::Justification::centredLeft, false);
    if (subtitle.isNotEmpty())
    {
        g.setFont(AlfaLookAndFeel::labelFont(11.5f));
        g.drawText(subtitle, row, juce::Justification::centredRight, true);
    }
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
    g.strokePath(path, juce::PathStrokeType(1.5f));
}
inline void spectrum(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t,
                     std::span<const float> bands)
{
    if (bands.empty())
        return;
    const auto width = area.getWidth() / static_cast<float>(bands.size());
    for (size_t i = 0; i < bands.size(); ++i)
    {
        const auto norm = juce::jlimit(0.0f, 1.0f, (bands[i] + 90) / 90);
        auto bar = juce::Rectangle<float>(area.getX() + static_cast<float>(i) * width, area.getY(), width,
                                          area.getHeight());
        bar = bar.withTop(bar.getBottom() - bar.getHeight() * norm).reduced(width * 0.1f, 0);
        g.setColour(t.accent.withAlpha(0.35f + 0.65f * norm));
        g.fillRect(bar);
    }
}
inline void meter(juce::Graphics& g, juce::Rectangle<float> area, const Theme& t, float left, float right,
                  juce::Colour colour)
{
    auto draw = [&](juce::Rectangle<float> r, float peak)
    {
        g.setColour(t.line);
        g.fillRoundedRectangle(r, 2);
        const auto db = juce::Decibels::gainToDecibels(juce::jmax(peak, 0.0f), -60.0f);
        const auto norm = juce::jlimit(0.0f, 1.0f, (db + 60) / 60);
        g.setColour(db > -1 ? t.warn : colour);
        g.fillRoundedRectangle(r.withTop(r.getBottom() - r.getHeight() * norm), 2);
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
