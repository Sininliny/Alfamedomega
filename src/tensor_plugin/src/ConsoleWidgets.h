#pragma once
#include "TensorProcessor.h"
#include "alfa/ui/AlfaLookAndFeel.h"
#include "alfa/ui/Visuals.h"

#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

namespace alfa::tensorhost
{
inline juce::String scalarText(double value)
{
    // The shortest round-trip representation keeps exact doubles without turning
    // ordinary decimal controls into unreadable 17-digit binary artifacts.
    // The JSON formatter is portable to the product's macOS 11 deployment target;
    // floating-point std::to_chars requires macOS 13.3 in Apple libc++.
    return std::isfinite(value) ? juce::String(nlohmann::json(value).dump()) : juce::String(value, 17);
}
// Compact display for knob readouts: at most four significant decimals, trailing zeros removed.
inline juce::String compactText(double value)
{
    if (!std::isfinite(value))
        return juce::String(value);
    const auto magnitude = std::abs(value);
    const int decimals = magnitude >= 1000 ? 0 : magnitude >= 100 ? 1 : magnitude >= 10 ? 2 : 3;
    auto text = juce::String(value, decimals);
    if (text.containsChar('.'))
        text = text.trimCharactersAtEnd("0").trimCharactersAtEnd(".");
    return text == "-0" ? juce::String("0") : text;
}
// Signed weights share one convention everywhere: amber is positive, blue is negative.
inline juce::Colour weightColour(const alfa::ui::Theme& t, double value)
{
    return value < 0 ? t.mod : t.accent;
}
// Identity colours, as in Pigments: every primitive family and every source kind keeps one hue on every
// view (instance cards, the signal map, relationship dots, live modulation marks). Signs keep amber/blue.
// Hues are fixed; saturation and brightness follow the theme so light themes stay readable.
inline juce::Colour identityColour(const alfa::ui::Theme& t, float hue, float saturation = 0.55f)
{
    const bool light = t.bg.getPerceivedBrightness() > 0.5f;
    return juce::Colour::fromHSV(hue, light ? saturation + 0.2f : saturation, light ? 0.55f : 0.9f, 1.0f);
}
inline juce::Colour familyColour(const alfa::ui::Theme& t, const juce::String& primitive)
{
    static const std::array<std::pair<const char*, float>, 13> hues{{{"control", -1.0f},
                                                                     {"osc", 0.95f},
                                                                     {"excitation", 0.80f},
                                                                     {"physical", 0.27f},
                                                                     {"cell", 0.19f},
                                                                     {"filter", 0.47f},
                                                                     {"nonlinear", 0.03f},
                                                                     {"spectral", 0.66f},
                                                                     {"pitchtime", 0.73f},
                                                                     {"time", 0.53f},
                                                                     {"spatial", 0.33f},
                                                                     {"math", 0.12f},
                                                                     {"granular", 0.88f}}};
    const auto family = primitive.upToFirstOccurrenceOf(".", false, false);
    for (const auto& [name, hue] : hues)
        if (family == name)
            return hue < 0 ? t.text2 : identityColour(t, hue);
    return t.text2;
}
inline juce::Colour sourceColour(const alfa::ui::Theme& t, const juce::String& kind)
{
    static const std::array<std::pair<const char*, float>, 8> hues{{{"midi", 0.54f},
                                                                    {"audio", 0.30f},
                                                                    {"gesture", 0.86f},
                                                                    {"context", 0.15f},
                                                                    {"user_preference", 0.75f},
                                                                    {"latent_sound", 0.70f},
                                                                    {"state", -1.0f},
                                                                    {"output_history", -1.0f}}};
    for (const auto& [name, hue] : hues)
        if (kind == name)
            return hue < 0 ? t.text2 : identityColour(t, hue, 0.5f);
    return t.mod;
}
// Live level in [0, 1] from a peak amplitude, on a 60 dB scale.
inline float levelFraction(float peak)
{
    return juce::jlimit(0.0f, 1.0f, (juce::Decibels::gainToDecibels(peak, -60.0f) + 60.0f) / 60.0f);
}
inline void surface(juce::Graphics& g, const alfa::ui::Theme& t, juce::Rectangle<int> r,
                    const juce::String& title, const juce::String& subtitle = {})
{
    alfa::ui::visuals::panel(g, t, r, title, subtitle);
}
// Double-click a control's title to rename it. `raw` is the editable name without decorations such
// as the rank; an empty entry restores the X[layer,index] default.
inline void allowRename(juce::Label& label, const juce::String& raw,
                        std::function<void(const juce::String&)> rename)
{
    label.setEditable(false, true, false);
    label.setTooltip("Double-click to rename");
    label.onEditorShow = [&label, raw]
    {
        if (auto* editor = label.getCurrentTextEditor())
        {
            editor->setText(raw, false);
            editor->selectAll();
        }
    };
    label.onTextChange = [&label, rename = std::move(rename)]
    {
        rename(label.getText().trim());
    };
}
class ParameterKnob final : public juce::Component
{
public:
    // A fader is a short horizontal slider with its name on the left and its value on the right.
    ParameterKnob(Processor& p, const juce::String& id, const juce::String& label, bool percent = false,
                  bool fader = false)
        : parameterId(id)
        , fader_(fader)
    {
        name.setText(label, juce::dontSendNotification);
        name.setJustificationType(fader ? juce::Justification::centredLeft : juce::Justification::centred);
        name.setFont(alfa::ui::AlfaLookAndFeel::labelFont(fader ? 11.0f : 12.0f, true));
        addAndMakeVisible(name);
        if (fader)
        {
            slider.setSliderStyle(juce::Slider::LinearHorizontal);
            slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 64, 18);
        }
        else
        {
            alfa::ui::visuals::rotaryGesture(slider,
                                             p.uiSettings().getProperty("knob_gesture", "vertical").toString());
            slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 86, 16);
        }
        addAndMakeVisible(slider);
        slider.setWantsKeyboardFocus(true);
        if (percent)
        {
            slider.textFromValueFunction = [](double v)
            {
                return juce::String(v * 100, 1) + " %";
            };
            slider.valueFromTextFunction = [](const juce::String& s)
            {
                return s.getDoubleValue() / 100;
            };
        }
        else if (id.contains("gain"))
        {
            slider.setTextValueSuffix(" dB");
            slider.setNumDecimalPlacesToDisplay(1);
        }
        slider.onDragStart = [this]
        {
            if (onTouched)
                onTouched();
        };
        if (auto* parameter = p.parameters().getParameter(id))
            slider.setDoubleClickReturnValue(true, parameter->convertFrom0to1(parameter->getDefaultValue()));
        attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(p.parameters(),
                                                                                            id, slider);
        // The attachment installs the parameter's own seven-decimal text; X controls read better compact.
        if (percent)
        {
            slider.textFromValueFunction = [](double v)
            {
                return juce::String(juce::roundToInt(v * 100)) + " %";
            };
            slider.updateText();
        }
        else if (!id.contains("gain"))
        {
            slider.textFromValueFunction = [](double v)
            {
                return compactText(v);
            };
            slider.updateText();
        }
    }
    void resized() override
    {
        // Name under the knob, value as quiet text between them; a fader reads left to right.
        auto r = getLocalBounds();
        if (fader_)
            name.setBounds(r.removeFromLeft(38));
        else
            name.setBounds(r.removeFromBottom(18));
        slider.setBounds(r);
    }
    // Built before it joins the editor, so it takes the editor's look (value text, fonts) only now.
    void parentHierarchyChanged() override { sendLookAndFeelChange(); }
    bool isFader() const noexcept { return fader_; }
    juce::Slider& control() { return slider; }
    juce::Label& title() { return name; }
    std::function<void()> onTouched;
    void mouseDown(const juce::MouseEvent&) override
    {
        if (onTouched)
            onTouched();
    }

private:
    juce::String parameterId;
    bool fader_ = false;
    juce::Label name;
    juce::Slider slider;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
};
class NumericRow final : public juce::Component
{
public:
    using Commit = std::function<bool(double, juce::String&)>;
    NumericRow(juce::String title, juce::String unit, double value, Commit write,
               std::function<void(bool)> lockWrite = {}, bool locked = false,
               std::optional<std::array<double, 3>> range = std::nullopt)
        : commit(std::move(write))
        , lockCallback(std::move(lockWrite))
        , value_(value)
        , range_(range)
    {
        name.setText(title, juce::dontSendNotification);
        name.setTooltip(title);
        // One size for every row: long names wrap or end in "..." instead of shrinking.
        name.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13));
        name.setMinimumHorizontalScale(1.0f);
        addAndMakeVisible(name);
        units.setText(unit, juce::dontSendNotification);
        units.setJustificationType(juce::Justification::centredLeft);
        units.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f));
        units.setColour(juce::Label::textColourId, findColour(juce::Label::textColourId).withAlpha(0.6f));
        addAndMakeVisible(units);
        entry.setText(scalarText(value));
        entry.setJustification(juce::Justification::centredRight);
        // The value reads as text beside the fader; it shows a field only while it has focus.
        entry.setFont(alfa::ui::AlfaLookAndFeel::monoFont(12.5f));
        entry.setColour(juce::TextEditor::backgroundColourId, juce::Colours::transparentBlack);
        entry.setColour(juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
        entry.setSelectAllWhenFocused(true);
        addAndMakeVisible(entry);
        entry.onReturnKey = [this]
        {
            submit();
        };
        entry.onFocusLost = [this]
        {
            submit();
        };
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        if (range_)
        {
            slider.setRange((*range_)[0], (*range_)[1], (*range_)[2]);
            // Times and frequencies span decades; a linear slider makes the short end unreachable.
            const auto low = (*range_)[0], high = (*range_)[1];
            if ((unit == "s" || unit == "ms" || unit == "Hz") && low > 0 && high / low > 20)
                slider.setSkewFactorFromMidPoint(std::sqrt(low * high));
            slider.setValue(value, juce::dontSendNotification);
            addAndMakeVisible(slider);
        }
        slider.onValueChange = [this]
        {
            entry.setText(scalarText(slider.getValue()), false);
        };
        slider.onDragEnd = [this]
        {
            submit();
        };
        lock.setButtonText("Lock");
        lock.setToggleState(locked, juce::dontSendNotification);
        if (lockCallback)
        {
            addAndMakeVisible(lock);
            lock.onClick = [this]
            {
                lockCallback(lock.getToggleState());
            };
        }
    }
    void resized() override
    {
        auto r = getLocalBounds().reduced(8, 5);
        if (lockCallback)
            lock.setBounds(r.removeFromRight(64));
        units.setBounds(r.removeFromRight(44).withTrimmedLeft(4));
        entry.setBounds(r.removeFromRight(84).reduced(0, 3));
        if (identity)
            r.removeFromLeft(14);
        if (range_)
        {
            name.setBounds(r.removeFromLeft(juce::jmin(240, r.getWidth() * 3 / 5)));
            slider.setBounds(r.reduced(8, 0));
        }
        else
            name.setBounds(r);
    }
    void paint(juce::Graphics& g) override
    {
        g.setColour(findColour(juce::TextEditor::outlineColourId).withAlpha(0.25f));
        g.drawHorizontalLine(getHeight() - 1, 8.0f, static_cast<float>(getWidth() - 8));
        if (identity)
        {
            g.setColour(*identity);
            g.fillEllipse(juce::Rectangle<float>(7, 7).withCentre(
                {12.5f, static_cast<float>(name.getBounds().getCentreY())}));
        }
        if (weightTheme && !range_)
        {
            // Diverging bar under the name: centre is zero, the side shows the sign.
            auto bar = name.getBounds().toFloat().withTrimmedTop(static_cast<float>(name.getHeight() - 4));
            const auto centre = bar.getCentreX();
            const auto amount =
                static_cast<float>(juce::jlimit(0.0, 1.0, std::abs(value_) / juce::jmax(1.0, weightScale)));
            g.setColour(weightTheme->line);
            g.fillRect(bar.withHeight(2.0f));
            g.setColour(weightColour(*weightTheme, value_));
            const auto half = bar.getWidth() * 0.5f * amount;
            g.fillRect(value_ < 0 ? juce::Rectangle<float>(centre - half, bar.getY(), half, 2.0f)
                                  : juce::Rectangle<float>(centre, bar.getY(), half, 2.0f));
        }
    }
    void showWeightBar(const alfa::ui::Theme& t, double scale)
    {
        weightTheme = &t;
        weightScale = scale;
        repaint();
    }
    // A coloured dot before the name: the source kind of a relationship, or the family of an instance.
    void setIdentity(juce::Colour colour)
    {
        identity = colour;
        resized();
        repaint();
    }
    // The value the runtime used in the last block, drawn on the slider as a band from the edited value.
    // This is where modulation (projections) moved the parameter; nothing is drawn while they agree.
    void setLive(std::optional<double> value, juce::Colour colour)
    {
        if (value == live && colour == liveColour)
            return;
        if (value && live && std::abs(*value - *live) <= 1e-6 * juce::jmax(1.0, std::abs(*value)) &&
            colour == liveColour)
            return;
        live = value;
        liveColour = colour;
        repaint();
    }
    void paintOverChildren(juce::Graphics& g) override
    {
        if (!live || !range_ || !slider.isVisible())
            return;
        const auto low = slider.getMinimum(), high = slider.getMaximum();
        const auto clamped = juce::jlimit(low, high, *live);
        const auto from = static_cast<float>(slider.getX()) + slider.getPositionOfValue(value_);
        const auto to = static_cast<float>(slider.getX()) + slider.getPositionOfValue(clamped);
        if (std::abs(to - from) < 1.0f &&
            std::abs(*live - value_) <= 1e-9 * juce::jmax(1.0, std::abs(value_)))
            return;
        const auto y = static_cast<float>(slider.getBounds().getCentreY());
        g.setColour(liveColour.withAlpha(0.45f));
        g.fillRoundedRectangle(juce::Rectangle<float>(juce::jmin(from, to), y - 5, std::abs(to - from), 10),
                               3);
        g.setColour(liveColour);
        g.fillRoundedRectangle(juce::Rectangle<float>(to - 1.5f, y - 8, 3, 16), 1.5f);
    }

private:
    const alfa::ui::Theme* weightTheme = nullptr;
    double weightScale = 1;
    std::optional<juce::Colour> identity;
    std::optional<double> live;
    juce::Colour liveColour;
    void submit()
    {
        try
        {
            const auto text = entry.getText().trim().toStdString();
            size_t used = 0;
            const auto parsed = std::stod(text, &used);
            if (used != text.size() || !std::isfinite(parsed))
                throw std::invalid_argument("Enter a finite number");
            if (parsed == value_)
                return;
            juce::String error;
            if (commit(parsed, error))
            {
                value_ = parsed;
                entry.setTooltip({});
                repaint();
            }
            else
            {
                entry.setText(scalarText(value_), false);
                entry.setTooltip(error);
            }
        }
        catch (const std::exception& e)
        {
            entry.setText(scalarText(value_), false);
            entry.setTooltip(e.what());
        }
    }
    Commit commit;
    std::function<void(bool)> lockCallback;
    double value_;
    std::optional<std::array<double, 3>> range_;
    juce::Label name, units;
    juce::TextEditor entry;
    juce::Slider slider;
    juce::ToggleButton lock;
};
// The whole sound at a glance: the output spectrum now (solid) and where its recent change is taking it
// (dashed: the measured change of the last half second carried on for as long again), amber where it
// rises and blue where it falls, with the measured change of a few broad properties beside it.
class LiveScope final : public juce::Component
{
public:
    static constexpr size_t bandCount = 120;
    static constexpr float lowest = 30.0f, highest = 16000.0f, floorDb = -72.0f;
    LiveScope(Processor& p, alfa::ui::AlfaLookAndFeel& l)
        : processor(p)
        , look(l)
    {
        fast.fill(floorDb);
        slow.fill(floorDb);
    }
    // Analyse the latest output; called at the editor's timer rate.
    void tick()
    {
        std::array<float, 2048> samples{};
        const auto start = processor.waveformCursor.load(std::memory_order_relaxed) % samples.size();
        for (size_t i = 0; i < samples.size(); ++i)
            samples[i] = processor.waveformSnapshot[(start + i) % samples.size()].load(std::memory_order_relaxed);
        std::array<float, 4096> data{};
        std::copy(samples.begin(), samples.end(), data.begin());
        window.multiplyWithWindowingTable(data.data(), 2048);
        fft.performFrequencyOnlyForwardTransform(data.data());
        rate = static_cast<float>(processor.getSampleRate() > 0 ? processor.getSampleRate() : 48000.0);
        const auto seed = !seeded;
        seeded = true;
        const auto bin = rate / 2048.0f;
        for (size_t k = 0; k < bandCount; ++k)
        {
            const auto low = frequency(static_cast<float>(k)), high = frequency(static_cast<float>(k) + 1.0f);
            float peak = 0;
            if (high - low < bin)
            {
                // Narrower than one bin: interpolate between the two nearest bins.
                const auto position = juce::jlimit(1.0f, 1022.0f, 0.5f * (low + high) / bin);
                const auto i = static_cast<size_t>(position);
                const auto f = position - static_cast<float>(i);
                peak = data[i] * (1 - f) + data[i + 1] * f;
            }
            else
                for (auto i = static_cast<size_t>(std::ceil(low / bin)); i <= static_cast<size_t>(high / bin) && i < 1024; ++i)
                    peak = juce::jmax(peak, data[i]);
            const auto db = juce::jmax(floorDb, juce::Decibels::gainToDecibels(peak / 512.0f, floorDb));
            fast[k] = seed ? db : fast[k] + 0.5f * (db - fast[k]);
            slow[k] = seed ? db : slow[k] + 0.07f * (db - slow[k]);
        }
        repaint();
    }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        alfa::ui::visuals::panel(g, t, getLocalBounds(), "Sound",
                                 "output spectrum, and where its recent change is heading", t.accent);
        auto inner = getLocalBounds().reduced(10).withTrimmedTop(26);
        const auto listWidth = inner.getWidth() > 520 ? 168 : 0;
        auto list = inner.removeFromRight(listWidth);
        if (listWidth)
            inner.removeFromRight(8);
        const auto screen = inner.toFloat();
        alfa::ui::visuals::recessed(g, t, screen, 8);
        auto plot = screen.reduced(30, 10).withTrimmedBottom(14).withTrimmedRight(-20);
        const auto x = [&](float hz)
        {
            return plot.getX() + plot.getWidth() * std::log(hz / lowest) / std::log(highest / lowest);
        };
        const auto y = [&](float db)
        {
            return plot.getY() + plot.getHeight() * juce::jlimit(0.0f, 1.0f, db / floorDb);
        };
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(9.5f));
        for (const auto db : {0.0f, -24.0f, -48.0f})
        {
            g.setColour(alfa::ui::visuals::gridColour(t));
            g.drawHorizontalLine(static_cast<int>(y(db)), plot.getX(), plot.getRight());
            g.setColour(t.text2.withAlpha(0.7f));
            g.drawText(juce::String(static_cast<int>(db)), juce::Rectangle<float>(screen.getX() + 2, y(db) - 6, 24, 12),
                       juce::Justification::centredRight);
        }
        for (const auto& [hz, label] : std::initializer_list<std::pair<float, const char*>>{
                 {50, "50"}, {100, "100"}, {200, "200"}, {500, "500"}, {1000, "1k"}, {2000, "2k"},
                 {5000, "5k"}, {10000, "10k"}})
        {
            g.setColour(alfa::ui::visuals::gridColour(t));
            g.drawVerticalLine(static_cast<int>(x(hz)), plot.getY(), plot.getBottom());
            g.setColour(t.text2.withAlpha(0.7f));
            g.drawText(label, juce::Rectangle<float>(x(hz) - 16, plot.getBottom() + 3, 32, 11),
                       juce::Justification::centred);
        }
        std::vector<juce::Point<float>> now, next;
        float level = floorDb;
        for (size_t k = 0; k < bandCount; ++k)
            level = juce::jmax(level, fast[k]);
        const auto sounding = level > floorDb + 6;
        for (size_t k = 0; k < bandCount; ++k)
        {
            const auto hz = frequency(static_cast<float>(k) + 0.5f);
            now.push_back({x(hz), y(fast[k])});
            next.push_back({x(hz), y(juce::jlimit(floorDb, 0.0f, fast[k] + (fast[k] - slow[k])))});
        }
        alfa::ui::visuals::deltaCurves(g, t, now, sounding ? next : std::vector<juce::Point<float>>{},
                                       plot.getBottom());
        if (!listWidth)
            return;
        // Broad properties of the output, now against half a second ago.
        auto rows = list.toFloat().reduced(4, 2);
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(9.5f, true));
        g.drawText("MEASURED CHANGE", rows.removeFromTop(16), juce::Justification::centredLeft);
        rows.removeFromTop(2);
        const auto measured = measure();
        const auto rowHeight = juce::jmin(20.0f, rows.getHeight() / static_cast<float>(measured.size()));
        for (const auto& [name, change, unit, digits] : measured)
        {
            auto row = rows.removeFromTop(rowHeight);
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
            g.drawText(name, row, juce::Justification::centredLeft);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f, true));
            const auto quiet = !sounding || std::abs(change) < std::pow(10.0, -digits) * 0.5;
            g.setColour(quiet ? t.text2 : change > 0 ? t.accent : t.mod);
            g.drawText(!sounding ? juce::String("-")
                                 : (change > 0 && !quiet ? "+" : "") +
                                       (digits == 0 ? juce::String(quiet ? 0 : juce::roundToInt(change))
                                                    : juce::String(quiet ? 0.0 : change, digits)) +
                                       " " + unit,
                       row, juce::Justification::centredRight);
        }
    }

private:
    struct Property
    {
        juce::String name;
        double change;
        juce::String unit;
        int digits;
    };
    Processor& processor;
    alfa::ui::AlfaLookAndFeel& look;
    std::array<float, bandCount> fast{}, slow{};
    float rate = 48000;
    bool seeded = false;
    juce::dsp::FFT fft{11};
    juce::dsp::WindowingFunction<float> window{2048, juce::dsp::WindowingFunction<float>::hann};
    static float frequency(float band)
    {
        return lowest * std::pow(highest / lowest, band / static_cast<float>(bandCount));
    }
    std::vector<Property> measure() const
    {
        const auto summary = [&](const std::array<float, bandCount>& bands)
        {
            // Power in the whole spectrum and in three regions, and the power-weighted centre frequency.
            std::array<double, 5> result{};
            double weighted = 0;
            for (size_t k = 0; k < bandCount; ++k)
            {
                const auto hz = frequency(static_cast<float>(k) + 0.5f);
                const auto power = std::pow(10.0, bands[k] / 10.0);
                result[0] += power;
                result[hz < 250 ? 1 : hz < 4000 ? 2 : 3] += power;
                weighted += power * hz;
            }
            result[4] = result[0] > 0 ? weighted / result[0] : 0;
            for (size_t i = 0; i < 4; ++i)
                result[i] = 10 * std::log10(juce::jmax(1e-12, result[i]));
            return result;
        };
        const auto a = summary(fast), b = summary(slow);
        return {{"Level", a[0] - b[0], "dB", 1},
                {"Centroid", a[4] - b[4], "Hz", 0},
                {"Lows < 250 Hz", a[1] - b[1], "dB", 1},
                {"Mids", a[2] - b[2], "dB", 1},
                {"Highs > 4 kHz", a[3] - b[3], "dB", 1}};
    }
};
// Input and output peaks as two thin horizontal bars, left over right channel.
class StereoMeter final : public juce::Component
{
public:
    StereoMeter(Processor& p, alfa::ui::AlfaLookAndFeel& l)
        : processor(p)
        , look(l)
    {
    }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        auto area = getLocalBounds().toFloat();
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(9.5f, true));
        const auto rowHeight = area.getHeight() / 2;
        for (int pair = 0; pair < 2; ++pair)
        {
            auto row = area.removeFromTop(rowHeight);
            g.setColour(t.text2);
            g.drawText(pair == 0 ? "IN" : "OUT", row.removeFromLeft(26), juce::Justification::centredLeft);
            const auto& peaks = pair == 0 ? processor.inputPeak : processor.outputPeak;
            for (int channel = 0; channel < 2; ++channel)
            {
                const auto bar = row.withHeight(3).withY(row.getCentreY() - 3.5f + 4.0f * static_cast<float>(channel));
                g.setColour(alfa::ui::visuals::screenColour(t));
                g.fillRoundedRectangle(bar, 1.5f);
                const auto peak = peaks[static_cast<size_t>(channel)].load();
                const auto fraction = levelFraction(peak);
                g.setColour(juce::Decibels::gainToDecibels(peak, -60.0f) > -1 ? t.warn
                                                                                 : alfa::ui::visuals::nowColour(t));
                g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * fraction), 1.5f);
            }
        }
    }

private:
    Processor& processor;
    alfa::ui::AlfaLookAndFeel& look;
};
// The on-screen keyboard: soft ivory and graphite keys, played keys in the accent.
class KeyboardView final : public juce::MidiKeyboardComponent
{
public:
    KeyboardView(juce::MidiKeyboardState& keys, alfa::ui::AlfaLookAndFeel& l)
        : juce::MidiKeyboardComponent(keys, horizontalKeyboard)
        , look(l)
    {
        setScrollButtonsVisible(false);
    }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        g.setColour(alfa::ui::visuals::screenColour(t));
        g.fillRoundedRectangle(getLocalBounds().toFloat(), 6);
        juce::MidiKeyboardComponent::paint(g);
    }
    void drawWhiteNote(int, juce::Graphics& g, juce::Rectangle<float> area, bool isDown, bool isOver,
                       juce::Colour, juce::Colour) override
    {
        const auto& t = look.theme();
        const auto light = alfa::ui::visuals::isLight(t);
        auto key = area.reduced(0.75f, 0).withTrimmedTop(1);
        auto colour = isDown ? t.accent : light ? juce::Colour(0xfffdfcfa) : juce::Colour(0xffe6e2d9);
        if (isOver && !isDown)
            colour = colour.darker(0.06f);
        g.setGradientFill(juce::ColourGradient(colour, key.getX(), key.getY(), colour.darker(0.07f), key.getX(),
                                               key.getBottom(), false));
        g.fillRoundedRectangle(key, 3);
        g.setColour(juce::Colours::black.withAlpha(light ? 0.16f : 0.25f));
        g.drawRoundedRectangle(key, 3, 0.75f);
    }
    void drawBlackNote(int, juce::Graphics& g, juce::Rectangle<float> area, bool isDown, bool isOver,
                       juce::Colour) override
    {
        const auto& t = look.theme();
        auto key = area.withTrimmedTop(1).reduced(0.5f, 0);
        auto colour = isDown ? t.accent.darker(0.15f) : juce::Colour(0xff1d1f24);
        if (isOver && !isDown)
            colour = colour.brighter(0.15f);
        g.setColour(juce::Colours::black.withAlpha(0.3f));
        g.fillRoundedRectangle(key.translated(0, 1.5f), 2.5f);
        g.setGradientFill(juce::ColourGradient(colour.brighter(0.25f), key.getX(), key.getY(), colour, key.getX(),
                                               key.getBottom(), false));
        g.fillRoundedRectangle(key, 2.5f);
        g.setColour(juce::Colours::white.withAlpha(isDown ? 0.2f : 0.08f));
        g.fillRoundedRectangle(key.reduced(2.0f, 0).removeFromBottom(4).translated(0, -2), 1.5f);
    }

private:
    alfa::ui::AlfaLookAndFeel& look;
};
class ExplorationPad final : public juce::Component
{
public:
    ExplorationPad(Processor& p, alfa::ui::AlfaLookAndFeel& l)
        : processor(p)
        , look(l)
    {
    }
    juce::String xId{"x0"}, yId{"x1"};
    // Display names of the two axes (user control names); the IDs are shown when empty.
    juce::String xName, yName;
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        auto r = getLocalBounds().toFloat().reduced(12);
        g.setColour(t.bg);
        g.fillRoundedRectangle(r, 6);
        g.setColour(t.line);
        for (int i = 1; i < 4; ++i)
        {
            auto x = r.getX() + r.getWidth() * static_cast<float>(i) / 4;
            auto y = r.getY() + r.getHeight() * static_cast<float>(i) / 4;
            g.drawLine(x, r.getY(), x, r.getBottom());
            g.drawLine(r.getX(), y, r.getRight(), y);
        }
        const auto x = value(xId), y = value(yId);
        const auto px = r.getX() + (x + 1) * 0.5f * r.getWidth();
        const auto py = r.getBottom() - (y + 1) * 0.5f * r.getHeight();
        g.setColour(t.accent.withAlpha(0.2f));
        g.fillEllipse(px - 15, py - 15, 30, 30);
        g.setColour(t.accent);
        g.fillEllipse(px - 4, py - 4, 8, 8);
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(11));
        g.drawText((xName.isEmpty() ? xId.toUpperCase() : xName) + "  " + juce::String(x, 2) + "  /  " +
                       (yName.isEmpty() ? yId.toUpperCase() : yName) + "  " + juce::String(y, 2),
                   r.removeFromBottom(18), juce::Justification::centred);
    }
    void mouseDown(const juce::MouseEvent& e) override
    {
        for (const auto& id : {xId, yId})
            if (auto* p = processor.parameters().getParameter(id))
                p->beginChangeGesture();
        move(e);
    }
    void mouseDrag(const juce::MouseEvent& e) override { move(e); }
    void mouseUp(const juce::MouseEvent&) override
    {
        for (const auto& id : {xId, yId})
            if (auto* p = processor.parameters().getParameter(id))
                p->endChangeGesture();
    }

private:
    float value(const juce::String& id) const
    {
        if (auto* p = processor.parameters().getRawParameterValue(id))
            return p->load();
        return 0;
    }
    void move(const juce::MouseEvent& e)
    {
        const auto r = getLocalBounds().toFloat().reduced(12);
        if (r.getWidth() <= 0 || r.getHeight() <= 0)
            return;
        const auto x = juce::jlimit(-1.0f, 1.0f, 2 * (e.position.x - r.getX()) / r.getWidth() - 1);
        const auto y = juce::jlimit(-1.0f, 1.0f, 1 - 2 * (e.position.y - r.getY()) / r.getHeight());
        for (const auto& item : {std::pair{xId, x}, std::pair{yId, y}})
            if (auto* p = processor.parameters().getParameter(item.first))
                p->setValueNotifyingHost(p->convertTo0to1(item.second));
        repaint();
    }
    Processor& processor;
    alfa::ui::AlfaLookAndFeel& look;
};
// Ranked, signed list: which bottom parameters one control reaches and how strongly.
class InfluenceView final : public juce::Component
{
public:
    struct Entry
    {
        juce::String label;
        double weight = 0;
        bool locked = false;
    };
    explicit InfluenceView(alfa::ui::AlfaLookAndFeel& l)
        : look(l)
    {
    }
    void setEntries(juce::String heading, std::vector<Entry> list, int reached)
    {
        title = std::move(heading);
        entries = std::move(list);
        reachedCount = reached;
        repaint();
    }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        auto r = getLocalBounds();
        g.setColour(t.text);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13, true));
        g.drawText(title, r.removeFromTop(22), juce::Justification::centredLeft);
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
        g.drawText(entries.empty() ? juce::String("No mapped parameters: this control reaches nothing yet.")
                                   : "Reaches " + juce::String(reachedCount) +
                                         " parameters. Linear weight through the authored matrices, "
                                         "before range clamps. Amber +, blue -.",
                   r.removeFromTop(18), juce::Justification::centredLeft);
        r.removeFromTop(6);
        double scale = 1e-12;
        for (const auto& e : entries)
            scale = juce::jmax(scale, std::abs(e.weight));
        const auto rowHeight = 20;
        for (const auto& e : entries)
        {
            if (r.getHeight() < rowHeight)
                break;
            auto row = r.removeFromTop(rowHeight);
            auto labelArea = row.removeFromLeft(juce::jmin(260, row.getWidth() * 2 / 5));
            auto valueArea = row.removeFromRight(70);
            g.setColour(e.locked ? t.text2.withAlpha(0.6f) : t.text);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
            g.drawText((e.locked ? "[locked] " : "") + e.label, labelArea.reduced(2, 0),
                       juce::Justification::centredLeft, true);
            auto bar = row.reduced(8, 6).toFloat();
            g.setColour(t.line);
            g.fillRect(bar.getCentreX() - 0.5f, bar.getY() - 3, 1.0f, bar.getHeight() + 6);
            const auto half = bar.getWidth() * 0.5f * static_cast<float>(std::abs(e.weight) / scale);
            g.setColour(weightColour(t, e.weight).withAlpha(e.locked ? 0.35f : 0.9f));
            g.fillRoundedRectangle(
                e.weight < 0
                    ? juce::Rectangle<float>(bar.getCentreX() - half, bar.getY(), half, bar.getHeight())
                    : juce::Rectangle<float>(bar.getCentreX(), bar.getY(), half, bar.getHeight()),
                2);
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(11));
            g.drawText(compactText(e.weight), valueArea, juce::Justification::centredRight);
        }
    }

private:
    alfa::ui::AlfaLookAndFeel& look;
    juce::String title;
    std::vector<Entry> entries;
    int reachedCount = 0;
};
// Heat map of one control layer matrix: destinations as rows, sources as columns.
class MatrixView final : public juce::Component, public juce::SettableTooltipClient
{
public:
    explicit MatrixView(alfa::ui::AlfaLookAndFeel& l)
        : look(l)
    {
    }
    std::function<void(int source, int destination)> onSelect;
    static constexpr int headerHeight = 24, legendHeight = 34;
    void setMatrix(std::vector<double> data, juce::StringArray rows, juce::StringArray columns)
    {
        values = std::move(data);
        rowLabels = std::move(rows);
        columnLabels = std::move(columns);
        scale = 1e-12;
        for (const auto v : values)
            scale = juce::jmax(scale, std::abs(v));
        selectedRow = juce::jlimit(-1, rowLabels.size() - 1, selectedRow);
        selectedColumn = juce::jlimit(-1, columnLabels.size() - 1, selectedColumn);
        repaint();
    }
    // Live source values as fractions of their ranges, drawn as a thin bar under each column number.
    void setLive(std::vector<float> fractions)
    {
        if (fractions == liveColumns)
            return;
        liveColumns = std::move(fractions);
        repaint(0, 0, getWidth(), headerHeight);
    }
    void select(int source, int destination)
    {
        selectedColumn = source;
        selectedRow = destination;
        repaint();
    }
    // Rows grow with the available height so small matrices fill their panel; large ones scroll.
    int preferredHeight(int available) const
    {
        const auto rows = juce::jmax(1, rowLabels.size());
        const auto fitted = (available - headerHeight - legendHeight) / rows;
        return headerHeight + rows * juce::jlimit(minimumRow, maximumRow, fitted) + legendHeight;
    }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        const auto rows = rowHeight();
        g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(11));
        for (int c = 0; c < columnLabels.size(); ++c)
        {
            g.setColour(c == selectedColumn ? t.accent : t.text2);
            g.drawText(juce::String(c), column(c).withY(0).withHeight(headerHeight - 6),
                       juce::Justification::centred, false);
            if (static_cast<size_t>(c) < liveColumns.size())
            {
                auto bar = column(c).withY(headerHeight - 6).withHeight(3).reduced(3, 0).toFloat();
                g.setColour(t.line);
                g.fillRoundedRectangle(bar, 1.5f);
                g.setColour(t.accent);
                g.fillRoundedRectangle(
                    bar.withWidth(bar.getWidth() *
                                  juce::jlimit(0.0f, 1.0f, liveColumns[static_cast<size_t>(c)])),
                    1.5f);
            }
        }
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(
            juce::jlimit(11.0f, 13.0f, static_cast<float>(rows) * 0.55f)));
        for (int r = 0; r < rowLabels.size(); ++r)
        {
            const auto y = headerHeight + r * rows;
            if (r == selectedRow)
            {
                g.setColour(t.accent.withAlpha(0.08f));
                g.fillRect(0, y, labelWidth() + columnLabels.size() * columnWidth(), rows);
            }
            g.setColour(r == selectedRow ? t.accent : t.text2);
            g.drawFittedText(rowLabels[r], 4, y, labelWidth() - 10, rows, juce::Justification::centredRight,
                             1, 0.9f);
            for (int c = 0; c < columnLabels.size(); ++c)
            {
                const auto index = static_cast<size_t>(r * columnLabels.size() + c);
                const auto v = index < values.size() ? values[index] : 0.0;
                auto box = column(c).withY(y).withHeight(rows).reduced(1).toFloat();
                g.setColour(t.bg);
                g.fillRoundedRectangle(box, 2);
                if (std::abs(v) > 0.0)
                {
                    // Square-root scale keeps small but real coefficients visible next to large ones.
                    const auto amount = static_cast<float>(std::sqrt(std::abs(v) / scale));
                    g.setColour(weightColour(t, v).withAlpha(0.22f + 0.78f * amount));
                    g.fillRoundedRectangle(box, 2);
                    if (box.getWidth() >= 34 && box.getHeight() >= 18)
                    {
                        g.setColour(amount > 0.6f ? t.bg : t.text);
                        g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(10));
                        g.drawText(compactText(v), box, juce::Justification::centred, false);
                        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(
                            juce::jlimit(11.0f, 13.0f, static_cast<float>(rows) * 0.55f)));
                    }
                }
                if (r == selectedRow && c == selectedColumn)
                {
                    g.setColour(t.text);
                    g.drawRoundedRectangle(box, 2, 1.5f);
                }
            }
        }
        paintLegend(g, t);
    }
    void mouseDown(const juce::MouseEvent& e) override
    {
        const auto hit = hitCell(e.getPosition());
        if (hit.first < 0)
            return;
        select(hit.first, hit.second);
        if (onSelect)
            onSelect(hit.first, hit.second);
    }
    void mouseMove(const juce::MouseEvent& e) override
    {
        const auto hit = hitCell(e.getPosition());
        if (hit.first < 0)
        {
            setTooltip({});
            return;
        }
        const auto index = static_cast<size_t>(hit.second * columnLabels.size() + hit.first);
        setTooltip(columnLabels[hit.first] + "  >  " + rowLabels[hit.second] + "  =  " +
                   scalarText(index < values.size() ? values[index] : 0.0));
    }

private:
    static constexpr int minimumRow = 20, maximumRow = 34;
    alfa::ui::AlfaLookAndFeel& look;
    std::vector<double> values;
    std::vector<float> liveColumns;
    juce::StringArray rowLabels, columnLabels;
    double scale = 1;
    int selectedRow = -1, selectedColumn = -1;
    int rowHeight() const
    {
        const auto rows = juce::jmax(1, rowLabels.size());
        return juce::jlimit(minimumRow, maximumRow, (getHeight() - headerHeight - legendHeight) / rows);
    }
    int labelWidth() const
    {
        // Wide enough for the longest destination name, never more than 45% of the view.
        const auto font = alfa::ui::AlfaLookAndFeel::labelFont(13);
        float widest = 40;
        for (const auto& label : rowLabels)
            widest = juce::jmax(widest, juce::GlyphArrangement::getStringWidth(font, label));
        return juce::jmin(static_cast<int>(widest) + 18, getWidth() * 45 / 100);
    }
    juce::Rectangle<int> cellArea() const { return getLocalBounds().withTrimmedLeft(labelWidth()); }
    int columnWidth() const
    {
        return columnLabels.isEmpty() ? 0 : juce::jlimit(14, 72, cellArea().getWidth() / columnLabels.size());
    }
    juce::Rectangle<int> column(int c) const
    {
        return {labelWidth() + c * columnWidth(), 0, columnWidth(), getHeight()};
    }
    void paintLegend(juce::Graphics& g, const alfa::ui::Theme& t)
    {
        if (columnLabels.isEmpty())
            return;
        // Diverging scale: blue negative, amber positive, both relative to this matrix's largest |weight|.
        const auto top = headerHeight + rowLabels.size() * rowHeight() + 10;
        auto area =
            juce::Rectangle<int>(labelWidth(), top, juce::jmin(260, columnLabels.size() * columnWidth()), 12);
        const auto half = area.getWidth() / 2;
        for (int x = 0; x < area.getWidth(); ++x)
        {
            const auto value = (x - half) / static_cast<double>(juce::jmax(1, half));
            const auto amount = static_cast<float>(std::sqrt(std::abs(value)));
            g.setColour(std::abs(value) < 0.02 ? t.bg
                                               : weightColour(t, value).withAlpha(0.22f + 0.78f * amount));
            g.fillRect(area.getX() + x, area.getY(), 1, area.getHeight());
        }
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(10));
        const auto text = area.translated(0, 13).withHeight(12);
        g.drawText("-" + compactText(scale), text, juce::Justification::left, false);
        g.drawText("0", text, juce::Justification::centred, false);
        g.drawText("+" + compactText(scale), text, juce::Justification::right, false);
    }
    std::pair<int, int> hitCell(juce::Point<int> p) const
    {
        if (p.x < labelWidth() || p.y < headerHeight || columnWidth() == 0)
            return {-1, -1};
        const auto c = (p.x - labelWidth()) / columnWidth(), r = (p.y - headerHeight) / rowHeight();
        if (c >= columnLabels.size() || r >= rowLabels.size())
            return {-1, -1};
        return {c, r};
    }
};
// Live values of every control layer: one row of bars per layer, the page shown as knobs outlined.
// Bars grow from zero for ranges that span it, from the minimum otherwise.
class LayerStateView final : public juce::Component, public juce::SettableTooltipClient
{
public:
    struct Layer
    {
        std::vector<std::array<double, 2>> ranges;
        std::vector<double> values;
    };
    explicit LayerStateView(alfa::ui::AlfaLookAndFeel& l)
        : look(l)
    {
    }
    std::function<void(int layer, int index)> onSelect;
    void setLayers(std::vector<Layer> list)
    {
        layers = std::move(list);
        repaint();
    }
    // Values concatenated layer by layer, as the processor publishes them.
    void setValues(const std::vector<float>& live)
    {
        size_t cursor = 0;
        bool changed = false;
        for (auto& layer : layers)
            for (auto& value : layer.values)
                if (cursor < live.size())
                {
                    changed = changed || std::abs(value - live[cursor]) > 1e-6;
                    value = live[cursor++];
                }
        if (changed)
            repaint();
    }
    void setPage(int layer, int begin, int end)
    {
        pageLayer = layer;
        pageBegin = begin;
        pageEnd = end;
        repaint();
    }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        auto r = getLocalBounds();
        if (layers.empty())
        {
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
            g.drawText("This sound has no control layer", r, juce::Justification::topLeft);
            return;
        }
        const auto rowHeight = juce::jlimit(34, 90, r.getHeight() / static_cast<int>(layers.size()));
        for (size_t l = 0; l < layers.size(); ++l)
        {
            auto row = r.removeFromTop(rowHeight);
            const auto& layer = layers[l];
            g.setColour(static_cast<int>(l) == pageLayer ? t.accent : t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11, true));
            g.drawText("LAYER " + juce::String(static_cast<int>(l)) + "   " +
                           juce::String(static_cast<int>(layer.values.size())) + " controls",
                       row.removeFromTop(16), juce::Justification::centredLeft, false);
            auto bars = row.reduced(0, 4).toFloat();
            const auto count = std::max<size_t>(1, layer.values.size());
            const auto w = bars.getWidth() / static_cast<float>(count);
            for (size_t i = 0; i < layer.values.size(); ++i)
            {
                auto box = juce::Rectangle<float>(bars.getX() + w * static_cast<float>(i), bars.getY(), w,
                                                  bars.getHeight())
                               .reduced(juce::jmin(2.0f, w * 0.15f), 0);
                const auto onPage = static_cast<int>(l) == pageLayer && static_cast<int>(i) >= pageBegin &&
                                    static_cast<int>(i) < pageEnd;
                g.setColour(onPage ? t.accent.withAlpha(0.10f) : t.bg);
                g.fillRoundedRectangle(box, 2);
                const auto low = i < layer.ranges.size() ? layer.ranges[i][0] : -1.0;
                const auto high = i < layer.ranges.size() ? layer.ranges[i][1] : 1.0;
                const auto span = high - low > 0 ? high - low : 1.0;
                const auto zero = low < 0 && high > 0 ? -low / span : 0.0;
                const auto fraction = juce::jlimit(0.0, 1.0, (layer.values[i] - low) / span);
                const auto top =
                    box.getBottom() - box.getHeight() * static_cast<float>(juce::jmax(zero, fraction));
                const auto bottom =
                    box.getBottom() - box.getHeight() * static_cast<float>(juce::jmin(zero, fraction));
                g.setColour(weightColour(t, layer.values[i]).withAlpha(0.85f));
                g.fillRect(box.withTop(top).withBottom(juce::jmax(bottom, top + 1.0f)));
                if (zero > 0)
                {
                    g.setColour(t.line);
                    const auto y = box.getBottom() - box.getHeight() * static_cast<float>(zero);
                    g.drawHorizontalLine(static_cast<int>(y), box.getX(), box.getRight());
                }
                if (onPage)
                {
                    g.setColour(t.accent.withAlpha(0.6f));
                    g.drawRoundedRectangle(box, 2, 1);
                }
            }
        }
    }
    void mouseMove(const juce::MouseEvent& e) override
    {
        const auto hit = hitTestBar(e.getPosition());
        setTooltip(
            hit.first < 0
                ? juce::String()
                : "X[" + juce::String(hit.first) + "," + juce::String(hit.second) + "] = " +
                      compactText(
                          layers[static_cast<size_t>(hit.first)].values[static_cast<size_t>(hit.second)]));
    }
    void mouseDown(const juce::MouseEvent& e) override
    {
        if (const auto hit = hitTestBar(e.getPosition()); hit.first >= 0 && onSelect)
            onSelect(hit.first, hit.second);
    }

private:
    alfa::ui::AlfaLookAndFeel& look;
    std::vector<Layer> layers;
    int pageLayer = -1, pageBegin = 0, pageEnd = 0;
    std::pair<int, int> hitTestBar(juce::Point<int> p) const
    {
        if (layers.empty())
            return {-1, -1};
        const auto rowHeight = juce::jlimit(34, 90, getHeight() / static_cast<int>(layers.size()));
        const auto l = p.y / rowHeight;
        if (l < 0 || l >= static_cast<int>(layers.size()) || p.y % rowHeight < 16)
            return {-1, -1};
        const auto count = static_cast<int>(layers[static_cast<size_t>(l)].values.size());
        if (count == 0 || getWidth() <= 0)
            return {-1, -1};
        return {l, juce::jlimit(0, count - 1, p.x * count / getWidth())};
    }
};
// Used / budget per resource of the compiled sound, as bars. Budgets come from the sound (or its
// authored tier profile); a full bar is normal, it means the sound uses all it declared.
class CapacityView final : public juce::Component
{
public:
    struct Entry
    {
        juce::String name;
        double used = 0, budget = 0;
    };
    explicit CapacityView(alfa::ui::AlfaLookAndFeel& l)
        : look(l)
    {
    }
    void setEntries(std::vector<Entry> e, juce::String note)
    {
        entries = std::move(e);
        caption = std::move(note);
        repaint();
    }
    // Reads the "Used / budget:" and "Budget source:" lines of the compiled inspection report.
    static std::vector<Entry> parse(const juce::String& report, juce::String& source)
    {
        std::vector<Entry> result;
        for (const auto& line : juce::StringArray::fromLines(report))
        {
            if (line.startsWith("Budget source: "))
                source = line.fromFirstOccurrenceOf("Budget source: ", false, false);
            if (!line.startsWith("Used / budget: "))
                continue;
            for (auto part :
                 juce::StringArray::fromTokens(line.fromFirstOccurrenceOf(": ", false, false), "|", ""))
            {
                part = part.trim();
                const auto name = part.upToLastOccurrenceOf(" ", false, false);
                const auto pair = part.fromLastOccurrenceOf(" ", false, false);
                if (name.isNotEmpty() && pair.containsChar('/'))
                    result.push_back({name.substring(0, 1).toUpperCase() + name.substring(1),
                                      pair.upToFirstOccurrenceOf("/", false, false).getDoubleValue(),
                                      pair.fromFirstOccurrenceOf("/", false, false).getDoubleValue()});
            }
        }
        return result;
    }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        auto r = getLocalBounds();
        if (entries.empty())
        {
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
            g.drawFittedText("No capacity report for this sound", r, juce::Justification::topLeft, 2);
            return;
        }
        const auto rowHeight = juce::jlimit(20, 26, (r.getHeight() - 40) / static_cast<int>(entries.size()));
        for (const auto& entry : entries)
        {
            auto row = r.removeFromTop(rowHeight);
            g.setColour(t.text);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12.5f));
            g.drawText(entry.name, row.removeFromLeft(90), juce::Justification::centredLeft, true);
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(11));
            g.drawText(compactText(entry.used) + " / " + compactText(entry.budget), row.removeFromRight(86),
                       juce::Justification::centredRight, true);
            auto bar = row.reduced(8, rowHeight / 2 - 4).toFloat();
            g.setColour(t.bg);
            g.fillRoundedRectangle(bar, 4);
            const auto amount = entry.budget > 0
                                    ? static_cast<float>(juce::jlimit(0.0, 1.0, entry.used / entry.budget))
                                    : 0.f;
            g.setColour(entry.used > entry.budget ? t.warn : t.accent.withAlpha(0.85f));
            g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * amount), 4);
        }
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f));
        g.drawFittedText(caption, r.withTrimmedTop(6), juce::Justification::topLeft, 3);
    }

private:
    alfa::ui::AlfaLookAndFeel& look;
    std::vector<Entry> entries;
    juce::String caption;
};
} // namespace alfa::tensorhost
