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
#include <map>
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
// Parameter names of instances that share a primitive follow one pattern: "Voice 3 partial 5 frequency offset"
// is voice 3, index 5 of "Partial frequency offset". Names without a voice or an index keep them at -1.
struct NamePattern
{
    juce::String key, display, withoutVoice;
    int voice = -1, index = -1;
};
inline NamePattern namePattern(const juce::String& name)
{
    NamePattern result;
    auto rest = name.trim();
    if (rest.startsWithIgnoreCase("voice "))
    {
        const auto after = rest.substring(6).trimStart();
        const auto digits = after.initialSectionContainingOnly("0123456789");
        if (digits.isNotEmpty())
        {
            result.voice = digits.getIntValue();
            rest = after.substring(digits.length()).trimStart();
        }
    }
    if (rest.isEmpty())
        rest = "value";
    result.withoutVoice = rest;
    juce::StringArray words;
    words.addTokens(rest, " ", "");
    juce::StringArray keyWords, displayWords;
    for (const auto& word : words)
        if (result.index < 0 && word.isNotEmpty() && word.containsOnly("0123456789"))
        {
            result.index = word.getIntValue();
            keyWords.add("#");
        }
        else
        {
            keyWords.add(word);
            displayWords.add(word);
        }
    result.key = keyWords.joinIntoString(" ");
    const auto display = displayWords.joinIntoString(" ");
    result.display = display.substring(0, 1).toUpperCase() + display.substring(1);
    return result;
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
            slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 58, 18);
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
            name.setBounds(r.removeFromLeft(34));
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
    // Text set before the row joined the editor took the default look's colours; take the theme's now.
    void parentHierarchyChanged() override { lookAndFeelChanged(); }
    void lookAndFeelChanged() override
    {
        entry.applyColourToAllText(entry.findColour(juce::TextEditor::textColourId));
        units.setColour(juce::Label::textColourId, findColour(juce::Label::textColourId).withAlpha(0.6f));
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
// What the focused control reaches, as controls a person can hold: parameters that differ only by voice
// share one row, and rows that differ only by an index (partial 1, 2, 3 ...) share one profile drawn as
// bars over that index. A row is a bipolar fader; a profile is drawn by dragging across it. Weights are
// the linear sensitivity through the authored matrices (parameter units per unit of the control).
class InfluenceView final : public juce::Component, public juce::SettableTooltipClient
{
public:
    struct Member
    {
        size_t parameter = 0;
        double weight = 0;
        bool locked = false;
    };
    struct Group
    {
        juce::String name, unit;
        int voices = 1;
        std::vector<int> indices;                 // one per column; -1 when the row has no index
        std::vector<std::vector<Member>> columns; // the parameters of each column, one per voice
    };
    explicit InfluenceView(alfa::ui::AlfaLookAndFeel& l)
        : look(l)
    {
        setTooltip("Drag a fader, or draw across a profile, to change how strongly this control moves "
                   "those parameters. Every voice moves together and keeps its own offset.");
    }
    void setGroups(juce::String control, std::vector<Group> list, int reached)
    {
        if (dragging >= 0)
            return;
        title = std::move(control);
        groups = std::move(list);
        reachedCount = reached;
        limits.clear();
        for (const auto& group : groups)
        {
            double largest = 0;
            for (size_t c = 0; c < group.columns.size(); ++c)
                largest = juce::jmax(largest, std::abs(mean(group, c)));
            limits.push_back(niceCeiling(juce::jmax(1e-9, largest * 1.5)));
        }
        scroll = juce::jlimit(0, juce::jmax(0, static_cast<int>(groups.size()) - 1), scroll);
        repaint();
    }
    // Final edits as (parameter, new weight) pairs, sent when a drag ends.
    std::function<void(const std::vector<std::pair<size_t, double>>&)> onEdit;

    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        auto r = getLocalBounds().toFloat();
        rowAreas.clear();
        auto head = r.removeFromTop(22);
        g.setColour(t.text);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13, true));
        const auto titleWidth = juce::GlyphArrangement::getStringWidth(g.getCurrentFont(), title) + 12;
        g.drawText(title, head.removeFromLeft(titleWidth), juce::Justification::centredLeft);
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
        g.drawText(groups.empty() ? juce::String("reaches nothing yet")
                                  : "reaches " + juce::String(reachedCount) + " parameters in " +
                                        juce::String(static_cast<int>(groups.size())) +
                                        (groups.size() == 1 ? " group" : " groups"),
                   head, juce::Justification::centredLeft, true);
        r.removeFromTop(8);
        // Rows that fit at their base height share what is left of the panel: profiles take three
        // shares and grow to 150 px, faders one share up to 54 px, so a short list fills its panel.
        std::vector<std::pair<size_t, float>> fitted;
        float used = 0, shares = 0;
        for (size_t i = static_cast<size_t>(scroll); i < groups.size(); ++i)
        {
            const auto height = groups[i].columns.size() > 1 ? 64.0f : 34.0f;
            if (used + height > r.getHeight())
                break;
            fitted.push_back({i, height});
            used += height;
            shares += groups[i].columns.size() > 1 ? 3.0f : 1.0f;
        }
        const auto spare = (r.getHeight() - (scroll > 0 || fitted.size() + static_cast<size_t>(scroll) < groups.size() ? 16.0f : 0.0f) - used);
        for (auto& [i, height] : fitted)
        {
            const auto profile = groups[i].columns.size() > 1;
            height = juce::jmin(profile ? 150.0f : 54.0f,
                                height + juce::jmax(0.0f, spare) * (profile ? 3.0f : 1.0f) / juce::jmax(1.0f, shares));
            auto row = r.removeFromTop(height);
            rowAreas.push_back({i, row});
            paintGroup(g, t, i, row.reduced(0, 3));
        }
        if (scroll > 0 || rowAreas.size() < groups.size() - static_cast<size_t>(scroll))
        {
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10));
            g.drawText(juce::String(scroll + 1) + "-" + juce::String(scroll + static_cast<int>(rowAreas.size())) +
                           " of " + juce::String(static_cast<int>(groups.size())) + ", scroll for more",
                       getLocalBounds().toFloat().removeFromBottom(14), juce::Justification::centredRight);
        }
    }
    void mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails& wheel) override
    {
        scroll = juce::jlimit(0, juce::jmax(0, static_cast<int>(groups.size()) - 1),
                              scroll + (wheel.deltaY < 0 ? 1 : -1));
        repaint();
    }
    void mouseDown(const juce::MouseEvent& e) override
    {
        dragging = -1;
        for (const auto& [index, area] : rowAreas)
            if (control(index, area).expanded(0, 4).contains(e.position))
                dragging = static_cast<int>(index);
        pending.clear();
        if (dragging >= 0)
            mouseDrag(e);
    }
    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (dragging < 0)
            return;
        const auto index = static_cast<size_t>(dragging);
        juce::Rectangle<float> area;
        for (const auto& [i, row] : rowAreas)
            if (i == index)
                area = control(i, row);
        const auto& group = groups[index];
        const auto limit = limits[index];
        if (group.columns.size() > 1)
        {
            // Draw across the profile: the column under the mouse takes the height of the mouse.
            const auto width = area.getWidth() / static_cast<float>(group.columns.size());
            const auto column = juce::jlimit(0, static_cast<int>(group.columns.size()) - 1,
                                             static_cast<int>((e.position.x - area.getX()) / width));
            const auto value = juce::jlimit(-1.0f, 1.0f, (area.getCentreY() - e.position.y) / (area.getHeight() * 0.5f));
            pending[static_cast<size_t>(column)] = snap(value * limit, limit);
        }
        else
        {
            const auto value = juce::jlimit(-1.0f, 1.0f, (e.position.x - area.getCentreX()) / (area.getWidth() * 0.5f));
            pending[0] = snap(value * limit, limit);
        }
        repaint();
    }
    void mouseUp(const juce::MouseEvent&) override
    {
        if (dragging < 0)
            return;
        const auto& group = groups[static_cast<size_t>(dragging)];
        std::vector<std::pair<size_t, double>> edits;
        for (const auto& [column, value] : pending)
        {
            // Every voice moves by the same amount, so voices keep their offsets from each other.
            const auto shift = value - mean(group, column);
            for (const auto& member : group.columns[column])
                edits.push_back({member.parameter, member.weight + shift});
        }
        dragging = -1;
        pending.clear();
        if (!edits.empty() && onEdit)
            onEdit(edits);
        repaint();
    }
    void mouseDoubleClick(const juce::MouseEvent& e) override
    {
        // Double-click a row to remove what this control does to it.
        for (const auto& [index, area] : rowAreas)
            if (area.contains(e.position) && onEdit)
            {
                std::vector<std::pair<size_t, double>> edits;
                for (const auto& column : groups[index].columns)
                    for (const auto& member : column)
                        edits.push_back({member.parameter, 0.0});
                dragging = -1;
                pending.clear();
                onEdit(edits);
                return;
            }
    }

private:
    alfa::ui::AlfaLookAndFeel& look;
    juce::String title;
    std::vector<Group> groups;
    std::vector<double> limits;
    std::vector<std::pair<size_t, juce::Rectangle<float>>> rowAreas;
    std::map<size_t, double> pending;
    int reachedCount = 0, scroll = 0, dragging = -1;

    static double mean(const Group& group, size_t column)
    {
        double sum = 0;
        for (const auto& member : group.columns[column])
            sum += member.weight;
        return group.columns[column].empty() ? 0.0 : sum / static_cast<double>(group.columns[column].size());
    }
    static double niceCeiling(double value)
    {
        const auto power = std::pow(10.0, std::floor(std::log10(value)));
        for (const auto step : {1.0, 2.0, 5.0, 10.0})
            if (value <= step * power)
                return step * power;
        return 10 * power;
    }
    // Values snap to a hundredth of the fader's reach, and to zero near the centre.
    static double snap(double value, double limit)
    {
        const auto step = limit / 100.0;
        const auto snapped = std::round(value / step) * step;
        return std::abs(snapped) < step * 1.5 ? 0.0 : snapped;
    }
    double shown(size_t index, size_t column) const
    {
        if (dragging == static_cast<int>(index))
            if (const auto it = pending.find(column); it != pending.end())
                return it->second;
        return mean(groups[index], column);
    }
    static juce::Rectangle<float> control(size_t, juce::Rectangle<float> row)
    {
        return row.reduced(0, 3).withTrimmedLeft(row.getWidth() * 0.36f).withTrimmedRight(70);
    }
    void paintGroup(juce::Graphics& g, const alfa::ui::Theme& t, size_t index, juce::Rectangle<float> row)
    {
        const auto& group = groups[index];
        const auto limit = limits[index];
        const auto profile = group.columns.size() > 1;
        bool locked = true;
        for (const auto& column : group.columns)
            for (const auto& member : column)
                locked = locked && member.locked;
        auto label = row.withWidth(row.getWidth() * 0.36f).reduced(2, 0);
        auto lines = profile ? label.withHeight(34.0f) : label;
        g.setColour(locked ? t.text2 : t.text);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12, true));
        g.drawText(group.name, lines.removeFromTop(lines.getHeight() * 0.55f), juce::Justification::bottomLeft, true);
        juce::StringArray detail;
        if (group.voices > 1)
            detail.add(juce::String(group.voices) + " voices");
        if (profile)
            detail.add("index " + juce::String(group.indices.front()) + juce::String(juce::CharPointer_UTF8("\xe2\x80\x93")) +
                       juce::String(group.indices.back()));
        if (group.unit.isNotEmpty() && group.unit != "linear")
            detail.add(group.unit);
        if (locked)
            detail.add("locked");
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10.5f));
        g.drawText(detail.joinIntoString("  ·  "), lines, juce::Justification::topLeft, true);
        const auto area = control(index, row.expanded(0, 3));
        // Value: the single weight, or the strongest of the profile.
        double value = 0;
        for (size_t c = 0; c < group.columns.size(); ++c)
            if (std::abs(shown(index, c)) > std::abs(value))
                value = shown(index, c);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f, true));
        g.setColour(std::abs(value) < 1e-12 ? t.text2 : weightColour(t, value));
        g.drawText((value > 0 ? "+" : "") + compactText(value),
                   juce::Rectangle<float>(area.getRight() + 6, area.getY(), 64, area.getHeight()),
                   juce::Justification::centredRight);
        const auto alpha = locked ? 0.35f : 0.95f;
        if (!profile)
        {
            const auto groove = area.withSizeKeepingCentre(area.getWidth(), 4);
            alfa::ui::visuals::recessed(g, t, groove, 2);
            g.setColour(t.text2.withAlpha(0.5f));
            g.fillRect(juce::Rectangle<float>(1, 10).withCentre(groove.getCentre()));
            const auto x = groove.getCentreX() + static_cast<float>(value / limit) * groove.getWidth() * 0.5f;
            g.setColour(weightColour(t, value).withAlpha(alpha));
            g.fillRoundedRectangle(juce::Rectangle<float>(juce::jmin(x, groove.getCentreX()), groove.getY(),
                                                          std::abs(x - groove.getCentreX()), 4),
                                   2);
            alfa::ui::visuals::raised(g, t, juce::Rectangle<float>(12, 12).withCentre({x, groove.getCentreY()}), 6);
            return;
        }
        alfa::ui::visuals::recessed(g, t, area, 5);
        g.setColour(alfa::ui::visuals::gridColour(t).withMultipliedAlpha(2.5f));
        g.drawHorizontalLine(static_cast<int>(area.getCentreY()), area.getX() + 4, area.getRight() - 4);
        const auto width = area.getWidth() / static_cast<float>(group.columns.size());
        for (size_t c = 0; c < group.columns.size(); ++c)
        {
            const auto v = static_cast<float>(shown(index, c) / limit);
            const auto bar = juce::Rectangle<float>(area.getX() + width * static_cast<float>(c), area.getY(), width,
                                                    area.getHeight())
                                 .reduced(juce::jmin(3.0f, width * 0.2f), 4);
            const auto height = bar.getHeight() * 0.5f * std::abs(v);
            g.setColour(weightColour(t, v).withAlpha(std::abs(v) < 1e-6f ? 0.0f : alpha));
            g.fillRoundedRectangle(v > 0 ? bar.withTop(bar.getCentreY() - height).withBottom(bar.getCentreY())
                                         : bar.withTop(bar.getCentreY()).withHeight(height),
                                   1.5f);
        }
    }
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
    static constexpr int minimumRow = 20, maximumRow = 64;
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
        return columnLabels.isEmpty() ? 0 : juce::jlimit(14, 160, cellArea().getWidth() / columnLabels.size());
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
        const auto rowHeight = juce::jmax(34, r.getHeight() / static_cast<int>(layers.size()));
        for (size_t l = 0; l < layers.size(); ++l)
        {
            auto row = r.removeFromTop(rowHeight);
            const auto& layer = layers[l];
            g.setColour(static_cast<int>(l) == pageLayer ? t.accent : t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11, true));
            g.drawText("LAYER " + juce::String(static_cast<int>(l)) + "   " +
                           juce::String(static_cast<int>(layer.values.size())) + " controls",
                       row.removeFromTop(16), juce::Justification::centredLeft, false);
            auto bars = row.reduced(0, 4).withTrimmedBottom(6).toFloat();
            const auto count = std::max<size_t>(1, layer.values.size());
            // Slim wells spread across the row, at most 40 px apart; the wells stay at most 14 px wide.
            const auto w = slot(bars.getWidth(), count);
            const auto well = juce::jlimit(2.0f, 14.0f, w * 0.5f);
            for (size_t i = 0; i < layer.values.size(); ++i)
            {
                auto box = juce::Rectangle<float>(bars.getX() + w * static_cast<float>(i), bars.getY(), w,
                                                  bars.getHeight())
                               .withSizeKeepingCentre(well, bars.getHeight());
                const auto onPage = static_cast<int>(l) == pageLayer && static_cast<int>(i) >= pageBegin &&
                                    static_cast<int>(i) < pageEnd;
                g.setColour(alfa::ui::visuals::screenColour(t));
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
                    g.setColour(t.accent.withAlpha(0.8f));
                    g.fillRoundedRectangle(box.withY(box.getBottom() + 3).withHeight(2), 1);
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
    static float slot(float width, size_t count)
    {
        return juce::jmin(40.0f, width / static_cast<float>(std::max<size_t>(1, count)));
    }
    std::pair<int, int> hitTestBar(juce::Point<int> p) const
    {
        if (layers.empty())
            return {-1, -1};
        const auto rowHeight = juce::jmax(34, getHeight() / static_cast<int>(layers.size()));
        const auto l = p.y / rowHeight;
        if (l < 0 || l >= static_cast<int>(layers.size()) || p.y % rowHeight < 16)
            return {-1, -1};
        const auto count = static_cast<int>(layers[static_cast<size_t>(l)].values.size());
        if (count == 0 || getWidth() <= 0)
            return {-1, -1};
        const auto w = slot(static_cast<float>(getWidth()), static_cast<size_t>(count));
        const auto index = static_cast<int>(static_cast<float>(p.x) / w);
        return index < count ? std::pair<int, int>{l, index} : std::pair<int, int>{-1, -1};
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
    // Height that shows every entry and the caption at this width.
    int preferredHeight(int width) const
    {
        if (entries.empty())
            return 40;
        const auto columns = static_cast<size_t>(width > 300 ? 3 : 2);
        if (exact())
            return static_cast<int>((entries.size() + columns - 1) / columns) * 54 + 44;
        return static_cast<int>(entries.size()) * 24 + 52;
    }
    // Smallest height that still shows every tile, without the caption.
    int minimumHeight(int width) const
    {
        const auto columns = static_cast<size_t>(width > 300 ? 3 : 2);
        return exact() ? static_cast<int>((entries.size() + columns - 1) / columns) * 44 : preferredHeight(width);
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
        // When the budget is exactly what the loaded sound uses, bars would always be full: show counts.
        if (exact())
        {
            const auto columns = r.getWidth() > 300 ? 3 : 2;
            const auto cellWidth = r.getWidth() / columns;
            const auto rows = static_cast<int>((entries.size() + static_cast<size_t>(columns) - 1) / static_cast<size_t>(columns));
            // Tiles grow with spare height so a roomy panel is not left half empty.
            // The caption goes first when the height is short; tiles then shrink to 44 px.
            const auto caption = r.getHeight() >= rows * 54 + 44 ? 44 : 0;
            const auto pitch = juce::jlimit(44, 110, (r.getHeight() - caption) / juce::jmax(1, rows));
            for (size_t i = 0; i < entries.size(); ++i)
            {
                const auto cell = juce::Rectangle<int>(r.getX() + static_cast<int>(i % static_cast<size_t>(columns)) * cellWidth,
                                                       r.getY() + static_cast<int>(i / static_cast<size_t>(columns)) * pitch,
                                                       cellWidth - 8, pitch - 8)
                                      .toFloat();
                alfa::ui::visuals::recessed(g, t, cell, 6);
                g.setColour(t.text);
                const auto text = cell.withSizeKeepingCentre(cell.getWidth(), 38).reduced(10, 0);
                g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(pitch > 70 ? 20.0f : 17.0f, true));
                g.drawText(compactText(entries[i].used), text.withTrimmedBottom(16), juce::Justification::bottomLeft);
                g.setColour(t.text2);
                g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10.5f));
                g.drawText(entries[i].name, text.withTrimmedTop(24), juce::Justification::topLeft);
            }
            r.removeFromTop(rows * pitch);
            if (caption == 0)
                return;
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f));
            g.drawFittedText(caption + " The sound uses exactly its budget.", r.withTrimmedTop(6),
                             juce::Justification::topLeft, 3);
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
    bool exact() const
    {
        return std::all_of(entries.begin(), entries.end(), [](const Entry& e)
                           { return std::abs(e.used - e.budget) < 0.5; });
    }
};
// How strongly each control of one matrix drives the next layer: per source column, the summed size of
// its coefficients, amber for what it raises and blue for what it lowers, with the matrix's totals above.
class ReachView final : public juce::Component, public juce::SettableTooltipClient
{
public:
    explicit ReachView(alfa::ui::AlfaLookAndFeel& l)
        : look(l)
    {
        setTooltip("Click a control to select its column in the matrix");
    }
    std::function<void(int source)> onSelect;
    void setMatrix(const std::vector<double>& values, const juce::StringArray& rows, const juce::StringArray& columns)
    {
        sources = columns;
        cells = static_cast<size_t>(rows.size() * columns.size());
        const auto count = static_cast<size_t>(columns.size());
        raise.assign(count, 0.0);
        lower.assign(count, 0.0);
        targets.assign(count, 0);
        nonZero = 0;
        largest = 0;
        for (size_t i = 0; i < values.size() && count > 0; ++i)
        {
            const auto v = values[i];
            const auto c = i % count;
            if (v == 0.0)
                continue;
            (v > 0 ? raise[c] : lower[c]) += std::abs(v);
            ++targets[c];
            ++nonZero;
            largest = juce::jmax(largest, std::abs(v));
        }
        repaint();
    }
    void select(int source)
    {
        selected = source;
        repaint();
    }
    // Height that lists every control in a compact row with the one-line totals.
    int preferredHeight() const { return 22 + 16 * sources.size(); }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        auto r = getLocalBounds().toFloat();
        if (sources.isEmpty())
        {
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
            g.drawText("No matrix in this layer", r, juce::Justification::topLeft);
            return;
        }
        // Totals as three readouts when every control still gets a 20 px row, as one line when 16 px rows
        // fit, and left out when the rows need all the height.
        const auto count = static_cast<float>(sources.size());
        if (r.getHeight() >= 98.0f + 20.0f * count)
        {
            paintTotals(g, t, r.removeFromTop(46));
            r.removeFromTop(12);
        }
        else if (r.getHeight() >= 22.0f + 16.0f * count)
        {
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
            g.drawText(juce::String(static_cast<int>(nonZero)) + " of " + juce::String(static_cast<int>(cells)) +
                           " coefficients  ·  largest " + compactText(largest),
                       r.removeFromTop(18), juce::Justification::centredLeft, true);
            r.removeFromTop(4);
        }
        if (r.getHeight() >= 98.0f + 20.0f * count - 58.0f)
        {
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10, true));
            g.drawText("DRIVE PER CONTROL", r.removeFromTop(16), juce::Justification::centredLeft);
            r.removeFromTop(4);
        }
        double strongest = 1e-12;
        for (size_t c = 0; c < raise.size(); ++c)
            strongest = juce::jmax(strongest, raise[c] + lower[c]);
        rowHeight = juce::jlimit(14.0f, 30.0f, r.getHeight() / count);
        rowsTop = r.getY();
        for (int c = 0; c < sources.size() && r.getHeight() >= rowHeight; ++c)
        {
            auto row = r.removeFromTop(rowHeight);
            const auto index = static_cast<size_t>(c);
            g.setColour(c == selected ? t.accent : t.text);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f, c == selected));
            g.drawText(sources[c], row.removeFromLeft(78), juce::Justification::centredLeft, true);
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10.5f));
            g.drawText(targets[index] == 0 ? juce::String("none")
                                           : compactText(raise[index] + lower[index]) + "  ·  " +
                                                 juce::String(targets[index]) + (targets[index] == 1 ? " target" : " targets"),
                       row.removeFromRight(98), juce::Justification::centredRight, true);
            auto groove = row.reduced(6, 0).withSizeKeepingCentre(row.getWidth() - 12, juce::jmin(8.0f, rowHeight - 8));
            alfa::ui::visuals::recessed(g, t, groove, groove.getHeight() / 2);
            const auto up = groove.getWidth() * static_cast<float>(raise[index] / strongest);
            const auto down = groove.getWidth() * static_cast<float>(lower[index] / strongest);
            g.setColour(t.accent.withAlpha(0.85f));
            g.fillRoundedRectangle(groove.withWidth(up), groove.getHeight() / 2);
            g.setColour(t.mod.withAlpha(0.85f));
            g.fillRoundedRectangle(groove.withX(groove.getX() + up).withWidth(down), groove.getHeight() / 2);
        }
    }
    void paintTotals(juce::Graphics& g, const alfa::ui::Theme& t, juce::Rectangle<float> tiles) const
    {
        const std::array<std::pair<juce::String, juce::String>, 3> totals{
            {{"Coefficients", juce::String(static_cast<int>(nonZero)) + " of " + juce::String(static_cast<int>(cells))},
             {"Largest", compactText(largest)},
             {"Density", juce::String(juce::roundToInt(cells ? 100.0 * static_cast<double>(nonZero) / static_cast<double>(cells) : 0.0)) + " %"}}};
        const auto tileWidth = tiles.getWidth() / 3.0f;
        for (const auto& [name, value] : totals)
        {
            const auto tile = tiles.removeFromLeft(tileWidth).withTrimmedRight(8);
            alfa::ui::visuals::recessed(g, t, tile, 6);
            g.setColour(t.text);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(15, true));
            g.drawText(value, tile.reduced(10, 4).withTrimmedBottom(16), juce::Justification::bottomLeft, true);
            g.setColour(t.text2);
            g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10.5f));
            g.drawText(name, tile.reduced(10, 5).withTrimmedTop(24), juce::Justification::topLeft, true);
        }
    }
    void mouseDown(const juce::MouseEvent& e) override
    {
        const auto c = static_cast<int>((e.position.y - rowsTop) / juce::jmax(1.0f, rowHeight));
        if (e.position.y >= rowsTop && c >= 0 && c < sources.size() && onSelect)
            onSelect(c);
    }

private:
    alfa::ui::AlfaLookAndFeel& look;
    juce::StringArray sources;
    std::vector<double> raise, lower;
    std::vector<int> targets;
    size_t cells = 0, nonZero = 0;
    double largest = 0;
    int selected = -1;
    float rowHeight = 20, rowsTop = 0;
};
} // namespace alfa::tensorhost
