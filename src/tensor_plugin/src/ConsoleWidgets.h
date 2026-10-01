#pragma once
#include "TensorProcessor.h"
#include "alfa/ui/AlfaLookAndFeel.h"
#include "alfa/ui/Visuals.h"

#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>

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
    ParameterKnob(Processor& p, const juce::String& id, const juce::String& label, bool percent = false)
        : processor(p)
        , parameterId(id)
    {
        name.setText(label, juce::dontSendNotification);
        name.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(name);
        alfa::ui::visuals::rotaryGesture(slider,
                                         p.uiSettings().getProperty("knob_gesture", "vertical").toString());
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 86, 22);
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
        if (!percent && !id.contains("gain"))
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
        auto r = getLocalBounds();
        name.setBounds(r.removeFromTop(22));
        slider.setBounds(r);
    }
    juce::Slider& control() { return slider; }
    juce::Label& title() { return name; }
    std::function<void()> onTouched;
    void mouseDown(const juce::MouseEvent&) override
    {
        if (onTouched)
            onTouched();
    }

private:
    Processor& processor;
    juce::String parameterId;
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
        units.setJustificationType(juce::Justification::centredRight);
        addAndMakeVisible(units);
        entry.setText(scalarText(value));
        entry.setJustification(juce::Justification::centredRight);
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
        units.setBounds(r.removeFromRight(48));
        entry.setBounds(r.removeFromRight(80).reduced(0, 3));
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

private:
    const alfa::ui::Theme* weightTheme = nullptr;
    double weightScale = 1;
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
class LiveScope final : public juce::Component
{
public:
    LiveScope(Processor& p, alfa::ui::AlfaLookAndFeel& l)
        : processor(p)
        , look(l)
    {
        bands.fill(-90);
    }
    void paint(juce::Graphics& g) override
    {
        const auto& t = look.theme();
        surface(g, t, getLocalBounds(), "OUTPUT");
        alfa::ui::visuals::badge(g, juce::Rectangle<float>(static_cast<float>(getWidth() - 132), 5, 120, 18),
                                 "CH 1 / FFT 2048", t.mod);
        auto area = getLocalBounds().reduced(16, 12).withTrimmedTop(24);
        auto spectrumArea = area.removeFromBottom(area.getHeight() / 3);
        std::array<float, 2048> samples{};
        const auto captureCursor = processor.waveformCursor.load(std::memory_order_relaxed) % samples.size();
        for (size_t i = 0; i < samples.size(); ++i)
            samples[i] = processor.waveformSnapshot[(captureCursor + i) % samples.size()].load(
                std::memory_order_relaxed);
        alfa::ui::visuals::waveform(g, area.toFloat(), t, samples, t.mod, false);
        std::array<float, 4096> data{};
        std::copy(samples.begin(), samples.end(), data.begin());
        window.multiplyWithWindowingTable(data.data(), 2048);
        fft.performFrequencyOnlyForwardTransform(data.data());
        alfa::ui::visuals::logSpectrum(data, 2048, static_cast<float>(processor.getSampleRate()), bands);
        alfa::ui::visuals::spectrum(g, spectrumArea.reduced(0, 3).toFloat(), t, bands);
    }

private:
    Processor& processor;
    alfa::ui::AlfaLookAndFeel& look;
    std::array<float, 64> bands{};
    juce::dsp::FFT fft{11};
    juce::dsp::WindowingFunction<float> window{2048, juce::dsp::WindowingFunction<float>::hann};
};
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
        auto area = getLocalBounds().reduced(8);
        g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(10));
        for (int pair = 0; pair < 2; ++pair)
        {
            auto bar = area.removeFromLeft(area.getWidth() / (2 - pair)).reduced(4, 0);
            g.setColour(t.text2);
            g.drawText(pair == 0 ? "IN" : "OUT", bar.removeFromTop(16), juce::Justification::centred);
            const auto& peaks = pair == 0 ? processor.inputPeak : processor.outputPeak;
            alfa::ui::visuals::meter(g, bar.toFloat(), t, peaks[0].load(), peaks[1].load(),
                                     pair == 0 ? t.mod : t.accent);
        }
    }

private:
    Processor& processor;
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
            auto x = r.getX() + r.getWidth() * i / 4;
            auto y = r.getY() + r.getHeight() * i / 4;
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
            g.drawText(juce::String(c), column(c).withY(0).withHeight(headerHeight),
                       juce::Justification::centred, false);
        }
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(juce::jlimit(11.0f, 13.0f, rows * 0.55f)));
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
                        g.setFont(
                            alfa::ui::AlfaLookAndFeel::labelFont(juce::jlimit(11.0f, 13.0f, rows * 0.55f)));
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
