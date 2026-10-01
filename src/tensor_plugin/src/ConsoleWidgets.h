#pragma once
#include "TensorProcessor.h"
#include "alfa/ui/AlfaLookAndFeel.h"
#include "alfa/ui/Visuals.h"

#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <charconv>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

namespace alfa::tensorhost
{
inline juce::String scalarText(double value)
{
    std::array<char, 128> buffer{};
    // The shortest round-trip representation keeps exact doubles without turning
    // ordinary decimal controls into unreadable 17-digit binary artifacts.
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                      std::chars_format::general);
    return result.ec == std::errc{} ? juce::String(std::string(buffer.data(), result.ptr))
                                    : juce::String(value, 17);
}
inline void surface(juce::Graphics& g, const alfa::ui::Theme& t, juce::Rectangle<int> r,
                    const juce::String& title, const juce::String& subtitle = {})
{
    alfa::ui::visuals::panel(g, t, r, title, subtitle);
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
        if (auto* parameter = p.parameters().getParameter(id))
            slider.setDoubleClickReturnValue(true, parameter->convertFrom0to1(parameter->getDefaultValue()));
        attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(p.parameters(),
                                                                                            id, slider);
    }
    void resized() override
    {
        auto r = getLocalBounds();
        name.setBounds(r.removeFromTop(22));
        slider.setBounds(r);
    }
    juce::Slider& control() { return slider; }

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
        units.setBounds(r.removeFromRight(54));
        entry.setBounds(r.removeFromRight(96));
        if (range_)
        {
            name.setBounds(r.removeFromLeft(juce::jmin(200, r.getWidth() / 2)));
            slider.setBounds(r.reduced(8, 0));
        }
        else
            name.setBounds(r);
    }
    void paint(juce::Graphics& g) override
    {
        g.setColour(findColour(juce::TextEditor::outlineColourId).withAlpha(0.25f));
        g.drawHorizontalLine(getHeight() - 1, 8.0f, static_cast<float>(getWidth() - 8));
    }

private:
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
        g.drawText(xId.toUpperCase() + "  " + juce::String(x, 2) + "  /  " + yId.toUpperCase() + "  " +
                       juce::String(y, 2),
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
} // namespace alfa::tensorhost
