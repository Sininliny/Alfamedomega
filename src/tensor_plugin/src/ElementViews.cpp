#include "ElementViews.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <nlohmann/json.hpp>
#include <numbers>

namespace alfa::tensorhost
{
namespace
{
using JSON = nlohmann::json;
juce::String js(const std::string& s)
{
    return juce::String(s);
}
juce::String timeText(double seconds)
{
    return seconds < 1 ? compactText(seconds * 1000) + " ms" : compactText(seconds) + " s";
}
juce::String hertzText(double hz)
{
    return hz >= 1000 ? compactText(hz / 1000) + " kHz" : compactText(hz) + " Hz";
}
// Card shared by every element: soft vertical gradient, hairline border, title row with family dot.
void card(juce::Graphics& g, const alfa::ui::Theme& t, juce::Rectangle<float> r, juce::Colour identity,
          const juce::String& title, const juce::String& readout)
{
    g.setGradientFill(juce::ColourGradient(t.panel.brighter(0.04f), r.getX(), r.getY(), t.panel.darker(0.06f),
                                           r.getX(), r.getBottom(), false));
    g.fillRoundedRectangle(r, 10);
    g.setColour(t.line);
    g.drawRoundedRectangle(r.reduced(0.5f), 10, 1);
    g.setColour(juce::Colours::white.withAlpha(0.03f));
    g.drawHorizontalLine(static_cast<int>(r.getY()) + 1, r.getX() + 10, r.getRight() - 10);
    auto row = r.reduced(14, 0).withHeight(28).withTrimmedTop(8);
    g.setColour(identity);
    g.fillEllipse(juce::Rectangle<float>(7, 7).withCentre({row.getX() + 3.5f, row.getCentreY()}));
    g.setColour(t.text);
    const auto titleFont = alfa::ui::AlfaLookAndFeel::labelFont(12.5f, true);
    g.setFont(titleFont);
    g.drawText(title, row.withTrimmedLeft(14), juce::Justification::centredLeft, true);
    // The readout takes what the title leaves; it tightens, then gives way rather than overlap.
    auto free = row.withTrimmedLeft(14 + juce::GlyphArrangement::getStringWidth(titleFont, title) + 12);
    g.setColour(t.text2);
    for (const auto& [size, text] :
         {std::pair{11.0f, readout}, std::pair{10.0f, readout}, std::pair{10.0f, readout.replace("  ", " ")}})
    {
        const auto font = alfa::ui::AlfaLookAndFeel::monoFont(size);
        if (juce::GlyphArrangement::getStringWidth(font, text) <= free.getWidth())
        {
            g.setFont(font);
            g.drawText(text, free, juce::Justification::centredRight, false);
            break;
        }
    }
}
float logX(juce::Rectangle<float> r, double hz)
{
    const auto fraction = std::log(juce::jlimit(20.0, 20000.0, hz) / 20.0) / std::log(1000.0);
    return r.getX() + r.getWidth() * static_cast<float>(fraction);
}
double hzAt(juce::Rectangle<float> r, float x)
{
    return 20.0 *
           std::pow(1000.0, juce::jlimit(0.0, 1.0, static_cast<double>((x - r.getX()) / r.getWidth())));
}
void frequencyGrid(juce::Graphics& g, const alfa::ui::Theme& t, juce::Rectangle<float> r)
{
    g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(9.5f));
    for (const auto hz : {100.0, 1000.0, 10000.0})
    {
        const auto x = logX(r, hz);
        g.setColour(t.line.withAlpha(0.6f));
        g.drawVerticalLine(static_cast<int>(x), r.getY(), r.getBottom());
        g.setColour(t.text2.withAlpha(0.7f));
        g.drawText(hz >= 1000 ? juce::String(static_cast<int>(hz / 1000)) + "k"
                              : juce::String(static_cast<int>(hz)),
                   juce::Rectangle<float>(x + 3, r.getBottom() - 13, 30, 12), juce::Justification::left);
    }
}
double lowpassDecibels(double hz, double cutoff, double q, double rate)
{
    // RBJ low-pass, as filter.biquad_lowpass computes it.
    const auto w0 = 2 * std::numbers::pi * juce::jlimit(1.0, rate * 0.49, cutoff) / rate;
    const auto alpha = std::sin(w0) / (2 * juce::jmax(1e-3, q));
    const auto c = std::cos(w0);
    const std::array<double, 3> b{(1 - c) / 2, 1 - c, (1 - c) / 2}, a{1 + alpha, -2 * c, 1 - alpha};
    const auto z = std::polar(1.0, -2 * std::numbers::pi * juce::jmin(hz, rate * 0.499) / rate);
    const auto h = (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z);
    return 20 * std::log10(juce::jmax(1e-9, std::abs(h)));
}
} // namespace

juce::String instanceTitle(const nlohmann::json& sound, size_t instance)
{
    juce::StringArray common;
    int parameters = 0;
    for (const auto& meta : sound.value("parameter_metadata", JSON::array()))
    {
        if (meta.value("instance", size_t(-1)) != instance)
            continue;
        const auto words = juce::StringArray::fromTokens(js(meta.value("name", std::string{})), " ", "");
        if (parameters++ == 0)
            common = words;
        else
        {
            int shared = 0;
            while (shared < common.size() && shared < words.size() && common[shared] == words[shared])
                ++shared;
            common.removeRange(shared, common.size());
        }
    }
    if (parameters > 1 && !common.isEmpty())
        return common.joinIntoString(" ");
    const auto& instances = sound.at("instances");
    if (instance >= instances.size())
        return "Instance " + juce::String(static_cast<int>(instance));
    return js(instances[instance].at("primitive").get<std::string>())
        .fromFirstOccurrenceOf(".", false, false)
        .replaceCharacter('_', ' ');
}

ElementView::ElementView(alfa::ui::AlfaLookAndFeel& look, juce::String primitive, juce::String title,
                         std::vector<Instance> instances, std::vector<std::array<double, 2>> ranges,
                         std::vector<juce::String> names)
    : look_(look)
    , primitive_(std::move(primitive))
    , title_(std::move(title))
    , kind_(kindOf(primitive_))
    , instances_(std::move(instances))
    , ranges_(std::move(ranges))
    , names_(std::move(names))
{
}

ElementView::Kind ElementView::kindOf(const juce::String& primitive)
{
    if (primitive == "control.adsr")
        return Kind::Envelope;
    if (primitive == "filter.biquad_lowpass")
        return Kind::Lowpass;
    if (primitive == "osc.additive")
        return Kind::Partials;
    if (primitive == "time.modulated_delay")
        return Kind::Delay;
    if (primitive == "spatial.stereo_width")
        return Kind::Width;
    if (primitive == "time.fdn_reverb")
        return Kind::Reverb;
    return Kind::Generic;
}

void ElementView::setBase(const std::vector<double>& parameters)
{
    base_.clear();
    for (const auto& instance : instances_)
    {
        std::vector<double> values;
        for (size_t p = 0; p < instance.count; ++p)
            values.push_back(instance.offset + p < parameters.size() ? parameters[instance.offset + p] : 0.0);
        base_.push_back(std::move(values));
    }
    repaint();
}

void ElementView::setLive(const std::vector<float>& parameters, const std::vector<float>& levels, double rate)
{
    sampleRate_ = rate > 0 ? rate : 48000;
    levels_.clear();
    for (const auto& instance : instances_)
        levels_.push_back(instance.index < levels.size() ? levels[instance.index] : 0.0f);
    live_.clear();
    if (!parameters.empty())
        for (const auto& instance : instances_)
        {
            if (instance.offset + instance.count > parameters.size())
            {
                live_.clear();
                break;
            }
            live_.emplace_back(parameters.begin() + static_cast<std::ptrdiff_t>(instance.offset),
                               parameters.begin() +
                                   static_cast<std::ptrdiff_t>(instance.offset + instance.count));
        }
    repaint();
}

double ElementView::value(size_t instance, size_t parameter, bool authored) const
{
    for (const auto& [p, v] : pending_)
        if (p == parameter)
            return v;
    const auto& source = !authored && live_.size() == instances_.size() ? live_ : base_;
    if (instance < source.size() && parameter < source[instance].size())
        return source[instance][parameter];
    return 0;
}

double ElementView::clampParameter(size_t parameter, double v) const
{
    if (parameter < ranges_.size() && ranges_[parameter][1] > ranges_[parameter][0])
        return juce::jlimit(ranges_[parameter][0], ranges_[parameter][1], v);
    return v;
}

juce::Rectangle<float> ElementView::plot() const
{
    return getLocalBounds().toFloat().reduced(14, 10).withTrimmedTop(26);
}

void ElementView::paint(juce::Graphics& g)
{
    const auto& t = look_.theme();
    const auto identity = familyColour(t, primitive_);
    juce::String readout;
    const auto v = [&](size_t p)
    {
        return value(0, p);
    };
    switch (kind_)
    {
    case Kind::Envelope:
        readout = "A " + timeText(v(1)) + "  D " + timeText(v(2)) + "  S " + compactText(v(3)) + "  R " +
                  timeText(v(4));
        break;
    case Kind::Lowpass:
        readout = hertzText(v(0)) + "  Q " + compactText(v(1));
        break;
    case Kind::Delay:
        readout = timeText(v(0)) + "  fb " + compactText(v(4));
        break;
    case Kind::Width:
        readout = "width " + compactText(v(0));
        break;
    case Kind::Reverb:
        readout = "feedback " + compactText(v(0));
        break;
    default:
        break;
    }
    const auto count = instances_.size();
    card(g, t, getLocalBounds().toFloat(), identity,
         title_ + (count > 1 ? juce::String(juce::CharPointer_UTF8(" \xc3\x97")) +
                                   juce::String(static_cast<int>(count))
                             : juce::String()),
         readout);
    auto area = plot();
    g.setColour(t.bg.withAlpha(0.55f));
    g.fillRoundedRectangle(area, 6);
    area = area.reduced(6, 6);
    switch (kind_)
    {
    case Kind::Envelope:
        paintEnvelope(g, area);
        break;
    case Kind::Lowpass:
        paintLowpass(g, area);
        break;
    case Kind::Partials:
        paintPartials(g, area);
        break;
    case Kind::Delay:
        paintDelay(g, area);
        break;
    case Kind::Width:
        paintWidth(g, area);
        break;
    case Kind::Reverb:
        paintReverb(g, area);
        break;
    case Kind::Generic:
        paintGeneric(g, area);
        break;
    }
    // Handles edit the authored values; they sit on the first instance's curve.
    for (const auto& h : handles())
    {
        g.setColour(t.bg);
        g.fillEllipse(juce::Rectangle<float>(13, 13).withCentre(h));
        g.setColour(identity);
        g.drawEllipse(juce::Rectangle<float>(13, 13).withCentre(h), 2.0f);
    }
}

std::vector<juce::Point<float>> ElementView::handles() const
{
    auto r = plot().reduced(6, 6);
    if (kind_ == Kind::Envelope && !base_.empty())
    {
        const auto a = value(0, 1, true), d = value(0, 2, true), s = value(0, 3, true),
                   rel = value(0, 4, true);
        const auto hold = juce::jmax(0.05, 0.25 * (a + d + rel));
        const auto total = a + d + hold + rel;
        const auto x = [&](double time)
        {
            return r.getX() + r.getWidth() * static_cast<float>(time / total);
        };
        const auto y = [&](double level)
        {
            return r.getBottom() - r.getHeight() * static_cast<float>(juce::jlimit(0.0, 1.0, level));
        };
        return {{x(a), y(1)}, {x(a + d), y(s)}, {x(total), y(0)}};
    }
    if (kind_ == Kind::Lowpass && !base_.empty())
    {
        const auto cutoff = value(0, 0, true), q = value(0, 1, true);
        const auto db = lowpassDecibels(cutoff, cutoff, q, sampleRate_);
        return {{logX(r, cutoff), r.getY() + r.getHeight() * static_cast<float>((18 - db) / 48)}};
    }
    return {};
}

void ElementView::paintEnvelope(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto colour = familyColour(t, primitive_);
    // One time scale for every voice: the first voice's stages plus a sustain hold.
    const auto a0 = value(0, 1), d0 = value(0, 2), r0 = value(0, 4);
    const auto hold = juce::jmax(0.05, 0.25 * (a0 + d0 + r0));
    const auto total = juce::jmax(1e-6, a0 + d0 + hold + r0);
    const auto x = [&](double time)
    {
        return r.getX() + r.getWidth() * static_cast<float>(juce::jlimit(0.0, 1.0, time / total));
    };
    const auto y = [&](double level)
    {
        return r.getBottom() - r.getHeight() * static_cast<float>(juce::jlimit(0.0, 1.0, level));
    };
    g.setColour(t.line.withAlpha(0.5f));
    g.drawHorizontalLine(static_cast<int>(y(0.5)), r.getX(), r.getRight());
    for (size_t k = instances_.size(); k-- > 0;)
    {
        const auto a = value(k, 1), d = value(k, 2), s = value(k, 3), rel = value(k, 4);
        juce::Path p;
        p.startNewSubPath(x(0), y(0));
        p.lineTo(x(a), y(1));
        p.lineTo(x(a + d), y(s));
        p.lineTo(x(a + d + hold), y(s));
        p.lineTo(x(a + d + hold + rel), y(0));
        if (k == 0)
        {
            auto fill = p;
            fill.closeSubPath();
            g.setGradientFill(juce::ColourGradient(colour.withAlpha(0.32f), r.getX(), r.getY(),
                                                   colour.withAlpha(0.02f), r.getX(), r.getBottom(), false));
            g.fillPath(fill);
        }
        g.setColour(colour.withAlpha(k == 0 ? 1.0f : 0.35f));
        g.strokePath(p, juce::PathStrokeType(k == 0 ? 2.2f : 1.2f, juce::PathStrokeType::curved,
                                             juce::PathStrokeType::rounded));
    }
    // Each sounding voice's envelope level in the last block, as a short mark at the right edge.
    for (size_t k = 0; k < levels_.size(); ++k)
        if (levels_[k] > 0.001f)
        {
            const auto yy = y(levels_[k]);
            g.setColour(colour.withAlpha(0.85f));
            g.fillRoundedRectangle(r.getRight() - 16, yy - 1.5f, 16, 3, 1.5f);
        }
}

void ElementView::paintLowpass(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto colour = familyColour(t, primitive_);
    frequencyGrid(g, t, r);
    const auto y = [&](double db)
    {
        return r.getY() + r.getHeight() * static_cast<float>((18 - db) / 48);
    };
    g.setColour(t.line.withAlpha(0.8f));
    g.drawHorizontalLine(static_cast<int>(y(0)), r.getX(), r.getRight());
    for (size_t k = instances_.size(); k-- > 0;)
    {
        const auto cutoff = value(k, 0), q = value(k, 1);
        juce::Path p;
        for (int i = 0; i <= 160; ++i)
        {
            const auto xx = r.getX() + r.getWidth() * static_cast<float>(i) / 160.0f;
            const auto yy = juce::jlimit(r.getY(), r.getBottom(),
                                         y(lowpassDecibels(hzAt(r, xx), cutoff, q, sampleRate_)));
            i == 0 ? p.startNewSubPath(xx, yy) : p.lineTo(xx, yy);
        }
        if (k == 0)
        {
            auto fill = p;
            fill.lineTo(r.getRight(), r.getBottom());
            fill.lineTo(r.getX(), r.getBottom());
            fill.closeSubPath();
            g.setGradientFill(juce::ColourGradient(colour.withAlpha(0.28f), r.getX(), r.getY(),
                                                   colour.withAlpha(0.02f), r.getX(), r.getBottom(), false));
            g.fillPath(fill);
        }
        g.setColour(colour.withAlpha(k == 0 ? 1.0f : 0.35f));
        g.strokePath(p, juce::PathStrokeType(k == 0 ? 2.2f : 1.2f));
    }
}

void ElementView::paintPartials(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto colour = familyColour(t, primitive_);
    frequencyGrid(g, t, r);
    double scale = 1e-9;
    for (size_t p = 1; p < ranges_.size(); p += 2)
        scale = juce::jmax(scale, ranges_[p][1]);
    for (size_t k = 0; k < instances_.size(); ++k)
        for (size_t p = 1; p < instances_[k].count; p += 2)
            scale = juce::jmax(scale, value(k, p));
    int sounding = 0;
    for (size_t k = 0; k < instances_.size(); ++k)
    {
        bool any = false;
        for (size_t p = 0; p + 1 < instances_[k].count; p += 2)
        {
            const auto hz = value(k, p), amplitude = value(k, p + 1);
            if (hz <= 0 || amplitude <= 0)
                continue;
            any = true;
            const auto x = logX(r, hz);
            const auto h =
                r.getHeight() * static_cast<float>(juce::jlimit(0.0, 1.0, std::sqrt(amplitude / scale)));
            g.setGradientFill(juce::ColourGradient(colour, x, r.getBottom() - h, colour.withAlpha(0.15f), x,
                                                   r.getBottom(), false));
            g.fillRoundedRectangle(x - 1.5f, r.getBottom() - h, 3, h, 1.5f);
        }
        sounding += any ? 1 : 0;
    }
    g.setColour(t.text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f));
    if (sounding == 0)
        g.drawText("Play a note: each voice's partials appear here", r, juce::Justification::centred, true);
    else
        g.drawText(juce::String(sounding) + (sounding == 1 ? " voice" : " voices"), r.withHeight(16),
                   juce::Justification::topRight, false);
}

void ElementView::paintDelay(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto colour = familyColour(t, primitive_);
    const auto delay = juce::jmax(1e-5, value(0, 0)), depth = value(0, 1), wet = value(0, 3),
               feedback = value(0, 4);
    const auto span = delay * 6.5;
    const auto x = [&](double time)
    {
        return r.getX() + r.getWidth() * static_cast<float>(time / span);
    };
    g.setColour(t.line.withAlpha(0.6f));
    g.drawHorizontalLine(static_cast<int>(r.getBottom() - 1), r.getX(), r.getRight());
    g.setColour(t.text.withAlpha(0.7f));
    g.fillRoundedRectangle(x(0) - 1.5f, r.getY(), 3, r.getHeight(), 1.5f);
    // Echoes at multiples of the base delay, each scaled by the feedback; the first carries the wet mix.
    auto level = wet;
    for (int k = 1; k <= 6; ++k)
    {
        const auto h = r.getHeight() * static_cast<float>(juce::jlimit(0.0, 1.0, std::abs(level)));
        const auto xx = x(delay * k);
        if (k == 1)
        {
            g.setColour(colour.withAlpha(0.18f));
            g.fillRect(juce::Rectangle<float>(x(delay - depth), r.getY(), x(delay + depth) - x(delay - depth),
                                              r.getHeight()));
        }
        g.setColour(weightColour(t, level).interpolatedWith(colour, 0.5f));
        g.fillRoundedRectangle(xx - 2, r.getBottom() - h, 4, h, 2);
        level *= feedback;
    }
}

void ElementView::paintWidth(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto colour = familyColour(t, primitive_);
    const auto width = juce::jlimit(0.0, 2.0, value(0, 0));
    const auto centre = juce::Point<float>(r.getCentreX(), r.getBottom() - 4);
    const auto radius = juce::jmin(r.getHeight() - 8, r.getWidth() * 0.5f - 8);
    const auto half = static_cast<float>(std::numbers::pi / 4 * width * 0.95);
    juce::Path wedge;
    wedge.addPieSegment(juce::Rectangle<float>(2 * radius, 2 * radius).withCentre(centre), -half, half, 0);
    g.setGradientFill(juce::ColourGradient(colour.withAlpha(0.45f), centre.x, centre.y,
                                           colour.withAlpha(0.05f), centre.x, centre.y - radius, true));
    g.fillPath(wedge);
    g.setColour(t.line);
    for (const auto angle : {-std::numbers::pi / 4, std::numbers::pi / 4})
        g.drawLine(
            juce::Line<float>(centre, centre.getPointOnCircumference(radius, static_cast<float>(angle))),
            1.0f);
    g.setColour(t.text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
    g.drawText("L", juce::Rectangle<float>(centre.x - radius * 0.75f - 10, r.getY(), 20, 14),
               juce::Justification::centred);
    g.drawText("R", juce::Rectangle<float>(centre.x + radius * 0.75f - 10, r.getY(), 20, 14),
               juce::Justification::centred);
}

void ElementView::paintReverb(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto colour = familyColour(t, primitive_);
    const auto feedback = juce::jlimit(0.0, 0.999, std::abs(value(0, 0)));
    // Level after n passes through the network is feedback^n; the axis ends where it reaches -60 dB.
    const auto passes = feedback > 0 ? juce::jlimit(2.0, 200.0, std::log(0.001) / std::log(feedback)) : 2.0;
    juce::Path p;
    for (int i = 0; i <= 80; ++i)
    {
        const auto n = passes * i / 80.0;
        const auto level = std::pow(feedback, n);
        const auto xx = r.getX() + r.getWidth() * static_cast<float>(i) / 80.0f;
        const auto yy = r.getBottom() - r.getHeight() * static_cast<float>(level);
        i == 0 ? p.startNewSubPath(xx, yy) : p.lineTo(xx, yy);
    }
    auto fill = p;
    fill.lineTo(r.getRight(), r.getBottom());
    fill.lineTo(r.getX(), r.getBottom());
    fill.closeSubPath();
    g.setGradientFill(juce::ColourGradient(colour.withAlpha(0.3f), r.getX(), r.getY(),
                                           colour.withAlpha(0.02f), r.getX(), r.getBottom(), false));
    g.fillPath(fill);
    g.setColour(colour);
    g.strokePath(p, juce::PathStrokeType(2.0f));
    g.setColour(t.text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
    g.drawText("-60 dB after " + juce::String(static_cast<int>(std::round(passes))) + " passes", r,
               juce::Justification::topRight, false);
}

void ElementView::paintGeneric(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto colour = familyColour(t, primitive_);
    const auto count = instances_.empty() ? size_t(0) : instances_[0].count;
    if (count == 0)
    {
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f));
        g.drawText("No parameters", r, juce::Justification::centred);
        return;
    }
    const auto w = r.getWidth() / static_cast<float>(count);
    for (size_t p = 0; p < count; ++p)
    {
        const auto low = p < ranges_.size() ? ranges_[p][0] : 0.0,
                   high = p < ranges_.size() ? ranges_[p][1] : 1.0;
        const auto fraction = high > low ? juce::jlimit(0.0, 1.0, (value(0, p) - low) / (high - low)) : 0.5;
        auto bar = juce::Rectangle<float>(r.getX() + w * static_cast<float>(p), r.getY(), w, r.getHeight())
                       .reduced(juce::jmin(4.0f, w * 0.2f), 0);
        g.setColour(t.line.withAlpha(0.5f));
        g.fillRoundedRectangle(bar, 3);
        g.setColour(colour.withAlpha(0.85f));
        g.fillRoundedRectangle(bar.withTop(bar.getBottom() - bar.getHeight() * static_cast<float>(fraction)),
                               3);
    }
}

void ElementView::mouseMove(const juce::MouseEvent& e)
{
    const auto points = handles();
    juce::String tip;
    for (size_t i = 0; i < points.size(); ++i)
        if (points[i].getDistanceFrom(e.position) < 10)
            tip = kind_ == Kind::Envelope ? juce::StringArray{"Drag: attack", "Drag: decay and sustain",
                                                              "Drag: release"}[static_cast<int>(i)]
                                          : "Drag: cutoff left and right, resonance up and down";
    setTooltip(tip.isEmpty() ? title_ + ". Double-click to open its parameters"
                             : tip + (instances_.size() > 1 ? ", every voice at once" : ""));
    setMouseCursor(tip.isEmpty() ? juce::MouseCursor::NormalCursor : juce::MouseCursor::DraggingHandCursor);
}

void ElementView::mouseDown(const juce::MouseEvent& e)
{
    handle_ = -1;
    const auto points = handles();
    for (size_t i = 0; i < points.size(); ++i)
        if (points[i].getDistanceFrom(e.position) < 10)
            handle_ = static_cast<int>(i);
    pending_.clear();
}

void ElementView::mouseDrag(const juce::MouseEvent& e)
{
    if (handle_ < 0)
        return;
    auto r = plot().reduced(6, 6);
    const auto p = e.position;
    pending_.clear();
    if (kind_ == Kind::Envelope)
    {
        // Time per pixel is fixed from the press, so the scale does not move under the pointer.
        const auto a = value(0, 1, true), d = value(0, 2, true), rel = value(0, 4, true);
        const auto hold = juce::jmax(0.05, 0.25 * (a + d + rel));
        const auto total = a + d + hold + rel;
        const auto seconds = [&](float dx)
        {
            return static_cast<double>(dx) / r.getWidth() * total;
        };
        const auto start = e.mouseDownPosition;
        if (handle_ == 0)
            pending_ = {{1, clampParameter(1, a + seconds(p.x - start.x))}};
        else if (handle_ == 1)
            pending_ = {
                {2, clampParameter(2, d + seconds(p.x - start.x))},
                {3,
                 clampParameter(
                     3, juce::jlimit(0.0, 1.0, static_cast<double>((r.getBottom() - p.y) / r.getHeight())))}};
        else
            pending_ = {{4, clampParameter(4, rel + seconds(p.x - start.x))}};
    }
    else if (kind_ == Kind::Lowpass)
        pending_ = {
            {0, clampParameter(0, hzAt(r, p.x))},
            {1, clampParameter(1, std::pow(10.0, (18 - 48 * (p.y - r.getY()) / r.getHeight()) / 20))}};
    repaint();
}

void ElementView::mouseUp(const juce::MouseEvent&)
{
    if (handle_ >= 0 && !pending_.empty() && onEdit)
        onEdit(pending_);
    handle_ = -1;
    pending_.clear();
    repaint();
}

void ElementView::mouseDoubleClick(const juce::MouseEvent&)
{
    if (onOpen)
        onOpen();
}

// ----------------------------------------------------------------------------------------------------------

ControlProjection projectControls(const nlohmann::json& sound, size_t layer,
                                  const std::function<juce::String(size_t)>& parameterName)
{
    ControlProjection result;
    result.layer = layer;
    const auto layers = sound.value("control_layers", JSON::array());
    if (layer >= layers.size())
        return result;
    const auto n = layers[layer].at("values").size();
    if (n < 2)
        return result;
    for (size_t i = 0; i < n; ++i)
    {
        const auto range = layers[layer].at("ranges").at(i);
        const auto low = range.at(0).get<double>(), high = range.at(1).get<double>();
        result.mid.push_back((low + high) / 2);
        result.half.push_back(high > low ? (high - low) / 2 : 0.0);
    }
    // Compose the authored matrices from this layer down to the parameters.
    std::vector<std::vector<double>> j;
    {
        const auto& matrix = layers[layer].at("matrix");
        const auto rows = matrix.at("shape").at(0).get<size_t>();
        const auto& values = matrix.at("values");
        for (size_t d = 0; d < rows; ++d)
        {
            std::vector<double> row(n);
            for (size_t s = 0; s < n; ++s)
                row[s] = values.at(d * n + s).get<double>();
            j.push_back(std::move(row));
        }
    }
    for (size_t l = layer + 1; l < layers.size(); ++l)
    {
        const auto& matrix = layers[l].at("matrix");
        const auto rows = matrix.at("shape").at(0).get<size_t>(),
                   columns = matrix.at("shape").at(1).get<size_t>();
        const auto& values = matrix.at("values");
        std::vector<std::vector<double>> next(rows, std::vector<double>(n, 0.0));
        for (size_t d = 0; d < rows; ++d)
            for (size_t m = 0; m < columns && m < j.size(); ++m)
                if (const auto w = values.at(d * columns + m).get<double>(); w != 0.0)
                    for (size_t s = 0; s < n; ++s)
                        next[d][s] += w * j[m][s];
        j = std::move(next);
    }
    // Scale to fractions of each parameter's range per half range of each control.
    const auto metadata = sound.value("parameter_metadata", JSON::array());
    const auto locks = sound.value("parameter_locks", JSON::array());
    std::vector<double> rowScale(j.size(), metadata.empty() ? 1.0 : 0.0);
    for (const auto& meta : metadata)
        if (const auto index = meta.value("index", size_t(-1));
            index < rowScale.size() && meta.contains("min") && meta.contains("max"))
            if (const auto span = meta["max"].get<double>() - meta["min"].get<double>(); span > 0)
                rowScale[index] = 1.0 / span;
    for (size_t i = 0; i < locks.size() && i < rowScale.size(); ++i)
        if (locks[i].get<int>() != 0)
            rowScale[i] = 0;
    for (size_t d = 0; d < j.size(); ++d)
        for (size_t s = 0; s < n; ++s)
            j[d][s] *= rowScale[d] * result.half[s];
    // G = J~^T J~, eigenvectors by cyclic Jacobi rotations.
    std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0)), v(n, std::vector<double>(n, 0.0));
    for (const auto& row : j)
        for (size_t p = 0; p < n; ++p)
            if (row[p] != 0.0)
                for (size_t q = 0; q < n; ++q)
                    a[p][q] += row[p] * row[q];
    for (size_t i = 0; i < n; ++i)
        v[i][i] = 1;
    for (int sweep = 0; sweep < 60; ++sweep)
    {
        double off = 0;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q)
                off += a[p][q] * a[p][q];
        if (off < 1e-24)
            break;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q)
            {
                if (std::abs(a[p][q]) < 1e-300)
                    continue;
                const auto theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const auto tangent =
                    (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                const auto c = 1 / std::sqrt(tangent * tangent + 1), s = tangent * c;
                for (size_t k = 0; k < n; ++k)
                {
                    const auto kp = a[k][p], kq = a[k][q];
                    a[k][p] = c * kp - s * kq;
                    a[k][q] = s * kp + c * kq;
                }
                for (size_t k = 0; k < n; ++k)
                {
                    const auto pk = a[p][k], qk = a[q][k];
                    a[p][k] = c * pk - s * qk;
                    a[q][k] = s * pk + c * qk;
                }
                for (size_t k = 0; k < n; ++k)
                {
                    const auto kp = v[k][p], kq = v[k][q];
                    v[k][p] = c * kp - s * kq;
                    v[k][q] = s * kp + c * kq;
                }
            }
    }
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return a[x][x] > a[y][y]; });
    double total = 0;
    for (size_t i = 0; i < n; ++i)
        total += juce::jmax(0.0, a[i][i]);
    if (total <= 1e-18 || a[order[0]][order[0]] <= 1e-18 * total)
        return result;
    const auto vector = [&](size_t column)
    {
        std::vector<double> out(n);
        size_t largest = 0;
        for (size_t i = 0; i < n; ++i)
        {
            out[i] = v[i][column];
            if (std::abs(out[i]) > std::abs(out[largest]))
                largest = i;
        }
        if (out[largest] < 0) // stable orientation: the dominant control points right or up
            for (auto& x : out)
                x = -x;
        return out;
    };
    result.first = vector(order[0]);
    result.second = vector(order[1]);
    result.firstShare = juce::jmax(0.0, a[order[0]][order[0]]) / total;
    result.secondShare = juce::jmax(0.0, a[order[1]][order[1]]) / total;
    const auto moves = [&](const std::vector<double>& direction)
    {
        std::vector<std::pair<juce::String, double>> list;
        for (size_t d = 0; d < j.size(); ++d)
        {
            double sum = 0;
            for (size_t s = 0; s < n; ++s)
                sum += j[d][s] * direction[s];
            if (std::abs(sum) <= 1e-9)
                continue;
            // "Voice 3 decay" and "Voice 5 decay" are one thing to a player: merge repeated voices.
            auto name = parameterName(d);
            if (name.startsWith("Voice "))
                name = name.fromFirstOccurrenceOf(" ", false, false).fromFirstOccurrenceOf(" ", false, false);
            const auto existing =
                std::find_if(list.begin(), list.end(), [&](const auto& e) { return e.first == name; });
            if (existing == list.end())
                list.push_back({name, sum});
            else
                existing->second += sum;
        }
        std::sort(list.begin(), list.end(),
                  [](const auto& x, const auto& y) { return std::abs(x.second) > std::abs(y.second); });
        if (list.size() > 4)
            list.resize(4);
        return list;
    };
    result.firstMoves = moves(result.first);
    result.secondMoves = moves(result.second);
    result.valid = true;
    return result;
}

SoundSpace::SoundSpace(alfa::ui::AlfaLookAndFeel& look)
    : look_(look)
{
    setTooltip("Drag to move every control along the two directions that change this sound most");
}

void SoundSpace::setProjection(ControlProjection projection, juce::StringArray controlNames)
{
    projection_ = std::move(projection);
    names_ = std::move(controlNames);
    extentX_ = extentY_ = 1;
    if (projection_.valid)
    {
        extentX_ = extentY_ = 0;
        for (size_t i = 0; i < projection_.first.size(); ++i)
        {
            extentX_ += std::abs(projection_.first[i]);
            extentY_ += std::abs(projection_.second[i]);
        }
        extentX_ = juce::jmax(1e-6, extentX_);
        extentY_ = juce::jmax(1e-6, extentY_);
    }
    trail_.clear();
    repaint();
}

void SoundSpace::setValues(const std::vector<double>& values)
{
    if (values == values_)
        return;
    values_ = values;
    if (projection_.valid && values_.size() == projection_.first.size())
    {
        const auto p = positionOf(values_);
        if (trail_.empty() || trail_.back().getDistanceFrom(p) > 1.5f)
            trail_.push_back(p);
        if (trail_.size() > 64)
            trail_.erase(trail_.begin());
    }
    repaint();
}

void SoundSpace::setMarkers(std::vector<std::pair<juce::String, std::vector<double>>> markers)
{
    markers_ = std::move(markers);
    repaint();
}

juce::Rectangle<float> SoundSpace::pad() const
{
    // Each direction is scaled to its own reach, so the pad may be wider than tall, within reason.
    const auto area = getLocalBounds().toFloat().reduced(4).withTrimmedBottom(44);
    return area.withSizeKeepingCentre(juce::jmin(area.getWidth(), area.getHeight() * 1.6f), area.getHeight());
}

juce::Point<float> SoundSpace::toScreen(juce::Point<double> p) const
{
    const auto r = pad().reduced(14);
    return {r.getCentreX() + static_cast<float>(p.x / extentX_) * r.getWidth() * 0.5f,
            r.getCentreY() - static_cast<float>(p.y / extentY_) * r.getHeight() * 0.5f};
}

juce::Point<float> SoundSpace::positionOf(const std::vector<double>& values) const
{
    juce::Point<double> p;
    for (size_t i = 0; i < values.size() && i < projection_.first.size(); ++i)
    {
        const auto u = projection_.half[i] > 0 ? (values[i] - projection_.mid[i]) / projection_.half[i] : 0.0;
        p.x += projection_.first[i] * u;
        p.y += projection_.second[i] * u;
    }
    return toScreen(p);
}

void SoundSpace::paint(juce::Graphics& g)
{
    const auto& t = look_.theme();
    const auto r = pad();
    g.setGradientFill(juce::ColourGradient(t.panel.brighter(0.05f), r.getCentreX(), r.getCentreY(),
                                           t.bg.withAlpha(0.8f), r.getX(), r.getY(), true));
    g.fillRoundedRectangle(r, 8);
    if (!projection_.valid)
    {
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
        g.drawFittedText("This layer has fewer than two controls that change the sound",
                         r.reduced(20).toNearestInt(), juce::Justification::centred, 3);
        return;
    }
    const auto inner = r.reduced(14);
    const auto centre = inner.getCentre();
    g.setColour(t.line.withAlpha(0.7f));
    for (const auto scale : {0.5f, 1.0f})
        g.drawEllipse(
            juce::Rectangle<float>(inner.getWidth() * scale, inner.getHeight() * scale).withCentre(centre),
            1.0f);
    g.drawHorizontalLine(static_cast<int>(centre.y), inner.getX(), inner.getRight());
    g.drawVerticalLine(static_cast<int>(centre.x), inner.getY(), inner.getBottom());
    // Each control as an arrow: where a full positive turn of that control alone would move the sound.
    std::vector<size_t> strongest(projection_.first.size());
    for (size_t i = 0; i < strongest.size(); ++i)
        strongest[i] = i;
    const auto length = [&](size_t i)
    {
        return std::hypot(projection_.first[i] / extentX_, projection_.second[i] / extentY_);
    };
    std::sort(strongest.begin(), strongest.end(), [&](size_t a, size_t b) { return length(a) > length(b); });
    std::vector<juce::Rectangle<float>> placed;
    for (size_t k = 0; k < strongest.size() && k < 8; ++k)
    {
        const auto i = strongest[k];
        if (length(i) < 0.02)
            break;
        const auto tip = toScreen({projection_.first[i], projection_.second[i]});
        g.setColour(t.text2.withAlpha(0.4f));
        g.drawArrow(juce::Line<float>(centre, tip), 1.2f, 7, 7);
        // Name the arrow beyond its tip, unless another name is already there.
        // Short arrows stay unnamed: their names would crowd the centre where the sound sits.
        if (tip.getDistanceFrom(centre) < 26)
            continue;
        const auto direction = (tip - centre) / tip.getDistanceFrom(centre);
        const auto font = alfa::ui::AlfaLookAndFeel::labelFont(11);
        const auto name = i < static_cast<size_t>(names_.size()) ? names_[static_cast<int>(i)]
                                                                 : juce::String(static_cast<int>(i));
        const auto width = juce::GlyphArrangement::getStringWidth(font, name) + 6;
        const auto box = juce::Rectangle<float>(width, 14)
                             .withCentre(tip + direction * (10.0f + std::abs(direction.x) * width * 0.5f))
                             .constrainedWithin(getLocalBounds().toFloat());
        if (std::any_of(placed.begin(), placed.end(),
                        [&](const auto& other) { return other.intersects(box); }))
            continue;
        placed.push_back(box);
        g.setColour(t.text2);
        g.setFont(font);
        g.drawText(name, box, juce::Justification::centred, false);
    }
    for (const auto& [label, values] : markers_)
    {
        const auto p = positionOf(values);
        g.setColour(t.text);
        g.drawEllipse(juce::Rectangle<float>(16, 16).withCentre(p), 1.3f);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10, true));
        g.drawText(label, juce::Rectangle<float>(16, 16).withCentre(p), juce::Justification::centred);
    }
    for (size_t i = 1; i < trail_.size(); ++i)
    {
        g.setColour(t.accent.withAlpha(0.6f * static_cast<float>(i) / static_cast<float>(trail_.size())));
        g.drawLine(juce::Line<float>(trail_[i - 1], trail_[i]), 2.0f);
    }
    if (values_.size() == projection_.first.size())
    {
        const auto p = positionOf(values_);
        g.setGradientFill(juce::ColourGradient(t.accent.withAlpha(0.45f), p.x, p.y, t.accent.withAlpha(0.0f),
                                               p.x + 22, p.y, true));
        g.fillEllipse(juce::Rectangle<float>(44, 44).withCentre(p));
        g.setColour(t.accent);
        g.fillEllipse(juce::Rectangle<float>(12, 12).withCentre(p));
        g.setColour(t.bg);
        g.drawEllipse(juce::Rectangle<float>(12, 12).withCentre(p), 1.5f);
    }
    // What each direction does, in the sound's own parameter names.
    const auto describe = [](const std::vector<std::pair<juce::String, double>>& moves)
    {
        juce::StringArray parts;
        for (const auto& [name, amount] : moves)
            parts.add(name + (amount > 0 ? " +" : " -"));
        return parts.joinIntoString(", ");
    };
    auto captions = getLocalBounds().toFloat().reduced(6, 0).withTop(r.getBottom() + 4);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
    const auto line = [&](const juce::String& head, double share, const juce::String& text)
    {
        auto row = captions.removeFromTop(19);
        g.setColour(t.text);
        g.drawText(head + "  " + juce::String(juce::roundToInt(share * 100)) + "%", row.removeFromLeft(96),
                   juce::Justification::centredLeft, false);
        g.setColour(t.text2);
        g.drawFittedText(text, row.toNearestInt(), juce::Justification::centredLeft, 1, 0.85f);
    };
    line(juce::String::fromUTF8("\xe2\x86\x94") + " Direction 1", projection_.firstShare,
         describe(projection_.firstMoves));
    line(juce::String::fromUTF8("\xe2\x86\x95") + " Direction 2", projection_.secondShare,
         describe(projection_.secondMoves));
}

std::vector<double> SoundSpace::valuesAt(juce::Point<float> to) const
{
    const auto r = pad().reduced(14);
    const auto dx = static_cast<double>(to.x - dragOrigin_.x) / (r.getWidth() * 0.5) * extentX_;
    const auto dy = -static_cast<double>(to.y - dragOrigin_.y) / (r.getHeight() * 0.5) * extentY_;
    auto values = dragStart_;
    for (size_t i = 0; i < values.size() && i < projection_.first.size(); ++i)
    {
        if (projection_.half[i] <= 0)
            continue;
        const auto u = (values[i] - projection_.mid[i]) / projection_.half[i] + projection_.first[i] * dx +
                       projection_.second[i] * dy;
        values[i] = projection_.mid[i] + projection_.half[i] * juce::jlimit(-1.0, 1.0, u);
    }
    return values;
}

void SoundSpace::mouseDown(const juce::MouseEvent& e)
{
    if (!projection_.valid || values_.size() != projection_.first.size())
        return;
    dragging_ = true;
    dragStart_ = values_;
    // Pressing away from the dot jumps toward the press, as on a pad.
    dragOrigin_ = positionOf(values_);
    if (onMove)
        onMove(valuesAt(e.position), false);
}

void SoundSpace::mouseDrag(const juce::MouseEvent& e)
{
    if (dragging_ && onMove)
        onMove(valuesAt(e.position), false);
}

void SoundSpace::mouseUp(const juce::MouseEvent& e)
{
    if (dragging_ && onMove)
        onMove(valuesAt(e.position), true);
    dragging_ = false;
}
} // namespace alfa::tensorhost
