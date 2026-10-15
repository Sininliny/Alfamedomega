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
namespace visuals = alfa::ui::visuals;
// Card shared by every element: a raised panel whose title row carries the family dot and the first
// readout that fits; the curve sits in a recessed screen below it.
void card(juce::Graphics& g, const alfa::ui::Theme& t, juce::Rectangle<float> r, juce::Colour identity,
          const juce::String& title, const juce::String& count, const juce::StringArray& readouts)
{
    visuals::raised(g, t, r, 12);
    auto row = r.reduced(14, 0).withHeight(34);
    g.setColour(identity);
    g.fillEllipse(juce::Rectangle<float>(5, 5).withCentre({row.getX() + 2.5f, row.getCentreY()}));
    row.removeFromLeft(13);
    const auto titleFont = alfa::ui::AlfaLookAndFeel::labelFont(12.5f, true);
    g.setColour(t.text);
    g.setFont(titleFont);
    g.drawText(title, row, juce::Justification::centredLeft, true);
    auto free = row.withTrimmedLeft(juce::GlyphArrangement::getStringWidth(titleFont, title) + 6);
    if (count.isNotEmpty())
    {
        const auto font = alfa::ui::AlfaLookAndFeel::labelFont(10.5f);
        g.setFont(font);
        g.setColour(t.text2.withAlpha(0.8f));
        g.drawText(count, free, juce::Justification::centredLeft, false);
        free.removeFromLeft(juce::GlyphArrangement::getStringWidth(font, count) + 10);
    }
    // The first readout that fits is shown; a narrow card shows a shorter one, or none.
    const auto font = alfa::ui::AlfaLookAndFeel::labelFont(11.0f);
    g.setFont(font);
    g.setColour(t.text2);
    for (const auto& text : readouts)
        if (juce::GlyphArrangement::getStringWidth(font, text) <= free.getWidth())
        {
            g.drawText(text, free, juce::Justification::centredRight, false);
            break;
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
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(9.5f));
    for (const auto hz : {100.0, 1000.0, 10000.0})
    {
        const auto x = logX(r, hz);
        g.setColour(visuals::gridColour(t));
        g.drawVerticalLine(static_cast<int>(x), r.getY(), r.getBottom());
        g.setColour(t.text2.withAlpha(0.6f));
        g.drawText(hz >= 1000 ? juce::String(static_cast<int>(hz / 1000)) + "k"
                              : juce::String(static_cast<int>(hz)),
                   juce::Rectangle<float>(x - 20, r.getBottom() + 2, 40, 12), juce::Justification::centred);
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
    // The recessed screen; curves use curveArea(), which leaves a row for axis labels.
    return getLocalBounds().toFloat().reduced(10, 0).withTrimmedTop(34).withTrimmedBottom(10);
}

juce::Rectangle<float> ElementView::curveArea() const
{
    return plot().reduced(10, 0).withTrimmedTop(12).withTrimmedBottom(18);
}

void ElementView::setHeading(const std::vector<double>& parameters)
{
    next_.clear();
    if (!parameters.empty())
        for (const auto& instance : instances_)
        {
            if (instance.offset + instance.count > parameters.size())
            {
                next_.clear();
                break;
            }
            next_.emplace_back(parameters.begin() + static_cast<std::ptrdiff_t>(instance.offset),
                               parameters.begin() + static_cast<std::ptrdiff_t>(instance.offset + instance.count));
        }
    repaint();
}

double ElementView::nextValue(size_t instance, size_t parameter) const
{
    if (instance < next_.size() && parameter < next_[instance].size() && pending_.empty())
        return next_[instance][parameter];
    return value(instance, parameter);
}

void ElementView::paint(juce::Graphics& g)
{
    const auto& t = look_.theme();
    const auto identity = familyColour(t, primitive_);
    juce::StringArray readouts;
    const auto v = [&](size_t p)
    {
        return value(0, p);
    };
    switch (kind_)
    {
    case Kind::Envelope:
        readouts = {"A " + timeText(v(1)) + " · D " + timeText(v(2)) + " · S " + compactText(v(3)) + " · R " +
                        timeText(v(4)),
                    "S " + compactText(v(3)) + " · R " + timeText(v(4)), "R " + timeText(v(4))};
        break;
    case Kind::Lowpass:
        readouts = {hertzText(v(0)) + " · Q " + compactText(v(1)), hertzText(v(0))};
        break;
    case Kind::Delay:
        readouts = {timeText(v(0)) + " · fb " + compactText(v(4)), timeText(v(0))};
        break;
    case Kind::Width:
        readouts = {"width " + compactText(v(0)), compactText(v(0))};
        break;
    case Kind::Reverb:
        readouts = {"feedback " + compactText(v(0)), compactText(v(0))};
        break;
    case Kind::Partials:
    case Kind::Generic:
        break;
    }
    const auto count = instances_.size();
    card(g, t, getLocalBounds().toFloat(), identity, title_,
         count > 1 ? juce::String(juce::CharPointer_UTF8("\xc3\x97")) + juce::String(static_cast<int>(count))
                   : juce::String(),
         readouts);
    visuals::recessed(g, t, plot(), 8);
    const auto area = curveArea();
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
    const auto points = handles();
    for (size_t i = 0; i < points.size(); ++i)
        visuals::handle(g, t, points[i], static_cast<int>(i) == handle_ || static_cast<int>(i) == hover_);
}

std::vector<juce::Point<float>> ElementView::handles() const
{
    const auto r = curveArea();
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
    g.setColour(visuals::gridColour(t));
    g.drawHorizontalLine(static_cast<int>(y(0.5)), r.getX(), r.getRight());
    // Level at time `time` of an envelope with these stages: straight segments, as the primitive runs them.
    const auto level = [&](double time, double a, double d, double s, double rel)
    {
        if (time < a)
            return a > 0 ? time / a : 1.0;
        if (time < a + d)
            return 1 - (1 - s) * (d > 0 ? (time - a) / d : 1.0);
        if (time < a + d + hold)
            return s;
        return rel > 0 ? juce::jmax(0.0, s * (1 - (time - a - d - hold) / rel)) : 0.0;
    };
    const auto curve = [&](auto&& parameter)
    {
        std::vector<juce::Point<float>> points;
        // Sample densely and add every corner so straight segments stay straight.
        std::vector<double> times;
        for (int i = 0; i <= 120; ++i)
            times.push_back(total * i / 120.0);
        for (const auto corner : {parameter(1), parameter(1) + parameter(2), parameter(1) + parameter(2) + hold,
                                  parameter(1) + parameter(2) + hold + parameter(4)})
            if (corner > 0 && corner < total)
                times.push_back(corner);
        std::sort(times.begin(), times.end());
        for (const auto time : times)
            points.push_back({x(time), y(level(time, parameter(1), parameter(2), parameter(3), parameter(4)))});
        return points;
    };
    for (size_t k = instances_.size(); k-- > 1;)
    {
        const auto points = curve([&](size_t p) { return value(k, p); });
        juce::Path p;
        p.startNewSubPath(points.front());
        for (const auto& point : points)
            p.lineTo(point);
        g.setColour(visuals::nowColour(t).withAlpha(0.22f));
        g.strokePath(p, juce::PathStrokeType(1.0f));
    }
    // Resample the heading on the same time points as now so the band between them lines up.
    auto now = curve([&](size_t p) { return value(0, p); });
    std::vector<juce::Point<float>> next;
    if (!next_.empty())
        for (const auto& point : now)
        {
            const auto time = (point.x - r.getX()) / r.getWidth() * total;
            next.push_back({point.x, y(level(time, nextValue(0, 1), nextValue(0, 2), nextValue(0, 3), nextValue(0, 4)))});
        }
    visuals::deltaCurves(g, t, now, next, r.getBottom());
    g.setColour(t.text2.withAlpha(0.6f));
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(9.5f));
    const std::array<std::pair<const char*, double>, 4> stages{
        {{"A", a0 / 2}, {"D", a0 + d0 / 2}, {"S", a0 + d0 + hold / 2}, {"R", a0 + d0 + hold + r0 / 2}}};
    for (const auto& [name, time] : stages)
        g.drawText(name, juce::Rectangle<float>(x(time) - 10, r.getBottom() + 3, 20, 12),
                   juce::Justification::centred);
    // Each sounding voice's envelope level in the last block, as a short mark at the right edge.
    for (size_t k = 0; k < levels_.size(); ++k)
        if (levels_[k] > 0.001f)
        {
            g.setColour(visuals::nowColour(t).withAlpha(0.8f));
            g.fillRoundedRectangle(r.getRight() - 12, y(levels_[k]) - 1.0f, 12, 2, 1);
        }
}

void ElementView::paintLowpass(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    frequencyGrid(g, t, r);
    const auto y = [&](double db)
    {
        return juce::jlimit(r.getY(), r.getBottom(), r.getY() + r.getHeight() * static_cast<float>((18 - db) / 48));
    };
    g.setColour(visuals::gridColour(t));
    g.drawHorizontalLine(static_cast<int>(y(0)), r.getX(), r.getRight());
    const auto response = [&](double cutoff, double q)
    {
        std::vector<juce::Point<float>> points;
        for (int i = 0; i <= 160; ++i)
        {
            const auto xx = r.getX() + r.getWidth() * static_cast<float>(i) / 160.0f;
            points.push_back({xx, y(lowpassDecibels(hzAt(r, xx), cutoff, q, sampleRate_))});
        }
        return points;
    };
    for (size_t k = instances_.size(); k-- > 1;)
    {
        const auto points = response(value(k, 0), value(k, 1));
        juce::Path p;
        p.startNewSubPath(points.front());
        for (const auto& point : points)
            p.lineTo(point);
        g.setColour(visuals::nowColour(t).withAlpha(0.22f));
        g.strokePath(p, juce::PathStrokeType(1.0f));
    }
    const auto now = response(value(0, 0), value(0, 1));
    const auto next = next_.empty() ? std::vector<juce::Point<float>>{} : response(nextValue(0, 0), nextValue(0, 1));
    visuals::deltaCurves(g, t, now, next, r.getBottom());
    const auto cx = logX(r, value(0, 0));
    for (float yy = y(lowpassDecibels(value(0, 0), value(0, 0), value(0, 1), sampleRate_)) + 8; yy < r.getBottom(); yy += 5)
    {
        g.setColour(visuals::nowColour(t).withAlpha(0.4f));
        g.drawVerticalLine(static_cast<int>(cx), yy, juce::jmin(r.getBottom(), yy + 2));
    }
}

void ElementView::paintPartials(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    frequencyGrid(g, t, r);
    for (const auto f : {0.25f, 0.5f, 0.75f})
    {
        g.setColour(visuals::gridColour(t));
        g.drawHorizontalLine(static_cast<int>(r.getY() + r.getHeight() * f), r.getX(), r.getRight());
    }
    double scale = 1e-9;
    for (size_t p = 1; p < ranges_.size(); p += 2)
        scale = juce::jmax(scale, ranges_[p][1]);
    for (size_t k = 0; k < instances_.size(); ++k)
        for (size_t p = 1; p < instances_[k].count; p += 2)
            scale = juce::jmax(scale, value(k, p), nextValue(k, p));
    const auto height = [&](double amplitude)
    {
        return r.getHeight() * static_cast<float>(juce::jlimit(0.0, 1.0, std::sqrt(juce::jmax(0.0, amplitude) / scale)));
    };
    int sounding = 0;
    const auto now = visuals::nowColour(t);
    for (size_t k = 0; k < instances_.size(); ++k)
    {
        bool any = false;
        for (size_t p = 0; p + 1 < instances_[k].count; p += 2)
        {
            const auto hz = value(k, p), amplitude = value(k, p + 1);
            if (hz <= 0 || (amplitude <= 0 && nextValue(k, p + 1) <= 0))
                continue;
            any = true;
            const auto x = logX(r, hz);
            const auto h = height(amplitude);
            const auto h2 = k < next_.size() ? height(nextValue(k, p + 1)) : h;
            const auto low = juce::jmin(h, h2);
            g.setColour(now.withAlpha(k == 0 ? 0.5f : 0.18f));
            g.fillRoundedRectangle(x - 2.5f, r.getBottom() - low, 5, low, 2.5f);
            if (k == 0 && std::abs(h2 - h) > 1.0f)
            {
                g.setColour((h2 > h ? t.accent : t.mod).withAlpha(h2 > h ? 0.9f : 0.6f));
                g.fillRoundedRectangle(x - 2.5f, r.getBottom() - juce::jmax(h, h2), 5, std::abs(h2 - h) + 2.5f, 2.5f);
            }
            if (k == 0)
            {
                g.setColour(now);
                g.drawHorizontalLine(static_cast<int>(r.getBottom() - h), x - 4.5f, x + 4.5f);
            }
        }
        sounding += any ? 1 : 0;
    }
    g.setColour(t.text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10.5f));
    if (sounding == 0)
        g.drawText("Play a note: each voice's partials appear here", r, juce::Justification::centred, true);
    else
        g.drawText(juce::String(sounding) + (sounding == 1 ? " voice" : " voices"), r.withHeight(14),
                   juce::Justification::topRight, false);
}

void ElementView::paintDelay(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto now = visuals::nowColour(t);
    const auto delay = juce::jmax(1e-5, value(0, 0)), depth = value(0, 1);
    const auto span = juce::jmax(delay, nextValue(0, 0)) * 6.5;
    const auto x = [&](double time)
    {
        return r.getX() + r.getWidth() * static_cast<float>(time / span);
    };
    g.setColour(visuals::gridColour(t));
    g.drawHorizontalLine(static_cast<int>(r.getBottom() - 1), r.getX(), r.getRight());
    g.setColour(now.withAlpha(0.7f));
    g.fillRoundedRectangle(x(0) - 1.5f, r.getY(), 3, r.getHeight(), 1.5f);
    g.setColour(now.withAlpha(0.08f));
    g.fillRect(juce::Rectangle<float>(x(delay - depth), r.getY(), x(delay + depth) - x(delay - depth), r.getHeight()));
    // Echoes at multiples of the delay, each scaled by the feedback; the first carries the wet mix.
    auto level = value(0, 3), heading = nextValue(0, 3);
    const auto nextDelay = juce::jmax(1e-5, nextValue(0, 0));
    for (int k = 1; k <= 6; ++k)
    {
        const auto h = r.getHeight() * static_cast<float>(juce::jlimit(0.0, 1.0, std::abs(level)));
        const auto h2 = r.getHeight() * static_cast<float>(juce::jlimit(0.0, 1.0, std::abs(heading)));
        const auto xx = x(delay * k);
        g.setColour(now.withAlpha(0.55f));
        g.fillRoundedRectangle(xx - 2.5f, r.getBottom() - juce::jmin(h, h2), 5, juce::jmin(h, h2), 2.5f);
        if (!next_.empty() && std::abs(h2 - h) > 1.0f)
        {
            g.setColour((h2 > h ? t.accent : t.mod).withAlpha(0.8f));
            g.fillRoundedRectangle(xx - 2.5f, r.getBottom() - juce::jmax(h, h2), 5, std::abs(h2 - h) + 2.5f, 2.5f);
        }
        if (!next_.empty() && std::abs(x(nextDelay * k) - xx) > 2.0f)
        {
            g.setColour(t.text2);
            g.drawRoundedRectangle(x(nextDelay * k) - 2.5f, r.getBottom() - h2, 5, juce::jmax(2.0f, h2), 2.5f, 1.0f);
        }
        level *= value(0, 4);
        heading *= nextValue(0, 4);
    }
}

void ElementView::paintWidth(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto width = juce::jlimit(0.0, 2.0, value(0, 0)), next = juce::jlimit(0.0, 2.0, nextValue(0, 0));
    const auto centre = juce::Point<float>(r.getCentreX(), r.getBottom() - 2);
    const auto radius = juce::jmin(r.getHeight() - 6, r.getWidth() * 0.5f - 8);
    const auto angle = [](double w)
    {
        return static_cast<float>(std::numbers::pi / 4 * w * 0.95);
    };
    const auto box = juce::Rectangle<float>(2 * radius, 2 * radius).withCentre(centre);
    juce::Path wedge;
    wedge.addPieSegment(box, -angle(width), angle(width), 0);
    g.setGradientFill(juce::ColourGradient(visuals::nowColour(t).withAlpha(0.32f), centre.x, centre.y,
                                           visuals::nowColour(t).withAlpha(0.04f), centre.x, centre.y - radius, true));
    g.fillPath(wedge);
    if (!next_.empty() && std::abs(next - width) > 1e-3)
    {
        const auto inner = angle(juce::jmin(width, next)), outer = angle(juce::jmax(width, next));
        juce::Path band;
        band.addPieSegment(box, inner, outer, 0);
        band.addPieSegment(box, -outer, -inner, 0);
        g.setColour((next > width ? t.accent : t.mod).withAlpha(0.45f));
        g.fillPath(band);
    }
    g.setColour(visuals::gridColour(t).withMultipliedAlpha(2.0f));
    for (const auto a : {-std::numbers::pi / 4, std::numbers::pi / 4})
        g.drawLine(juce::Line<float>(centre, centre.getPointOnCircumference(radius, static_cast<float>(a))), 1.0f);
    g.setColour(t.text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10.5f));
    g.drawText("L", juce::Rectangle<float>(centre.x - radius * 0.75f - 10, r.getY(), 20, 14), juce::Justification::centred);
    g.drawText("R", juce::Rectangle<float>(centre.x + radius * 0.75f - 10, r.getY(), 20, 14), juce::Justification::centred);
}

void ElementView::paintReverb(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto feedback = juce::jlimit(0.0, 0.999, std::abs(value(0, 0)));
    const auto heading = juce::jlimit(0.0, 0.999, std::abs(nextValue(0, 0)));
    // Level after n passes through the network is feedback^n; the axis ends where the longer of the two
    // reaches -60 dB.
    const auto passesOf = [](double f)
    {
        return f > 0 ? juce::jlimit(2.0, 200.0, std::log(0.001) / std::log(f)) : 2.0;
    };
    const auto passes = passesOf(feedback);
    const auto span = juce::jmax(passes, next_.empty() ? passes : passesOf(heading));
    const auto curve = [&](double f)
    {
        std::vector<juce::Point<float>> points;
        for (int i = 0; i <= 100; ++i)
        {
            const auto n = span * i / 100.0;
            points.push_back({r.getX() + r.getWidth() * static_cast<float>(i) / 100.0f,
                              r.getBottom() - r.getHeight() * static_cast<float>(std::pow(f, n))});
        }
        return points;
    };
    visuals::deltaCurves(g, t, curve(feedback), next_.empty() ? std::vector<juce::Point<float>>{} : curve(heading),
                         r.getBottom());
    g.setColour(t.text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10.5f));
    g.drawText("-60 dB after " + juce::String(static_cast<int>(std::round(passes))) + " passes", r,
               juce::Justification::topRight, false);
}

void ElementView::paintGeneric(juce::Graphics& g, juce::Rectangle<float> r)
{
    const auto& t = look_.theme();
    const auto count = instances_.empty() ? size_t(0) : instances_[0].count;
    if (count == 0)
    {
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
        g.drawText("No parameters", r, juce::Justification::centred);
        return;
    }
    const auto w = r.getWidth() / static_cast<float>(count);
    for (size_t p = 0; p < count; ++p)
    {
        const auto low = p < ranges_.size() ? ranges_[p][0] : 0.0,
                   high = p < ranges_.size() ? ranges_[p][1] : 1.0;
        const auto fraction = [&](double v)
        {
            return high > low ? static_cast<float>(juce::jlimit(0.0, 1.0, (v - low) / (high - low))) : 0.5f;
        };
        const auto now = fraction(value(0, p)), next = fraction(nextValue(0, p));
        auto bar = juce::Rectangle<float>(r.getX() + w * static_cast<float>(p), r.getY(), w, r.getHeight())
                       .reduced(juce::jmin(4.0f, w * 0.25f), 0);
        bar = bar.withSizeKeepingCentre(juce::jmin(bar.getWidth(), 8.0f), bar.getHeight());
        g.setColour(visuals::gridColour(t).withMultipliedAlpha(1.5f));
        g.fillRoundedRectangle(bar, 3);
        const auto top = [&](float f)
        {
            return bar.getBottom() - bar.getHeight() * f;
        };
        g.setColour(visuals::nowColour(t).withAlpha(0.6f));
        g.fillRoundedRectangle(bar.withTop(top(juce::jmin(now, next))), 3);
        if (!next_.empty() && std::abs(next - now) > 0.004f)
        {
            g.setColour((next > now ? t.accent : t.mod).withAlpha(0.85f));
            g.fillRoundedRectangle(bar.withTop(top(juce::jmax(now, next))).withBottom(top(juce::jmin(now, next)) + 3), 3);
        }
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
    int over = -1;
    for (size_t i = 0; i < points.size(); ++i)
        if (points[i].getDistanceFrom(e.position) < 10)
            over = static_cast<int>(i);
    if (over != hover_)
    {
        hover_ = over;
        repaint();
    }
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
    auto r = curveArea();
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
    result.perUnit.assign(j.size(), std::vector<double>(n, 0.0));
    for (size_t d = 0; d < j.size(); ++d)
        for (size_t s = 0; s < n; ++s)
            result.perUnit[d][s] = j[d][s] * result.half[s];
    result.rowScale = rowScale;
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

std::vector<double> stepParameters(const ControlProjection& projection, const std::vector<double>& parameters,
                                   const std::vector<double>& du,
                                   const std::vector<std::array<double, 2>>& ranges)
{
    auto result = parameters;
    for (size_t d = 0; d < result.size() && d < projection.perUnit.size(); ++d)
    {
        double change = 0;
        for (size_t s = 0; s < du.size() && s < projection.perUnit[d].size(); ++s)
            change += projection.perUnit[d][s] * du[s];
        result[d] += change;
        if (d < ranges.size() && ranges[d][1] > ranges[d][0])
            result[d] = juce::jlimit(ranges[d][0], ranges[d][1], result[d]);
    }
    return result;
}

SoundSpace::SoundSpace(alfa::ui::AlfaLookAndFeel& look)
    : look_(look)
{
    setTooltip("Drag to move every control along the two directions that change this sound most. Shading: "
               "where a step changes the sound more; it fades where parameters reach the end of their range");
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
    fieldDirty_ = true;
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
    fieldDirty_ = true;
    repaint();
}

void SoundSpace::setParameters(const std::vector<double>& parameters, std::vector<std::array<double, 2>> ranges)
{
    if (parameters == parameters_ && ranges == ranges_)
        return;
    parameters_ = parameters;
    ranges_ = std::move(ranges);
    fieldDirty_ = true;
    repaint();
}

void SoundSpace::setHeading(std::vector<double> du)
{
    if (du == heading_)
        return;
    heading_ = std::move(du);
    repaint();
}

void SoundSpace::setMarkers(std::vector<std::pair<juce::String, std::vector<double>>> markers)
{
    markers_ = std::move(markers);
    repaint();
}

void SoundSpace::resized()
{
    fieldDirty_ = true;
}

juce::Rectangle<float> SoundSpace::pad() const
{
    return getLocalBounds().toFloat();
}

juce::Point<float> SoundSpace::toScreen(juce::Point<double> p) const
{
    const auto r = pad().reduced(26, 22);
    return {r.getCentreX() + static_cast<float>(p.x / extentX_) * r.getWidth() * 0.5f,
            r.getCentreY() - static_cast<float>(p.y / extentY_) * r.getHeight() * 0.5f};
}

juce::Point<double> SoundSpace::planeOf(const std::vector<double>& values) const
{
    juce::Point<double> p;
    for (size_t i = 0; i < values.size() && i < projection_.first.size(); ++i)
    {
        const auto u = projection_.half[i] > 0 ? (values[i] - projection_.mid[i]) / projection_.half[i] : 0.0;
        p.x += projection_.first[i] * u;
        p.y += projection_.second[i] * u;
    }
    return p;
}

juce::Point<float> SoundSpace::positionOf(const std::vector<double>& values) const
{
    return toScreen(planeOf(values));
}

void SoundSpace::computeField()
{
    // How much one small step changes the sound at each point of the plane, reached from the current
    // point: the control step is clamped to the control ranges and the parameters to theirs, so the shading
    // fades where controls or parameters run out of range. Measured in fractions of parameter ranges.
    fieldDirty_ = false;
    fieldColour_ = look_.theme().accent;
    field_ = {};
    const auto n = projection_.first.size();
    if (!projection_.valid || values_.size() != n || projection_.perUnit.empty() || parameters_.empty())
        return;
    const auto rows = projection_.perUnit.size();
    std::vector<double> u0(n);
    for (size_t i = 0; i < n; ++i)
        u0[i] = projection_.half[i] > 0 ? (values_[i] - projection_.mid[i]) / projection_.half[i] : 0.0;
    const auto q0 = planeOf(values_);
    std::vector<double> scaled(rows);
    const auto evaluate = [&](juce::Point<double> q, std::vector<double>& out)
    {
        for (size_t d = 0; d < rows; ++d)
            out[d] = d < parameters_.size() ? parameters_[d] : 0.0;
        std::vector<double> du(n);
        for (size_t i = 0; i < n; ++i)
            du[i] = juce::jlimit(-1.0, 1.0, u0[i] + projection_.first[i] * (q.x - q0.x) +
                                                   projection_.second[i] * (q.y - q0.y)) -
                    u0[i];
        for (size_t d = 0; d < rows; ++d)
        {
            double change = 0;
            for (size_t i = 0; i < n; ++i)
                change += projection_.perUnit[d][i] * du[i];
            auto v = out[d] + change;
            if (d < ranges_.size() && ranges_[d][1] > ranges_[d][0])
                v = juce::jlimit(ranges_[d][0], ranges_[d][1], v);
            out[d] = v * (d < projection_.rowScale.size() ? projection_.rowScale[d] : 0.0);
        }
    };
    constexpr int columns = 30, lines = 20;
    const auto r = pad();
    const auto inner = pad().reduced(26, 22);
    const auto toPlane = [&](float x, float y)
    {
        return juce::Point<double>((x - inner.getCentreX()) / (inner.getWidth() * 0.5) * extentX_,
                                   -(y - inner.getCentreY()) / (inner.getHeight() * 0.5) * extentY_);
    };
    const auto step = 0.03 * juce::jmax(extentX_, extentY_);
    std::vector<double> a(rows), b(rows), c(rows), e(rows);
    std::vector<float> strength(static_cast<size_t>(columns * lines));
    for (int j = 0; j < lines; ++j)
        for (int i = 0; i < columns; ++i)
        {
            const auto q = toPlane(r.getX() + r.getWidth() * (static_cast<float>(i) + 0.5f) / columns,
                                   r.getY() + r.getHeight() * (static_cast<float>(j) + 0.5f) / lines);
            evaluate({q.x + step, q.y}, a);
            evaluate({q.x - step, q.y}, b);
            evaluate({q.x, q.y + step}, c);
            evaluate({q.x, q.y - step}, e);
            double size = 0;
            for (size_t d = 0; d < rows; ++d)
            {
                const auto gx = a[d] - b[d], gy = c[d] - e[d];
                size += gx * gx + gy * gy;
            }
            strength[static_cast<size_t>(j * columns + i)] = static_cast<float>(std::sqrt(size));
        }
    // One box blur, so the shading reads as a field rather than as cells.
    {
        const auto raw = strength;
        for (int j = 0; j < lines; ++j)
            for (int i = 0; i < columns; ++i)
            {
                float sum = 0;
                int count = 0;
                for (int dj = -1; dj <= 1; ++dj)
                    for (int di = -1; di <= 1; ++di)
                        if (const auto x = i + di, y = j + dj; x >= 0 && x < columns && y >= 0 && y < lines)
                        {
                            sum += raw[static_cast<size_t>(y * columns + x)];
                            ++count;
                        }
                strength[static_cast<size_t>(j * columns + i)] = sum / static_cast<float>(count);
            }
    }
    const auto [low, high] = std::minmax_element(strength.begin(), strength.end());
    if (*high <= 1e-12f || (*high - *low) / *high < 0.06f)
        return; // the same everywhere: nothing to shade
    const auto& t = look_.theme();
    field_ = juce::Image(juce::Image::ARGB, columns, lines, true);
    for (int j = 0; j < lines; ++j)
        for (int i = 0; i < columns; ++i)
        {
            const auto k = static_cast<size_t>(j * columns + i);
            const auto amount = (strength[k] - *low) / (*high - *low);
            field_.setPixelAt(i, j, t.accent.withAlpha(0.03f + 0.22f * amount * amount));
        }
}

void SoundSpace::paint(juce::Graphics& g)
{
    const auto& t = look_.theme();
    const auto r = pad();
    alfa::ui::visuals::recessed(g, t, r, 8);
    if (!projection_.valid)
    {
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
        g.drawFittedText("This layer has fewer than two controls that change the sound",
                         r.reduced(20).toNearestInt(), juce::Justification::centred, 3);
        return;
    }
    if (fieldDirty_ || fieldColour_ != t.accent)
        computeField();
    {
        const juce::Graphics::ScopedSaveState saved(g);
        juce::Path clip;
        clip.addRoundedRectangle(r, 8);
        g.reduceClipRegion(clip);
        if (field_.isValid())
        {
            g.setImageResamplingQuality(juce::Graphics::highResamplingQuality);
            g.drawImage(field_, r);
        }
    }
    const auto inner = r.reduced(26, 22);
    const auto centre = inner.getCentre();
    const auto grid = alfa::ui::visuals::gridColour(t);
    g.setColour(grid.withMultipliedAlpha(1.6f));
    for (const auto scale : {0.5f, 1.0f})
        g.drawEllipse(juce::Rectangle<float>(inner.getWidth() * scale, inner.getHeight() * scale).withCentre(centre),
                      1.0f);
    g.setColour(grid);
    g.drawHorizontalLine(static_cast<int>(centre.y), r.getX() + 8, r.getRight() - 8);
    g.drawVerticalLine(static_cast<int>(centre.x), r.getY() + 8, r.getBottom() - 8);
    // Each control as an arrow: where a full positive turn of that control alone would move the sound.
    std::vector<size_t> strongest(projection_.first.size());
    for (size_t i = 0; i < strongest.size(); ++i)
        strongest[i] = i;
    const auto length = [&](size_t i)
    {
        return std::hypot(projection_.first[i] / extentX_, projection_.second[i] / extentY_);
    };
    std::sort(strongest.begin(), strongest.end(), [&](size_t a, size_t b) { return length(a) > length(b); });
    // The current point and the heading arrow keep their space free of names.
    std::vector<juce::Rectangle<float>> placed;
    juce::Point<float> headingTip, headingStart;
    bool hasHeading = false;
    if (values_.size() == projection_.first.size())
    {
        const auto p = positionOf(values_);
        placed.push_back(juce::Rectangle<float>(28, 28).withCentre(p));
        juce::Point<double> h;
        for (size_t i = 0; i < heading_.size() && i < projection_.first.size(); ++i)
        {
            h.x += projection_.first[i] * heading_[i];
            h.y += projection_.second[i] * heading_[i];
        }
        const auto direction = toScreen(planeOf(values_) + h) - p;
        if (const auto distance = direction.getDistanceFromOrigin(); distance > 0.5f)
        {
            hasHeading = true;
            headingStart = p + direction / distance * 9.0f;
            headingTip = p + direction / distance * 38.0f;
            for (float f = 0.3f; f <= 1.0f; f += 0.35f)
                placed.push_back(juce::Rectangle<float>(14, 14).withCentre(p + (headingTip - p) * f));
        }
    }
    for (size_t k = 0; k < strongest.size() && k < 8; ++k)
    {
        const auto i = strongest[k];
        if (length(i) < 0.02)
            break;
        const auto tip = toScreen({projection_.first[i], projection_.second[i]});
        g.setColour(t.text2.withAlpha(0.65f));
        g.drawLine(juce::Line<float>(centre, tip), 1.0f);
        g.fillEllipse(juce::Rectangle<float>(4.5f, 4.5f).withCentre(tip));
        // Name the arrow beyond its tip, unless another name is already there.
        // Short arrows stay unnamed: their names would crowd the centre where the sound sits.
        if (tip.getDistanceFrom(centre) < 26)
            continue;
        const auto direction = (tip - centre) / tip.getDistanceFrom(centre);
        const auto font = alfa::ui::AlfaLookAndFeel::labelFont(10.5f);
        const auto name = i < static_cast<size_t>(names_.size()) ? names_[static_cast<int>(i)]
                                                                 : juce::String(static_cast<int>(i));
        const auto width = juce::GlyphArrangement::getStringWidth(font, name) + 6;
        const auto box = juce::Rectangle<float>(width, 14)
                             .withCentre(tip + direction * (9.0f + std::abs(direction.x) * width * 0.5f))
                             .constrainedWithin(r.reduced(2));
        if (std::any_of(placed.begin(), placed.end(), [&](const auto& other) { return other.intersects(box); }))
            continue;
        placed.push_back(box);
        g.setColour(t.text2);
        g.setFont(font);
        g.drawText(name, box, juce::Justification::centred, false);
    }
    for (const auto& [label, values] : markers_)
    {
        const auto p = positionOf(values);
        const auto box = juce::Rectangle<float>(18, 18).withCentre(p);
        g.setColour(alfa::ui::visuals::screenColour(t));
        g.fillEllipse(box);
        g.setColour(t.text2.withAlpha(0.7f));
        g.drawEllipse(box, 1.0f);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10, true));
        g.drawText(label, box, juce::Justification::centred);
    }
    const auto now = alfa::ui::visuals::nowColour(t);
    for (size_t i = 1; i < trail_.size(); ++i)
    {
        g.setColour(now.withAlpha(0.7f * static_cast<float>(i) / static_cast<float>(trail_.size())));
        g.drawLine(juce::Line<float>(trail_[i - 1], trail_[i]), 2.0f);
    }
    if (values_.size() == projection_.first.size())
    {
        const auto p = positionOf(values_);
        // Where the sound is heading: the current step, shown as a direction in the plane.
        if (hasHeading)
        {
            const auto unit = (headingTip - p) / headingTip.getDistanceFrom(p);
            const auto across = juce::Point<float>(-unit.y, unit.x);
            juce::Path dashed;
            const float pattern[] = {3.0f, 3.0f};
            juce::Path line;
            line.startNewSubPath(headingStart);
            line.lineTo(headingTip - unit * 6.0f);
            juce::PathStrokeType(1.5f).createDashedStroke(dashed, line, pattern, 2);
            g.setColour(t.accent);
            g.fillPath(dashed);
            juce::Path head;
            head.addTriangle(headingTip, headingTip - unit * 8.0f + across * 4.0f, headingTip - unit * 8.0f - across * 4.0f);
            g.fillPath(head);
        }
        g.setColour(now.withAlpha(0.1f));
        g.fillEllipse(juce::Rectangle<float>(26, 26).withCentre(p));
        g.setColour(now);
        g.fillEllipse(juce::Rectangle<float>(11, 11).withCentre(p));
        g.setColour(alfa::ui::visuals::screenColour(t));
        g.drawEllipse(juce::Rectangle<float>(11, 11).withCentre(p), 2.0f);
    }
}

std::vector<double> SoundSpace::valuesAt(juce::Point<float> to) const
{
    const auto r = pad().reduced(26, 22);
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
HeadingView::HeadingView(alfa::ui::AlfaLookAndFeel& look)
    : look_(look)
{
}

void HeadingView::setRows(std::vector<Row> rows, juce::String source)
{
    const auto same = [](const Row& a, const Row& b)
    {
        return a.group == b.group && a.parameter == b.parameter && std::abs(a.change - b.change) < 1e-4;
    };
    if (source == source_ && rows.size() == rows_.size() && std::equal(rows.begin(), rows.end(), rows_.begin(), same))
        return;
    rows_ = std::move(rows);
    source_ = std::move(source);
    repaint();
}

void HeadingView::paint(juce::Graphics& g)
{
    const auto& t = look_.theme();
    alfa::ui::visuals::panel(g, t, getLocalBounds(), "Heading", source_, t.accent);
    rowAreas_.clear();
    auto area = getLocalBounds().toFloat().reduced(14, 0).withTrimmedTop(34).withTrimmedBottom(8);
    if (rows_.empty())
    {
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11.5f));
        g.drawFittedText("Move a control or the sound space to see what changes", area.toNearestInt(),
                         juce::Justification::centred, 2);
        return;
    }
    const auto columns = rows_.size() > 6 && area.getWidth() > 520 ? 2 : 1;
    const auto perColumn = static_cast<int>((rows_.size() + static_cast<size_t>(columns) - 1) / static_cast<size_t>(columns));
    const auto rowHeight = juce::jmin(22.0f, area.getHeight() / static_cast<float>(juce::jmax(1, perColumn)));
    double largest = 0.05;
    for (const auto& row : rows_)
        largest = juce::jmax(largest, std::abs(row.change));
    const auto columnWidth = (area.getWidth() - 20.0f * static_cast<float>(columns - 1)) / static_cast<float>(columns);
    for (size_t i = 0; i < rows_.size(); ++i)
    {
        const auto column = static_cast<int>(i) / perColumn, line = static_cast<int>(i) % perColumn;
        auto r = juce::Rectangle<float>(area.getX() + static_cast<float>(column) * (columnWidth + 20.0f),
                                        area.getY() + static_cast<float>(line) * rowHeight, columnWidth, rowHeight);
        if (line >= perColumn || r.getBottom() > area.getBottom() + 0.5f)
            continue;
        rowAreas_.push_back(r);
        const auto& row = rows_[i];
        auto label = r.removeFromLeft(r.getWidth() * 0.5f);
        g.setColour(row.colour);
        g.fillEllipse(juce::Rectangle<float>(5, 5).withCentre({label.getX() + 2.5f, label.getCentreY()}));
        label.removeFromLeft(11);
        // The group, then the parameter of the group that moves most.
        const auto groupFont = alfa::ui::AlfaLookAndFeel::labelFont(11.5f, true);
        const auto groupWidth = juce::jmin(label.getWidth(), juce::GlyphArrangement::getStringWidth(groupFont, row.group) + 8);
        g.setColour(t.text);
        g.setFont(groupFont);
        g.drawText(row.group, label.removeFromLeft(groupWidth), juce::Justification::centredLeft, true);
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(10.5f));
        g.drawText(std::abs(row.change) < 0.005 ? juce::String("unchanged") : namePattern(row.parameter).withoutVoice,
                   label, juce::Justification::centredLeft, true);
        auto value = r.removeFromRight(42);
        const auto quiet = std::abs(row.change) < 0.005;
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11, true));
        g.setColour(quiet ? t.text2 : row.change > 0 ? t.accent : t.mod);
        g.drawText((row.change > 0 && !quiet ? "+" : "") + juce::String(quiet ? 0.0 : row.change, 2), value,
                   juce::Justification::centredRight);
        const auto track = r.reduced(6, 0).withSizeKeepingCentre(r.getWidth() - 12, 4);
        g.setColour(alfa::ui::visuals::screenColour(t));
        g.fillRoundedRectangle(track, 2);
        g.setColour(alfa::ui::visuals::gridColour(t).withMultipliedAlpha(3.0f));
        g.fillRect(juce::Rectangle<float>(1, 8).withCentre(track.getCentre()));
        if (!quiet)
        {
            const auto length = static_cast<float>(row.change / largest) * track.getWidth() * 0.5f;
            const auto bar = length > 0 ? juce::Rectangle<float>(track.getCentreX(), track.getY(), length, 4)
                                        : juce::Rectangle<float>(track.getCentreX() + length, track.getY(), -length, 4);
            g.setColour((row.change > 0 ? t.accent : t.mod).withAlpha(0.9f));
            g.fillRoundedRectangle(bar, 2);
        }
    }
}

int HeadingView::preferredHeight(size_t rows, int width)
{
    const auto shown = juce::jlimit<size_t>(1, 12, rows);
    const auto columns = shown > 6 && width - 28 > 520 ? 2 : 1;
    const auto perColumn = static_cast<int>((shown + static_cast<size_t>(columns) - 1) / static_cast<size_t>(columns));
    return 34 + perColumn * 22 + 10;
}

void HeadingView::mouseMove(const juce::MouseEvent& event)
{
    juce::String tip;
    for (size_t i = 0; i < rowAreas_.size() && i < rows_.size(); ++i)
        if (rowAreas_[i].contains(event.position))
            tip = rows_[i].group + ": " + rows_[i].parameter + " " + (rows_[i].change >= 0 ? "+" : "") +
                  juce::String(rows_[i].change * 100, 1) + " % of its range";
    setTooltip(tip);
}

} // namespace alfa::tensorhost
