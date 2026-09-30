#include "alfa/plugin/ParameterLayout.h"

#include <cmath>

namespace alfa::params
{
namespace id
{
juce::String knob(int index)
{
    return "knob_" + juce::String(index + 1).paddedLeft('0', 2);
}
} // namespace id

bool isValidHybridPair(Method a, Method b)
{
    auto isComponent = [](Method m)
    {
        return std::find(kHybridComponents.begin(), kHybridComponents.end(), m) != kHybridComponents.end();
    };
    if (a == b || !isComponent(a) || !isComponent(b))
        return false;
    return !((a == Method::additive && b == Method::physical) ||
             (a == Method::physical && b == Method::additive));
}

namespace
{
using F = KnobInfo::Format;

using KnobTable = std::array<KnobInfo, kNumKnobs>;

KnobInfo k(const char* name, const char* shortName, F format = F::percent, Tier minimumTier = Tier::lite,
           const char* note = "")
{
    KnobInfo info;
    info.name = name;
    info.shortName = shortName;
    info.format = format;
    info.used = true;
    info.minimumTier = minimumTier;
    info.note = note;
    return info;
}

KnobInfo onOff(const char* name, const char* shortName, float threshold, const char* note = "")
{
    auto info = k(name, shortName, F::onOff, Tier::lite, note);
    info.threshold = threshold;
    return info;
}

KnobTable emptyTable()
{
    KnobTable t;
    for (int i = 0; i < kNumKnobs; ++i)
    {
        t[static_cast<size_t>(i)].name = "Knob " + juce::String(i + 1) + " (unused)";
        t[static_cast<size_t>(i)].shortName = "K" + juce::String(i + 1);
    }
    return t;
}

void ampEnvelope(KnobTable& t)
{
    t[18] = k("Amp attack", "Attack", F::attackTime);
    t[19] = k("Amp decay", "Decay", F::decayTime);
    t[20] = k("Amp sustain", "Sustain");
    t[21] = k("Amp release", "Release", F::decayTime);
}

void envelope(KnobTable& t, int first, const char* what, const char* shortWhat)
{
    const juce::String w(what), s(shortWhat);
    t[static_cast<size_t>(first)] = k("", "", F::attackTime);
    t[static_cast<size_t>(first + 1)] = k("", "", F::decayTime);
    t[static_cast<size_t>(first + 2)] = k("", "");
    t[static_cast<size_t>(first + 3)] = k("", "", F::decayTime);
    const char* stages[] = {"attack", "decay", "sustain", "release"};
    const char* shortStages[] = {"A", "D", "S", "R"};
    for (int i = 0; i < 4; ++i)
    {
        t[static_cast<size_t>(first + i)].name = w + " " + stages[i];
        t[static_cast<size_t>(first + i)].shortName = s + " " + shortStages[i];
    }
}

void unison(KnobTable& t)
{
    t[22] = k("Unison", "Unison", F::unison, Tier::lite,
              "Below 50 % only the main oscillator plays; above it more detuned copies join, up to the model "
              "size's limit");
    t[23] = k("Unison spread", "Spread", F::spreadCents);
}

KnobTable build(Method m)
{
    auto t = emptyTable();
    switch (m)
    {
    case Method::subtractive:
        t[0] = k("Saw level", "Saw");
        t[1] = k("Pulse level", "Pulse");
        t[2] = k("Triangle level", "Tri");
        t[3] = k("Pulse width", "PW");
        t[6] = k("Filter cutoff", "Cutoff", F::cutoff);
        t[7] = k("Filter envelope amount", "F.Env", F::bipolar);
        t[8] = k("Filter key tracking", "Key trk");
        t[9] = k("Velocity to cutoff", "Vel>Cut");
        envelope(t, 10, "Filter", "Flt");
        t[14] = k("Mod wheel to cutoff", "MW>Cut");
        t[15] = k("Resonance", "Reso", F::resonance);
        t[16] = k("Filter mode", "Mode", F::filterMode);
        t[17] = k("Noise", "Noise");
        ampEnvelope(t);
        unison(t);
        break;
    case Method::wavetable:
        t[0] = k("Wavetable", "Table", F::table64);
        t[1] = k("Position", "Pos");
        t[2] = k("Position envelope amount", "P.Env", F::bipolar);
        t[3] = k("Mod wheel to position", "MW>Pos");
        envelope(t, 4, "Position", "Pos");
        t[8] = k("Brightness tilt", "Tilt", F::tilt);
        ampEnvelope(t);
        unison(t);
        break;
    case Method::additive:
        t[0] = k("Harmonic rolloff", "Rolloff");
        t[1] = k("Even harmonics cut", "Even cut");
        t[2] = k("Formant frequency", "Formant", F::formant);
        t[3] = k("Formant width", "F.Width");
        t[4] = k("Formant boost", "F.Boost");
        t[5] = k("High partial decay", "Hi decay");
        t[6] = onOff("Stretch tuning", "Stretch", 0.6f, "Piano-like stretched partials above 60 %");
        t[7] = k("Stretch amount", "Str amt");
        t[8] = k("Partial detune", "Detune");
        ampEnvelope(t);
        break;
    case Method::spectral:
        t[0] = k("Peak 1 start", "P1 start");
        t[1] = k("Peak 2 start", "P2 start");
        t[2] = k("Peak 3 start", "P3 start");
        t[3] = k("Peak 1 end", "P1 end");
        t[4] = k("Peak 2 end", "P2 end");
        t[5] = k("Peak 3 end", "P3 end");
        t[6] = k("Harmonic to noise", "Harm/Noise");
        t[7] = k("Morph LFO rate", "LFO rate", F::lfoRate);
        t[8] = k("Morph LFO depth", "LFO depth", F::percent, Tier::lite,
                 "The LFO and the mod wheel move each peak from its start to its end");
        ampEnvelope(t);
        break;
    case Method::fm:
        t[0] = k("Algorithm", "Algo", F::fmAlgorithm);
        for (int op = 2; op <= 8; ++op)
        {
            const auto minTier = op <= 2   ? Tier::lite
                                 : op <= 4 ? Tier::standard
                                 : op <= 6 ? Tier::pro
                                           : Tier::studio;
            auto& info = t[static_cast<size_t>(op)];
            info = k("", "", F::fmRatio, minTier,
                     "Its twin knob (8 further on) halves the ratio when it is below 15 %");
            info.name = "Operator " + juce::String(op) + " ratio";
            info.shortName = "Op" + juce::String(op) + " ratio";
        }
        envelope(t, 10, "Index", "Idx");
        for (int i = 10; i <= 16; ++i)
            t[static_cast<size_t>(i)].note =
                "Below 15 % it also halves operator " + juce::String(i - 8) + "'s ratio";
        t[14] = k("Modulation index", "Index", F::percent, Tier::lite,
                  "Below 15 % it also halves operator 6's ratio");
        t[15] = k("Mod wheel to index", "MW>Idx", F::percent, Tier::lite,
                  "Below 15 % it also halves operator 7's ratio");
        t[16] = k("Operator 8 half ratio", "Op8 half", F::percent, Tier::studio,
                  "Below 15 % operator 8 plays at half its ratio");
        t[17] = k("Operator detune", "Detune", F::bipolar);
        ampEnvelope(t);
        break;
    case Method::physical:
        t[0] = k("Resonator", "Resonator", F::resonator);
        t[1] =
            k("Stiffness", "Stiff", F::percent, Tier::lite, "For the plate it picks one of 8 plate shapes");
        t[2] = k("Decay time", "Decay", F::decaySeconds);
        t[3] = k("High mode damping", "Hi damp");
        t[4] = k("Release damping", "Rel damp");
        t[5] = k("Brightness", "Bright");
        t[6] = k("Strike position", "Position");
        t[7] = onOff("Bowed", "Bowed", 0.5f,
                     "Above 50 % the resonator is driven continuously, like a bow or breath");
        t[8] = k("Breath noise", "Breath");
        break;
    case Method::granular:
        t[0] = k("Grain source", "Source", F::source16);
        t[1] = k("Position", "Pos");
        t[2] = k("Scan speed", "Scan");
        t[3] = k("Grain pitch", "Pitch", F::octaves);
        t[4] = k("Layer detune", "L.Detune");
        t[5] = onOff("Pitched grains", "Pitched", 0.5f, "Above 50 % grains follow the played note");
        t[6] = k("Position jitter", "Jitter");
        t[7] = k("Layer spread", "L.Spread");
        ampEnvelope(t);
        break;
    case Method::sampling:
        t[0] = k("Sample", "Sample", F::sample8);
        t[1] = k("Fine tune", "Fine", F::fineTune);
        t[2] = k("Start offset", "Start");
        t[3] = k("EQ tilt", "Tilt", F::bipolar);
        t[4] = k("EQ bump", "Bump", F::bipolar);
        t[5] = k("EQ bump frequency", "Bump Hz");
        ampEnvelope(t);
        break;
    case Method::hybrid:
        break;
    }
    return t;
}

const KnobTable& table(Method m)
{
    static const std::array<KnobTable, kNumMethods> tables = []
    {
        std::array<KnobTable, kNumMethods> all;
        for (int i = 0; i < kNumMethods; ++i)
            all[static_cast<size_t>(i)] = build(static_cast<Method>(i));
        return all;
    }();
    return tables[static_cast<size_t>(m)];
}

// juce::String(float, 0) prints every digit; zero decimals means rounding here.
juce::String number(float v, int decimals)
{
    return decimals == 0 ? juce::String(juce::roundToInt(v)) : juce::String(v, decimals);
}

juce::String percent(float v)
{
    return juce::String(juce::roundToInt(v * 100.0f)) + " %";
}

juce::String seconds(float s)
{
    return s >= 1.0f ? juce::String(s, 2) + " s" : number(s * 1000.0f, s < 0.01f ? 1 : 0) + " ms";
}

juce::String hertz(float hz)
{
    return hz >= 1000.0f ? juce::String(hz / 1000.0f, 2) + " kHz" : number(hz, hz < 100.0f ? 1 : 0) + " Hz";
}
} // namespace

KnobInfo knobInfo(Method method, int knobIndex)
{
    return table(method)[static_cast<size_t>(juce::jlimit(0, kNumKnobs - 1, knobIndex))];
}

KnobInfo hybridKnobInfo(Method a, Method b, int knobIndex)
{
    const auto i = juce::jlimit(0, kNumKnobs - 1, knobIndex);
    if (i == kNumKnobs - 1)
    {
        auto info = k("A to B crossfade", "A/B", F::percent, Tier::lite, "The mod wheel pushes it towards B");
        return info;
    }

    const auto ia = table(a)[static_cast<size_t>(i)];
    const auto ib = table(b)[static_cast<size_t>((i + 11) % kNumKnobs)];
    const juce::String nameA(kMethods[static_cast<size_t>(a)].name),
        nameB(kMethods[static_cast<size_t>(b)].name);

    if (!ia.used && !ib.used)
        return ia;
    if (ia.used && !ib.used)
    {
        auto info = ia;
        info.name = "A " + ia.name;
        info.note = (nameA + ": " + ia.note).trimCharactersAtEnd(": ");
        return info;
    }
    if (!ia.used)
    {
        auto info = ib;
        info.name = "B " + ib.name;
        info.note = (nameB + ": " + ib.note).trimCharactersAtEnd(": ");
        return info;
    }

    // Both components read this knob: show A's meaning, name both.
    auto info = ia;
    info.name = "A " + ia.name + " / B " + ib.name;
    info.shortName = ia.shortName + "/" + ib.shortName;
    info.minimumTier = std::max(ia.minimumTier, ib.minimumTier);
    info.note = "Also sets " + nameB + "'s " + ib.name.toLowerCase();
    return info;
}

KnobInfo knobInfoFor(Method method, Method hybridA, Method hybridB, int knobIndex)
{
    return method == Method::hybrid ? hybridKnobInfo(hybridA, hybridB, knobIndex)
                                    : knobInfo(method, knobIndex);
}

juce::String formatKnob(const KnobInfo& info, float v)
{
    v = juce::jlimit(0.0f, 1.0f, v);
    switch (info.format)
    {
    case F::percent:
        return percent(v);
    case F::bipolar:
    {
        const auto p = juce::roundToInt((v * 2.0f - 1.0f) * 100.0f);
        return (p > 0 ? "+" : "") + juce::String(p) + " %";
    }
    case F::attackTime:
        return seconds(0.001f * std::exp2(v * 11.0f));
    case F::decayTime:
        return seconds(0.01f * std::exp2(v * 9.0f));
    case F::decaySeconds:
        return seconds(0.05f * std::exp2(v * 7.0f));
    case F::cutoff:
        return hertz(juce::jlimit(20.0f, 20000.0f, 30.0f * std::exp2(v * 9.5f)));
    case F::formant:
        return hertz(200.0f * std::exp2(v * 6.0f));
    case F::resonance:
        return "Q " + juce::String(0.5f + 19.5f * v * v, 1);
    case F::unison:
        return v < 0.5f ? juce::String("off") : percent((v - 0.5f) * 2.0f) + " of copies";
    case F::spreadCents:
        return juce::String(3.0f + 47.0f * v * v, 1) + " ct";
    case F::fmRatio:
        return juce::String::charToString(0x00d7) + juce::String(juce::roundToInt(1.0f + v * 7.0f));
    case F::fmAlgorithm:
    {
        const char* names[] = {"Stack", "Pairs", "One to many", "Additive"};
        return names[juce::jlimit(0, 3, static_cast<int>(v * 3.999f))];
    }
    case F::filterMode:
        return v < 0.25f ? "Low-pass" : v < 0.5f ? "Low/band" : v < 0.75f ? "Band/high" : "High-pass";
    case F::table64:
        return "Table " + juce::String(static_cast<int>(v * 63.999f) + 1);
    case F::source16:
        return "Source " + juce::String(static_cast<int>(v * 15.999f) + 1);
    case F::sample8:
        return "Sample " + juce::String(static_cast<int>(v * 7.999f) + 1);
    case F::resonator:
    {
        const char* names[] = {"String", "Beam", "Membrane", "Tube", "Plate"};
        return names[juce::jlimit(0, 4, static_cast<int>(v * 4.999f))];
    }
    case F::onOff:
        return v > info.threshold ? "On" : "Off";
    case F::lfoRate:
        return hertz(0.2f + 4.0f * v);
    case F::octaves:
    {
        const auto st = (v * 2.0f - 1.0f) * 12.0f;
        return (st > 0.0f ? "+" : "") + juce::String(st, 1) + " st";
    }
    case F::fineTune:
    {
        const auto cents = 1200.0f * std::log2(1.0f + (v * 2.0f - 1.0f) * 0.01f);
        return (cents > 0.0f ? "+" : "") + juce::String(cents, 1) + " ct";
    }
    case F::tilt:
    {
        const auto db = (v * 2.0f - 1.0f) * 4.0f;
        return (db > 0.0f ? "+" : "") + juce::String(db, 1) + " dB/oct";
    }
    }
    return percent(v);
}

std::pair<int, int> timbreKnobs(Method method)
{
    switch (method)
    {
    case Method::subtractive:
        return {6, 15}; // cutoff, resonance
    case Method::wavetable:
        return {1, 8}; // position, tilt
    case Method::additive:
        return {0, 2}; // rolloff, formant
    case Method::spectral:
        return {0, 6}; // peak 1, harmonic/noise
    case Method::fm:
        return {14, 2}; // index, operator 2 ratio
    case Method::physical:
        return {5, 2}; // brightness, decay
    case Method::granular:
        return {1, 6}; // position, jitter
    case Method::sampling:
        return {3, 5}; // tilt, bump frequency
    case Method::hybrid:
        return {31, 0}; // crossfade, A's first knob
    }
    return {0, 1};
}

std::pair<int, int> timbreKnobsFor(Method method, Method hybridA)
{
    if (method == Method::hybrid)
        return {kNumKnobs - 1, timbreKnobs(hybridA).first};
    return timbreKnobs(method);
}

namespace
{
ParamSpec make(const juce::String& id, const juce::String& name, const juce::String& shortName, Group group,
               Unit unit, float min, float max, float def, float skewCentre = -1.0f)
{
    ParamSpec s;
    s.id = id;
    s.name = name;
    s.shortName = shortName;
    s.group = group;
    s.unit = unit;
    s.min = min;
    s.max = max;
    s.defaultValue = def;
    s.skewCentre = skewCentre;
    s.isInteger = unit == Unit::count;
    return s;
}

std::vector<ParamSpec> buildSpecs()
{
    std::vector<ParamSpec> specs;
    auto add = [&specs](ParamSpec s)
    {
        specs.push_back(std::move(s));
    };

    auto tier = make(id::modelTier, "Model size", "Size", Group::model, Unit::none, 0.0f, 3.0f, 1.0f);
    tier.choices = {"Lite", "Standard", "Pro", "Studio"};
    add(tier);

    auto method =
        make(id::method, "Method", "Method", Group::sound, Unit::none, 0.0f, kNumMethods - 1.0f, 0.0f);
    for (const auto& m : kMethods)
        method.choices.add(m.name);
    add(method);

    juce::StringArray components;
    for (auto m : kHybridComponents)
        components.add(kMethods[static_cast<size_t>(m)].name);
    auto a = make(id::hybridA, "Hybrid A", "A", Group::sound, Unit::none, 0.0f, 4.0f, 0.0f);
    a.choices = components;
    add(a);
    auto b = make(id::hybridB, "Hybrid B", "B", Group::sound, Unit::none, 0.0f, 4.0f, 1.0f);
    b.choices = components;
    add(b);

    for (int i = 0; i < kNumKnobs; ++i)
    {
        auto spec = make(id::knob(i), "Knob " + juce::String(i + 1), "K" + juce::String(i + 1), Group::knob,
                         Unit::percent, 0.0f, 1.0f, 0.5f);
        spec.knobIndex = i;
        add(spec);
    }

    add(make(id::voices, "Voices", "Voices", Group::voice, Unit::count, 1.0f, 32.0f, 16.0f));
    add(make(id::glide, "Glide", "Glide", Group::voice, Unit::milliseconds, 0.0f, 2000.0f, 0.0f, 200.0f));
    auto bend = make(id::bendRange, "Bend range", "Bend", Group::voice, Unit::semitones, 0.0f, 24.0f, 2.0f);
    bend.isInteger = true;
    add(bend);

    add(make(id::outputGain, "Output", "Output", Group::output, Unit::decibels, -48.0f, 6.0f, -6.0f));

    return specs;
}

Method componentAt(int index)
{
    return kHybridComponents[static_cast<size_t>(juce::jlimit(0, 4, index))];
}
} // namespace

const std::vector<ParamSpec>& allSpecs()
{
    static const auto specs = buildSpecs();
    return specs;
}

const ParamSpec* findSpec(const juce::String& paramId)
{
    static const auto index = []
    {
        std::unordered_map<juce::String, const ParamSpec*> map;
        for (auto& s : allSpecs())
            map[s.id] = &s;
        return map;
    }();

    auto it = index.find(paramId);
    return it != index.end() ? it->second : nullptr;
}

SoundShape readShape(const juce::AudioProcessorValueTreeState& state)
{
    auto raw = [&state](const juce::String& id)
    {
        const auto* p = state.getRawParameterValue(id);
        return p != nullptr ? juce::roundToInt(p->load()) : 0;
    };
    SoundShape s;
    s.method = static_cast<Method>(juce::jlimit(0, kNumMethods - 1, raw(id::method)));
    s.hybridA = componentAt(raw(id::hybridA));
    s.hybridB = componentAt(raw(id::hybridB));
    return s;
}

int packShape(const SoundShape& s)
{
    return static_cast<int>(s.method) | (static_cast<int>(s.hybridA) << 4) |
           (static_cast<int>(s.hybridB) << 8);
}

SoundShape unpackShape(int packed)
{
    SoundShape s;
    s.method = static_cast<Method>(juce::jlimit(0, kNumMethods - 1, packed & 0xf));
    s.hybridA = static_cast<Method>(juce::jlimit(0, kNumMethods - 1, (packed >> 4) & 0xf));
    s.hybridB = static_cast<Method>(juce::jlimit(0, kNumMethods - 1, (packed >> 8) & 0xf));
    return s;
}

juce::String unitSuffix(Unit unit)
{
    switch (unit)
    {
    case Unit::percent:
        return "%";
    case Unit::hertz:
        return "Hz";
    case Unit::milliseconds:
        return "ms";
    case Unit::seconds:
        return "s";
    case Unit::decibels:
        return "dB";
    case Unit::semitones:
        return "st";
    case Unit::ratio:
        return ":1";
    case Unit::none:
    case Unit::count:
        break;
    }
    return {};
}

juce::String formatValue(const ParamSpec& spec, float v)
{
    if (!spec.choices.isEmpty())
        return spec.choices[juce::jlimit(0, spec.choices.size() - 1, juce::roundToInt(v))];

    switch (spec.unit)
    {
    case Unit::percent:
        return juce::String(juce::roundToInt(v * 100.0f)) + " %";
    case Unit::hertz:
        return hertz(v);
    case Unit::milliseconds:
        return v >= 1000.0f ? juce::String(v / 1000.0f, 2) + " s" : number(v, v < 10.0f ? 1 : 0) + " ms";
    case Unit::seconds:
        return juce::String(v, 2) + " s";
    case Unit::decibels:
        return (v > 0.0f ? "+" : "") + juce::String(v, 1) + " dB";
    case Unit::semitones:
        return juce::String(v, v < 1.0f ? 2 : 1) + " st";
    case Unit::ratio:
        return juce::String(v, 2) + ":1";
    case Unit::count:
        return juce::String(juce::roundToInt(v));
    case Unit::none:
        break;
    }
    return juce::String(v, 2);
}

float parseValue(const ParamSpec& spec, const juce::String& text)
{
    const auto t = text.trim().toLowerCase();

    if (!spec.choices.isEmpty())
    {
        for (int i = 0; i < spec.choices.size(); ++i)
            if (spec.choices[i].equalsIgnoreCase(t))
                return static_cast<float>(i);
    }

    auto v = t.retainCharacters("0123456789.-+").getFloatValue();

    switch (spec.unit)
    {
    case Unit::percent:
        v /= 100.0f;
        break;
    case Unit::hertz:
        if (t.contains("k"))
            v *= 1000.0f;
        break;
    case Unit::milliseconds:
        if (t.endsWith("s") && !t.endsWith("ms"))
            v *= 1000.0f;
        break;
    case Unit::none:
    case Unit::seconds:
    case Unit::decibels:
    case Unit::semitones:
    case Unit::ratio:
    case Unit::count:
        break;
    }

    return juce::jlimit(spec.min, spec.max, v);
}

juce::AudioProcessorValueTreeState::ParameterLayout createLayout(const std::atomic<int>* packedShape)
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    for (auto& spec : allSpecs())
    {
        const juce::ParameterID pid{spec.id, kLayoutVersion};

        if (!spec.choices.isEmpty())
        {
            layout.add(std::make_unique<juce::AudioParameterChoice>(pid, spec.name, spec.choices,
                                                                    juce::roundToInt(spec.defaultValue)));
            continue;
        }

        if (spec.isInteger)
        {
            layout.add(std::make_unique<juce::AudioParameterInt>(pid, spec.name, static_cast<int>(spec.min),
                                                                 static_cast<int>(spec.max),
                                                                 juce::roundToInt(spec.defaultValue)));
            continue;
        }

        juce::NormalisableRange<float> range{spec.min, spec.max};
        if (spec.skewCentre > spec.min && spec.skewCentre < spec.max)
            range.setSkewForCentre(spec.skewCentre);

        const auto* specPtr = &spec;
        auto toText = [specPtr, packedShape](float v, int)
        {
            if (specPtr->group == Group::knob && packedShape != nullptr)
            {
                const auto s = unpackShape(packedShape->load(std::memory_order_relaxed));
                return formatKnob(knobInfoFor(s.method, s.hybridA, s.hybridB, specPtr->knobIndex), v);
            }
            return formatValue(*specPtr, v);
        };
        layout.add(std::make_unique<juce::AudioParameterFloat>(
            pid, spec.name, range, spec.defaultValue,
            juce::AudioParameterFloatAttributes()
                .withLabel(spec.group == Group::knob ? juce::String() : unitSuffix(spec.unit))
                .withStringFromValueFunction(toText)
                .withValueFromStringFunction([specPtr](const juce::String& t)
                                             { return parseValue(*specPtr, t); })));
    }

    return layout;
}
} // namespace alfa::params
