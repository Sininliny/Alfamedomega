#include "alfa/plugin/ParameterLayout.h"

namespace alfa::params
{
namespace id
{
juce::String blend(int methodIndex)
{
    return "blend_" + juce::String(kMethods[static_cast<size_t>(methodIndex)].id);
}
juce::String macro(int macroIndex)
{
    return "macro_" + juce::String(macroIndex + 1);
}

juce::String engine(int methodIndex, int slot)
{
    return juce::String(kMethods[static_cast<size_t>(methodIndex)].id) + "_p" + juce::String(slot + 1);
}
} // namespace id

namespace
{
struct EngineSlot
{
    const char* name;
    const char* shortName;
    Unit unit;
    float min, max, def, skewCentre;
    Tier minimumTier;
};

// Six primary controls per engine, as drawn in the Sound design layer panel.
// A few need the capacity only the larger models have; they stay visible
// but dimmed on smaller tiers (see ui-design.md, "Controls a smaller model
// cannot drive").
constexpr EngineSlot engineSlots[kNumMethods][kNumEngineSlots] = {
    // subtractive
    {{"Cutoff", "Cutoff", Unit::hertz, 20.0f, 20000.0f, 2000.0f, 1000.0f, Tier::lite},
     {"Resonance", "Reso", Unit::percent, 0.0f, 1.0f, 0.2f, -1.0f, Tier::lite},
     {"Drive", "Drive", Unit::decibels, 0.0f, 24.0f, 0.0f, -1.0f, Tier::lite},
     {"Oscillator shape", "Shape", Unit::percent, 0.0f, 1.0f, 0.5f, -1.0f, Tier::lite},
     {"Detune", "Detune", Unit::semitones, 0.0f, 1.0f, 0.1f, -1.0f, Tier::lite},
     {"Filter envelope", "F.Env", Unit::percent, -1.0f, 1.0f, 0.3f, -1.0f, Tier::lite}},
    // additive
    {{"Partials", "Partials", Unit::count, 1.0f, 256.0f, 64.0f, 32.0f, Tier::lite},
     {"Tilt", "Tilt", Unit::decibels, -12.0f, 12.0f, -3.0f, -1.0f, Tier::lite},
     {"Odd / even", "Odd/Even", Unit::percent, -1.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Inharmonicity", "Inharm", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Spread", "Spread", Unit::percent, 0.0f, 1.0f, 0.2f, -1.0f, Tier::lite},
     {"Partial drift", "Drift", Unit::percent, 0.0f, 1.0f, 0.1f, -1.0f, Tier::pro}},
    // fm
    {{"Ratio", "Ratio", Unit::ratio, 0.25f, 16.0f, 2.0f, 2.0f, Tier::lite},
     {"Index", "Index", Unit::none, 0.0f, 20.0f, 2.0f, 4.0f, Tier::lite},
     {"Feedback", "Fdbk", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Algorithm", "Algo", Unit::count, 1.0f, 8.0f, 1.0f, -1.0f, Tier::lite},
     {"Index envelope", "I.Env", Unit::percent, -1.0f, 1.0f, 0.4f, -1.0f, Tier::lite},
     {"Operator drift", "Drift", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::pro}},
    // wavetable
    {{"Position", "Position", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Warp", "Warp", Unit::percent, -1.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Unison", "Unison", Unit::count, 1.0f, 16.0f, 1.0f, -1.0f, Tier::lite},
     {"Unison detune", "Detune", Unit::semitones, 0.0f, 1.0f, 0.15f, -1.0f, Tier::lite},
     {"Table morph", "Morph", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::standard},
     {"Brightness", "Bright", Unit::percent, 0.0f, 1.0f, 0.5f, -1.0f, Tier::lite}},
    // granular
    {{"Grain size", "Size", Unit::milliseconds, 5.0f, 500.0f, 60.0f, 60.0f, Tier::lite},
     {"Density", "Density", Unit::hertz, 1.0f, 200.0f, 30.0f, 30.0f, Tier::lite},
     {"Position", "Position", Unit::percent, 0.0f, 1.0f, 0.3f, -1.0f, Tier::lite},
     {"Jitter", "Jitter", Unit::percent, 0.0f, 1.0f, 0.2f, -1.0f, Tier::lite},
     {"Pitch spread", "P.Spread", Unit::semitones, 0.0f, 24.0f, 0.0f, -1.0f, Tier::lite},
     {"Grain texture", "Texture", Unit::percent, 0.0f, 1.0f, 0.5f, -1.0f, Tier::pro}},
    // physical
    {{"Exciter", "Exciter", Unit::percent, 0.0f, 1.0f, 0.5f, -1.0f, Tier::lite},
     {"Damping", "Damping", Unit::percent, 0.0f, 1.0f, 0.4f, -1.0f, Tier::lite},
     {"Stiffness", "Stiff", Unit::percent, 0.0f, 1.0f, 0.2f, -1.0f, Tier::lite},
     {"Body size", "Body", Unit::percent, 0.0f, 1.0f, 0.5f, -1.0f, Tier::lite},
     {"Pickup position", "Pickup", Unit::percent, 0.0f, 1.0f, 0.3f, -1.0f, Tier::lite},
     {"Material", "Material", Unit::percent, 0.0f, 1.0f, 0.5f, -1.0f, Tier::standard}},
    // sampling
    {{"Start", "Start", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Loop length", "Loop", Unit::milliseconds, 1.0f, 2000.0f, 200.0f, 200.0f, Tier::lite},
     {"Pitch", "Pitch", Unit::semitones, -24.0f, 24.0f, 0.0f, -1.0f, Tier::lite},
     {"Time stretch", "Stretch", Unit::ratio, 0.25f, 4.0f, 1.0f, 1.0f, Tier::lite},
     {"Formant", "Formant", Unit::semitones, -12.0f, 12.0f, 0.0f, -1.0f, Tier::lite},
     {"Neural resynthesis", "Resynth", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::pro}},
    // spectral
    {{"Freeze", "Freeze", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Blur", "Blur", Unit::percent, 0.0f, 1.0f, 0.1f, -1.0f, Tier::lite},
     {"Shift", "Shift", Unit::semitones, -24.0f, 24.0f, 0.0f, -1.0f, Tier::lite},
     {"Tonal / noise", "Tonal", Unit::percent, 0.0f, 1.0f, 0.5f, -1.0f, Tier::lite},
     {"Smear", "Smear", Unit::milliseconds, 0.0f, 1000.0f, 0.0f, 100.0f, Tier::lite},
     {"Spectral morph", "Morph", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::studio}},
    // hybrid
    {{"Source A / B", "A/B", Unit::percent, 0.0f, 1.0f, 0.5f, -1.0f, Tier::lite},
     {"Cross-mod", "X-Mod", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Ring mod", "Ring", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Vocoder", "Vocoder", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::standard},
     {"Envelope follow", "Follow", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::lite},
     {"Latent blend", "Latent", Unit::percent, 0.0f, 1.0f, 0.0f, -1.0f, Tier::pro}},
};

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

    // Default blend: a mostly-subtractive starting patch so a fresh
    // instance sounds like something familiar.
    for (int m = 0; m < kNumMethods; ++m)
    {
        const auto* name = kMethods[static_cast<size_t>(m)].name;
        auto spec = make(id::blend(m), juce::String(name) + " blend", name, Group::blend, Unit::percent, 0.0f,
                         1.0f, m == 0 ? 0.8f : (m == 3 ? 0.3f : 0.0f));
        spec.engineIndex = m;
        add(spec);
    }

    for (int i = 0; i < kNumMacros; ++i)
        add(make(id::macro(i), "Macro " + juce::String(i + 1), "M" + juce::String(i + 1), Group::macro,
                 Unit::percent, 0.0f, 1.0f, 0.0f));

    add(make(id::timbreX, "Warm to bright", "Bright", Group::timbre, Unit::percent, 0.0f, 1.0f, 0.5f));
    add(make(id::timbreY, "Stable to evolving", "Evolve", Group::timbre, Unit::percent, 0.0f, 1.0f, 0.3f));

    add(make(id::attack, "Attack", "Attack", Group::voice, Unit::milliseconds, 0.0f, 10000.0f, 5.0f, 200.0f));
    add(make(id::decay, "Decay", "Decay", Group::voice, Unit::milliseconds, 1.0f, 10000.0f, 300.0f, 400.0f));
    add(make(id::sustain, "Sustain", "Sustain", Group::voice, Unit::percent, 0.0f, 1.0f, 0.7f));
    add(make(id::release, "Release", "Release", Group::voice, Unit::milliseconds, 1.0f, 20000.0f, 400.0f,
             600.0f));
    add(make(id::voices, "Voices", "Voices", Group::voice, Unit::count, 1.0f, 32.0f, 16.0f));
    add(make(id::glide, "Glide", "Glide", Group::voice, Unit::milliseconds, 0.0f, 2000.0f, 0.0f, 200.0f));
    auto bend = make(id::bendRange, "Bend range", "Bend", Group::voice, Unit::semitones, 0.0f, 24.0f, 2.0f);
    bend.isInteger = true;
    add(bend);

    add(make(id::latentDrift, "Latent drift", "Drift", Group::neural, Unit::percent, 0.0f, 1.0f, 0.1f));
    add(make(id::seed, "Seed", "Seed", Group::neural, Unit::count, 0.0f, 9999.0f, 1.0f));
    add(make(id::followModel, "Follow model", "Follow", Group::neural, Unit::percent, 0.0f, 1.0f, 1.0f));

    add(make(id::outputGain, "Output", "Output", Group::output, Unit::decibels, -48.0f, 6.0f, -6.0f));

    for (int m = 0; m < kNumMethods; ++m)
    {
        for (int slot = 0; slot < kNumEngineSlots; ++slot)
        {
            const auto& e = engineSlots[m][slot];
            auto spec =
                make(id::engine(m, slot), juce::String(kMethods[static_cast<size_t>(m)].name) + " " + e.name,
                     e.shortName, Group::engine, e.unit, e.min, e.max, e.def, e.skewCentre);
            spec.engineIndex = m;
            spec.minimumTier = e.minimumTier;
            add(spec);
        }
    }

    return specs;
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

std::vector<const ParamSpec*> engineSpecs(int methodIndex)
{
    std::vector<const ParamSpec*> out;
    for (auto& s : allSpecs())
        if (s.group == Group::engine && s.engineIndex == methodIndex)
            out.push_back(&s);
    return out;
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
        return v >= 1000.0f ? juce::String(v / 1000.0f, 2) + " kHz"
                            : juce::String(v, v < 100.0f ? 1 : 0) + " Hz";
    case Unit::milliseconds:
        return v >= 1000.0f ? juce::String(v / 1000.0f, 2) + " s"
                            : juce::String(v, v < 10.0f ? 1 : 0) + " ms";
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

juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
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
        layout.add(std::make_unique<juce::AudioParameterFloat>(
            pid, spec.name, range, spec.defaultValue,
            juce::AudioParameterFloatAttributes()
                .withLabel(unitSuffix(spec.unit))
                .withStringFromValueFunction([specPtr](float v, int) { return formatValue(*specPtr, v); })
                .withValueFromStringFunction([specPtr](const juce::String& t)
                                             { return parseValue(*specPtr, t); })));
    }

    return layout;
}
} // namespace alfa::params
