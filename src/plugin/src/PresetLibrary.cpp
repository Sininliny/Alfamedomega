#include "alfa/plugin/PresetLibrary.h"

namespace alfa
{
namespace
{
using namespace params;

// Knob n as numbered in the UI (1-based).
juce::String kn(int n)
{
    return id::knob(n - 1);
}

float methodValue(Method m)
{
    return static_cast<float>(static_cast<int>(m));
}

float componentValue(Method m)
{
    const auto it = std::find(kHybridComponents.begin(), kHybridComponents.end(), m);
    return static_cast<float>(std::distance(kHybridComponents.begin(), it));
}

void setParam(juce::AudioProcessorValueTreeState& state, const juce::String& paramId, float value)
{
    if (auto* p = state.getParameter(paramId))
        p->setValueNotifyingHost(p->convertTo0to1(value));
}
} // namespace

PresetLibrary::PresetLibrary()
{
    // Placeholder sounds, one or two per method, so the browser, the method
    // picker and the knob labels have something to show. Knob numbers and
    // meanings follow the teacher programs (ParameterLayout.cpp).
    presets = {
        {"Init", "Basic", Tier::lite, {}},
        {"Warm Pad",
         "Pad",
         Tier::standard,
         {{id::method, methodValue(Method::wavetable)},
          {kn(1), 0.3f},
          {kn(2), 0.35f},
          {kn(3), 0.7f},
          {kn(5), 0.8f},
          {kn(6), 0.8f},
          {kn(7), 0.6f},
          {kn(9), 0.4f},
          {kn(19), 0.72f},
          {kn(22), 0.8f},
          {kn(23), 0.8f},
          {kn(24), 0.45f}}},
        {"Glass Keys",
         "Keys",
         Tier::standard,
         {{id::method, methodValue(Method::fm)},
          {kn(1), 0.0f},
          {kn(3), 0.43f},
          {kn(11), 0.1f},
          {kn(12), 0.55f},
          {kn(13), 0.2f},
          {kn(15), 0.45f},
          {kn(19), 0.05f},
          {kn(20), 0.7f},
          {kn(21), 0.15f},
          {kn(22), 0.6f}}},
        {"Plucked Wire",
         "Pluck",
         Tier::lite,
         {{id::method, methodValue(Method::physical)},
          {kn(1), 0.1f},
          {kn(2), 0.3f},
          {kn(3), 0.55f},
          {kn(4), 0.3f},
          {kn(6), 0.7f},
          {kn(7), 0.2f}}},
        {"Sampled Choir",
         "Vocal",
         Tier::pro,
         {{id::method, methodValue(Method::sampling)},
          {kn(1), 0.6f},
          {kn(5), 0.7f},
          {kn(6), 0.5f},
          {kn(19), 0.7f},
          {kn(22), 0.75f}}},
        {"Spectral Drift",
         "Texture",
         Tier::studio,
         {{id::method, methodValue(Method::spectral)},
          {kn(1), 0.3f},
          {kn(4), 0.7f},
          {kn(7), 0.6f},
          {kn(8), 0.15f},
          {kn(9), 0.8f},
          {kn(19), 0.8f},
          {kn(22), 0.85f}}},
        {"Struck Bell",
         "Bell",
         Tier::pro,
         {{id::method, methodValue(Method::hybrid)},
          {id::hybridA, componentValue(Method::fm)},
          {id::hybridB, componentValue(Method::physical)},
          {kn(1), 0.0f},
          {kn(3), 0.62f},
          {kn(15), 0.4f},
          {kn(19), 0.02f},
          {kn(20), 0.75f},
          {kn(21), 0.0f},
          {kn(22), 0.25f}, // A release, and B's (physical) resonator: beam
          {kn(24), 0.6f},  // B decay time
          {kn(32), 0.45f}}},
    };
}

void PresetLibrary::apply(const Preset& preset, juce::AudioProcessorValueTreeState& state,
                          const juce::StringArray& lockedParamIds)
{
    for (const auto& spec : allSpecs())
    {
        if (spec.group == Group::model || lockedParamIds.contains(spec.id))
            continue; // loading a preset never changes the model size

        float value = spec.defaultValue;
        for (const auto& [paramId, v] : preset.values)
            if (paramId == spec.id)
                value = v;

        setParam(state, spec.id, value);
    }
}

Preset PresetLibrary::capture(const juce::String& name, const juce::String& category, Tier madeOn,
                              juce::AudioProcessorValueTreeState& state)
{
    Preset p{name, category, madeOn, {}};
    for (const auto& spec : allSpecs())
    {
        if (spec.group == Group::model)
            continue;
        if (auto* param = state.getParameter(spec.id))
        {
            const auto v = param->convertFrom0to1(param->getValue());
            if (!juce::approximatelyEqual(v, spec.defaultValue))
                p.values.emplace_back(spec.id, v);
        }
    }
    return p;
}

} // namespace alfa
