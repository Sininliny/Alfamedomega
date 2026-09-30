#include "alfa/plugin/PresetLibrary.h"

namespace alfa
{
namespace
{
using namespace params;

juce::String blendId(Method m)
{
    return id::blend(static_cast<int>(m));
}
juce::String engineId(Method m, int slot)
{
    return id::engine(static_cast<int>(m), slot);
}

void setParam(juce::AudioProcessorValueTreeState& state, const juce::String& paramId, float value)
{
    if (auto* p = state.getParameter(paramId))
        p->setValueNotifyingHost(p->convertTo0to1(value));
}
} // namespace

PresetLibrary::PresetLibrary()
{
    // Placeholder sounds that exercise different engines, so the browser, the
    // Method blend and the tier badges have something to show.
    presets = {
        {"Init", "Basic", Tier::lite, {}},
        {"Warm Pad",
         "Pad",
         Tier::standard,
         {{blendId(Method::subtractive), 0.6f},
          {blendId(Method::wavetable), 0.5f},
          {blendId(Method::spectral), 0.2f},
          {id::attack, 600.0f},
          {id::release, 2500.0f},
          {engineId(Method::subtractive, 0), 1200.0f},
          {id::timbreX, 0.3f},
          {id::timbreY, 0.6f}}},
        {"Glass Keys",
         "Keys",
         Tier::standard,
         {{blendId(Method::subtractive), 0.0f},
          {blendId(Method::fm), 0.8f},
          {blendId(Method::additive), 0.3f},
          {engineId(Method::fm, 0), 3.5f},
          {engineId(Method::fm, 1), 4.0f},
          {id::decay, 900.0f},
          {id::sustain, 0.2f}}},
        {"Plucked Wire",
         "Pluck",
         Tier::lite,
         {{blendId(Method::subtractive), 0.2f},
          {blendId(Method::physical), 0.9f},
          {id::attack, 1.0f},
          {id::decay, 700.0f},
          {id::sustain, 0.0f},
          {id::release, 300.0f}}},
        {"Sampled Choir",
         "Vocal",
         Tier::pro,
         {{blendId(Method::subtractive), 0.0f},
          {blendId(Method::sampling), 0.9f},
          {blendId(Method::granular), 0.3f},
          {engineId(Method::sampling, 5), 0.6f},
          {id::attack, 300.0f},
          {id::release, 1500.0f}}},
        {"Spectral Drift",
         "Texture",
         Tier::studio,
         {{blendId(Method::subtractive), 0.0f},
          {blendId(Method::spectral), 0.8f},
          {blendId(Method::hybrid), 0.4f},
          {engineId(Method::spectral, 5), 0.7f},
          {id::latentDrift, 0.6f},
          {id::timbreY, 0.9f}}},
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
