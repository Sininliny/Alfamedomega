#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <array>
#include <vector>

// The single source of truth for Alfamedomega's parameters. The processor, the
// preset system and the UI all read IDs, ranges and display metadata from here,
// so a parameter is added in exactly one place.
namespace alfa::params
{
// Bump when a parameter's meaning changes; new parameters don't need a bump.
inline constexpr int kLayoutVersion = 1;

enum class Method
{
    subtractive,
    additive,
    fm,
    wavetable,
    granular,
    physical,
    sampling,
    spectral,
    hybrid
};

inline constexpr int kNumMethods = 9;

enum class Tier
{
    lite,
    standard,
    pro,
    studio
};

inline constexpr int kNumTiers = 4;

struct MethodInfo
{
    Method method;
    const char* id;      // used in parameter IDs, e.g. "blend_fm"
    const char* name;    // shown in the UI
    float capacityShare; // share of model capacity, from the capacity plan
};

// Expert capacity shares as computed by docs/tiers/tier_budget.py (the trunk is
// shared and not counted). Estimates; the trained model's manifest wins.
inline constexpr std::array<MethodInfo, kNumMethods> kMethods{{
    {Method::subtractive, "subtractive", "Subtractive", 0.150f},
    {Method::additive, "additive", "Additive", 0.039f},
    {Method::fm, "fm", "FM", 0.115f},
    {Method::wavetable, "wavetable", "Wavetable", 0.190f},
    {Method::granular, "granular", "Granular", 0.047f},
    {Method::physical, "physical", "Physical", 0.079f},
    {Method::sampling, "sampling", "Sampling", 0.287f},
    {Method::spectral, "spectral", "Spectral", 0.047f},
    {Method::hybrid, "hybrid", "Hybrid", 0.047f},
}};

struct TierInfo
{
    Tier tier;
    const char* name;
    const char* shortName;
    float totalParamsMillions;
    int budgetVoices; // voices per reference core, from tier_budget.py
    float latencyMs;  // model lookahead latency
    int voicesPerInstance;
};

inline constexpr float kControlRateHz = 250.0f;

// Figures follow docs/tiers (v3 spec) and are provisional.
inline constexpr std::array<TierInfo, kNumTiers> kTiers{{
    {Tier::lite, "Lite", "LITE", 4.0f, 167, 4.0f, 16},
    {Tier::standard, "Standard", "STD", 16.0f, 57, 8.0f, 16},
    {Tier::pro, "Pro", "PRO", 64.0f, 22, 8.0f, 16},
    {Tier::studio, "Studio", "STUDIO", 256.0f, 5, 8.0f, 8},
}};

enum class Unit
{
    none,
    percent,
    hertz,
    milliseconds,
    seconds,
    decibels,
    semitones,
    ratio,
    count
};

enum class Group
{
    blend,
    macro,
    timbre,
    voice,
    engine,
    neural,
    model,
    output
};

// Display metadata for one parameter. The APVTS owns the value; this owns how
// it is labelled, grouped and which model size it needs.
struct ParamSpec
{
    juce::String id;
    juce::String name;
    juce::String shortName;
    Group group;
    Unit unit = Unit::none;
    float min = 0.0f, max = 1.0f, defaultValue = 0.0f, skewCentre = -1.0f;
    int engineIndex = -1; // for Group::engine and Group::blend
    Tier minimumTier = Tier::lite;
    juce::StringArray choices; // non-empty for choice parameters
    bool isInteger = false;
};

// IDs that the processor and UI reference directly.
namespace id
{
inline const juce::String modelTier{"model_tier"};
inline const juce::String timbreX{"timbre_x"};
inline const juce::String timbreY{"timbre_y"};
inline const juce::String attack{"amp_attack"};
inline const juce::String decay{"amp_decay"};
inline const juce::String sustain{"amp_sustain"};
inline const juce::String release{"amp_release"};
inline const juce::String voices{"voices"};
inline const juce::String glide{"glide"};
inline const juce::String bendRange{"bend_range"};
inline const juce::String outputGain{"output_gain"};
inline const juce::String latentDrift{"latent_drift"};
inline const juce::String seed{"seed"};
inline const juce::String followModel{"follow_model"};

juce::String blend(int methodIndex);
juce::String macro(int macroIndex);             // 0-based, 24 macros
juce::String engine(int methodIndex, int slot); // slot 0-5
} // namespace id

inline constexpr int kNumMacros = 24;
inline constexpr int kNumEngineSlots = 6;

// Every parameter, in a stable order.
const std::vector<ParamSpec>& allSpecs();
const ParamSpec* findSpec(const juce::String& paramId);

// The six primary parameters of one engine, in slot order.
std::vector<const ParamSpec*> engineSpecs(int methodIndex);

juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

// Formats a value in its real unit ("1.20 kHz", "35 ms", "-6.0 dB").
juce::String formatValue(const ParamSpec& spec, float value);
juce::String unitSuffix(Unit unit);

// Parses typed text back to a real value ("1.2k", "1.2 kHz", "50 %", "0.4 s").
float parseValue(const ParamSpec& spec, const juce::String& text);
} // namespace alfa::params
