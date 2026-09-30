#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <array>
#include <atomic>
#include <vector>

// The single source of truth for Alfamedomega's parameters. The processor, the
// preset system and the UI all read IDs, ranges and display metadata from here,
// so a parameter is added in exactly one place.
//
// A sound (a "patch") has the shape the model was trained on (training/METHOD.md,
// src/inference/MODEL_CONTRACT.md): one synthesis method, or a hybrid of two
// components, plus 32 knobs in 0..1 whose meaning depends on the method.
namespace alfa::params
{
// Bump when a parameter's meaning changes; new parameters don't need a bump.
// Version 2: method + hybrid pair + 32 knobs replaced the blend, macros and
// per-engine slots of version 1.
inline constexpr int kLayoutVersion = 2;

// Order matches the model contract's one-hot and voice routing.
enum class Method
{
    subtractive,
    wavetable,
    additive,
    spectral,
    fm,
    physical,
    granular,
    sampling,
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
    const char* id;      // as training and the preference log name it
    const char* name;    // shown in the UI
    float capacityShare; // share of model capacity, from the capacity plan
    const char* summary; // one line for the method picker and Signal flow
};

// Expert capacity shares as computed by docs/tiers/tier_budget.py (the trunk is
// shared and not counted). Estimates; the trained model's manifest wins.
inline constexpr std::array<MethodInfo, kNumMethods> kMethods{{
    {Method::subtractive, "subtractive", "Subtractive", 0.150f, "Oscillators through a resonant filter"},
    {Method::wavetable, "wavetable", "Wavetable", 0.190f, "Scans through a table of single-cycle waves"},
    {Method::additive, "additive", "Additive", 0.039f, "Builds the sound partial by partial"},
    {Method::spectral, "spectral", "Spectral", 0.047f, "Shapes a moving spectral envelope"},
    {Method::fm, "fm", "FM", 0.115f, "Operators modulating each other's frequency"},
    {Method::physical, "physical", "Physical", 0.079f, "A struck, plucked or bowed resonator"},
    {Method::granular, "granular", "Granular", 0.047f, "Layers of short grains from a sound bank"},
    {Method::sampling, "sampling", "Sampling", 0.287f, "Plays and reshapes recorded samples"},
    {Method::hybrid, "hybrid", "Hybrid", 0.047f, "Two methods merged into one voice"},
}};

// Methods a hybrid can be built from, in the order of the hybrid choice
// parameters. Additive with physical is not allowed: both use the partials
// primitive.
inline constexpr std::array<Method, 5> kHybridComponents{Method::subtractive, Method::wavetable,
                                                         Method::additive, Method::fm, Method::physical};

bool isValidHybridPair(Method a, Method b);

struct TierInfo
{
    Tier tier;
    const char* name;
    const char* shortName;
    float totalParamsMillions;
    int budgetVoices; // voices per reference core, from tier_budget.py
    float latencyMs;  // model lookahead latency
    int voicesPerInstance;
    int fmOperators; // TierLimits in training/METHOD.md
    int unisonReads;
};

inline constexpr float kControlRateHz = 250.0f;

// Figures follow docs/tiers (v3 spec) and are provisional.
inline constexpr std::array<TierInfo, kNumTiers> kTiers{{
    {Tier::lite, "Lite", "LITE", 4.0f, 167, 4.0f, 16, 2, 2},
    {Tier::standard, "Standard", "STD", 16.0f, 57, 8.0f, 16, 4, 4},
    {Tier::pro, "Pro", "PRO", 64.0f, 22, 8.0f, 16, 6, 6},
    {Tier::studio, "Studio", "STUDIO", 256.0f, 5, 8.0f, 8, 8, 8},
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
    sound, // method and hybrid components
    knob,
    voice,
    model,
    output
};

// Display metadata for one parameter. The APVTS owns the value; this owns how
// it is labelled and grouped.
struct ParamSpec
{
    juce::String id;
    juce::String name;
    juce::String shortName;
    Group group;
    Unit unit = Unit::none;
    float min = 0.0f, max = 1.0f, defaultValue = 0.0f, skewCentre = -1.0f;
    int knobIndex = -1;        // for Group::knob
    juce::StringArray choices; // non-empty for choice parameters
    bool isInteger = false;
};

inline constexpr int kNumKnobs = 32;

// IDs that the processor and UI reference directly.
namespace id
{
inline const juce::String modelTier{"model_tier"};
inline const juce::String method{"method"};
inline const juce::String hybridA{"hybrid_a"};
inline const juce::String hybridB{"hybrid_b"};
inline const juce::String voices{"voices"};
inline const juce::String glide{"glide"};
inline const juce::String bendRange{"bend_range"};
inline const juce::String outputGain{"output_gain"};

juce::String knob(int index); // 0-based: "knob_01" .. "knob_32"
} // namespace id

// What one knob means for the current method, from the teacher programs in
// training/alfamedomega_train.py. The model learns to imitate those programs, so
// their knob meanings are the knob meanings.
struct KnobInfo
{
    enum class Format
    {
        percent,
        bipolar,     // -100..+100 %
        attackTime,  // 0.001 * 2^(11k) s
        decayTime,   // 0.01 * 2^(9k) s
        cutoff,      // 30 * 2^(9.5k) Hz
        formant,     // 200 * 2^(6k) Hz
        resonance,   // Q 0.5 + 19.5 k^2
        unison,      // off below 50 %
        spreadCents, // 3 + 47 k^2 cents
        fmRatio,     // round(1 + 7k)
        fmAlgorithm, // stack, pairs, one-to-many, additive
        filterMode,  // low-pass to band-pass to high-pass
        table64,
        source16,
        sample8,
        resonator,    // string, beam, membrane, tube, plate
        onOff,        // on above the threshold in `threshold`
        decaySeconds, // 0.05 * 2^(7k) s
        lfoRate,      // 0.2 + 4k Hz
        octaves,      // (2k - 1) octaves
        fineTune,     // +-1 % in cents
        tilt,         // (2k - 1) * 4 dB/oct
    };

    juce::String name;
    juce::String shortName;
    Format format = Format::percent;
    bool used = false;
    Tier minimumTier = Tier::lite;
    float threshold = 0.5f;
    juce::String note; // extra explanation for the tooltip, may be empty
};

// Knob meaning for a plain method. For Method::hybrid use hybridKnobInfo.
KnobInfo knobInfo(Method method, int knob);

// A hybrid runs component A on the knobs as they are and component B on the
// knobs rotated by 11 (knob i drives B's knob (i + 11) mod 32); knob 32 is the
// A-to-B crossfade.
KnobInfo hybridKnobInfo(Method a, Method b, int knob);

// The knob for the current sound: handles hybrid.
KnobInfo knobInfoFor(Method method, Method hybridA, Method hybridB, int knob);

juce::String formatKnob(const KnobInfo& info, float value);

// Two knobs the Timbre space pad drives for each method: the ones that change
// the character most, from the teacher programs.
std::pair<int, int> timbreKnobs(Method method);

// For a hybrid: the crossfade across, component A's main knob up.
std::pair<int, int> timbreKnobsFor(Method method, Method hybridA);

// Every parameter, in a stable order.
const std::vector<ParamSpec>& allSpecs();
const ParamSpec* findSpec(const juce::String& paramId);

// Reads the current sound's method and hybrid components from the parameters.
struct SoundShape
{
    Method method = Method::subtractive;
    Method hybridA = Method::subtractive;
    Method hybridB = Method::wavetable;
};
SoundShape readShape(const juce::AudioProcessorValueTreeState& state);

// `shape` lets knob parameters show their text in the current method's terms
// in the host; pass nullptr for generic percentages. It must outlive the layout.
juce::AudioProcessorValueTreeState::ParameterLayout
createLayout(const std::atomic<int>* packedShape = nullptr);

// Packs a SoundShape into one int for the atomic above.
int packShape(const SoundShape& s);
SoundShape unpackShape(int packed);

// Formats a value in its real unit ("1.20 kHz", "35 ms", "-6.0 dB").
juce::String formatValue(const ParamSpec& spec, float value);
juce::String unitSuffix(Unit unit);

// Parses typed text back to a real value ("1.2k", "1.2 kHz", "50 %", "0.4 s").
float parseValue(const ParamSpec& spec, const juce::String& text);
} // namespace alfa::params
