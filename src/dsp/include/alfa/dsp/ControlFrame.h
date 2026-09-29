#pragma once

// The control-frame contract between the network and the DSP.
//
// This header is the single source of truth for what one voice receives every frame. The
// inference host fills VoiceFrame from the expert graphs, the training script's PyTorch
// primitives render the same controls, and the export's golden vectors (training/METHOD.md
// section 7) check that both renders agree. Change anything here only together with METHOD.md,
// and bump kLayoutVersion when you do.
//
// Design (training/METHOD.md section 2): the network never emits audio and never emits
// per-method knobs. Method logic (a subtractive filter curve, a wavetable morph, FM routing)
// is fixed math inside the exported graphs, so the plugin renders only six primitives:
//
//   OSC       harmonic oscillator, a complex spectrum of harmonics 1..1024 per frame
//   PARTIALS  inharmonic sinusoid bank
//   FM        phase-modulation operators with a strictly upper-triangular matrix
//   NOISE     filtered noise, 64 log-spaced band gains
//   GRAIN     grain layers reading a grain-source bank
//   SAMPLER   sample player followed by a 64-band short-time EQ
//
// plus a per-primitive mix gain and a per-voice attack time constant.
//
// Every value is in physical units (linear gains, ratios, radians, seconds), never normalised.
// Arrays are sized for the largest tier; smaller tiers leave the tail zero and the DSP skips it
// (see TierLimits). The struct holds only floats, so it can be treated as a flat float array of
// kFrameFloats values; the offsets below are part of the contract.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace alfa::dsp
{

inline constexpr int kLayoutVersion = 1;

// Timing ------------------------------------------------------------------------------------

/// Frames per second produced by the network, independent of the host sample rate.
inline constexpr double kFrameRateHz = 250.0;

/// Samples per frame (the hop) at a given sample rate: round(sampleRate / 250), so 192 at
/// 48 kHz, 176 at 44.1 kHz and 384 at 96 kHz. Every hop in a voice has this exact length.
[[nodiscard]] constexpr int hopSamples(double sampleRate) noexcept
{
    return static_cast<int>(sampleRate / kFrameRateHz + 0.5);
}

// Sizes -------------------------------------------------------------------------------------

inline constexpr int kOscHarmonics = 1024;   ///< OSC harmonics 1..1024 (index 0 is harmonic 1)
inline constexpr int kOscTableSize = 2048;   ///< single-cycle wave length built from them
inline constexpr int kMaxPartials = 64;
inline constexpr int kMaxFmOperators = 8;
inline constexpr int kBands = 64;            ///< NOISE bands and SAMPLER EQ bands (same axis)
inline constexpr int kMaxGrainLayers = 8;
inline constexpr int kGrainSourcesPerLayer = 2;  ///< top-2 sparse selection over the bank
inline constexpr int kGrainSourceLength = 8192;  ///< samples per grain-source buffer

/// Lowest band centre. Band b (0..63) is centred at 20 * (nyquist / 20)^(b / 63) Hz, so the top
/// band sits at Nyquist. Per-bin gains are linear in log-frequency between centres, flat below
/// band 0 and zero at DC.
inline constexpr double kBandLowHz = 20.0;

enum class Primitive : std::uint8_t
{
    Osc = 0,
    Partials,
    Fm,
    Noise,
    Grain,
    Sampler,
};
inline constexpr int kPrimitiveCount = 6;

// Per-primitive controls ----------------------------------------------------------------------
//
// Interpolation inside a hop. A voice consumes one frame at each of its hop boundaries (the
// first at note-on). Across hop j the controls move from frame j-1 to frame j, reaching frame j
// at the end of the hop; for the first hop, frame -1 is frame 0. "Linear" below means a
// per-sample linear ramp between those two values; "held" means frame j's value is used for
// the whole hop.

/// OSC. The wave for a frame is x(phi) = sum_k re[k] cos(2 pi (k+1) phi) - im[k] sin(2 pi (k+1) phi),
/// phi in cycles, i.e. the real part of sum_k c_k e^{i 2 pi (k+1) phi}. Harmonics whose frequency
/// (k+1) * f0 is at or above Nyquist are zeroed, using f0 at the frame's hop boundary. The wave
/// is stored as a 2048-sample table read with linear interpolation by a phase accumulator at f0,
/// and each hop crossfades linearly from frame j-1's table to frame j's table.
struct OscControls
{
    std::array<float, kOscHarmonics> re{};
    std::array<float, kOscHarmonics> im{};
};

/// PARTIALS. Sinusoid m has frequency f0 * ratio[m] and amplitude amp[m] (both linear per
/// sample). Output is sum_m amp[m] sin(phase_m); phases start at 0 on note-on. A partial at or
/// above Nyquist is silent.
struct PartialsControls
{
    std::array<float, kMaxPartials> ratio{};
    std::array<float, kMaxPartials> amp{};
};

/// FM. Operator i runs at f0 * ratio[i] and outputs y_i = sin(theta_i + sum_{j > i} index[i][j] * y_j),
/// evaluated from the last operator to the first each sample, so there is no feedback and no
/// one-sample delay. index is in radians, stored row-major (index[i * 8 + j]); entries with
/// j <= i are ignored. Voice output is sum_i level[i] * y_i. All controls are linear per sample.
struct FmControls
{
    std::array<float, kMaxFmOperators> ratio{};
    std::array<float, kMaxFmOperators> level{};
    std::array<float, kMaxFmOperators * kMaxFmOperators> index{};
};

/// NOISE. Each hop starts one noise grain of 2 hops: uniform white noise in [-1, 1) shaped by
/// the band curve (linear gain per band, gain 1 everywhere means unfiltered), Hann-windowed and
/// overlap-added at the hop (50% overlap). Band gains are held for the grain.
struct NoiseControls
{
    std::array<float, kBands> band{};
};

/// GRAIN. Each hop, every active layer starts one Hann grain of 2 hops. The grain reads
///   w0 * bank[source[0]] + w1 * bank[source[1]]
/// with linear interpolation, starting at position * 8192 samples and advancing rate samples of
/// the bank per bank sample (negative plays backwards, reads wrap around the buffer), and is
/// scaled by gain. source is a bank index stored as a float and rounded; an index outside the
/// loaded bank reads silence. All values are held for the grain.
struct GrainControls
{
    std::array<float, kMaxGrainLayers * kGrainSourcesPerLayer> source{};  ///< [layer * 2 + s]
    std::array<float, kMaxGrainLayers * kGrainSourcesPerLayer> weight{};  ///< [layer * 2 + s]
    std::array<float, kMaxGrainLayers> position{};  ///< 0..1 of the source buffer
    std::array<float, kMaxGrainLayers> rate{};
    std::array<float, kMaxGrainLayers> gain{};
};

/// SAMPLER. Reads the loaded sample at (f0 / rootHz) * rate, with linear interpolation and the
/// sample's own loop if it has one. start is a fraction 0..1 of the sample length and is read
/// from the note-on frame only. rate is held per hop. The raw read is then equalised with the
/// band layout above: 2-hop Hann frames at 50% overlap, multiplied by the interpolated band
/// curve, overlap-added (band gains held per frame; gain 1 everywhere is transparent). gain is
/// applied after the EQ and is linear per sample.
struct SamplerControls
{
    float rate = 1.0f;
    float start = 0.0f;
    float gain = 0.0f;
    std::array<float, kBands> eq{};
};

// The frame ------------------------------------------------------------------------------------

/// Everything one voice receives per frame. Voice output is
///   attack(t) * sum_p mix[p] * primitive_p
/// where attack(t) = 1 - exp(-t / attackTau), t in seconds since the note-on sample, and
/// attackTau is read from the note-on frame only (clamped to at least 0.1 ms). mix is linear
/// per sample. A primitive whose mix is 0 in both frames of a hop is not rendered.
struct VoiceFrame
{
    std::array<float, kPrimitiveCount> mix{};  ///< indexed by Primitive
    float attackTau = 0.001f;                  ///< seconds
    float reserved = 0.0f;                     ///< keeps the primitive blocks 8-byte aligned

    OscControls osc{};
    PartialsControls partials{};
    FmControls fm{};
    NoiseControls noise{};
    GrainControls grain{};
    SamplerControls sampler{};

    [[nodiscard]] float mixOf(Primitive p) const noexcept { return mix[static_cast<std::size_t>(p)]; }
    void setMix(Primitive p, float g) noexcept { mix[static_cast<std::size_t>(p)] = g; }
};

/// Number of floats in a frame when it is treated as a flat array.
inline constexpr std::size_t kFrameFloats = sizeof(VoiceFrame) / sizeof(float);

// Flat offsets (in floats) -----------------------------------------------------------------------
// The inference host writes expert outputs at these offsets. They are checked against the struct
// by static_asserts in ControlFrame.cpp and exported to layout.json by alfa_dsp_layout.

struct FieldInfo
{
    std::string_view name;   ///< dotted path, e.g. "osc.re"
    std::size_t offset;      ///< in floats from the start of VoiceFrame
    std::size_t count;       ///< floats in the field
    std::string_view unit;
};

namespace offsets
{
inline constexpr std::size_t mix = 0;
inline constexpr std::size_t attackTau = 6;
inline constexpr std::size_t oscRe = 8;
inline constexpr std::size_t oscIm = oscRe + kOscHarmonics;
inline constexpr std::size_t partialsRatio = oscIm + kOscHarmonics;
inline constexpr std::size_t partialsAmp = partialsRatio + kMaxPartials;
inline constexpr std::size_t fmRatio = partialsAmp + kMaxPartials;
inline constexpr std::size_t fmLevel = fmRatio + kMaxFmOperators;
inline constexpr std::size_t fmIndex = fmLevel + kMaxFmOperators;
inline constexpr std::size_t noiseBand = fmIndex + kMaxFmOperators * kMaxFmOperators;
inline constexpr std::size_t grainSource = noiseBand + kBands;
inline constexpr std::size_t grainWeight = grainSource + kMaxGrainLayers * kGrainSourcesPerLayer;
inline constexpr std::size_t grainPosition = grainWeight + kMaxGrainLayers * kGrainSourcesPerLayer;
inline constexpr std::size_t grainRate = grainPosition + kMaxGrainLayers;
inline constexpr std::size_t grainGain = grainRate + kMaxGrainLayers;
inline constexpr std::size_t samplerRate = grainGain + kMaxGrainLayers;
inline constexpr std::size_t samplerStart = samplerRate + 1;
inline constexpr std::size_t samplerGain = samplerStart + 1;
inline constexpr std::size_t samplerEq = samplerGain + 1;
inline constexpr std::size_t end = samplerEq + kBands;
} // namespace offsets

/// Every field of VoiceFrame in layout order.
inline constexpr std::array<FieldInfo, 19> kFields{{
    {"mix", offsets::mix, kPrimitiveCount, "linear gain, [osc, partials, fm, noise, grain, sampler]"},
    {"attack_tau", offsets::attackTau, 1, "seconds, note-on frame only"},
    {"osc.re", offsets::oscRe, kOscHarmonics, "linear, harmonic k+1 cosine part"},
    {"osc.im", offsets::oscIm, kOscHarmonics, "linear, harmonic k+1 imaginary part"},
    {"partials.ratio", offsets::partialsRatio, kMaxPartials, "multiple of f0"},
    {"partials.amp", offsets::partialsAmp, kMaxPartials, "linear"},
    {"fm.ratio", offsets::fmRatio, kMaxFmOperators, "multiple of f0"},
    {"fm.level", offsets::fmLevel, kMaxFmOperators, "linear"},
    {"fm.index", offsets::fmIndex, kMaxFmOperators * kMaxFmOperators, "radians, row-major [i*8+j], j>i only"},
    {"noise.band", offsets::noiseBand, kBands, "linear gain"},
    {"grain.source", offsets::grainSource, kMaxGrainLayers * kGrainSourcesPerLayer, "bank index (rounded), [layer*2+s]"},
    {"grain.weight", offsets::grainWeight, kMaxGrainLayers * kGrainSourcesPerLayer, "linear, [layer*2+s]"},
    {"grain.position", offsets::grainPosition, kMaxGrainLayers, "0..1 of source buffer"},
    {"grain.rate", offsets::grainRate, kMaxGrainLayers, "bank samples per bank sample"},
    {"grain.gain", offsets::grainGain, kMaxGrainLayers, "linear"},
    {"sampler.rate", offsets::samplerRate, 1, "multiple of f0/root"},
    {"sampler.start", offsets::samplerStart, 1, "0..1 of sample length, note-on frame only"},
    {"sampler.gain", offsets::samplerGain, 1, "linear"},
    {"sampler.eq", offsets::samplerEq, kBands, "linear gain"},
}};

// Tiers ----------------------------------------------------------------------------------------

enum class Tier : std::uint8_t
{
    Lite = 0,
    Standard,
    Pro,
    Studio,
};

/// How much of each array a tier's network fills (docs/tiers/model-tiers.md, METHOD.md section 2).
/// OSC is always full: the graph extrapolates its explicit harmonics to 1024.
struct TierLimits
{
    int partials;
    int fmOperators;
    int grainLayers;
    int polyphony;
};

[[nodiscard]] constexpr TierLimits limitsFor(Tier tier) noexcept
{
    switch (tier)
    {
    case Tier::Lite: return {16, 2, 2, 16};
    case Tier::Standard: return {32, 4, 4, 16};
    case Tier::Pro: return {48, 6, 6, 16};
    case Tier::Studio: return {64, 8, 8, 8};
    }
    return {64, 8, 8, 16};
}

} // namespace alfa::dsp
