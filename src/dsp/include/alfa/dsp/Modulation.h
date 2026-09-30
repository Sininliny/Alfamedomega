#pragma once

// The modulation system of ControlFrame.h ("Processing slots and modulation", ModControls).
//
// Per voice a ModBank runs two envelopes, two LFOs, a sample-and-hold, a smoothed random source
// and an envelope follower; ModMatrix then adds depth * source to the destination table
// (slot params, slot out gains, primitive mixes, pitch, pan, dry gain). Everything runs at the
// control sub-block rate (kControlBlock = 32 samples): the caller, once per sub-block,
//
//   bank.update(frame.mod)            // at each hop boundary is enough; sources are held per hop
//   bank.advance(followerInput, n)    // n <= kControlBlock
//   copy base values into dest[kModDestinations]
//   ModMatrix::apply(frame.mod, limits.modRoutes, bank, dest)
//
// value() after advance(n) is the state at the END of those n samples, so a consumer that ramps
// linearly across the sub-block (ParamSmoother, or a processor's own per-block smoothing) lands
// on it exactly when the sub-block ends, the same convention the frame interpolation uses.
//
// Every class here allocates nothing and is noexcept after prepare(). The random sources are
// deterministic for a given seed (setSeed), which the tests and the golden vectors rely on.

#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/FastMath.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace alfa::dsp
{

// Parameter maps ---------------------------------------------------------------------------------

inline constexpr float kModMinSeconds = 1.0e-4f; ///< normalised time 0
inline constexpr float kModMaxSeconds = 10.0f;   ///< normalised time 1
inline constexpr float kModMinHz = 0.01f;        ///< normalised rate 0
inline constexpr float kModMaxHz = 100.0f;       ///< normalised rate 1

/// Normalised 0..1 to seconds, log 0.1 ms .. 10 s. Non-finite input maps to 0.
[[nodiscard]] float modTimeSeconds(float normalised) noexcept;
/// Normalised 0..1 to Hz, log 0.01 .. 100 Hz. Non-finite input maps to 0.
[[nodiscard]] float modRateHz(float normalised) noexcept;

// Envelopes ------------------------------------------------------------------------------------

/// One stage of a SegmentEnvelope.
struct EnvelopeStage
{
    enum class Kind : std::uint8_t
    {
        Ramp,    ///< from the level at stage entry to target over seconds, shaped by curve
        Hold,    ///< keeps the level at stage entry for seconds (delay, hold)
        Sustain, ///< follows target (5 ms slew) until note-off; seconds is ignored
    };

    Kind kind = Kind::Ramp;
    float seconds = 0.0f;
    float target = 0.0f;
    /// Ramp shape: 0 linear; k > 0 moves fast first, like an RC charge or decay,
    /// (1 - e^{-k p}) / (1 - e^{-k}) with p the stage progress 0..1; k < 0 starts slowly.
    /// The ramp always reaches target exactly at the end of the stage, so stage times are exact.
    float curve = 0.0f;
};

/// Generic N-stage envelope. Stages run in order from note-on; a Sustain stage waits for
/// note-off; note-off jumps to releaseStage from the current level (no jump in the output), and
/// note-on restarts at stage 0 from the current level. After the last stage the envelope is idle
/// and holds its level. A stage of 0 seconds completes instantly (its target is reached in the
/// same sample it is entered).
class SegmentEnvelope
{
public:
    static constexpr int kMaxStages = 8;

    void prepare(double sampleRate) noexcept;
    /// Idle at level 0.
    void reset() noexcept;

    /// count stages (clamped to kMaxStages); releaseStage is the stage note-off jumps to
    /// (count or more for none).
    void configure(int count, int releaseStage) noexcept;
    /// Changes a stage in place. If it is the running stage, its progress (as a fraction of the
    /// stage) is kept, so a live time change never jumps the level for a Hold or Ramp stage.
    void setStage(int index, const EnvelopeStage& stage) noexcept;
    [[nodiscard]] const EnvelopeStage& stage(int index) const noexcept;

    void noteOn() noexcept;
    void noteOff() noexcept;

    /// Advances n samples (any n >= 0) and returns the level at the end.
    float advance(int n) noexcept;
    /// Per-sample render: out[i] is the level after i + 1 samples.
    void render(float* out, int n) noexcept;

    [[nodiscard]] float level() const noexcept { return level_; }
    [[nodiscard]] int currentStage() const noexcept { return current_; } ///< count when idle
    [[nodiscard]] bool idle() const noexcept { return current_ >= count_; }
    [[nodiscard]] bool released() const noexcept { return current_ >= releaseStage_; }

private:
    void enter(int index) noexcept;
    void recomputeLength() noexcept;
    [[nodiscard]] float rampLevel(double progress) const noexcept;

    std::array<EnvelopeStage, kMaxStages> stages_{};
    int count_ = 0;
    int releaseStage_ = 0;
    double sampleRate_ = 48000.0;
    double sustainCoeffPerSample_ = 0.0; ///< -1 / (tau * sr) for the sustain slew

    int current_ = 0;       ///< running stage, count_ when idle
    double length_ = 0.0;   ///< running stage length in samples
    double position_ = 0.0; ///< samples into the running stage
    float start_ = 0.0f;    ///< level at stage entry
    float level_ = 0.0f;
};

/// DAHDSR (delay, attack, hold, decay, sustain, release) on top of SegmentEnvelope. Attack is
/// slightly convex (curve 1.5), decay and release are exponential (curve 5); release runs the
/// full release time from wherever the level is.
class DahdsrEnvelope
{
public:
    static constexpr float kAttackCurve = 1.5f;
    static constexpr float kDecayCurve = 5.0f;

    void prepare(double sampleRate) noexcept;
    void reset() noexcept { env_.reset(); }

    /// Physical values: seconds (>= 0) and sustain level (clamped 0..1).
    void setTimes(float delay, float attack, float hold, float decay, float sustain, float release) noexcept;
    /// Six normalised values in ModControls order: times log 0.1 ms .. 10 s, sustain linear.
    void setNormalised(const float* six) noexcept;

    void noteOn() noexcept { env_.noteOn(); }
    void noteOff() noexcept { env_.noteOff(); }
    float advance(int n) noexcept { return env_.advance(n); }
    void render(float* out, int n) noexcept { env_.render(out, n); }

    [[nodiscard]] float level() const noexcept { return env_.level(); }
    [[nodiscard]] bool idle() const noexcept { return env_.idle(); }
    [[nodiscard]] const SegmentEnvelope& engine() const noexcept { return env_; }

    enum StageIndex : int
    {
        Delay = 0,
        Attack,
        Hold,
        Decay,
        Sustain,
        Release,
        StageCount
    };

private:
    SegmentEnvelope env_;
};

// LFO and random sources -----------------------------------------------------------------------

/// Low-frequency oscillator, output -1..1. Shape morphs continuously over
/// sine (0) > triangle (0.25) > saw (0.5) > square (0.75) > sample-and-hold (1); all shapes are
/// aligned so phase 0 is a rising zero crossing (S&H picks a new value at each phase wrap).
/// The square, saw and S&H steps are replaced by linear ramps of kEdgeSeconds (at most a quarter
/// cycle), so a control-rate reader never sees an instantaneous step.
class Lfo
{
public:
    static constexpr float kEdgeSeconds = 0.002f;
    static constexpr float kShapeSmoothSeconds = 0.005f;

    void prepare(double sampleRate) noexcept;
    /// Phase to the start phase, shape snapped to its target, generator reseeded.
    void reset() noexcept;
    void setSeed(std::uint32_t seed) noexcept;

    void setRateHz(float hz) noexcept;         ///< clamped 0 .. 1000
    void setShape(float shape) noexcept;       ///< 0..1, smoothed
    void setStartPhase(float cycles) noexcept; ///< 0..1
    void setRetrigger(bool on) noexcept { retrigger_ = on; }
    /// ModControls order: rate (log 0.01..100 Hz), shape, start phase, retrigger (>= 0.5).
    void setNormalised(const float* four) noexcept;

    /// Restarts at the start phase when retrigger is on; otherwise free-running.
    void noteOn() noexcept;

    float advance(int n) noexcept;           ///< returns the value after n samples
    void render(float* out, int n) noexcept; ///< per sample, out[i] after i + 1 samples

    [[nodiscard]] float value() const noexcept { return value_; }
    [[nodiscard]] double phase() const noexcept { return phase_; }
    [[nodiscard]] float rateHz() const noexcept { return rateHz_; }

    /// The waveform itself, for tests and offline renders: shape 0..1, phase in cycles,
    /// edge = ramp width in cycles (0 for ideal edges), S&H from `held` to `next`.
    [[nodiscard]] static float shapeAt(float shape, double phase, float edge, float held,
                                       float next) noexcept;

private:
    void refreshValue() noexcept;

    double sampleRate_ = 48000.0;
    double phase_ = 0.0;
    double increment_ = 0.0; ///< cycles per sample
    float rateHz_ = 1.0f;
    float startPhase_ = 0.0f;
    bool retrigger_ = false;
    float shapeTarget_ = 0.0f;
    float shape_ = 0.0f;
    float shapePole_ = 0.0f; ///< per-sample one-pole coefficient of the shape smoothing
    float shHeld_ = 0.0f;    ///< S&H value before the last wrap (ramp start)
    float shNext_ = 0.0f;    ///< S&H value after the last wrap
    float value_ = 0.0f;
    std::uint32_t seed_ = 0x51u;
    detail::Random rng_;
};

/// Random value in [-1, 1) picked at `rate` Hz and held until the next tick (the step itself is
/// a Lfo::kEdgeSeconds ramp, at most a quarter period, for the same reason as the LFO edges).
class SampleHold
{
public:
    void prepare(double sampleRate) noexcept;
    void reset() noexcept; ///< reseeds, phase 0, draws the first value
    void setSeed(std::uint32_t seed) noexcept;
    void setRateHz(float hz) noexcept; ///< clamped 0 .. 1000
    float advance(int n) noexcept;
    void render(float* out, int n) noexcept;
    [[nodiscard]] float value() const noexcept { return value_; }
    [[nodiscard]] double phase() const noexcept { return phase_; }

private:
    void refresh() noexcept;

    double sampleRate_ = 48000.0;
    double phase_ = 0.0;
    double increment_ = 0.0;
    float rateHz_ = 1.0f;
    float held_ = 0.0f;
    float next_ = 0.0f;
    float value_ = 0.0f;
    std::uint32_t seed_ = 0x53u;
    detail::Random rng_;
};

/// Random targets in [-1, 1) at `rate` Hz joined by smoothstep (cubic, zero slope at each
/// target), so the output is continuous with a continuous slope and never leaves [-1, 1].
class SmoothRandom
{
public:
    void prepare(double sampleRate) noexcept;
    void reset() noexcept; ///< reseeds, phase 0, draws the first two targets
    void setSeed(std::uint32_t seed) noexcept;
    void setRateHz(float hz) noexcept; ///< clamped 0 .. 1000
    float advance(int n) noexcept;
    void render(float* out, int n) noexcept;
    [[nodiscard]] float value() const noexcept { return value_; }

private:
    void refresh() noexcept;

    double sampleRate_ = 48000.0;
    double phase_ = 0.0;
    double increment_ = 0.0;
    float rateHz_ = 1.0f;
    float from_ = 0.0f;
    float to_ = 0.0f;
    float value_ = 0.0f;
    std::uint32_t seed_ = 0x55u;
    detail::Random rng_;
};

/// Peak or RMS envelope follower with separate attack and release time constants (a step
/// reaches 1 - 1/e of its height after one time constant). value() is clamped to 0..1.
class EnvelopeFollower
{
public:
    enum class Mode : std::uint8_t
    {
        Peak,
        Rms
    };

    void prepare(double sampleRate) noexcept;
    void reset() noexcept;
    void setMode(Mode mode) noexcept { mode_ = mode; }
    void setTimes(float attackSeconds, float releaseSeconds) noexcept;
    /// Feeds n input samples (nullptr is silence) and returns value().
    float process(const float* input, int n) noexcept;
    [[nodiscard]] float value() const noexcept;
    [[nodiscard]] float raw() const noexcept { return state_; } ///< unclamped peak / power

private:
    double sampleRate_ = 48000.0;
    Mode mode_ = Mode::Peak;
    float attackSeconds_ = -1.0f;
    float releaseSeconds_ = -1.0f;
    float attackCoeff_ = 0.0f; ///< per-sample pole
    float releaseCoeff_ = 0.0f;
    float state_ = 0.0f; ///< amplitude (Peak) or power (Rms)
};

// Bank and matrix ----------------------------------------------------------------------------

enum class ModScope : std::uint8_t
{
    Voice,    ///< every source
    Instance, ///< only the LFOs and random sources run; the others read 0
};

/// All modulation sources of one voice (or of the instance, with ModScope::Instance).
class ModBank
{
public:
    explicit ModBank(ModScope scope = ModScope::Voice) noexcept
        : scope_(scope)
    {
    }

    void prepare(double sampleRate) noexcept;
    /// Everything back to its initial state, random generators reseeded from the current seed.
    void reset() noexcept;
    /// Seeds the four random generators (LFO S&H x2, sample-and-hold, smoothed random) with
    /// distinct streams derived from seed, and resets them. Same seed, same inputs: same output.
    void setSeed(std::uint32_t seed) noexcept;

    /// velocity 0..1, note 0..127 (both clamped). Retriggers the envelopes from their current
    /// level and the LFOs that have retrigger on. Ignored by an instance bank.
    void noteOn(float velocity, int note) noexcept;
    void noteOff() noexcept;

    /// Reads the 24 source params (normalised; non-finite reads as 0). Call once per hop.
    void update(const ModControls& controls) noexcept;
    /// Advances n (0 .. kControlBlock) samples. followerInput is the audio the follower tracks
    /// (the voice's primitive sum, n samples; nullptr is silence; ignored by an instance bank).
    void advance(const float* followerInput, int n) noexcept;

    /// Envelopes, follower, velocity and key 0..1; LFOs and random -1..1; None 0.
    [[nodiscard]] float value(ModSource source) const noexcept
    {
        const auto i = static_cast<std::size_t>(source);
        return i < values_.size() ? values_[i] : 0.0f;
    }

    [[nodiscard]] ModScope scope() const noexcept { return scope_; }
    [[nodiscard]] const DahdsrEnvelope& env1() const noexcept { return env1_; }
    [[nodiscard]] const DahdsrEnvelope& env2() const noexcept { return env2_; }
    [[nodiscard]] EnvelopeFollower& follower() noexcept { return follower_; }

private:
    void refreshValues() noexcept;

    ModScope scope_;
    std::uint32_t seed_ = 0x9E3779B9u;
    DahdsrEnvelope env1_;
    DahdsrEnvelope env2_;
    Lfo lfo1_;
    Lfo lfo2_;
    SampleHold sampleHold_;
    SmoothRandom smoothRandom_;
    EnvelopeFollower follower_;
    float velocity_ = 0.0f;
    float key_ = 0.0f;
    std::array<float, kModSourceCount> values_{};
};

/// Applies the routes of ModControls to a destination table (ControlFrame.h rules).
struct ModMatrix
{
    /// destinations holds kModDestinations base values filled by the caller. For each of the
    /// first routeLimit routes (clamped 0..kMaxModRoutes) in order, source and destination are
    /// rounded, depth is clamped to -1..1, and depth * source is added (pitch: 24 semitones per
    /// unit). A route with a non-finite field, an out-of-range source or destination, source
    /// None, or a destination the bank's scope does not have (instance: only 0..67 and 76) is
    /// ignored. Afterwards every destination is sanitised (non-finite becomes 0) and clamped:
    /// slot params 0..1, gains >= 0, pan -1..1, pitch +-kMaxPitchSemitones.
    static constexpr float kPitchSemitonesPerUnit = 24.0f;
    static constexpr float kMaxPitchSemitones = 96.0f;

    static void apply(const ModControls& controls, int routeLimit, const ModBank& bank,
                      float* destinations) noexcept;

    /// The clamping alone, for base tables with no routes.
    static void clampDestinations(float* destinations, ModScope scope = ModScope::Voice) noexcept;
};

// Smoothing ------------------------------------------------------------------------------------

/// Linear ramp for control-rate values (the slot graph's slot params and gains): setTarget
/// starts a ramp of `samples` samples from the current value, next() steps it. Removes the
/// zipper noise of values that change once per control block.
class ParamSmoother
{
public:
    /// Jumps to v with no ramp (non-finite reads as 0).
    void reset(float v = 0.0f) noexcept;
    /// Ramps from the current value to target over samples (<= 0 jumps). Non-finite targets are
    /// ignored.
    void setTarget(float target, int samples) noexcept;
    float next() noexcept;
    /// Advances n samples and returns the value.
    float skip(int n) noexcept;
    /// Writes the next n values.
    void fill(float* out, int n) noexcept;

    [[nodiscard]] float current() const noexcept { return current_; }
    [[nodiscard]] float target() const noexcept { return target_; }
    [[nodiscard]] bool smoothing() const noexcept { return remaining_ > 0; }

private:
    float current_ = 0.0f;
    float target_ = 0.0f;
    float step_ = 0.0f;
    int remaining_ = 0;
};

} // namespace alfa::dsp
