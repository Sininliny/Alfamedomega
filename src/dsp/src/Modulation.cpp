#include "alfa/dsp/Modulation.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp
{

namespace
{

constexpr float kSustainSlewSeconds = 0.005f;
constexpr float kMaxRateHz = 1000.0f;
constexpr float kMaxStageSeconds = 1000.0f;

[[nodiscard]] float finiteOr(float x, float fallback) noexcept
{
    return std::isfinite(x) ? x : fallback;
}

[[nodiscard]] float clamp01(float x) noexcept
{
    return std::isfinite(x) ? std::clamp(x, 0.0f, 1.0f) : 0.0f;
}

[[nodiscard]] float sanitiseRate(float hz) noexcept
{
    return std::isfinite(hz) ? std::clamp(hz, 0.0f, kMaxRateHz) : 0.0f;
}

/// Bijective 32-bit mixer (murmur3 finaliser), so every derived seed differs.
[[nodiscard]] std::uint32_t mixSeed(std::uint32_t seed, std::uint32_t stream) noexcept
{
    std::uint32_t x = seed + stream * 0x9E3779B9u;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

/// Phase ramp width (cycles) that turns a step into kEdgeSeconds of slope, capped at a quarter
/// cycle so fast LFOs keep their character.
[[nodiscard]] float edgeWidth(float rateHz) noexcept
{
    return std::min(0.25f, rateHz * Lfo::kEdgeSeconds);
}

[[nodiscard]] float frac(double x) noexcept
{
    return static_cast<float>(x - std::floor(x));
}

/// Accumulates n samples of phase, returning the number of wraps (capped: past a few wraps per
/// call nobody can hear which random values were skipped, and the cap bounds the work).
int advancePhase(double& phase, double increment, int n) noexcept
{
    phase += increment * static_cast<double>(n);
    if (phase < 1.0)
        return 0;
    const double wraps = std::floor(phase);
    phase -= wraps;
    return static_cast<int>(std::min(wraps, 4.0));
}

} // namespace

float modTimeSeconds(float normalised) noexcept
{
    if (!std::isfinite(normalised))
        return 0.0f;
    const float x = std::clamp(normalised, 0.0f, 1.0f);
    return kModMinSeconds * std::pow(kModMaxSeconds / kModMinSeconds, x);
}

float modRateHz(float normalised) noexcept
{
    if (!std::isfinite(normalised))
        return 0.0f;
    const float x = std::clamp(normalised, 0.0f, 1.0f);
    return kModMinHz * std::pow(kModMaxHz / kModMinHz, x);
}

// SegmentEnvelope --------------------------------------------------------------------------------

void SegmentEnvelope::prepare(double sampleRate) noexcept
{
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
    sustainCoeffPerSample_ = -1.0 / (static_cast<double>(kSustainSlewSeconds) * sampleRate_);
    reset();
}

void SegmentEnvelope::reset() noexcept
{
    current_ = count_;
    length_ = 0.0;
    position_ = 0.0;
    start_ = 0.0f;
    level_ = 0.0f;
}

void SegmentEnvelope::configure(int count, int releaseStage) noexcept
{
    count_ = std::clamp(count, 0, kMaxStages);
    releaseStage_ = std::clamp(releaseStage, 0, count_);
    current_ = std::min(current_, count_);
    if (!idle())
        recomputeLength();
}

void SegmentEnvelope::setStage(int index, const EnvelopeStage& stage) noexcept
{
    if (index < 0 || index >= kMaxStages)
        return;
    EnvelopeStage s = stage;
    s.seconds = std::clamp(finiteOr(s.seconds, 0.0f), 0.0f, kMaxStageSeconds);
    s.target = finiteOr(s.target, 0.0f);
    s.curve = std::clamp(finiteOr(s.curve, 0.0f), -30.0f, 30.0f);
    stages_[static_cast<std::size_t>(index)] = s;
    if (index == current_)
    {
        // Keep the fraction of the stage already done, so a new time bends the rest of the
        // ramp instead of jumping.
        const double fraction = length_ > 0.0 ? position_ / length_ : 0.0;
        recomputeLength();
        position_ = fraction * length_;
    }
}

const EnvelopeStage& SegmentEnvelope::stage(int index) const noexcept
{
    return stages_[static_cast<std::size_t>(std::clamp(index, 0, kMaxStages - 1))];
}

void SegmentEnvelope::recomputeLength() noexcept
{
    const auto& s = stages_[static_cast<std::size_t>(current_)];
    // Whole samples, so a stage of T seconds ends exactly round(T * sr) samples after it starts.
    length_ = std::round(static_cast<double>(s.seconds) * sampleRate_);
}

void SegmentEnvelope::enter(int index) noexcept
{
    current_ = std::min(index, count_);
    start_ = level_;
    position_ = 0.0;
    length_ = 0.0;
    if (!idle())
        recomputeLength();
}

void SegmentEnvelope::noteOn() noexcept
{
    enter(0);
}

void SegmentEnvelope::noteOff() noexcept
{
    if (current_ < releaseStage_ && releaseStage_ < count_)
        enter(releaseStage_);
}

float SegmentEnvelope::rampLevel(double progress) const noexcept
{
    const auto& s = stages_[static_cast<std::size_t>(current_)];
    const auto p = static_cast<float>(std::clamp(progress, 0.0, 1.0));
    float shape = p;
    if (std::fabs(s.curve) > 1.0e-4f)
        shape = std::expm1(-s.curve * p) / std::expm1(-s.curve);
    return start_ + (s.target - start_) * shape;
}

float SegmentEnvelope::advance(int n) noexcept
{
    while (current_ < count_)
    {
        const auto& s = stages_[static_cast<std::size_t>(current_)];
        if (s.kind == EnvelopeStage::Kind::Sustain)
        {
            if (n > 0)
            {
                const auto decay =
                    static_cast<float>(std::exp(sustainCoeffPerSample_ * static_cast<double>(n)));
                level_ = s.target + (level_ - s.target) * decay;
                if (std::fabs(level_ - s.target) < 1.0e-7f)
                    level_ = s.target;
            }
            break;
        }
        const double left = length_ - position_;
        if (left <= 1.0e-9)
        {
            level_ = s.kind == EnvelopeStage::Kind::Hold ? start_ : s.target;
            enter(current_ + 1);
            continue;
        }
        if (n <= 0)
            break;
        const auto available = static_cast<int>(std::min(std::ceil(left - 1.0e-9), 2.0e9));
        const int take = std::min(n, available);
        n -= take;
        if (take == available)
        {
            position_ = length_; // completed at the top of the next iteration
            continue;
        }
        position_ += static_cast<double>(take);
        level_ = s.kind == EnvelopeStage::Kind::Hold ? start_ : rampLevel(position_ / length_);
    }
    if (!std::isfinite(level_))
    {
        reset();
    }
    return level_;
}

void SegmentEnvelope::render(float* out, int n) noexcept
{
    for (int i = 0; i < n; ++i)
        out[i] = advance(1);
}

// DahdsrEnvelope ---------------------------------------------------------------------------------

void DahdsrEnvelope::prepare(double sampleRate) noexcept
{
    env_.prepare(sampleRate);
    env_.configure(StageCount, Release);
    setTimes(0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    env_.reset();
}

void DahdsrEnvelope::setTimes(float delay, float attack, float hold, float decay, float sustain,
                              float release) noexcept
{
    using Kind = EnvelopeStage::Kind;
    const float level = clamp01(sustain);
    env_.setStage(Delay, {Kind::Hold, delay, 0.0f, 0.0f});
    env_.setStage(Attack, {Kind::Ramp, attack, 1.0f, kAttackCurve});
    env_.setStage(Hold, {Kind::Hold, hold, 0.0f, 0.0f});
    env_.setStage(Decay, {Kind::Ramp, decay, level, kDecayCurve});
    env_.setStage(Sustain, {Kind::Sustain, 0.0f, level, 0.0f});
    env_.setStage(Release, {Kind::Ramp, release, 0.0f, kDecayCurve});
}

void DahdsrEnvelope::setNormalised(const float* six) noexcept
{
    setTimes(modTimeSeconds(six[0]), modTimeSeconds(six[1]), modTimeSeconds(six[2]), modTimeSeconds(six[3]),
             clamp01(six[4]), modTimeSeconds(six[5]));
}

// Lfo ------------------------------------------------------------------------------------------

void Lfo::prepare(double sampleRate) noexcept
{
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
    shapePole_ =
        static_cast<float>(std::exp(-1.0 / (static_cast<double>(kShapeSmoothSeconds) * sampleRate_)));
    setRateHz(rateHz_);
    reset();
}

void Lfo::reset() noexcept
{
    phase_ = startPhase_;
    shape_ = shapeTarget_;
    setSeed(seed_);
}

void Lfo::setSeed(std::uint32_t seed) noexcept
{
    seed_ = seed;
    rng_.setSeed(seed);
    shHeld_ = rng_.nextBipolar();
    shNext_ = shHeld_;
    refreshValue();
}

void Lfo::setRateHz(float hz) noexcept
{
    rateHz_ = sanitiseRate(hz);
    increment_ = static_cast<double>(rateHz_) / sampleRate_;
}

void Lfo::setShape(float shape) noexcept
{
    shapeTarget_ = clamp01(shape);
}

void Lfo::setStartPhase(float cycles) noexcept
{
    startPhase_ = clamp01(cycles);
    if (startPhase_ >= 1.0f)
        startPhase_ = 0.0f;
}

void Lfo::setNormalised(const float* four) noexcept
{
    setRateHz(modRateHz(four[0]));
    setShape(four[1]);
    setStartPhase(four[2]);
    setRetrigger(clamp01(four[3]) >= 0.5f);
}

void Lfo::noteOn() noexcept
{
    if (!retrigger_)
        return;
    phase_ = startPhase_;
    shHeld_ = rng_.nextBipolar();
    shNext_ = shHeld_;
    refreshValue();
}

float Lfo::shapeAt(float shape, double phase, float edge, float held, float next) noexcept
{
    const float p = frac(phase);
    const float e = std::clamp(edge, 0.0f, 0.5f);
    const auto basic = [&](int k) noexcept -> float
    {
        switch (k)
        {
        case 0:
            return detail::sin2pi(p);
        case 1:
            return 4.0f * std::fabs(frac(static_cast<double>(p) + 0.75) - 0.5f) - 1.0f;
        case 2:
        {
            // Rising ramp through 0 at phase 0; the drop at phase 0.5 is a slope of width e
            // centred on it.
            float q = frac(static_cast<double>(p) + 0.5);
            if (e <= 0.0f)
                return 2.0f * q - 1.0f;
            const float half = 0.5f * e;
            if (q >= half && q < 1.0f - half)
                return -1.0f + 2.0f * (q - half) / (1.0f - e);
            if (q < half)
                q += 1.0f;
            return 1.0f - 2.0f * (q - (1.0f - half)) / e;
        }
        case 3:
        {
            if (e <= 0.0f)
                return p < 0.5f ? 1.0f : -1.0f;
            // The triangle crosses zero at the square's edges with slope 4 per cycle, so
            // scaling it by 1 / (2e) and clipping gives edges exactly e wide.
            const float tri = 4.0f * std::fabs(frac(static_cast<double>(p) + 0.75) - 0.5f) - 1.0f;
            return std::clamp(tri / (2.0f * e), -1.0f, 1.0f);
        }
        default:
        {
            const float t = e > 0.0f ? std::min(p / e, 1.0f) : 1.0f;
            return held + (next - held) * t;
        }
        }
    };
    const float s4 = clamp01(shape) * 4.0f;
    const int i = std::min(3, static_cast<int>(s4));
    const float f = s4 - static_cast<float>(i);
    const float a = basic(i);
    if (f <= 0.0f)
        return a;
    return a + (basic(i + 1) - a) * f;
}

void Lfo::refreshValue() noexcept
{
    value_ = shapeAt(shape_, phase_, edgeWidth(rateHz_), shHeld_, shNext_);
}

float Lfo::advance(int n) noexcept
{
    if (n <= 0)
        return value_;
    if (shape_ != shapeTarget_)
    {
        const float pole = n == 1 ? shapePole_ : std::pow(shapePole_, static_cast<float>(n));
        shape_ = shapeTarget_ + (shape_ - shapeTarget_) * pole;
        if (std::fabs(shape_ - shapeTarget_) < 1.0e-6f)
            shape_ = shapeTarget_;
    }
    for (int w = advancePhase(phase_, increment_, n); w > 0; --w)
    {
        shHeld_ = shNext_;
        shNext_ = rng_.nextBipolar();
    }
    refreshValue();
    return value_;
}

void Lfo::render(float* out, int n) noexcept
{
    for (int i = 0; i < n; ++i)
        out[i] = advance(1);
}

// SampleHold ----------------------------------------------------------------------------------

void SampleHold::prepare(double sampleRate) noexcept
{
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
    setRateHz(rateHz_);
    reset();
}

void SampleHold::reset() noexcept
{
    phase_ = 0.0;
    setSeed(seed_);
}

void SampleHold::setSeed(std::uint32_t seed) noexcept
{
    seed_ = seed;
    rng_.setSeed(seed);
    held_ = rng_.nextBipolar();
    next_ = held_;
    refresh();
}

void SampleHold::setRateHz(float hz) noexcept
{
    rateHz_ = sanitiseRate(hz);
    increment_ = static_cast<double>(rateHz_) / sampleRate_;
}

void SampleHold::refresh() noexcept
{
    const float e = edgeWidth(rateHz_);
    const float t = e > 0.0f ? std::min(static_cast<float>(phase_) / e, 1.0f) : 1.0f;
    value_ = held_ + (next_ - held_) * t;
}

float SampleHold::advance(int n) noexcept
{
    if (n <= 0)
        return value_;
    for (int w = advancePhase(phase_, increment_, n); w > 0; --w)
    {
        held_ = next_;
        next_ = rng_.nextBipolar();
    }
    refresh();
    return value_;
}

void SampleHold::render(float* out, int n) noexcept
{
    for (int i = 0; i < n; ++i)
        out[i] = advance(1);
}

// SmoothRandom --------------------------------------------------------------------------------

void SmoothRandom::prepare(double sampleRate) noexcept
{
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
    setRateHz(rateHz_);
    reset();
}

void SmoothRandom::reset() noexcept
{
    phase_ = 0.0;
    setSeed(seed_);
}

void SmoothRandom::setSeed(std::uint32_t seed) noexcept
{
    seed_ = seed;
    rng_.setSeed(seed);
    from_ = rng_.nextBipolar();
    to_ = rng_.nextBipolar();
    refresh();
}

void SmoothRandom::setRateHz(float hz) noexcept
{
    rateHz_ = sanitiseRate(hz);
    increment_ = static_cast<double>(rateHz_) / sampleRate_;
}

void SmoothRandom::refresh() noexcept
{
    const auto p = static_cast<float>(phase_);
    const float s = p * p * (3.0f - 2.0f * p);
    value_ = from_ + (to_ - from_) * s;
}

float SmoothRandom::advance(int n) noexcept
{
    if (n <= 0)
        return value_;
    for (int w = advancePhase(phase_, increment_, n); w > 0; --w)
    {
        from_ = to_;
        to_ = rng_.nextBipolar();
    }
    refresh();
    return value_;
}

void SmoothRandom::render(float* out, int n) noexcept
{
    for (int i = 0; i < n; ++i)
        out[i] = advance(1);
}

// EnvelopeFollower ----------------------------------------------------------------------------

void EnvelopeFollower::prepare(double sampleRate) noexcept
{
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
    // Force the coefficients to be recomputed for the new rate.
    const float a = attackSeconds_ < 0.0f ? 0.01f : attackSeconds_;
    const float r = releaseSeconds_ < 0.0f ? 0.1f : releaseSeconds_;
    attackSeconds_ = -1.0f;
    releaseSeconds_ = -1.0f;
    setTimes(a, r);
    reset();
}

void EnvelopeFollower::reset() noexcept
{
    state_ = 0.0f;
}

void EnvelopeFollower::setTimes(float attackSeconds, float releaseSeconds) noexcept
{
    const auto pole = [this](float seconds) noexcept
    {
        return seconds > 0.0f
                   ? static_cast<float>(std::exp(-1.0 / (static_cast<double>(seconds) * sampleRate_)))
                   : 0.0f;
    };
    const float a = std::max(0.0f, finiteOr(attackSeconds, 0.0f));
    const float r = std::max(0.0f, finiteOr(releaseSeconds, 0.0f));
    if (a != attackSeconds_)
    {
        attackSeconds_ = a;
        attackCoeff_ = pole(a);
    }
    if (r != releaseSeconds_)
    {
        releaseSeconds_ = r;
        releaseCoeff_ = pole(r);
    }
}

float EnvelopeFollower::process(const float* input, int n) noexcept
{
    if (input == nullptr)
    {
        // Silence only releases: closed form of n release steps.
        if (n > 0 && state_ > 0.0f)
            state_ *= std::pow(releaseCoeff_, static_cast<float>(n));
    }
    else
    {
        const bool rms = mode_ == Mode::Rms;
        float s = state_;
        for (int i = 0; i < n; ++i)
        {
            const float x = input[i];
            const float v = rms ? x * x : std::fabs(x);
            if (!std::isfinite(v))
            {
                s = 0.0f;
                continue;
            }
            const float c = v > s ? attackCoeff_ : releaseCoeff_;
            s = v + (s - v) * c;
        }
        state_ = s;
    }
    if (!(state_ >= 1.0e-20f) || !std::isfinite(state_))
        state_ = 0.0f; // flushes denormals and anything odd
    return value();
}

float EnvelopeFollower::value() const noexcept
{
    const float v = mode_ == Mode::Rms ? std::sqrt(state_) : state_;
    return std::min(v, 1.0f);
}

// ModBank -------------------------------------------------------------------------------------

void ModBank::prepare(double sampleRate) noexcept
{
    env1_.prepare(sampleRate);
    env2_.prepare(sampleRate);
    lfo1_.prepare(sampleRate);
    lfo2_.prepare(sampleRate);
    sampleHold_.prepare(sampleRate);
    smoothRandom_.prepare(sampleRate);
    follower_.prepare(sampleRate);
    setSeed(seed_);
    reset();
}

void ModBank::reset() noexcept
{
    env1_.reset();
    env2_.reset();
    lfo1_.reset();
    lfo2_.reset();
    sampleHold_.reset();
    smoothRandom_.reset();
    follower_.reset();
    velocity_ = 0.0f;
    key_ = 0.0f;
    refreshValues();
}

void ModBank::setSeed(std::uint32_t seed) noexcept
{
    seed_ = seed;
    lfo1_.setSeed(mixSeed(seed, 1));
    lfo2_.setSeed(mixSeed(seed, 2));
    sampleHold_.setSeed(mixSeed(seed, 3));
    smoothRandom_.setSeed(mixSeed(seed, 4));
    refreshValues();
}

void ModBank::noteOn(float velocity, int note) noexcept
{
    if (scope_ == ModScope::Instance)
        return;
    velocity_ = clamp01(velocity);
    key_ = static_cast<float>(std::clamp(note, 0, 127)) / 127.0f;
    env1_.noteOn();
    env2_.noteOn();
    lfo1_.noteOn();
    lfo2_.noteOn();
    refreshValues();
}

void ModBank::noteOff() noexcept
{
    if (scope_ == ModScope::Instance)
        return;
    env1_.noteOff();
    env2_.noteOff();
}

void ModBank::update(const ModControls& controls) noexcept
{
    std::array<float, kModSourceParams> s{};
    for (std::size_t i = 0; i < s.size(); ++i)
        s[i] = clamp01(controls.sources[i]);
    if (scope_ == ModScope::Voice)
    {
        env1_.setNormalised(s.data());
        env2_.setNormalised(s.data() + 6);
        follower_.setTimes(modTimeSeconds(s[22]), modTimeSeconds(s[23]));
    }
    lfo1_.setNormalised(s.data() + 12);
    lfo2_.setNormalised(s.data() + 16);
    sampleHold_.setRateHz(modRateHz(s[20]));
    smoothRandom_.setRateHz(modRateHz(s[21]));
}

void ModBank::advance(const float* followerInput, int n) noexcept
{
    if (n <= 0)
        return;
    if (scope_ == ModScope::Voice)
    {
        env1_.advance(n);
        env2_.advance(n);
        follower_.process(followerInput, n);
    }
    lfo1_.advance(n);
    lfo2_.advance(n);
    sampleHold_.advance(n);
    smoothRandom_.advance(n);
    refreshValues();
}

void ModBank::refreshValues() noexcept
{
    const auto set = [this](ModSource s, float v) noexcept
    {
        values_[static_cast<std::size_t>(s)] = std::isfinite(v) ? v : 0.0f;
    };
    const bool voice = scope_ == ModScope::Voice;
    set(ModSource::None, 0.0f);
    set(ModSource::Env1, voice ? std::clamp(env1_.level(), 0.0f, 1.0f) : 0.0f);
    set(ModSource::Env2, voice ? std::clamp(env2_.level(), 0.0f, 1.0f) : 0.0f);
    set(ModSource::Lfo1, lfo1_.value());
    set(ModSource::Lfo2, lfo2_.value());
    set(ModSource::SampleHold, sampleHold_.value());
    set(ModSource::SmoothRandom, smoothRandom_.value());
    set(ModSource::Follower, voice ? follower_.value() : 0.0f);
    set(ModSource::Velocity, voice ? velocity_ : 0.0f);
    set(ModSource::Key, voice ? key_ : 0.0f);
}

// ModMatrix -----------------------------------------------------------------------------------

void ModMatrix::apply(const ModControls& controls, int routeLimit, const ModBank& bank,
                      float* destinations) noexcept
{
    const bool instance = bank.scope() == ModScope::Instance;
    const int limit = std::clamp(routeLimit, 0, kMaxModRoutes);
    for (int r = 0; r < limit; ++r)
    {
        const auto i = static_cast<std::size_t>(r);
        const float fs = controls.routeSource[i];
        const float fd = controls.routeDestination[i];
        const float depth = controls.routeDepth[i];
        // The comparisons are written so NaN fails them; Inf fails the range checks.
        if (!(fs >= 0.5f) || fs >= static_cast<float>(kModSourceCount) - 0.5f)
            continue; // None, out of range or non-finite
        if (!(fd >= -0.5f) || fd >= static_cast<float>(kModDestinations) - 0.5f)
            continue;
        if (!std::isfinite(depth))
            continue;
        const auto source = static_cast<ModSource>(static_cast<int>(fs + 0.5f));
        const int dest = static_cast<int>(fd + 0.5f);
        if (instance && dest >= moddest::mix && dest != moddest::dryGain)
            continue;
        float amount = std::clamp(depth, -1.0f, 1.0f) * bank.value(source);
        if (dest == moddest::pitch)
            amount *= kPitchSemitonesPerUnit;
        destinations[dest] += amount;
    }
    clampDestinations(destinations, bank.scope());
}

void ModMatrix::clampDestinations(float* destinations, ModScope /*scope*/) noexcept
{
    for (int d = 0; d < kModDestinations; ++d)
    {
        float v = finiteOr(destinations[d], 0.0f);
        if (d < moddest::slotOutGain)
            v = std::clamp(v, 0.0f, 1.0f);
        else if (d == moddest::pitch)
            v = std::clamp(v, -kMaxPitchSemitones, kMaxPitchSemitones);
        else if (d == moddest::pan)
            v = std::clamp(v, -1.0f, 1.0f);
        else
            v = std::max(v, 0.0f); // slot out gains, mix gains, dry gain
        destinations[d] = v;
    }
}

// ParamSmoother -------------------------------------------------------------------------------

void ParamSmoother::reset(float v) noexcept
{
    current_ = finiteOr(v, 0.0f);
    target_ = current_;
    step_ = 0.0f;
    remaining_ = 0;
}

void ParamSmoother::setTarget(float target, int samples) noexcept
{
    if (!std::isfinite(target))
        return;
    target_ = target;
    if (samples <= 0 || target == current_)
    {
        current_ = target;
        remaining_ = 0;
        step_ = 0.0f;
        return;
    }
    remaining_ = samples;
    step_ = (target_ - current_) / static_cast<float>(samples);
}

float ParamSmoother::next() noexcept
{
    if (remaining_ > 0)
    {
        --remaining_;
        current_ = remaining_ == 0 ? target_ : current_ + step_;
    }
    return current_;
}

float ParamSmoother::skip(int n) noexcept
{
    if (n <= 0 || remaining_ <= 0)
        return current_;
    if (n >= remaining_)
    {
        current_ = target_;
        remaining_ = 0;
    }
    else
    {
        current_ += step_ * static_cast<float>(n);
        remaining_ -= n;
    }
    return current_;
}

void ParamSmoother::fill(float* out, int n) noexcept
{
    for (int i = 0; i < n; ++i)
        out[i] = next();
}

} // namespace alfa::dsp
