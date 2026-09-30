#include "alfa/dsp/Nonlinear.h"

#include "alfa/dsp/detail/FastMath.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp
{

namespace
{

// Kaiser betas of the two halfband stages. Stage 1 (47 taps) gives ~80 dB stopband from
// 0.61 fs_base; stage 2 (23 taps) only has to separate content below 0.61 fs_base from its
// images, so ~70 dB is enough.
constexpr double kStage1Beta = 7.86;
constexpr double kStage2Beta = 6.76;

constexpr float kAutoGainRef = 0.25f; // -12 dBFS sine: the level auto gain tries to preserve
constexpr float kDcBlockHz = 10.0f;
constexpr float kTonePivotHz = 1000.0f;
constexpr float kToneShelfDb = 6.0f;
constexpr float kFadeStep = 1.0f / 32.0f; // oversampling switch fade, per sample

/// Rational tanh approximation (Pade-like), exact 1 with zero slope at |x| = 3.
inline float softSat(float x) noexcept
{
    x = std::clamp(x, -3.0f, 3.0f);
    const float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

inline float basicShape(int index, float x, float character) noexcept
{
    switch (index)
    {
    case 0:
        return softSat(x);
    case 1:
    {
        // Tube-like: same small-signal slope on both sides, but the negative half saturates
        // k times earlier and k times lower, which is what makes the even harmonics.
        const float k = 1.5f + 2.5f * character;
        return x >= 0.0f ? softSat(x) : softSat(k * x) / k;
    }
    case 2:
    {
        const float c = std::clamp(x, -1.0f, 1.0f);
        return c * (1.5f - 0.5f * c * c);
    }
    case 3:
        return std::clamp(x, -1.0f, 1.0f);
    case 4:
    {
        const float g = 1.0f + 3.0f * character;
        return detail::sin2pi(0.25f * g * x); // sin(pi/2 g x): folds past |x| = 1/g
    }
    case 5:
    {
        // Buchla-style: parallel cells, each adding a slope reversal past its threshold
        // (1, 3, 5, ...), summed with the input. With whole cells active this is an exact
        // triangle fold up to |x| = 2 cells + 1; past the last cell the output clips.
        const float cells = 2.0f + 6.0f * character;
        const float a = std::fabs(x);
        float y = a;
        float sign = 1.0f;
        for (int k = 0; k < 8; ++k)
        {
            const float w = std::clamp(cells - static_cast<float>(k), 0.0f, 1.0f);
            const float t = static_cast<float>(2 * k + 1);
            if (w <= 0.0f || a <= t)
                break;
            y -= 2.0f * sign * w * (a - t);
            sign = -sign;
        }
        return std::clamp(x >= 0.0f ? y : -y, -1.0f, 1.0f);
    }
    case 6:
        return std::max(softSat(x), 0.0f);
    default:
        return std::fabs(softSat(x));
    }
}

/// Crossfade between shape i and i + 1 by frac.
inline float morph(int i, float frac, float x, float character) noexcept
{
    float y = basicShape(i, x, character);
    if (frac > 0.0f)
        y += frac * (basicShape(i + 1, x, character) - y);
    return y;
}

inline float dbToGain(float db) noexcept
{
    return std::pow(10.0f, db / 20.0f);
}

const ParamTable kParams = []
{
    ParamTable t{};
    t[NonlinearProcessor::kDrive] = {"drive", 0.0f, 48.0f, Curve::Linear, 0.0f, "dB"};
    t[NonlinearProcessor::kType] = {"type", 0.0f, 7.0f, Curve::Linear, 0.0f, ""};
    t[NonlinearProcessor::kBias] = {"bias", -1.0f, 1.0f, Curve::Linear, 0.5f, ""};
    t[NonlinearProcessor::kCharacter] = {"character", 0.0f, 1.0f, Curve::Linear, 0.0f, ""};
    t[NonlinearProcessor::kAutoGain] = {"autoGain", 0.0f, 1.0f, Curve::Linear, 1.0f, ""};
    t[NonlinearProcessor::kTrim] = {"trim", -24.0f, 12.0f, Curve::Linear, 24.0f / 36.0f, "dB"};
    t[NonlinearProcessor::kBits] = {"bits", 1.0f, 24.0f, Curve::Linear, 1.0f, "bits"};
    t[NonlinearProcessor::kDownsample] = {"downsample", 1.0f, 64.0f, Curve::Log, 0.0f, "x"};
    t[NonlinearProcessor::kOversampling] = {"oversampling", 0.0f, 2.0f, Curve::Choice, 0.5f, ""};
    t[NonlinearProcessor::kTone] = {"tone", -1.0f, 1.0f, Curve::Linear, 0.5f, ""};
    t[NonlinearProcessor::kMix] = {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""};
    return t;
}();

} // namespace

const ParamTable& NonlinearProcessor::params() noexcept
{
    return kParams;
}

float NonlinearProcessor::shape(float x, float type, float character) noexcept
{
    const float t = std::clamp(type, 0.0f, static_cast<float>(kTypeCount - 1));
    const int i = std::min(static_cast<int>(t), kTypeCount - 2);
    return morph(i, t - static_cast<float>(i), x, character);
}

void NonlinearProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    stage1_.design(kStage1K, kStage1Beta);
    stage2_.design(kStage2K, kStage2Beta);
    for (auto& c : ch_)
    {
        c.up1.setKernel(&stage1_);
        c.down1.setKernel(&stage1_);
        c.up2.setKernel(&stage2_);
        c.down2.setKernel(&stage2_);
    }
    const auto sr = static_cast<float>(sampleRate_);
    dcR_ = std::exp(-detail::kTwoPi * kDcBlockHz / sr);
    toneA_ = 1.0f - std::exp(-detail::kTwoPi * kTonePivotHz / sr);
    reset();
}

void NonlinearProcessor::clearChannel(Channel& c) noexcept
{
    c.up1.reset();
    c.up2.reset();
    c.down1.reset();
    c.down2.reset();
    c.align = 0.0f;
    c.dcX = c.dcY = 0.0f;
    c.holdPhase = 1.0f;
    c.held = 0.0f;
    c.toneLp = 0.0f;
    c.dry.fill(0.0f);
    c.dryPos = 0;
}

void NonlinearProcessor::reset() noexcept
{
    for (auto& c : ch_)
        clearChannel(c);
    fade_ = 1.0f;
    mute_ = 0;
    factor_ = pendingFactor_;
    first_ = true;
}

void NonlinearProcessor::setFactor(int factor) noexcept
{
    factor_ = factor;
    for (auto& c : ch_)
    {
        c.up1.reset();
        c.up2.reset();
        c.down1.reset();
        c.down2.reset();
        c.align = 0.0f;
    }
}

int NonlinearProcessor::latencySamples() const noexcept
{
    constexpr int kLatency2x = 2 * kStage1K + 1;                    // 23
    constexpr int kLatency4x = kLatency2x + (2 * kStage2K + 2) / 2; // 29, incl. the align sample
    return factor_ == 4 ? kLatency4x : factor_ == 2 ? kLatency2x : 0;
}

int NonlinearProcessor::tailSamples() const noexcept
{
    // Latency, the longest hold, and ~5 time constants of the DC blocker (which also carries
    // the decay of any bias-induced DC).
    return latencySamples() + 64 + static_cast<int>(0.08 * sampleRate_);
}

void NonlinearProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const auto& t = kParams;
    const float drive = dbToGain(mapParam(t[kDrive], p[kDrive]));
    drive_.target = drive;
    type_.target = mapParam(t[kType], p[kType]);
    bias_.target = mapParam(t[kBias], p[kBias]);
    character_.target = mapParam(t[kCharacter], p[kCharacter]);

    // Auto gain keeps a -12 dBFS sine at about its level through the soft saturator: 1 at low
    // drive, falling towards kAutoGainRef as the shaper output approaches full scale.
    const float comp = softSat(kAutoGainRef) / softSat(drive * kAutoGainRef);
    const float amount = mapParam(t[kAutoGain], p[kAutoGain]);
    gain_.target = std::pow(comp, amount) * dbToGain(mapParam(t[kTrim], p[kTrim]));

    const float tilt = mapParam(t[kTone], p[kTone]);
    toneHigh_.target = dbToGain(kToneShelfDb * tilt);
    toneLow_.target = 1.0f / toneHigh_.target;
    mix_.target = mapParam(t[kMix], p[kMix]);

    bits_ = mapParam(t[kBits], p[kBits]);
    holdIncrement_ = 1.0f / mapParam(t[kDownsample], p[kDownsample]);
    pendingFactor_ = quality_ > 0 ? 1 : 1 << static_cast<int>(mapParam(t[kOversampling], p[kOversampling]));

    if (first_)
    {
        for (Ramp* r : {&drive_, &type_, &bias_, &character_, &gain_, &toneLow_, &toneHigh_, &mix_})
            r->snap();
        if (pendingFactor_ != factor_)
            setFactor(pendingFactor_);
        fade_ = 1.0f;
        first_ = false;
    }
}

float NonlinearProcessor::shaperAt(float x, const Sample& s) noexcept
{
    return morph(s.typeIndex, s.typeFrac, s.drive * x + s.bias, s.character) - s.offset;
}

float NonlinearProcessor::tick(Channel& c, float x, const Sample& s) const noexcept
{
    c.dry[static_cast<std::size_t>(c.dryPos)] = x;
    const float dry = c.dry[static_cast<std::size_t>((c.dryPos - latencySamples()) & 31)];
    c.dryPos = (c.dryPos + 1) & 31;

    float w;
    if (factor_ == 1)
    {
        w = shaperAt(x, s);
    }
    else if (factor_ == 2)
    {
        float a, b;
        c.up1.process(x, a, b);
        w = c.down1.process(shaperAt(a, s), shaperAt(b, s));
    }
    else
    {
        float a, b, p0, p1, p2, p3;
        c.up1.process(x, a, b);
        c.up2.process(a, p0, p1);
        c.up2.process(b, p2, p3);
        const float r0 = c.down2.process(shaperAt(p0, s), shaperAt(p1, s));
        const float r1 = c.down2.process(shaperAt(p2, s), shaperAt(p3, s));
        w = c.down1.process(c.align, r0);
        c.align = r1;
    }

    const float blocked = w - c.dcX + dcR_ * c.dcY;
    c.dcX = w;
    c.dcY = blocked;
    float y = (dcBlock_ ? blocked : w) * s.gain;

    if (c.holdPhase >= 1.0f)
    {
        c.holdPhase -= 1.0f; // the increment is at most 1, so the phase stays below 2
        c.held = y;
    }
    c.holdPhase += holdIncrement_;
    y = c.held;

    if (s.crush)
    {
        // Round half up after clamping to the level range (|v| < 2^23, so the int cast is
        // exact); a cast instead of std::floor avoids a libm call per sample.
        const float v = std::clamp(y * s.q, s.qLo, s.qHi) + 0.5f;
        auto r = static_cast<float>(static_cast<int>(v));
        if (r > v)
            r -= 1.0f;
        y = std::min(r, s.qHi) / s.q;
    }

    c.toneLp += toneA_ * (y - c.toneLp);
    if (s.toneOn)
        y = s.toneLow * c.toneLp + s.toneHigh * (y - c.toneLp);

    return s.fade * ((1.0f - s.mix) * dry + s.mix * y);
}

void NonlinearProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    if (first_)
    {
        // process() before any setParams(): run with the neutral defaults.
        std::array<float, kSlotParams> defaults{};
        for (std::size_t i = 0; i < defaults.size(); ++i)
            defaults[i] = kParams[i].defaultNorm;
        setParams(defaults.data(), {});
    }

    for (Ramp* r : {&drive_, &type_, &bias_, &character_, &gain_, &toneLow_, &toneHigh_, &mix_})
        r->begin(n);
    Sample s;
    s.toneOn = toneHigh_.cur != 1.0f || toneHigh_.target != 1.0f;
    s.crush = bits_ < 23.5f;
    s.q = std::exp2(bits_ - 1.0f);
    s.qLo = -std::floor(s.q);
    s.qHi = std::ceil(s.q) - 1.0f;

    // The static offset shape(bias) is ramped between its block end points rather than
    // evaluated per sample; during a ramp the small error is DC-like and the blocker removes it.
    const float offsetFrom = shape(bias_.cur, type_.cur, character_.cur);
    const float offsetStep =
        (shape(bias_.target, type_.target, character_.target) - offsetFrom) / static_cast<float>(n);

    float probe = 0.0f; // becomes NaN if any input or output sample is non-finite
    for (int i = 0; i < n; ++i)
    {
        // Oversampling switch: fade out, switch (clearing the resampler state), stay muted
        // while the new filters fill, then fade back in. Sample-major so both channels switch
        // on the same sample.
        if (pendingFactor_ != factor_)
        {
            fade_ = std::max(0.0f, fade_ - kFadeStep);
            if (fade_ == 0.0f)
            {
                setFactor(pendingFactor_);
                mute_ = 2 * latencySamples();
            }
        }
        else if (mute_ > 0)
            --mute_;
        else
            fade_ = std::min(1.0f, fade_ + kFadeStep);

        const auto fi = static_cast<float>(i + 1);
        s.drive = drive_.cur + drive_.step * fi;
        const float type = std::clamp(type_.cur + type_.step * fi, 0.0f, static_cast<float>(kTypeCount - 1));
        s.typeIndex = std::min(static_cast<int>(type), kTypeCount - 2);
        s.typeFrac = type - static_cast<float>(s.typeIndex);
        s.bias = bias_.cur + bias_.step * fi;
        s.character = character_.cur + character_.step * fi;
        s.offset = offsetFrom + offsetStep * fi;
        s.gain = gain_.cur + gain_.step * fi;
        s.toneLow = toneLow_.cur + toneLow_.step * fi;
        s.toneHigh = toneHigh_.cur + toneHigh_.step * fi;
        s.mix = mix_.cur + mix_.step * fi;
        s.fade = fade_;
        for (int chIndex = 0; chIndex < channels_; ++chIndex)
        {
            float& io = channels[chIndex][i];
            probe += io - io;
            io = tick(ch_[static_cast<std::size_t>(chIndex)], io, s);
            probe += io - io;
        }
    }

    for (Ramp* r : {&drive_, &type_, &bias_, &character_, &gain_, &toneLow_, &toneHigh_, &mix_})
        r->snap();
    for (auto& c : ch_)
    {
        if (std::fabs(c.dcY) < 1e-20f)
            c.dcY = 0.0f;
        if (std::fabs(c.toneLp) < 1e-20f)
            c.toneLp = 0.0f;
    }

    if (probe != probe)
    {
        for (auto& c : ch_)
            clearChannel(c);
        for (int chIndex = 0; chIndex < channels_; ++chIndex)
            std::fill_n(channels[chIndex], n, 0.0f);
    }
}

} // namespace alfa::dsp
