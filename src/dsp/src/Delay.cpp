#include "alfa/dsp/Delay.h"

#include "alfa/dsp/detail/FastMath.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace alfa::dsp
{

namespace
{

constexpr double kPiD = 3.14159265358979323846;

float onePoleCoefficient(double seconds, double sampleRate) noexcept
{
    return static_cast<float>(1.0 - std::exp(-1.0 / (seconds * sampleRate)));
}

/// Samples until a loop with gain g has decayed by 60 dB, for a loop of `period` samples.
double decaySamples(double period, double g) noexcept
{
    g = std::fabs(g);
    if (g < 1e-3)
        return period;
    if (g >= 0.9999)
        return std::numeric_limits<double>::max();
    return period * (1.0 + std::ceil(std::log(1e-3) / std::log(g)));
}

int clampTail(double samples, double sampleRate) noexcept
{
    const double cap = 60.0 * sampleRate; // longer tails are cut by the voice / transport anyway
    return static_cast<int>(std::min(std::max(samples, 0.0), cap));
}

void zero(float* const* channels, int numChannels, int n) noexcept
{
    for (int c = 0; c < numChannels; ++c)
        std::fill_n(channels[c], n, 0.0f);
}

bool blockFinite(float* const* channels, int numChannels, int n) noexcept
{
    bool ok = true;
    for (int c = 0; c < numChannels; ++c)
        ok = ok && detail::allFinite(channels[c], n);
    return ok;
}

float wrap01(float x) noexcept
{
    return x - std::floor(x);
}

} // namespace

// DelayProcessor --------------------------------------------------------------------------------

const ParamTable& DelayProcessor::params() noexcept
{
    static const ParamTable table{{
        {"time L", 1.0f, 2000.0f, Curve::Log, 0.7264f, "ms"},   // 250 ms
        {"time R", 1.0f, 2000.0f, Curve::Log, 0.7264f, "ms"},   // ignored when linked or mono
        {"link", 0.0f, 1.0f, Curve::Toggle, 0.0f, ""},          // R follows L
        {"feedback", 0.0f, 0.98f, Curve::Linear, 0.3571f, ""},  // 0.35
        {"ping-pong", 0.0f, 1.0f, Curve::Linear, 0.0f, ""},     // 1: input to L, repeats alternate
        {"low cut", 20.0f, 2000.0f, Curve::Log, 0.0f, "Hz"},    // in-loop high-pass
        {"high cut", 500.0f, 20000.0f, Curve::Log, 1.0f, "Hz"}, // in-loop low-pass
        {"wow depth", 0.0f, kMaxModMs, Curve::Linear, 0.0f, "ms"},
        {"wow rate", 0.05f, 10.0f, Curve::Log, 0.4346f, "Hz"}, // 0.5 Hz
        {"diffusion", 0.0f, 1.0f, Curve::Linear, 0.0f, ""},    // allpass gain 0..0.7 in the loop
        {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
    }};
    return table;
}

void DelayProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    channels_ = std::clamp(spec.channels, 1, 2);
    const double sr = sampleRate_;
    const auto maxLen = static_cast<int>(std::ceil(static_cast<double>(kMaxSeconds) * sr +
                                                   static_cast<double>(kMaxModMs) * 1e-3 * sr)) +
                        8;
    for (auto& c : ch_)
    {
        c.line.prepare(maxLen);
        c.diffuser.prepare(static_cast<int>(std::ceil(0.012 * sr)) + 2);
    }
    ch_[0].apBase = std::round(0.0071 * sr);
    ch_[1].apBase = std::round(0.0093 * sr);
    ch_[0].lfoOffset = 0.0f;
    ch_[1].lfoOffset = 0.25f; // 90 degrees: the wow is decorrelated between channels
    glide_ = 1.0 - std::exp(-1.0 / (0.06 * sr));
    fadeInc_ = static_cast<float>(1.0 / (0.03 * sr));
    jump_ = 0.02 * sr;
    reset();
}

void DelayProcessor::reset() noexcept
{
    for (auto& c : ch_)
    {
        c.line.reset();
        c.diffuser.reset();
        c.lowpass.y = 0.0f;
        c.highpass.y = 0.0f;
        c.fading = false;
        c.fade = 0.0f;
    }
    lfoPhase_ = 0.0f;
    first_ = true;
}

void DelayProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const auto& t = params();
    const auto v = [&](int i)
    {
        return mapParam(t[static_cast<std::size_t>(i)], p[i]);
    };
    const float msToSamples = static_cast<float>(sampleRate_ / 1000.0);

    const float timeL = v(kTimeL);
    const float timeR = v(kLink) > 0.5f ? timeL : v(kTimeR);
    ch_[0].target = static_cast<double>(timeL) * sampleRate_ / 1000.0;
    ch_[1].target = static_cast<double>(timeR) * sampleRate_ / 1000.0;

    fbTarget_ = std::clamp(v(kFeedback), 0.0f, 0.98f);
    ppTarget_ = channels_ == 2 ? v(kPingPong) : 0.0f;
    mixTarget_ = v(kMix);
    depthTarget_ = v(kModDepth) * msToSamples;
    lfoInc_ = static_cast<float>(static_cast<double>(v(kModRate)) / sampleRate_);
    const float apGain = 0.7f * v(kDiffusion);

    const float lp = detail::OnePole::coefficientFor(static_cast<double>(v(kHighCut)), sampleRate_);
    const float hp = detail::OnePole::coefficientFor(static_cast<double>(v(kLowCut)), sampleRate_);
    for (auto& c : ch_)
    {
        c.lowpass.p = lp;
        c.highpass.p = hp;
        c.diffuser.setGain(apGain);
        // The diffuser delay is part of the loop time, so it may take at most half of it.
        c.apTarget = std::min(c.apBase, 0.5 * c.target);
        if (first_)
        {
            c.cur = c.next = c.target;
            c.apDelay = c.apTarget;
        }
    }

    if (first_)
    {
        fb_ = fbTarget_;
        pp_ = ppTarget_;
        mix_ = mixTarget_;
        depth_ = depthTarget_;
        first_ = false;
    }
}

namespace
{
constexpr double kMaxGlideRate = 0.25; // samples of delay change per sample (+-25% speed)

double glideStep(double from, double to, double coefficient) noexcept
{
    return std::clamp(coefficient * (to - from), -kMaxGlideRate, kMaxGlideRate);
}
} // namespace

float DelayProcessor::readHeads(Channel& c, float mod) noexcept
{
    if (!c.fading)
    {
        if (std::fabs(c.target - c.cur) > jump_)
        {
            c.fading = true;
            c.fade = 0.0f;
            c.next = c.target;
        }
        else
        {
            c.cur += glideStep(c.cur, c.target, glide_);
        }
    }
    const double offset = static_cast<double>(mod) - c.apDelay - 1.0;
    const float a = c.line.readCubic(c.cur + offset);
    if (!c.fading)
        return a;

    c.next += glideStep(c.next, c.target, glide_);
    const float b = c.line.readCubic(c.next + offset);
    c.fade += fadeInc_;
    const float y = a + std::min(c.fade, 1.0f) * (b - a);
    if (c.fade >= 1.0f)
    {
        c.cur = c.next;
        c.fading = false;
    }
    return y;
}

void DelayProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    if (first_)
    {
        // process() before any setParams(): run with the neutral defaults.
        std::array<float, kSlotParams> d{};
        for (std::size_t i = 0; i < d.size(); ++i)
            d[i] = params()[i].defaultNorm;
        setParams(d.data(), {});
    }
    if (!blockFinite(channels, channels_, n))
    {
        reset();
        zero(channels, channels_, n);
        return;
    }

    const float inv = 1.0f / static_cast<float>(n);
    const float dFb = (fbTarget_ - fb_) * inv;
    const float dPp = (ppTarget_ - pp_) * inv;
    const float dMix = (mixTarget_ - mix_) * inv;
    const float dDepth = (depthTarget_ - depth_) * inv;
    const bool stereo = channels_ == 2;

    for (int i = 0; i < n; ++i)
    {
        fb_ += dFb;
        pp_ += dPp;
        mix_ += dMix;
        depth_ += dDepth;
        lfoPhase_ = wrap01(lfoPhase_ + lfoInc_);

        std::array<float, 2> tap{};
        std::array<float, 2> fb{};
        for (int c = 0; c < channels_; ++c)
        {
            auto& s = ch_[static_cast<std::size_t>(c)];
            s.apDelay += glide_ * (s.apTarget - s.apDelay);
            const float mod = depth_ > 0.0f ? 0.5f * depth_ * detail::sin2pi(lfoPhase_ + s.lfoOffset) : 0.0f;
            const float y = readHeads(s, mod);
            tap[static_cast<std::size_t>(c)] = y;
            fb[static_cast<std::size_t>(c)] =
                detail::softLimit(fb_ * s.highpass.highpass(s.lowpass.lowpass(y)));
        }

        const float xL = channels[0][i];
        if (stereo)
        {
            const float xR = channels[1][i];
            // Ping-pong: the input goes (as mono) to the left line only, and each repeat
            // crosses to the other side. The mixing matrix [[1-p, p], [p, 1-p]] has eigenvalues
            // 1 and 1-2p, so it never adds loop gain.
            const float toL = (1.0f - pp_) * fb[0] + pp_ * fb[1];
            const float toR = (1.0f - pp_) * fb[1] + pp_ * fb[0];
            const float inL = (1.0f - pp_) * xL + pp_ * 0.5f * (xL + xR);
            const float inR = (1.0f - pp_) * xR;
            ch_[0].line.write(
                ch_[0].diffuser.processWithDelay(inL + toL, static_cast<float>(ch_[0].apDelay)));
            ch_[1].line.write(
                ch_[1].diffuser.processWithDelay(inR + toR, static_cast<float>(ch_[1].apDelay)));
            channels[0][i] = xL + mix_ * (tap[0] - xL);
            channels[1][i] = xR + mix_ * (tap[1] - xR);
        }
        else
        {
            ch_[0].line.write(
                ch_[0].diffuser.processWithDelay(xL + fb[0], static_cast<float>(ch_[0].apDelay)));
            channels[0][i] = xL + mix_ * (tap[0] - xL);
        }
    }

    fb_ = fbTarget_;
    pp_ = ppTarget_;
    mix_ = mixTarget_;
    depth_ = depthTarget_;
    for (auto& c : ch_)
    {
        detail::flushDenormal(c.lowpass.y);
        detail::flushDenormal(c.highpass.y);
    }
    if (!blockFinite(channels, channels_, n))
    {
        reset();
        zero(channels, channels_, n);
    }
}

int DelayProcessor::tailSamples() const noexcept
{
    const double period = std::max(ch_[0].target, channels_ == 2 ? ch_[1].target : 0.0);
    // Diffusion smears each repeat by a few allpass periods; wow lengthens the read slightly.
    const double extra = static_cast<double>(depthTarget_) + 0.1 * sampleRate_;
    return clampTail(decaySamples(period, static_cast<double>(fbTarget_)) + extra, sampleRate_);
}

// ModDelayProcessor -----------------------------------------------------------------------------

namespace
{
constexpr int kModSub = 8; // samples per LFO evaluation
}

const ParamTable& ModDelayProcessor::params() noexcept
{
    static const ParamTable table{{
        {"base delay", 0.1f, kMaxBaseMs, Curve::Log, 0.7091f, "ms"}, // 7 ms; shortest delay
        {"depth", 0.0f, kMaxDepthMs, Curve::Linear, 0.3f, "ms"},     // sweep base..base+depth
        {"rate", 0.02f, 10.0f, Curve::Log, 0.518f, "Hz"},            // 0.5 Hz
        {"voices", 0.0f, 3.0f, Curve::Choice, 0.0f, "voices - 1"},   // 1..4 heads
        {"feedback", -0.95f, 0.95f, Curve::Linear, 0.5f, ""},
        {"stereo spread", 0.0f, 180.0f, Curve::Linear, 0.5f, "deg"}, // R LFO phase offset
        {"mix", 0.0f, 1.0f, Curve::Linear, 0.5f, ""},
    }};
    return table;
}

void ModDelayProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    channels_ = std::clamp(spec.channels, 1, 2);
    const auto maxLen =
        static_cast<int>(std::ceil(static_cast<double>(kMaxBaseMs + kMaxDepthMs) * 1e-3 * sampleRate_)) + 8;
    for (int c = 0; c < channels_; ++c)
        line_[static_cast<std::size_t>(c)].prepare(maxLen);
    // Per-sub-block one-pole: 20 ms time constant.
    smooth_ = static_cast<float>(1.0 - std::exp(-static_cast<double>(kModSub) / (0.02 * sampleRate_)));
    reset();
}

void ModDelayProcessor::reset() noexcept
{
    for (int c = 0; c < channels_; ++c)
        line_[static_cast<std::size_t>(c)].reset();
    lfoPhase_ = 0.0f;
    first_ = true;
}

void ModDelayProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const auto& t = params();
    const auto v = [&](int i)
    {
        return mapParam(t[static_cast<std::size_t>(i)], p[i]);
    };
    const float msToSamples = static_cast<float>(sampleRate_ / 1000.0);

    baseTarget_ = static_cast<double>(v(kBase) * msToSamples);
    depthTarget_ = static_cast<double>(v(kDepth) * msToSamples);
    lfoInc_ = static_cast<float>(static_cast<double>(v(kRate)) / sampleRate_);
    const int voices = std::clamp(static_cast<int>(v(kVoices)) + 1, 1, kMaxVoices);
    for (int k = 0; k < kMaxVoices; ++k)
    {
        const auto s = static_cast<std::size_t>(k);
        voiceGainTarget_[s] = k < voices ? 1.0f / static_cast<float>(voices) : 0.0f;
        // Inactive heads keep their phase so re-enabling them glides from where they were.
        if (k < voices)
            voiceOffsetTarget_[s] = static_cast<float>(k) / static_cast<float>(voices);
    }
    fbTarget_ = v(kFeedback);
    spreadTarget_ = v(kSpread) / 360.0f; // degrees to cycles
    mixTarget_ = v(kMix);

    if (first_)
    {
        base_ = baseTarget_;
        depth_ = depthTarget_;
        spread_ = spreadTarget_;
        fb_ = fbTarget_;
        mix_ = mixTarget_;
        voiceGain_ = voiceGainTarget_;
        voiceOffset_ = voiceOffsetTarget_;
        for (int c = 0; c < 2; ++c)
            for (int j = 0; j < kMaxVoices; ++j)
                pos_[static_cast<std::size_t>(c)][static_cast<std::size_t>(j)] = headDelay(
                    lfoPhase_ + (c == 1 ? spread_ : 0.0f) + voiceOffset_[static_cast<std::size_t>(j)]);
        first_ = false;
    }
}

void ModDelayProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    if (first_)
    {
        std::array<float, kSlotParams> d{};
        for (std::size_t i = 0; i < d.size(); ++i)
            d[i] = params()[i].defaultNorm;
        setParams(d.data(), {});
    }
    if (!blockFinite(channels, channels_, n))
    {
        reset();
        zero(channels, channels_, n);
        return;
    }

    const float inv = 1.0f / static_cast<float>(n);
    const float dFb = (fbTarget_ - fb_) * inv;
    const float dMix = (mixTarget_ - mix_) * inv;
    const float fb0 = fb_;
    const float mix0 = mix_;
    const float k = smooth_;
    const auto kd = static_cast<double>(smooth_);

    // The LFOs (<= 10 Hz) and the smoothed params are evaluated every kModSub samples and the
    // head delays and gains ramp linearly in between: indistinguishable from per-sample
    // evaluation, at an eighth of the sin2pi cost.
    for (int start = 0; start < n; start += kModSub)
    {
        const int m = std::min(kModSub, n - start);
        const float invM = 1.0f / static_cast<float>(m);
        base_ += kd * (baseTarget_ - base_);
        depth_ += kd * (depthTarget_ - depth_);
        spread_ += k * (spreadTarget_ - spread_);
        lfoPhase_ = wrap01(lfoPhase_ + lfoInc_ * static_cast<float>(m));
        std::array<float, kMaxVoices> g0{};
        std::array<float, kMaxVoices> gStep{};
        for (int j = 0; j < kMaxVoices; ++j)
        {
            const auto s = static_cast<std::size_t>(j);
            g0[s] = voiceGain_[s];
            voiceGain_[s] += k * (voiceGainTarget_[s] - voiceGain_[s]);
            gStep[s] = (voiceGain_[s] - g0[s]) * invM;
            voiceOffset_[s] += k * (voiceOffsetTarget_[s] - voiceOffset_[s]);
        }

        for (int c = 0; c < channels_; ++c)
        {
            const auto cs = static_cast<std::size_t>(c);
            auto& line = line_[cs];
            auto& pos = pos_[cs];
            const float phase = lfoPhase_ + (c == 1 ? spread_ : 0.0f);
            std::array<double, kMaxVoices> step{};
            std::array<bool, kMaxVoices> active{};
            for (int j = 0; j < kMaxVoices; ++j)
            {
                const auto s = static_cast<std::size_t>(j);
                // Every head keeps moving, audible or not, so fading one in never jumps.
                const double end = headDelay(phase + voiceOffset_[s]);
                step[s] = (end - pos[s]) / static_cast<double>(m);
                active[s] = g0[s] > 1e-5f || voiceGain_[s] > 1e-5f;
            }
            float* x = channels[c] + start;
            for (int i = 0; i < m; ++i)
            {
                float wet = 0.0f;
                for (int j = 0; j < kMaxVoices; ++j)
                {
                    const auto s = static_cast<std::size_t>(j);
                    pos[s] += step[s];
                    if (active[s])
                        wet += (g0[s] + gStep[s] * static_cast<float>(i + 1)) * line.readCubic(pos[s] - 1.0);
                }
                const auto t = static_cast<float>(start + i + 1);
                const float dry = x[i];
                line.write(dry + (fb0 + dFb * t) * detail::softLimit(wet));
                x[i] = dry + (mix0 + dMix * t) * (wet - dry);
            }
        }
    }

    fb_ = fbTarget_;
    mix_ = mixTarget_;
    if (!blockFinite(channels, channels_, n))
    {
        reset();
        zero(channels, channels_, n);
    }
}

double ModDelayProcessor::headDelay(float phase) const noexcept
{
    return base_ + depth_ * 0.5 * (1.0 + static_cast<double>(detail::sin2pi(phase)));
}

int ModDelayProcessor::tailSamples() const noexcept
{
    const double period = baseTarget_ + depthTarget_;
    return clampTail(decaySamples(period, static_cast<double>(fbTarget_)) + 1.0, sampleRate_);
}

// PhaserProcessor -------------------------------------------------------------------------------

namespace
{
constexpr int kPhaserSub = 8; // samples per coefficient update
}

const ParamTable& PhaserProcessor::params() noexcept
{
    static const ParamTable table{{
        {"stages", 0.0f, 5.0f, Curve::Choice, 0.2f, "stages / 2 - 1"}, // 2, 4, .. 12; default 4
        {"centre", 50.0f, 10000.0f, Curve::Log, 0.5233f, "Hz"},        // 800 Hz
        {"depth", 0.0f, 4.0f, Curve::Linear, 0.375f, "oct"},           // +-1.5 octaves
        {"rate", 0.02f, 10.0f, Curve::Log, 0.4358f, "Hz"},             // 0.3 Hz
        {"feedback", -0.95f, 0.95f, Curve::Linear, 0.5f, ""},
        {"stereo phase", 0.0f, 180.0f, Curve::Linear, 0.5f, "deg"}, // R LFO offset
        {"mix", 0.0f, 1.0f, Curve::Linear, 0.5f, ""},
    }};
    return table;
}

void PhaserProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    channels_ = std::clamp(spec.channels, 1, 2);
    smooth_ = static_cast<float>(1.0 - std::exp(-static_cast<double>(kPhaserSub) / (0.02 * sampleRate_)));
    stageFadeInc_ = static_cast<float>(1.0 / (0.01 * sampleRate_));
    reset();
}

void PhaserProcessor::reset() noexcept
{
    for (auto& s : state_)
        s.fill(0.0f);
    yPrev_.fill(0.0f);
    lfoPhase_ = 0.0f;
    stageFade_ = 1.0f;
    stagesOld_ = stages_;
    first_ = true;
}

float PhaserProcessor::coefficientAt(float phase) const noexcept
{
    const float oct = depth_ * detail::sin2pi(phase);
    const double f = std::clamp(static_cast<double>(centre_) * std::exp2(static_cast<double>(oct)), 10.0,
                                0.45 * sampleRate_);
    const double t = std::tan(kPiD * f / sampleRate_);
    return static_cast<float>((t - 1.0) / (t + 1.0));
}

void PhaserProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const auto& t = params();
    const auto v = [&](int i)
    {
        return mapParam(t[static_cast<std::size_t>(i)], p[i]);
    };

    stagesTarget_ = std::clamp(2 * (static_cast<int>(v(kStages)) + 1), 2, kMaxStages);
    centreTarget_ = v(kCentre);
    depthTarget_ = v(kDepth);
    lfoInc_ = static_cast<float>(static_cast<double>(v(kRate)) / sampleRate_);
    fbTarget_ = v(kFeedback);
    offsetTarget_ = v(kStereoPhase) / 360.0f;
    mixTarget_ = v(kMix);

    if (first_)
    {
        stages_ = stagesOld_ = stagesTarget_;
        stageFade_ = 1.0f;
        centre_ = centreTarget_;
        depth_ = depthTarget_;
        offset_ = offsetTarget_;
        fb_ = fbTarget_;
        mix_ = mixTarget_;
        for (int c = 0; c < 2; ++c)
            coef_[static_cast<std::size_t>(c)] = coefficientAt(lfoPhase_ + (c == 1 ? offset_ : 0.0f));
        first_ = false;
    }
}

void PhaserProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    if (first_)
    {
        std::array<float, kSlotParams> d{};
        for (std::size_t i = 0; i < d.size(); ++i)
            d[i] = params()[i].defaultNorm;
        setParams(d.data(), {});
    }
    if (!blockFinite(channels, channels_, n))
    {
        reset();
        zero(channels, channels_, n);
        return;
    }

    // Start a stage-count crossfade only from a settled state, so a fade is never cut short.
    if (stageFade_ >= 1.0f && stagesTarget_ != stages_)
    {
        stagesOld_ = stages_;
        stages_ = stagesTarget_;
        stageFade_ = 0.0f;
        // Newly added stages start from rest; their output is faded in.
        for (auto& s : state_)
            for (int k = stagesOld_; k < stages_; ++k)
                s[static_cast<std::size_t>(k)] = 0.0f;
    }

    const float inv = 1.0f / static_cast<float>(n);
    const float dFb = (fbTarget_ - fb_) * inv;
    const float dMix = (mixTarget_ - mix_) * inv;

    for (int start = 0; start < n; start += kPhaserSub)
    {
        const int m = std::min(kPhaserSub, n - start);
        centre_ += smooth_ * (centreTarget_ - centre_);
        depth_ += smooth_ * (depthTarget_ - depth_);
        offset_ += smooth_ * (offsetTarget_ - offset_);
        lfoPhase_ = wrap01(lfoPhase_ + lfoInc_ * static_cast<float>(m));

        for (int c = 0; c < channels_; ++c)
        {
            const auto cs = static_cast<std::size_t>(c);
            const float aEnd = coefficientAt(lfoPhase_ + (c == 1 ? offset_ : 0.0f));
            const float dA = (aEnd - coef_[cs]) / static_cast<float>(m);
            float a = coef_[cs];
            float yPrev = yPrev_[cs];
            float fb = fb_;
            float mix = mix_;
            float fade = stageFade_;
            auto& st = state_[cs];
            const int active = std::max(stages_, stagesOld_);
            float* x = channels[c] + start;

            for (int i = 0; i < m; ++i)
            {
                a += dA;
                fb += dFb;
                mix += dMix;
                const float dry = x[i];
                float u = dry + detail::softLimit(fb * yPrev);
                float yOld = 0.0f;
                float yNew = 0.0f;
                for (int k = 0; k < active; ++k)
                {
                    auto& s = st[static_cast<std::size_t>(k)];
                    const float y = a * u + s;
                    s = u - a * y;
                    u = y;
                    if (k + 1 == stagesOld_)
                        yOld = u;
                    if (k + 1 == stages_)
                        yNew = u;
                }
                float y = yNew;
                if (fade < 1.0f)
                {
                    fade = std::min(1.0f, fade + stageFadeInc_);
                    y = yOld + fade * (yNew - yOld);
                }
                yPrev = y;
                x[i] = dry + mix * (y - dry);
            }
            coef_[cs] = a;
            yPrev_[cs] = yPrev;
            if (c == channels_ - 1)
            {
                fb_ = fb;
                mix_ = mix;
                stageFade_ = fade;
            }
        }
    }

    fb_ = fbTarget_;
    mix_ = mixTarget_;
    if (stageFade_ >= 1.0f)
        stagesOld_ = stages_;
    for (auto& s : state_)
        for (auto& v : s)
            detail::flushDenormal(v);
    for (auto& y : yPrev_)
        detail::flushDenormal(y);
    if (!blockFinite(channels, channels_, n))
    {
        reset();
        zero(channels, channels_, n);
    }
}

int PhaserProcessor::tailSamples() const noexcept
{
    // Low-frequency group delay of one first-order allpass is about fs / (pi f) samples; the
    // impulse response of the chain rings for a few of those, repeated by the feedback.
    const double fMin =
        std::max(10.0, static_cast<double>(centreTarget_) * std::exp2(-static_cast<double>(depthTarget_)));
    const double ring = 3.0 * static_cast<double>(kMaxStages) * sampleRate_ / (kPiD * fMin);
    return clampTail(decaySamples(ring, static_cast<double>(fbTarget_)), sampleRate_);
}

// ReverbProcessor -------------------------------------------------------------------------------

namespace
{
// Early reflections at size 1 (ms) and their gains; odd taps come from the opposite channel.
constexpr std::array<float, ReverbProcessor::kEarlyTaps> kEarlyMs{7.1f, 11.3f, 17.9f, 23.3f, 31.7f, 41.9f};
constexpr std::array<float, ReverbProcessor::kEarlyTaps> kEarlyGain{0.72f, 0.61f, 0.53f, 0.44f, 0.37f, 0.30f};
constexpr float kMaxEarlyMs = 42.0f;
// Input diffuser delays (ms), different per channel so the two late inputs decorrelate.
constexpr std::array<std::array<float, 4>, 2> kDiffuserMs{
    {{1.7f, 2.9f, 4.3f, 6.1f}, {1.9f, 3.1f, 4.7f, 5.7f}}};
constexpr std::array<float, 4> kDiffuserGain{0.75f, 0.75f, 0.625f, 0.625f};
// Rows of the Sylvester 8x8 Hadamard matrix: two to inject L/R, two (orthogonal to each other
// and to the injection rows) to tap L/R, so the outputs are decorrelated.
constexpr std::array<float, 8> kInL{1, -1, -1, 1, 1, -1, -1, 1};
constexpr std::array<float, 8> kInR{1, -1, 1, -1, -1, 1, -1, 1};
constexpr std::array<float, 8> kOutL{1, -1, 1, -1, 1, -1, 1, -1};
constexpr std::array<float, 8> kOutR{1, 1, -1, -1, 1, 1, -1, -1};
constexpr float kInGain = 0.5f;
constexpr float kOutGain = 0.25f;
constexpr float kMaxModMs = 1.0f;
} // namespace

const ParamTable& ReverbProcessor::params() noexcept
{
    static const ParamTable table{{
        {"pre-delay", 0.0f, kMaxPreDelayMs, Curve::Linear, 0.05f, "ms"}, // 10 ms
        {"size", 0.2f, 1.0f, Curve::Linear, 0.5f, "x"},                  // FDN lines 9..97 ms
        {"decay", 0.1f, 30.0f, Curve::Log, 0.5252f, "s"},                // RT60, 2 s
        {"damping", 0.0f, 1.0f, Curve::Linear, 0.3f, ""},                // HF RT60 / (1 + 9 x)
        {"low damping", 0.0f, 1.0f, Curve::Linear, 0.0f, ""},            // LF RT60 / (1 + 4 x)
        {"diffusion", 0.0f, 1.0f, Curve::Linear, 0.7f, ""},
        {"modulation", 0.0f, 1.0f, Curve::Linear, 0.3f, ""}, // up to 1 ms line wobble
        {"width", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
        {"early/late", 0.0f, 1.0f, Curve::Linear, 0.5f, ""}, // 0 early, 1 late
        {"freeze", 0.0f, 1.0f, Curve::Toggle, 0.0f, ""},
        {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""},
    }};
    return table;
}

void ReverbProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    channels_ = std::clamp(spec.channels, 1, 2);
    const double msToSamples = sampleRate_ / 1000.0;
    const auto preLen =
        static_cast<int>(std::ceil(static_cast<double>(kMaxPreDelayMs + kMaxEarlyMs) * msToSamples)) + 8;
    for (int c = 0; c < 2; ++c)
    {
        const auto s = static_cast<std::size_t>(c);
        pre_[s].prepare(preLen);
        std::array<float, 4> d{};
        for (std::size_t k = 0; k < d.size(); ++k)
            d[k] = static_cast<float>(std::round(static_cast<double>(kDiffuserMs[s][k]) * msToSamples));
        diffuser_[s].prepare(d.data(), 4);
    }
    fdn_.prepare(sampleRate_, kMaxModMs);
    smooth_ = onePoleCoefficient(0.05, sampleRate_);
    reset();
}

void ReverbProcessor::reset() noexcept
{
    for (auto& p : pre_)
        p.reset();
    for (auto& d : diffuser_)
        d.reset();
    fdn_.reset();
    first_ = true;
}

void ReverbProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const auto& t = params();
    const auto v = [&](int i)
    {
        return mapParam(t[static_cast<std::size_t>(i)], p[i]);
    };

    preDelayTarget_ = v(kPreDelay) * static_cast<float>(sampleRate_ / 1000.0);
    sizeTarget_ = v(kSize);
    rt60_ = v(kDecay);
    const float hfRatio = 1.0f + 9.0f * v(kDamping);
    const float lfRatio = 1.0f + 4.0f * v(kLowDamping);
    const float diffusion = v(kDiffusion);
    std::array<float, 4> g{};
    for (std::size_t k = 0; k < g.size(); ++k)
        g[k] = kDiffuserGain[k] * diffusion;
    for (auto& d : diffuser_)
        d.setGains(g.data());
    const float balance = v(kEarlyLate);
    earlyTarget_ = std::min(1.0f, 2.0f * (1.0f - balance));
    lateTarget_ = std::min(1.0f, 2.0f * balance);
    widthTarget_ = v(kWidth);
    mixTarget_ = v(kMix);
    freeze_ = v(kFreeze) > 0.5f;
    inGainTarget_ = freeze_ ? 0.0f : 1.0f;
    fdn_.setParams(sizeTarget_, rt60_, hfRatio, lfRatio, kMaxModMs * v(kModulation), freeze_);

    if (first_)
    {
        preDelay_ = preDelayTarget_;
        size_ = sizeTarget_;
        inGain_ = inGainTarget_;
        early_ = earlyTarget_;
        late_ = lateTarget_;
        width_ = widthTarget_;
        mix_ = mixTarget_;
        first_ = false;
    }
}

void ReverbProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    if (first_)
    {
        std::array<float, kSlotParams> d{};
        for (std::size_t i = 0; i < d.size(); ++i)
            d[i] = params()[i].defaultNorm;
        setParams(d.data(), {});
    }
    if (!blockFinite(channels, channels_, n))
    {
        reset();
        zero(channels, channels_, n);
        return;
    }

    const float k = smooth_;
    const float msToSamples = static_cast<float>(sampleRate_ / 1000.0);
    std::array<float, 8> in{};
    std::array<float, 8> out{};

    for (int i = 0; i < n; ++i)
    {
        preDelay_ += k * (preDelayTarget_ - preDelay_);
        size_ += k * (sizeTarget_ - size_);
        inGain_ += k * (inGainTarget_ - inGain_);
        early_ += k * (earlyTarget_ - early_);
        late_ += k * (lateTarget_ - late_);
        width_ += k * (widthTarget_ - width_);
        mix_ += k * (mixTarget_ - mix_);

        const float xL = channels[0][i];
        const float xR = channels_ == 2 ? channels[1][i] : xL;
        pre_[0].write(xL * inGain_);
        pre_[1].write(xR * inGain_);

        // Early reflections, alternately from the same and the opposite channel.
        float eL = 0.0f;
        float eR = 0.0f;
        for (int e = 0; e < kEarlyTaps; ++e)
        {
            const auto s = static_cast<std::size_t>(e);
            const float d = preDelay_ + kEarlyMs[s] * msToSamples * size_;
            const float a = pre_[0].readLinear(d);
            const float b = pre_[1].readLinear(d);
            const bool cross = (e & 1) != 0;
            eL += kEarlyGain[s] * (cross ? b : a);
            eR += kEarlyGain[s] * (cross ? a : b);
        }

        const float sL = diffuser_[0].process(pre_[0].readLinear(preDelay_));
        const float sR = diffuser_[1].process(pre_[1].readLinear(preDelay_));
        for (std::size_t j = 0; j < 8; ++j)
            in[j] = kInGain * (kInL[j] * sL + kInR[j] * sR);
        fdn_.tick(in.data(), out.data());
        float lL = 0.0f;
        float lR = 0.0f;
        for (std::size_t j = 0; j < 8; ++j)
        {
            lL += kOutL[j] * out[j];
            lR += kOutR[j] * out[j];
        }

        const float wL = 0.5f * early_ * eL + kOutGain * late_ * lL;
        const float wR = 0.5f * early_ * eR + kOutGain * late_ * lR;
        const float mid = 0.5f * (wL + wR);
        const float side = 0.5f * (wL - wR) * width_;
        if (channels_ == 2)
        {
            channels[0][i] = xL + mix_ * (mid + side - xL);
            channels[1][i] = xR + mix_ * (mid - side - xR);
        }
        else
        {
            channels[0][i] = xL + mix_ * (mid - xL);
        }
    }

    fdn_.flush();
    if (!blockFinite(channels, channels_, n))
    {
        reset();
        zero(channels, channels_, n);
    }
}

int ReverbProcessor::tailSamples() const noexcept
{
    if (freeze_)
        return clampTail(60.0 * sampleRate_, sampleRate_);
    const double seconds = static_cast<double>(rt60_) + (static_cast<double>(kMaxPreDelayMs) + 150.0) * 1e-3;
    return clampTail(seconds * sampleRate_ + static_cast<double>(preDelayTarget_), sampleRate_);
}

} // namespace alfa::dsp
