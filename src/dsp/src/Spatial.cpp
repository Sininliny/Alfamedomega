#include "alfa/dsp/Spatial.h"

#include "alfa/dsp/detail/Denormals.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp
{

namespace
{
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

float dbToGain(float db) noexcept
{
    return std::pow(10.0f, db / 20.0f);
}

bool allFinite(const float* x, int n) noexcept
{
    // x - x is 0 for finite x and NaN otherwise; one test per block instead of per sample.
    float acc = 0.0f;
    for (int i = 0; i < n; ++i)
        acc += x[i] - x[i];
    return acc == 0.0f;
}
} // namespace

// StereoPanRenderer ----------------------------------------------------------------------------

void StereoPanRenderer::prepare(double sampleRate, int channels)
{
    sampleRate_ = sampleRate;
    channels_ = std::clamp(channels, 1, 2);
    for (auto& f : shelfLp_)
        f.setup(kShelfHz, sampleRate_);
    reset();
}

void StereoPanRenderer::reset() noexcept
{
    const StereoMatrix id = StereoMatrix::identity();
    matrix_[0].reset(id.ll);
    matrix_[1].reset(id.rl);
    matrix_[2].reset(id.lr);
    matrix_[3].reset(id.rr);
    level_.reset(1.0f);
    shelf_.reset(0.0f);
    for (auto& f : shelfLp_)
        f.reset();
    first_ = true;
}

void StereoPanRenderer::setPosition(const SpatialPosition& position) noexcept
{
    // The first position after a reset is taken at once, so a note does not start with a sweep.
    const int ramp = first_ ? 0 : kControlBlock;
    first_ = false;
    const StereoMatrix m = StereoMatrix::stereoPan(position.azimuth, position.law);
    matrix_[0].setTarget(m.ll, ramp);
    matrix_[1].setTarget(m.rl, ramp);
    matrix_[2].setTarget(m.lr, ramp);
    matrix_[3].setTarget(m.rr, ramp);
    const float d = std::clamp(position.distance, 0.0f, 1.0f);
    level_.setTarget(d > 0.0f ? dbToGain(kFarLevelDb * d) : 1.0f, ramp);
    shelf_.setTarget(d > 0.0f ? dbToGain(kFarShelfDb * d) - 1.0f : 0.0f, ramp);
}

void StereoPanRenderer::process(float* const* channels, int n) noexcept
{
    float* l = channels[0];
    float* r = channels_ > 1 ? channels[1] : nullptr;
    for (int i = 0; i < n; ++i)
    {
        const float ll = matrix_[0].next();
        const float rl = matrix_[1].next();
        const float lr = matrix_[2].next();
        const float rr = matrix_[3].next();
        const float level = level_.next();
        const float shelf = shelf_.next();
        if (r != nullptr)
        {
            const float x = l[i];
            const float y = r[i];
            l[i] = ll * x + rl * y;
            r[i] = lr * x + rr * y;
        }
        // First-order high shelf: x + (g - 1) * highpass(x). At distance 0 the shelf term is
        // exactly 0 and the level exactly 1, so the stage is transparent.
        const float xl = l[i];
        l[i] = level * (xl + shelf * (xl - shelfLp_[0].tick(xl)));
        if (r != nullptr)
        {
            const float xr = r[i];
            r[i] = level * (xr + shelf * (xr - shelfLp_[1].tick(xr)));
        }
    }
}

// SpatialProcessor -----------------------------------------------------------------------------

const ParamTable& SpatialProcessor::params() noexcept
{
    static const ParamTable table = []
    {
        ParamTable t{};
        t[0] = {"pan", -1.0f, 1.0f, Curve::Linear, 0.5f, ""};
        t[1] = {"pan_law", 0.0f, 2.0f, Curve::Choice, 0.0f, ""};
        t[2] = {"width", 0.0f, 2.0f, Curve::Linear, 0.5f, ""};
        t[3] = {"ms_balance", -1.0f, 1.0f, Curve::Linear, 0.5f, ""};
        t[4] = {"haas", 0.0f, static_cast<float>(kMaxHaasMs), Curve::Linear, 0.0f, "ms"};
        t[5] = {"haas_side", 0.0f, 1.0f, Curve::Toggle, 0.0f, ""};
        t[6] = {"rotation", -45.0f, 45.0f, Curve::Linear, 0.5f, "deg"};
        t[7] = {"mono_below", 0.0f, 500.0f, Curve::Linear, 0.0f, "Hz"};
        t[8] = {"distance", 0.0f, 1.0f, Curve::Linear, 0.0f, ""};
        t[9] = {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""};
        return t;
    }();
    return table;
}

void SpatialProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    const int maxDelay = static_cast<int>(std::ceil(kMaxHaasMs * 1e-3 * sampleRate_)) + 4;
    int size = 1;
    while (size < maxDelay)
        size <<= 1;
    delaySize_ = size;
    delayMask_ = size - 1;
    delay_.assign(static_cast<std::size_t>(2 * size), 0.0f);
    delaySmooth_ = detail::smoothingCoeff(0.02, sampleRate_);
    monoSmooth_ = detail::smoothingCoeff(0.03, sampleRate_, kControlBlock);
    renderer_.prepare(sampleRate_, channels_);
    reset();
}

void SpatialProcessor::clearState() noexcept
{
    std::fill(delay_.begin(), delay_.end(), 0.0f);
    writePos_ = 0;
    monoMid_.reset();
    for (auto& f : monoSide_)
        f.reset();
    renderer_.reset();
}

void SpatialProcessor::reset() noexcept
{
    clearState();
    for (auto& m : image_)
        m.reset(0.0f);
    image_[0].reset(1.0f);
    image_[3].reset(1.0f);
    delaySamples_ = {0.0f, 0.0f};
    delayTarget_ = {0.0f, 0.0f};
    monoHz_ = 0.0f;
    monoTargetHz_ = 0.0f;
    monoAmount_.reset(0.0f);
    mix_.reset(1.0f);
    position_ = {};
    first_ = true;
}

void SpatialProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const ParamTable& t = params();
    const float pan = mapParam(t[0], p[0]);
    const auto law = static_cast<PanLaw>(static_cast<int>(mapParam(t[1], p[1])));
    const float width = mapParam(t[2], p[2]);
    const float balance = mapParam(t[3], p[3]);
    const float haasMs = mapParam(t[4], p[4]);
    const bool haasRight = mapParam(t[5], p[5]) >= 0.5f;
    const float rotation = mapParam(t[6], p[6]) * kDegToRad;
    const float monoHz = mapParam(t[7], p[7]);
    const float distance = mapParam(t[8], p[8]);
    const float mix = mapParam(t[9], p[9]);

    const int ramp = first_ ? 0 : kControlBlock;

    // Width and M/S balance are both mid/side gains; balance only ever attenuates.
    const float gm = std::min(1.0f, 1.0f - balance);
    const float gs = std::min(1.0f, 1.0f + balance) * width;
    const StereoMatrix image = StereoMatrix::rotation(rotation).after(StereoMatrix::midSide(gm, gs));
    image_[0].setTarget(image.ll, ramp);
    image_[1].setTarget(image.rl, ramp);
    image_[2].setTarget(image.lr, ramp);
    image_[3].setTarget(image.rr, ramp);

    const float haas = haasMs * 1e-3f * static_cast<float>(sampleRate_);
    delayTarget_[0] = haasRight ? 0.0f : haas;
    delayTarget_[1] = haasRight ? haas : 0.0f;
    if (first_)
        delaySamples_ = delayTarget_;

    // Mono-maker: below kMonoOffHz it fades out and then bypasses completely (bit-exact).
    monoTargetHz_ = monoHz >= kMonoOffHz ? monoHz : 0.0f;
    const bool active = monoAmount_.value() > 0.0f || monoAmount_.ramping();
    if (monoTargetHz_ > 0.0f)
    {
        if (!active || first_)
            monoHz_ = monoTargetHz_; // nothing audible yet, so no glide needed
        else
            monoHz_ += (monoTargetHz_ - monoHz_) * monoSmooth_;
        monoAmount_.setTarget(1.0f, ramp);
    }
    else
    {
        monoAmount_.setTarget(0.0f, ramp);
    }
    if (monoHz_ > 0.0f)
    {
        monoMid_.setup(monoHz_, 0.70710678, sampleRate_);
        for (auto& f : monoSide_)
            f.setup(monoHz_, 0.70710678, sampleRate_);
    }

    position_.azimuth = pan;
    position_.distance = distance;
    position_.law = law;
    renderer_.setPosition(position_);
    mix_.setTarget(mix, ramp);
    first_ = false;
}

void SpatialProcessor::processStereo(float* l, float* r, int n) noexcept
{
    const bool mono = monoAmount_.value() > 0.0f || monoAmount_.ramping();
    const float k2 = 2.0f * monoMid_.k();
    float* dl = delay_.data();
    float* dr = delay_.data() + delaySize_;
    for (int i = 0; i < n; ++i)
    {
        float x = l[i];
        float y = r[i];
        if (mono)
        {
            float mid = 0.5f * (x + y);
            float side = 0.5f * (x - y);
            const float a = monoAmount_.next();
            // LR4 split: LP + HP of the side sums to the same 2nd-order all-pass the mid goes
            // through, so above the crossover L and R keep their relative phase.
            const float ap = mid - k2 * monoMid_.tick(mid).bp;
            const float hp = monoSide_[1].tick(monoSide_[0].tick(side).hp).hp;
            mid += a * (ap - mid);
            side += a * (hp - side);
            x = mid + side;
            y = mid - side;
        }

        const float ll = image_[0].next();
        const float rl = image_[1].next();
        const float lr = image_[2].next();
        const float rr = image_[3].next();
        const float il = ll * x + rl * y;
        const float ir = lr * x + rr * y;

        dl[writePos_] = il;
        dr[writePos_] = ir;
        float out[2];
        for (int c = 0; c < 2; ++c)
        {
            float& d = delaySamples_[static_cast<std::size_t>(c)];
            const float target = delayTarget_[static_cast<std::size_t>(c)];
            d += (target - d) * delaySmooth_;
            if (std::abs(target - d) < 1e-3f)
                d = target; // settle exactly, so an integer delay is sample-accurate
            const float* line = c == 0 ? dl : dr;
            const auto whole = static_cast<int>(d);
            const float frac = d - static_cast<float>(whole);
            const float a = line[(writePos_ - whole) & delayMask_];
            const float b = line[(writePos_ - whole - 1) & delayMask_];
            out[c] = a + frac * (b - a);
        }
        writePos_ = (writePos_ + 1) & delayMask_;
        l[i] = out[0];
        r[i] = out[1];
    }
    if (mono && !monoAmount_.ramping() && monoAmount_.value() <= 0.0f)
    {
        monoMid_.reset();
        for (auto& f : monoSide_)
            f.reset();
    }
}

void SpatialProcessor::processChunk(float* const* ch, int n) noexcept
{
    float* l = ch[0];
    float* r = channels_ > 1 ? ch[1] : nullptr;
    std::copy_n(l, n, dryL_.data());
    if (r != nullptr)
    {
        std::copy_n(r, n, dryR_.data());
        processStereo(l, r, n);
    }
    renderer_.process(ch, n);
    for (int i = 0; i < n; ++i)
    {
        const float m = mix_.next();
        const float dryGain = 1.0f - m;
        l[i] = dryGain * dryL_[static_cast<std::size_t>(i)] + m * l[i];
        if (r != nullptr)
            r[i] = dryGain * dryR_[static_cast<std::size_t>(i)] + m * r[i];
    }
}

void SpatialProcessor::process(float* const* ch, int n) noexcept
{
    detail::ScopedNoDenormals noDenormals;
    for (int done = 0; done < n;)
    {
        const int k = std::min(kControlBlock, n - done);
        float* chunk[2] = {ch[0] + done, channels_ > 1 ? ch[1] + done : nullptr};
        processChunk(chunk, k);
        bool ok = allFinite(chunk[0], k);
        if (chunk[1] != nullptr)
            ok = ok && allFinite(chunk[1], k);
        if (!ok)
        {
            // A non-finite input or state: start clean and output silence for this chunk.
            clearState();
            std::fill_n(chunk[0], k, 0.0f);
            if (chunk[1] != nullptr)
                std::fill_n(chunk[1], k, 0.0f);
        }
        done += k;
    }
}

int SpatialProcessor::tailSamples() const noexcept
{
    const float d = std::max({delaySamples_[0], delaySamples_[1], delayTarget_[0], delayTarget_[1]});
    // Plus a little for the filters to ring down.
    return static_cast<int>(std::ceil(d)) + static_cast<int>(0.05 * sampleRate_);
}

} // namespace alfa::dsp
