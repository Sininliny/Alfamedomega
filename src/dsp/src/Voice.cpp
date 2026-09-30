#include "alfa/dsp/Voice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace alfa::dsp
{
namespace
{
constexpr float kMinAttackTau = 1.0e-4f;
constexpr float kSilenceThreshold = 1.0e-5f; // -100 dBFS peak
constexpr int kSilentHopsToEnd = 25;         // 100 ms after release
constexpr double kMaxReleaseSeconds = 30.0;
} // namespace

Voice::Voice()
    : latest_(std::make_unique<VoiceFrame>())
    , frameA_(std::make_unique<VoiceFrame>())
    , frameB_(std::make_unique<VoiceFrame>())
{
    prev_ = frameA_.get();
    cur_ = frameB_.get();
}

Voice::~Voice() = default;

void Voice::prepare(const PrepareSpec& spec, std::uint32_t seed)
{
    spec_ = spec;
    hop_ = spec.hop;
    seed_ = seed;
    osc_.prepare(spec);
    partials_.prepare(spec);
    fm_.prepare(spec);
    noise_.prepare(spec);
    noise_.setSeed(seed);
    grain_.prepare(spec);
    sampler_.prepare(spec);

    ProcessSpec ps;
    ps.sampleRate = spec.sampleRate;
    ps.channels = 1;
    ps.limits = spec.limits;
    slots_.prepare(ps, true, hop_);
    mod_.prepare(spec.sampleRate);
    mod_.setSeed(seed);
    active_ = false;
}

void Voice::setGrainBank(const GrainBank* bank) noexcept
{
    grain_.setBank(bank);
}

void Voice::setSample(const SampleData* sample) noexcept
{
    sampler_.setSample(sample);
}

void Voice::start(int note, float velocity, float f0Hz, const VoiceFrame* firstFrame) noexcept
{
    note_ = note;
    velocity_ = velocity;
    f0Target_ = f0Hz;
    f0Last_ = f0Hz;
    f0Now_ = f0Hz;
    active_ = true;
    releasing_ = false;
    age_ = 0;
    releaseAge_ = 0;
    silentHops_ = 0;
    hopPeak_ = 0.0f;
    firstHop_ = true;
    hopPos_ = hop_; // the first process call starts hop 0 immediately
    haveFrame_ = firstFrame != nullptr;
    if (firstFrame != nullptr)
        *latest_ = *firstFrame;

    osc_.reset();
    partials_.reset();
    fm_.reset();
    noise_.reset();
    grain_.reset();
    sampler_.reset();
    slots_.reset();
    // Same seed every note, so a voice's random modulation is reproducible (golden vectors).
    mod_.setSeed(seed_);
    mod_.noteOn(velocity, note);
    dest_.fill(0.0f);
}

void Voice::release() noexcept
{
    if (!releasing_)
        mod_.noteOff();
    releasing_ = true;
}

void Voice::setFrame(const VoiceFrame& frame) noexcept
{
    std::memcpy(static_cast<void*>(latest_.get()), &frame, sizeof(VoiceFrame));
    haveFrame_ = true;
}

void Voice::beginHop() noexcept
{
    // Frame j-1 <- frame j, frame j <- latest.
    std::swap(prev_, cur_);
    std::memcpy(static_cast<void*>(cur_), latest_.get(), sizeof(VoiceFrame));
    if (firstHop_)
        std::memcpy(static_cast<void*>(prev_), cur_, sizeof(VoiceFrame));

    // Pitch modulation applies per hop: the matrix value from the last block sets this hop's
    // end f0, and f0 ramps to it from the previous hop's end.
    const float pitch = std::isfinite(dest_[moddest::pitch]) ? dest_[moddest::pitch] : 0.0f;
    const float f0End = f0Target_ * std::exp2(pitch / 12.0f);

    HopContext ctx;
    ctx.prev = prev_;
    ctx.cur = cur_;
    ctx.f0Start = firstHop_ ? f0End : f0Last_;
    ctx.f0End = f0End;
    ctx.firstHop = firstHop_;
    f0Last_ = f0End;
    f0Now_ = f0End;

    mod_.update(cur_->mod);

    if (firstHop_)
    {
        const float tau = std::max(cur_->attackTau, kMinAttackTau);
        attackDecay_ = static_cast<float>(std::exp(-1.0 / (static_cast<double>(tau) * spec_.sampleRate)));
        attackEnv_ = 1.0f;
        for (int p = 0; p < kPrimitiveCount; ++p)
            lastMix_[static_cast<std::size_t>(p)] = cur_->mix[static_cast<std::size_t>(p)];
        for (int k = 0; k < kMaxVoiceSlots; ++k)
            lastOut_[static_cast<std::size_t>(k)] = cur_->slots[static_cast<std::size_t>(k)].outGain;
        lastDry_ = cur_->dryGain;
        lastPan_ = cur_->pan;
    }

    slots_.beginHop(cur_->slots.data());

    // A primitive renders when anything can hear it this hop: its mix gain, a running slot's
    // input weight, or a modulation route onto its mix gain.
    const int routes = std::clamp(spec_.limits.modRoutes, 0, kMaxModRoutes);
    for (int p = 0; p < kPrimitiveCount; ++p)
    {
        const auto i = static_cast<std::size_t>(p);
        bool heard = prev_->mix[i] != 0.0f || cur_->mix[i] != 0.0f;
        for (int k = 0; k < slots_.slotCount() && !heard; ++k)
        {
            const auto& a = prev_->slots[static_cast<std::size_t>(k)];
            const auto& b = cur_->slots[static_cast<std::size_t>(k)];
            const bool running =
                familyFromFloat(a.family) != Family::Off || familyFromFloat(b.family) != Family::Off;
            heard = running && (a.input[i] != 0.0f || b.input[i] != 0.0f);
        }
        for (int r = 0; r < routes && !heard; ++r)
        {
            const auto u = static_cast<std::size_t>(r);
            const float d = cur_->mod.routeDestination[u];
            heard = cur_->mod.routeDepth[u] != 0.0f && d > static_cast<float>(moddest::mix + p) - 0.5f &&
                    d < static_cast<float>(moddest::mix + p) + 0.5f;
        }
        rendering_[i] = heard;
    }

    const auto dispatch = [&](auto& prim, Primitive p)
    {
        if (rendering_[static_cast<std::size_t>(p)])
            prim.beginHop(ctx);
        else
            prim.skipHop(ctx);
    };
    dispatch(osc_, Primitive::Osc);
    dispatch(partials_, Primitive::Partials);
    dispatch(fm_, Primitive::Fm);
    dispatch(noise_, Primitive::Noise);
    dispatch(grain_, Primitive::Grain);
    dispatch(sampler_, Primitive::Sampler);

    firstHop_ = false;
    hopPos_ = 0;
}

namespace
{
/// Equal-power pan, normalised so the centre is unity on both sides (hard left is +3 dB left).
void panGains(float pan, float& left, float& right) noexcept
{
    const float theta = (std::clamp(pan, -1.0f, 1.0f) + 1.0f) * 0.25f * 3.14159265358979f;
    left = 1.41421356f * std::cos(theta);
    right = 1.41421356f * std::sin(theta);
}
} // namespace

void Voice::renderBlock(float* left, float* right, int n) noexcept
{
    // Values at the block's end, so the per-sample ramps land exactly on frame j at the hop end.
    const float t = static_cast<float>(hopPos_ + n) / static_cast<float>(hop_);
    const int slotCount = slots_.slotCount();

    // 1. Base values at this block, interpolated between frames j-1 and j.
    dest_.fill(0.0f);
    for (int k = 0; k < slotCount; ++k)
    {
        const auto u = static_cast<std::size_t>(k);
        interpolateSlot(prev_->slots[u], cur_->slots[u], t, blocks_[u]);
        std::copy(blocks_[u].params.begin(), blocks_[u].params.end(),
                  dest_.begin() + moddest::slotParam + k * kSlotParams);
        dest_[static_cast<std::size_t>(moddest::slotOutGain + k)] = blocks_[u].outGain;
    }
    for (int p = 0; p < kPrimitiveCount; ++p)
    {
        const auto u = static_cast<std::size_t>(p);
        dest_[static_cast<std::size_t>(moddest::mix + p)] =
            prev_->mix[u] + t * (cur_->mix[u] - prev_->mix[u]);
    }
    dest_[moddest::pan] = prev_->pan + t * (cur_->pan - prev_->pan);
    dest_[moddest::dryGain] = prev_->dryGain + t * (cur_->dryGain - prev_->dryGain);

    // 2. Modulation on top (clamps and sanitises every destination).
    ModMatrix::apply(cur_->mod, spec_.limits.modRoutes, mod_, dest_.data());
    for (int k = 0; k < slotCount; ++k)
    {
        const auto u = static_cast<std::size_t>(k);
        std::copy_n(dest_.begin() + moddest::slotParam + k * kSlotParams, kSlotParams,
                    blocks_[u].params.begin());
        blocks_[u].outGain = dest_[static_cast<std::size_t>(moddest::slotOutGain + k)];
    }

    // 3. Attack envelope and primitives.
    float e = attackEnv_;
    for (int i = 0; i < n; ++i)
    {
        env_[static_cast<std::size_t>(i)] = 1.0f - e;
        e *= attackDecay_;
    }
    attackEnv_ = e < 1.0e-9f ? 0.0f : e;

    const auto renderPrim = [&](auto& prim, Primitive p)
    {
        const auto i = static_cast<std::size_t>(p);
        float* buf = prim_[i].data();
        if (!rendering_[i])
        {
            std::fill_n(buf, n, 0.0f);
            return;
        }
        prim.render(buf, n);
        for (int s = 0; s < n; ++s)
            buf[s] *= env_[static_cast<std::size_t>(s)];
    };
    renderPrim(osc_, Primitive::Osc);
    renderPrim(partials_, Primitive::Partials);
    renderPrim(fm_, Primitive::Fm);
    renderPrim(noise_, Primitive::Noise);
    renderPrim(grain_, Primitive::Grain);
    renderPrim(sampler_, Primitive::Sampler);

    const float invN = 1.0f / static_cast<float>(n);
    std::fill_n(dry_.data(), n, 0.0f);
    for (int p = 0; p < kPrimitiveCount; ++p)
    {
        const auto u = static_cast<std::size_t>(p);
        const float g1 = dest_[static_cast<std::size_t>(moddest::mix + p)];
        const float g0 = lastMix_[u];
        lastMix_[u] = g1;
        if (!rendering_[u] || (g0 == 0.0f && g1 == 0.0f))
            continue;
        const float dg = (g1 - g0) * invN;
        const float* src = prim_[u].data();
        for (int s = 0; s < n; ++s)
            dry_[static_cast<std::size_t>(s)] += src[s] * (g0 + dg * static_cast<float>(s + 1));
    }

    // 4. Slots.
    const float* sources[kPrimitiveCount];
    for (int p = 0; p < kPrimitiveCount; ++p)
        sources[p] = prim_[static_cast<std::size_t>(p)].data();
    ControlContext cc;
    cc.f0 = f0Now_;
    cc.velocity = velocity_;
    const bool anySlot = slotCount > 0 && slots_.process(sources, blocks_.data(), cc, n);

    // 5. Dry + slot outputs, then pan.
    {
        const float g1 = dest_[moddest::dryGain];
        const float dg = (g1 - lastDry_) * invN;
        for (int s = 0; s < n; ++s)
            mono_[static_cast<std::size_t>(s)] =
                dry_[static_cast<std::size_t>(s)] * (lastDry_ + dg * static_cast<float>(s + 1));
        lastDry_ = g1;
    }
    for (int k = 0; k < slotCount; ++k)
    {
        const auto u = static_cast<std::size_t>(k);
        const float g1 = blocks_[u].outGain;
        const float g0 = lastOut_[u];
        lastOut_[u] = g1;
        if (!anySlot || !slots_.ran(k) || (g0 == 0.0f && g1 == 0.0f))
            continue;
        const float dg = (g1 - g0) * invN;
        const float* out = slots_.output(k, 0);
        for (int s = 0; s < n; ++s)
            mono_[static_cast<std::size_t>(s)] += out[s] * (g0 + dg * static_cast<float>(s + 1));
    }

    float l0, r0, l1, r1;
    panGains(lastPan_, l0, r0);
    panGains(dest_[moddest::pan], l1, r1);
    lastPan_ = dest_[moddest::pan];
    const float dl = (l1 - l0) * invN;
    const float dr = (r1 - r0) * invN;
    float peak = hopPeak_;
    for (int s = 0; s < n; ++s)
    {
        const float y = mono_[static_cast<std::size_t>(s)];
        const float k = static_cast<float>(s + 1);
        left[s] += y * (l0 + dl * k);
        if (right != nullptr)
            right[s] += y * (r0 + dr * k);
        peak = std::max(peak, std::fabs(y));
    }
    hopPeak_ = peak;

    // 6. Modulation sources advance over this block (the follower hears the primitive sum).
    mod_.advance(dry_.data(), n);
}

bool Voice::primitiveActive(int p) const noexcept
{
    const auto i = static_cast<std::size_t>(p);
    return rendering_[i] || latest_->mix[i] != 0.0f;
}

Family Voice::requestedFamily(int slot) const noexcept
{
    return slots_.resolveFamily(slot, latest_->slots[static_cast<std::size_t>(slot)]);
}

bool Voice::finished() const noexcept
{
    const auto zero = [](float g)
    {
        return g == 0.0f;
    };
    const bool noSources = std::all_of(cur_->mix.begin(), cur_->mix.end(), zero) &&
                           std::all_of(latest_->mix.begin(), latest_->mix.end(), zero);
    bool noSlots = true;
    for (int k = 0; k < slots_.slotCount(); ++k)
        noSlots = noSlots &&
                  familyFromFloat(cur_->slots[static_cast<std::size_t>(k)].family) == Family::Off &&
                  familyFromFloat(latest_->slots[static_cast<std::size_t>(k)].family) == Family::Off;
    const bool tooLong = static_cast<double>(releaseAge_) > kMaxReleaseSeconds * spec_.sampleRate;
    // Slots can ring after their input stops (resonators, waveguides), so without them the
    // network's zero gains end the voice at once; with them only measured silence does.
    return (noSources && noSlots) || silentHops_ >= kSilentHopsToEnd || tooLong;
}

void Voice::process(float* left, float* right, int n) noexcept
{
    if (!active_)
        return;
    int done = 0;
    while (done < n)
    {
        if (hopPos_ >= hop_)
        {
            if (!haveFrame_)
                return; // waiting for the note-on frame
            if (!firstHop_ && releasing_)
            {
                silentHops_ = hopPeak_ < kSilenceThreshold ? silentHops_ + 1 : 0;
                if (finished())
                {
                    active_ = false;
                    return;
                }
            }
            hopPeak_ = 0.0f;
            beginHop();
        }
        const int chunk = std::min({n - done, hop_ - hopPos_, kControlBlock});
        renderBlock(left + done, right != nullptr ? right + done : nullptr, chunk);
        done += chunk;
        hopPos_ += chunk;
        age_ += chunk;
        if (releasing_)
            releaseAge_ += chunk;
    }
}

} // namespace alfa::dsp
