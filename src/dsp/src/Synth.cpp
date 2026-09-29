#include "alfa/dsp/Synth.h"

#include "alfa/dsp/Convolution.h"
#include "alfa/dsp/Voice.h"
#include "alfa/dsp/detail/Denormals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace alfa::dsp
{

Synth::Synth()
    : latestI_(std::make_unique<InstanceFrame>())
    , frameIA_(std::make_unique<InstanceFrame>())
    , frameIB_(std::make_unique<InstanceFrame>())
{
    prevI_ = frameIA_.get();
    curI_ = frameIB_.get();
}

Synth::~Synth() = default;

void Synth::prepare(double sampleRate, int maxBlock, Tier tier)
{
    spec_.sampleRate = sampleRate;
    spec_.hop = hopSamples(sampleRate);
    spec_.limits = limitsFor(tier);

    voices_.clear();
    voices_.reserve(static_cast<std::size_t>(spec_.limits.polyphony));
    for (int v = 0; v < spec_.limits.polyphony; ++v)
    {
        auto voice = std::make_unique<Voice>();
        voice->prepare(spec_, 0x1234567u + 0x9E3779B9u * static_cast<std::uint32_t>(v + 1));
        voices_.push_back(std::move(voice));
    }
    startOrder_.assign(voices_.size(), 0);
    counter_ = 0;

    maxBlock_ = std::max(1, maxBlock);
    busL_.assign(static_cast<std::size_t>(maxBlock_), 0.0f);
    busR_.assign(static_cast<std::size_t>(maxBlock_), 0.0f);

    ProcessSpec ps;
    ps.sampleRate = sampleRate;
    ps.channels = 2;
    ps.limits = spec_.limits;
    instSlots_.prepare(ps, false, spec_.hop);
    instMod_.prepare(sampleRate);
    instMod_.setSeed(0xA1FAu);
    instMod_.reset();
    *prevI_ = *latestI_;
    *curI_ = *latestI_;
    instHopPos_ = spec_.hop; // the first process call starts an instance hop at once
    instFirstHop_ = true;
    instDest_.fill(0.0f);
    lastInstOut_.fill(0.0f);
}

void Synth::setGrainBank(const GrainBank* bank) noexcept
{
    for (auto& v : voices_)
        v->setGrainBank(bank);
}

void Synth::setSample(const SampleData* sample) noexcept
{
    for (auto& v : voices_)
        v->setSample(sample);
}

float Synth::f0For(int note) const noexcept
{
    return static_cast<float>(440.0 *
                              std::pow(2.0, (note - 69 + static_cast<double>(bendSemitones_)) / 12.0));
}

int Synth::noteOn(int note, float velocity, const VoiceFrame* firstFrame) noexcept
{
    if (voices_.empty())
        return -1;
    int chosen = -1;
    // Free voice first, then the oldest released voice, then the oldest voice.
    for (std::size_t i = 0; i < voices_.size() && chosen < 0; ++i)
        if (!voices_[i]->active())
            chosen = static_cast<int>(i);
    for (int pass = 0; pass < 2 && chosen < 0; ++pass)
    {
        std::uint64_t oldest = UINT64_MAX;
        for (std::size_t i = 0; i < voices_.size(); ++i)
        {
            if (pass == 0 && !voices_[i]->releasing())
                continue;
            if (startOrder_[i] < oldest)
            {
                oldest = startOrder_[i];
                chosen = static_cast<int>(i);
            }
        }
    }
    const auto c = static_cast<std::size_t>(chosen);
    voices_[c]->start(note, velocity, f0For(note), firstFrame);
    startOrder_[c] = ++counter_;
    return chosen;
}

void Synth::noteOff(int note) noexcept
{
    for (auto& v : voices_)
        if (v->active() && !v->releasing() && v->note() == note)
            v->release();
}

void Synth::allNotesOff() noexcept
{
    for (auto& v : voices_)
        if (v->active())
            v->release();
}

void Synth::killAll() noexcept
{
    for (auto& v : voices_)
        v->kill();
}

void Synth::setPitchBend(float semitones) noexcept
{
    bendSemitones_ = semitones;
    for (auto& v : voices_)
        if (v->active())
            v->setF0(f0For(v->note()));
}

void Synth::setFrame(int voice, const VoiceFrame& frame) noexcept
{
    if (voice >= 0 && voice < voiceCount())
        voices_[static_cast<std::size_t>(voice)]->setFrame(frame);
}

void Synth::publishIrs(IrBank& bank)
{
    for (int k = 0; k < instSlots_.slotCount(); ++k)
        if (auto* p = static_cast<ConvolutionProcessor*>(instSlots_.processor(k, Family::Convolution));
            p != nullptr)
            bank.publish(*p);
}

void Synth::collectIrs(IrBank& bank)
{
    std::uint64_t oldest = UINT64_MAX;
    for (int k = 0; k < instSlots_.slotCount(); ++k)
        if (const auto* p =
                static_cast<const ConvolutionProcessor*>(instSlots_.processor(k, Family::Convolution));
            p != nullptr)
            oldest = std::min(oldest, p->acknowledgedGeneration());
    if (oldest != UINT64_MAX)
        bank.collectBefore(oldest);
}

void Synth::setInstanceFrame(const InstanceFrame& frame) noexcept
{
    std::memcpy(static_cast<void*>(latestI_.get()), &frame, sizeof(InstanceFrame));
}

void Synth::process(float* left, float* right, int n) noexcept
{
    detail::ScopedNoDenormals noDenormals;
    if (voices_.empty())
    {
        std::fill_n(left, n, 0.0f);
        if (right != nullptr)
            std::fill_n(right, n, 0.0f);
        return;
    }
    for (int done = 0; done < n;)
    {
        const int chunk = std::min(n - done, maxBlock_);
        renderChunk(left + done, right != nullptr ? right + done : nullptr, chunk);
        done += chunk;
    }
}

void Synth::renderChunk(float* left, float* right, int n) noexcept
{
    std::fill_n(busL_.data(), n, 0.0f);
    std::fill_n(busR_.data(), n, 0.0f);
    for (auto& v : voices_)
        v->process(busL_.data(), busR_.data(), n);

    float* outR = right;
    for (int done = 0; done < n;)
    {
        if (instHopPos_ >= spec_.hop)
            beginInstanceHop();
        const int chunk = std::min({n - done, spec_.hop - instHopPos_, kControlBlock});
        instanceBlock(left, outR, done, chunk);
        done += chunk;
        instHopPos_ += chunk;
    }
}

void Synth::beginInstanceHop() noexcept
{
    std::swap(prevI_, curI_);
    std::memcpy(static_cast<void*>(curI_), latestI_.get(), sizeof(InstanceFrame));
    if (instFirstHop_)
    {
        std::memcpy(static_cast<void*>(prevI_), curI_, sizeof(InstanceFrame));
        for (int k = 0; k < kMaxInstanceSlots; ++k)
            lastInstOut_[static_cast<std::size_t>(k)] = curI_->slots[static_cast<std::size_t>(k)].outGain;
        lastDry_ = curI_->dryGain;
        lastMaster_ = curI_->masterGain;
        lastWidth_ = curI_->width;
    }
    instFirstHop_ = false;
    instMod_.update(curI_->mod);
    instSlots_.beginHop(curI_->slots.data());
    instHopPos_ = 0;
}

namespace
{
float finiteNonNeg(float v, float fallback) noexcept
{
    return std::isfinite(v) ? std::max(v, 0.0f) : fallback;
}

/// Output safety: exactly linear up to +6 dBFS, then a tanh knee that never passes +12 dBFS.
/// Stacked high-Q slots (a resonator fed a sustained harmonic, a filter at full resonance)
/// can legitimately reach hundreds; this keeps a bad frame from reaching speakers at that level.
/// The network trains against this same function, so it is part of the instrument.
float safety(float x) noexcept
{
    constexpr float knee = 2.0f;
    const float a = std::fabs(x);
    if (a <= knee)
        return x;
    const float y = knee + knee * std::tanh((a - knee) / knee);
    return std::copysign(y, x);
}
} // namespace

void Synth::instanceBlock(float* left, float* right, int offset, int n) noexcept
{
    const float t = static_cast<float>(instHopPos_ + n) / static_cast<float>(spec_.hop);
    const int slotCount = instSlots_.slotCount();
    float* bl = busL_.data() + offset;
    float* br = busR_.data() + offset;

    instDest_.fill(0.0f);
    for (int k = 0; k < slotCount; ++k)
    {
        const auto u = static_cast<std::size_t>(k);
        interpolateSlot(prevI_->slots[u], curI_->slots[u], t, instBlocks_[u]);
        std::copy(instBlocks_[u].params.begin(), instBlocks_[u].params.end(),
                  instDest_.begin() + moddest::slotParam + k * kSlotParams);
        instDest_[static_cast<std::size_t>(moddest::slotOutGain + k)] = instBlocks_[u].outGain;
    }
    instDest_[moddest::dryGain] = prevI_->dryGain + t * (curI_->dryGain - prevI_->dryGain);
    ModMatrix::apply(curI_->mod, spec_.limits.modRoutes, instMod_, instDest_.data());
    for (int k = 0; k < slotCount; ++k)
    {
        const auto u = static_cast<std::size_t>(k);
        std::copy_n(instDest_.begin() + moddest::slotParam + k * kSlotParams, kSlotParams,
                    instBlocks_[u].params.begin());
        instBlocks_[u].outGain = instDest_[static_cast<std::size_t>(moddest::slotOutGain + k)];
    }

    const float* sources[2] = {bl, br};
    ControlContext cc; // instance slots have no pitch
    const bool anySlot = slotCount > 0 && instSlots_.process(sources, instBlocks_.data(), cc, n);

    const float invN = 1.0f / static_cast<float>(n);
    const float dry1 = instDest_[moddest::dryGain];
    const float master1 =
        finiteNonNeg(prevI_->masterGain + t * (curI_->masterGain - prevI_->masterGain), 0.0f);
    const float width1 =
        std::clamp(finiteNonNeg(prevI_->width + t * (curI_->width - prevI_->width), 1.0f), 0.0f, 2.0f);
    const float dDry = (dry1 - lastDry_) * invN;
    const float dMaster = (master1 - lastMaster_) * invN;
    const float dWidth = (width1 - lastWidth_) * invN;

    // Bus times dry gain, in place (the slots already read the bus).
    for (int i = 0; i < n; ++i)
    {
        const float g = lastDry_ + dDry * static_cast<float>(i + 1);
        bl[i] *= g;
        br[i] *= g;
    }
    lastDry_ = dry1;
    for (int k = 0; k < slotCount; ++k)
    {
        const auto u = static_cast<std::size_t>(k);
        const float g1 = instBlocks_[u].outGain;
        const float g0 = lastInstOut_[u];
        lastInstOut_[u] = g1;
        if (!anySlot || !instSlots_.ran(k) || (g0 == 0.0f && g1 == 0.0f))
            continue;
        const float dg = (g1 - g0) * invN;
        const float* ol = instSlots_.output(k, 0);
        const float* orr = instSlots_.output(k, 1);
        for (int i = 0; i < n; ++i)
        {
            const float g = g0 + dg * static_cast<float>(i + 1);
            bl[i] += g * ol[i];
            br[i] += g * orr[i];
        }
    }

    for (int i = 0; i < n; ++i)
    {
        const float k = static_cast<float>(i + 1);
        const float m = lastMaster_ + dMaster * k;
        const float w = lastWidth_ + dWidth * k;
        const float mid = 0.5f * (bl[i] + br[i]);
        const float side = 0.5f * (bl[i] - br[i]) * w;
        if (right != nullptr)
        {
            left[offset + i] = safety(m * (mid + side));
            right[offset + i] = safety(m * (mid - side));
        }
        else
        {
            left[offset + i] = safety(m * mid);
        }
    }
    lastMaster_ = master1;
    lastWidth_ = width1;

    instMod_.advance(bl, n);
}

VoiceInfo Synth::voiceInfo(int voice) const noexcept
{
    VoiceInfo info;
    if (voice < 0 || voice >= voiceCount())
        return info;
    const auto& v = *voices_[static_cast<std::size_t>(voice)];
    info.active = v.active();
    info.releasing = v.releasing();
    info.note = v.note();
    info.velocity = v.velocity();
    info.samplesSinceNoteOn = v.samplesSinceNoteOn();
    info.samplesSinceRelease = v.samplesSinceRelease();
    return info;
}

} // namespace alfa::dsp
