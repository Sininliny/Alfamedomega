#include "alfa/dsp/SlotGraph.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace alfa::dsp
{

SlotGraph::SlotGraph() = default;
SlotGraph::~SlotGraph() = default;

void SlotGraph::prepare(const ProcessSpec& spec, bool voice, int hop)
{
    voice_ = voice;
    channels_ = spec.channels;
    numSources_ = voice ? kPrimitiveCount : 1;
    hop_ = std::max(1, hop);
    const int count = voice ? spec.limits.voiceSlots : spec.limits.instanceSlots;
    const auto block = static_cast<std::size_t>(channels_ * kControlBlock);

    slots_.clear();
    slots_.resize(static_cast<std::size_t>(std::clamp(count, 0, voice ? kMaxVoiceSlots : kMaxInstanceSlots)));
    for (auto& slot : slots_)
    {
        for (int f = 1; f < kFamilyCount; ++f)
        {
            const auto family = static_cast<Family>(f);
            if (voice && !allowedInVoice(family))
                continue;
            auto p = makeProcessor(family);
            if (p != nullptr)
                p->prepare(spec);
            slot.processors[static_cast<std::size_t>(f)] = std::move(p);
        }
        slot.out.assign(block, 0.0f);
        slot.in.assign(block, 0.0f);
        slot.old.assign(block, 0.0f);
    }
    reset();
}

void SlotGraph::reset() noexcept
{
    for (auto& slot : slots_)
    {
        for (auto& p : slot.processors)
            if (p != nullptr)
                p->reset();
        slot.family = Family::Off;
        slot.fadingOut = Family::Off;
        slot.fadePos = 0;
        slot.ran = false;
        std::fill(slot.out.begin(), slot.out.end(), 0.0f);
    }
}

void SlotGraph::beginHop(const SlotControls* current) noexcept
{
    for (std::size_t k = 0; k < slots_.size(); ++k)
    {
        auto& slot = slots_[k];
        Family next = familyFromFloat(current[k].family);
        if (slot.processors[static_cast<std::size_t>(next)] == nullptr)
            next = Family::Off;  // not allowed here (or Off)
        if (next == slot.family)
            continue;
        // Start the new family clean and fade the old one out across this hop. A change during
        // a fade drops the older family (its output is already mostly gone).
        slot.fadingOut = slot.family;
        slot.fadePos = 0;
        slot.family = next;
        if (auto& p = slot.processors[static_cast<std::size_t>(next)]; p != nullptr)
            p->reset();
    }
}

void SlotGraph::runFamily(Slot& slot, Family f, const float* params, const ControlContext& ctx, float* buffer,
                          int n) noexcept
{
    auto& p = slot.processors[static_cast<std::size_t>(f)];
    float* ch[2] = {buffer, buffer + kControlBlock};
    p->setParams(params, ctx);
    p->process(ch, n);

    // A processor that produced a non-finite sample is reset and silenced for this block.
    float sum = 0.0f;
    for (int c = 0; c < channels_; ++c)
        for (int i = 0; i < n; ++i)
            sum += ch[c][i] * 0.0f;  // NaN or Inf anywhere makes this NaN
    if (!(sum == 0.0f))
    {
        p->reset();
        for (int c = 0; c < channels_; ++c)
            std::fill_n(ch[c], n, 0.0f);
    }
}

bool SlotGraph::process(const float* const* sources, const SlotBlock* blocks, const ControlContext& ctx, int n) noexcept
{
    bool any = false;
    for (std::size_t k = 0; k < slots_.size(); ++k)
    {
        auto& slot = slots_[k];
        const SlotBlock& b = blocks[k];
        slot.ran = false;
        if (slot.family == Family::Off && slot.fadingOut == Family::Off)
            continue;

        // Input: sources, then earlier slots.
        for (int c = 0; c < channels_; ++c)
        {
            float* in = slot.in.data() + c * kControlBlock;
            std::fill_n(in, n, 0.0f);
            for (int s = 0; s < numSources_; ++s)
            {
                const float w = b.input[static_cast<std::size_t>(s)];
                if (w == 0.0f || !std::isfinite(w))
                    continue;
                const float* src = sources[s * channels_ + c];
                for (int i = 0; i < n; ++i)
                    in[i] += w * src[i];
            }
            for (std::size_t e = 0; e < k; ++e)
            {
                const float w = b.input[static_cast<std::size_t>(numSources_) + e];
                if (w == 0.0f || !std::isfinite(w) || !slots_[e].ran)
                    continue;
                const float* src = slots_[e].out.data() + c * kControlBlock;
                for (int i = 0; i < n; ++i)
                    in[i] += w * src[i];
            }
        }

        const auto copyIn = [&](float* dst) {
            for (int c = 0; c < channels_; ++c)
                std::copy_n(slot.in.data() + c * kControlBlock, n, dst + c * kControlBlock);
        };

        if (slot.family != Family::Off)
        {
            copyIn(slot.out.data());
            runFamily(slot, slot.family, b.params.data(), ctx, slot.out.data(), n);
        }
        else
        {
            for (int c = 0; c < channels_; ++c)
                std::fill_n(slot.out.data() + c * kControlBlock, n, 0.0f);
        }

        if (slot.fadingOut != Family::Off)
        {
            copyIn(slot.old.data());
            runFamily(slot, slot.fadingOut, b.params.data(), ctx, slot.old.data(), n);
            const float step = 1.0f / static_cast<float>(hop_);
            for (int c = 0; c < channels_; ++c)
            {
                float* o = slot.out.data() + c * kControlBlock;
                const float* old = slot.old.data() + c * kControlBlock;
                for (int i = 0; i < n; ++i)
                {
                    const float g = std::min(1.0f, static_cast<float>(slot.fadePos + i) * step);
                    o[i] = g * o[i] + (1.0f - g) * old[i];
                }
            }
            slot.fadePos += n;
            if (slot.fadePos >= hop_)
                slot.fadingOut = Family::Off;
        }

        slot.ran = true;
        any = true;
    }
    return any;
}

const float* SlotGraph::output(int slot, int channel) const noexcept
{
    return slots_[static_cast<std::size_t>(slot)].out.data() + channel * kControlBlock;
}

int SlotGraph::tailSamples() const noexcept
{
    int tail = 0;
    for (const auto& slot : slots_)
    {
        for (const Family f : {slot.family, slot.fadingOut})
        {
            if (f == Family::Off)
                continue;
            if (const auto& p = slot.processors[static_cast<std::size_t>(f)]; p != nullptr)
                tail = std::max(tail, p->tailSamples());
        }
    }
    return tail;
}

void interpolateSlot(const SlotControls& prev, const SlotControls& cur, float t, SlotBlock& out) noexcept
{
    out.family = familyFromFloat(cur.family);
    out.outGain = prev.outGain + t * (cur.outGain - prev.outGain);
    for (int i = 0; i < kSlotParams; ++i)
    {
        const auto u = static_cast<std::size_t>(i);
        out.params[u] = prev.params[u] + t * (cur.params[u] - prev.params[u]);
    }
    for (int i = 0; i < kSlotInputs; ++i)
    {
        const auto u = static_cast<std::size_t>(i);
        out.input[u] = prev.input[u] + t * (cur.input[u] - prev.input[u]);
    }
}

} // namespace alfa::dsp
