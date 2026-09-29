#pragma once

// Shared interface for the six DSP primitives. A Voice owns one of each and drives them hop by
// hop; nothing here is virtual, the voice calls each primitive type directly.
//
// Lifecycle, all on the audio thread except prepare():
//   prepare(spec)      allocate everything; may be called again when the sample rate changes
//   reset()            forget all state (called at note-on, before the first beginHop)
//   beginHop(ctx)      a new hop starts; ctx.prev/ctx.cur are frames j-1 and j
//   render(out, n)     overwrite out[0..n) with the next n samples of the hop, unscaled by the
//                      voice mix gain; the render calls between two beginHop calls add up to
//                      exactly spec.hop samples
//   skipHop(ctx)       called INSTEAD of beginHop + render when the primitive's mix is 0 in
//                      both frames of the hop: advance time-dependent state (phases, read
//                      positions) cheaply as if the hop had rendered; output nothing
//
// Each hop is therefore either beginHop(ctx) followed by render calls, or one skipHop(ctx).
//
// Every function after prepare() is noexcept, allocation-free and lock-free.

#include "alfa/dsp/ControlFrame.h"

namespace alfa::dsp
{

struct PrepareSpec
{
    double sampleRate = 48000.0;
    int hop = 192;  ///< hopSamples(sampleRate)
    TierLimits limits = limitsFor(Tier::Studio);
};

struct HopContext
{
    const VoiceFrame* prev = nullptr;  ///< frame j-1 (equal to cur on the first hop)
    const VoiceFrame* cur = nullptr;   ///< frame j
    float f0Start = 440.0f;            ///< fundamental at the start of the hop, Hz
    float f0End = 440.0f;              ///< fundamental at the end of the hop, Hz (linear across)
    bool firstHop = false;             ///< true on the note-on hop
};

/// Grain-source bank, owned by the caller (normally loaded from the model folder). Buffers are
/// contiguous: buffer i starts at data + i * length.
struct GrainBank
{
    const float* data = nullptr;
    int count = 0;
    int length = kGrainSourceLength;
};

/// A mono sample for SAMPLER, owned by the caller.
struct SampleData
{
    const float* data = nullptr;
    long long length = 0;
    double sampleRate = 48000.0;
    double rootHz = 261.6255653f;  ///< pitch the sample was recorded at (C4 by default)
    long long loopStart = -1;      ///< loop [loopStart, loopEnd) when both are valid
    long long loopEnd = -1;
};

} // namespace alfa::dsp
