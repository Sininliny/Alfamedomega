#include "alfa/dsp/ControlFrame.h"

#include <cstddef>
#include <type_traits>

namespace alfa::dsp
{

static_assert(std::is_standard_layout_v<VoiceFrame>);
static_assert(std::is_trivially_copyable_v<VoiceFrame>);
static_assert(sizeof(VoiceFrame) % sizeof(float) == 0);
static_assert(offsetof(VoiceFrame, mix) / sizeof(float) == offsets::mix);
static_assert(offsetof(VoiceFrame, attackTau) / sizeof(float) == offsets::attackTau);
static_assert(offsetof(VoiceFrame, osc) / sizeof(float) == offsets::oscRe);
static_assert((offsetof(VoiceFrame, osc) + offsetof(OscControls, im)) / sizeof(float) == offsets::oscIm);
static_assert(offsetof(VoiceFrame, partials) / sizeof(float) == offsets::partialsRatio);
static_assert((offsetof(VoiceFrame, partials) + offsetof(PartialsControls, amp)) / sizeof(float) ==
              offsets::partialsAmp);
static_assert(offsetof(VoiceFrame, fm) / sizeof(float) == offsets::fmRatio);
static_assert((offsetof(VoiceFrame, fm) + offsetof(FmControls, level)) / sizeof(float) == offsets::fmLevel);
static_assert((offsetof(VoiceFrame, fm) + offsetof(FmControls, index)) / sizeof(float) == offsets::fmIndex);
static_assert(offsetof(VoiceFrame, noise) / sizeof(float) == offsets::noiseBand);
static_assert(offsetof(VoiceFrame, grain) / sizeof(float) == offsets::grainSource);
static_assert((offsetof(VoiceFrame, grain) + offsetof(GrainControls, weight)) / sizeof(float) ==
              offsets::grainWeight);
static_assert((offsetof(VoiceFrame, grain) + offsetof(GrainControls, position)) / sizeof(float) ==
              offsets::grainPosition);
static_assert((offsetof(VoiceFrame, grain) + offsetof(GrainControls, rate)) / sizeof(float) ==
              offsets::grainRate);
static_assert((offsetof(VoiceFrame, grain) + offsetof(GrainControls, gain)) / sizeof(float) ==
              offsets::grainGain);
static_assert(offsetof(VoiceFrame, sampler) / sizeof(float) == offsets::samplerRate);
static_assert((offsetof(VoiceFrame, sampler) + offsetof(SamplerControls, start)) / sizeof(float) ==
              offsets::samplerStart);
static_assert((offsetof(VoiceFrame, sampler) + offsetof(SamplerControls, gain)) / sizeof(float) ==
              offsets::samplerGain);
static_assert((offsetof(VoiceFrame, sampler) + offsetof(SamplerControls, eq)) / sizeof(float) ==
              offsets::samplerEq);
static_assert(offsetof(VoiceFrame, pan) / sizeof(float) == offsets::pan);
static_assert(offsetof(VoiceFrame, dryGain) / sizeof(float) == offsets::dryGain);
static_assert(offsetof(VoiceFrame, slots) / sizeof(float) == offsets::slots);
static_assert(sizeof(SlotControls) / sizeof(float) == offsets::slotFloats);
static_assert(offsetof(SlotControls, params) / sizeof(float) == 2);
static_assert(offsetof(SlotControls, input) / sizeof(float) == 2 + kSlotParams);
static_assert(offsetof(VoiceFrame, mod) / sizeof(float) == offsets::mod);
static_assert(sizeof(ModControls) / sizeof(float) == offsets::modFloats);
static_assert(offsetof(ModControls, routeSource) / sizeof(float) == kModSourceParams);
static_assert(kFrameFloats == offsets::end);
static_assert(std::is_standard_layout_v<InstanceFrame>);
static_assert(offsetof(InstanceFrame, slots) / sizeof(float) == offsets::instance::slots);
static_assert(offsetof(InstanceFrame, mod) / sizeof(float) == offsets::instance::mod);
static_assert(kInstanceFrameFloats == offsets::instance::end);
static_assert(kMaxVoiceSlots * kSlotParams + kMaxVoiceSlots + kPrimitiveCount + 3 == kModDestinations);

} // namespace alfa::dsp
