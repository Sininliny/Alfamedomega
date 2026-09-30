#pragma once

// Audio budget guard (tier spec: allocation_by_tier.json audio_budget_guard, chosen by the user
// 2026-09-30). Before each block the Synth sums the modelled CPU cost of everything that will
// run: every active voice's primitives and slots, and the instance chain. If the total exceeds
// the tier's maxShare of real time, it degrades voice slots until it fits: the heaviest slot
// first (ties: lowest family capacity weight, then the most recent note), first to reduced
// quality where the family has one that saves CPU, then to bypass. Instance slots are never
// degraded. Costs are per second of audio, so the same share holds at 44.1, 48 and 96 kHz.
//
// The cost model is a table of measured nanoseconds of CPU per sample at 48 and 96 kHz
// (CostTableGenerated.h, written by `alfa_dsp_calibrate`), interpolated linearly in the sample
// rate. Slot costs are the median over typical param sets, and every cost is multiplied by
// kSafetyMargin (1.2, per the tier spec). It is an estimate for the machine class it was measured on; hosts
// can scale it (setCostScale) after timing their own blocks. The guard never measures time on the audio
// thread, so its decisions are deterministic for a given input.

#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Family.h"

#include <array>
#include <cstdint>

namespace alfa::dsp
{

class Voice;

/// ns of CPU per sample of audio; [..][0] at 48 kHz, [..][1] at 96 kHz.
struct CostTable
{
    float voiceBase[4][2];                     ///< per active voice: mod, mixing, pan
    float primitive[4][kPrimitiveCount][2];    ///< per voice, by tier, on top of the base
    float voiceSlot[4][kFamilyCount][2][2];    ///< per voice (mono), by tier, quality
    float instanceSlot[4][kFamilyCount][2][2]; ///< stereo, by tier, quality
    float instanceBase[2];                     ///< bus and instance chain with slots off
};

} // namespace alfa::dsp

#include "alfa/dsp/CostTableGenerated.h"

namespace alfa::dsp
{

struct GuardReport
{
    float estimatedShare = 0.0f; ///< modelled share of real time before the guard acted
    float finalShare = 0.0f;     ///< after the guard's degradations
    float budgetShare = 0.0f;
    int downgraded = 0; ///< voice slots at reduced quality
    int bypassed = 0;   ///< voice slots bypassed
};

class BudgetGuard
{
public:
    static constexpr int kMaxVoices = 64;
    /// Every modelled cost is multiplied by this: the table holds typical-param costs, and
    /// the tier spec asked for a 1.2x margin rather than worst-case params.
    static constexpr float kSafetyMargin = 1.2f;

    void prepare(double sampleRate, Tier tier) noexcept;
    void setEnabled(bool enabled) noexcept { enabled_ = enabled; }
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    /// Multiplies every modelled cost (1 = the machine the table was measured on).
    void setCostScale(float scale) noexcept { scale_ = scale > 0.0f ? scale : 1.0f; }

    /// Sets every active voice's slot guard levels for its next hop. startOrder ranks notes
    /// (higher = more recent). instanceFamilies: the families the instance slots will run.
    GuardReport run(Voice* const* voices, const std::uint64_t* startOrder, int voiceCount,
                    const Family* instanceFamilies, int instanceSlots) noexcept;

    /// Modelled cost, as a share of real time, of one module (for tests and the bench).
    [[nodiscard]] float primitiveShare(int primitive) const noexcept;
    [[nodiscard]] float voiceSlotShare(Family f, int quality) const noexcept;
    [[nodiscard]] float instanceSlotShare(Family f) const noexcept;

private:
    struct Candidate
    {
        Voice* voice;
        std::uint64_t order;
        float cost[3]; ///< share at guard level 0, 1, 2
        float weight;
        int slot;
        int level;
    };

    [[nodiscard]] float share(const float (&ns)[2]) const noexcept;

    double sampleRate_ = 48000.0;
    int tier_ = 1;
    float budget_ = 0.25f;
    float scale_ = 1.0f;
    bool enabled_ = true;
    std::array<Candidate, kMaxVoices * kMaxVoiceSlots> candidates_{};
};

} // namespace alfa::dsp
