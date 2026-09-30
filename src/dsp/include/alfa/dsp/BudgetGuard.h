#pragma once

// Audio budget guard (tier spec: allocation_by_tier.json audio_budget_guard, chosen by the user
// 2026-09-30). Before each block the Synth asks the guard which voice slots may run. When the
// load is over the tier's maxShare of real time, it degrades voice slots: the heaviest slot
// first (ties: lowest family capacity weight, then the most recent note), first to reduced
// quality where the family has one that saves CPU, then to bypass. Instance slots and
// primitives are never degraded. Shares are per second of audio, so the same budget holds at
// 44.1, 48 and 96 kHz.
//
// Two triggers:
// - Measured (real-time hosts): the Synth times each process() call with a steady clock and the
//   guard keeps an exponential average of that load. It degrades when the predicted load goes
//   over maxShare and restores one step per block while it stays under kRestoreFraction of it,
//   so it does not flap. Levels persist between blocks and reset on a new note or family. The
//   prediction is the measured average plus the change in modelled cost since then, so a
//   note-on burst is caught in its first block, before the average sees it.
// - Model (the default, and offline renders): the decision comes from the cost model alone,
//   recomputed from scratch every block. No clock is read, so a given input always gets the
//   same decisions and training renders are reproducible.
//
// The cost model (CostTableGenerated.h, written by `alfa_dsp_calibrate`) holds measured
// nanoseconds of CPU per sample at 48 and 96 kHz, interpolated linearly in the sample rate.
// Slot costs are the median over typical param sets. One factor per rate (rateScale) rescales
// the model so its total matches the measured total on the calibration set, and every cost is
// then multiplied by kSafetyMargin (1.2, per the tier spec). Hosts can scale it further
// (setCostScale) for a slower or faster machine.

#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Family.h"

#include <array>
#include <cstddef>
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
    float rateScale[2]; ///< fitted so the model's total matches measured totals (0 reads as 1)
};

enum class GuardTrigger
{
    Model,    ///< modelled cost only: deterministic, for offline renders and tests
    Measured, ///< measured process() load, with the model for prediction and victim choice
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
    int downgraded = 0;         ///< voice slots at reduced quality
    int bypassed = 0;           ///< voice slots bypassed
    float measuredShare = 0.0f; ///< average measured load (Measured trigger; 0 before the first)
};

class BudgetGuard
{
public:
    static constexpr int kMaxVoices = 64;
    /// Every modelled cost is multiplied by this: the table holds typical-param costs, and
    /// the tier spec asked for a 1.2x margin rather than worst-case params.
    static constexpr float kSafetyMargin = 1.2f;
    /// Measured trigger: restore a step only while the load stays under this share of budget.
    static constexpr float kRestoreFraction = 0.8f;
    /// Measured trigger: time constant of the load average, in seconds.
    static constexpr double kLoadTau = 0.2;

    void prepare(double sampleRate, Tier tier) noexcept;
    void setEnabled(bool enabled) noexcept { enabled_ = enabled; }
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    /// Multiplies every modelled cost (1 = the machine the table was measured on).
    void setCostScale(float scale) noexcept { scale_ = scale > 0.0f ? scale : 1.0f; }
    /// Replaces the generated cost table (calibration). The table must outlive use; null restores
    /// the generated one.
    void setCostTable(const CostTable* table) noexcept;
    void setTrigger(GuardTrigger trigger) noexcept;
    [[nodiscard]] GuardTrigger trigger() const noexcept { return trigger_; }
    /// Forgets measurements and degradation levels.
    void reset() noexcept;
    /// Measured trigger: the CPU time one process() call of `samples` samples took.
    void observe(double seconds, int samples) noexcept;

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
        std::size_t index; ///< into levels_
    };

    [[nodiscard]] float share(const float (&ns)[2]) const noexcept;

    static constexpr std::size_t kMaxSlots = static_cast<std::size_t>(kMaxVoices) * kMaxVoiceSlots;

    double sampleRate_ = 48000.0;
    int tier_ = 1;
    float budget_ = 0.25f;
    float scale_ = 1.0f;
    bool enabled_ = true;
    GuardTrigger trigger_ = GuardTrigger::Model;
    const CostTable* table_ = &generated::kCostTable;
    std::array<Candidate, kMaxSlots> candidates_{};

    // Measured trigger state.
    std::array<std::int8_t, kMaxSlots> levels_{};
    std::array<Family, kMaxSlots> families_{};
    std::array<std::uint64_t, kMaxVoices> orders_{};
    bool measured_ = false;  ///< at least one observe() since reset
    float loadAvg_ = 0.0f;   ///< measured share of real time, averaged
    float modelAvg_ = 0.0f;  ///< modelled share of what ran, averaged the same way
    float lastModel_ = 0.0f; ///< modelled share of the last block's decisions
};

} // namespace alfa::dsp
