#pragma once

// FM primitive (ControlFrame.h, FmControls): up to eight phase-modulation operators with a
// strictly upper-triangular modulation matrix, evaluated from the last operator to the first.
//
// Interpolation inside a hop of H samples: sample k (0 .. H-1) uses
//   value_k = prev + (cur - prev) * k / H
// for ratio, level and index, and f0_k = f0Start + (f0End - f0Start) * k / H, so a hop starts
// exactly at frame j-1 (and ctx.f0Start) and frame j is reached at the next hop boundary.
// Operator i's phase (in cycles, wrapped to [0, 1)) is read at sample k and then advanced by
// f0_k * ratio_i,k / sampleRate. Phases are 0 at reset().
//
// Operators that output nothing (level 0 in both frames) and modulate no operator that is
// itself needed are skipped; their phases are still advanced (in closed form), so they come
// back in phase when they are turned on again.

#include "alfa/dsp/Primitive.h"

#include <array>

namespace alfa::dsp
{

class FmPrimitive
{
public:
    void prepare(const PrepareSpec& spec);
    void reset() noexcept;
    void beginHop(const HopContext& ctx) noexcept;
    void render(float* out, int n) noexcept;
    void skipHop(const HopContext& ctx) noexcept;

    /// Operators rendered in the current hop (after beginHop), for tests and profiling.
    [[nodiscard]] int activeOperators() const noexcept { return numActive_; }

private:
    static constexpr int kOps = kMaxFmOperators;
    static constexpr int kBlock = 64;  ///< samples evaluated per operator pass in render()

    /// Loads the f0 and ratio ramps of ctx (all operators) into gStart_/gStep_/ratio*_.
    void loadRamps(const HopContext& ctx) noexcept;
    /// Advances the phases of the operators in mask over the whole hop in closed form.
    void advanceClosedForm(unsigned mask) noexcept;

    double sampleRate_ = 48000.0;
    int hop_ = 192;
    int numOps_ = kOps;

    // Persistent state.
    std::array<double, kOps> phase_{};  ///< cycles, [0, 1), indexed by operator

    // Hop description (all operators), for closed-form advancing.
    double gStart_ = 0.0;  ///< f0 / sampleRate at sample 0 of the hop
    double gStep_ = 0.0;   ///< per-sample increment of g
    std::array<double, kOps> ratioStart_{};
    std::array<double, kOps> ratioStep_{};

    // Running per-sample values for the active operators, compact index a = 0 .. numActive_-1
    // in ascending operator order.
    int numActive_ = 0;
    unsigned activeMask_ = 0;
    std::array<int, kOps> active_{};
    double g_ = 0.0;  ///< running f0 / sampleRate; the phase path is double so notes do not drift
    double gInc_ = 0.0;
    std::array<double, kOps> ratio_{};
    std::array<double, kOps> ratioInc_{};
    int pos_ = 0;  ///< samples of the hop rendered so far
    std::array<float, kOps> level_{};  ///< at the hop start
    std::array<float, kOps> levelInc_{};
    std::array<float, kOps * kOps> mod_{};     ///< [a * 8 + b], b > a, in cycles, at the hop start
    std::array<float, kOps * kOps> modInc_{};
};

} // namespace alfa::dsp
