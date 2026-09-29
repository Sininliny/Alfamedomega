#pragma once

// PARTIALS primitive: an inharmonic sinusoid bank (ControlFrame.h, PartialsControls). Sinusoid m
// runs at f0(t) * ratio_m(t) with amplitude amp_m(t), all linear per sample across the hop; a
// partial at or above Nyquist is silent. Only the tier's first limits.partials partials render.

#include "alfa/dsp/Primitive.h"

#include <array>

namespace alfa::dsp
{

class PartialsPrimitive
{
public:
    void prepare(const PrepareSpec& spec);
    void reset() noexcept;
    void beginHop(const HopContext& ctx) noexcept;
    void render(float* out, int n) noexcept;
    void skipHop(const HopContext& ctx) noexcept;

    /// Partials rendered this hop (amp non-zero in either frame, within the tier, below Nyquist
    /// for at least part of the hop). For tests.
    [[nodiscard]] int activeCount() const noexcept { return count_; }
    /// Phase of partial m at the end of the current hop, cycles in [0, 1). For tests.
    [[nodiscard]] double phase(int m) const noexcept;

private:
    /// Advances every partial's phase to the end of the hop described by ctx (closed form) and,
    /// when collect is true, gathers the audible partials into the per-hop arrays.
    void startHop(const HopContext& ctx, bool collect) noexcept;

    static constexpr auto kN = static_cast<std::size_t>(kMaxPartials);

    double sampleRate_ = 48000.0;
    double invSampleRate_ = 1.0 / 48000.0;
    float nyquist_ = 24000.0f;
    int hop_ = 192;
    float invHop_ = 1.0f / 192.0f;
    int limit_ = kMaxPartials;

    std::array<double, kN> phase_{};      // per partial, cycles; end of the current hop
    std::array<double, kN> startPhase_{}; // per partial, cycles; start of the current hop

    // Per-hop, compacted to the audible partials (structure of arrays).
    alignas(64) std::array<float, kN> ph_{};
    alignas(64) std::array<float, kN> ratio0_{};
    alignas(64) std::array<float, kN> ratioD_{};
    alignas(64) std::array<float, kN> amp0_{};
    alignas(64) std::array<float, kN> ampD_{};
    int count_ = 0;     // partials rendered this hop
    int safeCount_ = 0; // the first safeCount_ of them stay below Nyquist for the whole hop
    int pos_ = 0;
    float f0Start_ = 440.0f;
    float f0Delta_ = 0.0f;
};

} // namespace alfa::dsp
