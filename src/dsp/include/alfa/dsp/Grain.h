#pragma once

// GRAIN primitive (ControlFrame.h, GrainControls): grain layers reading a grain-source bank.
//
// At every beginHop, each layer below spec.limits.grainLayers starts one grain lasting 2 hops,
// windowed by a periodic Hann of length 2 * hop, so at 50% overlap every layer has two grains
// sounding (the one started last hop and the new one) and a constant grain sums to exactly 1.
// A grain uses ctx.cur's values, held for its whole life:
//   out += gain * hann[age] * (w0 * src0(pos) + w1 * src1(pos)),   pos += rate per sample
// where src is read with linear interpolation, wrapping modulo bank->length, and pos starts at
// position * bank->length. source is rounded to the nearest integer; an index outside
// [0, bank->count), a zero weight or a missing bank contributes nothing. A layer with no
// audible source or a zero gain starts no grain.
//
// The bank pointer given to setBank() is sampled at reset() and beginHop(); a grain resolves
// its source pointers when it starts, so the bank's buffers must stay alive (and unchanged)
// while grains that read them are sounding.
//
// skipHop(ctx) (mix 0 for the hop) drops every live grain: a grain started before the skipped
// hop would have ended during it, and nothing starts in it.

#include "alfa/dsp/Primitive.h"

#include <array>
#include <vector>

namespace alfa::dsp
{

class GrainPrimitive
{
public:
    void prepare(const PrepareSpec& spec);
    void reset() noexcept;
    void setBank(const GrainBank* bank) noexcept { pendingBank_ = bank; }
    void beginHop(const HopContext& ctx) noexcept;
    void render(float* out, int n) noexcept;
    void skipHop(const HopContext& ctx) noexcept;

    /// Grains currently sounding, for tests and profiling.
    [[nodiscard]] int liveGrains() const noexcept;

private:
    struct Grain
    {
        const float* src0 = nullptr;  ///< nullptr when the source is silent
        const float* src1 = nullptr;
        float amp0 = 0.0f;            ///< gain * w0
        float amp1 = 0.0f;            ///< gain * w1
        double pos = 0.0;             ///< read position in bank samples, [0, length)
        double rate = 0.0;
        double length = 0.0;
        int age = 0;                  ///< samples since the grain started
        bool live = false;
    };

    void renderGrain(Grain& g, float* out, int n) noexcept;

    int hop_ = 192;
    int numLayers_ = kMaxGrainLayers;
    std::vector<float> window_;  ///< periodic Hann, 2 * hop
    const GrainBank* pendingBank_ = nullptr;
    const GrainBank* bank_ = nullptr;
    std::array<std::array<Grain, 2>, kMaxGrainLayers> grains_{};
    std::array<int, kMaxGrainLayers> next_{};  ///< slot the next grain of each layer goes to
};

} // namespace alfa::dsp
