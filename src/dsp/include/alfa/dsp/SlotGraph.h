#pragma once

// The routable processing slots of a voice or of the instance (ControlFrame.h "Processing
// slots"). Each slot owns one preallocated processor per family it may run, so the network can
// change a slot's family, its inputs and its output gain every frame without allocation, and
// the slot order is whatever DAG the input weights describe.
//
// Driven one control block (<= kControlBlock samples) at a time with fully resolved values:
// the caller interpolates frames and applies modulation first.

#include "alfa/dsp/Family.h"
#include "alfa/dsp/Processor.h"

#include <array>
#include <memory>
#include <vector>

namespace alfa::dsp
{

/// One slot's values for one control block, after interpolation and modulation.
struct SlotBlock
{
    Family family = Family::Off;
    float outGain = 0.0f;
    std::array<float, kSlotParams> params{};
    std::array<float, kSlotInputs> input{};
};

class SlotGraph
{
public:
    SlotGraph();
    ~SlotGraph();
    SlotGraph(const SlotGraph&) = delete;
    SlotGraph& operator=(const SlotGraph&) = delete;

    /// voice = true: mono, families up to Body, numSources = kPrimitiveCount.
    /// voice = false: stereo instance, every family, numSources = 1 (the summed voices).
    /// Allocates spec.limits.voiceSlots or instanceSlots slots.
    void prepare(const ProcessSpec& spec, bool voice, int hop);
    void reset() noexcept;

    [[nodiscard]] int slotCount() const noexcept { return static_cast<int>(slots_.size()); }
    [[nodiscard]] int numSources() const noexcept { return numSources_; }
    [[nodiscard]] int channels() const noexcept { return channels_; }

    /// Hop boundary: a family change starts a crossfade from the old family across this hop.
    void beginHop(const SlotControls* current) noexcept;

    /// Processes n samples. sources[s * channels + c] points at n samples of source s, channel c.
    /// blocks[k] holds slot k's resolved values. Returns true if any slot ran.
    bool process(const float* const* sources, const SlotBlock* blocks, const ControlContext& ctx,
                 int n) noexcept;

    /// Output of slot k, channel c, from the last process() call (n samples, zeros if it was off).
    [[nodiscard]] const float* output(int slot, int channel) const noexcept;

    /// The slot's instance of a family (null if not allowed here). For host-side setup such as
    /// handing an IrBank to a ConvolutionProcessor; not for the audio thread.
    [[nodiscard]] Processor* processor(int slot, Family family) const noexcept
    {
        return slots_[static_cast<std::size_t>(slot)].processors[static_cast<std::size_t>(family)].get();
    }

    /// True if slot k produced output in the last process() call.
    [[nodiscard]] bool ran(int slot) const noexcept { return slots_[static_cast<std::size_t>(slot)].ran; }

    /// Longest tail among the slots that are currently running, in samples.
    [[nodiscard]] int tailSamples() const noexcept;

private:
    struct Slot
    {
        std::array<std::unique_ptr<Processor>, kFamilyCount> processors;
        Family family = Family::Off;
        Family fadingOut = Family::Off;
        int fadePos = 0;
        std::vector<float> out; // channels * kControlBlock
        std::vector<float> in;  // channels * kControlBlock
        std::vector<float> old; // channels * kControlBlock, crossfade scratch
        bool ran = false;
    };

    void runFamily(Slot& slot, Family f, const float* params, const ControlContext& ctx, float* buffer,
                   int n) noexcept;

    std::vector<Slot> slots_;
    int channels_ = 1;
    int numSources_ = kPrimitiveCount;
    int hop_ = 192;
    bool voice_ = true;
};

/// Resolves a slot's frame values at position t (0..1) across the hop.
void interpolateSlot(const SlotControls& prev, const SlotControls& cur, float t, SlotBlock& out) noexcept;

} // namespace alfa::dsp
