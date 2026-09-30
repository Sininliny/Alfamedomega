#pragma once

#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Fm.h"
#include "alfa/dsp/Grain.h"
#include "alfa/dsp/Modulation.h"
#include "alfa/dsp/Noise.h"
#include "alfa/dsp/Osc.h"
#include "alfa/dsp/Partials.h"
#include "alfa/dsp/Primitive.h"
#include "alfa/dsp/Sampler.h"
#include "alfa/dsp/SlotGraph.h"

#include <array>
#include <cstdint>
#include <memory>

namespace alfa::dsp
{

/// One voice: the six primitives, the attack envelope, the voice slots, modulation, pan and the
/// hop clock (ControlFrame.h). The voice's hop grid starts at its own note-on sample; at each
/// boundary it takes the latest frame the host wrote with setFrame() (repeating the last one if
/// none arrived). Inside a hop it works in control blocks of at most kControlBlock samples:
/// interpolate the frame, apply the mod matrix, render primitives, run the slots, mix and pan.
///
/// The envelope follower tracks the voice's primitive sum; the matrix reads it at the start of a
/// block, so it lags the audio by one control block (at most 32 samples).
class Voice
{
public:
    Voice();
    ~Voice();
    Voice(const Voice&) = delete;
    Voice& operator=(const Voice&) = delete;

    void prepare(const PrepareSpec& spec, std::uint32_t seed);

    void setGrainBank(const GrainBank* bank) noexcept;
    void setSample(const SampleData* sample) noexcept;

    /// Starts a note. firstFrame may be null if the host sets it with setFrame() before the next
    /// process call; until then the voice is silent.
    void start(int note, float velocity, float f0Hz, const VoiceFrame* firstFrame) noexcept;
    void release() noexcept;
    void kill() noexcept { active_ = false; }

    /// Latest frame from the network; consumed at the next hop boundary.
    void setFrame(const VoiceFrame& frame) noexcept;
    void setF0(float f0Hz) noexcept { f0Target_ = f0Hz; }

    /// Adds n samples of this voice to left and right. Deactivates itself when finished.
    void process(float* left, float* right, int n) noexcept;

    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] bool releasing() const noexcept { return releasing_; }
    [[nodiscard]] int note() const noexcept { return note_; }
    [[nodiscard]] float velocity() const noexcept { return velocity_; }
    [[nodiscard]] std::int64_t samplesSinceNoteOn() const noexcept { return age_; }
    [[nodiscard]] std::int64_t samplesSinceRelease() const noexcept { return releaseAge_; }

    // For the audio budget guard (BudgetGuard.h).
    /// True if primitive p renders this hop or would next hop (latest frame).
    [[nodiscard]] bool primitiveActive(int p) const noexcept;
    /// The family slot k runs next hop per the latest frame, before the guard.
    [[nodiscard]] Family requestedFamily(int slot) const noexcept;
    [[nodiscard]] int slotCount() const noexcept { return slots_.slotCount(); }
    void setSlotGuard(int slot, int level) noexcept { slots_.setGuardLevel(slot, level); }
    [[nodiscard]] int slotGuard(int slot) const noexcept { return slots_.guardLevel(slot); }

private:
    void beginHop() noexcept;
    void renderBlock(float* left, float* right, int n) noexcept;
    [[nodiscard]] bool finished() const noexcept;

    PrepareSpec spec_;
    int hop_ = 0;
    int hopPos_ = 0;
    bool firstHop_ = true;
    std::uint32_t seed_ = 0;

    // Frames: latest_ is written by the host; prev_/cur_ are frames j-1 and j.
    std::unique_ptr<VoiceFrame> latest_, frameA_, frameB_;
    VoiceFrame* prev_ = nullptr;
    VoiceFrame* cur_ = nullptr;
    bool haveFrame_ = false;

    OscPrimitive osc_;
    PartialsPrimitive partials_;
    FmPrimitive fm_;
    NoisePrimitive noise_;
    GrainPrimitive grain_;
    SamplerPrimitive sampler_;
    std::array<bool, kPrimitiveCount> rendering_{};

    SlotGraph slots_;
    ModBank mod_{ModScope::Voice};
    std::array<float, kModDestinations> dest_{};
    std::array<SlotBlock, kMaxVoiceSlots> blocks_{};

    // Control-block buffers.
    std::array<std::array<float, kControlBlock>, kPrimitiveCount> prim_{};
    std::array<float, kControlBlock> dry_{};
    std::array<float, kControlBlock> mono_{};
    std::array<float, kControlBlock> env_{};

    // Gains at the end of the previous block; each block ramps from these to its own values.
    std::array<float, kPrimitiveCount> lastMix_{};
    std::array<float, kMaxVoiceSlots> lastOut_{};
    float lastDry_ = 0.0f;
    float lastPan_ = 0.0f;

    float attackEnv_ = 1.0f; // exp(-t / tau), counts down to 0
    float attackDecay_ = 0.0f;

    float f0Target_ = 440.0f;
    float f0Last_ = 440.0f;
    float f0Now_ = 440.0f;
    int note_ = -1;
    float velocity_ = 0.0f;
    bool active_ = false;
    bool releasing_ = false;
    std::int64_t age_ = 0;
    std::int64_t releaseAge_ = 0;
    int silentHops_ = 0;
    float hopPeak_ = 0.0f;
};

} // namespace alfa::dsp
