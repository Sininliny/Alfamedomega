#pragma once

#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Fm.h"
#include "alfa/dsp/Grain.h"
#include "alfa/dsp/Noise.h"
#include "alfa/dsp/Osc.h"
#include "alfa/dsp/Partials.h"
#include "alfa/dsp/Primitive.h"
#include "alfa/dsp/Sampler.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace alfa::dsp
{

/// One voice: the six primitives, the per-primitive mix gains, the attack envelope and the hop
/// clock. The voice's hop grid starts at its own note-on sample; at each boundary it takes the
/// latest frame the host has written with setFrame() (repeating the last one if none arrived).
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
    void release() noexcept { releasing_ = true; }
    void kill() noexcept { active_ = false; }

    /// Latest frame from the network; consumed at the next hop boundary.
    void setFrame(const VoiceFrame& frame) noexcept;
    void setF0(float f0Hz) noexcept { f0Target_ = f0Hz; }

    /// Adds n samples of this voice to out. Deactivates itself when finished.
    void process(float* out, int n) noexcept;

    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] bool releasing() const noexcept { return releasing_; }
    [[nodiscard]] int note() const noexcept { return note_; }
    [[nodiscard]] float velocity() const noexcept { return velocity_; }
    [[nodiscard]] std::int64_t samplesSinceNoteOn() const noexcept { return age_; }
    [[nodiscard]] std::int64_t samplesSinceRelease() const noexcept { return releaseAge_; }

private:
    void beginHop() noexcept;
    void renderChunk(float* out, int n) noexcept;

    PrepareSpec spec_;
    int hop_ = 0;
    int hopPos_ = 0;
    bool firstHop_ = true;

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
    std::array<float, kPrimitiveCount> mixFrom_{};
    std::array<float, kPrimitiveCount> mixStep_{};
    std::vector<float> scratch_;

    float attackEnv_ = 1.0f;   // exp(-t / tau), counts down to 0
    float attackDecay_ = 0.0f;

    float f0Target_ = 440.0f;
    float f0Last_ = 440.0f;
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
