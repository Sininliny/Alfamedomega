#pragma once

// The DSP side of one plugin instance: a fixed pool of voices driven by control frames.
//
// Threading: prepare(), setGrainBank() and setSample() run on a non-audio thread while
// process() is not running (the plugin calls them from prepareToPlay or after suspending
// processing). Everything else is audio-thread only, noexcept and allocation-free.
//
// Typical audio callback, with the inference host (src/inference) writing frames:
//   for each MIDI event:  noteOn()/noteOff() at its sample offset (split process() there)
//   for each active voice: setFrame(v, latest frame from the network)
//   process(left, right, n)
//
// Signal path: voices (mono inside, equal-power panned) sum to a stereo bus. The instance chain
// then runs on its own hop grid (starting at prepare): up to limits.instanceSlots stereo slots
// read the bus and each other, and the output is
//   masterGain * width( dryGain * bus + sum_k outGain_k * slot_k )
// with width applied as mid/side (side *= width), then a safety stage that is linear to +6 dBFS
// and never exceeds +12 dBFS. setInstanceFrame() feeds that chain.

#include "alfa/dsp/BudgetGuard.h"
#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Modulation.h"
#include "alfa/dsp/Primitive.h"
#include "alfa/dsp/SlotGraph.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace alfa::dsp
{

class Voice;
class IrBank;

struct VoiceInfo
{
    bool active = false;
    bool releasing = false;
    int note = -1;
    float velocity = 0.0f;
    std::int64_t samplesSinceNoteOn = 0;
    std::int64_t samplesSinceRelease = 0;
};

class Synth
{
public:
    Synth();
    ~Synth();
    Synth(const Synth&) = delete;
    Synth& operator=(const Synth&) = delete;

    /// Allocates the voice pool for the tier's polyphony. maxBlock bounds process() sizes.
    void prepare(double sampleRate, int maxBlock, Tier tier);

    void setGrainBank(const GrainBank* bank) noexcept;
    void setSample(const SampleData* sample) noexcept;

    /// Starts a note and returns its voice index. Takes a free voice, else steals the oldest
    /// released voice, else the oldest voice. firstFrame may be null (see Voice::start).
    int noteOn(int note, float velocity, const VoiceFrame* firstFrame = nullptr) noexcept;
    /// Marks every voice playing this note as released. The network keeps driving it (gate = 0)
    /// until its mix gains reach 0 or it stays below -100 dBFS for 100 ms.
    void noteOff(int note) noexcept;
    void allNotesOff() noexcept;
    void killAll() noexcept;
    /// Silences every voice and returns the instance chain to its state after prepare(): slot
    /// state cleared, default InstanceFrame, hop grid restarted. For offline renders that reuse
    /// one prepared Synth (the training binding).
    void reset() noexcept;

    /// Pitch bend in semitones, applied to every voice from its next hop.
    void setPitchBend(float semitones) noexcept;

    /// Latest frame for a voice; consumed at that voice's next hop boundary.
    void setFrame(int voice, const VoiceFrame& frame) noexcept;

    /// Hands the bank's impulse responses to every instance slot's CONVOLUTION family (after
    /// prepare, and after each change to the bank). Non-audio thread; the bank must outlive use.
    void publishIrs(IrBank& bank);
    /// Frees IR views no slot uses any more. Non-audio thread, any time after publishIrs().
    void collectIrs(IrBank& bank);

    /// Latest instance-chain frame; consumed at the next instance hop boundary.
    void setInstanceFrame(const InstanceFrame& frame) noexcept;

    /// Renders n samples, replacing the contents of left and right. With right null, left gets
    /// the mono sum (L + R) / 2.
    void process(float* left, float* right, int n) noexcept;

    /// Audio budget guard (BudgetGuard.h), on by default. The report is from the last block.
    void setBudgetGuardEnabled(bool enabled) noexcept { guard_.setEnabled(enabled); }
    void setCostScale(float scale) noexcept { guard_.setCostScale(scale); }
    [[nodiscard]] const GuardReport& guardReport() const noexcept { return guardReport_; }

    [[nodiscard]] int voiceCount() const noexcept { return static_cast<int>(voices_.size()); }
    [[nodiscard]] VoiceInfo voiceInfo(int voice) const noexcept;
    [[nodiscard]] int hop() const noexcept { return spec_.hop; }
    [[nodiscard]] double sampleRate() const noexcept { return spec_.sampleRate; }

private:
    [[nodiscard]] float f0For(int note) const noexcept;
    void renderChunk(float* left, float* right, int n) noexcept;
    void beginInstanceHop() noexcept;
    void instanceBlock(float* left, float* right, int offset, int n) noexcept;

    PrepareSpec spec_;
    int maxBlock_ = 0;
    std::vector<std::unique_ptr<Voice>> voices_;
    std::vector<Voice*> voicePtrs_;
    BudgetGuard guard_;
    GuardReport guardReport_;
    std::vector<std::uint64_t> startOrder_;
    std::uint64_t counter_ = 0;
    float bendSemitones_ = 0.0f;

    // Voice bus, maxBlock samples per channel.
    std::vector<float> busL_, busR_;

    // Instance chain.
    std::unique_ptr<InstanceFrame> latestI_, frameIA_, frameIB_;
    InstanceFrame* prevI_ = nullptr;
    InstanceFrame* curI_ = nullptr;
    int instHopPos_ = 0;
    bool instFirstHop_ = true;
    SlotGraph instSlots_;
    ModBank instMod_{ModScope::Instance};
    std::array<float, kModDestinations> instDest_{};
    std::array<SlotBlock, kMaxInstanceSlots> instBlocks_{};
    std::array<float, kMaxInstanceSlots> lastInstOut_{};
    float lastDry_ = 1.0f;
    float lastMaster_ = 1.0f;
    float lastWidth_ = 1.0f;
};

} // namespace alfa::dsp
