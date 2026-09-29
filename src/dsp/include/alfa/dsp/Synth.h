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
// Voices are mono; process() writes the same signal to both channels.

#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Primitive.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace alfa::dsp
{

class Voice;

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

    /// Pitch bend in semitones, applied to every voice from its next hop.
    void setPitchBend(float semitones) noexcept;

    /// Latest frame for a voice; consumed at that voice's next hop boundary.
    void setFrame(int voice, const VoiceFrame& frame) noexcept;

    /// Renders n <= maxBlock samples, replacing the contents of left and right (right may be null).
    void process(float* left, float* right, int n) noexcept;

    [[nodiscard]] int voiceCount() const noexcept { return static_cast<int>(voices_.size()); }
    [[nodiscard]] VoiceInfo voiceInfo(int voice) const noexcept;
    [[nodiscard]] int hop() const noexcept { return spec_.hop; }
    [[nodiscard]] double sampleRate() const noexcept { return spec_.sampleRate; }

private:
    [[nodiscard]] float f0For(int note) const noexcept;

    PrepareSpec spec_;
    int maxBlock_ = 0;
    std::vector<std::unique_ptr<Voice>> voices_;
    std::vector<std::uint64_t> startOrder_;
    std::uint64_t counter_ = 0;
    float bendSemitones_ = 0.0f;
};

} // namespace alfa::dsp
