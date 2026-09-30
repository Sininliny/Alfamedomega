#pragma once

#include "alfa/inference/ControlFrameLayout.h"
#include "alfa/inference/FrameClock.h"
#include "alfa/inference/ModelLibrary.h"
#include "alfa/inference/Tier.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace alfa::inference
{

/// What the audio thread tells the network about one voice slot, each frame.
struct VoiceInput
{
    /// training/METHOD.md section 1: pitch/127, velocity, gate, log1p(time since note-on)/log1p(4 s),
    /// log1p(time since note-off)/log1p(4 s), mod wheel, aftertouch, pitch bend.
    std::array<float, kNoteFeatures> features{};
    /// Index into kMethodNames, or -1 when the slot is idle (no inference, zero frame).
    int method = -1;
};

/// A control frame that takes effect partway through the current block.
struct FrameEvent
{
    int sampleOffset;   ///< Sample in this block where the frame starts.
    const float* frame; ///< kFloatsPerFrame floats, voice-major, laid out per ControlFrameLayout.h.
};

struct HostStats
{
    std::uint64_t framesProcessed = 0;
    /// Frames whose inference was not back in time; the previous frame was held instead.
    std::uint64_t framesLate = 0;
};

/// Runs the neural model for one plugin instance, off the audio thread, and hands the DSP a
/// control frame every 4 ms with a fixed, reportable latency.
///
/// Threading contract:
///  - Message thread: construction, loadTier(), prepare(), setPatch(), releaseRetired(),
///    destruction. prepare() must not run concurrently with process() (JUCE guarantees this
///    for prepareToPlay / processBlock).
///  - Audio thread: setVoice(), retrigger(), setGlobalMod(), process(), reset(). These are
///    wait-free and never allocate, lock or make system calls.
///  - Model inference runs on anira's static thread pool.
///
/// Latency: the frame the DSP receives at frame boundary n was computed from the inputs given at
/// boundary n - latencyFrames(). The plugin reports latencySamples() to the host and delays the
/// note events it gives the DSP by the same amount, so notes and their control frames line up.
class InferenceHost
{
public:
    InferenceHost();
    ~InferenceHost();

    InferenceHost(const InferenceHost&) = delete;
    InferenceHost& operator=(const InferenceHost&) = delete;

    // ---- Message thread -------------------------------------------------------------------

    /// Loads a tier's graphs (after checking their hashes) and builds its inference session.
    /// If prepare() has been called, the new tier is prepared too and is picked up by the audio
    /// thread at the start of its next block; the previous tier keeps playing until then.
    /// Switching tier changes latency, so the plugin should switch with the transport stopped.
    /// Returns false and sets error on failure; the current tier is left untouched.
    bool loadTier(const ModelEntry& model, std::string& error);

    /// Sets the host sample rate and maximum block size. Recomputes latency.
    void prepare(double sampleRate, int maxBlockSize);

    /// Runs the patch encoder for a new patch and publishes the embedding to the running model.
    /// knobs are normalised 0..1; method indexes kMethodNames. The patch is remembered and
    /// applied to any tier loaded later.
    bool setPatch(std::span<const float, kPatchKnobs> knobs, int method, std::string& error);

    /// As above, for a hybrid patch (method == the index of "hybrid"): hybridA and hybridB index
    /// the two component methods, which must be different and not hybrid themselves.
    bool setPatch(std::span<const float, kPatchKnobs> knobs, int method, int hybridA, int hybridB,
                  std::string& error);

    /// Conditions the patch on a sampler reference (mono, 48 kHz), as training does: the model's
    /// audio encoder embeds its log-mel spectrogram and the embedding is added to the patch's.
    /// An empty span removes it. Remembered and applied to any tier loaded later. Returns false
    /// if the model has no audio encoder or the audio is too short (under 1025 samples).
    bool setReferenceAudio(std::span<const float> mono48k, std::string& error);

    /// Copies the loaded tier's grain-source bank (grain_bank.npy), row-major, rows x
    /// dsp::kGrainSourceLength, for the GRAIN primitive. Returns false if the tier has none.
    bool copyGrainBank(std::vector<float>& bank, std::size_t& rows) const;

    /// Offline rendering (bounce, freeze): process() waits for every frame instead of holding the
    /// previous one when inference is late. Not real-time safe while on. Call while the audio
    /// thread is not processing, as JUCE does with AudioProcessor::setNonRealtime().
    void setNonRealtime(bool nonRealtime);

    /// Frees tiers the audio thread has switched away from. Also runs inside loadTier().
    void releaseRetired();

    /// Tier the audio thread will use for its next block (the last one loaded).
    [[nodiscard]] bool hasModel() const noexcept;
    [[nodiscard]] Tier tier() const noexcept;

    /// Latency of the last loaded tier at the prepared sample rate, in frames and in samples.
    [[nodiscard]] int latencyFrames() const noexcept;
    [[nodiscard]] int latencySamples() const noexcept;

    [[nodiscard]] HostStats stats() const noexcept;

    // ---- Audio thread ---------------------------------------------------------------------

    void setVoice(int voice, const VoiceInput& input) noexcept;
    /// Clears the voice's recurrent state at the next frame (call on note-on / voice steal).
    void retrigger(int voice) noexcept;
    void setGlobalMod(std::span<const float, kGlobalModInputs> mod) noexcept;

    /// Advances by one host block. Returns the frames that start inside it, in order; the frame
    /// pointers stay valid until the next process() call. Blocks with no boundary return an
    /// empty span: keep using the last frame.
    std::span<const FrameEvent> process(int numSamples) noexcept;

    /// Blocks until every frame process() has handed to the inference thread so far has been
    /// computed, or the timeout passes (returns false). For tests and offline tools that must
    /// never outrun inference without switching to setNonRealtime(): call it on the audio
    /// thread between process() calls. Not real-time safe.
    bool waitForInference(std::chrono::milliseconds timeout) const;

    /// Re-anchors the stream after a transport jump: drops in-flight frames, zeroes the model's
    /// recurrent state and restarts the frame clock.
    void reset() noexcept;

private:
    struct Engine;

    void installEngine(std::unique_ptr<Engine> engine);
    Engine* acquireEngineForBlock() noexcept;

    // Engine hand-off, lock-free: message thread writes pending_, audio thread moves it to
    // active_ and puts the old one into retired_, which the message thread frees.
    std::atomic<Engine*> pending_{nullptr};
    std::atomic<Engine*> retired_{nullptr};
    Engine* active_ = nullptr;             // audio thread only
    std::atomic<Engine*> newest_{nullptr}; // last engine loaded, for message-thread queries

    std::array<float, kPatchKnobs> patchKnobs_{}; // message thread only
    int patchMethod_ = -1;
    int patchHybridA_ = -1, patchHybridB_ = -1;
    std::vector<float> referenceAudio_; // message thread only
    bool nonRealtime_ = false;

    double sampleRate_ = 0.0;
    int maxBlockSize_ = 0;

    FrameClock clock_;
    std::vector<float> inputFrame_;
    std::vector<float> frameStorage_; // one output frame per possible event, plus the held one
    std::vector<FrameEvent> events_;
    std::size_t heldFrame_ = 0;
    std::array<bool, kMaxVoices> retriggerPending_{};

    std::atomic<std::uint64_t> framesProcessed_{0};
    std::atomic<std::uint64_t> framesLate_{0};
};

} // namespace alfa::inference
