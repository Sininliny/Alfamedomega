#pragma once

// One tier's graphs and recurrent state, run once per frame on an inference thread.
// Not real-time safe (ONNX Runtime allocates); never called from the audio thread.

#include "TripleBuffer.h"
#include "alfa/inference/ControlFrameLayout.h"
#include "alfa/inference/ModelLibrary.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace alfa::inference
{

/// Floats per voice in the packed input frame: note features, method index, retrigger flag.
inline constexpr std::size_t kInputFloatsPerVoice = kNoteFeatures + 2;
/// Packed input frame: global modulation, then each voice slot.
inline constexpr std::size_t kInputFloatsPerFrame = kGlobalModInputs + kInputFloatsPerVoice * kMaxVoices;

/// Index of "hybrid" in kMethodNames.
inline constexpr int kHybridMethod = kNumMethods - 1;

/// f0 in Hz from the note features, as training computes it: midi_hz(pitch + 2 * bend).
[[nodiscard]] float f0FromFeatures(const float* features) noexcept;

class FramePipeline
{
public:
    /// Opens every graph of the tier and checks it against the model contract.
    /// Throws std::runtime_error with a readable message on any mismatch.
    explicit FramePipeline(const ModelEntry& model);

    /// Runs patch_encoder and publishes the embedding. For a hybrid patch, hybridA and hybridB
    /// are the component methods; otherwise -1. Message thread.
    void setPatch(std::span<const float, kPatchKnobs> knobs, int method, int hybridA, int hybridB);

    /// Runs audio_encoder on a sampler reference (mono, 48 kHz) and adds its embedding to the
    /// patch embedding, as training does. An empty span removes it. Message thread.
    /// Returns false if this model has no audio encoder.
    bool setReferenceAudio(std::span<const float> mono48k);

    /// input: kInputFloatsPerFrame floats. output: kFloatsPerFrame floats. Inference thread.
    void runFrame(const float* input, float* output);

    /// Zeroes all recurrent state. Call only while no frame is running.
    void resetState();

    [[nodiscard]] std::size_t embeddingSize() const noexcept { return embedSize_; }
    [[nodiscard]] std::size_t contextSize() const noexcept { return contextSize_; }
    [[nodiscard]] std::size_t hiddenSize() const noexcept { return hiddenSize_; }
    [[nodiscard]] bool hasAudioEncoder() const noexcept { return audioEncoder_.session != nullptr; }

    /// The grain-source bank GRAIN reads (grain_bank.npy), row-major [rows, columns]. Empty if
    /// the model has none.
    [[nodiscard]] const std::vector<float>& grainBank() const noexcept { return grainBank_; }
    [[nodiscard]] std::size_t grainBankRows() const noexcept { return grainBankRows_; }

private:
    struct Graph
    {
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string> inputNames, outputNames;
        std::vector<std::vector<std::int64_t>> inputShapes, outputShapes; // -1 = dynamic (voices)
        std::vector<const char*> outputPtrs;
        [[nodiscard]] bool hasInput(const std::string& name) const;
    };

    struct ExpertOutput
    {
        std::size_t width;  // floats per voice this tier writes
        std::size_t rows;   // 1, or N for an N x N fm.index spread to the frame's 8-wide rows
        ControlField field; // where they go
    };

    struct Expert
    {
        Graph graph;
        bool takesF0 = false;
        bool hasModalState = false;
        std::vector<ExpertOutput> outputs; // graph outputs in order; modal_state_out is skipped
        std::vector<int> outputSlot;       // graph output index to outputs index, or -1
        bool writesOsc = false, writesNoise = false;
    };

    struct Patch
    {
        std::vector<float> embedding;
        int hybridA = -1, hybridB = -1;
    };

    Graph open(const ModelEntry& model, const std::string& file);
    void checkExpert(const std::string& file, Expert& expert);
    void publishPatch();
    void runInstanceStep(const float* globalMod, const Patch& patch);
    void runHybrid(const Patch& patch, float* output);
    // Runs expert `method` on n rows of hidden state for `voices`, writing each voice's fields
    // into dest[i] (a VoiceFrame).
    void runExpert(std::size_t method, const float* hidden, const int* voices, std::size_t n,
                   float* const* dest);

    Ort::MemoryInfo memory_;
    Graph patchEncoder_, instanceStep_, voiceCore_, hybrid_, audioEncoder_;
    std::array<Expert, kNumMethods - 1> experts_; // every method but hybrid

    std::size_t embedSize_ = 0, instanceStateSize_ = 0, contextSize_ = 0, hiddenSize_ = 0, modalSize_ = 0;
    bool patchTakesHybridParts_ = false;

    // Message thread: the parts of the embedding, summed and published to the inference thread.
    std::vector<float> patchEmbedding_, audioEmbedding_;
    int hybridA_ = -1, hybridB_ = -1;
    TripleBuffer<Patch> patch_;

    std::vector<float> grainBank_;
    std::size_t grainBankRows_ = 0;

    // Recurrent state and scratch, sized once in the constructor.
    std::vector<float> instanceState_, instanceStateOut_, context_;
    std::vector<float> voiceState_; // kMaxVoices x H
    std::vector<float> modalState_; // kMaxVoices x M
    std::vector<float> f0_;         // kMaxVoices
    std::vector<float> gatherFeatures_, gatherState_;
    std::vector<float> hidden_, voiceStateOut_;
    std::vector<float> expertHidden_, expertF0_, expertModal_, expertModalOut_;
    std::vector<float> hybridHiddenA_, hybridHiddenB_, hybridXf_;
    std::vector<std::vector<float>> expertScratch_; // per output slot, kMaxVoices x width
    std::vector<float> componentFrames_;            // 2 x kMaxVoices VoiceFrames for hybrid parts
    std::vector<int> activeVoices_;
    std::array<std::vector<int>, kNumMethods> voicesByMethod_;
    std::array<int, kMaxVoices> rowOfVoice_{};
};

} // namespace alfa::inference
