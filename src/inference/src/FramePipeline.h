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

class FramePipeline
{
public:
    /// Opens every graph of the tier and checks it against the model contract.
    /// Throws std::runtime_error with a readable message on any mismatch.
    explicit FramePipeline(const ModelEntry& model);

    /// Runs patch_encoder and publishes the embedding. Message thread.
    void setPatch(std::span<const float, kPatchKnobs> knobs, int method);

    /// input: kInputFloatsPerFrame floats. output: kFloatsPerFrame floats. Inference thread.
    void runFrame(const float* input, float* output);

    /// Zeroes all recurrent state. Call only while no frame is running.
    void resetState();

    [[nodiscard]] std::size_t embeddingSize() const noexcept { return embedSize_; }
    [[nodiscard]] std::size_t contextSize() const noexcept { return contextSize_; }
    [[nodiscard]] std::size_t hiddenSize() const noexcept { return hiddenSize_; }

private:
    struct Graph
    {
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string> inputNames, outputNames;
        std::vector<const char*> inputPtrs, outputPtrs;
        std::vector<std::vector<std::int64_t>> inputShapes, outputShapes; // -1 = dynamic (voices)
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
        std::vector<ExpertOutput> outputs; // in graph output order
    };

    Graph open(const ModelEntry& model, const std::string& file);
    void runInstanceStep(const float* globalMod);
    void runExpert(std::size_t method, const std::vector<int>& voices, float* output);

    Ort::MemoryInfo memory_;
    Graph patchEncoder_, instanceStep_, voiceCore_;
    std::array<Expert, kNumMethods> experts_;

    std::size_t embedSize_ = 0, contextSize_ = 0, hiddenSize_ = 0;

    TripleBuffer<std::vector<float>> embedding_;

    // Recurrent state and scratch, sized once in the constructor.
    std::vector<float> instanceState_, instanceStateOut_, context_;
    std::vector<float> voiceState_;                   // kMaxVoices x H
    std::vector<float> gatherFeatures_, gatherState_; // active voices x 8, x H
    std::vector<float> hidden_, voiceStateOut_;       // active voices x H
    std::vector<float> expertHidden_;                 // voices of one method x H
    std::vector<std::vector<float>> expertScratch_;   // per output slot, kMaxVoices x width
    std::vector<int> activeVoices_;
    std::array<std::vector<int>, kNumMethods> voicesByMethod_;
    std::array<int, kMaxVoices> rowOfVoice_{};
};

} // namespace alfa::inference
