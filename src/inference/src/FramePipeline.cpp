#include "FramePipeline.h"

#include "LogMel.h"
#include "Npy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace alfa::inference
{
namespace
{

Ort::Env& ortEnv()
{
    // One environment per process, created on first use from the message thread.
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "alfamedomega");
    return env;
}

Ort::SessionOptions sessionOptions()
{
    Ort::SessionOptions options;
    // One thread per session: the caller (an anira worker) is the only thread that runs it,
    // so ONNX Runtime creates no pool of its own and never competes with the audio thread.
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    return options;
}

std::string shapeString(const std::vector<std::int64_t>& shape)
{
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
        s += (i ? ", " : "") + (shape[i] < 0 ? std::string("voices") : std::to_string(shape[i]));
    return s + "]";
}

[[noreturn]] void fail(const std::string& graph, const std::string& what)
{
    throw std::runtime_error(graph + ": " + what + " (see src/inference/MODEL_CONTRACT.md)");
}

std::int64_t i64(std::size_t n)
{
    return static_cast<std::int64_t>(n);
}

// 0 in `expected` means "any positive size", -1 means "the dynamic voices dimension".
bool shapeMatches(const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& expected)
{
    if (shape.size() != expected.size())
        return false;
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (expected[i] < 0 ? shape[i] >= 0 : expected[i] == 0 ? shape[i] <= 0 : shape[i] != expected[i])
            return false;
    }
    return true;
}

// Checks that a graph has a tensor of the given name and shape. Returns the actual shape.
std::vector<std::int64_t> expectTensor(const std::string& graph, const std::vector<std::string>& names,
                                       const std::vector<std::vector<std::int64_t>>& shapes,
                                       const std::string& name, const std::vector<std::int64_t>& expected,
                                       const char* kind)
{
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end())
        fail(graph, std::string("no ") + kind + " named '" + name + "'");
    const auto& shape = shapes[static_cast<std::size_t>(it - names.begin())];
    if (!shapeMatches(shape, expected))
        fail(graph, std::string(kind) + " '" + name + "' has shape " + shapeString(shape) + ", expected " +
                        shapeString(expected));
    return shape;
}

constexpr std::size_t kMix = dsp::offsets::mix;
constexpr std::size_t kOscMix = static_cast<std::size_t>(dsp::Primitive::Osc);
constexpr std::size_t kNoiseMix = static_cast<std::size_t>(dsp::Primitive::Noise);

// Fields the hybrid merge combines itself; every other field comes from its one component.
bool mergedSeparately(std::string_view field)
{
    return field == "mix" || field == "attack_tau" || field == "osc.re" || field == "osc.im" ||
           field == "noise.band" || field == "osc.detune_ratio" || field == "osc.detune_level";
}

void fillDefaults(float* frames, std::size_t voices, const std::vector<float>& defaultVoice)
{
    for (std::size_t v = 0; v < voices; ++v)
        std::copy(defaultVoice.begin(), defaultVoice.end(), frames + v * kFloatsPerVoice);
}

} // namespace

float f0FromFeatures(const float* features) noexcept
{
    const double midi = static_cast<double>(features[0]) * 127.0 + 2.0 * static_cast<double>(features[7]);
    return static_cast<float>(440.0 * std::pow(2.0, (midi - 69.0) / 12.0));
}

bool FramePipeline::Graph::hasInput(const std::string& name) const
{
    return std::find(inputNames.begin(), inputNames.end(), name) != inputNames.end();
}

FramePipeline::Graph FramePipeline::open(const ModelEntry& model, const std::string& file)
{
    const auto path = model.folder / file;
    if (model.files.find(file) == model.files.end())
        fail(file, "missing from the manifest");

    Graph g;
    try
    {
        g.session = std::make_unique<Ort::Session>(ortEnv(), path.c_str(), sessionOptions());
    }
    catch (const Ort::Exception& e)
    {
        fail(file, std::string("cannot load: ") + e.what());
    }

    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < g.session->GetInputCount(); ++i)
    {
        g.inputNames.emplace_back(g.session->GetInputNameAllocated(i, allocator).get());
        g.inputShapes.push_back(g.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (std::size_t i = 0; i < g.session->GetOutputCount(); ++i)
    {
        g.outputNames.emplace_back(g.session->GetOutputNameAllocated(i, allocator).get());
        g.outputShapes.push_back(g.session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (const auto& n : g.outputNames)
        g.outputPtrs.push_back(n.c_str());
    return g;
}

void FramePipeline::checkExpert(const std::string& file, Expert& expert)
{
    const auto& g = expert.graph;
    // Inputs: hidden[V,H], optional f0[V] (the exporter prunes it where unused), and for
    // physical modal_state[V,M] with a matching modal_state_out output.
    expectTensor(file, g.inputNames, g.inputShapes, "hidden", {-1, i64(hiddenSize_)}, "input");
    for (const auto& name : g.inputNames)
        if (name != "hidden" && name != "f0" && name != "modal_state")
            fail(file, "unexpected input '" + name + "'");
    if (g.hasInput("f0"))
    {
        expectTensor(file, g.inputNames, g.inputShapes, "f0", {-1}, "input");
        expert.takesF0 = true;
    }
    if (g.hasInput("modal_state"))
    {
        const auto m = expectTensor(file, g.inputNames, g.inputShapes, "modal_state", {-1, 0}, "input")[1];
        if (modalSize_ != 0 && static_cast<std::size_t>(m) != modalSize_)
            fail(file, "modal_state width differs between experts");
        modalSize_ = static_cast<std::size_t>(m);
        expectTensor(file, g.outputNames, g.outputShapes, "modal_state_out", {-1, m}, "output");
        expert.hasModalState = true;
    }

    // Outputs: one per control-frame field the expert drives, named after it, [V, width].
    for (const auto& name : g.outputNames)
    {
        if (name == "modal_state_out")
        {
            if (!expert.hasModalState)
                fail(file, "output 'modal_state_out' without input 'modal_state'");
            expert.outputSlot.push_back(-1);
            continue;
        }
        const auto field = findControlField(name);
        if (!field)
            fail(file, "output '" + name + "' is not a control-frame field");
        const auto shape = expectTensor(file, g.outputNames, g.outputShapes, name, {-1, 0}, "output");
        const auto width = static_cast<std::size_t>(shape[1]);
        if (width > field->capacity)
            fail(file, "output '" + name + "' has " + std::to_string(width) + " values, the frame holds " +
                           std::to_string(field->capacity));
        // A square fm.index smaller than the frame's matrix keeps its row structure.
        std::size_t rows = 1;
        if (name == kFmIndexField && width < field->capacity)
        {
            rows = static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(width))));
            if (rows * rows != width)
                fail(file, "output 'fm.index' must be N x N operators, got " + std::to_string(width));
        }
        expert.outputSlot.push_back(static_cast<int>(expert.outputs.size()));
        expert.outputs.push_back({width, rows, *field});
        expert.writesOsc = expert.writesOsc || name == "osc.re" || name == "osc.im";
        expert.writesNoise = expert.writesNoise || name == "noise.band";
        expert.writesDetune = expert.writesDetune || name == "osc.detune_level";
    }
}

FramePipeline::FramePipeline(const ModelEntry& model)
    : memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    // The training export records the control-frame layout it was built for.
    if (model.files.count("export_info.json") != 0)
    {
        std::ifstream in(model.folder / "export_info.json");
        const auto info = nlohmann::json::parse(in, nullptr, false);
        if (info.is_discarded())
            fail("export_info.json", "not valid JSON");
        const int layout = info.value("layout_version", dsp::kLayoutVersion);
        if (layout != dsp::kLayoutVersion)
            fail("export_info.json", "model was exported for control-frame layout " + std::to_string(layout) +
                                         ", this build uses layout " + std::to_string(dsp::kLayoutVersion));
        const auto floats = info.value("voiceframe_floats", kFloatsPerVoice);
        if (floats != kFloatsPerVoice)
            fail("export_info.json", "model expects " + std::to_string(floats) +
                                         " floats per VoiceFrame, this build has " +
                                         std::to_string(kFloatsPerVoice));
    }

    patchEncoder_ = open(model, "patch_encoder.onnx");
    instanceStep_ = open(model, "instance_step.onnx");
    voiceCore_ = open(model, "voice_core.onnx");
    hybrid_ = open(model, "expert_hybrid.onnx");
    if (model.files.count("audio_encoder.onnx") != 0)
        audioEncoder_ = open(model, "audio_encoder.onnx");

    // patch_encoder: knobs[1,32], method[1,9], optional hybrid_parts[1,18] -> embedding[1,E]
    {
        const auto& g = patchEncoder_;
        expectTensor("patch_encoder", g.inputNames, g.inputShapes, "knobs", {1, kPatchKnobs}, "input");
        expectTensor("patch_encoder", g.inputNames, g.inputShapes, "method", {1, kNumMethods}, "input");
        if (g.hasInput("hybrid_parts"))
        {
            expectTensor("patch_encoder", g.inputNames, g.inputShapes, "hybrid_parts", {1, 2 * kNumMethods},
                         "input");
            patchTakesHybridParts_ = true;
        }
        embedSize_ = static_cast<std::size_t>(
            expectTensor("patch_encoder", g.outputNames, g.outputShapes, "embedding", {1, 0}, "output")[1]);
    }
    // instance_step: embedding[1,E], global_mod[1,4], state[1,S] -> context[1,C], state_out[1,S]
    {
        const auto& g = instanceStep_;
        expectTensor("instance_step", g.inputNames, g.inputShapes, "embedding", {1, i64(embedSize_)},
                     "input");
        expectTensor("instance_step", g.inputNames, g.inputShapes, "global_mod", {1, kGlobalModInputs},
                     "input");
        instanceStateSize_ = static_cast<std::size_t>(
            expectTensor("instance_step", g.inputNames, g.inputShapes, "state", {1, 0}, "input")[1]);
        contextSize_ = static_cast<std::size_t>(
            expectTensor("instance_step", g.outputNames, g.outputShapes, "context", {1, 0}, "output")[1]);
        expectTensor("instance_step", g.outputNames, g.outputShapes, "state_out",
                     {1, i64(instanceStateSize_)}, "output");
    }
    // voice_core: features[V,8], context[1,C], state[V,H] -> hidden[V,H], state_out[V,H]
    {
        const auto& g = voiceCore_;
        expectTensor("voice_core", g.inputNames, g.inputShapes, "features", {-1, kNoteFeatures}, "input");
        expectTensor("voice_core", g.inputNames, g.inputShapes, "context", {1, i64(contextSize_)}, "input");
        hiddenSize_ = static_cast<std::size_t>(
            expectTensor("voice_core", g.inputNames, g.inputShapes, "state", {-1, 0}, "input")[1]);
        const auto H = i64(hiddenSize_);
        expectTensor("voice_core", g.outputNames, g.outputShapes, "hidden", {-1, H}, "output");
        expectTensor("voice_core", g.outputNames, g.outputShapes, "state_out", {-1, H}, "output");
    }
    // expert_<method> for every method but hybrid.
    std::size_t maxOutputs = 0;
    for (std::size_t m = 0; m < experts_.size(); ++m)
    {
        const auto file = "expert_" + std::string(kMethodNames[m]) + ".onnx";
        experts_[m].graph = open(model, file);
        checkExpert(file, experts_[m]);
        maxOutputs = std::max(maxOutputs, experts_[m].outputs.size());
    }
    // expert_hybrid: hidden[V,H] -> hidden_a[V,H], hidden_b[V,H], xf[V] (weight of component A)
    {
        const auto& g = hybrid_;
        const auto H = i64(hiddenSize_);
        expectTensor("expert_hybrid", g.inputNames, g.inputShapes, "hidden", {-1, H}, "input");
        expectTensor("expert_hybrid", g.outputNames, g.outputShapes, "hidden_a", {-1, H}, "output");
        expectTensor("expert_hybrid", g.outputNames, g.outputShapes, "hidden_b", {-1, H}, "output");
        expectTensor("expert_hybrid", g.outputNames, g.outputShapes, "xf", {-1}, "output");
        if (g.outputNames.size() != 3 || g.inputNames.size() != 1)
            fail("expert_hybrid", "expected exactly input 'hidden' and outputs 'hidden_a', 'hidden_b', 'xf'");
    }
    // audio_encoder: logmel[1,64,frames] -> e_audio[1,E]
    if (audioEncoder_.session)
    {
        const auto& g = audioEncoder_;
        expectTensor("audio_encoder", g.inputNames, g.inputShapes, "logmel", {1, i64(kMelBands), -1},
                     "input");
        expectTensor("audio_encoder", g.outputNames, g.outputShapes, "e_audio", {1, i64(embedSize_)},
                     "output");
    }
    // grain_bank.npy: the learned grain-source bank GRAIN reads.
    if (model.files.count("grain_bank.npy") != 0)
    {
        std::size_t columns = 0;
        std::string error;
        if (!readNpyFloat2d(model.folder / "grain_bank.npy", grainBank_, grainBankRows_, columns, error))
            fail("grain_bank.npy", error);
        if (columns != static_cast<std::size_t>(dsp::kGrainSourceLength))
            fail("grain_bank.npy", "rows must hold " + std::to_string(dsp::kGrainSourceLength) +
                                       " samples, got " + std::to_string(columns));
    }

    const auto H = hiddenSize_;
    const auto M = modalSize_;
    instanceState_.assign(instanceStateSize_, 0.0f);
    instanceStateOut_.assign(instanceStateSize_, 0.0f);
    context_.assign(contextSize_, 0.0f);
    voiceState_.assign(kMaxVoices * H, 0.0f);
    modalState_.assign(kMaxVoices * M, 0.0f);
    f0_.assign(kMaxVoices, 0.0f);
    gatherFeatures_.assign(kMaxVoices * kNoteFeatures, 0.0f);
    gatherState_.assign(kMaxVoices * H, 0.0f);
    hidden_.assign(kMaxVoices * H, 0.0f);
    voiceStateOut_.assign(kMaxVoices * H, 0.0f);
    expertHidden_.assign(kMaxVoices * H, 0.0f);
    expertF0_.assign(kMaxVoices, 0.0f);
    expertModal_.assign(kMaxVoices * M, 0.0f);
    expertModalOut_.assign(kMaxVoices * M, 0.0f);
    hybridHiddenA_.assign(kMaxVoices * H, 0.0f);
    hybridHiddenB_.assign(kMaxVoices * H, 0.0f);
    hybridXf_.assign(kMaxVoices, 0.0f);
    std::size_t widest = 0;
    for (const auto& f : dsp::kFields)
        widest = std::max(widest, f.count);
    expertScratch_.assign(maxOutputs, std::vector<float>(kMaxVoices * widest, 0.0f));
    componentFrames_.assign(2 * kMaxVoices * kFloatsPerVoice, 0.0f);
    {
        const dsp::VoiceFrame defaults{};
        const auto* first = reinterpret_cast<const float*>(&defaults);
        defaultVoice_.assign(first, first + kFloatsPerVoice);
    }
    for (auto& v : voicesByMethod_)
        v.reserve(kMaxVoices);
    activeVoices_.reserve(kMaxVoices);

    // Until a patch is set the model sees a zero embedding.
    patchEmbedding_.assign(embedSize_, 0.0f);
    patch_.forEachSlot([this](Patch& p) { p.embedding.assign(embedSize_, 0.0f); });
}

void FramePipeline::publishPatch()
{
    patch_.write(
        [&](Patch& slot)
        {
            std::copy(patchEmbedding_.begin(), patchEmbedding_.end(), slot.embedding.begin());
            // Training conditions on e_patch + e_audio when a reference is given.
            if (!audioEmbedding_.empty())
                for (std::size_t i = 0; i < embedSize_; ++i)
                    slot.embedding[i] += audioEmbedding_[i];
            slot.hybridA = hybridA_;
            slot.hybridB = hybridB_;
        });
}

void FramePipeline::setPatch(std::span<const float, kPatchKnobs> knobs, int method, int hybridA, int hybridB)
{
    const auto isComponent = [](int m)
    {
        return m >= 0 && m < kHybridMethod;
    };
    std::array<float, kNumMethods> oneHot{};
    if (method >= 0 && method < kNumMethods)
        oneHot[static_cast<std::size_t>(method)] = 1.0f;
    const bool hybrid = method == kHybridMethod && isComponent(hybridA) && isComponent(hybridB);
    std::array<float, 2 * kNumMethods> parts{};
    if (hybrid)
    {
        // Training's merge raises on this too: only mix, attack, OSC and NOISE can be shared.
        for (const auto& oa : experts_[static_cast<std::size_t>(hybridA)].outputs)
            for (const auto& ob : experts_[static_cast<std::size_t>(hybridB)].outputs)
                if (oa.field.name == ob.field.name && !mergedSeparately(oa.field.name))
                    throw std::runtime_error(
                        "hybrid of " + std::string(kMethodNames[static_cast<std::size_t>(hybridA)]) +
                        " and " + std::string(kMethodNames[static_cast<std::size_t>(hybridB)]) +
                        ": both drive '" + std::string(oa.field.name) + "'");
        parts[static_cast<std::size_t>(hybridA)] = 1.0f;
        parts[static_cast<std::size_t>(kNumMethods + hybridB)] = 1.0f;
    }

    const std::array<std::int64_t, 2> knobShape{1, kPatchKnobs};
    const std::array<std::int64_t, 2> methodShape{1, kNumMethods};
    const std::array<std::int64_t, 2> partsShape{1, 2 * kNumMethods};
    const std::array<std::int64_t, 2> embedShape{1, i64(embedSize_)};
    std::vector<float> embedding(embedSize_);

    std::vector<Ort::Value> inputs;
    std::vector<const char*> inputNames{"knobs", "method"};
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(knobs.data()), knobs.size(),
                                                     knobShape.data(), 2));
    inputs.push_back(
        Ort::Value::CreateTensor<float>(memory_, oneHot.data(), oneHot.size(), methodShape.data(), 2));
    if (patchTakesHybridParts_)
    {
        inputNames.push_back("hybrid_parts");
        inputs.push_back(
            Ort::Value::CreateTensor<float>(memory_, parts.data(), parts.size(), partsShape.data(), 2));
    }
    const char* outputName = "embedding";
    auto output =
        Ort::Value::CreateTensor<float>(memory_, embedding.data(), embedding.size(), embedShape.data(), 2);
    patchEncoder_.session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(), inputs.size(),
                               &outputName, &output, 1);

    patchEmbedding_ = std::move(embedding);
    hybridA_ = hybrid ? hybridA : -1;
    hybridB_ = hybrid ? hybridB : -1;
    publishPatch();
}

bool FramePipeline::setReferenceAudio(std::span<const float> mono48k)
{
    if (!audioEncoder_.session)
        return false;
    if (mono48k.empty())
    {
        audioEmbedding_.clear();
        publishPatch();
        return true;
    }
    std::size_t frames = 0;
    auto mel = logMel(mono48k, frames);
    if (mel.empty())
        return false;

    const std::array<std::int64_t, 3> melShape{1, i64(kMelBands), i64(frames)};
    const std::array<std::int64_t, 2> embedShape{1, i64(embedSize_)};
    std::vector<float> embedding(embedSize_);
    auto input = Ort::Value::CreateTensor<float>(memory_, mel.data(), mel.size(), melShape.data(), 3);
    auto output =
        Ort::Value::CreateTensor<float>(memory_, embedding.data(), embedding.size(), embedShape.data(), 2);
    const char* inputName = "logmel";
    const char* outputName = "e_audio";
    audioEncoder_.session->Run(Ort::RunOptions{nullptr}, &inputName, &input, 1, &outputName, &output, 1);
    audioEmbedding_ = std::move(embedding);
    publishPatch();
    return true;
}

void FramePipeline::resetState()
{
    std::fill(instanceState_.begin(), instanceState_.end(), 0.0f);
    std::fill(voiceState_.begin(), voiceState_.end(), 0.0f);
    std::fill(modalState_.begin(), modalState_.end(), 0.0f);
}

void FramePipeline::runInstanceStep(const float* globalMod, const Patch& patch)
{
    const std::array<std::int64_t, 2> eShape{1, i64(embedSize_)};
    const std::array<std::int64_t, 2> gShape{1, kGlobalModInputs};
    const std::array<std::int64_t, 2> sShape{1, i64(instanceStateSize_)};
    const std::array<std::int64_t, 2> cShape{1, i64(contextSize_)};

    std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(patch.embedding.data()),
                                        patch.embedding.size(), eShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(globalMod), kGlobalModInputs,
                                        gShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, instanceState_.data(), instanceState_.size(), sShape.data(),
                                        2),
    };
    std::array<Ort::Value, 2> outputs{
        Ort::Value::CreateTensor<float>(memory_, context_.data(), context_.size(), cShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, instanceStateOut_.data(), instanceStateOut_.size(),
                                        sShape.data(), 2),
    };
    const std::array<const char*, 3> inputNames{"embedding", "global_mod", "state"};
    const std::array<const char*, 2> outputNames{"context", "state_out"};
    instanceStep_.session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(), inputs.size(),
                               outputNames.data(), outputs.data(), outputs.size());
    instanceState_.swap(instanceStateOut_);
}

void FramePipeline::runFrame(const float* input, float* output)
{
    fillDefaults(output, kMaxVoices, defaultVoice_);
    const auto& patch = patch_.read();
    runInstanceStep(input, patch);

    const auto H = hiddenSize_;
    const auto M = modalSize_;
    activeVoices_.clear();
    for (auto& v : voicesByMethod_)
        v.clear();

    for (std::size_t v = 0; v < kMaxVoices; ++v)
    {
        const float* in = input + kGlobalModInputs + v * kInputFloatsPerVoice;
        const auto method = static_cast<int>(std::lround(in[kNoteFeatures]));
        const bool retrigger = in[kNoteFeatures + 1] > 0.5f;
        const bool idle = method < 0 || method >= kNumMethods;
        if (retrigger || idle)
        {
            std::fill_n(voiceState_.data() + v * H, H, 0.0f);
            std::fill_n(modalState_.data() + v * M, M, 0.0f);
        }
        if (idle)
            continue;

        const auto row = activeVoices_.size();
        rowOfVoice_[v] = static_cast<int>(row);
        activeVoices_.push_back(static_cast<int>(v));
        voicesByMethod_[static_cast<std::size_t>(method)].push_back(static_cast<int>(v));
        f0_[v] = f0FromFeatures(in);
        std::copy(in, in + kNoteFeatures, gatherFeatures_.data() + row * kNoteFeatures);
        std::copy_n(voiceState_.data() + v * H, H, gatherState_.data() + row * H);
    }
    if (activeVoices_.empty())
        return;

    const auto n = activeVoices_.size();
    const std::array<std::int64_t, 2> fShape{i64(n), kNoteFeatures};
    const std::array<std::int64_t, 2> cShape{1, i64(contextSize_)};
    const std::array<std::int64_t, 2> hShape{i64(n), i64(H)};

    std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor<float>(memory_, gatherFeatures_.data(), n * kNoteFeatures, fShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, context_.data(), context_.size(), cShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, gatherState_.data(), n * H, hShape.data(), 2),
    };
    std::array<Ort::Value, 2> outputs{
        Ort::Value::CreateTensor<float>(memory_, hidden_.data(), n * H, hShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, voiceStateOut_.data(), n * H, hShape.data(), 2),
    };
    const std::array<const char*, 3> inputNames{"features", "context", "state"};
    const std::array<const char*, 2> outputNames{"hidden", "state_out"};
    voiceCore_.session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(), inputs.size(),
                            outputNames.data(), outputs.data(), outputs.size());

    for (std::size_t row = 0; row < n; ++row)
    {
        const auto v = static_cast<std::size_t>(activeVoices_[row]);
        std::copy_n(voiceStateOut_.data() + row * H, H, voiceState_.data() + v * H);
    }

    std::array<float*, kMaxVoices> dest{};
    for (std::size_t m = 0; m < experts_.size(); ++m)
    {
        const auto& voices = voicesByMethod_[m];
        if (voices.empty())
            continue;
        for (std::size_t i = 0; i < voices.size(); ++i)
        {
            const auto v = static_cast<std::size_t>(voices[i]);
            const auto row = static_cast<std::size_t>(rowOfVoice_[v]);
            std::copy_n(hidden_.data() + row * H, H, expertHidden_.data() + i * H);
            dest[i] = output + v * kFloatsPerVoice;
        }
        runExpert(m, expertHidden_.data(), voices.data(), voices.size(), dest.data());
    }
    if (!voicesByMethod_[kHybridMethod].empty())
        runHybrid(patch, output);
}

void FramePipeline::runExpert(std::size_t method, const float* hidden, const int* voices, std::size_t n,
                              float* const* dest)
{
    auto& expert = experts_[method];
    const auto H = hiddenSize_;
    const auto M = modalSize_;
    const std::array<std::int64_t, 2> hShape{i64(n), i64(H)};
    const std::array<std::int64_t, 1> f0Shape{i64(n)};
    const std::array<std::int64_t, 2> mShape{i64(n), i64(M)};

    std::array<Ort::Value, 3> inputs{Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
    std::array<const char*, 3> inputNames{};
    std::size_t inputCount = 0;
    inputNames[inputCount] = "hidden";
    inputs[inputCount++] =
        Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(hidden), n * H, hShape.data(), 2);
    if (expert.takesF0)
    {
        for (std::size_t i = 0; i < n; ++i)
            expertF0_[i] = f0_[static_cast<std::size_t>(voices[i])];
        inputNames[inputCount] = "f0";
        inputs[inputCount++] =
            Ort::Value::CreateTensor<float>(memory_, expertF0_.data(), n, f0Shape.data(), 1);
    }
    if (expert.hasModalState)
    {
        for (std::size_t i = 0; i < n; ++i)
            std::copy_n(modalState_.data() + static_cast<std::size_t>(voices[i]) * M, M,
                        expertModal_.data() + i * M);
        inputNames[inputCount] = "modal_state";
        inputs[inputCount++] =
            Ort::Value::CreateTensor<float>(memory_, expertModal_.data(), n * M, mShape.data(), 2);
    }

    // Ort::Value has no default constructor worth using here; build the output list in place.
    std::vector<Ort::Value> outs;
    outs.reserve(expert.outputSlot.size());
    std::vector<std::array<std::int64_t, 2>> shapes(expert.outputs.size());
    for (const int slot : expert.outputSlot)
    {
        if (slot < 0)
        {
            outs.push_back(
                Ort::Value::CreateTensor<float>(memory_, expertModalOut_.data(), n * M, mShape.data(), 2));
            continue;
        }
        const auto k = static_cast<std::size_t>(slot);
        shapes[k] = {i64(n), i64(expert.outputs[k].width)};
        outs.push_back(Ort::Value::CreateTensor<float>(memory_, expertScratch_[k].data(),
                                                       n * expert.outputs[k].width, shapes[k].data(), 2));
    }
    expert.graph.session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(), inputCount,
                              expert.graph.outputPtrs.data(), outs.data(), outs.size());

    for (std::size_t i = 0; i < n; ++i)
    {
        float* voiceFrame = dest[i];
        for (std::size_t k = 0; k < expert.outputs.size(); ++k)
        {
            const auto& o = expert.outputs[k];
            const float* src = expertScratch_[k].data() + i * o.width;
            if (o.rows == 1)
            {
                std::copy_n(src, o.width, voiceFrame + o.field.offset);
                continue;
            }
            const auto cols = o.width / o.rows;
            const auto stride = static_cast<std::size_t>(dsp::kMaxFmOperators);
            for (std::size_t r = 0; r < o.rows; ++r)
                std::copy_n(src + r * cols, cols, voiceFrame + o.field.offset + r * stride);
        }
        if (expert.hasModalState)
            std::copy_n(expertModalOut_.data() + i * M, M,
                        modalState_.data() + static_cast<std::size_t>(voices[i]) * M);
    }
}

void FramePipeline::runHybrid(const Patch& patch, float* output)
{
    // A hybrid patch without two components leaves its voices silent.
    if (patch.hybridA < 0 || patch.hybridB < 0)
        return;

    const auto& voices = voicesByMethod_[kHybridMethod];
    const auto n = voices.size();
    const auto H = hiddenSize_;
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto row = static_cast<std::size_t>(rowOfVoice_[static_cast<std::size_t>(voices[i])]);
        std::copy_n(hidden_.data() + row * H, H, expertHidden_.data() + i * H);
    }

    // expert_hybrid splits the hidden state into one per component, plus the crossfade.
    {
        const std::array<std::int64_t, 2> hShape{i64(n), i64(H)};
        const std::array<std::int64_t, 1> xShape{i64(n)};
        auto input = Ort::Value::CreateTensor<float>(memory_, expertHidden_.data(), n * H, hShape.data(), 2);
        std::vector<Ort::Value> outs;
        outs.reserve(3);
        for (const auto& name : hybrid_.outputNames)
        {
            if (name == "xf")
                outs.push_back(
                    Ort::Value::CreateTensor<float>(memory_, hybridXf_.data(), n, xShape.data(), 1));
            else
                outs.push_back(Ort::Value::CreateTensor<float>(
                    memory_, name == "hidden_a" ? hybridHiddenA_.data() : hybridHiddenB_.data(), n * H,
                    hShape.data(), 2));
        }
        const char* inputName = "hidden";
        hybrid_.session->Run(Ort::RunOptions{nullptr}, &inputName, &input, 1, hybrid_.outputPtrs.data(),
                             outs.data(), outs.size());
    }

    // Each component expert writes a scratch VoiceFrame; state such as modal_state still
    // belongs to the voice.
    fillDefaults(componentFrames_.data(), 2 * kMaxVoices, defaultVoice_);
    std::array<float*, kMaxVoices> destA{}, destB{};
    for (std::size_t i = 0; i < n; ++i)
    {
        destA[i] = componentFrames_.data() + i * kFloatsPerVoice;
        destB[i] = componentFrames_.data() + (kMaxVoices + i) * kFloatsPerVoice;
    }
    const auto a = static_cast<std::size_t>(patch.hybridA);
    const auto b = static_cast<std::size_t>(patch.hybridB);
    runExpert(a, hybridHiddenA_.data(), voices.data(), n, destA.data());
    runExpert(b, hybridHiddenB_.data(), voices.data(), n, destB.data());

    // Merge as training's merge_hybrid does: mixes crossfade, OSC spectra add weighted by each
    // component's OSC mix, NOISE bands add by power, attack takes the faster component, and
    // every other field comes from the one component that drives it.
    const auto& expA = experts_[a];
    const auto& expB = experts_[b];
    constexpr auto kOsc = dsp::offsets::oscRe;
    constexpr auto kOscFloats = 2 * static_cast<std::size_t>(dsp::kOscHarmonics);
    constexpr auto kNoise = dsp::offsets::noiseBand;
    constexpr auto kBands = static_cast<std::size_t>(dsp::kBands);
    constexpr auto kPrimitives = static_cast<std::size_t>(dsp::kPrimitiveCount);
    constexpr auto kDetuneRatio = dsp::offsets::oscDetuneRatio;
    constexpr auto kDetuneLevel = dsp::offsets::oscDetuneLevel;
    constexpr auto kDetune = static_cast<std::size_t>(dsp::kMaxOscDetune);
    for (std::size_t i = 0; i < n; ++i)
    {
        const float* fa = destA[i];
        const float* fb = destB[i];
        float* out = output + static_cast<std::size_t>(voices[i]) * kFloatsPerVoice;
        const float wa = hybridXf_[i];
        const float wb = 1.0f - wa;

        for (const auto* part : {&expA, &expB})
        {
            const float* src = part == &expA ? fa : fb;
            for (const auto& o : part->outputs)
                if (!mergedSeparately(o.field.name))
                    std::copy_n(src + o.field.offset, o.field.capacity, out + o.field.offset);
        }
        for (std::size_t p = 0; p < kPrimitives; ++p)
            out[kMix + p] = wa * fa[kMix + p] + wb * fb[kMix + p];
        out[dsp::offsets::attackTau] = std::min(fa[dsp::offsets::attackTau], fb[dsp::offsets::attackTau]);

        if (expA.writesOsc || expB.writesOsc)
        {
            const float ga = expA.writesOsc ? wa * fa[kMix + kOscMix] : 0.0f;
            const float gb = expB.writesOsc ? wb * fb[kMix + kOscMix] : 0.0f;
            for (std::size_t k = 0; k < kOscFloats; ++k)
                out[kOsc + k] = ga * fa[kOsc + k] + gb * fb[kOsc + k];
            out[kMix + kOscMix] = 1.0f;

            // Unison reads act on the merged table, so one set serves both: the OSC-weighted mean
            // of the components' levels and ratios. A component without unison reads as level 0,
            // ratio 1, which is what its default-filled scratch frame holds.
            if (expA.writesDetune || expB.writesDetune)
            {
                for (std::size_t k = 0; k < kDetune; ++k)
                {
                    float numR = 0.0f, numL = 0.0f;
                    for (const auto& [frame, g] : {std::pair{fa, ga}, std::pair{fb, gb}})
                    {
                        const float level = frame[kDetuneLevel + k];
                        numR += g * level * frame[kDetuneRatio + k];
                        numL += g * level;
                    }
                    out[kDetuneLevel + k] = numL / (ga + gb + 1e-6f);
                    out[kDetuneRatio + k] = numL > 1e-9f ? numR / (numL + 1e-12f) : 1.0f;
                }
            }
        }
        if (expA.writesNoise || expB.writesNoise)
        {
            const float ga = expA.writesNoise ? wa * fa[kMix + kNoiseMix] : 0.0f;
            const float gb = expB.writesNoise ? wb * fb[kMix + kNoiseMix] : 0.0f;
            for (std::size_t k = 0; k < kBands; ++k)
            {
                const float pa = ga * fa[kNoise + k];
                const float pb = gb * fb[kNoise + k];
                out[kNoise + k] = std::sqrt(pa * pa + pb * pb + 1e-12f);
            }
            out[kMix + kNoiseMix] = 1.0f;
        }
    }
}

} // namespace alfa::inference
