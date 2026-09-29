#include "FramePipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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

} // namespace

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
    return g;
}

namespace
{

// Checks that a graph has a tensor of the given name and shape (0 in `expected` means "any
// positive size", -1 means "the dynamic voices dimension"). Returns the actual shape.
std::vector<std::int64_t> expectTensor(const std::string& graph, const std::vector<std::string>& names,
                                       const std::vector<std::vector<std::int64_t>>& shapes,
                                       const std::string& name, const std::vector<std::int64_t>& expected,
                                       const char* kind)
{
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end())
        fail(graph, std::string("no ") + kind + " named '" + name + "'");
    const auto& shape = shapes[static_cast<std::size_t>(it - names.begin())];
    bool ok = shape.size() == expected.size();
    for (std::size_t i = 0; ok && i < shape.size(); ++i)
    {
        if (expected[i] < 0)
            ok = shape[i] < 0;
        else if (expected[i] == 0)
            ok = shape[i] > 0;
        else
            ok = shape[i] == expected[i];
    }
    if (!ok)
        fail(graph, std::string(kind) + " '" + name + "' has shape " + shapeString(shape) + ", expected " +
                        shapeString(expected));
    return shape;
}

} // namespace

FramePipeline::FramePipeline(const ModelEntry& model)
    : memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    patchEncoder_ = open(model, "patch_encoder.onnx");
    instanceStep_ = open(model, "instance_step.onnx");
    voiceCore_ = open(model, "voice_core.onnx");

    // patch_encoder: knobs[1,32], method[1,9] -> embedding[1,E]
    {
        const auto& g = patchEncoder_;
        expectTensor("patch_encoder", g.inputNames, g.inputShapes, "knobs", {1, kPatchKnobs}, "input");
        expectTensor("patch_encoder", g.inputNames, g.inputShapes, "method", {1, kNumMethods}, "input");
        embedSize_ = static_cast<std::size_t>(
            expectTensor("patch_encoder", g.outputNames, g.outputShapes, "embedding", {1, 0}, "output")[1]);
    }
    // instance_step: embedding[1,E], global_mod[1,4], state[1,C] -> context[1,C], state_out[1,C]
    {
        const auto& g = instanceStep_;
        const auto E = i64(embedSize_);
        expectTensor("instance_step", g.inputNames, g.inputShapes, "embedding", {1, E}, "input");
        expectTensor("instance_step", g.inputNames, g.inputShapes, "global_mod", {1, kGlobalModInputs},
                     "input");
        contextSize_ = static_cast<std::size_t>(
            expectTensor("instance_step", g.inputNames, g.inputShapes, "state", {1, 0}, "input")[1]);
        const auto C = i64(contextSize_);
        expectTensor("instance_step", g.outputNames, g.outputShapes, "context", {1, C}, "output");
        expectTensor("instance_step", g.outputNames, g.outputShapes, "state_out", {1, C}, "output");
    }
    // voice_core: features[V,8], context[1,C], state[V,H] -> hidden[V,H], state_out[V,H]
    {
        const auto& g = voiceCore_;
        const auto C = i64(contextSize_);
        expectTensor("voice_core", g.inputNames, g.inputShapes, "features", {-1, kNoteFeatures}, "input");
        expectTensor("voice_core", g.inputNames, g.inputShapes, "context", {1, C}, "input");
        hiddenSize_ = static_cast<std::size_t>(
            expectTensor("voice_core", g.inputNames, g.inputShapes, "state", {-1, 0}, "input")[1]);
        const auto H = i64(hiddenSize_);
        expectTensor("voice_core", g.outputNames, g.outputShapes, "hidden", {-1, H}, "output");
        expectTensor("voice_core", g.outputNames, g.outputShapes, "state_out", {-1, H}, "output");
    }
    // expert_<method>: hidden[V,H] -> one output per control field it drives, [V, width]
    std::size_t maxOutputs = 0;
    for (std::size_t m = 0; m < kNumMethods; ++m)
    {
        const auto file = "expert_" + std::string(kMethodNames[m]) + ".onnx";
        auto& expert = experts_[m];
        expert.graph = open(model, file);
        const auto& g = expert.graph;
        expectTensor(file, g.inputNames, g.inputShapes, "hidden", {-1, i64(hiddenSize_)}, "input");
        for (std::size_t o = 0; o < g.outputNames.size(); ++o)
        {
            const auto& name = g.outputNames[o];
            const auto field = findControlField(name);
            if (!field)
                fail(file, "output '" + name + "' is not a control-frame field");
            const auto shape = expectTensor(file, g.outputNames, g.outputShapes, name, {-1, 0}, "output");
            const auto width = static_cast<std::size_t>(shape[1]);
            if (width > field->capacity)
                fail(file, "output '" + name + "' has " + std::to_string(width) +
                               " values, the frame holds " + std::to_string(field->capacity));
            // A square fm.index smaller than the frame's matrix keeps its row structure.
            std::size_t rows = 1;
            if (name == kFmIndexField && width < field->capacity)
            {
                rows = static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(width))));
                if (rows * rows != width)
                    fail(file, "output 'fm.index' must be N x N operators, got " + std::to_string(width));
            }
            expert.outputs.push_back({width, rows, *field});
        }
        maxOutputs = std::max(maxOutputs, expert.outputs.size());
    }

    for (auto* g : {&patchEncoder_, &instanceStep_, &voiceCore_})
    {
        for (const auto& n : g->inputNames)
            g->inputPtrs.push_back(n.c_str());
        for (const auto& n : g->outputNames)
            g->outputPtrs.push_back(n.c_str());
    }
    for (auto& e : experts_)
        for (const auto& n : e.graph.outputNames)
            e.graph.outputPtrs.push_back(n.c_str());

    const auto H = hiddenSize_;
    instanceState_.assign(contextSize_, 0.0f);
    instanceStateOut_.assign(contextSize_, 0.0f);
    context_.assign(contextSize_, 0.0f);
    voiceState_.assign(kMaxVoices * H, 0.0f);
    gatherFeatures_.assign(kMaxVoices * kNoteFeatures, 0.0f);
    gatherState_.assign(kMaxVoices * H, 0.0f);
    hidden_.assign(kMaxVoices * H, 0.0f);
    voiceStateOut_.assign(kMaxVoices * H, 0.0f);
    expertHidden_.assign(kMaxVoices * H, 0.0f);
    std::size_t widest = 0;
    for (const auto& f : dsp::kFields)
        widest = std::max(widest, f.count);
    expertScratch_.assign(maxOutputs, std::vector<float>(kMaxVoices * widest, 0.0f));
    for (auto& v : voicesByMethod_)
        v.reserve(kMaxVoices);
    activeVoices_.reserve(kMaxVoices);

    // Until a patch is set the model sees a zero embedding.
    embedding_.forEachSlot([this](std::vector<float>& e) { e.assign(embedSize_, 0.0f); });
}

void FramePipeline::setPatch(std::span<const float, kPatchKnobs> knobs, int method)
{
    std::array<float, kNumMethods> oneHot{};
    if (method >= 0 && method < kNumMethods)
        oneHot[static_cast<std::size_t>(method)] = 1.0f;

    const std::array<std::int64_t, 2> knobShape{1, kPatchKnobs};
    const std::array<std::int64_t, 2> methodShape{1, kNumMethods};
    const std::array<std::int64_t, 2> embedShape{1, i64(embedSize_)};
    std::vector<float> embedding(embedSize_);

    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(knobs.data()), knobs.size(),
                                        knobShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, oneHot.data(), oneHot.size(), methodShape.data(), 2),
    };
    const std::array<const char*, 2> inputNames{"knobs", "method"};
    const char* outputName = "embedding";
    auto output =
        Ort::Value::CreateTensor<float>(memory_, embedding.data(), embedding.size(), embedShape.data(), 2);
    patchEncoder_.session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(), inputs.size(),
                               &outputName, &output, 1);

    embedding_.write([&](std::vector<float>& slot)
                     { std::copy(embedding.begin(), embedding.end(), slot.begin()); });
}

void FramePipeline::resetState()
{
    std::fill(instanceState_.begin(), instanceState_.end(), 0.0f);
    std::fill(voiceState_.begin(), voiceState_.end(), 0.0f);
}

void FramePipeline::runInstanceStep(const float* globalMod)
{
    const auto& embedding = embedding_.read();
    const std::array<std::int64_t, 2> eShape{1, i64(embedSize_)};
    const std::array<std::int64_t, 2> gShape{1, kGlobalModInputs};
    const std::array<std::int64_t, 2> cShape{1, i64(contextSize_)};

    std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(embedding.data()), embedding.size(),
                                        eShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(globalMod), kGlobalModInputs,
                                        gShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, instanceState_.data(), instanceState_.size(), cShape.data(),
                                        2),
    };
    std::array<Ort::Value, 2> outputs{
        Ort::Value::CreateTensor<float>(memory_, context_.data(), context_.size(), cShape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_, instanceStateOut_.data(), instanceStateOut_.size(),
                                        cShape.data(), 2),
    };
    const std::array<const char*, 3> inputNames{"embedding", "global_mod", "state"};
    const std::array<const char*, 2> outputNames{"context", "state_out"};
    instanceStep_.session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(), inputs.size(),
                               outputNames.data(), outputs.data(), outputs.size());
    instanceState_.swap(instanceStateOut_);
}

void FramePipeline::runFrame(const float* input, float* output)
{
    std::memset(output, 0, kFloatsPerFrame * sizeof(float));
    runInstanceStep(input);

    const auto H = hiddenSize_;
    activeVoices_.clear();
    for (auto& v : voicesByMethod_)
        v.clear();

    for (std::size_t v = 0; v < kMaxVoices; ++v)
    {
        const float* in = input + kGlobalModInputs + v * kInputFloatsPerVoice;
        const auto method = static_cast<int>(std::lround(in[kNoteFeatures]));
        const bool retrigger = in[kNoteFeatures + 1] > 0.5f;
        float* state = voiceState_.data() + v * H;
        if (retrigger || method < 0 || method >= kNumMethods)
            std::fill(state, state + H, 0.0f);
        if (method < 0 || method >= kNumMethods)
            continue;

        const auto row = activeVoices_.size();
        rowOfVoice_[v] = static_cast<int>(row);
        activeVoices_.push_back(static_cast<int>(v));
        voicesByMethod_[static_cast<std::size_t>(method)].push_back(static_cast<int>(v));
        std::copy(in, in + kNoteFeatures, gatherFeatures_.data() + row * kNoteFeatures);
        std::copy(state, state + H, gatherState_.data() + row * H);
    }
    if (activeVoices_.empty())
        return;

    const auto rows = i64(activeVoices_.size());
    const std::array<std::int64_t, 2> fShape{rows, kNoteFeatures};
    const std::array<std::int64_t, 2> cShape{1, i64(contextSize_)};
    const std::array<std::int64_t, 2> hShape{rows, i64(H)};
    const auto n = activeVoices_.size();

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

    for (std::size_t m = 0; m < kNumMethods; ++m)
        if (!voicesByMethod_[m].empty())
            runExpert(m, voicesByMethod_[m], output);
}

void FramePipeline::runExpert(std::size_t method, const std::vector<int>& voices, float* output)
{
    auto& expert = experts_[method];
    const auto H = hiddenSize_;
    const auto n = voices.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto row = static_cast<std::size_t>(rowOfVoice_[static_cast<std::size_t>(voices[i])]);
        std::copy_n(hidden_.data() + row * H, H, expertHidden_.data() + i * H);
    }

    const std::array<std::int64_t, 2> hShape{i64(n), i64(H)};
    auto input = Ort::Value::CreateTensor<float>(memory_, expertHidden_.data(), n * H, hShape.data(), 2);

    // Ort::Value has no default constructor worth using here; build the output list in place.
    std::vector<Ort::Value> outs;
    outs.reserve(expert.outputs.size());
    std::vector<std::array<std::int64_t, 2>> shapes(expert.outputs.size());
    for (std::size_t k = 0; k < expert.outputs.size(); ++k)
    {
        shapes[k] = {i64(n), i64(expert.outputs[k].width)};
        outs.push_back(Ort::Value::CreateTensor<float>(memory_, expertScratch_[k].data(),
                                                       n * expert.outputs[k].width, shapes[k].data(), 2));
    }
    const char* inputName = "hidden";
    expert.graph.session->Run(Ort::RunOptions{nullptr}, &inputName, &input, 1, expert.graph.outputPtrs.data(),
                              outs.data(), outs.size());

    for (std::size_t i = 0; i < n; ++i)
    {
        float* voiceFrame = output + static_cast<std::size_t>(voices[i]) * kFloatsPerVoice;
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
    }
}

} // namespace alfa::inference
