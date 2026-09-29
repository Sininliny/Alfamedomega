#include "alfa/inference/InferenceHost.h"

#include "FramePipeline.h"

#include <anira/anira.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace alfa::inference
{
namespace
{

constexpr auto kInputSize = kInputFloatsPerFrame;
constexpr auto kOutputSize = kFloatsPerFrame;

/// anira backend that runs the whole per-frame graph pipeline as one "inference".
class PipelineBackend final : public anira::BackendBase
{
public:
    PipelineBackend(anira::InferenceConfig& config, FramePipeline& pipeline,
                    std::atomic<bool>& resetRequested)
        : anira::BackendBase(config)
        , pipeline_(pipeline)
        , resetRequested_(resetRequested)
    {
    }

    void process(std::vector<anira::BufferF>& input, std::vector<anira::BufferF>& output,
                 std::shared_ptr<anira::SessionElement>) override
    {
        if (resetRequested_.exchange(false, std::memory_order_acq_rel))
            pipeline_.resetState();
        pipeline_.runFrame(input[0].get_read_pointer(0), output[0].get_write_pointer(0));
    }

private:
    FramePipeline& pipeline_;
    std::atomic<bool>& resetRequested_;
};

anira::InferenceConfig makeConfig(const ModelEntry& model)
{
    const auto& tier = spec(model.tier);
    const auto in = static_cast<std::int64_t>(kInputSize);
    const auto out = static_cast<std::int64_t>(kOutputSize);
    return anira::InferenceConfig(
        {anira::ModelData(std::string("alfamedomega-frame-pipeline"), anira::InferenceBackend::CUSTOM)},
        {anira::TensorShape({{1, in}}, {{1, out}})},
        // One streamable tensor each way, one channel; one inference consumes one packed input
        // frame and produces one packed output frame.
        anira::ProcessingSpec({1}, {1}, {kInputSize}, {kOutputSize}), tier.maxInferenceMs,
        /*warm_up*/ 2,
        /*session_exclusive_processor*/ true, // the pipeline holds recurrent state
        /*blocking_ratio*/ 0.0f,
        /*num_parallel_processors*/ 1);
}

} // namespace

struct InferenceHost::Engine
{
    explicit Engine(const ModelEntry& m)
        : model(m)
        , pipeline(m)
        , config(makeConfig(m))
        , pp(config)
        , backend(config, pipeline, resetRequested)
        , handler(pp, config, backend, anira::ContextConfig(spec(m.tier).inferenceThreads))
    {
        handler.set_inference_backend(anira::InferenceBackend::CUSTOM);
    }

    void prepare(const FrameClock& clock, int maxBlockSize, double sampleRate, bool nonRealtime)
    {
        handler.set_non_realtime(nonRealtime);
        const auto framesPerBlock = static_cast<float>(clock.maxFramesPerBlock(maxBlockSize));
        const auto frameRate = static_cast<float>(sampleRate / static_cast<double>(clock.hop()));
        // The reference stream is the packed input: kInputSize "samples" per frame.
        const anira::HostConfig host(framesPerBlock * static_cast<float>(kInputSize),
                                     frameRate * static_cast<float>(kInputSize),
                                     /*allow_smaller_buffers*/ true);
        handler.prepare(host);

        // Output latency is in output-stream samples. Round it up to whole frames so every
        // popped block is exactly one frame.
        const auto raw = handler.get_latency(0);
        latencyFrames = static_cast<int>((raw + kOutputSize - 1) / kOutputSize);
        if (static_cast<std::size_t>(latencyFrames) * kOutputSize != raw)
            handler.prepare(host,
                            static_cast<unsigned>(static_cast<std::size_t>(latencyFrames) * kOutputSize), 0);

        latencySamples = latencyFrames * clock.hop();
        pipeline.resetState(); // prepare() drained all work, including the warm-up inferences
        prepared = true;
    }

    ModelEntry model;
    std::atomic<bool> resetRequested{false};
    FramePipeline pipeline;
    anira::InferenceConfig config;
    anira::PrePostProcessor pp;
    PipelineBackend backend;
    anira::InferenceHandler handler;
    bool prepared = false;
    int latencyFrames = 0;
    int latencySamples = 0;
};

InferenceHost::InferenceHost()
{
    inputFrame_.assign(kInputSize, 0.0f);
    for (std::size_t v = 0; v < kMaxVoices; ++v)
        inputFrame_[kGlobalModInputs + v * kInputFloatsPerVoice + kNoteFeatures] = -1.0f; // idle
}

InferenceHost::~InferenceHost()
{
    releaseRetired();
    delete pending_.exchange(nullptr);
    delete active_;
}

bool InferenceHost::loadTier(const ModelEntry& model, std::string& error)
{
    releaseRetired();
    if (!ModelLibrary::verify(model, error))
        return false;

    std::unique_ptr<Engine> engine;
    try
    {
        engine = std::make_unique<Engine>(model);
        if (patchMethod_ >= 0)
            engine->pipeline.setPatch(patchKnobs_, patchMethod_);
        if (sampleRate_ > 0.0)
            engine->prepare(clock_, maxBlockSize_, sampleRate_, nonRealtime_);
    }
    catch (const std::exception& e)
    {
        error = std::string(toString(model.tier)) + " model: " + e.what();
        return false;
    }
    installEngine(std::move(engine));
    return true;
}

void InferenceHost::installEngine(std::unique_ptr<Engine> engine)
{
    newest_.store(engine.get(), std::memory_order_release);
    // A tier loaded but never picked up by the audio thread is simply replaced.
    delete pending_.exchange(engine.release(), std::memory_order_acq_rel);
}

void InferenceHost::releaseRetired()
{
    delete retired_.exchange(nullptr, std::memory_order_acq_rel);
}

void InferenceHost::prepare(double sampleRate, int maxBlockSize)
{
    // Not concurrent with process(), so the message thread may complete a pending switch here.
    if (auto* next = pending_.exchange(nullptr, std::memory_order_acq_rel))
    {
        delete active_;
        active_ = next;
    }
    releaseRetired();

    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;
    clock_.prepare(sampleRate);

    const auto slots = static_cast<std::size_t>(clock_.maxFramesPerBlock(maxBlockSize)) + 1;
    frameStorage_.assign(slots * kOutputSize, 0.0f);
    events_.assign(slots, FrameEvent{0, frameStorage_.data()});
    heldFrame_ = 0;

    if (active_)
        active_->prepare(clock_, maxBlockSize, sampleRate, nonRealtime_);
}

void InferenceHost::setNonRealtime(bool nonRealtime)
{
    nonRealtime_ = nonRealtime;
    for (auto* e : {active_, pending_.load(std::memory_order_acquire)})
        if (e != nullptr)
            e->handler.set_non_realtime(nonRealtime);
}

bool InferenceHost::setPatch(std::span<const float, kPatchKnobs> knobs, int method, std::string& error)
{
    std::copy(knobs.begin(), knobs.end(), patchKnobs_.begin());
    patchMethod_ = method;

    auto* newest = newest_.load(std::memory_order_acquire);
    if (newest == nullptr)
    {
        error = "no model loaded";
        return false;
    }
    try
    {
        newest->pipeline.setPatch(knobs, method);
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }
    return true;
}

bool InferenceHost::hasModel() const noexcept
{
    return newest_.load(std::memory_order_acquire) != nullptr;
}

Tier InferenceHost::tier() const noexcept
{
    const auto* e = newest_.load(std::memory_order_acquire);
    return e ? e->model.tier : kDefaultTier;
}

int InferenceHost::latencyFrames() const noexcept
{
    const auto* e = newest_.load(std::memory_order_acquire);
    return e && e->prepared ? e->latencyFrames : 0;
}

int InferenceHost::latencySamples() const noexcept
{
    const auto* e = newest_.load(std::memory_order_acquire);
    return e && e->prepared ? e->latencySamples : 0;
}

HostStats InferenceHost::stats() const noexcept
{
    return {framesProcessed_.load(std::memory_order_relaxed), framesLate_.load(std::memory_order_relaxed)};
}

// ---- Audio thread --------------------------------------------------------------------------

void InferenceHost::setVoice(int voice, const VoiceInput& input) noexcept
{
    if (voice < 0 || voice >= kMaxVoices)
        return;
    float* slot =
        inputFrame_.data() + kGlobalModInputs + static_cast<std::size_t>(voice) * kInputFloatsPerVoice;
    std::copy(input.features.begin(), input.features.end(), slot);
    slot[kNoteFeatures] = static_cast<float>(input.method);
}

void InferenceHost::retrigger(int voice) noexcept
{
    if (voice >= 0 && voice < kMaxVoices)
        retriggerPending_[static_cast<std::size_t>(voice)] = true;
}

void InferenceHost::setGlobalMod(std::span<const float, kGlobalModInputs> mod) noexcept
{
    std::copy(mod.begin(), mod.end(), inputFrame_.begin());
}

InferenceHost::Engine* InferenceHost::acquireEngineForBlock() noexcept
{
    if (pending_.load(std::memory_order_relaxed) != nullptr &&
        retired_.load(std::memory_order_relaxed) == nullptr)
    {
        auto* next = pending_.exchange(nullptr, std::memory_order_acq_rel);
        if (next != nullptr && next->prepared)
        {
            retired_.store(active_, std::memory_order_release);
            active_ = next;
        }
        else if (next != nullptr)
        {
            // Not prepared yet: put it back and keep playing the current tier.
            Engine* expected = nullptr;
            if (!pending_.compare_exchange_strong(expected, next, std::memory_order_acq_rel))
                retired_.store(next, std::memory_order_release); // a newer tier arrived meanwhile
        }
    }
    return active_;
}

std::span<const FrameEvent> InferenceHost::process(int numSamples) noexcept
{
    if (frameStorage_.empty() || numSamples <= 0)
        return {};

    Engine* engine = acquireEngineForBlock();
    const auto slots = events_.size();
    std::size_t count = 0;

    clock_.advance(numSamples,
                   [&](int offset)
                   {
                       if (count >= slots) // cannot happen when numSamples <= maxBlockSize
                           return;
                       for (std::size_t v = 0; v < kMaxVoices; ++v)
                       {
                           inputFrame_[kGlobalModInputs + v * kInputFloatsPerVoice + kNoteFeatures + 1] =
                               retriggerPending_[v] ? 1.0f : 0.0f;
                           retriggerPending_[v] = false;
                       }

                       const auto slot = (heldFrame_ + 1) % slots;
                       float* out = frameStorage_.data() + slot * kOutputSize;
                       if (engine != nullptr)
                       {
                           const float* in = inputFrame_.data();
                           engine->handler.push_data(&in, kInputSize, 0);
                           if (engine->handler.pop_data(&out, kOutputSize, 0) == kOutputSize)
                           {
                               heldFrame_ = slot;
                           }
                           else
                           {
                               // Inference missed its deadline: hold the previous frame. anira keeps the
                               // stream time-aligned and drops the late frame when it arrives.
                               std::memcpy(out, frameStorage_.data() + heldFrame_ * kOutputSize,
                                           kOutputSize * sizeof(float));
                               heldFrame_ = slot;
                               framesLate_.fetch_add(1, std::memory_order_relaxed);
                           }
                       }
                       else
                       {
                           std::memset(out, 0, kOutputSize * sizeof(float));
                           heldFrame_ = slot;
                       }
                       framesProcessed_.fetch_add(1, std::memory_order_relaxed);
                       events_[count++] = FrameEvent{offset, out};
                   });

    return {events_.data(), count};
}

void InferenceHost::reset() noexcept
{
    clock_.reset();
    retriggerPending_.fill(false);
    if (active_ != nullptr)
    {
        active_->handler.reset();
        active_->resetRequested.store(true, std::memory_order_release);
    }
}

} // namespace alfa::inference
