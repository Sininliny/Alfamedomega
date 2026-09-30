#include "SynthBackend.h"

#include <alfa/dsp/Synth.h>
#include <alfa/inference/InferenceHost.h>
#include <alfa/inference/ModelLibrary.h>
#include <cmath>
#include <limits>

namespace alfa
{
namespace
{
namespace inf = alfa::inference;

constexpr int kSlots = inf::kMaxVoices;
constexpr size_t kQueueSize = 4096;

/// The real synth: the inference host turns notes and the patch into control
/// frames, and alfa::dsp::Synth renders them.
///
/// Voices. The network sees 16 voice slots; the DSP allocates its own voices.
/// The plugin assigns a slot when a note arrives, and hands the note to the DSP
/// latencySamples() later, when the frames computed from it start arriving. At
/// that point the slot is mapped to whichever DSP voice took the note, and
/// every frame for the slot goes to that voice.
///
/// Threads. The constructor, requestTier(), the timer (tier loading, patch
/// changes) and prepare() are on the message thread; render() is on the audio
/// thread and never allocates or locks. A tier swap re-prepares the DSP, which
/// happens with processing suspended.
class RealBackend final : public SynthBackend, private juce::Timer
{
public:
    RealBackend(juce::AudioProcessorValueTreeState& s, juce::AudioProcessor& p, inf::ModelLibrary lib)
        : state(s)
        , processor(p)
        , library(std::move(lib))
    {
        method = raw(params::id::method);
        hybridA = raw(params::id::hybridA);
        hybridB = raw(params::id::hybridB);
        for (int i = 0; i < params::kNumKnobs; ++i)
            knobs[static_cast<size_t>(i)] = raw(params::id::knob(i));
        voiceLimit = raw(params::id::voices);
        bendRange = raw(params::id::bendRange);

        requestTier(static_cast<params::Tier>(juce::roundToInt(raw(params::id::modelTier)->load())));
        loadRequestedTier();
        startTimerHz(20);
    }

    ~RealBackend() override { stopTimer(); }

    void prepare(double newSampleRate, int newMaxBlock) override
    {
        sampleRate = newSampleRate;
        maxBlock = juce::jmax(1, newMaxBlock);
        host.prepare(sampleRate, maxBlock);
        prepareSynth();
    }

    void setNonRealtime(bool offline) override
    {
        host.setNonRealtime(offline);
        synth.setBudgetGuardTrigger(offline ? dsp::GuardTrigger::Model : dsp::GuardTrigger::Measured);
    }

    void requestTier(params::Tier tier) override { requestedTier.store(static_cast<int>(tier)); }

    int latencySamples(double) const override { return host.hasModel() ? host.latencySamples() : 0; }

    juce::String description() const override
    {
        juce::String text = "Neural model " + juce::String(loadedId) + " (" +
                            toJuce(inf::toString(host.tier())) + "), " + juce::String(host.latencySamples()) +
                            " samples latency";
        const auto stats = host.stats();
        if (stats.framesLate > 0)
            text << ", " << juce::String(static_cast<juce::int64>(stats.framesLate)) << " late frames";
        if (loadError.isNotEmpty())
            text << ". " << loadError;
        return text;
    }

    void render(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi, Telemetry& t) override
    {
        const auto numSamples = buffer.getNumSamples();
        if (!prepared || numSamples <= 0)
            return;

        const bool audible = patchValid.load(std::memory_order_relaxed);
        const int methodIndex = audible ? currentMethod.load(std::memory_order_relaxed) : -1;
        const auto latency = host.latencySamples();

        // 1. Notes as they arrive drive the network's inputs now...
        for (const auto meta : midi)
            handleMidi(meta.getMessage(), clock + meta.samplePosition, latency, t);
        updateInputs(methodIndex);
        const auto frames = host.process(numSamples);

        // 2. ...and reach the DSP `latency` samples later, together with the
        // frames computed from them.
        auto* left = buffer.getWritePointer(0);
        auto* right = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;
        size_t nextFrame = 0;
        int pos = 0;
        const auto end = clock + numSamples;
        while (pos < numSamples)
        {
            while (queueSize > 0 && queue[queueRead].time <= clock + pos)
            {
                dispatch(queue[queueRead]);
                queueRead = (queueRead + 1) % kQueueSize;
                --queueSize;
            }
            while (nextFrame < frames.size() && frames[nextFrame].sampleOffset <= pos)
            {
                applyFrame(frames[nextFrame++].frame, t);
            }

            int next = numSamples;
            if (nextFrame < frames.size())
                next = juce::jmin(next, frames[nextFrame].sampleOffset);
            if (queueSize > 0)
                next = juce::jmin(next, static_cast<int>(juce::jlimit<int64_t>(
                                            pos + 1, numSamples, queue[queueRead].time - clock)));
            next = juce::jmax(next, pos + 1);
            synth.process(left + pos, right != nullptr ? right + pos : nullptr, next - pos);
            pos = next;
        }
        clock = end;

        if (!audible)
            buffer.clear();

        releaseFinishedSlots();
        publishTelemetry(t, buffer);
    }

private:
    struct Slot
    {
        int note = -1;
        float velocity = 0.0f;
        bool held = false;
        bool busy = false; // playing or releasing, as far as the network knows
        int64_t onAt = 0;  // undelayed sample time
        int64_t offAt = 0;
        uint32_t age = 0;
        int dspVoice = -1; // mapped when the delayed note-on reaches the DSP
        int pending = 0;   // delayed events not yet given to the DSP
    };

    enum class EventType
    {
        noteOn,
        noteOff,
        bend,
        allOff
    };

    struct Event
    {
        int64_t time = 0;
        EventType type = EventType::noteOn;
        int slot = -1;
        int note = 0;
        float value = 0.0f;
    };

    static juce::String toJuce(std::string_view text) { return juce::String(text.data(), text.size()); }

    std::atomic<float>* raw(const juce::String& id)
    {
        auto* p = state.getRawParameterValue(id);
        jassert(p != nullptr);
        return p;
    }

    // ---- Message thread -----------------------------------------------------------------

    void timerCallback() override
    {
        if (requestedTier.load() != attemptedTier)
            loadRequestedTier();
        sendPatchIfChanged();
        host.releaseRetired();
    }

    void loadRequestedTier()
    {
        attemptedTier = requestedTier.load();
        const auto* entry = library.findNearest(static_cast<inf::Tier>(attemptedTier));
        if (entry == nullptr)
            return;

        std::string error;
        if (!host.loadTier(*entry, error))
        {
            loadError = "Could not load " + juce::String(entry->id) + ": " + juce::String(error);
            return;
        }
        loadError = entry->tier == static_cast<inf::Tier>(attemptedTier)
                        ? juce::String()
                        : "The requested size is not installed; using the nearest one";
        loadedId = entry->id;
        loadedTier.store(static_cast<int>(entry->tier));

        std::size_t rows = 0;
        grainBankData.clear();
        if (!host.copyGrainBank(grainBankData, rows))
            rows = 0;
        grainBank = {grainBankData.data(), static_cast<int>(rows), dsp::kGrainSourceLength};

        lastSentShape = -1; // resend the patch for the new tier
        sendPatchIfChanged();

        if (sampleRate > 0.0)
        {
            // Polyphony and grain bank are per tier: swap them while the
            // audio thread is held off.
            processor.suspendProcessing(true);
            prepareSynth();
            processor.suspendProcessing(false);
        }
    }

    void prepareSynth()
    {
        synth.prepare(sampleRate, maxBlock, static_cast<dsp::Tier>(loadedTier.load()));
        synth.setBudgetGuardTrigger(processor.isNonRealtime() ? dsp::GuardTrigger::Model
                                                              : dsp::GuardTrigger::Measured);
        synth.setGrainBank(grainBank.count > 0 ? &grainBank : nullptr);
        host.reset();
        slots = {};
        queueRead = queueWrite = queueSize = 0;
        clock = 0;
        prepared = true;
    }

    void sendPatchIfChanged()
    {
        const auto shape = params::readShape(state);
        std::array<float, params::kNumKnobs> values{};
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = knobs[i]->load();

        const auto packed = params::packShape(shape);
        if (packed == lastSentShape && values == lastSentKnobs)
            return;
        lastSentShape = packed;
        lastSentKnobs = values;

        const bool hybrid = shape.method == params::Method::hybrid;
        if (hybrid && !params::isValidHybridPair(shape.hybridA, shape.hybridB))
        {
            // Not a trained pair: silence rather than layering two engines.
            patchValid.store(false);
            return;
        }

        std::string error;
        const bool ok =
            hybrid ? host.setPatch(values, static_cast<int>(shape.method), static_cast<int>(shape.hybridA),
                                   static_cast<int>(shape.hybridB), error)
                   : host.setPatch(values, static_cast<int>(shape.method), error);
        patchValid.store(ok);
        currentMethod.store(static_cast<int>(shape.method));
        if (!ok)
            loadError = "Patch rejected: " + juce::String(error);
    }

    // ---- Audio thread -------------------------------------------------------------------

    void push(const Event& e) noexcept
    {
        if (queueSize == kQueueSize)
            return; // over 4096 events inside one latency window: drop
        queue[queueWrite] = e;
        queueWrite = (queueWrite + 1) % kQueueSize;
        ++queueSize;
        if (e.slot >= 0)
            ++slots[static_cast<size_t>(e.slot)].pending;
    }

    int limit() const noexcept
    {
        return juce::jlimit(1, juce::jmin(kSlots, synth.voiceCount()), static_cast<int>(voiceLimit->load()));
    }

    void handleMidi(const juce::MidiMessage& msg, int64_t time, int latency, Telemetry& t) noexcept
    {
        if (msg.isNoteOn())
        {
            const auto slot = allocate();
            auto& s = slots[static_cast<size_t>(slot)];
            if (s.note >= 0 && s.held)
                t.noteVelocity[static_cast<size_t>(s.note)].store(0.0f);
            s.note = msg.getNoteNumber();
            s.velocity = msg.getFloatVelocity();
            s.held = true;
            s.busy = true;
            s.onAt = time;
            s.age = ++ageCounter;
            host.retrigger(slot);
            push({time + latency, EventType::noteOn, slot, s.note, s.velocity});
            t.noteVelocity[static_cast<size_t>(s.note)].store(s.velocity);
        }
        else if (msg.isNoteOff())
        {
            for (int i = 0; i < kSlots; ++i)
            {
                auto& s = slots[static_cast<size_t>(i)];
                if (s.held && s.note == msg.getNoteNumber())
                {
                    s.held = false;
                    s.offAt = time;
                    push({time + latency, EventType::noteOff, i, s.note, 0.0f});
                }
            }
            t.noteVelocity[static_cast<size_t>(msg.getNoteNumber())].store(0.0f);
        }
        else if (msg.isPitchWheel())
        {
            bend = static_cast<float>(msg.getPitchWheelValue() - 8192) / 8192.0f;
            push({time + latency, EventType::bend, -1, 0, bend});
        }
        else if (msg.isControllerOfType(1))
        {
            modWheel = static_cast<float>(msg.getControllerValue()) / 127.0f;
        }
        else if (msg.isChannelPressure())
        {
            aftertouch = static_cast<float>(msg.getChannelPressureValue()) / 127.0f;
        }
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
        {
            for (auto& s : slots)
            {
                if (s.held)
                {
                    s.held = false;
                    s.offAt = time;
                }
                if (s.note >= 0)
                    t.noteVelocity[static_cast<size_t>(s.note)].store(0.0f);
            }
            push({time + latency, EventType::allOff, -1, 0, 0.0f});
        }
    }

    /// A free slot, else the oldest released one, else the oldest.
    int allocate() noexcept
    {
        int best = 0;
        int bestRank = 3;
        uint32_t bestAge = std::numeric_limits<uint32_t>::max();
        for (int i = 0; i < limit(); ++i)
        {
            const auto& s = slots[static_cast<size_t>(i)];
            const int rank = !s.busy ? 0 : (!s.held ? 1 : 2);
            if (rank < bestRank || (rank == bestRank && s.age < bestAge))
            {
                best = i;
                bestRank = rank;
                bestAge = s.age;
            }
        }
        return best;
    }

    void updateInputs(int methodIndex) noexcept
    {
        static const float logOf4s = std::log1p(4.0f);
        const auto secondsSince = [this](int64_t at)
        {
            return static_cast<float>(static_cast<double>(clock - at) / sampleRate);
        };

        for (int i = 0; i < kSlots; ++i)
        {
            const auto& s = slots[static_cast<size_t>(i)];
            inf::VoiceInput in;
            if (s.busy && methodIndex >= 0)
            {
                in.method = methodIndex;
                in.features = {static_cast<float>(s.note) / 127.0f,
                               s.velocity,
                               s.held ? 1.0f : 0.0f,
                               std::log1p(juce::jmax(0.0f, secondsSince(s.onAt))) / logOf4s,
                               s.held ? 0.0f : std::log1p(juce::jmax(0.0f, secondsSince(s.offAt))) / logOf4s,
                               modWheel,
                               aftertouch,
                               bend};
            }
            host.setVoice(i, in);
        }
        const std::array<float, inf::kGlobalModInputs> global{modWheel, aftertouch, bend, 0.0f};
        host.setGlobalMod(global);
    }

    void dispatch(const Event& e) noexcept
    {
        if (e.slot >= 0)
            --slots[static_cast<size_t>(e.slot)].pending;

        switch (e.type)
        {
        case EventType::noteOn:
        {
            auto& s = slots[static_cast<size_t>(e.slot)];
            if (s.dspVoice >= 0)
                silence(s.dspVoice); // the slot was stolen from a note still sounding
            // No first frame: the voice stays silent until the first
            // frame computed from this note arrives, at most one hop later.
            const auto voice = synth.noteOn(e.note, e.value, nullptr);
            for (auto& other : slots)
                if (other.dspVoice == voice)
                    other.dspVoice = -1; // the DSP stole that voice
            s.dspVoice = voice;
            break;
        }
        case EventType::noteOff:
            synth.noteOff(e.note);
            break;
        case EventType::bend:
            synth.setPitchBend(e.value * bendRange->load());
            break;
        case EventType::allOff:
            synth.allNotesOff();
            break;
        }
    }

    static const dsp::VoiceFrame* voiceFrame(const float* frame, int slot) noexcept
    {
        return reinterpret_cast<const dsp::VoiceFrame*>(frame +
                                                        static_cast<size_t>(slot) * inf::kFloatsPerVoice);
    }

    /// A zero frame fades a voice out: its mix gains ramp to 0 over one hop.
    void silence(int voice) noexcept { synth.setFrame(voice, zeroFrame); }

    void applyFrame(const float* frame, Telemetry& t) noexcept
    {
        bool shown = false;
        for (int i = 0; i < kSlots; ++i)
        {
            const auto& s = slots[static_cast<size_t>(i)];
            if (s.dspVoice < 0)
                continue;
            const auto& f = *voiceFrame(frame, i);
            synth.setFrame(s.dspVoice, f);
            if (!shown && s.held)
            {
                showControls(f, t);
                shown = true;
            }
        }
    }

    /// Sixteen readable numbers from a playing voice's frame for the model
    /// view: the six primitive mix gains, the attack, then the first nine
    /// harmonic magnitudes.
    static void showControls(const dsp::VoiceFrame& f, Telemetry& t) noexcept
    {
        const auto squash = [](float v)
        {
            return std::tanh(std::abs(v));
        };
        size_t i = 0;
        for (const auto m : f.mix)
            t.controlVector[i++].store(squash(m));
        t.controlVector[i++].store(juce::jlimit(0.0f, 1.0f, f.attackTau * 10.0f));
        for (size_t h = 0; i < t.controlVector.size(); ++h)
            t.controlVector[i++].store(squash(std::hypot(f.osc.re[h], f.osc.im[h]) * 4.0f));
    }

    /// A released slot is free once its DSP voice has finished.
    void releaseFinishedSlots() noexcept
    {
        for (auto& s : slots)
        {
            if (!s.busy || s.held || s.pending > 0)
                continue;
            const bool finished = s.dspVoice < 0 || !synth.voiceInfo(s.dspVoice).active ||
                                  synth.voiceInfo(s.dspVoice).note != s.note;
            if (finished)
            {
                s.busy = false;
                s.dspVoice = -1;
            }
        }
    }

    void publishTelemetry(Telemetry& t, const juce::AudioBuffer<float>& buffer) noexcept
    {
        int active = 0, held = 0;
        for (const auto& s : slots)
        {
            active += s.busy ? 1 : 0;
            held += s.held ? 1 : 0;
        }
        t.activeVoices.store(active);
        t.envelopeStage.store(held > 0 ? 3 : (active > 0 ? 4 : 0));
        t.envelopeLevel.store(juce::jmin(1.0f, buffer.getMagnitude(0, 0, buffer.getNumSamples()) * 2.0f));

        // Which experts run: the patch's method, and for a hybrid both components.
        std::array<float, params::kNumMethods> level{};
        if (patchValid.load(std::memory_order_relaxed))
        {
            const auto m = currentMethod.load(std::memory_order_relaxed);
            level[static_cast<size_t>(m)] = 1.0f;
            if (m == static_cast<int>(params::Method::hybrid))
            {
                const auto component = [](float v)
                {
                    return params::kHybridComponents[static_cast<size_t>(
                        juce::jlimit(0, 4, juce::roundToInt(v)))];
                };
                const auto xf = knobs[31]->load();
                level[static_cast<size_t>(component(hybridA->load()))] = 1.0f - xf;
                level[static_cast<size_t>(component(hybridB->load()))] = xf;
            }
        }
        const float playing = active > 0 ? 1.0f : 0.0f;
        for (size_t i = 0; i < level.size(); ++i)
        {
            t.engineLevel[i].store(level[i] * playing);
            t.expertActivity[i].store(level[i] > 0.0f ? juce::jmax(0.25f, level[i]) * (0.4f + 0.6f * playing)
                                                      : 0.0f);
        }

        const auto tier = loadedTier.load();
        t.loadedTier.store(tier);
        t.modelIsStub.store(false);
        t.modelFrameMs.store(0.0f); // not measured per frame by the host yet
    }

    juce::AudioProcessorValueTreeState& state;
    juce::AudioProcessor& processor;
    inf::ModelLibrary library;
    inf::InferenceHost host;
    dsp::Synth synth;

    std::atomic<float>* method = nullptr;
    std::atomic<float>* hybridA = nullptr;
    std::atomic<float>* hybridB = nullptr;
    std::array<std::atomic<float>*, params::kNumKnobs> knobs{};
    std::atomic<float>* voiceLimit = nullptr;
    std::atomic<float>* bendRange = nullptr;

    // Message thread
    std::atomic<int> requestedTier{static_cast<int>(params::Tier::standard)};
    int attemptedTier = -1;
    std::atomic<int> loadedTier{static_cast<int>(inf::kDefaultTier)};
    std::string loadedId;
    juce::String loadError;
    std::vector<float> grainBankData;
    dsp::GrainBank grainBank;
    int lastSentShape = -1;
    std::array<float, params::kNumKnobs> lastSentKnobs{};
    double sampleRate = 0.0;
    int maxBlock = 0;

    // Shared
    std::atomic<bool> patchValid{false};
    std::atomic<int> currentMethod{0};

    // Audio thread
    bool prepared = false;
    int64_t clock = 0;
    std::array<Slot, kSlots> slots{};
    uint32_t ageCounter = 0;
    float bend = 0.0f, modWheel = 0.0f, aftertouch = 0.0f;
    std::array<Event, kQueueSize> queue{};
    size_t queueRead = 0, queueWrite = 0, queueSize = 0;
    const dsp::VoiceFrame zeroFrame{};
};

} // namespace

std::unique_ptr<SynthBackend> createBackend(juce::AudioProcessorValueTreeState& state,
                                            juce::AudioProcessor& processor)
{
    inf::ModelLibrary library;
    std::string error;
    if (library.load(inf::ModelLibrary::defaultModelsDir(), error) && !library.entries().empty())
        return std::make_unique<RealBackend>(state, processor, std::move(library));
    return createStubBackend(state);
}

} // namespace alfa
