#include "SynthBackend.h"

#include <cmath>

namespace alfa
{
namespace
{
using params::kNumMethods;

constexpr int kMaxVoices = 32;
constexpr float kTwoPi = juce::MathConstants<float>::twoPi;

/// Stand-in for the real engines: one cheap waveform per synthesis method,
/// mixed by the Method blend sliders. It exists so the plugin makes sound and
/// the UI has something real to visualise; it is not an approximation of the
/// final synth.
class StubBackend final : public SynthBackend
{
public:
    explicit StubBackend(juce::AudioProcessorValueTreeState& s)
    {
        auto raw = [&s](const juce::String& id)
        {
            auto* p = s.getRawParameterValue(id);
            jassert(p != nullptr);
            return p;
        };

        for (int m = 0; m < kNumMethods; ++m)
        {
            blend[static_cast<size_t>(m)] = raw(params::id::blend(m));
            for (int slot = 0; slot < params::kNumEngineSlots; ++slot)
                engine[static_cast<size_t>(m)][static_cast<size_t>(slot)] = raw(params::id::engine(m, slot));
        }

        for (int i = 0; i < params::kNumMacros; ++i)
            macros[static_cast<size_t>(i)] = raw(params::id::macro(i));

        attack = raw(params::id::attack);
        decay = raw(params::id::decay);
        sustain = raw(params::id::sustain);
        release = raw(params::id::release);
        voiceCount = raw(params::id::voices);
        timbreX = raw(params::id::timbreX);
        timbreY = raw(params::id::timbreY);
        drift = raw(params::id::latentDrift);
        bendRange = raw(params::id::bendRange);
    }

    void prepare(double newSampleRate, int) override
    {
        sampleRate = newSampleRate;
        for (auto& v : voices)
        {
            v = {};
            v.env.setSampleRate(sampleRate);
        }
    }

    void requestTier(params::Tier t) override { tier.store(static_cast<int>(t)); }

    int latencySamples(double sr) const override
    {
        const auto& info = params::kTiers[static_cast<size_t>(tier.load())];
        return juce::roundToInt(info.latencyMs * 0.001 * sr);
    }

    juce::String description() const override
    {
        return "Stub engines (DSP and inference components not linked yet)";
    }

    void render(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi, Telemetry& t) override
    {
        const auto numSamples = buffer.getNumSamples();
        updateEnvelopes();

        std::array<float, kNumMethods> weights{};
        float weightSum = 0.0f;
        for (size_t m = 0; m < weights.size(); ++m)
        {
            weights[m] = blend[m]->load();
            weightSum += weights[m];
        }
        const float norm = weightSum > 1.0f ? 1.0f / weightSum : 1.0f;

        int pos = 0;
        for (const auto meta : midi)
        {
            renderRange(buffer, pos, meta.samplePosition, weights, norm);
            pos = meta.samplePosition;
            handleMidi(meta.getMessage(), t);
        }
        renderRange(buffer, pos, numSamples, weights, norm);

        publishTelemetry(t, weights, norm);
    }

private:
    struct Voice
    {
        int note = -1;
        float velocity = 0.0f;
        float phase = 0.0f, modPhase = 0.0f, lp = 0.0f, pluck = 0.0f;
        uint32_t age = 0;
        juce::ADSR env;
    };

    void updateEnvelopes()
    {
        juce::ADSR::Parameters p;
        p.attack = attack->load() * 0.001f;
        p.decay = decay->load() * 0.001f;
        p.sustain = sustain->load();
        p.release = release->load() * 0.001f;
        for (auto& v : voices)
            v.env.setParameters(p);
    }

    void handleMidi(const juce::MidiMessage& msg, Telemetry& t)
    {
        if (msg.isNoteOn())
        {
            const auto limit = juce::jlimit(1, kMaxVoices, static_cast<int>(voiceCount->load()));
            auto* target = &voices[0];
            for (int i = 0; i < limit; ++i)
            {
                auto& v = voices[static_cast<size_t>(i)];
                if (!v.env.isActive())
                {
                    target = &v;
                    break;
                }
                if (v.age < target->age)
                    target = &v;
            }
            if (target->note >= 0)
                t.noteVelocity[static_cast<size_t>(target->note)].store(0.0f);

            target->note = msg.getNoteNumber();
            target->velocity = msg.getFloatVelocity();
            target->age = ++ageCounter;
            target->pluck = 1.0f;
            target->env.noteOn();
            t.noteVelocity[static_cast<size_t>(target->note)].store(target->velocity);
        }
        else if (msg.isNoteOff())
        {
            for (auto& v : voices)
                if (v.note == msg.getNoteNumber() && v.env.isActive())
                    v.env.noteOff();
            t.noteVelocity[static_cast<size_t>(msg.getNoteNumber())].store(0.0f);
        }
        else if (msg.isPitchWheel())
        {
            bend = (static_cast<float>(msg.getPitchWheelValue()) - 8192.0f) / 8192.0f * bendRange->load();
        }
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
        {
            for (auto& v : voices)
                v.env.reset();
            for (auto& n : t.noteVelocity)
                n.store(0.0f);
        }
    }

    float param(int method, int slot) const
    {
        return engine[static_cast<size_t>(method)][static_cast<size_t>(slot)]->load();
    }

    void renderRange(juce::AudioBuffer<float>& buffer, int start, int end,
                     const std::array<float, kNumMethods>& w, float norm)
    {
        if (end <= start)
            return;

        const float bright = timbreX->load();
        const float cutoff = param(0, 0);
        const float lpCoeff = 1.0f - std::exp(-kTwoPi * juce::jmin(cutoff * (0.5f + bright), 18000.0f) /
                                              static_cast<float>(sampleRate));
        const float fmRatio = param(2, 0);
        const float fmIndex = param(2, 1);
        const float wtPos = param(3, 0);
        const float damping = 0.99990f - 0.0005f * param(5, 1);

        auto* left = buffer.getWritePointer(0);
        auto* right = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;

        for (auto& v : voices)
        {
            if (!v.env.isActive())
                continue;

            const float freq =
                static_cast<float>(juce::MidiMessage::getMidiNoteInHertz(v.note)) * std::exp2(bend / 12.0f);
            const float inc = freq / static_cast<float>(sampleRate);

            for (int i = start; i < end; ++i)
            {
                const float ph = v.phase * kTwoPi;
                const float saw = 2.0f * v.phase - 1.0f;
                v.lp += lpCoeff * (saw - v.lp);

                float s = 0.0f;
                s += w[0] * v.lp; // subtractive
                s += w[1] * (std::sin(ph) + 0.5f * std::sin(2 * ph) + 0.33f * std::sin(3 * ph)) *
                     0.5f;                                                              // additive
                s += w[2] * std::sin(ph + fmIndex * std::sin(v.modPhase * kTwoPi));     // fm
                s += w[3] * std::tanh(std::sin(ph) * (1.0f + 8.0f * wtPos));            // wavetable
                s += w[4] * std::sin(ph) * (0.5f + 0.5f * std::sin(ph * 0.013f));       // granular
                s += w[5] * std::sin(ph) * v.pluck;                                     // physical
                s += w[6] * (1.0f - 4.0f * std::abs(v.phase - 0.5f));                   // sampling
                s += w[7] * (std::sin(ph) + 0.2f * (random.nextFloat() * 2.0f - 1.0f)); // spectral
                s += w[8] * std::sin(ph) * std::sin(ph * 1.5f);                         // hybrid

                const float out = s * norm * v.velocity * v.env.getNextSample() * 0.25f;
                left[i] += out;
                if (right != nullptr)
                    right[i] += out;

                v.phase += inc;
                v.phase -= std::floor(v.phase);
                v.modPhase += inc * fmRatio;
                v.modPhase -= std::floor(v.modPhase);
                v.pluck *= damping;
            }

            if (!v.env.isActive())
                v.note = -1;
        }
    }

    void publishTelemetry(Telemetry& t, const std::array<float, kNumMethods>& w, float norm)
    {
        int active = 0;
        float envSum = 0.0f;
        for (auto& v : voices)
        {
            if (v.env.isActive())
            {
                ++active;
                envSum = juce::jmax(envSum, v.velocity);
            }
        }
        t.activeVoices.store(active);

        const float playing = active > 0 ? 1.0f : 0.0f;
        for (size_t m = 0; m < w.size(); ++m)
            t.engineLevel[m].store(w[m] * norm * playing);

        // The stub has no model, so the "control vector" is a readable function of
        // the inputs a real model would see: timbre position, macros and drift.
        framePhase += 0.05f;
        const float x = timbreX->load(), y = timbreY->load(), d = drift->load();
        for (size_t i = 0; i < t.controlVector.size(); ++i)
        {
            const float base = i % 2 == 0 ? x : y;
            const float mac = macros[i % macros.size()]->load();
            const float wobble = d * 0.3f * std::sin(framePhase * (1.0f + 0.37f * static_cast<float>(i)));
            t.controlVector[i].store(juce::jlimit(0.0f, 1.0f, 0.5f * base + 0.3f * mac + 0.2f + wobble));
        }

        // Mixture-of-experts routing: the top engines by blend weight are active,
        // and bigger models keep more experts running at once.
        const int tierIndex = tier.load();
        const int topK = tierIndex + 2;
        std::array<size_t, kNumMethods> order{};
        for (size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&w](size_t a, size_t b) { return w[a] > w[b]; });
        for (size_t rank = 0; rank < order.size(); ++rank)
        {
            const auto m = order[rank];
            const bool on = static_cast<int>(rank) < topK && w[m] > 0.001f;
            t.expertActivity[m].store(on ? juce::jmax(0.2f, w[m] * norm) * (0.4f + 0.6f * playing) : 0.0f);
        }

        t.loadedTier.store(tierIndex);
        t.modelIsStub.store(true);
        t.modelFrameMs.store(0.0f);
    }

    double sampleRate = 48000.0;
    std::array<Voice, kMaxVoices> voices{};
    uint32_t ageCounter = 0;
    float bend = 0.0f, framePhase = 0.0f;
    juce::Random random;
    std::atomic<int> tier{1};

    std::array<std::atomic<float>*, kNumMethods> blend{};
    std::array<std::array<std::atomic<float>*, params::kNumEngineSlots>, kNumMethods> engine{};
    std::array<std::atomic<float>*, params::kNumMacros> macros{};
    std::atomic<float>* attack = nullptr;
    std::atomic<float>* decay = nullptr;
    std::atomic<float>* sustain = nullptr;
    std::atomic<float>* release = nullptr;
    std::atomic<float>* voiceCount = nullptr;
    std::atomic<float>* timbreX = nullptr;
    std::atomic<float>* timbreY = nullptr;
    std::atomic<float>* drift = nullptr;
    std::atomic<float>* bendRange = nullptr;
};
} // namespace

std::unique_ptr<SynthBackend> createStubBackend(juce::AudioProcessorValueTreeState& state)
{
    return std::make_unique<StubBackend>(state);
}

} // namespace alfa
