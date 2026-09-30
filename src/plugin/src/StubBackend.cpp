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
/// shaped by a few of the method's knobs. It exists so the plugin makes sound
/// and the UI has something real to visualise; it is not an approximation of
/// the final synth.
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

        method = raw(params::id::method);
        hybridA = raw(params::id::hybridA);
        hybridB = raw(params::id::hybridB);
        for (int i = 0; i < params::kNumKnobs; ++i)
            knobs[static_cast<size_t>(i)] = raw(params::id::knob(i));
        voiceCount = raw(params::id::voices);
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
        const auto m =
            static_cast<Method>(juce::jlimit(0, kNumMethods - 1, juce::roundToInt(method->load())));
        updateEnvelopes(m);

        int pos = 0;
        for (const auto meta : midi)
        {
            renderRange(buffer, pos, meta.samplePosition, m);
            pos = meta.samplePosition;
            handleMidi(meta.getMessage(), t);
        }
        renderRange(buffer, pos, numSamples, m);

        publishTelemetry(t, m);
    }

private:
    using Method = params::Method;

    struct Voice
    {
        int note = -1;
        float velocity = 0.0f;
        float phase = 0.0f, modPhase = 0.0f, lp = 0.0f, pluck = 0.0f;
        uint32_t age = 0;
        juce::ADSR env;
    };

    float knob(int i) const { return knobs[static_cast<size_t>(i)]->load(); }

    void updateEnvelopes(Method m)
    {
        // Knobs 18-21 are every method's amp envelope, with the teacher programs'
        // curves; physical has none and rings out on its decay knob instead.
        juce::ADSR::Parameters p;
        if (m == Method::physical)
        {
            p.attack = 0.001f;
            p.decay = 0.05f * std::exp2(knob(2) * 7.0f);
            p.sustain = knob(7) > 0.5f ? 0.7f : 0.0f;
            p.release = 0.05f + 0.3f * knob(4);
        }
        else
        {
            p.attack = 0.001f * std::exp2(knob(18) * 11.0f);
            p.decay = 0.01f * std::exp2(knob(19) * 9.0f);
            p.sustain = knob(20);
            p.release = 0.01f * std::exp2(knob(21) * 9.0f);
        }
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

    float sample(Method m, Voice& v, float lpCoeff) const
    {
        const float ph = v.phase * kTwoPi;
        switch (m)
        {
        case Method::subtractive:
        {
            const float saw = 2.0f * v.phase - 1.0f;
            const float pulse = v.phase < 0.05f + 0.9f * knob(3) ? 1.0f : -1.0f;
            const float tri = 1.0f - 4.0f * std::abs(v.phase - 0.5f);
            const float a = knob(0) * knob(0), b = knob(1) * knob(1), c = knob(2) * knob(2);
            const float mix = (a * saw + b * pulse + c * tri) / (a + b + c + 1e-6f);
            v.lp += lpCoeff * (mix - v.lp);
            return v.lp;
        }
        case Method::wavetable:
            return std::tanh(std::sin(ph) * (1.0f + 8.0f * knob(1)));
        case Method::additive:
        {
            const float roll = 0.5f + 2.5f * knob(0);
            float s = 0.0f;
            for (int h = 1; h <= 6; ++h)
                s += std::sin(ph * static_cast<float>(h)) * std::pow(static_cast<float>(h), -roll) *
                     (h % 2 == 0 ? 1.0f - knob(1) : 1.0f);
            return s * 0.6f;
        }
        case Method::spectral:
            return std::sin(ph) + (1.0f - knob(6)) * 0.3f * (random.nextFloat() * 2.0f - 1.0f);
        case Method::fm:
        {
            const float index = knob(14) * 8.0f;
            return std::sin(ph + index * std::sin(v.modPhase * kTwoPi));
        }
        case Method::physical:
            return std::sin(ph) * v.pluck;
        case Method::granular:
            return std::sin(ph) * (0.5f + 0.5f * std::sin(ph * 0.013f));
        case Method::sampling:
            return 1.0f - 4.0f * std::abs(v.phase - 0.5f);
        case Method::hybrid:
            break;
        }
        return 0.0f;
    }

    void renderRange(juce::AudioBuffer<float>& buffer, int start, int end, Method m)
    {
        if (end <= start)
            return;

        const float cutoff = juce::jlimit(20.0f, 18000.0f, 30.0f * std::exp2(knob(6) * 9.5f));
        const float lpCoeff = 1.0f - std::exp(-kTwoPi * cutoff / static_cast<float>(sampleRate));
        const float fmRatio = std::round(1.0f + knob(2) * 7.0f);
        const float damping = 0.99990f - 0.0005f * knob(3);

        // A hybrid plays both components, crossfaded by knob 32.
        const auto componentAt = [](float v)
        {
            return params::kHybridComponents[static_cast<size_t>(juce::jlimit(0, 4, juce::roundToInt(v)))];
        };
        const auto a = m == Method::hybrid ? componentAt(hybridA->load()) : m;
        const auto b = componentAt(hybridB->load());
        const float xf = m == Method::hybrid ? knob(31) : 0.0f;
        const bool valid = m != Method::hybrid || params::isValidHybridPair(a, b);

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
                float s = sample(a, v, lpCoeff) * (1.0f - xf);
                if (xf > 0.0f)
                    s += sample(b, v, lpCoeff) * xf;
                if (!valid)
                    s = 0.0f; // the contract: a hybrid without two valid components is silent

                const float out = s * v.velocity * v.env.getNextSample() * 0.25f;
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

    void publishTelemetry(Telemetry& t, Method m)
    {
        int active = 0;
        for (auto& v : voices)
            if (v.env.isActive())
                ++active;
        t.activeVoices.store(active);

        const float playing = active > 0 ? 1.0f : 0.0f;
        const auto componentAt = [](float v)
        {
            return params::kHybridComponents[static_cast<size_t>(juce::jlimit(0, 4, juce::roundToInt(v)))];
        };
        std::array<float, kNumMethods> level{};
        if (m == Method::hybrid)
        {
            level[static_cast<size_t>(Method::hybrid)] = 1.0f;
            level[static_cast<size_t>(componentAt(hybridA->load()))] = 1.0f - knob(31);
            level[static_cast<size_t>(componentAt(hybridB->load()))] = knob(31);
        }
        else
        {
            level[static_cast<size_t>(m)] = 1.0f;
        }

        // Each voice runs its method's expert (a hybrid runs the hybrid expert and
        // both components' experts), which is what the model graph lights up.
        for (size_t i = 0; i < level.size(); ++i)
        {
            t.engineLevel[i].store(level[i] * playing);
            t.expertActivity[i].store(level[i] > 0.0f ? juce::jmax(0.25f, level[i]) * (0.4f + 0.6f * playing)
                                                      : 0.0f);
        }

        // The stub has no model, so the "control vector" is a readable function of
        // what a real model would see: the knobs, with a little movement while
        // notes play.
        framePhase += 0.05f;
        for (size_t i = 0; i < t.controlVector.size(); ++i)
        {
            const float a = knobs[i * 2]->load(), b = knobs[i * 2 + 1]->load();
            const float wobble =
                playing * 0.05f * std::sin(framePhase * (1.0f + 0.37f * static_cast<float>(i)));
            t.controlVector[i].store(juce::jlimit(0.0f, 1.0f, 0.5f * (a + b) + wobble));
        }

        t.loadedTier.store(tier.load());
        t.modelIsStub.store(true);
        t.modelFrameMs.store(0.0f);
    }

    double sampleRate = 48000.0;
    std::array<Voice, kMaxVoices> voices{};
    uint32_t ageCounter = 0;
    float bend = 0.0f, framePhase = 0.0f;
    mutable juce::Random random;
    std::atomic<int> tier{1};

    std::atomic<float>* method = nullptr;
    std::atomic<float>* hybridA = nullptr;
    std::atomic<float>* hybridB = nullptr;
    std::array<std::atomic<float>*, params::kNumKnobs> knobs{};
    std::atomic<float>* voiceCount = nullptr;
    std::atomic<float>* bendRange = nullptr;
};
} // namespace

std::unique_ptr<SynthBackend> createStubBackend(juce::AudioProcessorValueTreeState& state)
{
    return std::make_unique<StubBackend>(state);
}

} // namespace alfa
