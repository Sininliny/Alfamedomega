// AlfaTestSynth: three programs with known behaviour, for the render host's tests.
//   0 "Sine Pad"         pitched, every key
//   1 "Saw Bass C1-C4"   pitched, silent outside MIDI 24..60, so range finding has edges to find
//   2 "Noise Hit"        unpitched decaying noise
#include <juce_audio_processors/juce_audio_processors.h>

#include <array>
#include <cmath>

namespace
{
struct Voice
{
    int note = -1;
    float velocity = 0.0f;
    double phase = 0.0;
    double env = 0.0;
    bool releasing = false;
};

constexpr std::array<const char*, 3> kPrograms { "Sine Pad", "Saw Bass C1-C4", "Noise Hit" };
} // namespace

class TestSynth final : public juce::AudioProcessor
{
public:
    TestSynth()
        : AudioProcessor (BusesProperties().withOutput ("Out", juce::AudioChannelSet::stereo(), true))
    {
    }

    const juce::String getName() const override { return "AlfaTestSynth"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.5; }
    bool isBusesLayoutSupported (const BusesLayout& l) const override
    {
        return l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
    }

    int getNumPrograms() override { return (int) kPrograms.size(); }
    int getCurrentProgram() override { return program; }
    void setCurrentProgram (int i) override { program = juce::jlimit (0, getNumPrograms() - 1, i); }
    const juce::String getProgramName (int i) override { return kPrograms[(size_t) juce::jlimit (0, 2, i)]; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& dest) override { dest.append (&program, sizeof (program)); }
    void setStateInformation (const void* data, int size) override
    {
        if (size >= (int) sizeof (int))
            setCurrentProgram (*static_cast<const int*> (data));
    }

    bool hasEditor() const override { return false; }
    juce::AudioProcessorEditor* createEditor() override { return nullptr; }

    void prepareToPlay (double sr, int) override
    {
        sampleRate = sr;
        voice = {};
    }
    void releaseResources() override {}
    void reset() override { voice = {}; }

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override
    {
        buffer.clear();
        int pos = 0;
        for (const auto meta : midi)
        {
            render (buffer, pos, meta.samplePosition);
            pos = meta.samplePosition;
            const auto m = meta.getMessage();
            if (m.isNoteOn())
                voice = { m.getNoteNumber(), m.getFloatVelocity(), 0.0, program == 2 ? 1.0 : 0.0, false };
            else if ((m.isNoteOff() && m.getNoteNumber() == voice.note) || m.isAllNotesOff())
                voice.releasing = true;
        }
        render (buffer, pos, buffer.getNumSamples());
    }

private:
    void render (juce::AudioBuffer<float>& buffer, int start, int end)
    {
        if (voice.note < 0)
            return;
        const bool audible = program != 1 || (voice.note >= 24 && voice.note <= 60);
        const double f0 = 440.0 * std::pow (2.0, (voice.note - 69) / 12.0);
        const double attack = 1.0 / (0.01 * sampleRate), release = std::exp (-1.0 / (0.15 * sampleRate));
        const double decay = std::exp (-1.0 / (0.2 * sampleRate));
        for (int i = start; i < end; ++i)
        {
            if (program == 2)
                voice.env *= voice.releasing ? release : decay;
            else
                voice.env = voice.releasing ? voice.env * release : std::min (1.0, voice.env + attack);
            double s = 0.0;
            if (program == 0)
                s = std::sin (2.0 * juce::MathConstants<double>::pi * voice.phase);
            else if (program == 1)
                s = 2.0 * voice.phase - 1.0;
            else
                s = random.nextFloat() * 2.0f - 1.0f;
            voice.phase += f0 / sampleRate;
            voice.phase -= std::floor (voice.phase);
            const auto out = (float) (audible ? 0.5 * voice.velocity * voice.env * s : 0.0);
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                buffer.addSample (ch, i, out);
        }
    }

    double sampleRate = 48000.0;
    int program = 0;
    Voice voice;
    juce::Random random { 1234 };
};

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new TestSynth(); }
