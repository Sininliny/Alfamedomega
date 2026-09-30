#pragma once
#include "alfa/tensor/Model.h"
#include "alfa/tensor/Runtime.h"

#include <juce_audio_utils/juce_audio_utils.h>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace alfa::tensorhost
{
class Processor final : public juce::AudioProcessor,
                        private juce::Timer,
                        private juce::MidiKeyboardState::Listener
{
public:
    Processor();
    ~Processor() override;
    void prepareToPlay(double, int) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout&) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Alfamedomega Tensor"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Tensor configuration"; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;
    bool loadJSON(const juce::String&, juce::String& error);
    static juce::String referenceDocument(int index);
    juce::String document() const
    {
        std::lock_guard lock(configurationMutex_);
        return json_;
    }
    juce::String inspection() const
    {
        std::lock_guard lock(configurationMutex_);
        return inspection_;
    }
    std::atomic<int> mode{0}; // instrument, FX, hybrid; same executable tensor space.
    std::array<std::atomic<float>, 5> intentions{};
    std::atomic<uint64_t> samples{0}, failures{0};
    std::atomic<double> blockMilliseconds{0}, modelMilliseconds{0};
    std::array<std::atomic<float>, 16> stateSnapshot{};
    juce::MidiKeyboardState keyboard;

private:
    struct Mix
    {
        size_t output;
        int channel;
        float gain;
    };
    struct Program
    {
        alfa::tensor::Runtime runtime;
        std::vector<Mix> mix;
        std::unique_ptr<alfa::tensor::TensorModel> model;
        alfa::tensor::Clock modelClock;
        alfa::tensor::Tensor latentSound;
    };
    std::vector<std::unique_ptr<Program>> owned_;
    juce::MidiBuffer mergedMidi_;
    struct KeyEvent
    {
        int channel = 1, note = 60;
        float velocity = 0;
        bool on = false;
    };
    std::array<KeyEvent, 256> keyEvents_{};
    std::atomic<size_t> keyRead_{0}, keyWrite_{0};
    void queueKey(int channel, int note, float velocity, bool on) noexcept;
    void handleNoteOn(juce::MidiKeyboardState*, int, int, float) override;
    void handleNoteOff(juce::MidiKeyboardState*, int, int, float) override;
    std::atomic<Program*> pending_{nullptr};
    Program* active_ = nullptr;
    juce::String json_, inspection_;
    mutable std::mutex configurationMutex_;
    double rate_ = 48000;
    uint64_t clock_ = 0;
    std::array<float, 128 * 10> notes_{};
    size_t noteCount_ = 0;
    std::array<Program*, 64> retired_{};
    std::atomic<size_t> retireRead_{0}, retireWrite_{0};
    std::array<float, 16 * 3> gestures_{};
    std::array<float, 8> context_{};
    std::array<float, 5> preferences_{};
    std::array<std::string, 2> midiAxes_{"note", "feature"};
    std::array<size_t, 2> midiShape_{0, 10};
    std::array<std::string, 2> gestureAxes_{"channel", "feature"};
    std::array<size_t, 2> gestureShape_{16, 3};
    std::array<std::string, 1> audioAxes_{"channel"}, contextAxes_{"feature"}, preferenceAxes_{"intention"};
    std::array<size_t, 1> stereoShape_{2}, contextShape_{8}, preferenceShape_{5};
    void event(const juce::MidiMessage&) noexcept;
    void timerCallback() override;
};
} // namespace alfa::tensorhost
