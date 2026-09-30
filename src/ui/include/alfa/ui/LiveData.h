#pragma once

#include "alfa/ui/EditorContext.h"

#include <juce_dsp/juce_dsp.h>

namespace alfa::ui
{

/// One copy of everything the visualisations draw, taken on the UI timer.
struct Snapshot
{
    static constexpr int kScopePoints = 512;
    static constexpr int kSpectrumBands = 64;

    std::array<float, 128> notes{};
    std::array<float, Telemetry::kControlVectorSize> controlVector{};
    std::array<float, params::kNumMethods> experts{};
    std::array<float, params::kNumMethods> engines{};
    std::array<float, params::kNumMethods> blend{};
    std::array<float, kScopePoints> scope{};
    std::array<float, kSpectrumBands> spectrum{}; // dB, -90..0
    float attackMs = 0, decayMs = 0, sustain = 0, releaseMs = 0;
    float peakL = 0, peakR = 0, cpu = 0;
    int voices = 0;
    int heldNotes = 0;
    int loadedTier = 1;
    bool modelIsStub = true;
    double sampleRate = 48000.0;
    int blockSize = 512;
};

/// Polls Telemetry and parameters at 30 Hz, keeps a short note history for the
/// piano roll, and tells views to repaint. The UI never waits on the audio
/// thread.
class LiveData : private juce::Timer
{
public:
    static constexpr int kHistoryLength = 90; // 3 s at 30 Hz

    explicit LiveData(EditorContext& context);
    ~LiveData() override;

    const Snapshot& snapshot() const noexcept { return snap; }

    /// Notes held in each of the last kHistoryLength frames, oldest first.
    const std::vector<std::array<bool, 128>>& noteHistory() const noexcept { return history; }

    struct Listener
    {
        virtual ~Listener() = default;
        virtual void liveDataUpdated() = 0;
    };

    void addListener(Listener* l) { listeners.add(l); }
    void removeListener(Listener* l) { listeners.remove(l); }

    /// Takes one snapshot now (tests and screenshot rendering call this).
    void update();

private:
    void timerCallback() override { update(); }
    void computeSpectrum(const std::array<float, 2048>& samples);

    EditorContext& ctx;
    Snapshot snap;
    std::vector<std::array<bool, 128>> history;
    juce::dsp::FFT fft{11};
    juce::dsp::WindowingFunction<float> window{2048, juce::dsp::WindowingFunction<float>::hann};
    juce::ListenerList<Listener> listeners;
};

/// Paint helpers shared by the Perform, Signal flow and Sound design views.
namespace viz
{
void pianoRoll(juce::Graphics&, juce::Rectangle<float>, const Theme&, const LiveData&);
void controlVector(juce::Graphics&, juce::Rectangle<float>, const Theme&, const Snapshot&);
void modelGraph(juce::Graphics&, juce::Rectangle<float>, const Theme&, const Snapshot&, bool showLabels);
void blendBars(juce::Graphics&, juce::Rectangle<float>, const Theme&, const Snapshot&, bool showLabels);
void waveform(juce::Graphics&, juce::Rectangle<float>, const Theme&, const Snapshot&, juce::Colour colour);
void envelope(juce::Graphics&, juce::Rectangle<float>, const Theme&, const Snapshot&);
void spectrum(juce::Graphics&, juce::Rectangle<float>, const Theme&, const Snapshot&);
void meter(juce::Graphics&, juce::Rectangle<float>, const Theme&, float peakL, float peakR);
} // namespace viz

} // namespace alfa::ui
