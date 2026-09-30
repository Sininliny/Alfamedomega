#pragma once

// SPECTRAL family (Family::Spectral, instance slots, stereo): STFT-domain processing built from
// small bin operations a network can address independently.
//
// The bin operations (namespace spectral) each own their parameters and state, and each is
// bypassed exactly at its neutral setting (active() == false):
//
//   op              domain     params (neutral)                        what it does
//   Freeze          complex    amount (0)                              captures the spectrum when amount
//                                                                      rises above 0, then resynthesises
//                                                                      it with coherent phase advance;
//                                                                      output = lerp(live, frozen, amount)
//   Smear           magnitude  time constant s (0)                     one-pole average of magnitudes
//                                                                      over frames (a spectral tail)
//   Blur            magnitude  radius in bins (0)                      box average of magnitudes across
//                                                                      neighbouring bins
//   BandShape       magnitude  64 band gains dB (0), tilt dB/oct (0)   gain curve on the shared 64-band
//                                                                      log layout (Bands.h), tilt pivots
//                                                                      at 1 kHz; total capped at +36 dB
//   Gate            magnitude  threshold dB re frame max (<= -96 off), keeps bins above the threshold
//                              knee dB, invert                         with a smoothstep soft knee;
//                                                                      invert keeps the bins below
//   Sparsify        magnitude  keep fraction (1)                       keeps the strongest fraction of
//                                                                      bins (top-N), zeroes the rest
//   FrequencyShift  complex    shift Hz (0)                            linear shift of all bins by a
//                                                                      fixed Hz amount (not harmonic)
//   PhaseMorph      complex    amount -1..1 (0)                        -1 robotise (zero phase, frame
//                                                                      centred), +1 whisperise (uniform
//                                                                      random phase), between: blend /
//                                                                      partial randomisation (blur)
//
// Magnitude operations work on an array of bin magnitudes; the processor measures the
// magnitudes once, runs every active magnitude op, and rescales the complex bins by new / old,
// so the analysed phases are kept. Complex operations work on the spectrum in the RealFft
// ordered layout (Fft.h).
//
// SpectralProcessor runs them in a fixed order:
//   Freeze -> Smear -> Blur -> BandShape -> Gate -> Sparsify -> FrequencyShift -> PhaseMorph
// on a SpectralStft (detail/SpectralStft.h) with sqrt-Hann windows at 75% overlap, then mixes
// with the dry signal delayed by the same latency. With every op at its neutral value the output
// is the input delayed by latencySamples() = fftSize, to float precision.
//
// Params (slot param i = entry i):
//    0 fft size        Choice 512 / 1024 / 2048 (default 1024); takes effect at reset()
//    1 band gain       -24 .. 24 dB   (0)     height of a Gaussian bump drawn on the 64 bands
//    2 band centre     40 .. 16000 Hz, log (1000)
//    3 band width      0.25 .. 8 octaves, log (2); the bump's full width at about 60% height
//    4 tilt            -6 .. 6 dB/octave (0), pivot 1 kHz
//    5 gate threshold  -96 .. 0 dB relative to the frame's loudest bin; -96 (x = 0) is off
//    6 gate knee       0 .. 24 dB (6)
//    7 gate invert     Toggle (off)
//    8 sparsify keep   1 .. 0.01 fraction of bins kept, log (1 = off)
//    9 freeze          0 .. 1 (0)
//   10 smear           0 .. 2 s time constant (0 = off)
//   11 blur            0 .. 32 bins radius (0 = off)
//   12 frequency shift -2000 .. 2000 Hz (0)
//   13 phase           -1 .. 1 (0): -1 robot, +1 whisper
//   14 mix             0 .. 1 (1) wet / dry (dry is latency compensated)
//   15 output gain     -24 .. 24 dB (0)

#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/Bands.h"
#include "alfa/dsp/detail/FastMath.h"
#include "alfa/dsp/detail/SpectralStft.h"

#include <array>
#include <vector>

namespace alfa::dsp
{

namespace spectral
{

/// What an op needs to know about the frame it is given.
struct FrameInfo
{
    int fftSize = 0;
    int hop = 0;
    double sampleRate = 48000.0;
    [[nodiscard]] int bins() const noexcept { return fftSize / 2 + 1; }
};

/// Magnitude of every bin of an ordered spectrum, 0 .. fftSize/2 (DC and Nyquist signed values
/// are taken by absolute value).
void magnitudes(const float* spectrum, int fftSize, float* mag) noexcept;

class Freeze
{
public:
    void prepare(int maxFftSize);
    void reset() noexcept;
    void setAmount(float amount) noexcept;
    [[nodiscard]] bool active() const noexcept { return amount_ > 0.0f; }
    [[nodiscard]] bool captured() const noexcept { return captured_; }
    /// Call every frame, also when inactive (it keeps the previous frame for the frequency
    /// estimate at capture time).
    void apply(float* spectrum, const FrameInfo& info) noexcept;

private:
    std::vector<float> prev_;    // previous live spectrum
    std::vector<float> mag_;     // frozen magnitude per bin (DC / Nyquist signed)
    std::vector<float> phase_;   // frozen synthesis phase
    std::vector<float> advance_; // phase advance per frame
    float amount_ = 0.0f;
    bool havePrev_ = false;
    bool captured_ = false;
};

class Smear
{
public:
    void prepare(int maxFftSize);
    void reset() noexcept { primed_ = false; }
    void setTime(float seconds) noexcept;
    [[nodiscard]] bool active() const noexcept { return seconds_ > 0.0f; }
    void apply(float* mag, const FrameInfo& info) noexcept;

private:
    std::vector<float> state_;
    float seconds_ = 0.0f;
    bool primed_ = false;
};

class Blur
{
public:
    void prepare(int maxFftSize);
    void setRadius(float bins) noexcept;
    [[nodiscard]] bool active() const noexcept { return radius_ > 0.0f; }
    void apply(float* mag, const FrameInfo& info) noexcept;

private:
    std::vector<double> prefix_;
    float radius_ = 0.0f;
};

class BandShape
{
public:
    /// Prepares band maps for every power-of-two FFT size in [minFftSize, maxFftSize].
    void prepare(double sampleRate, int minFftSize, int maxFftSize);
    /// kBands gains in dB on the Bands.h layout.
    void setBandsDb(const float* db) noexcept;
    void setTilt(float dbPerOctave) noexcept;
    [[nodiscard]] bool active() const noexcept { return active_; }
    void apply(float* mag, const FrameInfo& info) const noexcept;

    /// Centre frequency of band b at the prepared sample rate.
    [[nodiscard]] double bandHz(int b) const noexcept;
    /// Draws a Gaussian bump in log-frequency into db[kBands]: gainDb at centreHz, falling to
    /// about 60% of it widthOctaves / 2 either side.
    void bump(float* db, float gainDb, float centreHz, float widthOctaves) const noexcept;

private:
    void update() noexcept;

    std::array<detail::BandMap, 17> maps_; // by log2(fftSize)
    std::array<float, kBands> db_{};
    std::array<float, kBands> gain_{};
    double sampleRate_ = 48000.0;
    float tilt_ = 0.0f;
    bool active_ = false;
};

class Gate
{
public:
    static constexpr float kOffDb = -96.0f;
    void setThreshold(float dbRelativeToMax) noexcept;
    void setKnee(float db) noexcept;
    void setInvert(bool invert) noexcept { invert_ = invert; }
    [[nodiscard]] bool active() const noexcept { return thresholdDb_ > kOffDb; }
    void apply(float* mag, const FrameInfo& info) const noexcept;

private:
    float thresholdDb_ = kOffDb;
    float kneeDb_ = 0.0f;
    bool invert_ = false;
};

class Sparsify
{
public:
    void prepare(int maxFftSize);
    void setKeep(float fraction) noexcept;
    [[nodiscard]] bool active() const noexcept { return keep_ < 1.0f; }
    void apply(float* mag, const FrameInfo& info) noexcept;

private:
    std::vector<float> scratch_;
    float keep_ = 1.0f;
};

class FrequencyShift
{
public:
    void reset() noexcept { rotation_ = 0.0; }
    void setShift(float hz) noexcept;
    [[nodiscard]] bool active() const noexcept { return hz_ != 0.0f; }
    /// Moves every bin by round(shift / binHz) bins and rotates the frame's phase by the whole
    /// shift's phase advance per hop, so a partial at f comes out at exactly f + shift.
    void apply(float* spectrum, const FrameInfo& info) noexcept;

private:
    double rotation_ = 0.0;
    float hz_ = 0.0f;
};

class PhaseMorph
{
public:
    void reset() noexcept { random_.setSeed(seed_); }
    void setSeed(unsigned int seed) noexcept
    {
        seed_ = seed;
        random_.setSeed(seed);
    }
    void setAmount(float amount) noexcept;
    [[nodiscard]] bool active() const noexcept { return amount_ != 0.0f; }
    void apply(float* spectrum, const FrameInfo& info) noexcept;

private:
    detail::Random random_;
    unsigned int seed_ = 0x9E3779B9u;
    float amount_ = 0.0f;
};

} // namespace spectral

class SpectralProcessor final : public Processor
{
public:
    static constexpr int kMinFftSize = 512;
    static constexpr int kMaxFftSize = 2048;

    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;
    /// Output delay relative to the input, for latency compensation (= FFT size).
    [[nodiscard]] int latencySamples() const noexcept { return fftSize_; }

private:
    struct Channel
    {
        detail::SpectralStft stft;
        spectral::Freeze freeze;
        spectral::Smear smear;
        spectral::FrequencyShift shift;
        spectral::PhaseMorph phase;
        detail::Random random;  // phases for bins that gain energy from silence
        std::vector<float> dry; // latency-compensation delay line
        int dryPos = 0;
    };

    void processFrame(Channel& c, float* spectrum) noexcept;
    void processChannel(Channel& c, float* x, int n, float mixFrom, float mixTo, float gainFrom,
                        float gainTo) noexcept;

    std::array<Channel, 2> ch_;
    spectral::Blur blur_;
    spectral::BandShape band_;
    spectral::Gate gate_;
    spectral::Sparsify sparse_;
    std::vector<float> mag_;
    std::vector<float> mag0_;
    std::array<float, kControlBlock> dryBlock_{};

    double sampleRate_ = 48000.0;
    int channels_ = 2;
    int fftSize_ = 1024;
    int pendingFftSize_ = 1024;
    std::array<float, 4> bandParams_{}; // gain, centre, width, tilt of the last update
    bool bandValid_ = false;
    float smearSeconds_ = 0.0f;
    bool freezeOn_ = false;
    float mix_ = 1.0f;
    float mixTarget_ = 1.0f;
    float gain_ = 1.0f;
    float gainTarget_ = 1.0f;
};

} // namespace alfa::dsp
