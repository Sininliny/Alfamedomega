#pragma once

// Streaming STFT engine and a phase-vocoder frame helper. Used by the SPECTRAL family
// (Spectral.h), the offline phase vocoder (PitchTime.h) and, later, by neural control of bins.
//
// Both classes allocate only in prepare() and are allocation-free, lock-free and noexcept
// afterwards. Spectra use the RealFft "ordered" layout (Fft.h): s[0] = DC, s[1] = Nyquist, then
// s[2k], s[2k+1] = Re, Im of bin k for k = 1 .. n/2 - 1.

#include "alfa/dsp/detail/Fft.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace alfa::dsp::detail
{

enum class StftWindow : unsigned char
{
    SqrtHann, ///< sqrt-Hann analysis and synthesis (their product is Hann): the default
    Hann,     ///< Hann analysis and synthesis (better sidelobes, product is Hann^2)
};

/// Wraps an angle into [-pi, pi).
[[nodiscard]] inline float wrapPhase(float x) noexcept
{
    constexpr float kTwoPiF = 6.28318530717958647692f;
    constexpr float kInvTwoPi = 0.15915494309189533577f;
    return x - kTwoPiF * std::floor(x * kInvTwoPi + 0.5f);
}

/// Streaming short-time Fourier transform with 75% overlap and weighted overlap-add.
///
/// Input arrives in blocks of any size. Every hop (fftSize / 4 samples) the last fftSize input
/// samples are windowed, transformed, handed to the frame callback as a spectrum it may modify
/// in place, transformed back, windowed again and overlap-added. The window pair is normalised
/// so an untouched spectrum reconstructs the input exactly (to float precision), delayed by
/// latency() = fftSize samples: a frame can only be emitted once all four frames overlapping
/// it have been analysed, and a hop is emitted while the next one is collected.
///
/// The FFT size can be any power of two in the prepared range; changing it resets the state.
class SpectralStft
{
public:
    static constexpr int kOverlap = 4;

    /// Allocates FFT setups and buffers for every power of two in [minFftSize, maxFftSize]
    /// (both powers of two, >= 32) and selects maxFftSize.
    void prepare(int minFftSize, int maxFftSize, StftWindow window = StftWindow::SqrtHann);

    /// Selects a prepared FFT size (rounded up to a power of two and clamped to the prepared
    /// range) and resets. Allocation-free.
    void setFftSize(int n) noexcept;
    void reset() noexcept;

    [[nodiscard]] int fftSize() const noexcept { return fftSize_; }
    [[nodiscard]] int hop() const noexcept { return hop_; }
    [[nodiscard]] int bins() const noexcept { return fftSize_ / 2 + 1; }
    [[nodiscard]] int latency() const noexcept { return fftSize_; }

    /// Processes n samples (in and out may alias). onFrame(float* spectrum) is called once per
    /// completed hop with the analysed frame; it is a template parameter so the call inlines
    /// (no std::function on the audio path).
    template <typename OnFrame>
    void process(const float* in, float* out, int n, OnFrame&& onFrame) noexcept
    {
        int i = 0;
        while (i < n)
        {
            const int k = std::min(n - i, hop_ - fill_);
            // Input first: when in == out the samples are consumed before being overwritten.
            std::copy_n(in + i, k, input_.data() + (fftSize_ - hop_ + fill_));
            std::copy_n(ready_.data() + fill_, k, out + i);
            fill_ += k;
            i += k;
            if (fill_ == hop_)
            {
                analyse();
                onFrame(frame_.data());
                synthesise();
                fill_ = 0;
            }
        }
    }

private:
    static constexpr int kMaxOrder = 16; // FFT sizes up to 2^16

    void analyse() noexcept;
    void synthesise() noexcept;

    std::array<RealFft, kMaxOrder + 1> ffts_;
    RealFft* fft_ = nullptr;
    StftWindow window_ = StftWindow::SqrtHann;
    int minFftSize_ = 0;
    int maxFftSize_ = 0;
    int fftSize_ = 0;
    int hop_ = 0;
    int fill_ = 0;

    AlignedBuffer analysisWindow_;
    AlignedBuffer synthesisWindow_; // includes the 1 / (fftSize * overlap sum) normalisation
    AlignedBuffer input_;           // the last fftSize input samples
    AlignedBuffer frame_;
    AlignedBuffer acc_;   // overlap-add accumulator, acc_[0] is the next hop to emit
    AlignedBuffer ready_; // the hop being emitted
};

/// Phase-vocoder view of one STFT frame: magnitude and instantaneous frequency per bin, and the
/// inverse (phase accumulation). Real-time safe after prepare(), so it can run inside a
/// SpectralStft callback, where a network can then edit magnitudes and frequencies directly.
///
/// analyse() estimates each bin's frequency from the phase advance since the previous analysed
/// frame, given the analysis hop in samples. synthesise() advances each bin's output phase by
/// frequency * synthesis hop and writes magnitude * e^{i phase}. With equal hops and untouched
/// arrays the round trip reproduces the analysed spectrum. With lockPhases (identity phase
/// locking, Laroche and Dolson 1999) only spectral peaks accumulate phase; the other bins keep
/// their analysed phase offset to the peak of their region, which keeps partials coherent under
/// time stretching.
///
/// The first frame after reset() (or after a size change) takes the analysed phases as they are
/// and reports every bin at its centre frequency.
class PhaseVocoderFrame
{
public:
    void prepare(int maxFftSize);
    void reset() noexcept;

    void analyse(const float* spectrum, int fftSize, double analysisHop) noexcept;
    void synthesise(float* spectrum, double synthesisHop, bool lockPhases = false) noexcept;

    [[nodiscard]] int bins() const noexcept { return fftSize_ / 2 + 1; }
    [[nodiscard]] int fftSize() const noexcept { return fftSize_; }
    /// Linear magnitude per bin, 0 .. fftSize/2. Writable before synthesise().
    [[nodiscard]] float* magnitude() noexcept { return mag_.data(); }
    /// Instantaneous frequency in bins (multiply by sampleRate / fftSize for Hz). Writable.
    [[nodiscard]] float* frequency() noexcept { return freq_.data(); }
    /// Phase of the last analysed frame, radians in [-pi, pi).
    [[nodiscard]] const float* phase() const noexcept { return phase_.data(); }

private:
    std::vector<float> mag_;
    std::vector<float> freq_;
    std::vector<float> phase_;     // analysis phase, current frame
    std::vector<float> prevPhase_; // analysis phase, previous frame
    std::vector<float> outPhase_;  // synthesis phase
    std::vector<int> peaks_;
    int fftSize_ = 0;
    bool haveAnalysis_ = false;
    bool haveSynthesis_ = false;
};

} // namespace alfa::dsp::detail
