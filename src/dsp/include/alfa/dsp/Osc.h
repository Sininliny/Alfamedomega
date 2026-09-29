#pragma once

// OSC primitive: a harmonic oscillator read from single-cycle wavetables (ControlFrame.h,
// OscControls). Each hop builds frame j's 2048-sample table with one inverse real FFT and
// crossfades linearly from frame j-1's table to it while a phase accumulator runs at f0.

#include "alfa/dsp/Primitive.h"
#include "alfa/dsp/detail/Fft.h"

namespace alfa::dsp
{

class OscPrimitive
{
public:
    void prepare(const PrepareSpec& spec);
    void reset() noexcept;
    void beginHop(const HopContext& ctx) noexcept;
    void render(float* out, int n) noexcept;
    void skipHop(const HopContext& ctx) noexcept;

    /// Current phase in cycles, [0, 1). For tests.
    [[nodiscard]] double phase() const noexcept { return phase_; }

private:
    /// Fills table (kOscTableSize + guard) from frame->osc, zeroing harmonics at or above Nyquist
    /// for fundamental f0. A null frame gives silence.
    void buildTable(const VoiceFrame* frame, float f0, float* table) noexcept;
    /// Moves to the next hop: the current table becomes the previous one.
    void rotate(const HopContext& ctx) noexcept;
    void advanceDetuned(int n) noexcept;

    detail::RealFft fft_;
    detail::AlignedBuffer spectrum_;
    detail::AlignedBuffer tableA_;
    detail::AlignedBuffer tableB_;

    float* prev_ = nullptr;
    float* cur_ = nullptr;
    bool prevValid_ = false; // prev_ holds frame j-1's table
    bool curValid_ = false;  // cur_ holds frame j's table
    float prevF0_ = 440.0f;  // f0 at frame j-1's hop boundary (for its Nyquist cut)
    float curF0_ = 440.0f;
    bool started_ = false; // a hop has begun since reset()

    double sampleRate_ = 48000.0;
    double invSampleRate_ = 1.0 / 48000.0;
    int hop_ = 192;
    float invHop_ = 1.0f / 192.0f;

    double phase_ = 0.0;  // cycles, [0, 1)
    double phase2_ = 0.0; // detuned read (VoiceFrame::oscDetuneRatio), cycles
    bool detune_ = false; // detune level non-zero in frame j-1 or j
    float ratioStart_ = 1.0f;
    float ratioDelta_ = 0.0f;
    float levelStart_ = 0.0f;
    float levelDelta_ = 0.0f;
    int pos_ = 0; // sample index within the hop
    float f0Start_ = 440.0f;
    float f0Delta_ = 0.0f; // f0End - f0Start
};

} // namespace alfa::dsp
