#pragma once

#include "alfa/dsp/Primitive.h"
#include "alfa/dsp/detail/Bands.h"
#include "alfa/dsp/detail/Fft.h"

#include <vector>

namespace alfa::dsp
{

/// SAMPLER primitive (ControlFrame.h): plays the loaded sample at (f0 / root) * rate and runs
/// it through a 64-band short-time EQ.
///
/// The EQ uses 2-hop Hann frames at 50% overlap. To avoid adding a hop of latency, the raw read
/// runs two hops ahead of the output: at hop boundary j the sampler reads raw samples for hop
/// j+1 (using frame j's rate and the end-of-hop f0) and equalises the frame covering hops j and
/// j+1. Rate changes therefore reach the output one hop late, which is inaudible for pitch and
/// keeps note-on transients exact: at note-on the first two hops are read immediately and a
/// frame starting one hop before the note (silence plus the attack) completes the overlap.
/// When every EQ band is exactly 1 the FFT is skipped and the output equals the raw read.
class SamplerPrimitive
{
public:
    void prepare(const PrepareSpec& spec);
    void setSample(const SampleData* sample) noexcept { pending_ = sample; }
    void reset() noexcept;
    void beginHop(const HopContext& ctx) noexcept;
    void render(float* out, int n) noexcept;
    void skipHop(const HopContext& ctx) noexcept;

private:
    void readRaw(float* dst, int n, double increment) noexcept;
    void equalise(const float* rawFrame, const float* eq, int outOffset, int frameSkip) noexcept;
    [[nodiscard]] double incrementFor(const HopContext& ctx) const noexcept;
    void advance(const HopContext& ctx, bool equaliseFrame) noexcept;

    double sampleRate_ = 48000.0;
    int hop_ = 0;
    int fftSize_ = 0;
    detail::RealFft fft_;
    detail::BandMap bands_;
    std::vector<float> window_;
    detail::AlignedBuffer frame_;
    std::vector<float> raw_;     // 2 hops: raw read for [jH, (j+2)H)
    std::vector<float> acc_;     // overlap-add accumulator starting at the current hop
    std::vector<float> scratch_; // 2 hops, stages the note-on frame

    const SampleData* pending_ = nullptr;
    const SampleData* sample_ = nullptr;
    double readPos_ = 0.0;
    bool finished_ = true;

    float gainFrom_ = 0.0f;
    float gainStep_ = 0.0f;
    int pos_ = 0;
};

} // namespace alfa::dsp
