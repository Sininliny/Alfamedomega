#pragma once

#include "alfa/dsp/Primitive.h"
#include "alfa/dsp/detail/Bands.h"
#include "alfa/dsp/detail/FastMath.h"
#include "alfa/dsp/detail/Fft.h"

#include <cstdint>
#include <vector>

namespace alfa::dsp
{

/// NOISE primitive (ControlFrame.h): each hop starts a 2-hop Hann grain of band-shaped white
/// noise, overlap-added at 50%.
///
/// The grain is synthesised directly in the frequency domain (a random complex spectrum times
/// the band curve, one inverse FFT), which has the same statistics as filtering windowed white
/// noise and costs one FFT instead of two. With every band at 1 the output has the variance of
/// uniform [-1, 1) noise (1/3) before windowing.
class NoisePrimitive
{
public:
    void prepare(const PrepareSpec& spec);
    void setSeed(std::uint32_t seed) noexcept { random_.setSeed(seed); }
    void reset() noexcept;
    void beginHop(const HopContext& ctx) noexcept;
    void render(float* out, int n) noexcept;
    void skipHop(const HopContext& ctx) noexcept;

private:
    int hop_ = 0;
    int fftSize_ = 0;
    float scale_ = 0.0f;
    detail::RealFft fft_;
    detail::BandMap bands_;
    detail::Random random_;
    std::vector<float> window_;   // periodic Hann, 2 hops
    detail::AlignedBuffer spectrum_;
    detail::AlignedBuffer grain_;  // fftSize_, first 2 hops used
    std::vector<float> tail_;     // second half of the previous grain
    std::vector<float> hopOut_;   // this hop's output
    int pos_ = 0;
};

} // namespace alfa::dsp
