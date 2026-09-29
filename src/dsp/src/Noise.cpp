#include "alfa/dsp/Noise.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp
{

void NoisePrimitive::prepare(const PrepareSpec& spec)
{
    hop_ = spec.hop;
    const int grainLength = 2 * hop_;
    fftSize_ = std::max(32, detail::nextPowerOfTwo(grainLength));
    fft_.prepare(fftSize_);
    bands_.prepare(spec.sampleRate, fftSize_);
    window_ = detail::periodicHann(grainLength);
    spectrum_.resize(static_cast<std::size_t>(fftSize_));
    grain_.resize(static_cast<std::size_t>(fftSize_));
    tail_.assign(static_cast<std::size_t>(hop_), 0.0f);
    hopOut_.assign(static_cast<std::size_t>(hop_), 0.0f);
    // Bins get (a + ib) * gain with a, b uniform in [-1, 1) (variance 1/3 each). The unscaled
    // inverse gives variance ~ 4/3 per bin * fftSize/2 bins = 2 fftSize/3; scale to 1/3.
    scale_ = static_cast<float>(1.0 / std::sqrt(2.0 * fftSize_));
    reset();
}

void NoisePrimitive::reset() noexcept
{
    std::fill(tail_.begin(), tail_.end(), 0.0f);
    std::fill(hopOut_.begin(), hopOut_.end(), 0.0f);
    pos_ = 0;
}

void NoisePrimitive::beginHop(const HopContext& ctx) noexcept
{
    const float* gains = ctx.cur->noise.band.data();
    float* s = spectrum_.data();
    const int half = fftSize_ / 2;
    s[0] = 0.0f;
    s[1] = bands_.gain(gains, half) * scale_ * random_.nextBipolar();
    for (int k = 1; k < half; ++k)
    {
        const float g = bands_.gain(gains, k) * scale_;
        const auto i = static_cast<std::size_t>(2 * k);
        s[i] = g * random_.nextBipolar();
        s[i + 1] = g * random_.nextBipolar();
    }
    fft_.inverse(s, grain_.data());

    const float* g = grain_.data();
    const float* w = window_.data();
    for (int t = 0; t < hop_; ++t)
    {
        const auto i = static_cast<std::size_t>(t);
        const auto j = static_cast<std::size_t>(t + hop_);
        hopOut_[i] = tail_[i] + g[i] * w[i];
        tail_[i] = g[j] * w[j];
    }
    pos_ = 0;
}

void NoisePrimitive::render(float* out, int n) noexcept
{
    std::copy_n(hopOut_.data() + pos_, n, out);
    pos_ += n;
}

void NoisePrimitive::skipHop(const HopContext&) noexcept
{
    std::fill(tail_.begin(), tail_.end(), 0.0f);
    pos_ = 0;
}

} // namespace alfa::dsp
