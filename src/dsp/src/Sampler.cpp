#include "alfa/dsp/Sampler.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp
{

void SamplerPrimitive::prepare(const PrepareSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    hop_ = spec.hop;
    const int frameLength = 2 * hop_;
    fftSize_ = std::max(32, detail::nextPowerOfTwo(frameLength));
    fft_.prepare(fftSize_);
    bands_.prepare(spec.sampleRate, fftSize_);
    window_ = detail::periodicHann(frameLength);
    frame_.resize(static_cast<std::size_t>(fftSize_));
    raw_.assign(static_cast<std::size_t>(frameLength), 0.0f);
    acc_.assign(static_cast<std::size_t>(fftSize_ + hop_), 0.0f);
    scratch_.assign(static_cast<std::size_t>(frameLength), 0.0f);
    reset();
}

void SamplerPrimitive::reset() noexcept
{
    sample_ = pending_;
    finished_ = sample_ == nullptr || sample_->data == nullptr || sample_->length < 2;
    readPos_ = 0.0;
    std::fill(raw_.begin(), raw_.end(), 0.0f);
    std::fill(acc_.begin(), acc_.end(), 0.0f);
    gainFrom_ = 0.0f;
    gainStep_ = 0.0f;
    pos_ = 0;
}

double SamplerPrimitive::incrementFor(const HopContext& ctx) const noexcept
{
    if (finished_)
        return 0.0;
    const double root = sample_->rootHz > 0.0 ? sample_->rootHz : 261.6255653;
    return static_cast<double>(ctx.f0End) / root * static_cast<double>(ctx.cur->sampler.rate) *
           sample_->sampleRate / sampleRate_;
}

void SamplerPrimitive::readRaw(float* dst, int n, double increment) noexcept
{
    if (finished_)
    {
        std::fill_n(dst, n, 0.0f);
        return;
    }
    const float* data = sample_->data;
    const long long length = sample_->length;
    const long long loopStart = sample_->loopStart;
    const long long loopEnd = sample_->loopEnd;
    const bool looped = loopStart >= 0 && loopEnd > loopStart + 1 && loopEnd <= length;
    const long long end = looped ? loopEnd : length;
    const auto loopLength = static_cast<double>(loopEnd - loopStart);

    for (int t = 0; t < n; ++t)
    {
        if (readPos_ < 0.0 || readPos_ >= static_cast<double>(end))
        {
            finished_ = true;
            std::fill_n(dst + t, n - t, 0.0f);
            return;
        }
        const auto i = static_cast<long long>(readPos_);
        const auto frac = static_cast<float>(readPos_ - static_cast<double>(i));
        const float a = data[i];
        float b = 0.0f;
        if (i + 1 < end)
            b = data[i + 1];
        else if (looped)
            b = data[loopStart];
        dst[t] = a + frac * (b - a);

        readPos_ += increment;
        if (looped && readPos_ >= static_cast<double>(loopEnd))
            readPos_ -= loopLength;
    }
}

void SamplerPrimitive::equalise(const float* rawFrame, const float* eq, int outOffset, int frameSkip) noexcept
{
    const int frameLength = 2 * hop_;
    const bool flat = std::all_of(eq, eq + kBands, [](float g) { return g == 1.0f; });
    const float* w = window_.data();

    if (flat)
    {
        for (int t = frameSkip; t < frameLength; ++t)
            acc_[static_cast<std::size_t>(t + outOffset)] += rawFrame[t] * w[t];
        return;
    }

    float* f = frame_.data();
    for (int t = 0; t < frameLength; ++t)
        f[t] = rawFrame[t] * w[t];
    std::fill(f + frameLength, f + fftSize_, 0.0f);
    fft_.forward(f, f);

    const int half = fftSize_ / 2;
    const float scale = 1.0f / static_cast<float>(fftSize_);
    f[0] *= bands_.gain(eq, 0) * scale;
    f[1] *= bands_.gain(eq, half) * scale;
    for (int k = 1; k < half; ++k)
    {
        const float g = bands_.gain(eq, k) * scale;
        f[2 * k] *= g;
        f[2 * k + 1] *= g;
    }
    fft_.inverse(f, f);
    for (int t = frameSkip; t < fftSize_; ++t)
        acc_[static_cast<std::size_t>(t + outOffset)] += f[t];
}

void SamplerPrimitive::advance(const HopContext& ctx, bool equaliseFrame) noexcept
{
    const float* eq = ctx.cur->sampler.eq.data();
    const auto h = static_cast<std::size_t>(hop_);

    if (ctx.firstHop)
    {
        std::fill(acc_.begin(), acc_.end(), 0.0f);
        if (!finished_)
        {
            const float start = std::clamp(ctx.cur->sampler.start, 0.0f, 1.0f);
            readPos_ = static_cast<double>(start) * static_cast<double>(sample_->length - 1);
        }
        readRaw(raw_.data(), 2 * hop_, incrementFor(ctx));
        if (equaliseFrame)
        {
            // Frame -1 covers [-H, H): silence, then the first hop of the note. Only its part
            // from time 0 onwards lands in the accumulator.
            std::fill_n(scratch_.data(), hop_, 0.0f);
            std::copy_n(raw_.data(), hop_, scratch_.data() + hop_);
            equalise(scratch_.data(), eq, -hop_, hop_);
            equalise(raw_.data(), eq, 0, 0);
        }
        return;
    }

    std::copy(acc_.begin() + static_cast<std::ptrdiff_t>(h), acc_.end(), acc_.begin());
    std::fill(acc_.end() - static_cast<std::ptrdiff_t>(h), acc_.end(), 0.0f);
    std::copy_n(raw_.data() + h, h, raw_.data());
    readRaw(raw_.data() + h, hop_, incrementFor(ctx));
    if (equaliseFrame)
        equalise(raw_.data(), eq, 0, 0);
}

void SamplerPrimitive::beginHop(const HopContext& ctx) noexcept
{
    advance(ctx, true);
    gainFrom_ = ctx.prev->sampler.gain;
    gainStep_ = (ctx.cur->sampler.gain - gainFrom_) / static_cast<float>(hop_);
    pos_ = 0;
}

void SamplerPrimitive::render(float* out, int n) noexcept
{
    const float* a = acc_.data() + pos_;
    for (int t = 0; t < n; ++t)
        out[t] = a[t] * (gainFrom_ + gainStep_ * static_cast<float>(pos_ + t));
    pos_ += n;
}

void SamplerPrimitive::skipHop(const HopContext& ctx) noexcept
{
    advance(ctx, false);
    std::fill(acc_.begin(), acc_.end(), 0.0f);
    pos_ = 0;
}

} // namespace alfa::dsp
