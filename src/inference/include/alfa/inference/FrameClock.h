#pragma once

#include <alfa/dsp/ControlFrame.h>
#include <cstdint>

namespace alfa::inference
{

/// Maps the host's sample clock onto the network's frame clock.
///
/// A frame boundary falls every alfa::dsp::hopSamples(sampleRate) samples (192 at 48 kHz, 176
/// at 44.1 kHz), the same integer hop the DSP renders with, so the DSP and the network never
/// drift apart. Real-time safe.
class FrameClock
{
public:
    void prepare(double sampleRate) noexcept
    {
        hop_ = dsp::hopSamples(sampleRate);
        reset();
    }

    void reset() noexcept
    {
        untilNextFrame_ = 0;
        framesElapsed_ = 0;
    }

    /// Samples per frame.
    [[nodiscard]] int hop() const noexcept { return hop_; }

    /// Largest number of frame boundaries a block of maxBlockSize samples can contain.
    [[nodiscard]] int maxFramesPerBlock(int maxBlockSize) const noexcept
    {
        return (maxBlockSize - 1) / hop_ + 1;
    }

    /// Calls onFrame(offsetInBlock) for every frame boundary inside the next numSamples samples,
    /// in order, then advances the clock. Returns how many boundaries there were.
    template <typename Fn>
    int advance(int numSamples, Fn&& onFrame) noexcept
    {
        int count = 0;
        int offset = untilNextFrame_;
        for (; offset < numSamples; offset += hop_)
        {
            onFrame(offset);
            ++count;
        }
        untilNextFrame_ = offset - numSamples;
        framesElapsed_ += count;
        return count;
    }

    [[nodiscard]] std::int64_t framesElapsed() const noexcept { return framesElapsed_; }

private:
    int hop_ = 192;
    int untilNextFrame_ = 0;
    std::int64_t framesElapsed_ = 0;
};

} // namespace alfa::inference
