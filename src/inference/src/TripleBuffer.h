#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace alfa::inference
{

/// Single-writer, single-reader latest-value hand-off. Neither side ever waits or allocates
/// (as long as T's copy assignment does not). The reader always sees a complete value.
template <typename T>
class TripleBuffer
{
public:
    /// Writer: fills the back slot, then publishes it.
    template <typename Fill>
    void write(Fill&& fill)
    {
        fill(slots_[back_]);
        const auto previous =
            state_.exchange(static_cast<std::uint8_t>(back_ | kFresh), std::memory_order_acq_rel);
        back_ = previous & kIndexMask;
    }

    /// Reader: returns the newest published value (or the last one it read if nothing is new).
    const T& read()
    {
        if (state_.load(std::memory_order_relaxed) & kFresh)
        {
            const auto previous =
                state_.exchange(static_cast<std::uint8_t>(front_), std::memory_order_acq_rel);
            front_ = previous & kIndexMask;
        }
        return slots_[front_];
    }

    /// Not thread safe: use before the reader starts.
    template <typename Fn>
    void forEachSlot(Fn&& fn)
    {
        for (auto& s : slots_)
            fn(s);
    }

private:
    static constexpr std::uint8_t kFresh = 0x4;
    static constexpr std::uint8_t kIndexMask = 0x3;

    std::array<T, 3> slots_{};
    std::atomic<std::uint8_t> state_{1}; // middle slot index, plus the fresh bit
    std::uint8_t back_ = 0;              // writer only
    std::uint8_t front_ = 2;             // reader only
};

} // namespace alfa::inference
