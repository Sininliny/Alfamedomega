#pragma once

#include "alfa/plugin/ParameterLayout.h"

#include <array>
#include <atomic>

namespace alfa
{
// Everything the UI visualises about the running synth. The audio thread (and
// later the inference thread) write; the UI reads on a 30 Hz timer. All fields
// are lock-free atomics so neither side ever waits on the other.
struct Telemetry
{
    static constexpr int kScopeSize = 2048;
    static constexpr int kControlVectorSize = 16;
    static constexpr int kMaxExperts = params::kNumMethods;

    // Output
    std::atomic<float> peakL{0.0f}, peakR{0.0f};
    std::atomic<float> cpuLoad{0.0f}; // fraction of the block budget
    std::atomic<double> sampleRate{48000.0};
    std::atomic<int> blockSize{512};

    // Notes in
    std::atomic<int> activeVoices{0};
    std::array<std::atomic<float>, 128> noteVelocity{}; // 0 when off

    // Controls and model
    std::array<std::atomic<float>, kControlVectorSize> controlVector{};
    std::array<std::atomic<float>, kMaxExperts> expertActivity{}; // 0..1 per engine expert
    std::atomic<float> modelFrameMs{0.0f};                        // last inference frame time
    std::atomic<bool> modelIsStub{true};                          // true until the inference host lands
    std::atomic<int> loadedTier{1};

    // Engines and envelope
    std::array<std::atomic<float>, params::kNumMethods> engineLevel{};
    std::atomic<float> envelopeLevel{0.0f};
    std::atomic<int> envelopeStage{0}; // 0 idle, 1 attack, 2 decay, 3 sustain, 4 release

    // Output scope, written as a ring buffer.
    std::array<std::atomic<float>, kScopeSize> scope{};
    std::atomic<int> scopeWritePos{0};

    void pushScope(const float* samples, int num) noexcept
    {
        auto pos = scopeWritePos.load(std::memory_order_relaxed);
        for (int i = 0; i < num; ++i)
        {
            scope[static_cast<size_t>(pos)].store(samples[i], std::memory_order_relaxed);
            pos = (pos + 1) % kScopeSize;
        }
        scopeWritePos.store(pos, std::memory_order_release);
    }

    // Copies the most recent `num` samples, oldest first.
    void readScope(float* dest, int num) const noexcept
    {
        num = juce::jmin(num, kScopeSize);
        const auto end = scopeWritePos.load(std::memory_order_acquire);
        auto pos = (end - num + kScopeSize) % kScopeSize;
        for (int i = 0; i < num; ++i)
        {
            dest[i] = scope[static_cast<size_t>(pos)].load(std::memory_order_relaxed);
            pos = (pos + 1) % kScopeSize;
        }
    }
};
} // namespace alfa
