#pragma once

#include <cmath>
#include <cstdint>

namespace alfa::dsp::detail
{

inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float kTwoPi = 6.28318530717958647692f;

/// sin(2 pi x) for x in cycles, any range. Branch-free so loops over it vectorise.
/// Max abs error about 1e-6 over a full cycle (checked in the unit tests).
[[nodiscard]] inline float sin2pi(float x) noexcept
{
    const float r = x - std::floor(x + 0.5f);  // [-0.5, 0.5)
    const float a = std::fabs(r);
    const float m = (a > 0.25f) ? 0.5f - a : a;  // fold into [0, 0.25]
    const float t = kTwoPi * m;                  // [0, pi/2]
    const float t2 = t * t;
    // Taylor to t^11, accurate to ~6e-8 on [0, pi/2].
    float p = -2.5052108e-8f;
    p = p * t2 + 2.7557319e-6f;
    p = p * t2 - 1.9841270e-4f;
    p = p * t2 + 8.3333333e-3f;
    p = p * t2 - 1.6666667e-1f;
    p = p * t2 + 1.0f;
    const float s = p * t;
    return std::copysign(s, r);
}

/// Small, fast, deterministic PRNG (xorshift32). One per voice; not shared between threads.
class Random
{
public:
    explicit Random(std::uint32_t seed = 0x9E3779B9u) noexcept { setSeed(seed); }

    void setSeed(std::uint32_t seed) noexcept { state_ = seed != 0 ? seed : 0x9E3779B9u; }

    std::uint32_t nextU32() noexcept
    {
        std::uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        return x;
    }

    /// Uniform in [-1, 1).
    float nextBipolar() noexcept
    {
        return static_cast<float>(static_cast<std::int32_t>(nextU32())) * (1.0f / 2147483648.0f);
    }

private:
    std::uint32_t state_;
};

} // namespace alfa::dsp::detail
