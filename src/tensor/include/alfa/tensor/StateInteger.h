#pragma once
#include <cmath>
#include <cstdint>
#include <span>
#include <type_traits>
namespace alfa::tensor
{
// Exact persistent integer memory uses the same Scalar dtype as all other state.
// Four little-endian base65536 limbs represent uint64 exactly in float32/float64.
// Integer temporaries and structural indices do not introduce a second state buffer.
template <class Scalar>
bool decodeStateInteger(std::span<const Scalar> limbs, std::uint64_t& value) noexcept
{
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);
    if (limbs.empty() || limbs.size() > 4)
        return false;
    std::uint64_t candidate = 0;
    for (size_t i = 0; i < limbs.size(); ++i)
    {
        const auto limb = limbs[i];
        if (!std::isfinite(limb) || limb < 0 || limb > 65535 || std::floor(limb) != limb)
            return false;
        candidate |= static_cast<std::uint64_t>(limb) << (16 * i);
    }
    value = candidate;
    return true;
}
template <class Scalar>
bool encodeStateInteger(std::uint64_t value, std::span<Scalar> limbs) noexcept
{
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);
    if (limbs.empty() || limbs.size() > 4 || (limbs.size() < 4 && (value >> (16 * limbs.size())) != 0))
        return false;
    for (size_t i = 0; i < limbs.size(); ++i)
        limbs[i] = static_cast<Scalar>((value >> (16 * i)) & 65535u);
    return true;
}
} // namespace alfa::tensor
