#pragma once

#include <bit>
#include <cstdint>
#include <span>
#include <type_traits>

namespace alfa::tensor::detail
{
// True when no value is NaN or infinite. Same result as std::all_of(std::isfinite), but it tests
// the exponent bits with an integer OR reduction, which compilers vectorise. The runtime checks
// every parameter, state and output slice on every tick, so this is on the audio hot path.
template <typename Scalar>
[[nodiscard]] inline bool allFinite(std::span<const Scalar> values) noexcept
{
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);
    using Bits = std::conditional_t<std::is_same_v<Scalar, float>, std::uint32_t, std::uint64_t>;
    constexpr Bits exponent = std::is_same_v<Scalar, float> ? static_cast<Bits>(0x7f800000u)
                                                            : static_cast<Bits>(0x7ff0000000000000ull);
    Bits nonFinite = 0;
    for (const auto value : values)
        nonFinite |= static_cast<Bits>((std::bit_cast<Bits>(value) & exponent) == exponent);
    return nonFinite == 0;
}
} // namespace alfa::tensor::detail
