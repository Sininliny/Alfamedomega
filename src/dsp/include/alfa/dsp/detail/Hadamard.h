#pragma once
#include <array>
#include <cstddef>
namespace alfa::dsp::detail
{
// A specific normalized8-vector operation; it does not impose runtime slot counts.
// Pure math shared by legacy DelayFdn and typed tensor production FDN8.
template <class Scalar>
void hadamard8(std::array<Scalar, 8>& values) noexcept
{
    for (int h = 1; h < 8; h <<= 1)
        for (int i = 0; i < 8; i += 2 * h)
            for (int j = i; j < i + h; ++j)
            {
                const auto a = static_cast<std::size_t>(j), b = static_cast<std::size_t>(j + h);
                const Scalar x = values[a], z = values[b];
                values[a] = x + z;
                values[b] = x - z;
            }
    constexpr Scalar normalization = static_cast<Scalar>(0.35355339059327373);
    for (auto& value : values)
        value *= normalization;
}
} // namespace alfa::dsp::detail
