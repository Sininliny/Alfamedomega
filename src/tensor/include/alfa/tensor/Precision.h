#pragma once
#include <cstddef>
#include <string>
#include <string_view>
namespace alfa::tensor
{
enum class ScalarType
{
    Float32,
    Float64
};
struct PrecisionProfile
{
    std::string tier;
    ScalarType scalarType = ScalarType::Float32;
};
// Legacy documents omit tier/type and retain the existing float32 representation.
// Named tiers use one policy dtype for all numerical tensors, states and assets.
bool resolvePrecisionJSON(std::string_view, PrecisionProfile&, std::string& error);
std::string_view scalarTypeName(ScalarType) noexcept;
constexpr std::size_t scalarBytes(ScalarType type) noexcept
{
    return type == ScalarType::Float64 ? 8 : 4;
}
} // namespace alfa::tensor
