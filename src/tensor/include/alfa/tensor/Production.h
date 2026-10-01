#pragma once
#include "alfa/tensor/Kernels.h"

namespace alfa::tensor::production
{
// Versioned mathematical operations. Capacity is explicit in stateElements;
// delay lengths are measured in this primitive's independently scheduled updates.
std::span<const kernels::Metadata> catalog() noexcept;
template <typename Scalar>
bool createPrimitive(std::string_view id, std::size_t stateElements, PrimitiveT<Scalar>& result,
                     std::string& error);
} // namespace alfa::tensor::production
