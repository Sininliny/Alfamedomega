#pragma once
#include "alfa/tensor/Kernels.h"

namespace alfa::tensor::advanced
{
std::span<const kernels::Metadata> catalog() noexcept;
template <class S>
bool createPrimitive(std::string_view, std::size_t, PrimitiveT<S>&, std::string&);
// Off-thread IR partitioning. B is an explicit power-of-two block size.
template <class S>
TensorT<S> partitionIR(const TensorT<S>& impulse, std::size_t block);
template <class S>
PrimitiveT<S> partitionedPrimitive(std::size_t block, std::size_t partitions);
} // namespace alfa::tensor::advanced
