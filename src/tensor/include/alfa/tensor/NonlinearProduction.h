#pragma once
#include "alfa/tensor/Kernels.h"

namespace alfa::tensor::nonlinear
{
// The stage length is explicit: M=2*K+2 dense halfband taps, even and 2..24.
// Assets are immutable Scalar tensors; design is an off-thread convenience.
template <typename Scalar>
TensorT<Scalar> designHalfband(std::size_t halfK, double beta, std::string& error);
std::size_t stateElements(std::size_t firstDenseTaps, std::size_t secondDenseTaps = 0) noexcept;
// Group delay for symmetric linear-phase halfband taps (including designHalfband).
// Arbitrary authored coefficients still execute but need not have this group delay.
double latencyUpdates(std::size_t firstDenseTaps, std::size_t secondDenseTaps = 0) noexcept;
std::span<const kernels::Metadata> catalog() noexcept;
// IDs explicitly select 1x/2x/4x. No runtime switch clears or replaces hidden state.
template <typename Scalar>
bool createPrimitive(std::string_view id, std::size_t stateElements, PrimitiveT<Scalar>& result,
                     std::string& error);
} // namespace alfa::tensor::nonlinear
