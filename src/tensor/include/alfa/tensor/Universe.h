#pragma once

#include "alfa/tensor/Kernels.h"

#include <functional>

namespace alfa::tensor::universe
{
// Describes executable operations, not a finite list of instrument types. Every ID
// listed here has deterministic F/G callbacks. Clock, assets, initial state and all
// parameters must be explicitly supplied by the producer.
std::span<const kernels::Metadata> catalog() noexcept;
template <typename Scalar>
bool createPrimitive(std::string_view id, std::size_t stateElements, PrimitiveT<Scalar>& result, std::string& error);

// Registration and creation happen off the audio thread. Extensions own the lifetime
// of strings/parameter spans in their metadata, and retain all runtime state in S_t.
template <typename Scalar = float>
class FactoryT
{
public:
    using Creator = std::function<bool(std::size_t, PrimitiveT<Scalar>&, std::string&)>;
    FactoryT();
    bool add(kernels::Metadata metadata, Creator creator, std::string& error);
    bool create(std::string_view id, std::size_t stateElements, PrimitiveT<Scalar>& result, std::string& error) const;

private:
    struct Entry
    {
        kernels::Metadata metadata;
        Creator creator;
    };
    std::vector<Entry> entries_;
};
using Factory = FactoryT<float>;
using Factory64 = FactoryT<double>;
} // namespace alfa::tensor::universe
