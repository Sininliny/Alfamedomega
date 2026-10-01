#pragma once

#include "alfa/tensor/Runtime.h"

#include <functional>
#include <string>
#include <string_view>

namespace alfa::tensor
{
// Prototype versioned serialization of tensor computation descriptions. Parsing and
// factory resolution occur off the audio thread. This does not infer a synthesis method.
template <typename Scalar>
using PrimitiveResolverT =
    std::function<bool(std::string_view, std::size_t, PrimitiveT<Scalar>&, std::string&)>;
using PrimitiveResolver = PrimitiveResolverT<float>;
using PrimitiveResolver64 = PrimitiveResolverT<double>;
// An explicit extension resolver may implement new mathematical operations without
// changing the compiler's tensor space. Resolution never occurs on the audio thread.
template <typename Scalar>
bool parseJSON(std::string_view text, ConfigurationT<Scalar>& output, std::string& error,
               const PrimitiveResolverT<Scalar>& extension = {});
template <typename Scalar>
std::string serializeJSON(const ConfigurationT<Scalar>& configuration);
bool readBudgetJSON(std::string_view text, Budget& output, std::string& error);

// A concrete operational report produced by deterministic lowering. Runtime::prepare
// validates the supplied hardware/tier capacity and owns the prepared state/schedule.
template <typename Scalar>
bool compileJSON(std::string_view text, double sampleRate, RuntimeT<Scalar>& runtime, std::string& error);

} // namespace alfa::tensor
