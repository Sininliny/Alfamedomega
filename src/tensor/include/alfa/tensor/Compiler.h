#pragma once

#include "alfa/tensor/Runtime.h"

#include <functional>
#include <string>
#include <string_view>

namespace alfa::tensor
{
// Prototype versioned serialization of tensor computation descriptions. Parsing and
// factory resolution occur off the audio thread. This does not infer a synthesis method.
using PrimitiveResolver = std::function<bool(std::string_view, std::size_t, Primitive&, std::string&)>;
// An explicit extension resolver may implement new mathematical operations without
// changing the compiler's tensor space. Resolution never occurs on the audio thread.
bool parseJSON(std::string_view text, Configuration& output, std::string& error,
               const PrimitiveResolver& extension = {});
std::string serializeJSON(const Configuration& configuration);
bool readBudgetJSON(std::string_view text, Budget& output, std::string& error);

// A concrete operational report produced by deterministic lowering. Runtime::prepare
// validates the supplied hardware/tier capacity and owns the prepared state/schedule.
bool compileJSON(std::string_view text, double sampleRate, Runtime& runtime, std::string& error);

} // namespace alfa::tensor
