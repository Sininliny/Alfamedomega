#pragma once

#include "alfa/tensor/Runtime.h"

#include <string_view>

namespace alfa::tensor::kernels
{
// Describes these reference CPU kernels, not a closed primitive universe. Numeric
// parameter order is local to each operation; it does not impose a global knob list.
struct Parameter
{
    std::string_view name;
    std::string_view units;
    std::string_view validRange;
    bool trainable;
    bool controllable;
    bool modulationTarget;
};

struct Metadata
{
    std::string_view id;
    std::string_view inputRepresentation;
    std::string_view outputRepresentation;
    std::span<const Parameter> parameters;
    std::string_view stateRepresentation;
    std::string_view costPerUpdate;
    std::string_view supportedRates;
    std::string_view reusableCount;
};

const Metadata& sineMetadata() noexcept;
const Metadata& onePoleMetadata() noexcept;
const Metadata& feedbackCellMetadata() noexcept;
const Metadata& delayMetadata() noexcept;

Primitive sine();
Primitive onePole();
Primitive feedbackCell();
// Caller supplies the delay length in updates, and explicitly chooses its clock.
// Reference shift register costs O(length); no claimed production delay optimization.
Primitive delay(std::size_t lengthInUpdates);

// All readouts observe S_t before transition and scale only the output by activation.
// They never invent tensor routing semantics. The compiler must explicitly bind
// parameter slices and scalar audio inputs. Dynamic modulation can supply parameters
// per Step; applying routing/modulation tensors is the compiler's responsibility.
// Absent audio means zero excitation. Missing/invalid parameters hold state and
// silence readout; invalid state is silenced. No callback allocation or locks.
} // namespace alfa::tensor::kernels
