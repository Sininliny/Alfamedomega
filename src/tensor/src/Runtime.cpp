#include "alfa/tensor/Runtime.h"

#include "alfa/tensor/Compiler.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

namespace alfa::tensor
{
namespace
{
// Patricia trie: discriminating bits strictly decrease along every path.
// Two word-width-bounded walks/index; <=2*k-1 initialized nodes. Old nodes are
// unreachable, so no capacity-sized clear, timestamp wrap, hash collision chain,
// or full-history index mask is needed during a callback.
template <class Node>
bool insertSparseIndex(std::span<Node> nodes, std::size_t& used, std::size_t& root, std::size_t key) noexcept
{
    if (used == 0)
    {
        if (nodes.empty())
            return false;
        nodes[0] = {key, {}, 0, true};
        used = 1;
        root = 0;
        return true;
    }
    auto leaf = root;
    while (!nodes[leaf].leaf)
        leaf = nodes[leaf].children[(key >> nodes[leaf].bit) & 1u];
    const auto different = nodes[leaf].key ^ key;
    if (different == 0)
        return false;
    const auto bit = static_cast<unsigned>(std::bit_width(different) - 1);
    auto* link = &root;
    while (!nodes[*link].leaf && nodes[*link].bit > bit)
        link = &nodes[*link].children[(key >> nodes[*link].bit) & 1u];
    if (nodes.size() - used < 2)
        return false;
    const auto oldChild = *link, newLeaf = used++, branch = used++;
    nodes[newLeaf] = {key, {}, 0, true};
    nodes[branch] = {0, {}, bit, false};
    const auto direction = (key >> bit) & 1u;
    nodes[branch].children[direction] = newLeaf;
    nodes[branch].children[1u - direction] = oldChild;
    *link = branch;
    return true;
}
template <typename Scalar>
bool finite(std::span<const Scalar> values) noexcept
{
    return std::all_of(values.begin(), values.end(), [](Scalar value) { return std::isfinite(value); });
}

template <typename Scalar>
bool validTensor(TensorViewT<Scalar> tensor) noexcept
{
    if (tensor.axes.empty() && tensor.shape.empty() && tensor.values.empty() && tensor.presence.empty())
        return true; // Explicitly absent input/tensor.
    if (tensor.axes.size() != tensor.shape.size())
        return false;
    std::size_t elements = 1;
    for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis)
    {
        if (tensor.axes[axis].empty())
            return false;
        for (std::size_t previous = 0; previous < axis; ++previous)
            if (tensor.axes[axis] == tensor.axes[previous])
                return false;
        const auto dimension = tensor.shape[axis];
        if (dimension != 0 && elements > std::numeric_limits<std::size_t>::max() / dimension)
            return false;
        elements *= dimension;
    }
    if (elements != tensor.values.size() || !finite<Scalar>(tensor.values))
        return false;
    if (tensor.presence.empty())
        return true;
    if (tensor.presence.size() != elements)
        return false;
    for (std::size_t index = 0; index < elements; ++index)
        if (tensor.presence[index] > 1 || (tensor.presence[index] == 0 && tensor.values[index] != 0.0f))
            return false;
    return true;
}

template <typename Scalar>
std::size_t representedElements(const TensorT<Scalar>& tensor) noexcept
{
    if (tensor.presence.empty())
        return tensor.values.size();
    return static_cast<std::size_t>(std::count(tensor.presence.begin(), tensor.presence.end(), 1));
}

bool fits(Slice slice, std::size_t elements) noexcept
{
    return slice.offset <= elements && slice.count <= elements - slice.offset;
}

bool append(std::size_t& used, std::size_t count, std::size_t capacity) noexcept
{
    if (used > capacity || count > capacity - used)
        return false;
    used += count;
    return true;
}

template <typename Scalar>
TensorViewT<Scalar> view(const TensorT<Scalar>& tensor) noexcept
{
    return {tensor.axes, tensor.shape, tensor.values, tensor.presence};
}

template <typename Scalar>
bool matches(const TensorT<Scalar>& tensor, TensorViewT<Scalar> incoming) noexcept
{
    return validTensor<Scalar>(incoming) &&
           std::equal(tensor.axes.begin(), tensor.axes.end(), incoming.axes.begin(), incoming.axes.end()) &&
           std::equal(tensor.shape.begin(), tensor.shape.end(), incoming.shape.begin(),
                      incoming.shape.end()) &&
           std::equal(tensor.presence.begin(), tensor.presence.end(), incoming.presence.begin(),
                      incoming.presence.end()) &&
           tensor.values.size() == incoming.values.size();
}

template <typename Scalar>
struct SourceValuesT
{
    std::span<const Scalar> values;
    std::span<const std::uint8_t> presence;
};

template <typename Scalar>
SourceValuesT<Scalar> sourceValues(SourceKind source, const InputsT<Scalar>& inputs,
                                   std::span<const Scalar> state, std::span<const Scalar> history) noexcept
{
    switch (source)
    {
    case SourceKind::Audio:
        return {inputs.audio.values, inputs.audio.presence};
    case SourceKind::Midi:
        return {inputs.midi.values, inputs.midi.presence};
    case SourceKind::Gesture:
        return {inputs.gesture.values, inputs.gesture.presence};
    case SourceKind::UserPreference:
        return {inputs.userPreference.values, inputs.userPreference.presence};
    case SourceKind::LatentSound:
        return {inputs.latentSound.values, inputs.latentSound.presence};
    case SourceKind::Context:
        return {inputs.context.values, inputs.context.presence};
    case SourceKind::State:
        return {state, {}};
    case SourceKind::OutputHistory:
        return {history, {}};
    }
    return {};
}

template <typename Scalar>
bool represented(const TensorT<Scalar>& tensor, std::size_t index) noexcept
{
    return index < tensor.values.size() && (tensor.presence.empty() || tensor.presence[index] != 0);
}
} // namespace

template <typename Scalar>
bool RuntimeT<Scalar>::prepare(const ConfigurationT<Scalar>& configuration, const Budget& budget,
                               double sampleRate, std::string& error)
{
    error.clear();
    const auto fail = [&error](const char* reason)
    {
        error = reason;
        return false;
    };
    if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
        return fail("Sample rate must be finite and positive");
    if (!validTensor<Scalar>(view<Scalar>(configuration.activation)) ||
        !validTensor<Scalar>(view<Scalar>(configuration.parameters)) ||
        !validTensor<Scalar>(view<Scalar>(configuration.routing)) ||
        !validTensor<Scalar>(view<Scalar>(configuration.modulation)))
        return fail("Tensor shape, axis names or values are invalid");
    if (representedElements<Scalar>(configuration.parameters) > budget.parameterElements ||
        representedElements<Scalar>(configuration.routing) > budget.routingElements ||
        representedElements<Scalar>(configuration.modulation) > budget.modulationElements)
        return fail("Tensor exceeds the supplied capacity; compression must be explicit");
    std::size_t stored = 0;
    for (const auto* tensor : {&configuration.activation, &configuration.parameters, &configuration.routing,
                               &configuration.modulation})
        if (!append(stored, tensor->values.size(), budget.storedTensorElements))
            return fail("Full padded tensor storage exceeds the supplied hardware capacity");
    std::size_t projectionTerms = configuration.projections.size();
    for (const auto& instance : configuration.instances)
    {
        if (!append(stored, instance.inputChannels, budget.storedTensorElements) ||
            !append(projectionTerms, instance.audioBindings.size(), budget.projectionTerms))
            return fail("Audio operands or binding terms exceed supplied capacity");
        for (const auto& asset : instance.assets)
            if (!validTensor<Scalar>(view<Scalar>(asset)) ||
                !append(stored, asset.values.size(), budget.storedTensorElements))
                return fail("Immutable primitive asset is invalid or exceeds storage capacity");
    }
    if (configuration.initialState.size() != configuration.instances.size() ||
        !finite<Scalar>(configuration.sharedInitialState))
        return fail("Initial state does not match the configuration");
    if ((!configuration.parameterLocks.empty() &&
         configuration.parameterLocks.size() != configuration.parameters.values.size()) ||
        std::any_of(configuration.parameterLocks.begin(), configuration.parameterLocks.end(),
                    [](std::uint8_t value) { return value > 1; }))
        return fail("Parameter locks must be a binary mask matching the parameter tensor");
    if (configuration.projections.size() > budget.projectionTerms)
        return fail("Affine projection terms exceed the supplied capacity");
    if (std::any_of(configuration.activation.values.begin(), configuration.activation.values.end(),
                    [](Scalar value) { return value < 0.0f; }))
        return fail("Primitive activation must be nonnegative");

    // Build a candidate so a refused configuration leaves the playing runtime intact.
    RuntimeT<Scalar> candidate;
    candidate.configuration_ = configuration;
    candidate.initialActivation_ = configuration.activation.values;
    candidate.initialParameters_ = configuration.parameters.values;
    candidate.initialRouting_ = configuration.routing.values;
    candidate.initialModulation_ = configuration.modulation.values;
    candidate.primitives_.reserve(configuration.instances.size());
    candidate.sampleRate_ = sampleRate;
    candidate.states_.resize(configuration.instances.size());
    candidate.outputs_.resize(configuration.instances.size());
    candidate.allocated_.resize(configuration.instances.size(), 0);
    candidate.assets_.resize(configuration.instances.size());
    candidate.audioOperands_.resize(configuration.instances.size());
    candidate.audioShapes_.resize(configuration.instances.size());
    candidate.denseScratch_.resize(configuration.instances.size());
    candidate.journalSlices_.resize(configuration.instances.size());
    candidate.journalCounts_.resize(configuration.instances.size(), 0);
    candidate.denseUpdated_.resize(configuration.instances.size(), 0);
    candidate.initial_ = configuration.sharedInitialState;
    std::size_t stateCount = configuration.sharedInitialState.size();
    std::size_t outputCount = 0;
    std::size_t denseCount = 0, journalCount = 0, scratchCount = 0;
    if (stateCount > budget.stateElements)
        return fail("Shared state exceeds the supplied capacity");

    for (std::size_t index = 0; index < configuration.instances.size(); ++index)
    {
        const auto& instance = configuration.instances[index];
        const auto* primitive = instance.primitive;
        if (primitive == nullptr || primitive->id.empty() ||
            (primitive->transition == nullptr && primitive->sparseTransition == nullptr) ||
            primitive->readout == nullptr)
            return fail("Primitive must provide an identity, transition and readout");
        if (primitive->sparseWriteCapacity > primitive->stateElements)
            return fail("Sparse journal capacity cannot exceed unique local state elements");
        if (primitive->sparseTransition == nullptr && primitive->sparseWriteCapacity != 0)
            return fail("Sparse write capacity requires a sparse transition");
        candidate.primitives_.push_back(*primitive);
        candidate.configuration_.instances[index].primitive = &candidate.primitives_.back();
        for (const auto& asset : candidate.configuration_.instances[index].assets)
            candidate.assets_[index].push_back(view<Scalar>(asset));
        if (instance.activationIndex >= configuration.activation.values.size() ||
            !fits(instance.parameters, configuration.parameters.values.size()))
            return fail("Primitive tensor selection is out of range");
        if (instance.clock.periodSamples == 0 || instance.clock.phaseSamples >= instance.clock.periodSamples)
            return fail("Each primitive must supply a valid explicit clock");
        const auto& localInitial = configuration.initialState[index];
        if (localInitial.size() != primitive->stateElements || !finite<Scalar>(localInitial))
            return fail("Primitive initial state is invalid");

        candidate.states_[index] = {stateCount, 0};
        candidate.outputs_[index] = {outputCount, 0};
        if (configuration.activation.values[instance.activationIndex] == 0.0f)
            continue;
        if (!configuration.parameters.presence.empty())
            for (std::size_t element = 0; element < instance.parameters.count; ++element)
                if (configuration.parameters.presence[instance.parameters.offset + element] == 0)
                    return fail("Active primitive parameters cannot select padding");
        if (candidate.schedule_.size() >= budget.activePrimitives)
            return fail("Active primitives exceed the supplied capacity");
        if (primitive->scratchElements > budget.scratchElements)
            return fail("Primitive numerical workspace exceeds explicit scratch capacity");
        scratchCount = std::max(scratchCount, primitive->scratchElements);
        candidate.states_[index].count = primitive->stateElements;
        candidate.outputs_[index].count = primitive->outputElements;
        if (!append(stateCount, primitive->stateElements, budget.stateElements) ||
            !append(outputCount, primitive->outputElements, budget.outputElements))
            return fail("Primitive state or output exceeds the supplied capacity");
        if (primitive->sparseTransition)
        {
            candidate.journalSlices_[index] = {journalCount, primitive->sparseWriteCapacity};
            if (!append(journalCount, primitive->sparseWriteCapacity, budget.sparseStateWrites))
                return fail("Sparse state journal exceeds the supplied scratch capacity");
        }
        else
        {
            candidate.denseScratch_[index] = {denseCount, primitive->stateElements};
            if (!append(denseCount, primitive->stateElements, budget.stateElements))
                return fail("Dense state scratch exceeds state capacity");
        }
        candidate.initial_.insert(candidate.initial_.end(), localInitial.begin(), localInitial.end());
        candidate.schedule_.push_back({index, instance.clock});
        candidate.allocated_[index] = 1;
    }
    candidate.state_ = candidate.initial_;
    for (const auto& projection : configuration.projections)
    {
        if (projection.source < SourceKind::Audio || projection.source > SourceKind::OutputHistory ||
            (projection.weightTensor != WeightTensor::Routing &&
             projection.weightTensor != WeightTensor::Modulation) ||
            (projection.missingSource != MissingSource::Reject &&
             projection.missingSource != MissingSource::Zero))
            return fail("Affine projection has an unknown source, weight or missing-input policy");
        const auto& weights = projection.weightTensor == WeightTensor::Routing ? configuration.routing
                                                                               : configuration.modulation;
        if (!represented<Scalar>(configuration.parameters, projection.parameterElement) ||
            !represented<Scalar>(weights, projection.weightElement) ||
            (projection.source == SourceKind::State && projection.sourceElement >= stateCount) ||
            (projection.source == SourceKind::OutputHistory && projection.sourceElement >= outputCount))
            return fail("Affine projection selects missing or padded tensor/state elements");
    }
    candidate.next_.resize(denseCount, Scalar{0});
    candidate.journal_.resize(journalCount);
    if (journalCount > candidate.sparseIndices_.max_size() / 2)
        return fail("Sparse index validation capacity overflows structural storage");
    candidate.sparseIndices_.resize(journalCount * 2);
    candidate.scratch_.resize(scratchCount, Scalar{0});
    for (size_t i = 0; i < configuration.instances.size(); ++i)
    {
        const auto& instance = configuration.instances[i];
        candidate.audioOperands_[i].resize(instance.inputChannels, 0);
        candidate.audioShapes_[i] = {instance.inputChannels};
        for (const auto& binding : instance.audioBindings)
        {
            const auto& weights = binding.weightTensor == WeightTensor::Routing ? configuration.routing
                                                                                : configuration.modulation;
            if (binding.source < SourceKind::Audio || binding.source > SourceKind::OutputHistory ||
                (binding.weightTensor != WeightTensor::Routing &&
                 binding.weightTensor != WeightTensor::Modulation) ||
                (binding.missingSource != MissingSource::Zero &&
                 binding.missingSource != MissingSource::Reject) ||
                binding.destinationElement >= instance.inputChannels ||
                !represented<Scalar>(weights, binding.weightElement) ||
                (binding.source == SourceKind::State && binding.sourceElement >= stateCount) ||
                (binding.source == SourceKind::OutputHistory && binding.sourceElement >= outputCount))
                return fail("Invalid recurrent audio operand binding");
        }
    }
    candidate.output_.resize(outputCount, 0.0f);
    candidate.candidateOutput_.resize(outputCount, 0.0f);
    candidate.committedOutput_.resize(outputCount, 0.0f);
    candidate.effectiveParameters_ = configuration.parameters.values;
    // Fingerprint structural compatibility once, off thread. Numeric control edits
    // retain recurrence; identities, assets, clocks and operand layouts do not.
    auto identityConfiguration = configuration;
    if (identityConfiguration.initialState.empty())
    {
        identityConfiguration.initialState.resize(configuration.instances.size());
        for (size_t i = 0; i < configuration.instances.size(); ++i)
            identityConfiguration.initialState[i].resize(configuration.instances[i].primitive->stateElements);
    }
    auto identity = nlohmann::json::parse(serializeJSON(identityConfiguration));
    identity["allocated_instances"] = candidate.allocated_;
    identity["allocation"] = nlohmann::json::array();
    for (size_t i = 0; i < candidate.states_.size(); ++i)
        identity["allocation"].push_back({candidate.states_[i].offset, candidate.states_[i].count,
                                          candidate.outputs_[i].offset, candidate.outputs_[i].count});
    for (const auto& instance : configuration.instances)
        identity["callbacks"].push_back(
            {reinterpret_cast<std::uintptr_t>(instance.primitive->transition),
             reinterpret_cast<std::uintptr_t>(instance.primitive->readout),
             reinterpret_cast<std::uintptr_t>(instance.primitive->sparseTransition)});
    for (auto& tensor : identity["tensors"])
        tensor.erase("values");
    identity.erase("parameter_locks");
    const auto bytes = identity.dump();
    candidate.stateIdentity_ = {14695981039346656037ULL, 7809847782465536322ULL};
    for (const unsigned char byte : bytes)
    {
        candidate.stateIdentity_[0] = (candidate.stateIdentity_[0] ^ byte) * 1099511628211ULL;
        candidate.stateIdentity_[1] = (candidate.stateIdentity_[1] ^ byte) * 14029467366897019727ULL;
    }
    candidate.prepared_ = true;
    *this = std::move(candidate);
    return true;
}

template <typename Scalar>
bool RuntimeT<Scalar>::tick(const InputsT<Scalar>& inputs) noexcept
{
    if (!prepared_ || !validTensor<Scalar>(inputs.audio) || !validTensor<Scalar>(inputs.midi) ||
        !validTensor<Scalar>(inputs.gesture) || !validTensor<Scalar>(inputs.userPreference) ||
        !validTensor<Scalar>(inputs.latentSound) || !validTensor<Scalar>(inputs.context) ||
        sample_ == std::numeric_limits<std::uint64_t>::max())
    {
        std::fill(output_.begin(), output_.end(), 0.0f);
        return false;
    }
    std::fill(denseUpdated_.begin(), denseUpdated_.end(), std::uint8_t{0});
    std::fill(journalCounts_.begin(), journalCounts_.end(), std::size_t{0});
    std::copy(committedOutput_.begin(), committedOutput_.end(), candidateOutput_.begin());
    std::copy(configuration_.parameters.values.begin(), configuration_.parameters.values.end(),
              effectiveParameters_.begin());
    for (const auto& projection : configuration_.projections)
    {
        if (!configuration_.parameterLocks.empty() &&
            configuration_.parameterLocks[projection.parameterElement] != 0)
            continue;
        const auto sources = sourceValues<Scalar>(projection.source, inputs, state_, committedOutput_);
        if (projection.sourceElement >= sources.values.size() ||
            (!sources.presence.empty() && sources.presence[projection.sourceElement] == 0))
        {
            if (projection.missingSource == MissingSource::Zero)
                continue;
            std::fill(output_.begin(), output_.end(), 0.0f);
            return false;
        }
        const auto& weights = projection.weightTensor == WeightTensor::Routing ? configuration_.routing
                                                                               : configuration_.modulation;
        effectiveParameters_[projection.parameterElement] +=
            weights.values[projection.weightElement] * sources.values[projection.sourceElement];
    }
    if (!finite<Scalar>(effectiveParameters_))
    {
        std::fill(output_.begin(), output_.end(), 0.0f);
        return false;
    }
    for (const auto& update : schedule_)
    {
        const auto index = update.instance;
        const auto& instance = configuration_.instances[index];
        const auto out = outputs_[index];
        auto output = std::span<Scalar>(candidateOutput_).subspan(out.offset, out.count);
        if (configuration_.activation.values[instance.activationIndex] == 0.0f)
        {
            std::fill(output.begin(), output.end(), 0.0f);
            continue; // Retain state so a later activation can resume this allocated primitive.
        }
        if (sample_ % update.clock.periodSamples != update.clock.phaseSamples)
        {
            if (instance.outputBetweenUpdates == OutputBetweenUpdates::Zero)
                std::fill(output.begin(), output.end(), 0.0f);
            continue;
        }
        const auto local = states_[index];
        auto operandInputs = inputs;
        if (instance.inputChannels > 0)
        {
            auto& operands = audioOperands_[index];
            std::fill(operands.begin(), operands.end(), 0.0f);
            for (const auto& binding : instance.audioBindings)
            {
                const auto source = sourceValues<Scalar>(binding.source, inputs, state_, committedOutput_);
                if (binding.sourceElement >= source.values.size() ||
                    (!source.presence.empty() && source.presence[binding.sourceElement] == 0))
                {
                    if (binding.missingSource == MissingSource::Zero)
                        continue;
                    std::fill(output_.begin(), output_.end(), 0.0f);
                    return false;
                }
                const auto& weights = binding.weightTensor == WeightTensor::Routing
                                          ? configuration_.routing
                                          : configuration_.modulation;
                operands[binding.destinationElement] +=
                    weights.values[binding.weightElement] * source.values[binding.sourceElement];
            }
            if (!finite<Scalar>(operands))
            {
                std::fill(output_.begin(), output_.end(), 0.0f);
                return false;
            }
            operandInputs.audio = {audioAxes_, audioShapes_[index], operands, {}};
        }
        const StepT<Scalar> step{sample_,
                                 sampleRate_,
                                 update.clock,
                                 configuration_.activation.values[instance.activationIndex],
                                 operandInputs,
                                 state_,
                                 std::span<const Scalar>(state_).subspan(local.offset, local.count),
                                 states_,
                                 std::span<const Scalar>(effectiveParameters_)
                                     .subspan(instance.parameters.offset, instance.parameters.count),
                                 view<Scalar>(configuration_.routing),
                                 view<Scalar>(configuration_.modulation),
                                 assets_[index],
                                 std::span<Scalar>(scratch_).first(instance.primitive->scratchElements)};
        std::fill(output.begin(), output.end(), 0.0f);
        instance.primitive->readout(step, output);
        if (instance.primitive->sparseTransition)
        {
            const auto scratch = journalSlices_[index];
            auto writes = std::span<StateWriteT<Scalar>>(journal_).subspan(scratch.offset, scratch.count);
            for (auto& write : writes)
                write = {local.count, Scalar(0)};
            const auto count = instance.primitive->sparseTransition(step, writes);
            if (count > writes.size())
            {
                std::fill(output_.begin(), output_.end(), Scalar{0});
                return false;
            }
            auto indices =
                std::span<SparseIndexNode>(sparseIndices_).subspan(scratch.offset * 2, scratch.count * 2);
            std::size_t used = 0, root = 0;
            for (size_t i = 0; i < count; ++i)
            {
                if (writes[i].element >= local.count || !std::isfinite(writes[i].value) ||
                    !insertSparseIndex(indices, used, root, writes[i].element))
                {
                    std::fill(output_.begin(), output_.end(), Scalar{0});
                    return false;
                }
            }
            journalCounts_[index] = count;
        }
        else
        {
            const auto scratch = denseScratch_[index];
            auto next = std::span<Scalar>(next_).subspan(scratch.offset, scratch.count);
            std::copy(step.localState.begin(), step.localState.end(), next.begin());
            instance.primitive->transition(step, next);
            if (!finite<Scalar>(next))
            {
                std::fill(output_.begin(), output_.end(), Scalar{0});
                return false;
            }
            denseUpdated_[index] = 1;
        }
    }
    if (!finite<Scalar>(candidateOutput_))
    {
        std::fill(output_.begin(), output_.end(), 0.0f);
        return false;
    }
    // Commit only after every due primitive and its output has validated.
    // No callback can see these writes while another callback still executes.
    for (const auto& update : schedule_)
    {
        const auto index = update.instance;
        const auto local = states_[index];
        if (denseUpdated_[index])
        {
            const auto scratch = denseScratch_[index];
            std::copy_n(next_.begin() + static_cast<std::ptrdiff_t>(scratch.offset), scratch.count,
                        state_.begin() + static_cast<std::ptrdiff_t>(local.offset));
        }
        const auto journal = journalSlices_[index];
        for (size_t i = 0; i < journalCounts_[index]; ++i)
        {
            const auto& write = journal_[journal.offset + i];
            state_[local.offset + write.element] = write.value;
        }
    }
    output_.swap(candidateOutput_);
    std::copy(output_.begin(), output_.end(), committedOutput_.begin());
    ++sample_;
    return true;
}

template <typename Scalar>
void RuntimeT<Scalar>::reset() noexcept
{
    for (const auto& pair : {std::pair{&configuration_.activation.values, &initialActivation_},
                             std::pair{&configuration_.parameters.values, &initialParameters_},
                             std::pair{&configuration_.routing.values, &initialRouting_},
                             std::pair{&configuration_.modulation.values, &initialModulation_}})
        std::copy(pair.second->begin(), pair.second->end(), pair.first->begin());
    std::copy(initial_.begin(), initial_.end(), state_.begin());
    std::fill(next_.begin(), next_.end(), Scalar{0});
    std::fill(output_.begin(), output_.end(), 0.0f);
    std::fill(candidateOutput_.begin(), candidateOutput_.end(), 0.0f);
    std::fill(committedOutput_.begin(), committedOutput_.end(), Scalar{0});
    std::fill(journalCounts_.begin(), journalCounts_.end(), std::size_t{0});
    std::fill(denseUpdated_.begin(), denseUpdated_.end(), std::uint8_t{0});
    sample_ = 0;
}
template <typename Scalar>
bool RuntimeT<Scalar>::adoptStateFrom(const RuntimeT<Scalar>& previous) noexcept
{
    if (!prepared_ || !previous.prepared_ || stateIdentity_ != previous.stateIdentity_ ||
        sampleRate_ != previous.sampleRate_ || states_.size() != previous.states_.size() ||
        state_.size() != previous.state_.size() || output_.size() != previous.output_.size())
        return false;
    for (size_t i = 0; i < states_.size(); ++i)
        if (configuration_.instances[i].primitive->id != previous.configuration_.instances[i].primitive->id ||
            states_[i].offset != previous.states_[i].offset ||
            states_[i].count != previous.states_[i].count ||
            outputs_[i].offset != previous.outputs_[i].offset ||
            outputs_[i].count != previous.outputs_[i].count)
            return false;
    std::copy(previous.state_.begin(), previous.state_.end(), state_.begin());
    std::fill(next_.begin(), next_.end(), Scalar{0});
    std::fill(journalCounts_.begin(), journalCounts_.end(), std::size_t{0});
    std::fill(denseUpdated_.begin(), denseUpdated_.end(), std::uint8_t{0});
    std::copy(previous.committedOutput_.begin(), previous.committedOutput_.end(), committedOutput_.begin());
    std::copy(previous.output_.begin(), previous.output_.end(), output_.begin());
    std::copy(output_.begin(), output_.end(), candidateOutput_.begin());
    sample_ = previous.sample_;
    return true;
}

template <typename Scalar>
bool RuntimeT<Scalar>::takeStateFrom(RuntimeT<Scalar>& previous) noexcept
{
    if (!prepared_ || !previous.prepared_ || sampleRate_ != previous.sampleRate_ ||
        stateIdentity_ != previous.stateIdentity_ || state_.size() != previous.state_.size() ||
        output_.size() != previous.output_.size())
        return false;
    state_.swap(previous.state_);
    output_.swap(previous.output_);
    committedOutput_.swap(previous.committedOutput_);
    candidateOutput_.swap(previous.candidateOutput_);
    sample_ = previous.sample_;
    return true;
}

template <typename Scalar>
bool RuntimeT<Scalar>::updateTensors(TensorViewT<Scalar> activation, TensorViewT<Scalar> parameters,
                                     TensorViewT<Scalar> routing, TensorViewT<Scalar> modulation) noexcept
{
    if (!prepared_ || !matches<Scalar>(configuration_.activation, activation) ||
        !matches<Scalar>(configuration_.parameters, parameters) ||
        !matches<Scalar>(configuration_.routing, routing) ||
        !matches<Scalar>(configuration_.modulation, modulation) ||
        std::any_of(activation.values.begin(), activation.values.end(),
                    [](Scalar value) { return value < 0.0f; }))
        return false;
    for (std::size_t index = 0; index < configuration_.instances.size(); ++index)
        if (allocated_[index] == 0 &&
            activation.values[configuration_.instances[index].activationIndex] != 0.0f)
            return false;
    for (std::size_t element = 0; element < configuration_.parameterLocks.size(); ++element)
        if (configuration_.parameterLocks[element] != 0 &&
            parameters.values[element] != configuration_.parameters.values[element])
            return false;
    for (const auto& pair :
         {std::pair{&configuration_.activation, activation},
          std::pair{&configuration_.parameters, parameters}, std::pair{&configuration_.routing, routing},
          std::pair{&configuration_.modulation, modulation}})
        std::copy(pair.second.values.begin(), pair.second.values.end(), pair.first->values.begin());
    return true;
}

template <typename Scalar>
std::size_t RuntimeT<Scalar>::numericStorageBytes() const noexcept
{
    size_t elements = initialActivation_.size() + initialParameters_.size() + initialRouting_.size() +
                      initialModulation_.size() + initial_.size() + state_.size() + next_.size() +
                      output_.size() + candidateOutput_.size() + committedOutput_.size() +
                      effectiveParameters_.size() + journal_.size() + scratch_.size();
    elements += configuration_.activation.values.size() + configuration_.parameters.values.size() +
                configuration_.routing.values.size() + configuration_.modulation.values.size() +
                configuration_.sharedInitialState.size();
    for (const auto& values : configuration_.initialState)
        elements += values.size();
    for (const auto& instance : configuration_.instances)
        for (const auto& asset : instance.assets)
            elements += asset.values.size();
    for (const auto& values : audioOperands_)
        elements += values.size();
    return elements * sizeof(Scalar);
}
template class RuntimeT<float>;
template class RuntimeT<double>;
} // namespace alfa::tensor
