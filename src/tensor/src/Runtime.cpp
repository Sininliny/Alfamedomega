#include "alfa/tensor/Runtime.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace alfa::tensor
{
namespace
{
bool finite(std::span<const float> values) noexcept
{
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool validTensor(TensorView tensor) noexcept
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
    if (elements != tensor.values.size() || !finite(tensor.values))
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

std::size_t representedElements(const Tensor& tensor) noexcept
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

TensorView view(const Tensor& tensor) noexcept
{
    return {tensor.axes, tensor.shape, tensor.values, tensor.presence};
}

bool matches(const Tensor& tensor, TensorView incoming) noexcept
{
    return validTensor(incoming) &&
           std::equal(tensor.axes.begin(), tensor.axes.end(), incoming.axes.begin(), incoming.axes.end()) &&
           std::equal(tensor.shape.begin(), tensor.shape.end(), incoming.shape.begin(),
                      incoming.shape.end()) &&
           std::equal(tensor.presence.begin(), tensor.presence.end(), incoming.presence.begin(),
                      incoming.presence.end()) &&
           tensor.values.size() == incoming.values.size();
}

struct SourceValues
{
    std::span<const float> values;
    std::span<const std::uint8_t> presence;
};

SourceValues sourceValues(SourceKind source, const Inputs& inputs, std::span<const float> state) noexcept
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
    }
    return {};
}

bool represented(const Tensor& tensor, std::size_t index) noexcept
{
    return index < tensor.values.size() && (tensor.presence.empty() || tensor.presence[index] != 0);
}
} // namespace

bool Runtime::prepare(const Configuration& configuration, const Budget& budget, double sampleRate,
                      std::string& error)
{
    error.clear();
    const auto fail = [&error](const char* reason)
    {
        error = reason;
        return false;
    };
    if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
        return fail("Sample rate must be finite and positive");
    if (!validTensor(view(configuration.activation)) || !validTensor(view(configuration.parameters)) ||
        !validTensor(view(configuration.routing)) || !validTensor(view(configuration.modulation)))
        return fail("Tensor shape, axis names or values are invalid");
    if (representedElements(configuration.parameters) > budget.parameterElements ||
        representedElements(configuration.routing) > budget.routingElements ||
        representedElements(configuration.modulation) > budget.modulationElements)
        return fail("Tensor exceeds the supplied capacity; compression must be explicit");
    std::size_t stored = 0;
    for (const auto* tensor : {&configuration.activation, &configuration.parameters, &configuration.routing,
                               &configuration.modulation})
        if (!append(stored, tensor->values.size(), budget.storedTensorElements))
            return fail("Full padded tensor storage exceeds the supplied hardware capacity");
    for (const auto& instance : configuration.instances)
        for (const auto& asset : instance.assets)
            if (!validTensor(view(asset)) ||
                !append(stored, asset.values.size(), budget.storedTensorElements))
                return fail("Immutable primitive asset is invalid or exceeds storage capacity");
    if (configuration.initialState.size() != configuration.instances.size() ||
        !finite(configuration.sharedInitialState))
        return fail("Initial state does not match the configuration");
    if ((!configuration.parameterLocks.empty() &&
         configuration.parameterLocks.size() != configuration.parameters.values.size()) ||
        std::any_of(configuration.parameterLocks.begin(), configuration.parameterLocks.end(),
                    [](std::uint8_t value) { return value > 1; }))
        return fail("Parameter locks must be a binary mask matching the parameter tensor");
    if (configuration.projections.size() > budget.projectionTerms)
        return fail("Affine projection terms exceed the supplied capacity");
    if (std::any_of(configuration.activation.values.begin(), configuration.activation.values.end(),
                    [](float value) { return value < 0.0f; }))
        return fail("Primitive activation must be nonnegative");

    // Build a candidate so a refused configuration leaves the playing runtime intact.
    Runtime candidate;
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
    candidate.initial_ = configuration.sharedInitialState;
    std::size_t stateCount = configuration.sharedInitialState.size();
    std::size_t outputCount = 0;
    if (stateCount > budget.stateElements)
        return fail("Shared state exceeds the supplied capacity");

    for (std::size_t index = 0; index < configuration.instances.size(); ++index)
    {
        const auto& instance = configuration.instances[index];
        const auto* primitive = instance.primitive;
        if (primitive == nullptr || primitive->id.empty() || primitive->transition == nullptr ||
            primitive->readout == nullptr)
            return fail("Primitive must provide an identity, transition and readout");
        candidate.primitives_.push_back(*primitive);
        candidate.configuration_.instances[index].primitive = &candidate.primitives_.back();
        for (const auto& asset : candidate.configuration_.instances[index].assets)
            candidate.assets_[index].push_back(view(asset));
        if (instance.activationIndex >= configuration.activation.values.size() ||
            !fits(instance.parameters, configuration.parameters.values.size()))
            return fail("Primitive tensor selection is out of range");
        if (instance.clock.periodSamples == 0 || instance.clock.phaseSamples >= instance.clock.periodSamples)
            return fail("Each primitive must supply a valid explicit clock");
        const auto& localInitial = configuration.initialState[index];
        if (localInitial.size() != primitive->stateElements || !finite(localInitial))
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
        candidate.states_[index].count = primitive->stateElements;
        candidate.outputs_[index].count = primitive->outputElements;
        if (!append(stateCount, primitive->stateElements, budget.stateElements) ||
            !append(outputCount, primitive->outputElements, budget.outputElements))
            return fail("Primitive state or output exceeds the supplied capacity");
        candidate.initial_.insert(candidate.initial_.end(), localInitial.begin(), localInitial.end());
        candidate.schedule_.push_back({index, instance.clock});
        candidate.allocated_[index] = 1;
    }
    candidate.state_ = candidate.initial_;
    for (const auto& projection : configuration.projections)
    {
        if (projection.source < SourceKind::Audio || projection.source > SourceKind::State ||
            (projection.weightTensor != WeightTensor::Routing &&
             projection.weightTensor != WeightTensor::Modulation) ||
            (projection.missingSource != MissingSource::Reject &&
             projection.missingSource != MissingSource::Zero))
            return fail("Affine projection has an unknown source, weight or missing-input policy");
        const auto& weights = projection.weightTensor == WeightTensor::Routing ? configuration.routing
                                                                               : configuration.modulation;
        if (!represented(configuration.parameters, projection.parameterElement) ||
            !represented(weights, projection.weightElement) ||
            (projection.source == SourceKind::State && projection.sourceElement >= stateCount))
            return fail("Affine projection selects missing or padded tensor/state elements");
    }
    candidate.next_ = candidate.initial_;
    candidate.output_.resize(outputCount, 0.0f);
    candidate.candidateOutput_.resize(outputCount, 0.0f);
    candidate.effectiveParameters_ = configuration.parameters.values;
    candidate.prepared_ = true;
    *this = std::move(candidate);
    return true;
}

bool Runtime::tick(const Inputs& inputs) noexcept
{
    if (!prepared_ || !validTensor(inputs.audio) || !validTensor(inputs.midi) ||
        !validTensor(inputs.gesture) || !validTensor(inputs.userPreference) ||
        !validTensor(inputs.latentSound) || !validTensor(inputs.context) ||
        sample_ == std::numeric_limits<std::uint64_t>::max())
    {
        std::fill(output_.begin(), output_.end(), 0.0f);
        return false;
    }
    std::copy(state_.begin(), state_.end(), next_.begin());
    std::copy(output_.begin(), output_.end(), candidateOutput_.begin());
    std::copy(configuration_.parameters.values.begin(), configuration_.parameters.values.end(),
              effectiveParameters_.begin());
    for (const auto& projection : configuration_.projections)
    {
        if (!configuration_.parameterLocks.empty() &&
            configuration_.parameterLocks[projection.parameterElement] != 0)
            continue;
        const auto sources = sourceValues(projection.source, inputs, state_);
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
    if (!finite(effectiveParameters_))
    {
        std::fill(output_.begin(), output_.end(), 0.0f);
        return false;
    }
    for (const auto& update : schedule_)
    {
        const auto index = update.instance;
        const auto& instance = configuration_.instances[index];
        const auto out = outputs_[index];
        auto output = std::span<float>(candidateOutput_).subspan(out.offset, out.count);
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
        const Step step{sample_,
                        sampleRate_,
                        update.clock,
                        configuration_.activation.values[instance.activationIndex],
                        inputs,
                        state_,
                        std::span<const float>(state_).subspan(local.offset, local.count),
                        states_,
                        std::span<const float>(effectiveParameters_)
                            .subspan(instance.parameters.offset, instance.parameters.count),
                        view(configuration_.routing),
                        view(configuration_.modulation),
                        assets_[index]};
        std::fill(output.begin(), output.end(), 0.0f);
        instance.primitive->readout(step, output);
        instance.primitive->transition(step, std::span<float>(next_).subspan(local.offset, local.count));
    }
    if (!finite(next_) || !finite(candidateOutput_))
    {
        std::fill(output_.begin(), output_.end(), 0.0f);
        return false;
    }
    state_.swap(next_);
    output_.swap(candidateOutput_);
    ++sample_;
    return true;
}

void Runtime::reset() noexcept
{
    for (const auto& pair : {std::pair{&configuration_.activation.values, &initialActivation_},
                             std::pair{&configuration_.parameters.values, &initialParameters_},
                             std::pair{&configuration_.routing.values, &initialRouting_},
                             std::pair{&configuration_.modulation.values, &initialModulation_}})
        std::copy(pair.second->begin(), pair.second->end(), pair.first->begin());
    std::copy(initial_.begin(), initial_.end(), state_.begin());
    std::copy(initial_.begin(), initial_.end(), next_.begin());
    std::fill(output_.begin(), output_.end(), 0.0f);
    std::fill(candidateOutput_.begin(), candidateOutput_.end(), 0.0f);
    sample_ = 0;
}

bool Runtime::updateTensors(TensorView activation, TensorView parameters, TensorView routing,
                            TensorView modulation) noexcept
{
    if (!prepared_ || !matches(configuration_.activation, activation) ||
        !matches(configuration_.parameters, parameters) || !matches(configuration_.routing, routing) ||
        !matches(configuration_.modulation, modulation) ||
        std::any_of(activation.values.begin(), activation.values.end(),
                    [](float value) { return value < 0.0f; }))
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

} // namespace alfa::tensor
