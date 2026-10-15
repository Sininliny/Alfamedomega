#include "alfa/tensor/Model.h"

#include "alfa/tensor/Schema.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

namespace alfa::tensor
{
namespace
{
template <typename Scalar>
Scalar sigmoid(Scalar value) noexcept
{
    if (value >= Scalar{0})
        return Scalar{1} / (Scalar{1} + std::exp(-value));
    const auto exponential = std::exp(value);
    return exponential / (Scalar{1} + exponential);
}

template <typename Scalar>
bool finite(std::span<const Scalar> values) noexcept
{
    return std::all_of(values.begin(), values.end(), [](Scalar value) { return std::isfinite(value); });
}

template <typename Scalar>
void linear(std::span<const Scalar> input, std::span<const Scalar> weight, std::span<const Scalar> bias,
            std::span<Scalar> output) noexcept
{
    for (std::size_t row = 0; row < output.size(); ++row)
    {
        Scalar value = bias[row];
        for (std::size_t column = 0; column < input.size(); ++column)
            value += weight[row * input.size() + column] * input[column];
        output[row] = value;
    }
}

std::size_t positiveSize(const nlohmann::json& value, bool allowZero = false)
{
    if (!value.is_number_integer() || value.get<std::int64_t>() < 0)
        throw std::invalid_argument("Invalid model dimension");
    const auto result = value.get<std::size_t>();
    if (!allowZero && result == 0)
        throw std::invalid_argument("Model dimension cannot be zero");
    return result;
}
} // namespace

template <typename Scalar>
bool TensorModelT<Scalar>::prepare(std::string_view json, const ConfigurationT<Scalar>& controls,
                                   std::string& error)
{
    error.clear();
    try
    {
        const auto document = nlohmann::json::parse(json);
        if (!alfa::tensor::schemaIs(document.at("schema"), "amo.runtime.gru/1") ||
            !alfa::tensor::schemaIs(document.at("ontology"), "amo.math/1"))
            throw std::invalid_argument("Unsupported runtime compiler model");
        TensorModelT<Scalar> candidate;
        const std::array<std::string, 7> categories{"audio",  "midi",    "gesture",     "user_preference",
                                                    "latent", "context", "wave_feature"};
        std::size_t inputWidth = 0;
        for (std::size_t category = 0; category < categories.size(); ++category)
        {
            candidate.widths_[category] =
                positiveSize(document.at("input_widths").at(categories[category]), true);
            if (candidate.widths_[category] > std::numeric_limits<std::size_t>::max() - inputWidth)
                throw std::invalid_argument("Input dimensions overflow");
            inputWidth += candidate.widths_[category];
        }
        candidate.hidden_ = positiveSize(document.at("hidden_width"));
        candidate.clock_ = {positiveSize(document.at("clock").at("period_samples")),
                            positiveSize(document.at("clock").at("phase_samples"), true)};
        if (candidate.clock_.phaseSamples >= candidate.clock_.periodSamples)
            throw std::invalid_argument("Invalid model execution clock");
        candidate.threshold_ = document.at("presence_threshold").get<Scalar>();
        if (!(candidate.threshold_ > Scalar{0} && candidate.threshold_ < Scalar{1}))
            throw std::invalid_argument("Invalid model presence threshold");
        candidate.controls_ = {controls.activation, controls.parameters, controls.routing,
                               controls.modulation};
        candidate.locks_ = controls.parameterLocks;
        candidate.lockedValues_ = controls.parameters.values;
        const std::array<std::string, 4> names{"activation", "parameters", "routing", "modulation"};
        std::array<bool, 4> seen{};
        std::size_t outputs = 0;
        for (const auto& head : document.at("schemas"))
        {
            const auto name = head.at("name").get<std::string>();
            const auto found = std::find(names.begin(), names.end(), name);
            if (found == names.end())
                throw std::invalid_argument("Unknown prediction tensor category");
            const auto index = static_cast<std::size_t>(std::distance(names.begin(), found));
            const auto& tensor = candidate.controls_[index];
            if (seen[index] || head.at("axes").get<std::vector<std::string>>() != tensor.axes ||
                head.at("shape").get<std::vector<std::size_t>>() != tensor.shape)
                throw std::invalid_argument("Prediction coordinates differ from the shared tensor ontology");
            seen[index] = true;
            candidate.heads_.push_back({index, {outputs, tensor.values.size()}});
            if (tensor.values.size() > std::numeric_limits<std::size_t>::max() - outputs)
                throw std::invalid_argument("Prediction dimensions overflow");
            outputs += tensor.values.size();
        }
        if (inputWidth == 0 || outputs == 0 || std::find(seen.begin(), seen.end(), false) != seen.end())
            throw std::invalid_argument("Model requires all four prediction tensor categories");
        if (candidate.hidden_ > std::numeric_limits<std::size_t>::max() / 3)
            throw std::invalid_argument("Hidden width overflow");
        const auto gates = 3 * candidate.hidden_;
        const auto& weights = document.at("weights");
        const auto& budget = document.at("budget");
        const auto availableBytes = positiveSize(budget.at("available_ram_bytes"));
        const auto estimatedBytes = positiveSize(budget.at("estimated_peak_inference_bytes"));
        if (estimatedBytes > availableBytes)
            throw std::invalid_argument("Compiler model exceeds its independent RAM allowance");
        std::size_t parameters = 0;
        const auto read =
            [&](const char* name, std::size_t rows, std::size_t columns, std::vector<Scalar>& destination)
        {
            if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns)
                throw std::invalid_argument("Weight dimensions overflow");
            const auto& source = weights.at(name);
            if (!source.is_array() || source.size() != rows * columns)
                throw std::invalid_argument("Invalid compiler model weights");
            if (source.size() > std::numeric_limits<std::size_t>::max() - parameters)
                throw std::invalid_argument("Weight capacity overflow");
            parameters += source.size();
            if (static_cast<long double>(parameters) * sizeof(Scalar) > availableBytes)
                throw std::invalid_argument("Compiler weights exceed RAM allowance before allocation");
            destination = source.get<std::vector<Scalar>>();
            if (!finite<Scalar>(destination))
                throw std::invalid_argument("Invalid compiler model weights");
        };
        read("cell.weight_ih", gates, inputWidth, candidate.inputWeights_);
        read("cell.weight_hh", gates, candidate.hidden_, candidate.stateWeights_);
        read("cell.bias_ih", gates, 1, candidate.inputBias_);
        read("cell.bias_hh", gates, 1, candidate.stateBias_);
        read("values.weight", outputs, candidate.hidden_, candidate.valueWeights_);
        read("values.bias", outputs, 1, candidate.valueBias_);
        read("presence.weight", outputs, candidate.hidden_, candidate.maskWeights_);
        read("presence.bias", outputs, 1, candidate.maskBias_);
        if (parameters != positiveSize(budget.at("parameter_elements")))
            throw std::invalid_argument("Compiler model parameter count differs from its budget");
        const long double minimumBytes =
            static_cast<long double>(sizeof(Scalar)) *
            (static_cast<long double>(parameters) + inputWidth + 9.0L * candidate.hidden_ + 6.0L * outputs +
             candidate.lockedValues_.size());
        if (estimatedBytes > availableBytes || minimumBytes > availableBytes)
            throw std::invalid_argument("Compiler model exceeds its independent RAM allowance");
        candidate.bounds_ = document.at("output_bounds").get<std::vector<std::array<Scalar, 2>>>();
        if (candidate.bounds_.size() != outputs)
            throw std::invalid_argument("Every output requires explicit bounds");
        for (const auto& bound : candidate.bounds_)
            if (!std::isfinite(bound[0]) || !std::isfinite(bound[1]) || bound[0] > bound[1])
                throw std::invalid_argument("Invalid compiler output bounds");
        candidate.state_ = document.at("initial_state").get<std::vector<Scalar>>();
        if (candidate.state_.size() != candidate.hidden_ || !finite<Scalar>(candidate.state_))
            throw std::invalid_argument("Invalid neural context state");
        candidate.initialState_ = candidate.state_;
        candidate.input_.resize(inputWidth);
        candidate.nextState_.resize(candidate.hidden_);
        candidate.inputGates_.resize(gates);
        candidate.stateGates_.resize(gates);
        candidate.values_.resize(outputs);
        candidate.logits_.resize(outputs);
        for (std::size_t i = 0; i < 4; ++i)
            candidate.scopedBackup_[i].resize(candidate.controls_[i].values.size());
        *this = std::move(candidate);
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}

template <typename Scalar>
bool TensorModelT<Scalar>::infer(const InputsT<Scalar>& inputs) noexcept
{
    if (hidden_ == 0)
        return false;
    const std::array<TensorViewT<Scalar>, 7> sources{
        inputs.audio,       inputs.midi,    inputs.gesture,    inputs.userPreference,
        inputs.latentSound, inputs.context, inputs.waveFeature};
    std::size_t offset = 0;
    for (std::size_t category = 0; category < sources.size(); ++category)
    {
        if (sources[category].values.size() < widths_[category])
            return false;
        const auto values = sources[category].values.first(widths_[category]);
        if (!finite<Scalar>(values))
            return false;
        std::copy(values.begin(), values.end(), input_.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += widths_[category];
    }
    linear<Scalar>(input_, inputWeights_, inputBias_, inputGates_);
    linear<Scalar>(state_, stateWeights_, stateBias_, stateGates_);
    for (std::size_t index = 0; index < hidden_; ++index)
    {
        const auto reset = sigmoid(inputGates_[index] + stateGates_[index]);
        const auto update = sigmoid(inputGates_[hidden_ + index] + stateGates_[hidden_ + index]);
        const auto candidate =
            std::tanh(inputGates_[2 * hidden_ + index] + reset * stateGates_[2 * hidden_ + index]);
        nextState_[index] = (Scalar{1} - update) * candidate + update * state_[index];
    }
    linear<Scalar>(nextState_, valueWeights_, valueBias_, values_);
    linear<Scalar>(nextState_, maskWeights_, maskBias_, logits_);
    if (!finite<Scalar>(nextState_) || !finite<Scalar>(values_) || !finite<Scalar>(logits_))
        return false;
    state_.swap(nextState_);
    return true;
}

template <typename Scalar>
bool TensorModelT<Scalar>::apply(const InputsT<Scalar>& inputs, RuntimeT<Scalar>& runtime) noexcept
{
    if (hidden_ == 0)
        return false;
    if (runtime.sample() % clock_.periodSamples != clock_.phaseSamples)
        return true;
    if (!infer(inputs))
        return false;
    for (const auto& head : heads_)
    {
        auto& tensor = controls_[head.first];
        for (std::size_t element = 0; element < head.second.count; ++element)
        {
            const auto index = head.second.offset + element;
            const bool represented = tensor.presence.empty() || tensor.presence[element] != 0;
            const bool predicted = sigmoid(logits_[index]) >= threshold_;
            if (represented != predicted)
            {
                state_.swap(nextState_);
                return false; // A structural change must be recompiled off-thread.
            }
            tensor.values[element] =
                represented ? std::clamp(values_[index], bounds_[index][0], bounds_[index][1]) : Scalar{0};
            if (head.first == 1 && !locks_.empty() && locks_[element] != 0)
                tensor.values[element] = lockedValues_[element];
        }
    }
    const auto view = [](const TensorT<Scalar>& tensor) -> TensorViewT<Scalar>
    {
        return {tensor.axes, tensor.shape, tensor.values, tensor.presence};
    };
    if (!runtime.updateTensors(view(controls_[0]), view(controls_[1]), view(controls_[2]),
                               view(controls_[3])))
    {
        state_.swap(nextState_);
        return false;
    }
    return true;
}

template <typename Scalar>
bool TensorModelT<Scalar>::applyScoped(const InputsT<Scalar>& inputs, RuntimeT<Scalar>& runtime,
                                       std::array<TensorT<Scalar>*, 4> controls,
                                       const std::array<std::size_t, 4>& offsets) noexcept
{
    if (hidden_ == 0)
        return false;
    if (runtime.sample() % clock_.periodSamples != clock_.phaseSamples)
        return true;
    for (std::size_t category = 0; category < 4; ++category)
    {
        const auto count = controls_[category].values.size();
        if (offsets[category] > controls[category]->values.size() ||
            count > controls[category]->values.size() - offsets[category])
            return false;
        std::copy_n(controls[category]->values.begin() + static_cast<std::ptrdiff_t>(offsets[category]),
                    count, scopedBackup_[category].begin());
    }
    if (!infer(inputs))
        return false;
    bool valid = true;
    for (const auto& head : heads_)
        for (std::size_t element = 0; element < head.second.count; ++element)
        {
            const auto index = head.second.offset + element;
            const bool represented =
                controls_[head.first].presence.empty() || controls_[head.first].presence[element] != 0;
            if (represented != (sigmoid(logits_[index]) >= threshold_))
            {
                valid = false;
                break;
            }
            auto& destination = controls[head.first]->values[offsets[head.first] + element];
            destination =
                represented ? std::clamp(values_[index], bounds_[index][0], bounds_[index][1]) : Scalar{0};
            if (head.first == 1 && !locks_.empty() && locks_[element])
                destination = lockedValues_[element];
        }
    const auto view = [](const TensorT<Scalar>* t) -> TensorViewT<Scalar>
    {
        return {t->axes, t->shape, t->values, t->presence};
    };
    if (valid &&
        runtime.updateTensors(view(controls[0]), view(controls[1]), view(controls[2]), view(controls[3])))
        return true;
    for (std::size_t i = 0; i < 4; ++i)
        std::copy(scopedBackup_[i].begin(), scopedBackup_[i].end(),
                  controls[i]->values.begin() + static_cast<std::ptrdiff_t>(offsets[i]));
    state_.swap(nextState_);
    return false;
}

template <typename Scalar>
bool TensorModelT<Scalar>::canAdoptStateFrom(const TensorModelT& previous) const noexcept
{
    return hidden_ > 0 && hidden_ == previous.hidden_ && widths_ == previous.widths_ &&
           clock_.periodSamples == previous.clock_.periodSamples &&
           clock_.phaseSamples == previous.clock_.phaseSamples && state_.size() == previous.state_.size() &&
           nextState_.size() == previous.nextState_.size() && values_.size() == previous.values_.size() &&
           logits_.size() == previous.logits_.size();
}
template <typename Scalar>
bool TensorModelT<Scalar>::adoptStateFrom(const TensorModelT& previous) noexcept
{
    if (!canAdoptStateFrom(previous))
        return false;
    std::copy(previous.state_.begin(), previous.state_.end(), state_.begin());
    std::copy(previous.nextState_.begin(), previous.nextState_.end(), nextState_.begin());
    std::copy(previous.values_.begin(), previous.values_.end(), values_.begin());
    std::copy(previous.logits_.begin(), previous.logits_.end(), logits_.begin());
    return true;
}
template <typename Scalar>
bool TensorModelT<Scalar>::takeStateFrom(TensorModelT& previous) noexcept
{
    if (!canAdoptStateFrom(previous))
        return false;
    state_.swap(previous.state_);
    nextState_.swap(previous.nextState_);
    values_.swap(previous.values_);
    logits_.swap(previous.logits_);
    return true;
}

template <typename Scalar>
std::size_t TensorModelT<Scalar>::numericStorageBytes() const noexcept
{
    size_t count = inputWeights_.size() + stateWeights_.size() + inputBias_.size() + stateBias_.size() +
                   valueWeights_.size() + valueBias_.size() + maskWeights_.size() + maskBias_.size() +
                   input_.size() + state_.size() + nextState_.size() + inputGates_.size() +
                   stateGates_.size() + values_.size() + logits_.size() + initialState_.size() +
                   2 * bounds_.size() + lockedValues_.size();
    for (size_t i = 0; i < 4; ++i)
        count += controls_[i].values.size() + scopedBackup_[i].size();
    return count * sizeof(Scalar);
}

template <typename Scalar>
void TensorModelT<Scalar>::reset() noexcept
{
    std::copy(initialState_.begin(), initialState_.end(), state_.begin());
}
template class TensorModelT<float>;
template class TensorModelT<double>;
} // namespace alfa::tensor
