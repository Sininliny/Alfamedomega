#include "alfa/tensor/Model.h"

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
float sigmoid(float value) noexcept
{
    if (value >= 0.0f)
        return 1.0f / (1.0f + std::exp(-value));
    const auto exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

bool finite(std::span<const float> values) noexcept
{
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

void linear(std::span<const float> input, std::span<const float> weight, std::span<const float> bias,
            std::span<float> output) noexcept
{
    for (std::size_t row = 0; row < output.size(); ++row)
    {
        float value = bias[row];
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

bool TensorModel::prepare(std::string_view json, const Configuration& controls, std::string& error)
{
    error.clear();
    try
    {
        const auto document = nlohmann::json::parse(json);
        if (document.at("schema") != "aom.runtime.gru/1" || document.at("ontology") != "aom.math/1")
            throw std::invalid_argument("Unsupported runtime compiler model");
        TensorModel candidate;
        const std::array<std::string, 6> categories{"audio",           "midi",   "gesture",
                                                    "user_preference", "latent", "context"};
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
        candidate.threshold_ = document.at("presence_threshold").get<float>();
        if (!(candidate.threshold_ > 0.0f && candidate.threshold_ < 1.0f))
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
            [&](const char* name, std::size_t rows, std::size_t columns, std::vector<float>& destination)
        {
            if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns)
                throw std::invalid_argument("Weight dimensions overflow");
            const auto& source = weights.at(name);
            if (!source.is_array() || source.size() != rows * columns)
                throw std::invalid_argument("Invalid compiler model weights");
            if (source.size() > std::numeric_limits<std::size_t>::max() - parameters)
                throw std::invalid_argument("Weight capacity overflow");
            parameters += source.size();
            if (static_cast<long double>(parameters) * sizeof(float) > availableBytes)
                throw std::invalid_argument("Compiler weights exceed RAM allowance before allocation");
            destination = source.get<std::vector<float>>();
            if (!finite(destination))
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
        const long double minimumBytes = 4.0L * (static_cast<long double>(parameters) + inputWidth +
                                                 8.0L * candidate.hidden_ + 6.0L * outputs);
        if (estimatedBytes > availableBytes || minimumBytes > availableBytes)
            throw std::invalid_argument("Compiler model exceeds its independent RAM allowance");
        candidate.bounds_ = document.at("output_bounds").get<std::vector<std::array<float, 2>>>();
        if (candidate.bounds_.size() != outputs)
            throw std::invalid_argument("Every output requires explicit bounds");
        for (const auto& bound : candidate.bounds_)
            if (!std::isfinite(bound[0]) || !std::isfinite(bound[1]) || bound[0] > bound[1])
                throw std::invalid_argument("Invalid compiler output bounds");
        candidate.state_ = document.at("initial_state").get<std::vector<float>>();
        if (candidate.state_.size() != candidate.hidden_ || !finite(candidate.state_))
            throw std::invalid_argument("Invalid neural context state");
        candidate.initialState_ = candidate.state_;
        candidate.input_.resize(inputWidth);
        candidate.nextState_.resize(candidate.hidden_);
        candidate.inputGates_.resize(gates);
        candidate.stateGates_.resize(gates);
        candidate.values_.resize(outputs);
        candidate.logits_.resize(outputs);
        *this = std::move(candidate);
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}

bool TensorModel::infer(const Inputs& inputs) noexcept
{
    if (hidden_ == 0)
        return false;
    const std::array<TensorView, 6> sources{inputs.audio,          inputs.midi,        inputs.gesture,
                                            inputs.userPreference, inputs.latentSound, inputs.context};
    std::size_t offset = 0;
    for (std::size_t category = 0; category < sources.size(); ++category)
    {
        if (sources[category].values.size() < widths_[category])
            return false;
        const auto values = sources[category].values.first(widths_[category]);
        if (!finite(values))
            return false;
        std::copy(values.begin(), values.end(), input_.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += widths_[category];
    }
    linear(input_, inputWeights_, inputBias_, inputGates_);
    linear(state_, stateWeights_, stateBias_, stateGates_);
    for (std::size_t index = 0; index < hidden_; ++index)
    {
        const auto reset = sigmoid(inputGates_[index] + stateGates_[index]);
        const auto update = sigmoid(inputGates_[hidden_ + index] + stateGates_[hidden_ + index]);
        const auto candidate =
            std::tanh(inputGates_[2 * hidden_ + index] + reset * stateGates_[2 * hidden_ + index]);
        nextState_[index] = (1.0f - update) * candidate + update * state_[index];
    }
    linear(nextState_, valueWeights_, valueBias_, values_);
    linear(nextState_, maskWeights_, maskBias_, logits_);
    if (!finite(nextState_) || !finite(values_) || !finite(logits_))
        return false;
    state_.swap(nextState_);
    return true;
}

bool TensorModel::apply(const Inputs& inputs, Runtime& runtime) noexcept
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
                represented ? std::clamp(values_[index], bounds_[index][0], bounds_[index][1]) : 0.0f;
            if (head.first == 1 && !locks_.empty() && locks_[element] != 0)
                tensor.values[element] = lockedValues_[element];
        }
    }
    const auto view = [](const Tensor& tensor) -> TensorView
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

void TensorModel::reset() noexcept
{
    std::copy(initialState_.begin(), initialState_.end(), state_.begin());
}
} // namespace alfa::tensor
