#include "TensorExecution.h"

#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Model.h"

#include <juce_cryptography/juce_cryptography.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace alfa::tensorhost
{
namespace
{
using JSON = nlohmann::json;
using namespace alfa::tensor;
template <typename Scalar>
TensorT<Scalar> inputTensor(const JSON& data)
{
    TensorT<Scalar> tensor;
    tensor.axes = data.at("axes").get<std::vector<std::string>>();
    if (!data.at("shape").is_array())
        throw std::invalid_argument("Input tensor shape must be an array");
    size_t count = 1;
    for (const auto& value : data.at("shape"))
    {
        if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
            throw std::invalid_argument("Input tensor dimensions must be nonnegative integers");
        const auto dimension = value.get<uint64_t>();
        if (dimension > std::numeric_limits<size_t>::max() ||
            (dimension && count > std::numeric_limits<size_t>::max() / dimension))
            throw std::invalid_argument("Input tensor shape overflows");
        count *= static_cast<size_t>(dimension);
        tensor.shape.push_back(static_cast<size_t>(dimension));
    }
    tensor.values = data.at("values").get<std::vector<Scalar>>();
    if (tensor.axes.empty() && tensor.shape.empty() && tensor.values.empty())
        count = 0;
    if (tensor.axes.size() != tensor.shape.size() || tensor.values.size() != count)
        throw std::invalid_argument("Input tensor axes/shape/values differ");
    for (size_t axis = 0; axis < tensor.axes.size(); ++axis)
    {
        if (tensor.axes[axis].empty() ||
            std::find(tensor.axes.begin(), tensor.axes.begin() + static_cast<std::ptrdiff_t>(axis),
                      tensor.axes[axis]) != tensor.axes.begin() + static_cast<std::ptrdiff_t>(axis))
            throw std::invalid_argument("Input tensor axes must be unique and named");
    }
    for (const auto value : tensor.values)
        if (!std::isfinite(value))
            throw std::invalid_argument("Input tensor values must be finite");
    if (data.contains("presence"))
    {
        const auto& mask = data.at("presence");
        if (!mask.is_array() || (!mask.empty() && mask.size() != count))
            throw std::invalid_argument("Input presence must match values");
        for (size_t i = 0; i < mask.size(); ++i)
        {
            if (!mask[i].is_boolean() &&
                (!mask[i].is_number_integer() || mask[i].get<int64_t>() < 0 || mask[i].get<int64_t>() > 1))
                throw std::invalid_argument("Input presence must be binary");
            const auto bit = mask[i].is_boolean() ? (mask[i].get<bool>() ? 1 : 0) : mask[i].get<int>();
            if (!bit && tensor.values[i] != Scalar(0))
                throw std::invalid_argument("Input padding must be zero");
            tensor.presence.push_back(static_cast<uint8_t>(bit));
        }
    }
    return tensor;
}
template <typename Scalar>
TensorViewT<Scalar> view(const TensorT<Scalar>& tensor)
{
    return {tensor.axes, tensor.shape, tensor.values, tensor.presence};
}
template <typename Scalar>
class Execution final : public TensorExecution
{
    struct Layer
    {
        std::vector<Scalar> base, values, matrix;
        std::vector<std::array<Scalar, 2>> ranges;
        size_t destinations = 0;
    };
    struct Mix
    {
        size_t output;
        int channel;
        Scalar gain;
    };
    RuntimeT<Scalar> runtime_;
    ConfigurationT<Scalar> controls_;
    std::vector<Scalar> mapped_, highest_;
    std::array<std::vector<Scalar>, 4> initialControls_;
    std::vector<Layer> layers_;
    std::vector<Mix> mix_;
    std::unique_ptr<TensorModelT<Scalar>> model_;
    juce::String modelIdentity_;
    Clock modelClock_;
    struct ScopedModel
    {
        TensorModelT<Scalar> model;
        juce::String identity;
        std::array<size_t, 4> offsets{};
        TensorT<Scalar> latent;
        Clock clock;
    };
    std::vector<ScopedModel> scopedModels_;
    TensorT<Scalar> latent_;
    std::array<std::vector<Scalar>, 6> boundary_;
    std::string report_;

public:
    bool prepare(std::string_view text, double rate, std::string& error)
    {
        try
        {
            const auto data = JSON::parse(text);
            readNativePolicy(data);
            for (const auto* name :
                 {"host_alignment_bytes", "boundary_conversion_bytes", "compiler_model_bytes"})
                if (data.at("budget").contains(name))
                {
                    const auto& capacity = data.at("budget").at(name);
                    if (!capacity.is_number_integer() ||
                        (!capacity.is_number_unsigned() && capacity.get<int64_t>() < 0) ||
                        capacity.get<uint64_t>() > std::numeric_limits<size_t>::max())
                        throw std::invalid_argument("Native host capacities must be nonnegative integers");
                }
            if (data.contains("host_latency_samples"))
            {
                const auto& latency = data.at("host_latency_samples");
                if ((!latency.is_number_unsigned() && !latency.is_number_integer()) ||
                    latency.get<double>() < 0 || latency.get<double>() > std::numeric_limits<int>::max())
                    throw std::invalid_argument("Invalid explicit native host latency");
                if (latency.get<uint64_t>() * 8 >
                    data.at("budget").value("host_alignment_bytes", uint64_t(0)))
                    throw std::invalid_argument("Dry/wet alignment exceeds host_alignment_bytes budget");
            }
            Budget budget;
            if (!parseJSON(text, controls_, error) || !readBudgetJSON(text, budget, error) ||
                !runtime_.prepare(controls_, budget, rate, error))
                return false;
            mapped_ = controls_.parameters.values;
            size_t stored = controls_.activation.values.size() + controls_.parameters.values.size() +
                            controls_.routing.values.size() + controls_.modulation.values.size();
            for (const auto& instance : controls_.instances)
            {
                stored += instance.inputChannels;
                for (const auto& asset : instance.assets)
                    stored += asset.values.size();
            }
            if (data.contains("control_layers"))
            {
                const auto& layers = data.at("control_layers");
                if (!layers.is_array())
                    throw std::invalid_argument("Control layers must be an array");
                for (size_t i = 0; i < layers.size(); ++i)
                {
                    Layer layer;
                    layer.base = layers[i].at("values").get<std::vector<Scalar>>();
                    layer.values = layer.base;
                    layer.ranges = layers[i].at("ranges").get<std::vector<std::array<Scalar, 2>>>();
                    const auto matrix = inputTensor<Scalar>(layers[i].at("matrix"));
                    layer.destinations = i + 1 < layers.size() ? layers[i + 1].at("values").size()
                                                               : controls_.parameters.values.size();
                    if (layer.ranges.size() != layer.base.size() ||
                        matrix.shape != std::vector<size_t>{layer.destinations, layer.base.size()} ||
                        matrix.axes != std::vector<std::string>{"destination", "source"})
                        throw std::invalid_argument("Control layer matrix has incompatible coordinates");
                    for (size_t n = 0; n < layer.base.size(); ++n)
                        if (!std::isfinite(layer.base[n]) || !std::isfinite(layer.ranges[n][0]) ||
                            !std::isfinite(layer.ranges[n][1]) || layer.ranges[n][0] > layer.ranges[n][1] ||
                            layer.base[n] < layer.ranges[n][0] || layer.base[n] > layer.ranges[n][1])
                            throw std::invalid_argument("Invalid X control range or value");
                    const auto count = layer.base.size() + matrix.values.size();
                    if (stored > budget.storedTensorElements || count > budget.storedTensorElements - stored)
                        throw std::invalid_argument("Control layers exceed stored tensor capacity");
                    stored += count;
                    layer.matrix = matrix.values;
                    layers_.push_back(std::move(layer));
                }
            }
            const auto accountLatent = [&](size_t count)
            {
                if (stored > budget.storedTensorElements || count > budget.storedTensorElements - stored)
                    throw std::invalid_argument("Latent inputs exceed stored tensor capacity");
                stored += count;
            };
            const auto validateModelInputs = [](const JSON& model, size_t latentCount)
            {
                const std::array<const char*, 6> names{"audio",           "midi",   "gesture",
                                                       "user_preference", "latent", "context"};
                const std::array<size_t, 6> capacities{2, 1280, 48, 5, latentCount, 8};
                for (size_t i = 0; i < names.size(); ++i)
                    if (model.at("input_widths").at(names[i]).get<uint64_t>() > capacities[i])
                        throw std::invalid_argument(
                            "Compiler input width exceeds the native host source capacity");
            };
            if (!layers_.empty())
                highest_ = layers_[0].base;
            if (data.contains("latentSound"))
            {
                latent_ = inputTensor<Scalar>(data.at("latentSound"));
                accountLatent(latent_.values.size());
            }
            if (data.contains("runtimeModel"))
            {
                const auto serializedModel = data.at("runtimeModel").dump();
                modelIdentity_ = juce::SHA256(serializedModel.data(), serializedModel.size()).toHexString();
                model_ = std::make_unique<TensorModelT<Scalar>>();
                if (!model_->prepare(data.at("runtimeModel").dump(), controls_, error))
                    return false;
                validateModelInputs(data.at("runtimeModel"), latent_.values.size());
                if (data.at("budget").contains("compiler_model_bytes") &&
                    model_->numericStorageBytes() >
                        data.at("budget").at("compiler_model_bytes").get<size_t>())
                    throw std::invalid_argument("Compiler model exceeds compiler_model_bytes budget");
                const auto& clock = data.at("runtimeModel").at("clock");
                modelClock_ = {clock.at("period_samples").get<uint64_t>(),
                               clock.at("phase_samples").get<uint64_t>()};
            }
            if (data.contains("runtimeModels"))
            {
                if (model_)
                    throw std::invalid_argument("Global and scoped compiler models cannot overlap");
                if (!data.at("runtimeModels").is_array())
                    throw std::invalid_argument("Scoped compiler models must be an array");
                std::set<std::string> identities;
                std::array<std::vector<uint8_t>, 4> claimed;
                const std::array<const TensorT<Scalar>*, 4> tensors{
                    &controls_.activation, &controls_.parameters, &controls_.routing, &controls_.modulation};
                for (size_t i = 0; i < 4; ++i)
                    claimed[i].resize(tensors[i]->values.size());
                for (const auto& entry : data.at("runtimeModels"))
                {
                    const auto id = entry.at("id").get<std::string>();
                    if (id.empty() || !identities.insert(id).second)
                        throw std::invalid_argument("Scoped compiler model IDs must be nonempty and unique");
                    ScopedModel unit;
                    const auto& offsets = entry.at("offsets");
                    if (!offsets.is_array() || offsets.size() != 4)
                        throw std::invalid_argument("Scoped models require exactly four tensor offsets");
                    for (const auto& offset : offsets)
                        if (!offset.is_number_integer() ||
                            (!offset.is_number_unsigned() && offset.get<int64_t>() < 0) ||
                            offset.get<uint64_t>() > std::numeric_limits<size_t>::max())
                            throw std::invalid_argument("Scoped model offsets must be nonnegative integers");
                    unit.offsets = offsets.get<std::array<size_t, 4>>();
                    ConfigurationT<Scalar> local;
                    auto localDocument = entry.at("controls");
                    localDocument.erase("tier");
                    localDocument["scalar_type"] = std::is_same_v<Scalar, double> ? "float64" : "float32";
                    if (!parseJSON(localDocument.dump(), local, error))
                        return false;
                    const std::array<const TensorT<Scalar>*, 4> source{&local.activation, &local.parameters,
                                                                       &local.routing, &local.modulation};
                    for (size_t category = 0; category < 4; ++category)
                    {
                        if (unit.offsets[category] > claimed[category].size() ||
                            source[category]->values.size() >
                                claimed[category].size() - unit.offsets[category])
                            throw std::invalid_argument("Scoped model lies outside composed tensors");
                        for (size_t element = 0; element < source[category]->values.size(); ++element)
                        {
                            const auto index = unit.offsets[category] + element;
                            if (claimed[category][index]++)
                                throw std::invalid_argument("Scoped models overlap tensor authority");
                            const auto& target = *tensors[category];
                            if ((source[category]->presence.empty() ? 1
                                                                    : source[category]->presence[element]) !=
                                (target.presence.empty() ? 1 : target.presence[index]))
                                throw std::invalid_argument("Scoped model presence differs from target");
                        }
                    }
                    local.parameterLocks.resize(local.parameters.values.size());
                    for (size_t i = 0; i < local.parameterLocks.size(); ++i)
                        if (!controls_.parameterLocks.empty())
                            local.parameterLocks[i] = controls_.parameterLocks[unit.offsets[1] + i];
                    const std::array<TensorT<Scalar>*, 4> localTensors{&local.activation, &local.parameters,
                                                                       &local.routing, &local.modulation};
                    for (size_t category = 0; category < 4; ++category)
                        std::copy_n(tensors[category]->values.begin() +
                                        static_cast<std::ptrdiff_t>(unit.offsets[category]),
                                    localTensors[category]->values.size(),
                                    localTensors[category]->values.begin());
                    if (!unit.model.prepare(entry.at("model").dump(), local, error))
                        return false;
                    if (entry.at("controls").contains("latentSound"))
                        unit.latent = inputTensor<Scalar>(entry.at("controls").at("latentSound"));
                    validateModelInputs(entry.at("model"), unit.latent.values.size());
                    accountLatent(unit.latent.values.size());
                    const auto& clock = entry.at("model").at("clock");
                    unit.clock = {clock.at("period_samples").get<uint64_t>(),
                                  clock.at("phase_samples").get<uint64_t>()};
                    const auto serialized = entry.at("model").dump() + entry.at("offsets").dump();
                    unit.identity = juce::SHA256(serialized.data(), serialized.size()).toHexString();
                    scopedModels_.push_back(std::move(unit));
                }
                size_t modelBytes = 0;
                for (const auto& unit : scopedModels_)
                {
                    const auto bytes = unit.model.numericStorageBytes();
                    const auto allowance = data.at("budget").value("compiler_model_bytes", size_t(0));
                    if (modelBytes > allowance || bytes > allowance - modelBytes)
                        throw std::invalid_argument(
                            "Scoped compiler models exceed aggregate compiler_model_bytes budget");
                    modelBytes += bytes;
                }
            }
            if (!data.at("outputMix").is_array())
                throw std::invalid_argument("outputMix must be an array");
            for (const auto& item : data.at("outputMix"))
            {
                if (!item.is_array() || item.size() != 3 || !item.at(0).is_number_integer() ||
                    (!item.at(0).is_number_unsigned() && item.at(0).get<int64_t>() < 0) ||
                    !item.at(1).is_number_integer() || item.at(1).get<double>() < 0 ||
                    item.at(1).get<double>() > 1 || !item.at(2).is_number())
                    throw std::invalid_argument("outputMix requires output/channel/gain triples");
                Mix mix{item.at(0).get<size_t>(), item.at(1).get<int>(), item.at(2).get<Scalar>()};
                if (mix.output >= runtime_.output().size() || mix.channel < 0 || mix.channel > 1 ||
                    !std::isfinite(mix.gain))
                    throw std::invalid_argument("Invalid explicit outputMix");
                mix_.push_back(mix);
            }
            if constexpr (std::is_same_v<Scalar, double>)
            {
                // These are native-host conversion capacities, not primitive slots.
                const std::array<size_t, 6> capacities{2, 1280, 48, 5, latent_.values.size(), 8};
                size_t elements = 0;
                for (const auto count : capacities)
                    elements += count;
                if (data.at("budget").value("boundary_conversion_bytes", size_t(0)) <
                    elements * sizeof(Scalar))
                    throw std::invalid_argument(
                        "Float64 host input conversion exceeds declared boundary_conversion_bytes");
                for (size_t i = 0; i < boundary_.size(); ++i)
                    if (i != 4)
                        boundary_[i].resize(capacities[i]);
            }
            if (model_ || !scopedModels_.empty())
                initialControls_ = {controls_.activation.values, controls_.parameters.values,
                                    controls_.routing.values, controls_.modulation.values};
            std::ostringstream report;
            report << "Uniform scalar type: " << (std::is_same_v<Scalar, double> ? "float64" : "float32")
                   << " | tier: " << data.value("tier", std::string("legacy / explicit"));
            report << "\nCompiled state elements: " << runtime_.state().size()
                   << " | output elements: " << runtime_.output().size()
                   << " | scheduled primitives: " << runtime_.schedule().size();
            report << "\nDeclared host latency: " << data.value("host_latency_samples", 0) << " samples";
            report
                << "\nDeclared budgets, routing, activation, allocation and clocks are inspectable in JSON.";
            report << (model_ ? "\nRuntime GRU compiler model loaded; explicit model clock."
                              : "\nNo global compiler model is loaded.");
            report << "\nScoped fragment compiler models: " << scopedModels_.size();
            for (size_t i = 0; i < controls_.instances.size(); ++i)
            {
                const auto& instance = controls_.instances[i];
                const auto state = runtime_.stateAllocation()[i];
                report << "\n"
                       << instance.primitive->id
                       << " activation=" << controls_.activation.values[instance.activationIndex]
                       << " clock=" << instance.clock.periodSamples << " samples; state [" << state.offset
                       << "," << state.count << "]";
            }
            report << "\nProjection terms: " << controls_.projections.size()
                   << "\nParameter locks: " << data.value("parameter_locks", JSON::array()).dump();
            report_ = report.str();
            error.clear();
            return true;
        }
        catch (const std::exception& failure)
        {
            error = failure.what();
            return false;
        }
    }
    bool updateControls(std::span<const float> values) noexcept override
    {
        const auto count = std::min(values.size(), highest_.size());
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(values[i]))
                return false;
        for (size_t i = 0; i < highest_.size(); ++i)
            highest_[i] = i < count ? std::clamp(static_cast<Scalar>(values[i]), layers_[0].ranges[i][0],
                                                 layers_[0].ranges[i][1])
                                    : layers_[0].base[i];
        return mapControls();
    }
    bool mapControls() noexcept
    {
        if (layers_.empty())
            return true;
        for (auto& layer : layers_)
            std::copy(layer.base.begin(), layer.base.end(), layer.values.begin());
        std::copy(highest_.begin(), highest_.end(), layers_[0].values.begin());
        std::copy(controls_.parameters.values.begin(), controls_.parameters.values.end(), mapped_.begin());
        for (size_t i = 0; i < layers_.size(); ++i)
        {
            auto& layer = layers_[i];
            auto& destination = i + 1 < layers_.size() ? layers_[i + 1].values : mapped_;
            for (size_t d = 0; d < layer.destinations; ++d)
                for (size_t s = 0; s < layer.values.size(); ++s)
                    destination[d] += layer.matrix[d * layer.values.size() + s] * layer.values[s];
            if (i + 1 < layers_.size())
                for (size_t d = 0; d < destination.size(); ++d)
                    destination[d] =
                        std::clamp(destination[d], layers_[i + 1].ranges[d][0], layers_[i + 1].ranges[d][1]);
        }
        for (size_t i = 0; i < controls_.parameterLocks.size(); ++i)
            if (controls_.parameterLocks[i])
                mapped_[i] = controls_.parameters.values[i];
        auto parameters = view(controls_.parameters);
        parameters.values = mapped_;
        return runtime_.updateTensors(view(controls_.activation), parameters, view(controls_.routing),
                                      view(controls_.modulation));
    }
    TickResult tick(const Inputs& native, double& modelMilliseconds) noexcept override
    {
        InputsT<Scalar> inputs;
        if constexpr (std::is_same_v<Scalar, float>)
            inputs = native;
        else
        {
            const std::array<TensorView, 6> source{native.audio,          native.midi,        native.gesture,
                                                   native.userPreference, native.latentSound, native.context};
            std::array<TensorViewT<Scalar>, 6> converted;
            for (size_t i = 0; i < source.size(); ++i)
            {
                if (i == 4)
                    continue;
                if (source[i].values.size() > boundary_[i].size())
                    return {false, true};
                std::transform(source[i].values.begin(), source[i].values.end(), boundary_[i].begin(),
                               [](float value) { return static_cast<Scalar>(value); });
                converted[i] = {source[i].axes, source[i].shape,
                                std::span<const Scalar>(boundary_[i]).first(source[i].values.size()),
                                source[i].presence};
            }
            inputs = {converted[0], converted[1], converted[2], converted[3], converted[4], converted[5]};
        }
        inputs.latentSound = view(latent_);
        bool accepted = true, modelEvent = false;
        if (model_ && runtime_.sample() % modelClock_.periodSamples == modelClock_.phaseSamples)
        {
            modelEvent = true;
            const auto began = std::chrono::steady_clock::now();
            accepted = model_->applyScoped(
                inputs, runtime_,
                {&controls_.activation, &controls_.parameters, &controls_.routing, &controls_.modulation},
                {0, 0, 0, 0});
            modelMilliseconds +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
        }
        for (auto& unit : scopedModels_)
        {
            if (runtime_.sample() % unit.clock.periodSamples != unit.clock.phaseSamples)
                continue;
            modelEvent = true;
            auto localInputs = inputs;
            localInputs.latentSound = view(unit.latent);
            const auto began = std::chrono::steady_clock::now();
            accepted = unit.model.applyScoped(localInputs, runtime_,
                                              {&controls_.activation, &controls_.parameters,
                                               &controls_.routing, &controls_.modulation},
                                              unit.offsets) &&
                       accepted;
            modelMilliseconds +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
        }
        // Predictions are the current base tensors. User matrices are applied
        // afterwards, including between model events, without accumulating offsets.
        if (modelEvent && !mapControls())
            return {false, false};
        return {runtime_.tick(inputs), accepted};
    }
    float mixedOutput(int channel) const noexcept override
    {
        Scalar result = 0;
        const auto output = runtime_.output();
        for (const auto& mix : mix_)
            if (mix.channel == channel)
                result += mix.gain * output[mix.output];
        return static_cast<float>(result);
    }
    void reset() noexcept override
    {
        runtime_.reset();
        if (model_)
            model_->reset();
        for (auto& unit : scopedModels_)
            unit.model.reset();
        if (model_ || !scopedModels_.empty())
        {
            const std::array<TensorT<Scalar>*, 4> tensors{&controls_.activation, &controls_.parameters,
                                                          &controls_.routing, &controls_.modulation};
            for (size_t i = 0; i < tensors.size(); ++i)
                std::copy(initialControls_[i].begin(), initialControls_[i].end(), tensors[i]->values.begin());
        }
        if (!layers_.empty())
            std::copy(layers_[0].base.begin(), layers_[0].base.end(), highest_.begin());
        mapControls();
    }
    bool adopt(TensorExecution& old) noexcept override
    {
        auto* previous = dynamic_cast<Execution*>(&old);
        if (!previous || static_cast<bool>(model_) != static_cast<bool>(previous->model_))
            return false;
        if (scopedModels_.size() != previous->scopedModels_.size())
            return false;
        for (size_t i = 0; i < scopedModels_.size(); ++i)
            if (scopedModels_[i].identity != previous->scopedModels_[i].identity ||
                !scopedModels_[i].model.canAdoptStateFrom(previous->scopedModels_[i].model))
                return false;
        if (model_ &&
            (modelIdentity_ != previous->modelIdentity_ || !model_->canAdoptStateFrom(*previous->model_)))
            return false;
        if (!runtime_.takeStateFrom(previous->runtime_))
            return false;
        for (size_t i = 0; i < scopedModels_.size(); ++i)
            scopedModels_[i].model.takeStateFrom(previous->scopedModels_[i].model);
        return !model_ || model_->takeStateFrom(*previous->model_);
    }
    void snapshot(std::span<std::atomic<float>> destination) const noexcept override
    {
        const auto state = runtime_.state();
        for (size_t i = 0; i < destination.size(); ++i)
            destination[i].store(i < state.size() ? static_cast<float>(state[i]) : 0.0f,
                                 std::memory_order_relaxed);
    }
    std::vector<float> highestDefaults() const override
    {
        std::vector<float> result;
        if (!layers_.empty())
            for (const auto value : layers_[0].base)
                result.push_back(static_cast<float>(value));
        return result;
    }
    std::string inspection() const override { return report_; }
    ScalarType scalarType() const noexcept override
    {
        return std::is_same_v<Scalar, double> ? ScalarType::Float64 : ScalarType::Float32;
    }
};
} // namespace
NativePolicy readNativePolicy(const nlohmann::json& data)
{
    NativePolicy result;
    if (data.contains("performance"))
    {
        const auto& performance = data.at("performance");
        if (!performance.is_object())
            throw std::invalid_argument("Performance policy must be an object");
        const auto layout = performance.value("midi_layout", std::string("compact"));
        if (layout == "voice_indexed")
        {
            const auto& capacity = performance.at("voice_capacity");
            if (!capacity.is_number_integer() || capacity.get<double>() < 1 || capacity.get<double>() > 128)
                throw std::invalid_argument("Voice capacity must be an integer between 1 and 128");
            result.voiceCapacity = capacity.get<size_t>();
            result.releaseSeconds = performance.at("release_seconds").get<double>();
            result.bendSemitones = performance.value("bend_range_semitones", 0.0f);
            if (!std::isfinite(result.releaseSeconds) || result.releaseSeconds < 0 ||
                result.releaseSeconds > 600 || !std::isfinite(result.bendSemitones) ||
                result.bendSemitones < 0 || result.bendSemitones > 96)
                throw std::invalid_argument("Invalid performance release or pitch movement range");
        }
        else if (layout != "compact")
            throw std::invalid_argument("Unknown MIDI layout");
    }
    result.tailSeconds = data.value("tail_seconds", result.releaseSeconds);
    if (!std::isfinite(result.tailSeconds) || result.tailSeconds < 0 || result.tailSeconds > 600)
        throw std::invalid_argument("Invalid declared audio tail");
    if (data.contains("host_latency_samples"))
    {
        const auto& value = data.at("host_latency_samples");
        if (!value.is_number_integer() || value.get<double>() < 0 ||
            value.get<double>() > std::numeric_limits<int>::max())
            throw std::invalid_argument("Declared host latency must be a nonnegative supported sample count");
        result.latencySamples = value.get<int>();
    }
    if (data.contains("preferred_mode"))
    {
        const auto& mode = data.at("preferred_mode");
        if (!mode.is_number_integer() || mode.get<double>() < 0 || mode.get<double>() > 2)
            throw std::invalid_argument("Preferred mode must be instrument, FX or hybrid (0..2)");
    }
    return result;
}
std::unique_ptr<TensorExecution> prepareExecution(std::string_view text, double rate, std::string& error)
{
    PrecisionProfile precision;
    if (!resolvePrecisionJSON(text, precision, error))
        return {};
    if (precision.scalarType == ScalarType::Float64)
    {
        auto result = std::make_unique<Execution<double>>();
        if (result->prepare(text, rate, error))
            return result;
    }
    else
    {
        auto result = std::make_unique<Execution<float>>();
        if (result->prepare(text, rate, error))
            return result;
    }
    return {};
}
} // namespace alfa::tensorhost
