#include "alfa/tensor/Compiler.h"

#include "alfa/tensor/AdvancedProduction.h"
#include "alfa/tensor/Kernels.h"
#include "alfa/tensor/Precision.h"
#include "alfa/tensor/Schema.h"
#include "alfa/tensor/Universe.h"

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace alfa::tensor
{
namespace
{
using Json = nlohmann::json;

std::size_t size(const Json& value)
{
    if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<std::int64_t>() < 0))
        throw std::invalid_argument("Expected a nonnegative integer");
    const auto result = value.get<std::uint64_t>();
    if (result > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Integer exceeds addressable capacity");
    return static_cast<std::size_t>(result);
}

std::vector<std::uint8_t> mask(const Json& value)
{
    if (!value.is_array())
        throw std::invalid_argument("Mask must be an array");
    std::vector<std::uint8_t> result;
    result.reserve(value.size());
    for (const auto& element : value)
    {
        const auto bit = element.is_boolean() ? static_cast<std::size_t>(element.get<bool>()) : size(element);
        if (bit > 1)
            throw std::invalid_argument("Masks must be binary");
        result.push_back(static_cast<std::uint8_t>(bit));
    }
    return result;
}

template <typename Scalar>
TensorT<Scalar> tensor(const Json& value)
{
    TensorT<Scalar> result;
    result.axes = value.at("axes").get<std::vector<std::string>>();
    for (const auto& extent : value.at("shape"))
        result.shape.push_back(size(extent));
    result.values = value.at("values").get<std::vector<Scalar>>();
    if (value.contains("presence"))
        result.presence = mask(value.at("presence"));
    return result;
}

template <typename Scalar>
Json tensorJSON(const TensorT<Scalar>& value)
{
    Json result{{"axes", value.axes}, {"shape", value.shape}, {"values", value.values}};
    if (!value.presence.empty())
        result["presence"] = value.presence;
    return result;
}

SourceKind source(std::string_view name)
{
    if (name == "audio")
        return SourceKind::Audio;
    if (name == "midi")
        return SourceKind::Midi;
    if (name == "gesture")
        return SourceKind::Gesture;
    if (name == "user_preference")
        return SourceKind::UserPreference;
    if (name == "latent_sound")
        return SourceKind::LatentSound;
    if (name == "context")
        return SourceKind::Context;
    if (name == "state")
        return SourceKind::State;
    if (name == "output_history")
        return SourceKind::OutputHistory;
    if (name == "wave_feature")
        return SourceKind::WaveFeature;
    throw std::invalid_argument("Unknown source representation");
}

std::string_view sourceName(SourceKind kind)
{
    switch (kind)
    {
    case SourceKind::Audio:
        return "audio";
    case SourceKind::Midi:
        return "midi";
    case SourceKind::Gesture:
        return "gesture";
    case SourceKind::UserPreference:
        return "user_preference";
    case SourceKind::LatentSound:
        return "latent_sound";
    case SourceKind::Context:
        return "context";
    case SourceKind::State:
        return "state";
    case SourceKind::OutputHistory:
        return "output_history";
    case SourceKind::WaveFeature:
        return "wave_feature";
    }
    throw std::invalid_argument("Unknown source representation");
}

template <typename Scalar>
PrimitiveT<Scalar> primitive(std::string_view id, std::size_t stateElements,
                             const PrimitiveResolverT<Scalar>& extension)
{
    PrimitiveT<Scalar> result;
    if (id == "math.sine" && stateElements == 1)
        return kernels::sine<Scalar>();
    if (id == "math.one_pole_lowpass" && stateElements == 1)
        return kernels::onePole<Scalar>();
    if (id == "math.feedback_cell" && stateElements == 1)
        return kernels::feedbackCell<Scalar>();
    if (id == "math.integer_delay" && stateElements > 0)
        return kernels::delay<Scalar>(stateElements);
    std::string error;
    if (!universe::createPrimitive(id, stateElements, result, error))
    {
        if (!extension || !extension(id, stateElements, result, error))
            throw std::invalid_argument(error.empty() ? "Unknown primitive or invalid state layout" : error);
        if (result.id != id || result.stateElements != stateElements)
            throw std::invalid_argument("Extension primitive does not match its declared identity/state");
    }
    return result;
}

void validateHeader(const Json& document)
{
    if (!schemaIs(document.at("schema"), "amo.tensor/1") || !schemaIs(document.at("ontology"), "amo.math/1"))
        throw std::invalid_argument("Unsupported tensor schema or ontology version");
}
} // namespace

template <typename Scalar>
bool parseJSON(std::string_view text, ConfigurationT<Scalar>& output, std::string& error,
               const PrimitiveResolverT<Scalar>& extension)
{
    error.clear();
    try
    {
        const auto document = Json::parse(text);
        validateHeader(document);
        PrecisionProfile precision;
        if (!resolvePrecisionJSON(text, precision, error))
            return false;
        constexpr auto actual = std::is_same_v<Scalar, double> ? ScalarType::Float64 : ScalarType::Float32;
        if (precision.scalarType != actual)
        {
            error = "Compiler scalar type differs from declared tier/type";
            return false;
        }
        ConfigurationT<Scalar> candidate;
        const auto& tensors = document.at("tensors");
        candidate.activation = tensor<Scalar>(tensors.at("activation"));
        candidate.parameters = tensor<Scalar>(tensors.at("parameters"));
        candidate.routing = tensor<Scalar>(tensors.at("routing"));
        candidate.modulation = tensor<Scalar>(tensors.at("modulation"));
        for (const auto& serialized : document.at("instances"))
        {
            InstanceT<Scalar> instance;
            instance.instanceId = serialized.value("instance_id", std::string{});
            if (!instance.instanceId.empty())
                for (const auto& prior : candidate.instances)
                    if (prior.instanceId == instance.instanceId)
                        throw std::invalid_argument("Duplicate instance_id");
            const auto stateElements = size(serialized.at("state_size"));
            instance.descriptor = std::make_shared<PrimitiveT<Scalar>>(
                primitive<Scalar>(serialized.at("primitive").get<std::string>(), stateElements, extension));
            instance.primitive = instance.descriptor.get();
            instance.inputChannels =
                serialized.contains("input_channels") ? size(serialized.at("input_channels")) : 0;
            if (serialized.contains("audio_bindings"))
                for (const auto& binding : serialized.at("audio_bindings"))
                {
                    const auto weights = binding.at("weights").get<std::string>();
                    const auto missing = binding.at("missing_source").get<std::string>();
                    if ((weights != "routing" && weights != "modulation") ||
                        (missing != "zero" && missing != "reject"))
                        throw std::invalid_argument("Invalid audio binding policy");
                    instance.audioBindings.push_back(
                        {source(binding.at("source").get<std::string>()), size(binding.at("source_element")),
                         size(binding.at("destination_element")),
                         weights == "routing" ? WeightTensor::Routing : WeightTensor::Modulation,
                         size(binding.at("weight_element")),
                         missing == "zero" ? MissingSource::Zero : MissingSource::Reject});
                }
            instance.activationIndex = size(serialized.at("activation"));
            const auto& parameters = serialized.at("parameters");
            instance.parameters = {size(parameters.at("offset")), size(parameters.at("count"))};
            const auto& clock = serialized.at("clock");
            instance.clock = {size(clock.at("period_samples")), size(clock.at("phase_samples"))};
            const auto between = serialized.at("output_between_updates").get<std::string>();
            if (between != "hold" && between != "zero")
                throw std::invalid_argument("Unknown between-update output policy");
            instance.outputBetweenUpdates =
                between == "hold" ? OutputBetweenUpdates::Hold : OutputBetweenUpdates::Zero;
            if (serialized.contains("assets"))
                for (const auto& asset : serialized.at("assets"))
                    instance.assets.push_back(tensor<Scalar>(asset));
            if (instance.primitive->id == "time.partitioned_convolution.v1")
            {
                if (instance.parameters.count != 1 ||
                    instance.parameters.offset >= candidate.parameters.values.size() ||
                    instance.assets.size() != 1)
                    throw std::invalid_argument(
                        "Partitioned convolution requires block parameter and one IR asset");
                const auto blockValue = candidate.parameters.values[instance.parameters.offset];
                if (!std::isfinite(blockValue) || blockValue < 8 || blockValue > 4096 ||
                    std::floor(blockValue) != blockValue)
                    throw std::invalid_argument("Invalid partition size");
                const auto block = static_cast<std::size_t>(blockValue);
                if (instance.assets[0].axes != std::vector<std::string>{"partition", "bin", "complex"})
                {
                    const auto taps = instance.assets[0].values.size();
                    if (taps == 0)
                        throw std::invalid_argument("Empty convolution IR");
                    const auto partitions = (taps - 1) / block + 1;
                    if (document.contains("budget"))
                    {
                        const auto allowed = document.at("budget").at("stored_tensor_elements").get<size_t>();
                        if (partitions > allowed / (4 * block))
                            throw std::invalid_argument(
                                "Prepared IR exceeds stored tensor budget before allocation");
                    }
                    instance.assets[0] = advanced::partitionIR(instance.assets[0], block);
                }
                const auto& shape = instance.assets[0].shape;
                if (shape.size() != 3 || shape[1] != 2 * block || shape[2] != 2)
                    throw std::invalid_argument("Invalid prepared IR partition shape");
                auto descriptor = advanced::partitionedPrimitive<Scalar>(block, shape[0]);
                if (descriptor.stateElements != stateElements)
                    throw std::invalid_argument(
                        "Partitioned convolution state does not match IR/block layout");
                instance.descriptor = std::make_shared<PrimitiveT<Scalar>>(std::move(descriptor));
                instance.primitive = instance.descriptor.get();
            }
            candidate.initialState.push_back(serialized.at("initial_state").get<std::vector<Scalar>>());
            candidate.instances.push_back(std::move(instance));
        }
        if (document.contains("projections"))
            for (const auto& serialized : document.at("projections"))
            {
                Projection projection;
                projection.source = source(serialized.at("source").get<std::string>());
                projection.sourceElement = size(serialized.at("source_element"));
                projection.parameterElement = size(serialized.at("parameter_element"));
                projection.weightElement = size(serialized.at("weight_element"));
                const auto weights = serialized.at("weights").get<std::string>();
                if (weights != "routing" && weights != "modulation")
                    throw std::invalid_argument("Unknown weight tensor");
                projection.weightTensor =
                    weights == "routing" ? WeightTensor::Routing : WeightTensor::Modulation;
                const auto missing = serialized.at("missing_source").get<std::string>();
                if (missing != "reject" && missing != "zero")
                    throw std::invalid_argument("Unknown missing-source policy");
                projection.missingSource = missing == "reject" ? MissingSource::Reject : MissingSource::Zero;
                candidate.projections.push_back(projection);
            }
        if (document.contains("parameter_locks"))
            candidate.parameterLocks = mask(document.at("parameter_locks"));
        if (document.contains("shared_initial_state"))
            candidate.sharedInitialState = document.at("shared_initial_state").get<std::vector<Scalar>>();
        output = std::move(candidate);
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}

bool readBudgetJSON(std::string_view text, Budget& output, std::string& error)
{
    error.clear();
    try
    {
        const auto document = Json::parse(text);
        validateHeader(document);
        const auto& budget = document.at("budget");
        Budget candidate{size(budget.at("active_primitives")),      size(budget.at("state_elements")),
                         size(budget.at("parameter_elements")),     size(budget.at("routing_elements")),
                         size(budget.at("modulation_elements")),    size(budget.at("output_elements")),
                         size(budget.at("stored_tensor_elements")), size(budget.at("projection_terms"))};
        candidate.sparseStateWrites =
            budget.contains("sparse_state_writes") ? size(budget.at("sparse_state_writes")) : 0;
        candidate.scratchElements =
            budget.contains("scratch_elements") ? size(budget.at("scratch_elements")) : 0;
        output = candidate;
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}

template <typename Scalar>
std::string serializeJSON(const ConfigurationT<Scalar>& configuration)
{
    Json document{{"schema", "amo.tensor/1"},
                  {"ontology", "amo.math/1"},
                  {"tensors",
                   {{"activation", tensorJSON<Scalar>(configuration.activation)},
                    {"parameters", tensorJSON<Scalar>(configuration.parameters)},
                    {"routing", tensorJSON<Scalar>(configuration.routing)},
                    {"modulation", tensorJSON<Scalar>(configuration.modulation)}}},
                  {"parameter_locks", configuration.parameterLocks},
                  {"shared_initial_state", configuration.sharedInitialState}};
    document["instances"] = Json::array();
    for (std::size_t index = 0; index < configuration.instances.size(); ++index)
    {
        const auto& instance = configuration.instances[index];
        if (instance.primitive == nullptr || index >= configuration.initialState.size())
            throw std::invalid_argument("Cannot serialize an incomplete instance");
        Json entry{
            {"primitive", instance.primitive->id},
            {"activation", instance.activationIndex},
            {"parameters", {{"offset", instance.parameters.offset}, {"count", instance.parameters.count}}},
            {"clock",
             {{"period_samples", instance.clock.periodSamples},
              {"phase_samples", instance.clock.phaseSamples}}},
            {"output_between_updates",
             instance.outputBetweenUpdates == OutputBetweenUpdates::Hold ? "hold" : "zero"},
            {"state_size", instance.primitive->stateElements},
            {"initial_state", configuration.initialState[index]}};
        entry["assets"] = Json::array();
        if (!instance.instanceId.empty())
            entry["instance_id"] = instance.instanceId;
        entry["input_channels"] = instance.inputChannels;
        entry["audio_bindings"] = Json::array();
        for (const auto& binding : instance.audioBindings)
            entry["audio_bindings"].push_back(
                {{"source", sourceName(binding.source)},
                 {"source_element", binding.sourceElement},
                 {"destination_element", binding.destinationElement},
                 {"weights", binding.weightTensor == WeightTensor::Routing ? "routing" : "modulation"},
                 {"weight_element", binding.weightElement},
                 {"missing_source", binding.missingSource == MissingSource::Zero ? "zero" : "reject"}});
        for (const auto& asset : instance.assets)
            entry["assets"].push_back(tensorJSON<Scalar>(asset));
        document["instances"].push_back(std::move(entry));
    }
    document["projections"] = Json::array();
    for (const auto& projection : configuration.projections)
        document["projections"].push_back(
            {{"source", sourceName(projection.source)},
             {"source_element", projection.sourceElement},
             {"parameter_element", projection.parameterElement},
             {"weights", projection.weightTensor == WeightTensor::Routing ? "routing" : "modulation"},
             {"weight_element", projection.weightElement},
             {"missing_source", projection.missingSource == MissingSource::Reject ? "reject" : "zero"}});
    document["scalar_type"] = std::is_same_v<Scalar, double> ? "float64" : "float32";
    return document.dump(2);
}

template <typename Scalar>
bool compileJSON(std::string_view text, double sampleRate, RuntimeT<Scalar>& runtime, std::string& error)
{
    ConfigurationT<Scalar> configuration;
    Budget budget;
    return parseJSON(text, configuration, error) && readBudgetJSON(text, budget, error) &&
           runtime.prepare(configuration, budget, sampleRate, error);
}

template bool parseJSON<float>(std::string_view, ConfigurationT<float>&, std::string&,
                               const PrimitiveResolverT<float>&);
template std::string serializeJSON<float>(const ConfigurationT<float>&);
template bool compileJSON<float>(std::string_view, double, RuntimeT<float>&, std::string&);
template bool parseJSON<double>(std::string_view, ConfigurationT<double>&, std::string&,
                                const PrimitiveResolverT<double>&);
template std::string serializeJSON<double>(const ConfigurationT<double>&);
template bool compileJSON<double>(std::string_view, double, RuntimeT<double>&, std::string&);
} // namespace alfa::tensor
