#include "alfa/tensor/Compiler.h"

#include "alfa/tensor/Kernels.h"
#include "alfa/tensor/Universe.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
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

Tensor tensor(const Json& value)
{
    Tensor result;
    result.axes = value.at("axes").get<std::vector<std::string>>();
    for (const auto& extent : value.at("shape"))
        result.shape.push_back(size(extent));
    result.values = value.at("values").get<std::vector<float>>();
    if (value.contains("presence"))
        result.presence = mask(value.at("presence"));
    return result;
}

Json tensorJSON(const Tensor& value)
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
    }
    throw std::invalid_argument("Unknown source representation");
}

Primitive primitive(std::string_view id, std::size_t stateElements, const PrimitiveResolver& extension)
{
    Primitive result;
    if (id == "math.sine" && stateElements == 1)
        return kernels::sine();
    if (id == "math.one_pole_lowpass" && stateElements == 1)
        return kernels::onePole();
    if (id == "math.feedback_cell" && stateElements == 1)
        return kernels::feedbackCell();
    if (id == "math.integer_delay" && stateElements > 0)
        return kernels::delay(stateElements);
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
    if (document.at("schema") != "aom.tensor/1" || document.at("ontology") != "aom.math/1")
        throw std::invalid_argument("Unsupported tensor schema or ontology version");
}
} // namespace

bool parseJSON(std::string_view text, Configuration& output, std::string& error,
               const PrimitiveResolver& extension)
{
    error.clear();
    try
    {
        const auto document = Json::parse(text);
        validateHeader(document);
        Configuration candidate;
        const auto& tensors = document.at("tensors");
        candidate.activation = tensor(tensors.at("activation"));
        candidate.parameters = tensor(tensors.at("parameters"));
        candidate.routing = tensor(tensors.at("routing"));
        candidate.modulation = tensor(tensors.at("modulation"));
        for (const auto& serialized : document.at("instances"))
        {
            Instance instance;
            const auto stateElements = size(serialized.at("state_size"));
            instance.descriptor = std::make_shared<Primitive>(
                primitive(serialized.at("primitive").get<std::string>(), stateElements, extension));
            instance.primitive = instance.descriptor.get();
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
                    instance.assets.push_back(tensor(asset));
            candidate.initialState.push_back(serialized.at("initial_state").get<std::vector<float>>());
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
            candidate.sharedInitialState = document.at("shared_initial_state").get<std::vector<float>>();
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
        output = candidate;
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}

std::string serializeJSON(const Configuration& configuration)
{
    Json document{{"schema", "aom.tensor/1"},
                  {"ontology", "aom.math/1"},
                  {"tensors",
                   {{"activation", tensorJSON(configuration.activation)},
                    {"parameters", tensorJSON(configuration.parameters)},
                    {"routing", tensorJSON(configuration.routing)},
                    {"modulation", tensorJSON(configuration.modulation)}}},
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
        for (const auto& asset : instance.assets)
            entry["assets"].push_back(tensorJSON(asset));
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
    return document.dump(2);
}

bool compileJSON(std::string_view text, double sampleRate, Runtime& runtime, std::string& error)
{
    Configuration configuration;
    Budget budget;
    return parseJSON(text, configuration, error) && readBudgetJSON(text, budget, error) &&
           runtime.prepare(configuration, budget, sampleRate, error);
}
} // namespace alfa::tensor
