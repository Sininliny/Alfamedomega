#include "TensorComposition.h"

#include "TensorExecution.h"
#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Precision.h"

#include <juce_cryptography/juce_cryptography.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace alfa::tensorhost
{
namespace
{
using JSON = nlohmann::json;
constexpr std::array<const char*, 4> names{"activation", "parameters", "routing", "modulation"};
constexpr std::array<const char*, 6> properties{"tier_profiles",  "tier_candidates", "tier_conversion",
                                                "input_capacity", "tail_seconds",    "preferred_mode"};
std::size_t checkedAdd(std::size_t first, std::size_t second)
{
    if (second > std::numeric_limits<std::size_t>::max() - first)
        throw std::invalid_argument("Composition coordinate overflow");
    return first + second;
}
std::size_t checkedProduct(std::size_t first, std::size_t second)
{
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
        throw std::invalid_argument("Composition matrix size overflow");
    return first * second;
}
JSON tensor()
{
    return {{"axes", {"composition_coordinate"}},
            {"shape", {0}},
            {"values", JSON::array()},
            {"presence", JSON::array()}};
}
std::string fingerprint(JSON document)
{
    document.erase("composition");
    const auto serialized = document.dump();
    return juce::SHA256(serialized.data(), serialized.size()).toHexString().toStdString();
}
void append(JSON& destination, const JSON& source)
{
    for (const auto& value : source)
        destination.push_back(value);
}
struct Fragment
{
    std::string id;
    JSON document;
    std::array<std::size_t, 4> offsets{};
    std::size_t shared = 0, sharedOffset = 0, local = 0, localOffset = 0;
    std::size_t outputCount = 0, outputOffset = 0, instanceOffset = 0;
    std::vector<alfa::tensor::Slice> states, outputs;
};
template <typename Scalar>
void inspect(Fragment& fragment, double rate)
{
    alfa::tensor::ConfigurationT<Scalar> config;
    alfa::tensor::RuntimeT<Scalar> runtime;
    alfa::tensor::Budget budget;
    std::string error;
    const auto text = fragment.document.dump();
    if (!alfa::tensor::parseJSON(text, config, error) || !alfa::tensor::readBudgetJSON(text, budget, error) ||
        !runtime.prepare(config, budget, rate, error))
        throw std::invalid_argument(fragment.id + ": " + error);
    fragment.shared = config.sharedInitialState.size();
    fragment.local = runtime.state().size() - fragment.shared;
    fragment.outputCount = runtime.output().size();
    fragment.states.assign(runtime.stateAllocation().begin(), runtime.stateAllocation().end());
    fragment.outputs.assign(runtime.outputAllocation().begin(), runtime.outputAllocation().end());
    if (!prepareExecution(text, rate, error))
        throw std::invalid_argument(fragment.id + ": " + error);
}
std::size_t stateCoordinate(const Fragment& fragment, std::size_t index, std::size_t sharedTotal)
{
    if (index >= checkedAdd(fragment.shared, fragment.local))
        throw std::invalid_argument("State source lies outside its fragment");
    return index < fragment.shared ? fragment.sharedOffset + index
                                   : sharedTotal + fragment.localOffset + index - fragment.shared;
}
void remapSource(JSON& relation, const Fragment& fragment, std::size_t sharedTotal)
{
    const auto source = relation.at("source").get<std::string>();
    const auto index = relation.at("source_element").get<std::size_t>();
    if (source == "state")
        relation["source_element"] = stateCoordinate(fragment, index, sharedTotal);
    else if (source == "output_history")
    {
        if (index >= fragment.outputCount)
            throw std::invalid_argument("Output source lies outside its fragment");
        relation["source_element"] = fragment.outputOffset + index;
    }
    const auto weights = relation.at("weights").get<std::string>();
    const auto category = weights == "routing" ? 2 : weights == "modulation" ? 3 : -1;
    if (category < 0)
        throw std::invalid_argument("Invalid composition weight tensor");
    relation["weight_element"] = checkedAdd(fragment.offsets[static_cast<std::size_t>(category)],
                                            relation.at("weight_element").get<std::size_t>());
}
JSON lower(const JSON& recipe, double rate)
{
    if (recipe.at("schema") != "aom.composition/1" || !recipe.at("fragments").is_array() ||
        !recipe.at("recurrent_bindings").is_array() || !recipe.at("budget").is_object())
        throw std::invalid_argument("Unsupported composition recipe");
    std::vector<Fragment> fragments;
    alfa::tensor::PrecisionProfile target;
    std::string error;
    const auto targetTier = recipe.value("target_tier", std::string{});
    bool precisionSet = !targetTier.empty();
    if (precisionSet && !alfa::tensor::resolvePrecisionJSON(JSON{{"tier", targetTier}}.dump(), target, error))
        throw std::invalid_argument(error);
    std::array<std::size_t, 4> tensorCounts{};
    std::size_t sharedTotal = 0, localTotal = 0, outputTotal = 0, instanceTotal = 0, depth = 0;
    JSON result{{"schema", "aom.tensor/1"},         {"ontology", "aom.math/1"},
                {"budget", recipe.at("budget")},    {"tensors", JSON::object()},
                {"instances", JSON::array()},       {"projections", JSON::array()},
                {"parameter_locks", JSON::array()}, {"shared_initial_state", JSON::array()},
                {"outputMix", JSON::array()}};
    for (const auto* name : names)
        result["tensors"][name] = tensor();
    if (recipe.contains("host_latency_samples"))
        result["host_latency_samples"] = recipe.at("host_latency_samples");
    for (const auto& entry : recipe.at("fragments"))
    {
        Fragment fragment;
        fragment.id = entry.at("id").get<std::string>();
        if (fragment.id.empty() || std::any_of(fragments.begin(), fragments.end(), [&](const Fragment& other)
                                               { return other.id == fragment.id; }))
            throw std::invalid_argument("Fragment identities must be unique and nonempty");
        fragment.document = entry.at("document");
        alfa::tensor::PrecisionProfile precision;
        if (!alfa::tensor::resolvePrecisionJSON(fragment.document.dump(), precision, error))
            throw std::invalid_argument(error);
        if (!precisionSet)
        {
            target = precision;
            precisionSet = true;
        }
        else if (targetTier.empty() && precision.scalarType != target.scalarType)
            throw std::invalid_argument("Mixed fragment precision requires an explicit target tier");
        if (precision.scalarType == alfa::tensor::ScalarType::Float64)
            inspect<double>(fragment, rate);
        else
            inspect<float>(fragment, rate);
        fragment.offsets = tensorCounts;
        for (std::size_t category = 0; category < names.size(); ++category)
            tensorCounts[category] =
                checkedAdd(tensorCounts[category],
                           fragment.document.at("tensors").at(names[category]).at("values").size());
        fragment.sharedOffset = sharedTotal;
        fragment.localOffset = localTotal;
        fragment.outputOffset = outputTotal;
        fragment.instanceOffset = instanceTotal;
        sharedTotal = checkedAdd(sharedTotal, fragment.shared);
        localTotal = checkedAdd(localTotal, fragment.local);
        outputTotal = checkedAdd(outputTotal, fragment.outputCount);
        instanceTotal = checkedAdd(instanceTotal, fragment.document.at("instances").size());
        std::size_t tensorTotal = 0;
        for (const auto count : tensorCounts)
            tensorTotal = checkedAdd(tensorTotal, count);
        if (tensorTotal > recipe.at("budget").at("stored_tensor_elements").get<std::size_t>() ||
            checkedAdd(sharedTotal, localTotal) >
                recipe.at("budget").at("state_elements").get<std::size_t>() ||
            outputTotal > recipe.at("budget").at("output_elements").get<std::size_t>())
            throw std::invalid_argument(
                "Composition exceeds explicit storage/state/output capacity before merging");
        depth = std::max(depth, fragment.document.value("control_layers", JSON::array()).size());
        for (const auto* policy : {"performance", "latentSound"})
        {
            if (recipe.contains(policy))
            {
                result[policy] = recipe.at(policy);
                continue;
            }
            if (!fragments.empty())
            {
                const auto& first = fragments.front().document;
                if (first.contains(policy) != fragment.document.contains(policy) ||
                    (first.contains(policy) && first.at(policy) != fragment.document.at(policy)))
                    throw std::invalid_argument(std::string("Different ") + policy +
                                                " policies require explicit reconciliation");
            }
            if (fragment.document.contains(policy))
            {
                if (result.contains(policy) && result.at(policy) != fragment.document.at(policy))
                    throw std::invalid_argument(std::string("Different ") + policy +
                                                " policies require explicit reconciliation");
                result[policy] = fragment.document.at(policy);
            }
        }
        if (fragment.document.contains("input_capacity"))
        {
            const auto& capacity = fragment.document.at("input_capacity");
            if (!capacity.is_array() || capacity.size() != 6)
                throw std::invalid_argument("Fragment input_capacity must contain six widths");
            if (!result.contains("input_capacity"))
                result["input_capacity"] = {0, 0, 0, 0, 0, 0};
            for (std::size_t category = 0; category < 6; ++category)
                result["input_capacity"][category] =
                    std::max(result["input_capacity"][category].get<std::size_t>(),
                             capacity[category].get<std::size_t>());
        }
        fragments.push_back(std::move(fragment));
    }
    result["scalar_type"] = alfa::tensor::scalarTypeName(target.scalarType);
    if (!targetTier.empty())
        result["tier"] = targetTier;
    else if (!target.tier.empty() &&
             std::all_of(fragments.begin(), fragments.end(), [&](const Fragment& fragment)
                         { return fragment.document.value("tier", std::string{}) == target.tier; }))
        result["tier"] = target.tier;
    JSON coordinates = JSON::array();
    for (const auto& fragment : fragments)
    {
        if (fragment.document.contains("runtimeModel"))
        {
            auto controls = fragment.document;
            controls.erase("runtimeModel");
            controls.erase("composition");
            result["runtimeModels"].push_back({{"id", fragment.id},
                                               {"model", fragment.document.at("runtimeModel")},
                                               {"controls", controls},
                                               {"offsets", fragment.offsets}});
        }
        for (auto unit : fragment.document.value("runtimeModels", JSON::array()))
        {
            unit["id"] = fragment.id + "/" + unit.at("id").get<std::string>();
            for (size_t category = 0; category < 4; ++category)
                unit["offsets"][category] =
                    checkedAdd(fragment.offsets[category], unit["offsets"][category].get<size_t>());
            result["runtimeModels"].push_back(std::move(unit));
        }
        JSON offsets = JSON::object();
        for (std::size_t category = 0; category < names.size(); ++category)
        {
            offsets[names[category]] = fragment.offsets[category];
            auto& destination = result["tensors"][names[category]];
            const auto& source = fragment.document.at("tensors").at(names[category]);
            append(destination["values"], source.at("values"));
            if (source.contains("presence") && !source.at("presence").empty())
                append(destination["presence"], source.at("presence"));
            else
                for (std::size_t element = 0; element < source.at("values").size(); ++element)
                    destination["presence"].push_back(1);
        }
        const auto locks = fragment.document.value("parameter_locks", JSON::array());
        if (!locks.empty())
            append(result["parameter_locks"], locks);
        else
            for (std::size_t parameter = 0;
                 parameter < fragment.document.at("tensors").at("parameters").at("values").size();
                 ++parameter)
                result["parameter_locks"].push_back(0);
        append(result["shared_initial_state"],
               fragment.document.value("shared_initial_state", JSON::array()));
        std::size_t localInstance = 0;
        for (auto instance : fragment.document.at("instances"))
        {
            instance["instance_id"] =
                fragment.id + "/" + instance.value("instance_id", std::to_string(localInstance++));
            instance["fragment_id"] = fragment.id;
            instance["activation"] =
                checkedAdd(fragment.offsets[0], instance.at("activation").get<std::size_t>());
            instance["parameters"]["offset"] =
                checkedAdd(fragment.offsets[1], instance.at("parameters").at("offset").get<std::size_t>());
            if (instance.contains("audio_bindings"))
                for (auto& binding : instance["audio_bindings"])
                    remapSource(binding, fragment, sharedTotal);
            result["instances"].push_back(std::move(instance));
        }
        for (auto projection : fragment.document.value("projections", JSON::array()))
        {
            projection["parameter_element"] =
                checkedAdd(fragment.offsets[1], projection.at("parameter_element").get<std::size_t>());
            remapSource(projection, fragment, sharedTotal);
            result["projections"].push_back(std::move(projection));
        }
        for (auto mix : fragment.document.at("outputMix"))
        {
            mix[0] = checkedAdd(fragment.outputOffset, mix.at(0).get<std::size_t>());
            result["outputMix"].push_back(std::move(mix));
        }
        coordinates.push_back({{"id", fragment.id},
                               {"tensor_offsets", offsets},
                               {"shared_state_offset", fragment.sharedOffset},
                               {"local_state_offset", sharedTotal + fragment.localOffset},
                               {"output_offset", fragment.outputOffset},
                               {"instance_offset", fragment.instanceOffset}});
    }
    const auto find = [&](const std::string& id) -> const Fragment&
    {
        const auto found = std::find_if(fragments.begin(), fragments.end(),
                                        [&](const Fragment& fragment) { return fragment.id == id; });
        if (found == fragments.end())
            throw std::invalid_argument("Recurrent binding references an absent fragment");
        return *found;
    };
    std::vector<std::string> bindingIds;
    for (const auto& binding : recipe.at("recurrent_bindings"))
    {
        const auto id = binding.at("id").get<std::string>();
        if (id.empty() || std::find(bindingIds.begin(), bindingIds.end(), id) != bindingIds.end())
            throw std::invalid_argument("Binding identities must be unique and nonempty");
        bindingIds.push_back(id);
        const auto& source = find(binding.at("source_fragment").get<std::string>());
        const auto& destination = find(binding.at("destination_fragment").get<std::string>());
        const auto kind = binding.at("source").get<std::string>();
        const auto localSource = binding.at("source_element").get<std::size_t>();
        std::size_t sourceIndex = 0;
        if (kind == "state")
            sourceIndex = stateCoordinate(source, localSource, sharedTotal);
        else if (kind == "output_history" && localSource < source.outputCount)
            sourceIndex = source.outputOffset + localSource;
        else
            throw std::invalid_argument(
                "Inter-fragment links require valid committed state or output_history");
        const auto weights = binding.at("weights").get<std::string>();
        if (weights != "routing" && weights != "modulation")
            throw std::invalid_argument("Invalid recurrent weight tensor");
        const auto weight = binding.at("weight").get<double>();
        if (!std::isfinite(weight))
            throw std::invalid_argument("Recurrent weights must be finite");
        auto& tensorValue = result["tensors"][weights];
        JSON lowered{{"source", kind},
                     {"source_element", sourceIndex},
                     {"weights", weights},
                     {"weight_element", tensorValue["values"].size()},
                     {"missing_source", "reject"}};
        tensorValue["values"].push_back(weight);
        tensorValue["presence"].push_back(1);
        const auto element = binding.at("destination_element").get<std::size_t>();
        if (binding.at("destination") == "parameter")
        {
            if (element >= destination.document.at("tensors").at("parameters").at("values").size())
                throw std::invalid_argument("Recurrent parameter destination is out of range");
            lowered["parameter_element"] = destination.offsets[1] + element;
            result["projections"].push_back(std::move(lowered));
        }
        else if (binding.at("destination") == "audio")
        {
            const auto index = binding.at("destination_instance").get<std::size_t>();
            if (index >= destination.document.at("instances").size() ||
                element >= destination.document.at("instances")[index].value("input_channels", std::size_t{}))
                throw std::invalid_argument("Recurrent audio destination is out of range");
            lowered["destination_element"] = element;
            auto& instance = result["instances"][destination.instanceOffset + index];
            if (!instance.contains("audio_bindings"))
                instance["audio_bindings"] = JSON::array();
            instance["audio_bindings"].push_back(std::move(lowered));
        }
        else
            throw std::invalid_argument("Unsupported recurrent destination");
    }
    for (const auto* name : names)
        result["tensors"][name]["shape"] = {result["tensors"][name]["values"].size()};
    // Independent stacks are right-aligned. Empty leading layers introduce no
    // new mappings or values; every authored matrix occupies its own block.
    if (depth != 0)
    {
        result["control_layers"] = JSON::array();
        for (std::size_t level = 0; level < depth; ++level)
        {
            JSON values = JSON::array(), ranges = JSON::array(), labels = JSON::array();
            bool named = false;
            std::vector<std::size_t> widths, destinations;
            std::vector<JSON> matrices;
            std::size_t totalSources = 0, totalDestinations = 0;
            for (const auto& fragment : fragments)
            {
                const auto layers = fragment.document.value("control_layers", JSON::array());
                const auto start = depth - layers.size();
                const bool active = level >= start && !layers.empty();
                const auto local = active ? level - start : std::size_t{};
                const auto width = active ? layers[local].at("values").size() : 0;
                std::size_t destination =
                    fragment.document.at("tensors").at("parameters").at("values").size();
                if (level + 1 < depth)
                    destination = level + 1 >= start && !layers.empty()
                                      ? layers[level + 1 - start].at("values").size()
                                      : 0;
                widths.push_back(width);
                destinations.push_back(destination);
                totalSources = checkedAdd(totalSources, width);
                totalDestinations = checkedAdd(totalDestinations, destination);
                matrices.push_back(active ? layers[local].at("matrix").at("values") : JSON::array());
                if (active)
                {
                    append(values, layers[local].at("values"));
                    append(ranges, layers[local].at("ranges"));
                    // User control names follow their controls; a learned ranking does not, because it
                    // was computed for the fragment alone.
                    if (layers[local].contains("labels"))
                    {
                        append(labels, layers[local].at("labels"));
                        named = true;
                    }
                    else
                        for (std::size_t i = 0; i < width; ++i)
                            labels.push_back("");
                }
            }
            const auto count = checkedProduct(totalSources, totalDestinations);
            if (count > recipe.at("budget").at("stored_tensor_elements").get<std::size_t>())
                throw std::invalid_argument("Merged control matrices exceed supplied storage capacity");
            std::vector<double> matrix(count, 0.);
            std::size_t sourceOffset = 0, destinationOffset = 0;
            for (std::size_t fragment = 0; fragment < fragments.size(); ++fragment)
            {
                for (std::size_t d = 0; d < destinations[fragment]; ++d)
                    for (std::size_t s = 0; s < widths[fragment]; ++s)
                        matrix[(destinationOffset + d) * totalSources + sourceOffset + s] =
                            matrices[fragment].at(d * widths[fragment] + s).get<double>();
                sourceOffset += widths[fragment];
                destinationOffset += destinations[fragment];
            }
            result["control_layers"].push_back({{"values", values},
                                                {"ranges", ranges},
                                                {"matrix",
                                                 {{"axes", {"destination", "source"}},
                                                  {"shape", {totalDestinations, totalSources}},
                                                  {"values", matrix}}}});
            if (named)
                result["control_layers"][level]["labels"] = std::move(labels);
            for (const auto& binding : recipe.value("control_bindings", JSON::array()))
            {
                if (binding.at("layer_from_bottom").get<size_t>() != depth - 1 - level)
                    continue;
                const auto& source = find(binding.at("source_fragment").get<std::string>());
                const auto& destination = find(binding.at("destination_fragment").get<std::string>());
                size_t sourceIndex = 0, destinationIndex = 0;
                for (size_t i = 0; i < fragments.size(); ++i)
                {
                    if (fragments[i].id == source.id)
                        break;
                    sourceIndex += widths[i];
                }
                for (size_t i = 0; i < fragments.size(); ++i)
                {
                    if (fragments[i].id == destination.id)
                        break;
                    destinationIndex += destinations[i];
                }
                const auto sourceCoordinate = binding.at("source_element").get<size_t>();
                const auto destinationCoordinate = binding.at("destination_element").get<size_t>();
                const auto sourceFragment = static_cast<size_t>(&source - fragments.data());
                const auto destinationFragment = static_cast<size_t>(&destination - fragments.data());
                if (sourceCoordinate >= widths[sourceFragment] ||
                    destinationCoordinate >= destinations[destinationFragment])
                    throw std::invalid_argument(
                        "Control relationship lies outside fragment layer coordinates");
                const auto weight = binding.at("weight").get<double>();
                if (!std::isfinite(weight))
                    throw std::invalid_argument("Control relationship coefficient must be finite");
                result["control_layers"][level]["matrix"]["values"]
                      [(destinationIndex + destinationCoordinate) * totalSources + sourceIndex +
                       sourceCoordinate] = weight;
            }
        }
    }
    for (const auto& binding : recipe.value("control_bindings", JSON::array()))
        if (binding.at("layer_from_bottom").get<size_t>() >= depth)
            throw std::invalid_argument("Control relationship references absent layer");
    for (const auto* property : properties)
    {
        if (recipe.contains("properties") && recipe.at("properties").contains(property))
            result[property] = recipe.at("properties").at(property);
        else if (!fragments.empty() && fragments.front().document.contains(property))
            result[property] = fragments.front().document.at(property);
    }
    const auto digest = fingerprint(result);
    result["composition"] = recipe;
    result["composition"]["coordinates"] = std::move(coordinates);
    result["composition"]["flattened_fingerprint"] = digest;
    if (!prepareExecution(result.dump(), rate, error))
        throw std::invalid_argument("Composed configuration: " + error);
    return result;
}
JSON bindingJSON(const TensorComposition::Binding& binding)
{
    return {{"id", binding.id.toStdString()},
            {"source_fragment", binding.sourceFragment.toStdString()},
            {"destination_fragment", binding.destinationFragment.toStdString()},
            {"source", binding.source.toStdString()},
            {"source_element", binding.sourceElement},
            {"destination", binding.destination.toStdString()},
            {"destination_instance", binding.destinationInstance},
            {"destination_element", binding.destinationElement},
            {"weights", binding.weights.toStdString()},
            {"weight", binding.weight}};
}
} // namespace

TensorComposition::TensorComposition(double sampleRate, juce::String budgetJSON, juce::String targetTier)
    : sampleRate_(sampleRate)
{
    recipe_ = JSON{{"schema", "aom.composition/1"},
                   {"budget", JSON::parse(budgetJSON.toStdString())},
                   {"target_tier", targetTier.toStdString()},
                   {"fragments", JSON::array()},
                   {"recurrent_bindings", JSON::array()}}
                  .dump();
}
bool TensorComposition::commit(const juce::String& candidate, juce::String& error)
{
    try
    {
        const auto recipe = JSON::parse(candidate.toStdString());
        const auto lowered = lower(recipe, sampleRate_).dump(2);
        recipe_ = recipe.dump(2);
        document_ = lowered;
        error.clear();
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
bool TensorComposition::add(const juce::String& id, const juce::String& document, juce::String& error)
{
    try
    {
        auto candidate = JSON::parse(recipe_.toStdString());
        const auto parsed = JSON::parse(document.toStdString());
        if (candidate.at("fragments").empty() && parsed.contains("host_latency_samples"))
            candidate["host_latency_samples"] = parsed.at("host_latency_samples");
        candidate["fragments"].push_back({{"id", id.toStdString()}, {"document", parsed}});
        return commit(candidate.dump(), error);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
bool TensorComposition::clone(const juce::String& existing, const juce::String& id, juce::String& error)
{
    const auto current = JSON::parse(recipe_.toStdString());
    for (const auto& fragment : current.at("fragments"))
        if (fragment.at("id") == existing.toStdString())
            return add(id, fragment.at("document").dump(), error);
    error = "Fragment to clone does not exist";
    return false;
}
bool TensorComposition::remove(const juce::String& id, juce::String& error)
{
    auto candidate = JSON::parse(recipe_.toStdString());
    for (const auto& binding : candidate.at("recurrent_bindings"))
        if (binding.at("source_fragment") == id.toStdString() ||
            binding.at("destination_fragment") == id.toStdString())
        {
            error = "Remove external recurrent bindings explicitly before removing their fragment";
            return false;
        }
    auto& fragments = candidate["fragments"];
    const auto size = fragments.size();
    fragments.erase(std::remove_if(fragments.begin(), fragments.end(),
                                   [&](const JSON& entry) { return entry.at("id") == id.toStdString(); }),
                    fragments.end());
    if (fragments.size() == size)
    {
        error = "Fragment to remove does not exist";
        return false;
    }
    return commit(candidate.dump(), error);
}
bool TensorComposition::bind(const Binding& binding, juce::String& error)
{
    auto candidate = JSON::parse(recipe_.toStdString());
    candidate["recurrent_bindings"].push_back(bindingJSON(binding));
    return commit(candidate.dump(), error);
}
bool TensorComposition::unbind(const juce::String& id, juce::String& error)
{
    auto candidate = JSON::parse(recipe_.toStdString());
    auto& bindings = candidate["recurrent_bindings"];
    const auto size = bindings.size();
    bindings.erase(std::remove_if(bindings.begin(), bindings.end(),
                                  [&](const JSON& entry) { return entry.at("id") == id.toStdString(); }),
                   bindings.end());
    if (bindings.size() == size)
    {
        error = "Binding to remove does not exist";
        return false;
    }
    return commit(candidate.dump(), error);
}
bool TensorComposition::restore(const juce::String& recipe, juce::String& error)
{
    return commit(recipe, error);
}
bool TensorComposition::restoreDocument(const juce::String& text, juce::String& error)
{
    try
    {
        const auto document = JSON::parse(text.toStdString());
        const auto& recipe = document.at("composition");
        auto candidate = recipe;
        candidate.erase("coordinates");
        candidate.erase("flattened_fingerprint");
        candidate["target_tier"] = document.value("tier", std::string{});
        candidate["budget"] = document.at("budget");
        for (const auto* property : properties)
            if (document.contains(property))
                candidate["properties"][property] = document.at(property);
        if (recipe.at("flattened_fingerprint") != fingerprint(document))
        {
            const auto expected = lower(candidate, sampleRate_);
            const auto structure = [](JSON value)
            {
                value.erase("composition");
                value.erase("parameter_locks");
                for (auto& tensor : value["tensors"])
                    tensor.erase("values");
                for (auto& layer : value["control_layers"])
                {
                    layer.erase("values");
                    layer.erase("labels");
                    layer.erase("ranking");
                    layer["matrix"].erase("values");
                }
                if (value["control_layers"].is_null())
                    value.erase("control_layers");
                if (value.contains("runtimeModels"))
                    for (auto& unit : value["runtimeModels"])
                    {
                        for (auto& tensor : unit["controls"]["tensors"])
                            tensor.erase("values");
                        unit["controls"].erase("parameter_locks");
                        unit["controls"].erase("control_layers");
                    }
                return value;
            };
            if (structure(expected) != structure(document))
                throw std::invalid_argument("Composition recipe is stale after structural edits; use the "
                                            "current sound as a new fragment explicitly");
            std::array<size_t, 4> offset{};
            const auto depth = expected.value("control_layers", JSON::array()).size();
            std::vector<size_t> layerOffsets(depth);
            for (auto& entry : candidate["fragments"])
            {
                auto& fragment = entry["document"];
                for (size_t category = 0; category < 4; ++category)
                {
                    auto& values = fragment["tensors"][names[category]]["values"];
                    for (size_t i = 0; i < values.size(); ++i)
                        values[i] =
                            document.at("tensors").at(names[category]).at("values").at(offset[category] + i);
                    offset[category] += values.size();
                }
                const auto parameterCount = fragment["tensors"]["parameters"]["values"].size();
                fragment["parameter_locks"] = JSON::array();
                for (size_t i = 0; i < parameterCount; ++i)
                    fragment["parameter_locks"].push_back(
                        document.value("parameter_locks", JSON::array()).empty()
                            ? JSON(0)
                            : document["parameter_locks"][offset[1] - parameterCount + i]);
                auto& layers = fragment["control_layers"];
                if (layers.is_null())
                {
                    fragment.erase("control_layers");
                    continue;
                }
                for (size_t local = 0; local < layers.size(); ++local)
                {
                    const auto level = depth - layers.size() + local;
                    auto& values = layers[local]["values"];
                    for (size_t i = 0; i < values.size(); ++i)
                        values[i] = document["control_layers"][level]["values"][layerOffsets[level] + i];
                    layers[local].erase("labels");
                    layers[local].erase("ranking");
                    if (const auto& merged = document["control_layers"][level]; merged.contains("labels"))
                    {
                        JSON labels = JSON::array();
                        for (size_t i = 0; i < values.size(); ++i)
                            labels.push_back(merged.at("labels").at(layerOffsets[level] + i));
                        if (std::any_of(labels.begin(), labels.end(),
                                        [](const JSON& name) { return !name.get<std::string>().empty(); }))
                            layers[local]["labels"] = std::move(labels);
                    }
                    layerOffsets[level] += values.size();
                }
            }
            for (size_t level = 0; level < depth; ++level)
            {
                struct Coordinates
                {
                    std::string id;
                    size_t source = 0, destination = 0, sourceOffset = 0, destinationOffset = 0;
                };
                std::vector<Coordinates> coordinates;
                size_t sourceOffset = 0, destinationOffset = 0;
                for (const auto& entry : candidate["fragments"])
                {
                    const auto& fragment = entry.at("document");
                    const auto layers = fragment.value("control_layers", JSON::array());
                    const auto start = depth - layers.size();
                    const auto sources =
                        level >= start && !layers.empty() ? layers[level - start].at("values").size() : 0;
                    const auto destinations =
                        level + 1 == depth ? fragment.at("tensors").at("parameters").at("values").size()
                                           : (level + 1 >= start && !layers.empty()
                                                  ? layers[level + 1 - start].at("values").size()
                                                  : 0);
                    coordinates.push_back({entry.at("id").get<std::string>(), sources, destinations,
                                           sourceOffset, destinationOffset});
                    sourceOffset += sources;
                    destinationOffset += destinations;
                }
                for (const auto& source : coordinates)
                    for (const auto& destination : coordinates)
                        for (size_t d = 0; d < destination.destination; ++d)
                            for (size_t s = 0; s < source.source; ++s)
                            {
                                const auto index = (destination.destinationOffset + d) * sourceOffset +
                                                   source.sourceOffset + s;
                                const auto value =
                                    document["control_layers"][level]["matrix"]["values"][index];
                                if (value == expected["control_layers"][level]["matrix"]["values"][index])
                                    continue;
                                JSON binding{{"layer_from_bottom", depth - 1 - level},
                                             {"source_fragment", source.id},
                                             {"destination_fragment", destination.id},
                                             {"source_element", s},
                                             {"destination_element", d},
                                             {"weight", value}};
                                bool replaced = false;
                                for (auto& prior : candidate["control_bindings"])
                                    if (prior.at("layer_from_bottom") == binding["layer_from_bottom"] &&
                                        prior.at("source_fragment") == source.id &&
                                        prior.at("destination_fragment") == destination.id &&
                                        prior.at("source_element") == s &&
                                        prior.at("destination_element") == d)
                                    {
                                        prior = std::move(binding);
                                        replaced = true;
                                        break;
                                    }
                                if (!replaced)
                                    candidate["control_bindings"].push_back(std::move(binding));
                            }
            }
            for (auto& binding : candidate["recurrent_bindings"])
            {
                const auto category = binding.at("weights") == "routing" ? size_t(2) : size_t(3);
                binding["weight"] = document["tensors"][names[category]]["values"][offset[category]++];
            }
            const auto reconciled = lower(candidate, sampleRate_);
            auto normalized = document;
            for (auto& layer : normalized["control_layers"])
                layer.erase("ranking");
            if (normalized["control_layers"].is_null())
                normalized.erase("control_layers");
            if (reconciled.contains("runtimeModels"))
                normalized["runtimeModels"] = reconciled["runtimeModels"];
            if (fingerprint(reconciled) != fingerprint(normalized))
                throw std::invalid_argument("Composition edits cannot be reconciled without changing sound; "
                                            "use current sound as base");
        }
        return commit(candidate.dump(), error);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
juce::StringArray TensorComposition::fragmentIds() const
{
    juce::StringArray result;
    const auto current = JSON::parse(recipe_.toStdString());
    for (const auto& fragment : current.at("fragments"))
        result.add(fragment.at("id").get<std::string>());
    return result;
}
} // namespace alfa::tensorhost
