#include "TierConversion.h"

#include "TensorComposition.h"
#include "TensorExecution.h"
#include "alfa/tensor/Precision.h"
#include "alfa/tensor/Universe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace alfa::tensorhost
{
namespace
{
using JSON = nlohmann::json;
constexpr std::array<const char*, 4> tensors{"activation", "parameters", "routing", "modulation"};
constexpr std::array<const char*, 6> categories{"audio",           "midi",         "gesture",
                                                "user_preference", "latent_sound", "context"};
JSON mergeEquivalent(JSON document)
{
    if (document.contains("runtimeModel") || document.contains("runtimeModels") ||
        document.contains("composition") || !document.value("projections", JSON::array()).empty() ||
        !document.value("control_layers", JSON::array()).empty())
        return JSON();
    for (const auto& lock : document.value("parameter_locks", JSON::array()))
        if (lock.get<int>() != 0)
            return JSON();
    auto& instances = document["instances"];
    const auto active = [&](size_t i)
    {
        return document["tensors"]["activation"]["values"][instances[i]["activation"].get<size_t>()]
                   .get<double>() != 0;
    };
    for (size_t first = 0; first < instances.size(); ++first)
        for (size_t second = first + 1; second < instances.size(); ++second)
        {
            if (!active(first) || !active(second) ||
                instances[first]["activation"] == instances[second]["activation"])
                continue;
            bool sharedActivation = false;
            for (size_t i = 0; i < instances.size(); ++i)
                if (i != second && instances[i]["activation"] == instances[second]["activation"])
                    sharedActivation = true;
            if (sharedActivation)
                continue;
            const auto id = instances[first].at("primitive").get<std::string>();
            if (id != "math.sine" && id != "osc.polyblep_saw" && id != "osc.wavetable" &&
                id != "physical.modal_rotation.v1" && id != "filter.biquad_lowpass" &&
                id != "filter.biquad_highpass" && id != "filter.biquad_bandpass")
                continue;
            auto a = instances[first], b = instances[second];
            if (!a.value("audio_bindings", JSON::array()).empty() ||
                !b.value("audio_bindings", JSON::array()).empty())
                continue;
            const auto values = [&](const JSON& instance)
            {
                JSON slice = JSON::array();
                const auto start = instance.at("parameters").at("offset").get<size_t>();
                for (size_t i = 0; i < instance.at("parameters").at("count").get<size_t>(); ++i)
                    slice.push_back(document["tensors"]["parameters"]["values"].at(start + i));
                return slice;
            };
            if (values(a) != values(b))
                continue;
            if (document["tensors"]["activation"]["values"][a["activation"].get<size_t>()] !=
                document["tensors"]["activation"]["values"][b["activation"].get<size_t>()])
                continue;
            for (auto* instance : {&a, &b})
            {
                instance->erase("instance_id");
                instance->erase("fragment_id");
                instance->erase("activation");
                (*instance)["parameters"]["offset"] = 0;
            }
            if (a != b)
                continue;
            std::vector<size_t> offsets(instances.size()), counts(instances.size());
            size_t total = 0;
            for (size_t i = 0; i < instances.size(); ++i)
            {
                offsets[i] = total;
                if (!active(i))
                    continue;
                alfa::tensor::Primitive primitive;
                std::string error;
                const auto operation = instances[i].at("primitive").get<std::string>();
                if (operation == "math.sine")
                    counts[i] = 1;
                else if (!alfa::tensor::universe::createPrimitive(
                             operation, instances[i].at("state_size").get<size_t>(), primitive, error))
                    return JSON();
                else
                    counts[i] = primitive.outputElements;
                total += counts[i];
            }
            if (counts[first] != counts[second])
                continue;
            for (auto& mix : document["outputMix"])
            {
                const auto index = mix.at(0).get<size_t>();
                size_t owner = instances.size();
                for (size_t i = 0; i < instances.size(); ++i)
                    if (index >= offsets[i] && index - offsets[i] < counts[i])
                    {
                        owner = i;
                        break;
                    }
                if (owner == instances.size())
                    return JSON();
                if (owner == second)
                    mix[0] = offsets[first] + index - offsets[second];
                else
                    mix[0] = index - (owner > second ? counts[second] : 0);
            }
            document["tensors"]["activation"]["values"][instances[second]["activation"].get<size_t>()] = 0;
            document["tier_strategy"] = "merge_equivalent_recurrence";
            return document;
        }
    return JSON();
}
void protect(const JSON& original, const JSON& proposal)
{
    if (original.at("ontology") != proposal.at("ontology"))
        throw std::invalid_argument("Tier conversion must retain ontology");
    for (const auto* name : tensors)
    {
        const auto& a = original.at("tensors").at(name);
        const auto& b = proposal.at("tensors").at(name);
        if (a.at("axes") != b.at("axes") || a.at("shape") != b.at("shape"))
            throw std::invalid_argument("Candidates must retain padded coordinates");
    }
    if (original.value("host_latency_samples", 0) != proposal.value("host_latency_samples", 0) ||
        original.value("performance", JSON()) != proposal.value("performance", JSON()))
        throw std::invalid_argument("Candidates must retain host timing and performance policy");
    const auto locks = original.value("parameter_locks", JSON::array());
    for (size_t i = 0; i < locks.size(); ++i)
        if (locks[i].get<int>() != 0)
        {
            if (original.at("tensors").at("parameters").at("values").at(i) !=
                proposal.at("tensors").at("parameters").at("values").at(i))
                throw std::invalid_argument("Tier candidate changes a locked value");
            const auto& instances = original.at("instances");
            for (size_t instance = 0; instance < instances.size(); ++instance)
            {
                const auto& a = instances[instance];
                const auto start = a.at("parameters").at("offset").get<size_t>();
                if (i >= start && i - start < a.at("parameters").at("count").get<size_t>())
                {
                    if (instance >= proposal.at("instances").size() ||
                        a.at("primitive") != proposal["instances"][instance]["primitive"] ||
                        a.at("parameters") != proposal["instances"][instance]["parameters"] ||
                        a.value("instance_id", std::string{}) !=
                            proposal["instances"][instance].value("instance_id", std::string{}))
                        throw std::invalid_argument("Tier candidate remaps a locked primitive");
                }
            }
        }
}
std::vector<double> render(const JSON& document, const JSON& fixture, double rate)
{
    std::string error;
    auto execution = prepareExecution(document.dump(), rate, error);
    if (!execution)
        throw std::invalid_argument(error);
    const auto defaults = execution->highestDefaults();
    if (!execution->updateControls(defaults))
        throw std::invalid_argument("Tier validation control publication failed");
    const auto frames = fixture.at("frames").get<size_t>();
    std::array<std::vector<float>, 6> values;
    const auto inputs = fixture.value("inputs", JSON::object());
    for (size_t i = 0; i < 6; ++i)
        if (inputs.contains(categories[i]))
            values[i] = inputs.at(categories[i]).get<std::vector<float>>();
    const std::array<std::string, 1> axes{"feature"};
    std::array<std::array<size_t, 1>, 6> shapes;
    for (size_t i = 0; i < 6; ++i)
        shapes[i] = {values[i].size()};
    std::vector<double> audio(2 * frames);
    for (size_t sample = 0; sample < frames; ++sample)
    {
        if (fixture.contains("audio_frames"))
        {
            const auto& framesData = fixture.at("audio_frames");
            if (sample < framesData.size())
                values[0] = framesData.at(sample).get<std::vector<float>>();
            else
                std::fill(values[0].begin(), values[0].end(), 0.f);
            shapes[0] = {values[0].size()};
        }
        alfa::tensor::Inputs in;
        std::array<alfa::tensor::TensorView, 6> view;
        for (size_t i = 0; i < 6; ++i)
            if (!values[i].empty())
                view[i] = {axes, shapes[i], values[i], {}};
        in = {view[0], view[1], view[2], view[3], view[4], view[5]};
        double time = 0;
        const auto result = execution->tick(in, time);
        if (!result.audioRendered || !result.modelAccepted)
            throw std::invalid_argument("Tier validation render rejected an event");
        audio[2 * sample] = execution->mixedOutput(0);
        audio[2 * sample + 1] = execution->mixedOutput(1);
    }
    return audio;
}
double pitch(const std::vector<double>& signal, double rate, double nominal)
{
    if (!std::isfinite(nominal) || nominal <= 0 || nominal >= rate / 2)
        throw std::invalid_argument("Explicit valid fundamental required");
    const auto n = signal.size() / 2;
    const auto lo = static_cast<size_t>(std::max(2., std::floor(rate / nominal * .8)));
    const auto hi = static_cast<size_t>(std::ceil(rate / nominal * 1.2));
    if (hi + 2 >= n / 2)
        throw std::invalid_argument("Pitch fixture too short for its declared fundamental");
    const auto correlation = [&](size_t lag)
    {
        double ab = 0, aa = 0, bb = 0;
        for (size_t i = hi + 2; i < n; ++i)
        {
            const auto a = signal[2 * i], b = signal[2 * (i - lag)];
            ab += a * b;
            aa += a * a;
            bb += b * b;
        }
        return ab / std::sqrt(std::max(aa * bb, 1e-30));
    };
    size_t best = lo;
    double score = -2;
    for (size_t lag = lo; lag <= hi; ++lag)
    {
        const auto c = correlation(lag);
        if (c > score)
        {
            score = c;
            best = lag;
        }
    }
    if (score < .5)
        throw std::invalid_argument("Pitch fixture lacks a stable measurable fundamental");
    const auto left = correlation(best - 1), right = correlation(best + 1), denom = left - 2 * score + right;
    const auto fraction = std::abs(denom) > 1e-12 ? std::clamp(.5 * (left - right) / denom, -.5, .5) : 0.;
    return rate / (static_cast<double>(best) + fraction);
}
JSON evaluate(const std::vector<double>& original, const std::vector<double>& candidate, const JSON& fixture,
              double rate)
{
    double energy = 0, difference = 0;
    for (size_t i = 0; i < original.size(); ++i)
    {
        energy += original[i] * original[i];
        const auto delta = original[i] - candidate[i];
        difference += delta * delta;
    }
    if (energy < 1e-20)
        throw std::invalid_argument("Identity validation requires a non-silent original");
    const auto relative = std::sqrt(difference / energy);
    const auto cents =
        1200. * std::abs(std::log2(pitch(candidate, rate, fixture.at("fundamental_hz").get<double>()) /
                                   pitch(original, rate, fixture.at("fundamental_hz").get<double>())));
    const auto search = fixture.at("timing_search_samples").get<int>();
    if (search < 0 || static_cast<size_t>(search) >= original.size() / 4)
        throw std::invalid_argument("Invalid timing search range");
    int best = 0;
    double score = -2;
    const auto frames = static_cast<int>(original.size() / 2);
    for (int lag = -search; lag <= search; ++lag)
    {
        double ab = 0, aa = 0, bb = 0;
        for (int i = search; i < frames - search; ++i)
        {
            const auto a = original[2 * static_cast<size_t>(i)],
                       b = candidate[2 * static_cast<size_t>(i + lag)];
            ab += a * b;
            aa += a * a;
            bb += b * b;
        }
        const auto c = ab / std::sqrt(std::max(aa * bb, 1e-30));
        if (c > score)
        {
            score = c;
            best = lag;
        }
    }
    return {
        {"relative_error", relative}, {"pitch_error_cents", cents}, {"timing_error_samples", std::abs(best)}};
}
} // namespace
bool convertTier(std::string_view sound, std::string_view tier, double rate, std::string& converted,
                 std::string& error)
{
    try
    {
        const auto original = JSON::parse(sound);
        const auto& policy = original.at("tier_profiles").at(std::string(tier));
        const auto& budget = policy.at("budget");
        for (const auto& value : budget)
            if (!value.is_number_unsigned() && (!value.is_number_integer() || value.get<int64_t>() < 0))
                throw std::invalid_argument("Tier budgets must be nonnegative integers");
        std::vector<JSON> proposals{original};
        for (const auto& strategy : policy.value("strategies", JSON::array()))
        {
            if (strategy != "merge_equivalent_instances")
                throw std::invalid_argument("Unknown tier optimization strategy");
            auto merged = mergeEquivalent(original);
            if (!merged.is_null())
            {
                for (size_t step = 0; step < original.at("instances").size(); ++step)
                {
                    auto next = mergeEquivalent(merged);
                    if (next.is_null())
                        break;
                    merged = std::move(next);
                }
                proposals.push_back(std::move(merged));
            }
        }
        for (const auto& candidate :
             original.value("tier_candidates", JSON::object()).value(std::string(tier), JSON::array()))
            proposals.push_back(candidate);
        double best = std::numeric_limits<double>::infinity();
        JSON selected, rejections = JSON::array();
        size_t ticks = 0;
        for (size_t index = 0; index < proposals.size(); ++index)
        {
            try
            {
                auto candidate = proposals[index];
                protect(original, candidate);
                candidate["tier"] = std::string(tier);
                candidate.erase("scalar_type");
                candidate["budget"] = budget;
                candidate["parameter_locks"] = original.value("parameter_locks", JSON::array());
                alfa::tensor::PrecisionProfile precision;
                std::string diagnostic;
                if (!alfa::tensor::resolvePrecisionJSON(candidate.dump(), precision, diagnostic))
                    throw std::invalid_argument(diagnostic);
                candidate["scalar_type"] = alfa::tensor::scalarTypeName(precision.scalarType);
                candidate["input_capacity"] = {
                    2,
                    1280,
                    48,
                    5,
                    candidate.contains("latentSound") ? candidate["latentSound"]["values"].size() : size_t(0),
                    8};
                if (!prepareExecution(candidate.dump(), rate, diagnostic))
                    throw std::invalid_argument(diagnostic);
                JSON metrics = JSON::array();
                double score = 0;
                if (index != 0 || policy.contains("validation"))
                {
                    const auto& validation = policy.at("validation");
                    const auto& cases = validation.at("cases");
                    if (!cases.is_array() || cases.empty())
                        throw std::invalid_argument("Measured conversion requires explicit validation cases");
                    for (const auto& fixture : cases)
                    {
                        const auto frames = fixture.at("frames").get<size_t>();
                        if (frames == 0 || frames > 1000000 ||
                            frames > validation.at("evaluation_bytes").get<size_t>() / (4 * sizeof(double)))
                            throw std::invalid_argument("Tier evaluation exceeds memory allowance");
                        if (ticks > validation.at("evaluation_ticks").get<size_t>() ||
                            2 * frames > validation.at("evaluation_ticks").get<size_t>() - ticks)
                            throw std::invalid_argument("Tier evaluation exceeds tick allowance");
                        ticks += 2 * frames;
                        auto metric = evaluate(render(original, fixture, rate),
                                               render(candidate, fixture, rate), fixture, rate);
                        for (const auto* name :
                             {"relative_error", "pitch_error_cents", "timing_error_samples"})
                        {
                            const auto maximum = validation.at("maximum_errors").at(name).get<double>();
                            if (!std::isfinite(maximum) || maximum < 0 ||
                                metric.at(name).get<double>() > maximum)
                                throw std::invalid_argument(std::string("Candidate exceeds protected ") +
                                                            name);
                        }
                        score += metric["relative_error"].get<double>();
                        metrics.push_back(std::move(metric));
                    }
                }
                if (score < best)
                {
                    best = score;
                    selected = std::move(candidate);
                    selected["tier_conversion"] = {
                        {"candidate", index}, {"metrics", metrics}, {"score", score}};
                }
            }
            catch (const std::exception& failure)
            {
                rejections.push_back({{"candidate", index}, {"reason", failure.what()}});
            }
        }
        if (selected.is_null())
            throw std::invalid_argument("No candidate fits tier capacity and identity constraints: " +
                                        rejections.dump());
        selected["tier_profiles"] = original.at("tier_profiles");
        if (original.contains("tier_candidates"))
            selected["tier_candidates"] = original.at("tier_candidates");
        selected["tier_conversion"]["rejected"] = rejections;
        if (selected.contains("composition"))
        {
            TensorComposition composition(rate, selected.at("budget").dump(), std::string(tier));
            juce::String diagnostic;
            if (!composition.restoreDocument(selected.dump(), diagnostic))
                throw std::invalid_argument(diagnostic.toStdString());
            selected = JSON::parse(composition.document().toStdString());
        }
        converted = selected.dump(2);
        error.clear();
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
} // namespace alfa::tensorhost
