#include "FactoryPrograms.h"

#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Universe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace alfa::tensorhost
{
namespace
{
using Json = nlohmann::json;
struct Builder
{
    Json document{{"schema", "aom.tensor/1"},         {"ontology", "aom.math/1"},
                  {"instances", Json::array()},       {"projections", Json::array()},
                  {"parameter_locks", Json::array()}, {"shared_initial_state", Json::array()},
                  {"outputMix", Json::array()},       {"parameter_metadata", Json::array()}};
    std::vector<double> parameters, weights;
    std::size_t stateElements = 0, outputElements = 0, bindingTerms = 0;
    static Json tensor(std::string axis, const std::vector<double>& values)
    {
        return {{"axes", Json::array({axis})}, {"shape", Json::array({values.size()})}, {"values", values}};
    }
    std::size_t add(std::string_view id, std::size_t stateCount, std::vector<double> p,
                    std::size_t outputs = 1, Json assets = Json::array())
    {
        const auto offset = parameters.size(), index = document["instances"].size();
        document["instances"].push_back({{"primitive", id},
                                         {"activation", index},
                                         {"parameters", {{"offset", offset}, {"count", p.size()}}},
                                         {"clock", {{"period_samples", 1}, {"phase_samples", 0}}},
                                         {"output_between_updates", "zero"},
                                         {"state_size", stateCount},
                                         {"initial_state", std::vector<double>(stateCount, 0.0)},
                                         {"assets", std::move(assets)}});
        parameters.insert(parameters.end(), p.begin(), p.end());
        stateElements += stateCount;
        outputElements += outputs;
        return offset;
    }
    void projection(std::string source, std::size_t sourceElement, std::size_t parameter, double weight)
    {
        const auto index = weights.size();
        weights.push_back(weight);
        document["projections"].push_back({{"source", source},
                                           {"source_element", sourceElement},
                                           {"parameter_element", parameter},
                                           {"weights", "routing"},
                                           {"weight_element", index},
                                           {"missing_source", "zero"}});
    }
    void metadata(std::size_t parameter, std::string name, std::string unit, double minimum, double maximum,
                  double step, bool driven = false, std::string link = {})
    {
        Json m{
            {"index", parameter}, {"name", name}, {"unit", unit},     {"min", minimum},
            {"max", maximum},     {"step", step}, {"driven", driven}, {"default", parameters.at(parameter)}};
        for (std::size_t instance = 0; instance < document["instances"].size(); ++instance)
        {
            const auto& entry = document["instances"][instance];
            const auto offset = entry["parameters"]["offset"].get<std::size_t>();
            const auto count = entry["parameters"]["count"].get<std::size_t>();
            if (parameter >= offset && parameter - offset < count)
            {
                m["instance"] = instance;
                m["primitive"] = entry["primitive"];
                m["parameter"] = parameter - offset;
                break;
            }
        }
        if (!link.empty())
            m["link"] = link;
        document["parameter_metadata"].push_back(std::move(m));
    }
    void mix(std::size_t output, int channel, double gain)
    {
        document["outputMix"].push_back(Json::array({output, channel, gain}));
    }
    void bind(std::size_t instance, std::string source, std::size_t sourceElement, double weight)
    {
        auto& entry = document["instances"][instance];
        entry["input_channels"] = 1;
        if (!entry.contains("audio_bindings"))
            entry["audio_bindings"] = Json::array();
        const auto index = weights.size();
        weights.push_back(weight);
        ++bindingTerms;
        entry["audio_bindings"].push_back({{"source", source},
                                           {"source_element", sourceElement},
                                           {"destination_element", 0},
                                           {"weights", "routing"},
                                           {"weight_element", index},
                                           {"missing_source", "zero"}});
    }
    Json finish()
    {
        const auto count = document["instances"].size();
        std::size_t storage = count + parameters.size() + weights.size();
        for (const auto& instance : document["instances"])
        {
            storage += instance.value("input_channels", std::size_t{0});
            for (const auto& asset : instance["assets"])
                storage += asset["values"].size();
        }
        document["tensors"] = {{"activation", tensor("primitive", std::vector<double>(count, 1.0))},
                               {"parameters", tensor("parameter", parameters)},
                               {"routing", tensor("coefficient", weights)},
                               {"modulation", tensor("relationship", {})}};
        document["budget"] = {{"active_primitives", count},
                              {"state_elements", stateElements},
                              {"parameter_elements", parameters.size()},
                              {"routing_elements", weights.size()},
                              {"modulation_elements", 0},
                              {"output_elements", outputElements},
                              {"stored_tensor_elements", storage},
                              {"projection_terms", document["projections"].size() + bindingTerms}};
        return std::move(document);
    }
};

Json layer(std::vector<double> values, std::size_t destinations, std::vector<double> matrix)
{
    const auto sources = values.size();
    return {{"values", std::move(values)},
            {"ranges", std::vector<std::array<double, 2>>(sources, {-1.0, 1.0})},
            {"matrix",
             {{"axes", {"destination", "source"}},
              {"shape", {destinations, sources}},
              {"values", std::move(matrix)}}}};
}
void accountControlStorage(Json& document)
{
    auto storage = document["budget"]["stored_tensor_elements"].get<std::size_t>();
    for (const auto& l : document["control_layers"])
        storage += l["values"].size() + l["matrix"]["values"].size();
    document["budget"]["stored_tensor_elements"] = storage;
}
void instrumentLayers(Json& document, std::size_t voices, std::size_t partials, std::size_t filterParameter,
                      std::optional<std::size_t> widthParameter, const std::vector<double>& ratios)
{
    constexpr std::size_t high = 6, middle = 10;
    std::vector<double> first(high * middle, 0.0);
    for (std::size_t i = 0; i < high; ++i)
        first[i * high + i] = 1;
    first[6 * high] = 0.4;
    first[6 * high + 2] = 0.6;
    first[7 * high + 1] = 0.3;
    first[7 * high + 3] = 0.7;
    first[8 * high + 4] = 0.3;
    first[8 * high + 5] = 0.7;
    first[9 * high] = -0.2;
    first[9 * high + 4] = 0.8;
    const auto count = document["tensors"]["parameters"]["values"].size();
    std::vector<double> final(count * middle, 0.0);
    for (std::size_t voice = 0; voice < voices; ++voice)
    {
        const auto p = voice * (5 + partials * 2);
        final[(p + 1) * middle] = 0.001;
        final[(p + 2) * middle + 1] = 0.07;
        final[(p + 2) * middle + 6] = 0.02;
        final[(p + 3) * middle + 2] = 0.06;
        final[(p + 3) * middle + 7] = 0.02;
        final[(p + 4) * middle + 3] = 0.03;
        final[(p + 4) * middle + 7] = 0.01;
        for (std::size_t partial = 0; partial < partials; ++partial)
        {
            final[(p + 5 + 2 * partial) * middle + 8] = 4 * ratios[partial];
            final[(p + 5 + 2 * partial) * middle + 9] = 0.5 * ratios[partial];
        }
    }
    final[filterParameter * middle + 4] = 1200;
    final[filterParameter * middle + 6] = 200;
    final[(filterParameter + 1) * middle + 5] = 0.12;
    final[(filterParameter + 1) * middle + 9] = 0.03;
    if (widthParameter)
        final[*widthParameter * middle + 4] = 0.15;
    document["control_layers"] =
        Json::array({layer(std::vector<double>(high, 0.0), middle, std::move(first)),
                     layer(std::vector<double>(middle, 0.0), count, std::move(final))});
    accountControlStorage(document);
}
void fxLayers(Json& document, const std::vector<std::array<double, 3>>& coefficients)
{
    const auto count = document["tensors"]["parameters"]["values"].size();
    if (count == 0)
    {
        document["control_layers"] = Json::array();
        return;
    }
    std::vector<double> final(count * 3, 0.0);
    for (std::size_t p = 0; p < count; ++p)
        for (std::size_t x = 0; x < 3; ++x)
            final[p * 3 + x] = coefficients.at(p)[x];
    document["control_layers"] = Json::array(
        {layer({0, 0}, 3, {1, 0, 0.2, 0.8, -0.5, 0.5}), layer({0, 0, 0}, count, std::move(final))});
    accountControlStorage(document);
}

struct Envelope
{
    double attack, decay, sustain, release;
    double velocityAttack, velocityDecay;
};
FactoryProgram instrument(std::string id, std::string name, std::string description,
                          const std::vector<double>& ratios, const std::vector<double>& gains,
                          Envelope envelope, bool hybrid = false)
{
    constexpr std::size_t voices = 8;
    Builder b;
    std::vector<std::size_t> partialOutputs;
    for (std::size_t voice = 0; voice < voices; ++voice)
    {
        const auto state = b.stateElements;
        const auto envelopeParameters = b.add(
            "control.adsr", 4, {0, envelope.attack, envelope.decay, envelope.sustain, envelope.release});
        const auto oscillatorOutput = b.outputElements;
        partialOutputs.push_back(oscillatorOutput);
        std::vector<double> p(ratios.size() * 2, 0.0);
        const auto partialParameters = b.add("osc.additive", ratios.size(), std::move(p));
        b.projection("midi", voice * 10 + 2, envelopeParameters, 1);
        if (envelope.velocityAttack != 0)
            b.projection("midi", voice * 10 + 1, envelopeParameters + 1, envelope.velocityAttack);
        if (envelope.velocityDecay != 0)
            b.projection("midi", voice * 10 + 1, envelopeParameters + 2, envelope.velocityDecay);
        const auto prefix = "Voice " + std::to_string(voice + 1) + " ";
        b.metadata(envelopeParameters, prefix + "gate", "linear", 0, 0, 0, true);
        b.metadata(envelopeParameters + 1, prefix + "attack", "s",
                   std::max(0.0005, -envelope.velocityAttack + 0.0005), 2, 0.0005, false, "attack");
        b.metadata(envelopeParameters + 2, prefix + "decay", "s", 0.001, 4, 0.001, false, "decay");
        b.metadata(envelopeParameters + 3, prefix + "sustain", "linear", 0, 1, 0.01, false, "sustain");
        b.metadata(envelopeParameters + 4, prefix + "release", "s", 0.001, envelope.release + 0.25, 0.001,
                   false, "release");
        for (std::size_t partial = 0; partial < ratios.size(); ++partial)
        {
            b.projection("midi", voice * 10, partialParameters + 2 * partial, ratios[partial]);
            b.projection("state", state, partialParameters + 2 * partial + 1, gains[partial]);
            if (hybrid)
                b.projection("audio", 0, partialParameters + 2 * partial, 14 * ratios[partial]);
            b.metadata(partialParameters + 2 * partial,
                       prefix + "partial " + std::to_string(partial + 1) + " frequency offset", "Hz",
                       -24 * ratios[partial], 24 * ratios[partial], 0.01, true);
            b.metadata(partialParameters + 2 * partial + 1,
                       prefix + "partial " + std::to_string(partial + 1) + " amplitude offset", "linear", 0,
                       0.02, 0.001, true);
        }
        b.mix(oscillatorOutput, 0, 0.5);
        b.mix(oscillatorOutput, 1, 0.5);
    }
    const auto bodyInstance = b.document["instances"].size(), bodyOutput = b.outputElements;
    const auto filter = b.add("filter.biquad_lowpass", 4, {3400, 0.70710678});
    for (auto output : partialOutputs)
        b.bind(bodyInstance, "output_history", output, 1);
    b.bind(bodyInstance, "output_history", bodyOutput, 0.03);
    if (hybrid)
        b.bind(bodyInstance, "audio", 0, 0.15);
    b.metadata(filter, "Shared biquad cutoff", "Hz", 100, 16000, 1);
    b.metadata(filter + 1, "Shared biquad Q", "ratio", 0.2, 2, 0.01);
    b.mix(bodyOutput, 0, 0.1);
    b.mix(bodyOutput, 1, 0.1);
    std::optional<std::size_t> widthParameter;
    if (hybrid)
    {
        const auto dry = b.outputElements;
        const auto width = b.add("spatial.stereo_width", 0, {1}, 2);
        widthParameter = width;
        b.metadata(width, "Input stereo width", "linear", 0, 2, 0.01);
        b.mix(dry, 0, 0.2);
        b.mix(dry + 1, 1, 0.2);
    }
    auto j = b.finish();
    j["preferred_mode"] = hybrid ? 2 : 0;
    j["performance"] = {{"mode", hybrid ? "hybrid" : "instrument"},
                        {"midi_layout", "voice_indexed"},
                        {"voice_capacity", voices},
                        {"release_seconds", envelope.release + 0.3},
                        {"bend_range_semitones", 2}};
    j["description"] = description;
    instrumentLayers(j, voices, ratios.size(), filter, widthParameter, ratios);
    j["authoring"] = {
        {"kind", "explicit mathematical configuration"},
        {"semantic_mappings", false},
        {"velocity_behavior", "attack and decay only; amplitude follows explicitly allocated envelope state"},
        {"partial_frequencies",
         "explicit ratios; this additive reference does not suppress harmonics above Nyquist"}};
    j["preset"] = {{"schema", "aom.preset/1"},
                   {"id", id},
                   {"name", name},
                   {"category", hybrid ? "Hybrid" : "Instrument"}};
    return {std::move(id), std::move(name), hybrid ? "Hybrid" : "Instrument", std::move(description),
            j.dump(2)};
}

FactoryProgram fx(std::string id, std::string name, std::string description, Builder b,
                  const std::vector<std::array<double, 3>>& controlCoefficients = {})
{
    auto j = b.finish();
    j["preferred_mode"] = 1;
    j["performance"] = {{"mode", "fx"}};
    j["description"] = description;
    fxLayers(j, controlCoefficients);
    j["authoring"] = {{"kind", "explicit mathematical configuration"}, {"semantic_mappings", false}};
    j["preset"] = {{"schema", "aom.preset/1"}, {"id", id}, {"name", name}, {"category", "FX"}};
    return {std::move(id), std::move(name), "FX", std::move(description), j.dump(2)};
}
std::vector<FactoryProgram> makePrograms()
{
    std::vector<FactoryProgram> programs;
    programs.push_back(instrument("harmonic-field", "Harmonic Field",
                                  "Eight independent note envelopes share MIDI and recurrent control tensors "
                                  "with harmonic partials. Velocity shortens attack and lengthens decay.",
                                  {1, 2, 3, 4, 6}, {0.075, 0.035, 0.022, 0.013, 0.007},
                                  {0.055, 0.24, 0.72, 0.3, -0.035, 0.18}));
    programs.push_back(instrument("glass-partials", "Glass Partials",
                                  "Eight note voices with explicitly inharmonic partial ratios, a fast onset "
                                  "and a fading held response. No instrument-method selection is encoded.",
                                  {1, 1.41421356, 2.72, 3.9}, {0.07, 0.043, 0.025, 0.015},
                                  {0.012, 0.7, 0.16, 0.6, -0.009, 0.25}));
    programs.push_back(instrument("subharmonic-field", "Subharmonic Field",
                                  "Eight envelopes control explicit half-frequency and harmonic partials, "
                                  "with a slower onset and a sustained body.",
                                  {0.5, 1, 1.5, 2.5}, {0.075, 0.045, 0.02, 0.01},
                                  {0.11, 0.24, 0.6, 0.4, -0.06, 0.08}));
    {
        Builder b;
        const auto p = b.add("spatial.stereo_width", 0, {1}, 2);
        b.metadata(p, "Stereo width", "linear", 0, 2, 0.01);
        b.mix(0, 0, 1);
        b.mix(1, 1, 1);
        programs.push_back(fx(
            "stereo-field", "Stereo Field",
            "Native stereo mid/side transformation. Width is an explicitly editable mathematical parameter.",
            std::move(b), {{0.4, 0, 0.15}}));
    }
    {
        Builder b;
        const auto p = b.add("time.modulated_delay", 1025, {0.003, 0.001, 0.35, 0.4, 0.62});
        b.bind(0, "audio", 0, 1);
        b.bind(0, "output_history", 0, 0.03);
        b.metadata(p, "Base delay", "s", 0.0024, 0.0036, 0.0001);
        b.metadata(p + 1, "Depth", "s", 0, 0.0012, 0.0001);
        b.metadata(p + 2, "Modulation rate", "Hz", 0.01, 5, 0.01);
        b.metadata(p + 3, "Wet mix", "linear", 0, 1, 0.01);
        b.metadata(p + 4, "Feedback", "linear", -0.95, 0.95, 0.01);
        b.mix(0, 0, 0.8);
        b.mix(0, 1, 0.8);
        programs.push_back(
            fx("moving-delay", "Moving Delay",
               "A short fractional feedback delay receives the left input channel and its previous output, "
               "then mirrors the result to stereo. Modulation and feedback are explicit tensor weights.",
               std::move(b), {{0.0004, 0, 0}, {0, 0.00015, 0}, {0, 0.1, 0}, {0.18, 0, 0}, {0, 0, 0.14}}));
    }
    {
        Builder b;
        const auto feedback = b.add("time.fdn_reverb", 444, {0.86}, 1,
                                    Json::array({Builder::tensor("delay_line", {61, 83, 127, 173})}));
        b.bind(0, "audio", 0, 0.5);
        b.bind(0, "audio", 1, 0.5);
        const auto width = b.add("spatial.stereo_width", 0, {1}, 2);
        b.metadata(feedback, "Network feedback", "linear", 0, 0.97, 0.01);
        b.metadata(width, "Dry stereo width", "linear", 0, 2, 0.01);
        b.mix(0, 0, 0.18);
        b.mix(0, 1, 0.18);
        b.mix(1, 0, 0.8);
        b.mix(2, 1, 0.8);
        programs.push_back(fx("short-diffuse-space", "Short Diffuse Space",
                              "Explicit unequal delay memories form a short Householder feedback response "
                              "alongside the native stereo input. This is a small diffusion space.",
                              std::move(b), {{0.055, 0, 0.02}, {0, 0.2, 0}}));
    }
    {
        Builder b;
        std::vector<double> ir(64, 0.0);
        ir[0] = 0.8;
        ir[11] = 0.12;
        ir[29] = -0.09;
        ir[47] = 0.05;
        ir[63] = 0.03;
        b.add("time.convolution_fir", 63, {}, 1, Json::array({Builder::tensor("impulse_sample", ir)}));
        b.bind(0, "audio", 0, 0.5);
        b.bind(0, "audio", 1, 0.5);
        b.mix(0, 0, 1);
        b.mix(0, 1, 1);
        programs.push_back(
            fx("fir-sculpt", "FIR Sculpt",
               "A documented short authored impulse response shapes the mean stereo input and mirrors its "
               "result to stereo; this is not a measured acoustic or HRTF dataset.",
               std::move(b)));
    }
    programs.push_back(instrument(
        "audio-motion-field", "Audio Motion Field",
        "Eight note envelopes and harmonic partials also receive native audio-rate frequency influence, with "
        "an explicit stereo input contribution. Audio, MIDI and recurrent state coexist.",
        {1, 2, 3, 5}, {0.07, 0.04, 0.025, 0.012}, {0.04, 0.28, 0.65, 0.4, -0.02, 0.12}, true));
    for (auto& program : programs)
    {
        auto data = Json::parse(program.document);
        data["tier"] = "Standard";
        data["scalar_type"] = "float32";
        program.document = data.dump(2);
    }
    return programs;
}
} // namespace
std::span<const FactoryProgram> factoryPrograms()
{
    static const auto programs = makePrograms();
    return programs;
}
bool validateFactoryPrograms(double sampleRate, std::string& error)
{
    for (const auto& program : factoryPrograms())
    {
        alfa::tensor::Runtime runtime;
        if (!alfa::tensor::compileJSON(program.document, sampleRate, runtime, error))
        {
            error = program.name + ": " + error;
            return false;
        }
    }
    error.clear();
    return true;
}
} // namespace alfa::tensorhost
