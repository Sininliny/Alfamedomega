#include "FactoryPrograms.h"
#include "TensorProcessor.h"
#include "alfa/tensor/Compiler.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numbers>

namespace
{
using namespace alfa::tensor;
using namespace alfa::tensorhost;
using Json = nlohmann::json;
const FactoryProgram& program(std::string_view id)
{
    for (const auto& p : factoryPrograms())
        if (p.id == id)
            return p;
    throw std::runtime_error("Factory program not found");
}
struct Performance
{
    std::array<std::string, 2> midiAxes{"voice", "field"};
    std::array<std::size_t, 2> midiShape{8, 10};
    std::array<float, 80> midi{};
    std::array<std::string, 1> audioAxes{"channel"};
    std::array<std::size_t, 1> audioShape{2};
    std::array<float, 2> audio{};
    Inputs input()
    {
        Inputs result;
        result.midi = {midiAxes, midiShape, midi, {}};
        result.audio = {audioAxes, audioShape, audio, {}};
        return result;
    }
};
float mix(const Runtime& runtime, const Json& j, int channel = 0)
{
    double value = 0;
    for (const auto& m : j.at("outputMix"))
        if (m[1].get<int>() == channel)
            value += static_cast<double>(runtime.output()[m[0].get<std::size_t>()]) * m[2].get<double>();
    return static_cast<float>(value);
}
} // namespace

TEST_CASE("Every authored factory program compiles at supported host sample rates", "[tensor-host][factory]")
{
    std::string error;
    for (double rate : {44100.0, 48000.0, 96000.0, 192000.0})
    {
        INFO(rate);
        REQUIRE(validateFactoryPrograms(rate, error));
        INFO(error);
    }
    REQUIRE(factoryPrograms().size() == 8);
    std::vector<std::string> ids;
    for (const auto& p : factoryPrograms())
    {
        const auto j = Json::parse(p.document);
        INFO(p.name);
        REQUIRE_FALSE(p.name.empty());
        REQUIRE_FALSE(p.description.empty());
        REQUIRE(j["schema"] == "amo.tensor/1");
        REQUIRE(j["ontology"] == "amo.math/1");
        REQUIRE_FALSE(j["authoring"]["semantic_mappings"].get<bool>());
        REQUIRE(j["parameter_metadata"].size() == j["tensors"]["parameters"]["values"].size());
        for (const auto& m : j["parameter_metadata"])
        {
            REQUIRE(m.contains("default"));
            REQUIRE(m.contains("instance"));
            REQUIRE(m.contains("primitive"));
            REQUIRE(m["min"].get<double>() <= m["default"].get<double>());
            REQUIRE(m["max"].get<double>() >= m["default"].get<double>());
            REQUIRE(m["default"] == j["tensors"]["parameters"]["values"][m["index"].get<std::size_t>()]);
        }
        if (p.category != "FX")
        {
            REQUIRE(j["performance"]["midi_layout"] == "voice_indexed");
            REQUIRE(j["performance"]["voice_capacity"] == 8);
            REQUIRE(j["performance"]["bend_range_semitones"] == 2);
            REQUIRE(j["preferred_mode"] == (p.category == "Hybrid" ? 2 : 0));
        }
        else
            REQUIRE(j["preferred_mode"] == 1);
        for (const auto& previous : ids)
            REQUIRE(previous != p.id);
        ids.push_back(p.id);
    }
}

TEST_CASE("Factory instruments are silent without gates and play two independent simultaneous notes",
          "[tensor-host][factory]")
{
    const auto& p = program("harmonic-field");
    const auto j = Json::parse(p.document);
    Runtime runtime;
    std::string error;
    REQUIRE(compileJSON(p.document, 48000, runtime, error));
    Performance performance;
    for (int n = 0; n < 256; ++n)
    {
        REQUIRE(runtime.tick(performance.input()));
        REQUIRE(mix(runtime, j) == 0);
    }
    performance.midi[0] = 440;
    performance.midi[1] = 0.8f;
    performance.midi[2] = 1;
    performance.midi[10] = 659.255f;
    performance.midi[11] = 0.6f;
    performance.midi[12] = 1;
    std::array<std::complex<double>, 2> tones{};
    double power = 0;
    constexpr int frames = 8192, settling = 4096;
    for (int n = 0; n < frames; ++n)
    {
        REQUIRE(runtime.tick(performance.input()));
        const double y = mix(runtime, j);
        REQUIRE(std::isfinite(y));
        REQUIRE(std::abs(y) < 1);
        if (n < settling)
            continue;
        power += y * y;
        for (std::size_t note = 0; note < 2; ++note)
        {
            const double frequency = note == 0 ? 440 : 659.255;
            const double phase = -2 * std::numbers::pi * frequency * static_cast<double>(n) / 48000;
            tones[note] += y * std::complex<double>{std::cos(phase), std::sin(phase)};
        }
    }
    REQUIRE(power / static_cast<double>(frames - settling) > 0.0005);
    for (const auto& t : tones)
        REQUIRE(2 * std::abs(t) / static_cast<double>(frames - settling) > 0.015);
    performance.midi[2] = 0;
    performance.midi[12] = 0; // Pitch/velocity persist through explicit release.
    float last = 0;
    for (int n = 0; n < 24000; ++n)
    {
        REQUIRE(runtime.tick(performance.input()));
        last = mix(runtime, j);
    }
    REQUIRE(std::abs(last) < 0.000001f);
}

TEST_CASE("Factory audio bindings implement explicit FIR and native stereo response",
          "[tensor-host][factory]")
{
    Performance performance;
    performance.audio = {0.25f, 0.75f};
    const auto& p = program("fir-sculpt");
    const auto j = Json::parse(p.document);
    Runtime runtime;
    std::string error;
    REQUIRE(compileJSON(p.document, 48000, runtime, error));
    REQUIRE(runtime.tick(performance.input()));
    REQUIRE(mix(runtime, j) == Catch::Approx(0.4));
    performance.audio = {0, 0};
    for (int n = 1; n <= 63; ++n)
    {
        REQUIRE(runtime.tick(performance.input()));
        const double expected = n == 11 ? 0.06 : n == 29 ? -0.045 : n == 47 ? 0.025 : n == 63 ? 0.015 : 0;
        REQUIRE(mix(runtime, j) == Catch::Approx(expected).margin(0.000001));
    }
    const auto& stereo = program("stereo-field");
    const auto sj = Json::parse(stereo.document);
    REQUIRE(compileJSON(stereo.document, 48000, runtime, error));
    performance.audio = {0.35f, -0.2f};
    REQUIRE(runtime.tick(performance.input()));
    REQUIRE(mix(runtime, sj, 0) == Catch::Approx(0.35));
    REQUIRE(mix(runtime, sj, 1) == Catch::Approx(-0.2));
}

TEST_CASE("Factory short moving delay remains audible and finite at high host sample rate",
          "[tensor-host][factory]")
{
    const auto& p = program("moving-delay");
    const auto j = Json::parse(p.document);
    Runtime runtime;
    std::string error;
    REQUIRE(compileJSON(p.document, 192000, runtime, error));
    Performance performance;
    double tail = 0;
    for (int n = 0; n < 2048; ++n)
    {
        performance.audio[0] = n == 0 ? 1.0f : 0.0f;
        REQUIRE(runtime.tick(performance.input()));
        const double value = mix(runtime, j);
        REQUIRE(std::isfinite(value));
        if (n > 400)
            tail += value * value;
    }
    REQUIRE(tail > 0.001);
}

TEST_CASE("Factory hierarchy matrices are arbitrary-sized explicit fanout with valid scalar bounds",
          "[tensor-host][factory]")
{
    for (const auto& p : factoryPrograms())
    {
        const auto j = Json::parse(p.document);
        const auto& layers = j["control_layers"];
        if (layers.empty())
            continue;
        const auto sourceCount = layers[0]["values"].size();
        REQUIRE(sourceCount == (p.category == "FX" ? 2 : 6));
        for (std::size_t corner = 0; corner < (std::size_t{1} << sourceCount); ++corner)
        {
            std::vector<double> values(sourceCount);
            for (std::size_t x = 0; x < sourceCount; ++x)
                values[x] = (corner & (std::size_t{1} << x)) ? 1 : -1;
            for (const auto& l : layers)
            {
                const auto& matrix = l["matrix"];
                const auto rows = matrix["shape"][0].get<std::size_t>(),
                           columns = matrix["shape"][1].get<std::size_t>();
                REQUIRE(columns == values.size());
                REQUIRE(matrix["values"].size() == rows * columns);
                std::vector<double> next(rows, 0.0);
                for (std::size_t row = 0; row < rows; ++row)
                    for (std::size_t col = 0; col < columns; ++col)
                        next[row] += values[col] * matrix["values"][row * columns + col].get<double>();
                values = std::move(next);
            }
            REQUIRE(values.size() == j["parameter_metadata"].size());
            for (const auto& m : j["parameter_metadata"])
            {
                const auto i = m["index"].get<std::size_t>();
                const double value = m["default"].get<double>() + values[i];
                REQUIRE(value >= m["min"].get<double>() - 0.0000001);
                REQUIRE(value <= m["max"].get<double>() + 0.0000001);
            }
        }
    }
}

TEST_CASE("Export the authored factory documents when explicitly requested", "[.factory-export]")
{
    const char* folder = std::getenv("ALFA_FACTORY_EXPORT_DIR");
    REQUIRE(folder != nullptr);
    const std::filesystem::path target(folder);
    std::filesystem::create_directories(target);
    for (const auto& p : factoryPrograms())
    {
        std::ofstream output(target / (p.id + ".amo.json"), std::ios::binary);
        output << p.document << '\n';
        REQUIRE(output.good());
    }
}

TEST_CASE("Production fragments import render and round trip through the native plugin",
          "[tensor-host][production]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    for (const auto* file : {"ring-delay", "fdn-hadamard8", "causal-fir"})
    {
        INFO(file);
        const auto document = juce::File(juce::String(ALFA_SOURCE_DIR))
                                  .getChildFile("programs/primitives")
                                  .getChildFile(juce::String(file) + ".amo.json")
                                  .loadFileAsString();
        REQUIRE(document.isNotEmpty());
        alfa::tensorhost::Processor processor;
        processor.prepareToPlay(48000, 128);
        juce::String error;
        INFO(error.toStdString());
        REQUIRE(processor.loadJSON(document, error));
        processor.mode.store(1);
        juce::AudioBuffer<float> audio(2, 128);
        juce::MidiBuffer midi;
        double energy = 0;
        for (int block = 0; block < 16; ++block)
        {
            audio.clear();
            if (block == 0)
                audio.setSample(0, 0, 1);
            processor.processBlock(audio, midi);
            for (int channel = 0; channel < 2; ++channel)
                for (int sample = 0; sample < 128; ++sample)
                {
                    const auto value = audio.getSample(channel, sample);
                    REQUIRE(std::isfinite(value));
                    energy += static_cast<double>(value) * value;
                }
        }
        REQUIRE(energy > 0);
        REQUIRE(processor.failures.load() == 0);
        juce::MemoryBlock state;
        processor.getStateInformation(state);
        alfa::tensorhost::Processor restored;
        restored.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
        REQUIRE(restored.document() == processor.document());
    }
}
