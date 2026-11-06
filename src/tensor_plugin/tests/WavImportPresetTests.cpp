// Presets written by AMO Centre's WAV to Preset (training/amo/corpus/wav_to_preset.py) must load
// in the plugin's library and play like an authored instrument: idle voices are silent and finite,
// and two simultaneous voices sound. Fixtures: tests/data/wav-import (regenerate with that script).
#include "PresetStore.h"
#include "alfa/tensor/Compiler.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace
{
using namespace alfa::tensor;
using Json = nlohmann::json;

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    std::stringstream text;
    text << stream.rdbuf();
    return text.str();
}
} // namespace

TEST_CASE("WAV-imported presets validate in the library and play polyphonically", "[tensor-host][presets][wav-import]")
{
    const auto folder = std::filesystem::path(ALFA_SOURCE_DIR) / "src/tensor_plugin/tests/data/wav-import";
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(folder))
    {
        if (entry.path().extension() != ".json")
            continue;
        ++count;
        const auto text = readFile(entry.path());
        INFO(entry.path().filename().string());
        juce::String error;
        REQUIRE(alfa::tensorhost::PresetStore::validateDocument(juce::String::fromUTF8(text.c_str()), error));
        INFO(error.toStdString());
        const auto j = Json::parse(text);
        REQUIRE(j["performance"]["midi_layout"] == "voice_indexed");
        REQUIRE(j["performance"]["voice_capacity"] == 8);
        REQUIRE(j["preset"]["schema"] == "amo.preset/1");
        for (double rate : {44100.0, 48000.0, 96000.0})
        {
            INFO(rate);
            Runtime runtime;
            std::string message;
            REQUIRE(compileJSON(text, rate, runtime, message));
            std::array<std::string, 2> midiAxes{"voice", "field"};
            std::array<std::size_t, 2> midiShape{8, 10};
            std::array<float, 80> midi{};
            std::array<std::string, 1> audioAxes{"channel"};
            std::array<std::size_t, 1> audioShape{2};
            std::array<float, 2> audio{};
            const auto input = [&]
            {
                Inputs result;
                result.midi = {midiAxes, midiShape, midi, {}};
                result.audio = {audioAxes, audioShape, audio, {}};
                return result;
            };
            const auto mixed = [&]
            {
                double value = 0;
                for (const auto& m : j.at("outputMix"))
                    if (m[1].get<int>() == 0)
                        value += static_cast<double>(runtime.output()[m[0].get<std::size_t>()]) * m[2].get<double>();
                return value;
            };
            for (int n = 0; n < 512; ++n)
            {
                REQUIRE(runtime.tick(input()));
                REQUIRE(std::isfinite(mixed()));
            }
            midi[0] = 220.0f, midi[1] = 0.8f, midi[2] = 1.0f;
            midi[10] = 329.63f, midi[11] = 0.6f, midi[12] = 1.0f;
            double power = 0;
            const int frames = static_cast<int>(rate / 4);
            for (int n = 0; n < frames; ++n)
            {
                REQUIRE(runtime.tick(input()));
                const auto y = mixed();
                REQUIRE(std::isfinite(y));
                REQUIRE(std::abs(y) < 4.0);
                power += y * y;
            }
            REQUIRE(power / frames > 1e-8);
        }
    }
    REQUIRE(count >= 1);
}
