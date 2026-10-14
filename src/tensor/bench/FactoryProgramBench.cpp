// Renders every factory program the way the plugin feeds it (stereo audio in, voice-indexed
// MIDI rows, gestures, context, preferences) and reports the runtime cost per sample.
// Usage: alfa_tensor_factory_bench [programs/factory] [seconds] [sample rate]
#include "alfa/tensor/Compiler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace
{
using namespace alfa::tensor;

std::string slurp(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

void measure(const std::filesystem::path& path, double seconds, double rate)
{
    const auto text = slurp(path);
    Configuration configuration;
    Budget budget;
    Runtime runtime;
    std::string error;
    if (!parseJSON(text, configuration, error) || !readBudgetJSON(text, budget, error) ||
        !runtime.prepare(configuration, budget, rate, error))
    {
        std::cout << std::left << std::setw(32) << path.filename().string()
                  << " not float32-loadable: " << error << "\n";
        return;
    }
    // Eight held notes spread over the keyboard, the plugin's voice-indexed MIDI layout.
    constexpr std::size_t voices = 8;
    std::array<float, voices * 10> notes{};
    for (std::size_t v = 0; v < voices; ++v)
    {
        const int key = 36 + static_cast<int>(v) * 7;
        auto* row = notes.data() + v * 10;
        row[0] = static_cast<float>(440.0 * std::exp2((key - 69) / 12.0));
        row[1] = 0.8f;
        row[2] = 1;
        row[8] = 1;
        row[9] = static_cast<float>(key);
    }
    std::array<float, 48> gestures{};
    std::array<float, 8> context{};
    std::array<float, 5> preferences{};
    std::array<float, 2> audio{};
    const std::array<std::string, 2> midiAxes{"note", "feature"}, gestureAxes{"channel", "feature"};
    const std::array<std::string, 1> channel{"channel"}, feature{"feature"};
    const std::array<std::size_t, 2> midiShape{voices, 10}, gestureShape{16, 3};
    const std::array<std::size_t, 1> stereo{2}, contextShape{8}, preferenceShape{5};
    Inputs inputs;
    inputs.audio = {channel, stereo, audio, {}};
    inputs.midi = {midiAxes, midiShape, notes, {}};
    inputs.gesture = {gestureAxes, gestureShape, gestures, {}};
    inputs.context = {feature, contextShape, context, {}};
    inputs.userPreference = {feature, preferenceShape, preferences, {}};

    // Best of five runs: the minimum is the least disturbed by other load on the machine.
    const auto total = static_cast<std::size_t>(seconds * rate);
    std::size_t rejected = 0, n = 0;
    double sink = 0, elapsed = std::numeric_limits<double>::max();
    for (int run = 0; run < 5; ++run)
    {
        const auto began = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < total; ++i, ++n)
        {
            const double phase = static_cast<double>(n) * 220.0 / rate;
            audio[0] = static_cast<float>(0.25 * std::sin(6.283185307179586 * phase));
            audio[1] = audio[0];
            context[0] = static_cast<float>(static_cast<double>(n) / rate);
            context[1] = static_cast<float>(rate);
            for (std::size_t v = 0; v < voices; ++v)
                notes[v * 10 + 3] += static_cast<float>(1.0 / rate);
            if (!runtime.tick(inputs))
                ++rejected;
            for (const auto value : runtime.output())
                sink += value;
        }
        elapsed = std::min(elapsed,
                           std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
    }
    std::cout << std::left << std::setw(32) << path.filename().string() << std::right << std::fixed
              << std::setprecision(1) << std::setw(9) << elapsed * 1e9 / static_cast<double>(total)
              << " ns/sample " << std::setw(7) << std::setprecision(2) << 100.0 * elapsed / seconds
              << " % of one core  primitives " << runtime.schedule().size() << "  rejected " << rejected
              << (std::isfinite(sink) ? "" : "  NON-FINITE") << "\n";
}
} // namespace

int main(int argc, char** argv)
{
    const std::filesystem::path directory = argc > 1 ? argv[1] : "programs/factory";
    const double seconds = argc > 2 ? std::atof(argv[2]) : 1.0;
    const double rate = argc > 3 ? std::atof(argv[3]) : 48000.0;
    std::vector<std::filesystem::path> programs;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        if (entry.path().extension() == ".json")
            programs.push_back(entry.path());
    std::sort(programs.begin(), programs.end());
    std::cout << "best of 5 x " << seconds << " s at " << rate << " Hz per program, eight held voices\n";
    for (const auto& path : programs)
        measure(path, seconds, rate);
}
