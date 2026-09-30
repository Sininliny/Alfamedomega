#include "alfa/tensor/Kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace alfa::tensor::kernels
{
namespace
{
constexpr Parameter sineParameters[]{{"frequency", "Hz", "finite signed frequency", true, true, true},
                                     {"amplitude", "linear", "finite", true, true, true}};
constexpr Parameter poleParameters[]{{"cutoff", "Hz", "finite >= 0", true, true, true}};
constexpr Parameter feedbackParameters[]{
    {"feedback", "linear", "finite -1 < value < 1 (stable reference kernel)", true, true, true}};
const Metadata sineInfo{"math.sine",
                        "no excitation; parameters selected by compiler",
                        "one scalar sample",
                        sineParameters,
                        "phase in cycles [0,1)",
                        "O(1), one sine evaluation",
                        "explicit positive clock; phase advances by clock period / sample rate",
                        "configuration/hardware capacity; no per-kind instance limit"};
const Metadata poleInfo{
    "math.one_pole_lowpass",
    "audio.values[0], absent = 0",
    "one scalar sample from previous state",
    poleParameters,
    "one filter memory value",
    "O(1), one exponential coefficient evaluation",
    "sample rate or explicitly chosen positive update clock; sampled input held across update interval",
    "configuration/hardware capacity; no per-kind instance limit"};
const Metadata cellInfo{"math.feedback_cell",
                        "audio.values[0], absent = 0",
                        "one scalar sample from previous state",
                        feedbackParameters,
                        "one recurrence value",
                        "O(1)",
                        "explicit positive update clock; one recurrence per due update",
                        "configuration/hardware capacity; no per-kind instance limit"};
const Metadata delayInfo{"math.integer_delay",
                         "audio.values[0], absent = 0",
                         "one scalar sample delayed by allocated state length in updates",
                         feedbackParameters,
                         "shift register: newest to oldest, length supplied by compiler",
                         "O(state length) reference kernel",
                         "explicit positive update clock; delay length measured in updates",
                         "configuration/hardware capacity; no per-kind instance limit"};

bool common(const Step& s, std::size_t parameters) noexcept
{
    if (s.parameters.size() != parameters || !std::isfinite(s.sampleRate) || s.sampleRate <= 0 ||
        s.clock.periodSamples == 0 || !std::isfinite(s.activation) || s.activation < 0)
        return false;
    return std::all_of(s.parameters.begin(), s.parameters.end(), [](float v) { return std::isfinite(v); }) &&
           std::all_of(s.localState.begin(), s.localState.end(), [](float v) { return std::isfinite(v); });
}
float excitation(const Step& s) noexcept
{
    return s.inputs.audio.values.empty() || !std::isfinite(s.inputs.audio.values.front())
               ? 0.0f
               : s.inputs.audio.values.front();
}
float bounded(double value) noexcept
{
    if (!std::isfinite(value))
        return 0;
    constexpr double limit = std::numeric_limits<float>::max();
    return static_cast<float>(std::clamp(value, -limit, limit));
}
void silence(std::span<float> out) noexcept
{
    std::fill(out.begin(), out.end(), 0.0f);
}
void hold(const Step& s, std::span<float> next) noexcept
{
    silence(next);
    std::copy_n(s.localState.begin(), std::min(s.localState.size(), next.size()), next.begin());
}
bool validSine(const Step& s) noexcept
{
    return common(s, 2) && s.localState.size() == 1 && s.localState[0] >= 0 && s.localState[0] < 1;
}
void sineTransition(const Step& s, std::span<float> next) noexcept
{
    hold(s, next);
    if (!validSine(s) || next.size() != 1)
        return;
    const double advance =
        static_cast<double>(s.parameters[0]) / s.sampleRate * static_cast<double>(s.clock.periodSamples);
    if (!std::isfinite(advance))
        return;
    const double phase = std::fmod(s.localState[0] + std::fmod(advance, 1.0), 1.0);
    next[0] = static_cast<float>(phase < 0 ? phase + 1 : phase);
    if (next[0] >= 1)
        next[0] = 0; // Rounding cannot produce an invalid phase.
}
void sineReadout(const Step& s, std::span<float> out) noexcept
{
    silence(out);
    if (!validSine(s) || out.size() != 1)
        return;
    out[0] = bounded(std::sin(2 * std::numbers::pi * s.localState[0]) * static_cast<double>(s.parameters[1]) *
                     s.activation);
}
bool validPole(const Step& s) noexcept
{
    return common(s, 1) && s.localState.size() == 1 && s.parameters[0] >= 0;
}
void poleTransition(const Step& s, std::span<float> next) noexcept
{
    hold(s, next);
    if (!validPole(s) || next.size() != 1)
        return;
    const double interval = static_cast<double>(s.clock.periodSamples) / s.sampleRate;
    const double alpha =
        s.parameters[0] == 0 ? 0 : -std::expm1(-2 * std::numbers::pi * s.parameters[0] * interval);
    next[0] = bounded(s.localState[0] + alpha * (static_cast<double>(excitation(s)) - s.localState[0]));
}
void poleReadout(const Step& s, std::span<float> out) noexcept
{
    silence(out);
    if (validPole(s) && out.size() == 1)
        out[0] = bounded(static_cast<double>(s.localState[0]) * s.activation);
}
bool validFeedback(const Step& s) noexcept
{
    return common(s, 1) && !s.localState.empty() && std::abs(s.parameters[0]) < 1;
}
void delayTransition(const Step& s, std::span<float> next) noexcept
{
    hold(s, next);
    if (!validFeedback(s) || next.size() != s.localState.size())
        return;
    next[0] = bounded(static_cast<double>(excitation(s)) +
                      static_cast<double>(s.parameters[0]) * s.localState.back());
    for (std::size_t i = 1; i < next.size(); ++i)
        next[i] = s.localState[i - 1];
}
void delayReadout(const Step& s, std::span<float> out) noexcept
{
    silence(out);
    if (validFeedback(s) && out.size() == 1)
        out[0] = bounded(static_cast<double>(s.localState.back()) * s.activation);
}
} // namespace
const Metadata& sineMetadata() noexcept
{
    return sineInfo;
}
const Metadata& onePoleMetadata() noexcept
{
    return poleInfo;
}
const Metadata& feedbackCellMetadata() noexcept
{
    return cellInfo;
}
const Metadata& delayMetadata() noexcept
{
    return delayInfo;
}
Primitive sine()
{
    return {std::string(sineInfo.id), 1, 1, sineTransition, sineReadout};
}
Primitive onePole()
{
    return {std::string(poleInfo.id), 1, 1, poleTransition, poleReadout};
}
Primitive feedbackCell()
{
    return {std::string(cellInfo.id), 1, 1, delayTransition, delayReadout};
}
Primitive delay(std::size_t length)
{
    return {std::string(delayInfo.id), length, 1, delayTransition, delayReadout};
}
} // namespace alfa::tensor::kernels
