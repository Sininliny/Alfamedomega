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

template <typename Scalar>
bool common(const StepT<Scalar>& s, std::size_t parameters) noexcept
{
    if (s.parameters.size() != parameters || !std::isfinite(s.sampleRate) || s.sampleRate <= 0 ||
        s.clock.periodSamples == 0 || !std::isfinite(s.activation) || s.activation < 0)
        return false;
    return std::all_of(s.parameters.begin(), s.parameters.end(), [](Scalar v) { return std::isfinite(v); }) &&
           std::all_of(s.localState.begin(), s.localState.end(), [](Scalar v) { return std::isfinite(v); });
}
template <typename Scalar>
Scalar excitation(const StepT<Scalar>& s) noexcept
{
    return s.inputs.audio.values.empty() || !std::isfinite(s.inputs.audio.values.front())
               ? 0.0f
               : s.inputs.audio.values.front();
}
template <typename Scalar>
Scalar bounded(double value) noexcept
{
    if (!std::isfinite(value))
        return 0;
    constexpr double limit = std::numeric_limits<Scalar>::max();
    return static_cast<Scalar>(std::clamp(value, -limit, limit));
}
template <typename Scalar>
void silence(std::span<Scalar> out) noexcept
{
    std::fill(out.begin(), out.end(), 0.0f);
}
template <typename Scalar>
void hold(const StepT<Scalar>& s, std::span<Scalar> next) noexcept
{
    silence<Scalar>(next);
    std::copy_n(s.localState.begin(), std::min(s.localState.size(), next.size()), next.begin());
}
template <typename Scalar>
bool validSine(const StepT<Scalar>& s) noexcept
{
    return common<Scalar>(s, 2) && s.localState.size() == 1 && s.localState[0] >= 0 && s.localState[0] < 1;
}
template <typename Scalar>
void sineTransition(const StepT<Scalar>& s, std::span<Scalar> next) noexcept
{
    hold<Scalar>(s, next);
    if (!validSine<Scalar>(s) || next.size() != 1)
        return;
    const double advance =
        static_cast<double>(s.parameters[0]) / s.sampleRate * static_cast<double>(s.clock.periodSamples);
    if (!std::isfinite(advance))
        return;
    const double phase = std::fmod(s.localState[0] + std::fmod(advance, 1.0), 1.0);
    next[0] = static_cast<Scalar>(phase < 0 ? phase + 1 : phase);
    if (next[0] >= 1)
        next[0] = 0; // Rounding cannot produce an invalid phase.
}
template <typename Scalar>
void sineReadout(const StepT<Scalar>& s, std::span<Scalar> out) noexcept
{
    silence<Scalar>(out);
    if (!validSine<Scalar>(s) || out.size() != 1)
        return;
    out[0] = bounded<Scalar>(std::sin(2 * std::numbers::pi * s.localState[0]) *
                             static_cast<double>(s.parameters[1]) * s.activation);
}
template <typename Scalar>
bool validPole(const StepT<Scalar>& s) noexcept
{
    return common<Scalar>(s, 1) && s.localState.size() == 1 && s.parameters[0] >= 0;
}
template <typename Scalar>
void poleTransition(const StepT<Scalar>& s, std::span<Scalar> next) noexcept
{
    hold<Scalar>(s, next);
    if (!validPole<Scalar>(s) || next.size() != 1)
        return;
    const double interval = static_cast<double>(s.clock.periodSamples) / s.sampleRate;
    const double alpha =
        s.parameters[0] == 0 ? 0 : -std::expm1(-2 * std::numbers::pi * s.parameters[0] * interval);
    next[0] = bounded<Scalar>(s.localState[0] +
                              alpha * (static_cast<double>(excitation<Scalar>(s)) - s.localState[0]));
}
template <typename Scalar>
void poleReadout(const StepT<Scalar>& s, std::span<Scalar> out) noexcept
{
    silence<Scalar>(out);
    if (validPole<Scalar>(s) && out.size() == 1)
        out[0] = bounded<Scalar>(static_cast<double>(s.localState[0]) * s.activation);
}
template <typename Scalar>
bool validFeedback(const StepT<Scalar>& s) noexcept
{
    return common<Scalar>(s, 1) && !s.localState.empty() && std::abs(s.parameters[0]) < 1;
}
template <typename Scalar>
void delayTransition(const StepT<Scalar>& s, std::span<Scalar> next) noexcept
{
    hold<Scalar>(s, next);
    if (!validFeedback<Scalar>(s) || next.size() != s.localState.size())
        return;
    next[0] = bounded<Scalar>(static_cast<double>(excitation<Scalar>(s)) +
                              static_cast<double>(s.parameters[0]) * s.localState.back());
    for (std::size_t i = 1; i < next.size(); ++i)
        next[i] = s.localState[i - 1];
}
template <typename Scalar>
void delayReadout(const StepT<Scalar>& s, std::span<Scalar> out) noexcept
{
    silence<Scalar>(out);
    if (validFeedback<Scalar>(s) && out.size() == 1)
        out[0] = bounded<Scalar>(static_cast<double>(s.localState.back()) * s.activation);
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
template <typename Scalar>
PrimitiveT<Scalar> sine()
{
    return {std::string(sineInfo.id), 1, 1, sineTransition<Scalar>, sineReadout<Scalar>};
}
template <typename Scalar>
PrimitiveT<Scalar> onePole()
{
    return {std::string(poleInfo.id), 1, 1, poleTransition<Scalar>, poleReadout<Scalar>};
}
template <typename Scalar>
PrimitiveT<Scalar> feedbackCell()
{
    return {std::string(cellInfo.id), 1, 1, delayTransition<Scalar>, delayReadout<Scalar>};
}
template <typename Scalar>
PrimitiveT<Scalar> delay(std::size_t length)
{
    return {std::string(delayInfo.id), length, 1, delayTransition<Scalar>, delayReadout<Scalar>};
}
template PrimitiveT<float> sine<float>();
template PrimitiveT<float> onePole<float>();
template PrimitiveT<float> feedbackCell<float>();
template PrimitiveT<float> delay<float>(std::size_t);
template PrimitiveT<double> sine<double>();
template PrimitiveT<double> onePole<double>();
template PrimitiveT<double> feedbackCell<double>();
template PrimitiveT<double> delay<double>(std::size_t);
} // namespace alfa::tensor::kernels
