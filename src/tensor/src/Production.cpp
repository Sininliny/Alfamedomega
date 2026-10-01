#include "alfa/tensor/Production.h"

#include "alfa/dsp/detail/DelayInterpolation.h"
#include "alfa/dsp/detail/Hadamard.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace alfa::tensor::production
{
namespace
{
constexpr std::size_t rejected = std::numeric_limits<std::size_t>::max();
constexpr kernels::Parameter delayParams[]{
    {"delay", "updates", "integer 1..history capacity", true, true, true},
    {"feedback", "linear", "finite -1 < feedback < 1", true, true, true}};
constexpr kernels::Parameter cubicParams[]{
    {"delay", "updates", "finite 2..history capacity-2; Catmull-Rom read", true, true, true},
    {"feedback", "linear", "finite -1 < feedback < 1", true, true, true}};
constexpr kernels::Parameter allpassParams[]{
    {"delay", "updates", "finite 1..history capacity-1; linear read", true, true, true},
    {"gain", "linear", "finite -1 < gain < 1", true, true, true}};
constexpr kernels::Parameter fdnParams[]{
    {"delay[i]", "updates", "integer 1..per-line capacity; parameters[3*i], i=0..7", true, true, true},
    {"gain[i]", "linear", "-1 < gain < 1; parameters[3*i+1]", true, true, true},
    {"damping[i]", "pole", "0 <= damping < 1; parameters[3*i+2]", true, true, true}};
constexpr std::array<kernels::Parameter, 0> noParameters{};
constexpr auto rates = "any explicit positive clock; one update is one scheduled event, no global rate";
constexpr auto reuse = "arbitrary independent instances within declared resource budget";
const std::array<kernels::Metadata, 5> metadata{
    {{"time.ring_delay.v1", "audio[0]; absent/padding=0", "delayed scalar; activation scales readout",
      delayParams,
      "[cursor low16, cursor high16, C history samples]; cursor is next write index; state size=C+2",
      "O(1) read/update, exactly 3 journal writes, no history shift", rates, reuse},
     {"time.ring_delay_cubic.v1", "audio[0]; delay may change each update", "fractional delayed scalar",
      cubicParams, "two exact16-bit cursor limbs followed by C history samples; C>=4",
      "O(1), four-point shared legacy Hermite read, 3 journal writes", rates, reuse},
     {"time.ring_allpass.v1", "audio[0]; explicit gain and fractional delay",
      "w-g*(x+g*w); scalar diffusion allpass", allpassParams,
      "two cursor limbs followed by C history samples; writes x+g*w; C>=2",
      "O(1), linear read, 3 journal writes; fixed integer delay is allpass for abs(g)<1", rates, reuse},
     {"time.fdn_hadamard8.v1", "audio[0..7]; independent per-line excitation",
      "8 damped delayed scalars before loop gains", fdnParams,
      "8 line blocks [cursor low16, high16, damping memory, C history samples]; size=8*(C+3)",
      "O(8 log2(8)) butterfly+8 fixed ring reads, 32 journal writes; no line shifting", rates, reuse},
     {"time.causal_fir_ring.v1", "audio[0]; asset[0] contains causal coefficients h[0..K-1]",
      "h[0]*x[t]+sum(h[k]*x[t-k]); no extra computational sample latency", noParameters,
      "two cursor limbs plus C input history samples; 1<=K<=C+1; padded taps=0",
      "O(K) readout, O(1) update, 3 journal writes; direct FIR, not partitioned FFT convolution", rates,
      reuse}}};

template <typename S>
bool common(const StepT<S>& s, std::size_t count) noexcept
{
    if (s.parameters.size() != count || s.clock.periodSamples == 0 || !std::isfinite(s.sampleRate) ||
        s.sampleRate <= 0 || !std::isfinite(s.activation) || s.activation < 0)
        return false;
    for (const auto value : s.parameters)
        if (!std::isfinite(value))
            return false;
    return true;
}
template <typename S>
bool excitation(const StepT<S>& s, std::size_t channel, S& value) noexcept
{
    const auto& input = s.inputs.audio;
    value = 0;
    if (channel >= input.values.size())
        return true;
    if (!input.presence.empty())
    {
        if (input.presence.size() != input.values.size())
            return false;
        if (input.presence[channel] == 0)
            return true;
        if (input.presence[channel] != 1)
            return false;
    }
    value = input.values[channel];
    return std::isfinite(value);
}
template <typename S>
bool cursor(std::span<const S> state, std::size_t base, std::size_t capacity, std::size_t& index) noexcept
{
    if (capacity == 0 || base > state.size() || state.size() - base < 2)
        return false;
    const auto lo = state[base], hi = state[base + 1];
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo < 0 || lo > 65535 || hi < 0 || hi > 65535 ||
        std::floor(lo) != lo || std::floor(hi) != hi)
        return false;
    index = static_cast<std::size_t>(lo) + static_cast<std::size_t>(hi) * 65536;
    return index < capacity;
}
template <typename S>
S past(std::span<const S> state, std::size_t start, std::size_t capacity, std::size_t index,
       std::size_t delay) noexcept
{
    return state[start + (index >= delay ? index - delay : capacity - (delay - index))];
}
template <typename S>
S linear(std::span<const S> state, std::size_t start, std::size_t capacity, std::size_t index,
         S delay) noexcept
{
    const auto integer = static_cast<std::size_t>(std::floor(delay));
    const auto fraction = delay - static_cast<S>(integer);
    const auto a = past(state, start, capacity, index, integer),
               b = past(state, start, capacity, index, integer + 1);
    return a + fraction * (b - a);
}
template <typename S>
S cubic(std::span<const S> state, std::size_t start, std::size_t capacity, std::size_t index,
        S delay) noexcept
{
    const auto integer = static_cast<std::size_t>(std::floor(delay));
    const auto fraction = delay - static_cast<S>(integer);
    return alfa::dsp::detail::hermite(past(state, start, capacity, index, integer - 1),
                                      past(state, start, capacity, index, integer),
                                      past(state, start, capacity, index, integer + 1),
                                      past(state, start, capacity, index, integer + 2), fraction);
}
template <typename S>
void cursorWrites(std::span<StateWriteT<S>> writes, std::size_t first, std::size_t base,
                  std::size_t index) noexcept
{
    writes[first] = {base, static_cast<S>(index % 65536)};
    writes[first + 1] = {base + 1, static_cast<S>(index / 65536)};
}
enum class Ring
{
    Integer,
    Cubic,
    Allpass
};
template <typename S, Ring kind>
bool ringValues(const StepT<S>& s, std::size_t& index, S& input, S& delayed) noexcept
{
    constexpr std::size_t minimum = kind == Ring::Integer ? 1 : kind == Ring::Cubic ? 4 : 2;
    if (!common(s, 2) || s.localState.size() < minimum + 2)
        return false;
    const auto capacity = s.localState.size() - 2;
    if (capacity > std::numeric_limits<std::uint32_t>::max() || !cursor(s.localState, 0, capacity, index) ||
        !excitation(s, 0, input) || std::abs(s.parameters[1]) >= 1)
        return false;
    const auto delay = s.parameters[0];
    if constexpr (kind == Ring::Integer)
    {
        if (delay < 1 || static_cast<double>(delay) > static_cast<double>(capacity) ||
            std::floor(delay) != delay)
            return false;
        delayed = past(s.localState, 2, capacity, index, static_cast<std::size_t>(delay));
    }
    else if constexpr (kind == Ring::Cubic)
    {
        if (delay < 2 || static_cast<double>(delay) > static_cast<double>(capacity - 2))
            return false;
        delayed = cubic(s.localState, 2, capacity, index, delay);
    }
    else
    {
        if (delay < 1 || static_cast<double>(delay) > static_cast<double>(capacity - 1))
            return false;
        delayed = linear(s.localState, 2, capacity, index, delay);
    }
    return std::isfinite(delayed);
}
template <typename S, Ring kind>
std::size_t ringTransition(const StepT<S>& s, std::span<StateWriteT<S>> writes) noexcept
{
    std::size_t index = 0;
    S input = 0, delayed = 0;
    if (writes.size() < 3 || !ringValues<S, kind>(s, index, input, delayed))
        return rejected;
    const auto value = input + s.parameters[1] * delayed;
    if (!std::isfinite(value))
        return rejected;
    const auto capacity = s.localState.size() - 2;
    cursorWrites(writes, 0, 0, (index + 1) % capacity);
    writes[2] = {2 + index, value};
    return 3;
}
template <typename S, Ring kind>
void ringReadout(const StepT<S>& s, std::span<S> out) noexcept
{
    std::fill(out.begin(), out.end(), S(0));
    std::size_t index = 0;
    S input = 0, delayed = 0;
    if (out.size() != 1 || !ringValues<S, kind>(s, index, input, delayed))
        return;
    if constexpr (kind == Ring::Allpass)
        out[0] = (delayed - s.parameters[1] * (input + s.parameters[1] * delayed)) * s.activation;
    else
        out[0] = delayed * s.activation;
}
template <typename S>
bool firValues(const StepT<S>& s, std::size_t& index, S& input) noexcept
{
    if (!common(s, 0) || s.localState.size() < 3 || s.assets.size() != 1)
        return false;
    const auto capacity = s.localState.size() - 2;
    const auto& coefficients = s.assets[0];
    if (capacity > std::numeric_limits<std::uint32_t>::max() || coefficients.values.empty() ||
        coefficients.values.size() > capacity + 1 ||
        (!coefficients.presence.empty() && coefficients.presence.size() != coefficients.values.size()) ||
        !cursor(s.localState, 0, capacity, index) || !excitation(s, 0, input))
        return false;
    return true;
}
template <typename S>
std::size_t firTransition(const StepT<S>& s, std::span<StateWriteT<S>> writes) noexcept
{
    std::size_t index = 0;
    S input = 0;
    if (writes.size() < 3 || !firValues(s, index, input))
        return rejected;
    cursorWrites(writes, 0, 0, (index + 1) % (s.localState.size() - 2));
    writes[2] = {2 + index, input};
    return 3;
}
template <typename S>
void firReadout(const StepT<S>& s, std::span<S> out) noexcept
{
    std::fill(out.begin(), out.end(), S(0));
    std::size_t index = 0;
    S input = 0;
    if (out.size() != 1 || !firValues(s, index, input))
        return;
    const auto& coefficients = s.assets[0];
    S result = 0;
    for (std::size_t tap = 0; tap < coefficients.values.size(); ++tap)
    {
        if (!coefficients.presence.empty() && coefficients.presence[tap] == 0)
            continue;
        const auto sample = tap == 0 ? input : past(s.localState, 2, s.localState.size() - 2, index, tap);
        result += coefficients.values[tap] * sample;
    }
    out[0] = result * s.activation;
}
template <typename S>
bool fdnValues(const StepT<S>& s, std::array<std::size_t, 8>& positions, std::array<S, 8>& output,
               std::array<S, 8>& input) noexcept
{
    if (!common(s, 24) || s.localState.size() < 32 || s.localState.size() % 8 != 0)
        return false;
    const auto stride = s.localState.size() / 8, capacity = stride - 3;
    if (capacity > std::numeric_limits<std::uint32_t>::max())
        return false;
    for (std::size_t line = 0; line < 8; ++line)
    {
        const auto base = line * stride;
        const auto delay = s.parameters[3 * line], gain = s.parameters[3 * line + 1],
                   damp = s.parameters[3 * line + 2];
        if (!cursor(s.localState, base, capacity, positions[line]) || !excitation(s, line, input[line]) ||
            delay < 1 || static_cast<double>(delay) > static_cast<double>(capacity) ||
            std::floor(delay) != delay || std::abs(gain) >= 1 || damp < 0 || damp >= 1 ||
            !std::isfinite(s.localState[base + 2]))
            return false;
        const auto old =
            past(s.localState, base + 3, capacity, positions[line], static_cast<std::size_t>(delay));
        output[line] = (1 - damp) * old + damp * s.localState[base + 2];
        if (!std::isfinite(output[line]))
            return false;
    }
    return true;
}
template <typename S>
std::size_t fdnTransition(const StepT<S>& s, std::span<StateWriteT<S>> writes) noexcept
{
    std::array<std::size_t, 8> positions{};
    std::array<S, 8> output{}, input{}, mixed{};
    if (writes.size() < 32 || !fdnValues(s, positions, output, input))
        return rejected;
    for (std::size_t i = 0; i < 8; ++i)
        mixed[i] = output[i] * s.parameters[3 * i + 1];
    alfa::dsp::detail::hadamard8(mixed);
    const auto stride = s.localState.size() / 8, capacity = stride - 3;
    for (std::size_t i = 0; i < 8; ++i)
    {
        const auto value = input[i] + mixed[i];
        if (!std::isfinite(value))
            return rejected;
        const auto base = i * stride;
        cursorWrites(writes, 4 * i, base, (positions[i] + 1) % capacity);
        writes[4 * i + 2] = {base + 2, output[i]};
        writes[4 * i + 3] = {base + 3 + positions[i], value};
    }
    return 32;
}
template <typename S>
void fdnReadout(const StepT<S>& s, std::span<S> out) noexcept
{
    std::fill(out.begin(), out.end(), S(0));
    std::array<std::size_t, 8> positions{};
    std::array<S, 8> output{}, input{};
    if (out.size() != 8 || !fdnValues(s, positions, output, input))
        return;
    for (std::size_t i = 0; i < 8; ++i)
        out[i] = output[i] * s.activation;
}
} // namespace
std::span<const kernels::Metadata> catalog() noexcept
{
    return metadata;
}
template <typename S>
bool createPrimitive(std::string_view id, std::size_t count, PrimitiveT<S>& result, std::string& error)
{
    if (id == "time.fdn_hadamard8.v1")
    {
        if (count < 32 || count % 8 != 0 || count / 8 - 3 > std::numeric_limits<std::uint32_t>::max())
        {
            error = "FDN state must be 8*(C+3), C>=1, <=32-bit cursor capacity";
            return false;
        }
        result = {std::string(id), count, 8, nullptr, fdnReadout<S>, 32, fdnTransition<S>};
        return true;
    }
    const auto known = id == "time.ring_delay.v1" || id == "time.ring_delay_cubic.v1" ||
                       id == "time.ring_allpass.v1" || id == "time.causal_fir_ring.v1";
    const std::size_t minimum = id == "time.ring_delay_cubic.v1" ? 6 : id == "time.ring_allpass.v1" ? 4 : 3;
    if (!known)
    {
        error = "Unknown production primitive ID";
        return false;
    }
    if (count < minimum || count - 2 > std::numeric_limits<std::uint32_t>::max())
    {
        error = "Invalid explicit ring capacity for production operation";
        return false;
    }
    result = {std::string(id), count, 1, nullptr, nullptr, 3, nullptr};
    if (id == "time.ring_delay.v1")
    {
        result.readout = ringReadout<S, Ring::Integer>;
        result.sparseTransition = ringTransition<S, Ring::Integer>;
    }
    else if (id == "time.ring_delay_cubic.v1")
    {
        result.readout = ringReadout<S, Ring::Cubic>;
        result.sparseTransition = ringTransition<S, Ring::Cubic>;
    }
    else if (id == "time.ring_allpass.v1")
    {
        result.readout = ringReadout<S, Ring::Allpass>;
        result.sparseTransition = ringTransition<S, Ring::Allpass>;
    }
    else
    {
        result.readout = firReadout<S>;
        result.sparseTransition = firTransition<S>;
    }
    return true;
}
template bool createPrimitive<float>(std::string_view, std::size_t, PrimitiveT<float>&, std::string&);
template bool createPrimitive<double>(std::string_view, std::size_t, PrimitiveT<double>&, std::string&);
} // namespace alfa::tensor::production
