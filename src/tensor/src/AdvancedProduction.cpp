#include "alfa/tensor/AdvancedProduction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace alfa::tensor::advanced
{
namespace
{
constexpr auto rejected = std::numeric_limits<std::size_t>::max();
constexpr kernels::Parameter blockParams[]{
    {"partition size", "updates", "power of two 8..4096; immutable layout", false, true, false}};
constexpr kernels::Parameter sampleParams[]{
    {"speed", "frames/sec", "finite; ratio abs(speed)/local rate <=32", true, true, true},
    {"amplitude", "linear", "finite", true, true, true},
    {"half width", "base taps", "integer 8..128; actual support expands with downsampling ratio", true, true,
     true}};
constexpr kernels::Parameter modalParams[]{
    {"frequency[i]", "Hz", "0..local Nyquist; triples frequency/decay/gain", true, true, true},
    {"decay[i]", "seconds", "positive T60", true, true, true},
    {"gain[i]", "linear", "finite excitation gain", true, true, true}};
constexpr kernels::Parameter waveguideParams[]{
    {"frequency", "Hz", "positive; tuned roundtrip must fit ring capacity", true, true, true},
    {"feedback", "linear", "abs(feedback)<1", true, true, true},
    {"damping", "pole", "0<=damping<1", true, true, true},
    {"dispersion", "allpass coefficient", "abs(coefficient)<1", true, true, true}};
constexpr kernels::Parameter vocoderParams[]{{"analysis hop", "updates", "positive <=N", true, true, true},
                                             {"synthesis hop", "updates", "positive <=N", true, true, true},
                                             {"pitch ratio", "ratio", "0.0625..16", true, true, true}};
const std::array<kernels::Metadata, 5> metadata{
    {{"time.partitioned_convolution.v1",
      "audio[0]; asset0 real IR, prepared off-thread to complex partitions",
      "uniform overlap-add convolution, B updates algorithm latency", blockParams,
      "phase low16/high16, FDL cursor low16/high16, input B, overlap B, queued output B, P complex FFT "
      "frames 4B each",
      "O(P*B + B log B) per B updates; O(1) other updates; sparse frame journal",
      "independent explicit clock", "unbounded instance count within budget"},
     {"pitchtime.asset_resample_antialias.v1",
      "asset0 mono sample; rate-aware windowed sinc; zero outside asset", "ratio-dependent low-pass playback",
      sampleParams, "fractional asset cursor", "O(halfWidth*max(1,ratio)), no allocation",
      "independent explicit clock", "budgeted independent instances"},
     {"physical.modal_rotation.v1", "audio[0] excitation; parameter triples per mode",
      "sum of real modal coordinates", modalParams, "two real rotation coordinates per mode",
      "O(mode count), typed long-decay recurrence", "independent explicit clock",
      "mode count selected by state size, no fixed bank"},
     {"physical.tuned_waveguide.v1", "audio[0] excitation", "fractional tuned damped dispersive loop",
      waveguideParams,
      "cursor low16/high16, LP memory, fractional x/y, dispersion x/y, C ring samples; size=C+7",
      "O(1), 8 sparse writes; filter/allpass phase delay compensated at requested frequency",
      "independent explicit clock", "explicit ring capacity, arbitrary instances"},
     {"pitchtime.phase_vocoder_locked.v1", "audio[2N] Hermitian complex FFT plus frame-valid flag",
      "complex output frame plus one-update delayed valid flag", vocoderParams,
      "primed flag, pending frame flag, analysis phases N, synthesis phases N, output complex 2N; size=4N+2",
      "O(N) identity peak phase locking and bin redistribution per valid frame; one journal write otherwise; "
      "6N scratch",
      "sample clock or explicit producer cadence; frame flags gate phase updates",
      "explicit frame size and budgets"}}};
bool power(std::size_t n) noexcept
{
    return n >= 8 && n <= 4096 && !(n & (n - 1));
}
template <class S>
void fft(std::span<S> z, bool inverse) noexcept
{
    const auto n = z.size() / 2;
    for (std::size_t i = 1, j = 0; i < n; ++i)
    {
        auto bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
        {
            std::swap(z[2 * i], z[2 * j]);
            std::swap(z[2 * i + 1], z[2 * j + 1]);
        }
    }
    for (std::size_t len = 2; len <= n; len *= 2)
    {
        const S angle = (inverse ? S(2) : S(-2)) * std::numbers::pi_v<S> / static_cast<S>(len);
        const S cr = std::cos(angle), ci = std::sin(angle);
        for (std::size_t start = 0; start < n; start += len)
        {
            S wr = 1, wi = 0;
            for (std::size_t k = 0; k < len / 2; ++k)
            {
                const auto a = 2 * (start + k), b = 2 * (start + k + len / 2);
                const S vr = z[b] * wr - z[b + 1] * wi, vi = z[b] * wi + z[b + 1] * wr, ur = z[a],
                        ui = z[a + 1];
                z[a] = ur + vr;
                z[a + 1] = ui + vi;
                z[b] = ur - vr;
                z[b + 1] = ui - vi;
                const S next = wr * cr - wi * ci;
                wi = wr * ci + wi * cr;
                wr = next;
            }
        }
        if (len == n)
            break;
    }
    if (inverse)
        for (auto& v : z)
            v /= static_cast<S>(n);
}
template <class S>
S input(const StepT<S>& s) noexcept
{
    return s.inputs.audio.values.empty() || (!s.inputs.audio.presence.empty() && !s.inputs.audio.presence[0])
               ? S(0)
               : s.inputs.audio.values[0];
}
template <class S>
bool index(std::span<const S> s, std::size_t off, std::size_t bound, std::size_t& v) noexcept
{
    if (s[off] < 0 || s[off] > 65535 || s[off + 1] < 0 || s[off + 1] > 65535 ||
        std::floor(s[off]) != s[off] || std::floor(s[off + 1]) != s[off + 1])
        return false;
    v = static_cast<std::size_t>(s[off]) + 65536 * static_cast<std::size_t>(s[off + 1]);
    return v < bound;
}
template <class S>
bool convolution(const StepT<S>& s, std::size_t& b, std::size_t& p, std::size_t& phase,
                 std::size_t& cursor) noexcept
{
    if (s.parameters.size() != 1 || !std::isfinite(s.parameters[0]) || s.parameters[0] < 8 ||
        s.parameters[0] > 4096 || std::floor(s.parameters[0]) != s.parameters[0])
        return false;
    b = static_cast<std::size_t>(s.parameters[0]);
    if (!power(b) || s.localState.size() < 4 + 7 * b || (s.localState.size() - 4 - 3 * b) % (4 * b))
        return false;
    p = (s.localState.size() - 4 - 3 * b) / (4 * b);
    return p <= std::numeric_limits<std::uint32_t>::max() && s.assets.size() == 1 &&
           s.assets[0].values.size() == 4 * b * p && index(s.localState, 0, b, phase) &&
           index(s.localState, 2, p, cursor) && std::isfinite(input(s));
}
template <class S>
void convolutionOutput(const StepT<S>& s, std::span<S> out) noexcept
{
    std::size_t b = 0, p = 0, phase = 0, cursor = 0;
    out[0] = convolution(s, b, p, phase, cursor) ? s.localState[4 + 2 * b + phase] * s.activation : S(0);
}
template <class S>
std::size_t convolutionUpdate(const StepT<S>& s, std::span<StateWriteT<S>> j) noexcept
{
    std::size_t b = 0, p = 0, phase = 0, cursor = 0;
    if (!convolution(s, b, p, phase, cursor) || j.size() < 6 * b + 5 || s.scratch.size() < 8 * b)
        return rejected;
    std::size_t used = 0;
    const auto put = [&](std::size_t off, S v)
    {
        j[used++] = {off, v};
    };
    const auto nextPhase = (phase + 1) % b;
    put(0, static_cast<S>(nextPhase % 65536));
    put(1, static_cast<S>(nextPhase / 65536));
    put(4 + phase, input(s));
    if (nextPhase)
        return used;
    auto x = s.scratch.first(4 * b), y = s.scratch.subspan(4 * b, 4 * b);
    std::fill(x.begin(), x.end(), S(0));
    std::fill(y.begin(), y.end(), S(0));
    for (std::size_t i = 0; i < b; ++i)
        x[2 * i] = i == phase ? input(s) : s.localState[4 + i];
    fft(x, false);
    const auto history = 4 + 3 * b;
    for (std::size_t part = 0; part < p; ++part)
    {
        const auto slot = (cursor + p - part) % p;
        for (std::size_t k = 0; k < 4 * b; k += 2)
        {
            const S xr = part == 0 ? x[k] : s.localState[history + slot * 4 * b + k];
            const S xi = part == 0 ? x[k + 1] : s.localState[history + slot * 4 * b + k + 1];
            const auto tap = part * 4 * b + k;
            const S hr = s.assets[0].values[tap], hi = s.assets[0].values[tap + 1];
            y[k] += xr * hr - xi * hi;
            y[k + 1] += xr * hi + xi * hr;
        }
    }
    for (std::size_t k = 0; k < 4 * b; ++k)
        put(history + cursor * 4 * b + k, x[k]);
    fft(y, true);
    for (std::size_t i = 0; i < b; ++i)
    {
        put(4 + b + i, y[2 * (b + i)]);
        put(4 + 2 * b + i, y[2 * i] + s.localState[4 + b + i]);
    }
    cursor = (cursor + 1) % p;
    put(2, static_cast<S>(cursor % 65536));
    put(3, static_cast<S>(cursor / 65536));
    return used;
}
template <class S>
bool sampleValid(const StepT<S>& s, S& increment, std::size_t& radius) noexcept
{
    if (s.parameters.size() != 3 || s.assets.size() != 1 || s.localState.size() != 1 ||
        s.clock.periodSamples == 0)
        return false;
    const S rate = static_cast<S>(s.sampleRate / static_cast<double>(s.clock.periodSamples));
    increment = s.parameters[0] / rate;
    const S width = s.parameters[2];
    if (!std::isfinite(increment) || std::abs(increment) > 32 || !std::isfinite(width) || width < 8 ||
        width > 128 || std::floor(width) != width)
        return false;
    radius = static_cast<std::size_t>(std::ceil(width * std::max(S(1), std::abs(increment))));
    return std::isfinite(s.localState[0]) && std::isfinite(s.parameters[1]);
}
template <class S>
void sampleUpdate(const StepT<S>& s, std::span<S> next) noexcept
{
    S increment = 0;
    std::size_t radius = 0;
    next[0] =
        sampleValid(s, increment, radius) ? s.localState[0] + increment : std::numeric_limits<S>::quiet_NaN();
}
template <class S>
void sampleOutput(const StepT<S>& s, std::span<S> out) noexcept
{
    out[0] = 0;
    S increment = 0;
    std::size_t radius = 0;
    if (!sampleValid(s, increment, radius))
        return;
    const auto position = s.localState[0];
    const auto& asset = s.assets[0];
    if (position < 0 || position >= static_cast<S>(asset.values.size()))
        return;
    const S cutoff = S(1) / std::max(S(1), std::abs(increment));
    const auto center = static_cast<std::int64_t>(std::floor(position));
    S total = 0, weight = 0;
    for (auto k = -static_cast<std::int64_t>(radius); k <= static_cast<std::int64_t>(radius); ++k)
    {
        const auto i = center + k;
        const S distance = static_cast<S>(i) - position;
        const S a = std::numbers::pi_v<S> * distance * cutoff;
        const S sinc = std::abs(a) < S(1e-6) ? S(1) : std::sin(a) / a;
        const S w = cutoff * sinc *
                    (S(.5) + S(.5) * std::cos(std::numbers::pi_v<S> * distance / static_cast<S>(radius)));
        weight += w;
        if (i >= 0 && static_cast<std::size_t>(i) < asset.values.size() &&
            (asset.presence.empty() || asset.presence[static_cast<std::size_t>(i)]))
            total += w * asset.values[static_cast<std::size_t>(i)];
    }
    if (std::abs(weight) > S(1e-8))
        out[0] = total / weight * s.parameters[1] * s.activation;
}
template <class S>
void modalUpdate(const StepT<S>& s, std::span<S> next) noexcept
{
    const auto modes = next.size() / 2;
    if (s.parameters.size() != 3 * modes || s.clock.periodSamples == 0)
    {
        std::fill(next.begin(), next.end(), std::numeric_limits<S>::quiet_NaN());
        return;
    }
    const S rate = static_cast<S>(s.sampleRate / static_cast<double>(s.clock.periodSamples));
    for (std::size_t i = 0; i < modes; ++i)
    {
        const S f = s.parameters[3 * i], decay = s.parameters[3 * i + 1], gain = s.parameters[3 * i + 2];
        if (!std::isfinite(f) || f < 0 || f >= rate / S(2) || !std::isfinite(decay) || decay <= 0 ||
            !std::isfinite(gain))
        {
            next[2 * i] = std::numeric_limits<S>::quiet_NaN();
            return;
        }
        const S angle = S(2) * std::numbers::pi_v<S> * f / rate,
                damp = std::exp(std::log(S(.001)) / (decay * rate));
        const S c = damp * std::cos(angle), d = damp * std::sin(angle);
        next[2 * i] = c * s.localState[2 * i] - d * s.localState[2 * i + 1] + gain * input(s);
        next[2 * i + 1] = d * s.localState[2 * i] + c * s.localState[2 * i + 1];
    }
}
template <class S>
void modalOutput(const StepT<S>& s, std::span<S> out) noexcept
{
    S value = 0;
    for (std::size_t i = 0; i < s.localState.size(); i += 2)
        value += s.localState[i];
    out[0] = value * s.activation;
}
template <class S>
bool waveguide(const StepT<S>& s, std::size_t& cursor, S& low, S& fractional, S& dispersed) noexcept
{
    if (s.localState.size() < 9 || s.parameters.size() != 4 || s.clock.periodSamples == 0)
        return false;
    const auto capacity = s.localState.size() - 7;
    const S rate = static_cast<S>(s.sampleRate / static_cast<double>(s.clock.periodSamples));
    const S f = s.parameters[0], gain = s.parameters[1], damp = s.parameters[2], dispersion = s.parameters[3];
    if (!std::isfinite(f) || f <= 0 || f >= rate / S(2) || !std::isfinite(gain) || std::abs(gain) >= 1 ||
        !std::isfinite(damp) || damp < 0 || damp >= 1 || !std::isfinite(dispersion) ||
        std::abs(dispersion) >= 1 || capacity > std::numeric_limits<uint32_t>::max() ||
        !index(s.localState, 0, capacity, cursor))
        return false;
    const S w = S(2) * std::numbers::pi_v<S> * f / rate;
    const S lpDelay = std::atan2(damp * std::sin(w), S(1) - damp * std::cos(w)) / w;
    const S dispersionDelay =
        S(1) - S(2) * std::atan2(dispersion * std::sin(w), S(1) + dispersion * std::cos(w)) / w;
    const S delay = rate / f - lpDelay - dispersionDelay;
    if (!std::isfinite(delay) || delay < 1 || delay > static_cast<S>(capacity))
        return false;
    const auto integer = static_cast<size_t>(std::floor(delay));
    const S part = delay - static_cast<S>(integer);
    const auto position = cursor >= integer ? cursor - integer : capacity - (integer - cursor);
    low = (S(1) - damp) * s.localState[7 + position] + damp * s.localState[2];
    const S ratio = std::tan(w * part / S(2)) / std::tan(w / S(2));
    const S coefficient = (S(1) - ratio) / (S(1) + ratio);
    fractional = coefficient * low + s.localState[3] - coefficient * s.localState[4];
    dispersed = dispersion * fractional + s.localState[5] - dispersion * s.localState[6];
    return std::isfinite(input(s)) && std::isfinite(dispersed);
}
template <class S>
void waveguideOutput(const StepT<S>& s, std::span<S> out) noexcept
{
    size_t cursor = 0;
    S low = 0, fractional = 0, dispersed = 0;
    out[0] = waveguide(s, cursor, low, fractional, dispersed) ? dispersed * s.activation : S(0);
}
template <class S>
size_t waveguideUpdate(const StepT<S>& s, std::span<StateWriteT<S>> j) noexcept
{
    size_t cursor = 0;
    S low = 0, fractional = 0, dispersed = 0;
    if (j.size() < 8 || !waveguide(s, cursor, low, fractional, dispersed))
        return rejected;
    const auto next = (cursor + 1) % (s.localState.size() - 7);
    j[0] = {0, static_cast<S>(next % 65536)};
    j[1] = {1, static_cast<S>(next / 65536)};
    j[2] = {2, low};
    j[3] = {3, low};
    j[4] = {4, fractional};
    j[5] = {5, fractional};
    j[6] = {6, dispersed};
    j[7] = {7 + cursor, input(s) + s.parameters[1] * dispersed};
    return 8;
}
template <class S>
bool vocoderValid(const StepT<S>& s, size_t& n, bool& frame) noexcept
{
    if (s.localState.size() < 34 || (s.localState.size() - 2) % 4 || s.parameters.size() != 3)
        return false;
    n = (s.localState.size() - 2) / 4;
    if (!power(n) || s.scratch.size() < 6 * n || s.inputs.audio.values.size() < 2 * n + 1)
        return false;
    for (size_t i = 0; i < 2; ++i)
        if (!std::isfinite(s.parameters[i]) || s.parameters[i] <= 0 || s.parameters[i] > static_cast<S>(n))
            return false;
    if (!std::isfinite(s.parameters[2]) || s.parameters[2] < S(.0625) || s.parameters[2] > S(16))
        return false;
    const auto flag = s.inputs.audio.values[2 * n];
    if (flag != S(0) && flag != S(1))
        return false;
    frame = flag == S(1) && (s.inputs.audio.presence.empty() || s.inputs.audio.presence[2 * n]);
    return true;
}
template <class S>
void vocoderOutput(const StepT<S>& s, std::span<S> out) noexcept
{
    std::fill(out.begin(), out.end(), S(0));
    size_t n = 0;
    bool frame = false;
    if (!vocoderValid(s, n, frame) || s.localState[1] != S(1))
        return;
    for (size_t i = 0; i < 2 * n; ++i)
        out[i] = s.localState[2 + 2 * n + i] * s.activation;
    out[2 * n] = 1;
}
template <class S>
size_t vocoderUpdate(const StepT<S>& s, std::span<StateWriteT<S>> j) noexcept
{
    size_t n = 0;
    bool frame = false;
    if (!vocoderValid(s, n, frame) || j.size() < 4 * n + 2)
        return rejected;
    if (!frame)
    {
        j[0] = {1, 0};
        return 1;
    }
    auto phase = s.scratch.first(n), magnitude = s.scratch.subspan(n, n),
         synthesis = s.scratch.subspan(2 * n, n), owner = s.scratch.subspan(3 * n, n),
         spectrum = s.scratch.subspan(4 * n, 2 * n);
    std::fill(spectrum.begin(), spectrum.end(), S(0));
    const S tau = S(2) * std::numbers::pi_v<S>;
    for (size_t k = 0; k < n; ++k)
    {
        const S real = s.inputs.audio.values[2 * k], imag = s.inputs.audio.values[2 * k + 1];
        phase[k] = std::atan2(imag, real);
        magnitude[k] = std::hypot(real, imag);
        const S signedBin = k <= n / 2 ? static_cast<S>(k) : static_cast<S>(k) - static_cast<S>(n);
        const S expected = tau * signedBin * s.parameters[0] / static_cast<S>(n);
        const S advance = (expected + std::remainder(phase[k] - s.localState[2 + k] - expected, tau)) *
                          s.parameters[1] / s.parameters[0] * s.parameters[2];
        synthesis[k] =
            s.localState[0] == S(0) ? phase[k] : std::remainder(s.localState[2 + n + k] + advance, tau);
    }
    size_t last = 0;
    const auto peak = [&](size_t k)
    {
        return (k == 0 || magnitude[k] >= magnitude[k - 1]) &&
               (k == n / 2 || magnitude[k] > magnitude[k + 1]);
    };
    for (size_t k = 0; k <= n / 2; ++k)
    {
        if (peak(k))
            last = k;
        owner[k] = static_cast<S>(last);
    }
    last = n / 2;
    for (size_t k = n / 2 + 1; k-- > 0;)
    {
        if (peak(k))
            last = k;
        const auto left = static_cast<size_t>(owner[k]);
        if (last - k < k - left)
            owner[k] = static_cast<S>(last);
    }
    for (size_t k = 0; k <= n / 2; ++k)
    {
        const auto anchor = static_cast<size_t>(owner[k]);
        const S locked = synthesis[anchor] + std::remainder(phase[k] - phase[anchor], tau);
        const S position = static_cast<S>(k) * s.parameters[2];
        if (position > static_cast<S>(n / 2))
            continue;
        const auto first = static_cast<size_t>(std::floor(position));
        const S fraction = position - static_cast<S>(first);
        for (size_t part = 0; part < 2; ++part)
        {
            const auto bin = first + part;
            if (bin > n / 2)
                continue;
            const S scale = magnitude[k] * (part ? fraction : S(1) - fraction);
            spectrum[2 * bin] += scale * std::cos(locked);
            if (bin != 0 && bin != n / 2)
                spectrum[2 * bin + 1] += scale * std::sin(locked);
        }
    }
    for (size_t k = 1; k < n / 2; ++k)
    {
        spectrum[2 * (n - k)] = spectrum[2 * k];
        spectrum[2 * (n - k) + 1] = -spectrum[2 * k + 1];
    }
    j[0] = {0, 1};
    j[1] = {1, 1};
    for (size_t k = 0; k < n; ++k)
    {
        j[2 + k] = {2 + k, phase[k]};
        j[2 + n + k] = {2 + n + k, synthesis[k]};
    }
    for (size_t k = 0; k < 2 * n; ++k)
        j[2 + 2 * n + k] = {2 + 2 * n + k, spectrum[k]};
    return 4 * n + 2;
}
} // namespace
std::span<const kernels::Metadata> catalog() noexcept
{
    return metadata;
}
template <class S>
PrimitiveT<S> partitionedPrimitive(std::size_t b, std::size_t p)
{
    if (!power(b) || !p || p > (std::numeric_limits<std::size_t>::max() - 4 - 3 * b) / (4 * b))
        throw std::invalid_argument("Invalid partitioned convolution layout");
    return {"time.partitioned_convolution.v1",
            4 + 3 * b + 4 * b * p,
            1,
            nullptr,
            convolutionOutput<S>,
            6 * b + 5,
            convolutionUpdate<S>,
            8 * b};
}
template <class S>
TensorT<S> partitionIR(const TensorT<S>& impulse, std::size_t b)
{
    if (!power(b) || impulse.values.empty())
        throw std::invalid_argument("Nonempty impulse and power-of-two partition size 8..4096 required");
    const auto p = (impulse.values.size() - 1) / b + 1;
    auto primitive = partitionedPrimitive<S>(b, p);
    (void)primitive;
    TensorT<S> result{{"partition", "bin", "complex"}, {p, 2 * b, 2}, std::vector<S>(4 * b * p), {}};
    for (std::size_t part = 0; part < p; ++part)
    {
        auto frame = std::span<S>(result.values).subspan(part * 4 * b, 4 * b);
        for (std::size_t k = 0; k < b && part * b + k < impulse.values.size(); ++k)
            frame[2 * k] = !impulse.presence.empty() && !impulse.presence[part * b + k]
                               ? S(0)
                               : impulse.values[part * b + k];
        fft(frame, false);
    }
    return result;
}
template <class S>
bool createPrimitive(std::string_view id, std::size_t n, PrimitiveT<S>& out, std::string& error)
{
    if (id == metadata[0].id && n >= 60 && n <= std::numeric_limits<std::size_t>::max() / 2)
        out = {std::string(id), n, 1, nullptr, convolutionOutput<S>, n, convolutionUpdate<S>, 2 * n};
    else if (id == metadata[1].id && n == 1)
        out = {std::string(id), n, 1, sampleUpdate<S>, sampleOutput<S>};
    else if (id == metadata[2].id && n >= 2 && n % 2 == 0)
        out = {std::string(id), n, 1, modalUpdate<S>, modalOutput<S>};
    else if (id == metadata[3].id && n >= 9)
        out = {std::string(id), n, 1, nullptr, waveguideOutput<S>, 8, waveguideUpdate<S>};
    else if (id == metadata[4].id && n >= 34 && (n - 2) % 4 == 0 && power((n - 2) / 4))
        out = {std::string(id),  n, (n - 2) / 2 + 1,  nullptr,
               vocoderOutput<S>, n, vocoderUpdate<S>, 6 * ((n - 2) / 4)};
    else
    {
        error = "Unknown advanced operation or invalid state size";
        return false;
    }
    error.clear();
    return true;
}
template bool createPrimitive<float>(std::string_view, std::size_t, PrimitiveT<float>&, std::string&);
template bool createPrimitive<double>(std::string_view, std::size_t, PrimitiveT<double>&, std::string&);
template TensorT<float> partitionIR(const TensorT<float>&, std::size_t);
template TensorT<double> partitionIR(const TensorT<double>&, std::size_t);
template PrimitiveT<float> partitionedPrimitive(std::size_t, std::size_t);
template PrimitiveT<double> partitionedPrimitive(std::size_t, std::size_t);
} // namespace alfa::tensor::advanced
