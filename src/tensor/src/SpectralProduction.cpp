#include "alfa/tensor/SpectralProduction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace alfa::tensor::spectralProduction
{
namespace
{
constexpr auto reject = std::numeric_limits<std::size_t>::max();
constexpr kernels::Parameter hopParameter[]{
    {"hop", "scheduled updates", "integer 1..N; changing hop preserves capture history", true, true, true}};
constexpr std::array<kernels::Parameter, 0> none{};
const std::array<kernels::Metadata, 2> metadata{
    {{"spectral.stft_analysis.v1", "audio[0]; asset0 analysis window N",
      "full interleaved complex spectrum[2*N], then binary frame-valid flag; zero between frames",
      hopParameter, "cursor low16/high16, hop phase low16/high16, N input ring; size=N+4",
      "O(1) capture, O(N) output clearing, O(N log N) each frame; 5 journal writes; output buffer FFT "
      "scratch",
      "explicit instance clock; hop counts its updates", "budget-limited independent instances"},
     {"spectral.stft_synthesis.v1",
      "audio[2*N] complex frame plus binary flag; asset0 synthesis window N; asset1 COLA reciprocal N",
      "old overlap-add cursor scalar; new frame starts next update; inverse FFT includes 1/N", none,
      "cursor low16/high16, N+1 overlap-add ring; size=N+3",
      "O(1) emission, O(N log N) on valid frame; N+3 journal entries and 2*N typed scratch values",
      "explicit instance clock, supplied frames determine hop", "budget-limited independent instances"}}};
bool power(std::size_t n) noexcept
{
    return n >= 2 && !(n & (n - 1));
}
template <class S>
bool limbs(std::span<const S> state, std::size_t offset, std::size_t bound, std::size_t& value) noexcept
{
    const S a = state[offset], b = state[offset + 1];
    if (!std::isfinite(a) || !std::isfinite(b) || a < 0 || b < 0 || a > 65535 || b > 65535 ||
        std::floor(a) != a || std::floor(b) != b)
        return false;
    value = static_cast<std::size_t>(a) + 65536 * static_cast<std::size_t>(b);
    return value < bound;
}
template <class S>
void put(std::span<StateWriteT<S>> j, std::size_t pos, std::size_t offset, std::size_t value) noexcept
{
    j[pos] = {offset, static_cast<S>(value % 65536)};
    j[pos + 1] = {offset + 1, static_cast<S>(value / 65536)};
}
template <class S>
S asset(const StepT<S>& s, std::size_t a, std::size_t i) noexcept
{
    const auto& v = s.assets[a];
    return !v.presence.empty() && !v.presence[i] ? S(0) : v.values[i];
}
template <class S>
bool assets(const StepT<S>& s, std::size_t count, std::size_t n) noexcept
{
    if (s.assets.size() != count)
        return false;
    for (const auto& v : s.assets)
    {
        if (v.values.size() != n || (!v.presence.empty() && v.presence.size() != n))
            return false;
    }
    return true;
}
template <class S>
bool base(const StepT<S>& s) noexcept
{
    return s.clock.periodSamples > 0 && std::isfinite(s.sampleRate) && s.sampleRate > 0 &&
           std::isfinite(s.activation) && s.activation >= 0 &&
           (s.inputs.audio.presence.empty() ||
            s.inputs.audio.presence.size() == s.inputs.audio.values.size());
}
template <class S>
S input(const StepT<S>& s, std::size_t i) noexcept
{
    const auto& v = s.inputs.audio;
    return i >= v.values.size() || (!v.presence.empty() && !v.presence[i]) ? S(0) : v.values[i];
}
// Typed radix-2 butterfly: same forward sign and inverse normalization as existing
// tensor FFT; legacy pffft inverse instead leaves N normalization to its window.
// Accessor permits a preallocated typed workspace or the output frame buffer.
template <class S, class Access>
void fft(std::size_t n, bool inverse, Access z) noexcept
{
    for (std::size_t i = 1, j = 0; i < n; ++i)
    {
        auto bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
        {
            std::swap(z(2 * i), z(2 * j));
            std::swap(z(2 * i + 1), z(2 * j + 1));
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
                const S vr = z(b) * wr - z(b + 1) * wi, vi = z(b) * wi + z(b + 1) * wr, ur = z(a),
                        ui = z(a + 1);
                z(a) = ur + vr;
                z(a + 1) = ui + vi;
                z(b) = ur - vr;
                z(b + 1) = ui - vi;
                const S next = wr * cr - wi * ci;
                wi = wr * ci + wi * cr;
                wr = next;
            }
        }
        if (len == n)
            break;
    }
    if (inverse)
        for (std::size_t i = 0; i < 2 * n; ++i)
            z(i) /= static_cast<S>(n);
}
template <class S>
bool analysisValid(const StepT<S>& s, std::size_t& n, std::size_t& cursor, std::size_t& phase,
                   std::size_t& hop) noexcept
{
    if (!base(s) || s.localState.size() < 6 || s.parameters.size() != 1)
        return false;
    n = s.localState.size() - 4;
    const S h = s.parameters[0];
    if (!power(n) || !std::isfinite(h) || h < 1 || h > static_cast<S>(n) || std::floor(h) != h ||
        !assets(s, 1, n))
        return false;
    hop = static_cast<std::size_t>(h);
    return limbs(s.localState, 0, n, cursor) && limbs(s.localState, 2, n, phase) &&
           std::isfinite(input(s, 0));
}
template <class S>
void analysisRead(const StepT<S>& s, std::span<S> out) noexcept
{
    std::fill(out.begin(), out.end(), S(0));
    std::size_t n = 0, c = 0, p = 0, h = 0;
    if (!analysisValid(s, n, c, p, h) || out.size() != 2 * n + 1)
    {
        if (!out.empty())
            out[0] = std::numeric_limits<S>::quiet_NaN();
        return;
    }
    if (p + 1 < h)
        return;
    for (std::size_t i = 0; i < n; ++i)
    {
        const S x = i + 1 == n ? input(s, 0) : s.localState[4 + (c + 1 + i) % n];
        out[2 * i] = x * asset(s, 0, i);
    }
    fft<S>(n, false, [&](std::size_t i) -> S& { return out[i]; });
    for (std::size_t i = 0; i < 2 * n; ++i)
        out[i] *= s.activation;
    out[2 * n] = 1;
}
template <class S>
std::size_t analysisUpdate(const StepT<S>& s, std::span<StateWriteT<S>> j) noexcept
{
    std::size_t n = 0, c = 0, p = 0, h = 0;
    if (j.size() < 5 || !analysisValid(s, n, c, p, h))
        return reject;
    put(j, 0, 0, (c + 1) % n);
    put(j, 2, 2, p + 1 >= h ? 0 : p + 1);
    j[4] = {4 + c, input(s, 0)};
    return 5;
}
template <class S>
bool synthesisValid(const StepT<S>& s, std::size_t& n, std::size_t& c, bool& frame) noexcept
{
    if (!base(s) || s.localState.size() < 5 || !s.parameters.empty())
        return false;
    n = s.localState.size() - 3;
    if (!power(n) || !assets(s, 2, n) || !limbs(s.localState, 0, n + 1, c))
        return false;
    const S flag = input(s, 2 * n);
    if (flag != 0 && flag != 1)
        return false;
    frame = flag == 1;
    if (frame)
    {
        if (s.inputs.audio.values.size() != 2 * n + 1)
            return false;
        for (std::size_t i = 0; i < 2 * n; ++i)
            if (!std::isfinite(input(s, i)))
                return false;
    }
    return std::isfinite(s.localState[2 + c]);
}
template <class S>
void synthesisRead(const StepT<S>& s, std::span<S> out) noexcept
{
    std::size_t n = 0, c = 0;
    bool frame = false;
    if (out.size() != 1)
        return;
    out[0] = synthesisValid(s, n, c, frame) ? s.localState[2 + c] * s.activation
                                            : std::numeric_limits<S>::quiet_NaN();
}
template <class S>
std::size_t synthesisUpdate(const StepT<S>& s, std::span<StateWriteT<S>> j) noexcept
{
    std::size_t n = 0, c = 0;
    bool frame = false;
    if (!synthesisValid(s, n, c, frame) || j.size() < n + 3)
        return reject;
    if (!frame)
    {
        put(j, 0, 0, (c + 1) % (n + 1));
        j[2] = {2 + c, 0};
        return 3;
    }
    if (s.scratch.size() < 2 * n)
        return reject;
    for (std::size_t i = 0; i < 2 * n; ++i)
        s.scratch[i] = input(s, i);
    fft<S>(n, true, [&](std::size_t i) -> S& { return s.scratch[i]; });
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto index = 2 + (c + 1 + i) % (n + 1);
        j[i] = {index, s.localState[index] + s.scratch[2 * i] * asset(s, 0, i) * asset(s, 1, i)};
    }
    j[n] = {2 + c, 0};
    put(j, n + 1, 0, (c + 1) % (n + 1));
    return n + 3;
}
} // namespace
std::span<const kernels::Metadata> catalog() noexcept
{
    return metadata;
}
template <class S>
bool createPrimitive(std::string_view id, std::size_t size, PrimitiveT<S>& r, std::string& error)
{
    const bool a = id == "spectral.stft_analysis.v1", b = id == "spectral.stft_synthesis.v1";
    const auto overhead = a ? 4u : 3u;
    if ((!a && !b) || size <= overhead || !power(size - overhead) ||
        size - overhead > std::numeric_limits<std::uint32_t>::max() / 2)
    {
        error = "unknown STFT operation or state size must encode power-of-two N >=2";
        return false;
    }
    const auto n = size - overhead;
    r = {std::string(id),
         size,
         a ? 2 * n + 1 : 1,
         nullptr,
         a ? analysisRead<S> : synthesisRead<S>,
         a ? 5 : n + 3,
         a ? analysisUpdate<S> : synthesisUpdate<S>,
         a ? 0 : 2 * n};
    error.clear();
    return true;
}
template bool createPrimitive<float>(std::string_view, std::size_t, PrimitiveT<float>&, std::string&);
template bool createPrimitive<double>(std::string_view, std::size_t, PrimitiveT<double>&, std::string&);
} // namespace alfa::tensor::spectralProduction
