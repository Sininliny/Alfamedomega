#include "alfa/tensor/NonlinearProduction.h"
#include "alfa/dsp/detail/NonlinearHalfbandMath.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace alfa::tensor::nonlinear
{
namespace
{
constexpr std::size_t rejected = std::numeric_limits<std::size_t>::max();
constexpr std::size_t maxWrites = 15;
enum class Shape { Tanh, Clip, Table };
constexpr kernels::Parameter basicParameters[]{
    {"drive", "linear gain", "finite >=0", true, true, true},
    {"bias", "linear offset", "finite", true, true, true},
    {"output_gain", "linear gain", "finite", true, true, true}};
constexpr kernels::Parameter tableParameters[]{
    basicParameters[0], basicParameters[1], basicParameters[2],
    {"table_min", "input coordinate", "finite <table_max", true, true, true},
    {"table_max", "input coordinate", "finite >table_min", true, true, true}};
constexpr auto rates = "explicit positive clock: filtering measured in scheduled updates; no global rate";
constexpr auto reuse = "independent arbitrary instances within supplied budget";
constexpr auto states = "three rings/stage [cursor,M input history,cursor,M shaped-even history,cursor,M/2 shaped-odd history]; Scalar state only";
constexpr auto assets = "audio[0], absent=0; halfband dense taps asset0 (2x), asset0/1 (4x); table asset follows stages; declared group delay assumes symmetric FIR taps";
const std::array<kernels::Metadata, 9> metadata{{
    {"nonlinear.tanh.v1", "audio[0], absent=0", "gain*tanh(drive*x+bias)*activation", basicParameters, "none", "one tanh", rates, reuse},
    {"nonlinear.clip.v1", "audio[0], absent=0", "gain*clamp(drive*x+bias,-1,1)*activation", basicParameters, "none", "one clamp", rates, reuse},
    {"nonlinear.table.v1", "audio[0]; asset0>=2 uniformly spaced table values", "linear lookup of drive*x+bias, clamped endpoints", tableParameters, "none", "two table reads", rates, reuse},
    {"nonlinear.tanh_polyphase2x.v1", assets, "filtered 2x tanh; linear FIR latency M1-1 updates", basicParameters, states, "two tanh + FIR dots; six sparse writes", rates, reuse},
    {"nonlinear.clip_polyphase2x.v1", assets, "filtered 2x clipping; linear FIR latency M1-1 updates", basicParameters, states, "two clamp + FIR dots; six sparse writes", rates, reuse},
    {"nonlinear.table_polyphase2x.v1", assets, "filtered 2x table lookup; linear FIR latency M1-1 updates", tableParameters, states, "two lookup + FIR dots; six sparse writes", rates, reuse},
    {"nonlinear.tanh_polyphase4x.v1", assets, "filtered 4x tanh; linear FIR latency M1-1+(M2-1)/2 updates", basicParameters, states, "four tanh + cascaded FIR dots; <=15 sparse writes", rates, reuse},
    {"nonlinear.clip_polyphase4x.v1", assets, "filtered 4x clipping; linear FIR latency M1-1+(M2-1)/2 updates", basicParameters, states, "four clamp + cascaded FIR dots; <=15 sparse writes", rates, reuse},
    {"nonlinear.table_polyphase4x.v1", assets, "filtered 4x table lookup; linear FIR latency M1-1+(M2-1)/2 updates", tableParameters, states, "four lookup + cascaded FIR dots; <=15 sparse writes", rates, reuse}}};
bool validM(std::size_t m) noexcept { return m >= 2 && m <= 24 && m % 2 == 0; }
template <class S>
bool assetValue(const TensorViewT<S>& asset, std::size_t i, S& result) noexcept
{
    if (i >= asset.values.size()) return false;
    if (!asset.presence.empty())
    {
        if (asset.presence.size() != asset.values.size() || asset.presence[i] > 1) return false;
        if (asset.presence[i] == 0) { result = S(0); return asset.values[i] == S(0); }
    }
    result = asset.values[i];
    return std::isfinite(result);
}
// Local bounded overlay: oversampled substeps see their own earlier writes, while
// all other runtime primitives still see immutable S_t. Entries remain unique.
template <class S>
struct Overlay
{
    std::span<const S> state;
    std::array<StateWriteT<S>, maxWrites> writes{};
    std::size_t count = 0;
    bool okay = true;
    S get(std::size_t index) noexcept
    {
        for (std::size_t i = 0; i < count; ++i) if (writes[i].element == index) return writes[i].value;
        if (index >= state.size() || !std::isfinite(state[index])) { okay = false; return S(0); }
        return state[index];
    }
    void put(std::size_t index, S value) noexcept
    {
        if (index >= state.size() || !std::isfinite(value)) { okay = false; return; }
        for (std::size_t i = 0; i < count; ++i)
            if (writes[i].element == index) { writes[i].value = value; return; }
        if (count == writes.size()) { okay = false; return; }
        writes[count++] = {index, value};
    }
    std::size_t position(std::size_t base, std::size_t length) noexcept
    {
        const auto v = get(base);
        if (v < S(0) || v >= static_cast<S>(length) || std::floor(v) != v) { okay = false; return 0; }
        return static_cast<std::size_t>(v);
    }
    void push(std::size_t base, std::size_t length, S value) noexcept
    {
        const auto p = position(base, length);
        put(base + 1 + p, value);
        put(base, static_cast<S>((p + 1) % length));
    }
    S recent(std::size_t base, std::size_t length, std::size_t age) noexcept
    {
        const auto p = position(base, length);
        return get(base + 1 + (p + length - 1 - age) % length);
    }
};
template <class S>
std::array<S, 2> up(Overlay<S>& o, std::size_t base, const TensorViewT<S>& h, S x) noexcept
{
    const auto m = h.values.size();
    o.push(base, m, x);
    S sum = 0;
    for (std::size_t j = 0; j < m; ++j)
    {
        S tap = 0;
        if (!assetValue(h, j, tap)) o.okay = false;
        sum += tap * o.recent(base, m, j);
    }
    return {S(2) * sum, o.recent(base, m, m / 2 - 1)};
}
template <class S>
S down(Overlay<S>& o, std::size_t base, const TensorViewT<S>& h, S even, S odd) noexcept
{
    const auto m = h.values.size(), oddBase = base + m + 1;
    o.push(base, m, even);
    S sum = 0;
    for (std::size_t j = 0; j < m; ++j)
    {
        S tap = 0;
        if (!assetValue(h, j, tap)) o.okay = false;
        sum += tap * o.recent(base, m, j);
    }
    sum += S(0.5) * o.recent(oddBase, m / 2, m / 2 - 1);
    o.push(oddBase, m / 2, odd);
    return sum;
}
template <class S, Shape shape>
S transform(const StepT<S>& s, S x, std::size_t tableAsset, bool& okay) noexcept
{
    x = s.parameters[0] * x + s.parameters[1];
    if (!std::isfinite(x)) { okay = false; return S(0); }
    if constexpr (shape == Shape::Tanh) return std::tanh(x);
    if constexpr (shape == Shape::Clip) return std::clamp(x, S(-1), S(1));
    if constexpr (shape == Shape::Table)
    {
        if (tableAsset >= s.assets.size()) { okay = false; return S(0); }
        const auto& table = s.assets[tableAsset];
        if (table.values.size() < 2) { okay = false; return S(0); }
        const auto lo = s.parameters[3], hi = s.parameters[4];
        if (!(lo < hi)) { okay = false; return S(0); }
        // Halving before subtraction avoids overflow for valid wide double domains.
        const double ratio = (static_cast<double>(x) * 0.5 - static_cast<double>(lo) * 0.5) /
                             (static_cast<double>(hi) * 0.5 - static_cast<double>(lo) * 0.5);
        if (!std::isfinite(ratio)) { okay = false; return S(0); }
        const double index = std::clamp(ratio, 0.0, 1.0) * static_cast<double>(table.values.size() - 1);
        const auto a = std::min(static_cast<std::size_t>(index), table.values.size() - 2);
        S v0 = 0, v1 = 0;
        if (!assetValue(table, a, v0) || !assetValue(table, a + 1, v1)) okay = false;
        const auto fraction = static_cast<S>(index - static_cast<double>(a));
        return v0 * (S(1) - fraction) + v1 * fraction;
    }
}
template <class S, Shape shape, int factor>
bool evaluate(const StepT<S>& s, Overlay<S>& o, S& result) noexcept
{
    constexpr std::size_t parameterCount = shape == Shape::Table ? 5 : 3;
    if (s.parameters.size() != parameterCount || s.clock.periodSamples == 0 ||
        !std::isfinite(s.sampleRate) || s.sampleRate <= 0 || !std::isfinite(s.activation) || s.activation < 0)
        return false;
    for (auto p : s.parameters) if (!std::isfinite(p)) return false;
    if (s.parameters[0] < S(0)) return false;
    S input = 0;
    if (!s.inputs.audio.values.empty() && !assetValue(s.inputs.audio, 0, input)) return false;
    if constexpr (factor == 1)
    {
        if (!s.localState.empty()) return false;
        result = transform<S, shape>(s, input, 0, o.okay);
    }
    else
    {
        constexpr std::size_t stages = factor == 4 ? 2 : 1;
        if (s.assets.size() < stages) return false;
        const auto m1 = s.assets[0].values.size();
        const auto m2 = factor == 4 ? s.assets[1].values.size() : 0;
        if (!validM(m1) || (factor == 4 && !validM(m2)) || s.localState.size() != stateElements(m1, m2)) return false;
        auto pair = up(o, 0, s.assets[0], input);
        if constexpr (factor == 4)
        {
            const auto stage2 = stateElements(m1);
            for (auto& sample : pair)
            {
                auto high = up(o, stage2, s.assets[1], sample);
                for (auto& v : high) v = transform<S, shape>(s, v, stages, o.okay);
                sample = down(o, stage2 + m2 + 1, s.assets[1], high[0], high[1]);
            }
        }
        else for (auto& v : pair) v = transform<S, shape>(s, v, stages, o.okay);
        result = down(o, m1 + 1, s.assets[0], pair[0], pair[1]);
    }
    result *= s.parameters[2] * s.activation;
    return o.okay && std::isfinite(result);
}
template <class S, Shape shape, int factor>
std::size_t transition(const StepT<S>& s, std::span<StateWriteT<S>> journal) noexcept
{
    Overlay<S> o{s.localState};
    S output = 0;
    if (!evaluate<S, shape, factor>(s, o, output) || o.count > journal.size()) return rejected;
    std::copy_n(o.writes.begin(), o.count, journal.begin());
    return o.count;
}
template <class S, Shape shape, int factor>
void readout(const StepT<S>& s, std::span<S> out) noexcept
{
    std::fill(out.begin(), out.end(), S(0));
    Overlay<S> o{s.localState};
    S value = 0;
    if (out.size() != 1 || !evaluate<S, shape, factor>(s, o, value))
    {
        if (!out.empty()) out[0] = std::numeric_limits<S>::quiet_NaN();
        return;
    }
    out[0] = value;
}
template <class S, Shape shape, int factor>
void descriptor(std::string_view id, std::size_t n, PrimitiveT<S>& result)
{
    result = {std::string(id), n, 1, nullptr, readout<S, shape, factor>,
              factor == 1 ? 0u : factor == 2 ? 6u : 15u, transition<S, shape, factor>};
}
}
std::size_t stateElements(std::size_t first, std::size_t second) noexcept
{
    if (!validM(first) || (second != 0 && !validM(second))) return 0;
    return 3 + 5 * first / 2 + (second == 0 ? 0 : 3 + 5 * second / 2);
}
double latencyUpdates(std::size_t first, std::size_t second) noexcept
{
    if (stateElements(first, second) == 0) return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(first - 1) + (second == 0 ? 0.0 : static_cast<double>(second - 1) * 0.5);
}
std::span<const kernels::Metadata> catalog() noexcept { return metadata; }
template <class S>
TensorT<S> designHalfband(std::size_t halfK, double beta, std::string& error)
{
    TensorT<S> result;
    error.clear();
    if (halfK > 11 || !std::isfinite(beta) || beta < 0 || beta > 30)
    { error = "Halfband requires K=0..11 and finite Kaiser beta=0..30"; return result; }
    const auto m = 2 * halfK + 2;
    result.axes = {"halfband_dense_tap"}; result.shape = {m}; result.values.resize(m);
    alfa::dsp::detail::designNonlinearHalfbandDense<S>(result.values, static_cast<int>(halfK), beta);
    return result;
}
template <class S>
bool createPrimitive(std::string_view id, std::size_t n, PrimitiveT<S>& result, std::string& error)
{
    error.clear();
    const auto match = std::find_if(metadata.begin(), metadata.end(), [id](const auto& m) { return m.id == id; });
    if (match == metadata.end()) { error = "Unknown production nonlinear primitive"; return false; }
    const auto index = static_cast<std::size_t>(match - metadata.begin());
    const auto factor = index < 3 ? 1 : index < 6 ? 2 : 4;
    bool valid = factor == 1 && n == 0;
    for (std::size_t a = 2; a <= 24 && !valid; a += 2)
        if (factor == 2) valid = n == stateElements(a);
        else if (factor == 4) for (std::size_t b = 2; b <= 24; b += 2) valid = valid || n == stateElements(a, b);
    if (!valid) { error = "State size must exactly match explicitly supplied halfband assets"; return false; }
    switch (index)
    {
    case 0: descriptor<S, Shape::Tanh, 1>(id,n,result); break;
    case 1: descriptor<S, Shape::Clip, 1>(id,n,result); break;
    case 2: descriptor<S, Shape::Table, 1>(id,n,result); break;
    case 3: descriptor<S, Shape::Tanh, 2>(id,n,result); break;
    case 4: descriptor<S, Shape::Clip, 2>(id,n,result); break;
    case 5: descriptor<S, Shape::Table, 2>(id,n,result); break;
    case 6: descriptor<S, Shape::Tanh, 4>(id,n,result); break;
    case 7: descriptor<S, Shape::Clip, 4>(id,n,result); break;
    default: descriptor<S, Shape::Table, 4>(id,n,result); break;
    }
    return true;
}
template TensorT<float> designHalfband<float>(std::size_t, double, std::string&);
template TensorT<double> designHalfband<double>(std::size_t, double, std::string&);
template bool createPrimitive<float>(std::string_view, std::size_t, PrimitiveT<float>&, std::string&);
template bool createPrimitive<double>(std::string_view, std::size_t, PrimitiveT<double>&, std::string&);
}
