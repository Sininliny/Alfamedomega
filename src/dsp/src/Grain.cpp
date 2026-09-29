#include "alfa/dsp/Grain.h"

#include "alfa/dsp/detail/Bands.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace alfa::dsp
{

namespace
{

inline std::size_t u(int i) noexcept
{
    return static_cast<std::size_t>(i);
}

/// Wraps x into [0, len).
inline double wrapPos(double x, double len) noexcept
{
    if (x >= len || x < 0.0)
    {
        x -= std::floor(x / len) * len;
        if (x >= len || x < 0.0)
            x = 0.0;
    }
    return x;
}

/// Buffer for a source index stored as a float, or nullptr when it reads silence.
inline const float* sourceBuffer(const GrainBank* bank, float source, float weight) noexcept
{
    if (bank == nullptr || bank->data == nullptr || bank->length <= 0 || weight == 0.0f
        || !std::isfinite(weight) || !std::isfinite(source))
        return nullptr;
    const double idx = std::floor(static_cast<double>(source) + 0.5);
    if (idx < 0.0 || idx >= static_cast<double>(bank->count))
        return nullptr;
    return bank->data + static_cast<std::ptrdiff_t>(idx) * bank->length;
}

template <bool Use0, bool Use1>
void readGrain(const float* s0, const float* s1, float a0, float a1, double& posRef, double rate,
               double len, int lenInt, const float* win, float* out, int n) noexcept
{
    double pos = posRef;
    for (int k = 0; k < n; ++k)
    {
        const int i0 = static_cast<int>(pos);
        const auto fr = static_cast<float>(pos - i0);
        const int i1 = i0 + 1 < lenInt ? i0 + 1 : 0;
        float v = 0.0f;
        if constexpr (Use0)
            v += a0 * (s0[i0] + fr * (s0[i1] - s0[i0]));
        if constexpr (Use1)
            v += a1 * (s1[i0] + fr * (s1[i1] - s1[i0]));
        out[k] += win[k] * v;
        pos = wrapPos(pos + rate, len);
    }
    posRef = pos;
}

} // namespace

void GrainPrimitive::prepare(const PrepareSpec& spec)
{
    hop_ = std::max(1, spec.hop);
    numLayers_ = std::clamp(spec.limits.grainLayers, 0, kMaxGrainLayers);
    window_ = detail::periodicHann(2 * hop_);
    reset();
}

void GrainPrimitive::reset() noexcept
{
    for (auto& layer : grains_)
        for (auto& g : layer)
            g.live = false;
    next_.fill(0);
    bank_ = pendingBank_;
}

int GrainPrimitive::liveGrains() const noexcept
{
    int c = 0;
    for (const auto& layer : grains_)
        for (const auto& g : layer)
            c += g.live ? 1 : 0;
    return c;
}

void GrainPrimitive::beginHop(const HopContext& ctx) noexcept
{
    bank_ = pendingBank_;
    const int life = 2 * hop_;
    for (int l = 0; l < numLayers_; ++l)
    {
        auto& layer = grains_[u(l)];
        for (auto& g : layer)
            if (g.live && g.age >= life)
                g.live = false;

        Grain& g = layer[u(next_[u(l)])];
        next_[u(l)] ^= 1;
        g.live = false;
        if (ctx.cur == nullptr || bank_ == nullptr)
            continue;

        const GrainControls& c = ctx.cur->grain;
        const float gain = c.gain[u(l)];
        const float rate = c.rate[u(l)];
        const float position = c.position[u(l)];
        const float w0 = c.weight[u(2 * l)];
        const float w1 = c.weight[u(2 * l + 1)];
        const float* s0 = sourceBuffer(bank_, c.source[u(2 * l)], w0);
        const float* s1 = sourceBuffer(bank_, c.source[u(2 * l + 1)], w1);
        if (gain == 0.0f || !std::isfinite(gain) || !std::isfinite(rate) || (s0 == nullptr && s1 == nullptr))
            continue;

        g.src0 = s0;
        g.src1 = s1;
        g.amp0 = gain * w0;
        g.amp1 = gain * w1;
        g.length = static_cast<double>(bank_->length);
        g.rate = static_cast<double>(rate);
        const double start = std::isfinite(position) ? static_cast<double>(position) * g.length : 0.0;
        g.pos = wrapPos(start, g.length);
        g.age = 0;
        g.live = true;
    }
}

void GrainPrimitive::renderGrain(Grain& g, float* out, int n) noexcept
{
    const int life = 2 * hop_;
    const int count = std::min(n, life - g.age);
    if (count <= 0)
    {
        g.live = false;
        return;
    }
    const float* win = window_.data() + g.age;
    const int lenInt = static_cast<int>(g.length);
    if (g.src0 != nullptr && g.src1 != nullptr)
        readGrain<true, true>(g.src0, g.src1, g.amp0, g.amp1, g.pos, g.rate, g.length, lenInt, win, out, count);
    else if (g.src0 != nullptr)
        readGrain<true, false>(g.src0, g.src1, g.amp0, g.amp1, g.pos, g.rate, g.length, lenInt, win, out, count);
    else
        readGrain<false, true>(g.src0, g.src1, g.amp0, g.amp1, g.pos, g.rate, g.length, lenInt, win, out, count);
    g.age += count;
    if (g.age >= life)
        g.live = false;
}

void GrainPrimitive::render(float* out, int n) noexcept
{
    std::fill(out, out + n, 0.0f);
    for (int l = 0; l < numLayers_; ++l)
        for (auto& g : grains_[u(l)])
            if (g.live)
                renderGrain(g, out, n);
}

void GrainPrimitive::skipHop(const HopContext&) noexcept
{
    bank_ = pendingBank_;
    for (auto& layer : grains_)
        for (auto& g : layer)
            g.live = false;
}

} // namespace alfa::dsp
