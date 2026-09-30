#pragma once

// Building blocks shared by the physical-model families (Physical.h): the precomputed modal
// ratio tables and the vectorised modal bank that renders them.
//
// ModalBank knows nothing about strings or bars: it takes per-mode frequency (cycles per
// sample), decay (nepers per sample) and gain arrays and rings them. New mode sources (plates
// with other boundary conditions, measured instrument bodies, 2-D meshes reduced to modes)
// produce those arrays and reuse the bank unchanged.

#include "alfa/dsp/detail/FastMath.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace alfa::dsp::detail
{

inline constexpr int kPhysicalMaxModes = 32;

/// The mode sets the Resonator structure param morphs across, in morph order. This order is part
/// of the param contract (structure value i is exactly table i); append new sets only through a
/// new param or family, never by inserting here.
enum class PhysicalModeSet : int
{
    IdealString = 0, ///< k                                   (harmonic)
    StiffString,     ///< k sqrt(1 + B k^2) / sqrt(1 + B), B = kStiffStringB
    FreeBar,         ///< (beta_k / beta_1)^2, free-free Euler-Bernoulli beam: 1, 2.756, 5.404, ...
    Membrane,        ///< Bessel zeros j_mn / j_01, ideal circular membrane: 1, 1.594, 2.136, ...
    SquarePlate,     ///< (m^2 + n^2) / 2, simply supported square plate, degenerate pairs merged
    Tube,            ///< 2k - 1, closed-open tube (odd harmonics)
};
inline constexpr int kPhysicalModeSetCount = 6;
inline constexpr double kStiffStringB = 0.001;

struct PhysicalModeTables
{
    /// ratio[set][k]: frequency of mode k over mode 0, ascending, ratio[set][0] == 1.
    std::array<std::array<float, kPhysicalMaxModes>, kPhysicalModeSetCount> ratio{};
    /// log2 of the above (the morph interpolates in log frequency).
    std::array<std::array<float, kPhysicalMaxModes>, kPhysicalModeSetCount> log2Ratio{};
};

/// Built once on first use (thread-safe static init); call it from prepare(), not the audio
/// thread. The membrane uses tabulated Bessel zeros (no C++17 special math, for portability).
[[nodiscard]] const PhysicalModeTables& physicalModeTables();

/// 2^x for |x| < 60, relative error about 3e-7, branch-free so loops over it vectorise.
[[nodiscard]] inline float physicalExp2(float x) noexcept
{
    x = std::fmin(std::fmax(x, -60.0f), 60.0f);
    const float xi = std::floor(x + 0.5f);
    const float f = x - xi; // [-0.5, 0.5]
    // Taylor series of e^(f ln 2) to degree 6; truncation error below 1.3e-7 on this range.
    float p = 1.5252734e-5f;
    p = p * f + 1.5403530e-4f;
    p = p * f + 1.3333558e-3f;
    p = p * f + 9.6181291e-3f;
    p = p * f + 5.5504109e-2f;
    p = p * f + 2.4022651e-1f;
    p = p * f + 6.9314718e-1f;
    p = p * f + 1.0f;
    const auto e = static_cast<std::int32_t>(xi) + 127;
    return p * std::bit_cast<float>(static_cast<std::uint32_t>(e) << 23);
}

/// Bank of up to kPhysicalMaxModes decaying complex phasors (z <- r e^{i theta} z + g x, output
/// Im z), one state set per channel. The rotation form is unconditionally stable for r < 1 and
/// keeps its accuracy at very long decays, unlike a direct-form biquad whose poles crowd z = 1.
/// Data is structure-of-arrays so the per-sample loop vectorises over modes.
class ModalBank
{
public:
    static constexpr int kMaxChannels = 2;

    void reset() noexcept;
    /// freq in cycles per sample (modes at or above maxFreq are silenced), decay in nepers per
    /// sample (amplitude falls by e^-decay each sample), gain linear. count <= kPhysicalMaxModes.
    void setModes(const float* freq, const float* decay, const float* gain, int count,
                  float maxFreq = 0.48f) noexcept;
    /// in and out may alias. Returns false (and resets the channel, zeroing out) if the state
    /// went non-finite.
    bool process(int channel, const float* in, float* out, int n) noexcept;

    [[nodiscard]] int activeModes() const noexcept { return active_; }
    [[nodiscard]] float coefficientMagnitude(int k) const noexcept
    {
        const auto i = static_cast<std::size_t>(k);
        return std::sqrt(c_[i] * c_[i] + s_[i] * s_[i]);
    }

    void resetChannel(int channel) noexcept;

private:
    alignas(64) std::array<float, kPhysicalMaxModes> c_{}; // r cos theta
    alignas(64) std::array<float, kPhysicalMaxModes> s_{}; // r sin theta
    alignas(64) std::array<float, kPhysicalMaxModes> g_{};
    alignas(64) std::array<std::array<float, kPhysicalMaxModes>, kMaxChannels> re_{};
    alignas(64) std::array<std::array<float, kPhysicalMaxModes>, kMaxChannels> im_{};
    int padded_ = 0; // loop length: highest active mode rounded up to 8
    int active_ = 0; // modes below maxFreq with nonzero gain
};

} // namespace alfa::dsp::detail
