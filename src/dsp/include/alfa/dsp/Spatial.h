#pragma once

// Stereo-image helpers and the SPATIAL processing family (Family::Spatial, instance slots).
//
// Helpers (header-only, noexcept, usable from any family):
//   panGains / PanLaw        equal-power, linear and -4.5 dB pan laws
//   midSideEncode / Decode   M = (L + R) / 2, S = (L - R) / 2 and back
//   widthGains / applyWidth  side scaled by width (0 mono, 1 unchanged, 2 double side)
//   balanceGains             attenuate the opposite channel, no boost
//   crossfadeGains           constant-power (sin / cos) crossfade for uncorrelated signals
//   StereoMatrix             a 2x2 real matrix, how width, M/S balance, rotation and pan compose
//
// Extension points for richer spatialisation (HRTF binaural, ambisonics, object panning):
//   - SpatialRenderer is the positional stage of SpatialProcessor: it receives the image-shaped
//     stereo (or mono) signal and a SpatialPosition (azimuth, distance, spread) and renders the
//     output channels. The built-in StereoPanRenderer implements pan law + distance cue. A
//     binaural renderer would implement the same interface (HRTF pair per azimuth, crossfaded
//     like the convolution family's IR swap); SpatialProcessor owns a fixed array of
//     preallocated renderers and picks one per control block, so adding one never allocates on
//     the audio thread.
//   - Parameter slot 10 ("renderer") is reserved: today it is unused (empty name, stereo
//     renderer only). When a second renderer exists it becomes a Choice whose entry 0 stays the
//     stereo renderer, so existing patches keep their meaning.
//   - Renderers with more than two outputs (ambisonics, surround) do not fit an instance slot,
//     which is stereo. They belong in a new Family appended to Family.h, reusing
//     SpatialPosition and the SpatialRenderer interface with a wider output count.

#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/ConvspatialUtil.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace alfa::dsp
{

// Helpers --------------------------------------------------------------------------------------

enum class PanLaw : unsigned char
{
    EqualPower, ///< L^2 + R^2 = 1 everywhere, -3 dB at centre
    Minus4_5dB, ///< geometric mean of equal-power and linear, -4.5 dB at centre
    Linear,     ///< L + R = 1 everywhere, -6 dB at centre
};

struct StereoGains
{
    float left = 1.0f;
    float right = 1.0f;
};

/// Gains for a mono source at pan position `pan` in [-1, 1] (-1 hard left). The extremes are
/// exact: -1 gives (1, 0) and +1 gives (0, 1) for every law.
[[nodiscard]] inline StereoGains panGains(float pan, PanLaw law) noexcept
{
    const float u = 0.5f * (std::clamp(std::isfinite(pan) ? pan : 0.0f, -1.0f, 1.0f) + 1.0f); // 0..1
    const float halfPi = 0.5f * detail::kPi;
    // sin of the complementary angle instead of cos keeps both ends exact in float.
    const float epL = std::sin((1.0f - u) * halfPi);
    const float epR = std::sin(u * halfPi);
    switch (law)
    {
    case PanLaw::EqualPower:
        return {epL, epR};
    case PanLaw::Linear:
        return {1.0f - u, u};
    case PanLaw::Minus4_5dB:
        return {std::sqrt(epL * (1.0f - u)), std::sqrt(epR * u)};
    }
    return {epL, epR};
}

struct MidSide
{
    float mid = 0.0f;
    float side = 0.0f;
};

[[nodiscard]] inline MidSide midSideEncode(float left, float right) noexcept
{
    return {0.5f * (left + right), 0.5f * (left - right)};
}

/// Inverse of midSideEncode (exact up to one rounding of the sum and difference).
[[nodiscard]] inline StereoGains midSideDecode(float mid, float side) noexcept
{
    return {mid + side, mid - side};
}

/// Scales the side signal by `width` (0 mono, 1 unchanged, 2 double side), in place.
inline void applyWidth(float& left, float& right, float width) noexcept
{
    const MidSide ms = midSideEncode(left, right);
    const StereoGains lr = midSideDecode(ms.mid, ms.side * width);
    left = lr.left;
    right = lr.right;
}

/// Balance in [-1, 1]: positive attenuates the left channel linearly to 0 at +1, negative the
/// right. Never boosts, so 0 is the identity.
[[nodiscard]] inline StereoGains balanceGains(float balance) noexcept
{
    const float b = std::clamp(std::isfinite(balance) ? balance : 0.0f, -1.0f, 1.0f);
    return {b > 0.0f ? 1.0f - b : 1.0f, b < 0.0f ? 1.0f + b : 1.0f};
}

/// Constant-power crossfade at position t in [0, 1]: from = cos(t pi/2), to = sin(t pi/2), so
/// from^2 + to^2 = 1. Use it between uncorrelated signals; correlated signals want a linear
/// fade (from = 1 - t) instead.
[[nodiscard]] inline StereoGains crossfadeGains(float t) noexcept
{
    const float u = std::clamp(std::isfinite(t) ? t : 0.0f, 0.0f, 1.0f);
    const float halfPi = 0.5f * detail::kPi;
    return {std::sin((1.0f - u) * halfPi), std::sin(u * halfPi)};
}

/// out = [[ll, rl], [lr, rr]] * in, i.e. outL = ll L + rl R, outR = lr L + rr R.
struct StereoMatrix
{
    float ll = 1.0f, rl = 0.0f, lr = 0.0f, rr = 1.0f;

    [[nodiscard]] static StereoMatrix identity() noexcept { return {}; }

    /// this applied after `first`.
    [[nodiscard]] StereoMatrix after(const StereoMatrix& first) const noexcept
    {
        return {ll * first.ll + rl * first.lr, ll * first.rl + rl * first.rr, lr * first.ll + rr * first.lr,
                lr * first.rl + rr * first.rr};
    }

    /// Mid gain gm and side gain gs in L/R terms: L' = M gm + S gs, R' = M gm - S gs.
    [[nodiscard]] static StereoMatrix midSide(float gm, float gs) noexcept
    {
        const float a = 0.5f * (gm + gs);
        const float b = 0.5f * (gm - gs);
        return {a, b, b, a};
    }

    /// Rotates the (L, R) vector by `radians` (positive turns the image to the right). Energy
    /// preserving; a centred mono source reaches hard right at +pi/4.
    [[nodiscard]] static StereoMatrix rotation(float radians) noexcept
    {
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        return {c, -s, s, c};
    }

    /// Stereo pan: the left input channel sits at -1 and the right at +1; both move by `pan` and
    /// are clamped to [-1, 1], then each is placed with the pan law. pan = 0 is the identity.
    [[nodiscard]] static StereoMatrix stereoPan(float pan, PanLaw law) noexcept
    {
        const StereoGains gl = panGains(-1.0f + 2.0f * std::max(pan, 0.0f), law);
        const StereoGains gr = panGains(1.0f + 2.0f * std::min(pan, 0.0f), law);
        return {gl.left, gr.left, gl.right, gr.right};
    }
};

// Renderer extension point ----------------------------------------------------------------------

/// Where the (already image-shaped) signal should appear. azimuth in [-1, 1] (-1 hard left;
/// renderers with a real angle map it to -90..+90 degrees), distance 0 (at the listener) to 1
/// (far), spread reserved for renderers that model source width.
struct SpatialPosition
{
    float azimuth = 0.0f;
    float distance = 0.0f;
    float spread = 1.0f;
    PanLaw law = PanLaw::EqualPower;
};

/// The positional stage of SpatialProcessor (see the header comment). prepare() may allocate;
/// the rest must be real-time safe. process() works in place on `channels` channels (1 or 2),
/// n <= kControlBlock, and smooths position changes itself.
class SpatialRenderer
{
public:
    virtual ~SpatialRenderer() = default;
    virtual void prepare(double sampleRate, int channels) = 0;
    virtual void reset() noexcept = 0;
    virtual void setPosition(const SpatialPosition& position) noexcept = 0;
    virtual void process(float* const* channels, int n) noexcept = 0;
    [[nodiscard]] virtual int latencySamples() const noexcept { return 0; }
};

/// Built-in renderer: stereo pan with a selectable law, plus a distance cue (level falls to
/// -18 dB and a first-order high shelf at 4 kHz to -12 dB as distance goes 0 -> 1).
class StereoPanRenderer final : public SpatialRenderer
{
public:
    static constexpr float kFarLevelDb = -18.0f;
    static constexpr float kFarShelfDb = -12.0f;
    static constexpr double kShelfHz = 4000.0;

    void prepare(double sampleRate, int channels) override;
    void reset() noexcept override;
    void setPosition(const SpatialPosition& position) noexcept override;
    void process(float* const* channels, int n) noexcept override;

private:
    double sampleRate_ = 48000.0;
    int channels_ = 2;
    std::array<detail::LinearRamp, 4> matrix_; // ll, rl, lr, rr
    detail::LinearRamp level_;
    detail::LinearRamp shelf_; // linear HF gain minus 1
    std::array<detail::TptOnePole, 2> shelfLp_;
    bool first_ = true;
};

// SPATIAL family -------------------------------------------------------------------------------

/// Stereo image processor for instance slots. Signal flow per sample:
///   mono-maker (M through an all-pass, S through a matching LR4 high-pass below the crossover)
///   -> width + M/S balance + rotation (one 2x2 matrix, ramped per block)
///   -> Haas delay on one side
///   -> renderer (pan law + distance cue)
///   -> mix with the dry input (linear).
/// With the neutral defaults the output is bit-identical to the input. With one channel only
/// the distance cue and mix apply (the stereo stages need two channels).
///
/// Params (normalised 0..1, order is the contract):
///   0 pan           -1..1          stereo pan (0 = identity), Linear
///   1 pan law       0..2 Choice    equal-power, -4.5 dB, linear
///   2 width         0..2           0 mono, 1 unchanged, 2 double side, Linear
///   3 ms balance    -1..1          -1 mid only, 0 unchanged, +1 side only, Linear
///   4 haas          0..30 ms       delay applied to one side, Linear
///   5 haas side     Toggle         off: delay the left channel, on: the right
///   6 rotation      -45..45 deg    rotates the L/R vector, Linear
///   7 mono below    0..500 Hz      mono-maker crossover, below 10 Hz = off, Linear
///   8 distance      0..1           level -18 dB and HF shelf -12 dB at 1, Linear
///   9 mix           0..1           dry / processed, Linear
///  10 (reserved: renderer choice, see the header comment)
class SpatialProcessor final : public Processor
{
public:
    static constexpr double kMaxHaasMs = 30.0;
    static constexpr float kMonoOffHz = 10.0f;

    [[nodiscard]] static const ParamTable& params() noexcept;

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;

private:
    void processStereo(float* l, float* r, int n) noexcept;
    void processChunk(float* const* channels, int n) noexcept;
    void clearState() noexcept;

    double sampleRate_ = 48000.0;
    int channels_ = 2;
    bool first_ = true;

    // Mono-maker: M all-pass (one SVF), S LR4 high-pass (two SVFs), crossfaded in and out.
    float monoHz_ = 0.0f;       // smoothed crossover
    float monoTargetHz_ = 0.0f; // 0 = off
    float monoSmooth_ = 1.0f;
    detail::TptSvf monoMid_;
    std::array<detail::TptSvf, 2> monoSide_;
    detail::LinearRamp monoAmount_;

    std::array<detail::LinearRamp, 4> image_; // width / balance / rotation matrix

    // Haas: one delay line per channel, per-channel smoothed delay (in samples), so switching
    // side glides instead of jumping.
    std::vector<float> delay_; // 2 x delaySize_
    int delaySize_ = 0;
    int delayMask_ = 0;
    int writePos_ = 0;
    std::array<float, 2> delaySamples_{};
    std::array<float, 2> delayTarget_{};
    float delaySmooth_ = 1.0f;

    StereoPanRenderer renderer_;
    SpatialPosition position_;
    detail::LinearRamp mix_;

    std::array<float, kControlBlock> dryL_{}, dryR_{};
};

} // namespace alfa::dsp
