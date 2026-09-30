#pragma once

// 8-line feedback delay network for reverbs (and anything else that wants a dense, smooth,
// frequency-dependent decay).
//
// Each sample: every line is read (modulated length), passed through its absorption
// filter and gain, the 8 results are the network outputs; they are mixed by a normalised
// Hadamard matrix (orthogonal, so the mixing itself is lossless) and written back together with
// the 8 inputs. Per-line gains g_i = 10^(-3 L_i / (RT60 fs)) make every recirculation lose the
// same dB per second, so the whole network decays at RT60 regardless of which lines the energy
// travels through.
//
// Absorption per line (Jot-style, consistent across lines because it is derived from g_i):
//   high band: one-pole low-pass with DC gain 1 and Nyquist gain g_i^(hfRatio - 1)
//   low band:  low shelf below ~250 Hz with gain g_i^(lfRatio - 1)
// so RT60 at Nyquist is RT60 / hfRatio and at DC RT60 / lfRatio.
//
// Line lengths glide (one-pole, ~0.25 s) towards base * size, and slow sine LFOs with
// incommensurate rates move each length by up to the modulation depth, which breaks up the
// metallic periodicity of a static FDN. Lines are read with first-order allpass (Thiran)
// interpolation: its magnitude is exactly flat, so fractional and slowly modulated lengths do
// not low-pass the loop on every recirculation (cubic reads shortened the RT60 by ~6%). Freeze
// sets the loop gain to 1 - 1e-6 and glides lengths and modulation to exact integers: the held
// sound then neither decays audibly nor grows.

#include "alfa/dsp/detail/DelayLine.h"
#include "alfa/dsp/detail/FastMath.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace alfa::dsp::detail
{

class DelayFdn
{
public:
    static constexpr int kLines = 8;

    /// Base line lengths at size 1, in milliseconds (mutually prime-ish, 43..97 ms).
    static constexpr std::array<float, kLines> kBaseMs{43.1f, 49.7f, 57.3f, 63.1f,
                                                       71.9f, 79.3f, 88.7f, 97.1f};
    /// LFO rates (Hz) per line, incommensurate so the modulation never lines up.
    static constexpr std::array<float, kLines> kLfoHz{0.31f, 0.43f, 0.53f, 0.67f, 0.79f, 0.89f, 1.03f, 1.13f};

    /// maxModMs is the largest modulation depth that will be requested.
    void prepare(double sampleRate, float maxModMs = 1.0f)
    {
        sampleRate_ = sampleRate;
        const double msToSamples = sampleRate / 1000.0;
        for (int i = 0; i < kLines; ++i)
        {
            const auto s = static_cast<std::size_t>(i);
            base_[s] = static_cast<float>(static_cast<double>(kBaseMs[s]) * msToSamples);
            const auto maxLen =
                static_cast<int>(std::ceil(base_[s] + static_cast<float>(maxModMs * msToSamples))) + 4;
            lines_[s].prepare(maxLen);
            lfoInc_[s] = static_cast<float>(static_cast<double>(kLfoHz[s]) / sampleRate);
        }
        glide_ = static_cast<float>(1.0 - std::exp(-1.0 / (0.25 * sampleRate)));
        lowSplit_ = OnePole::coefficientFor(250.0, sampleRate);
        maxMod_ = static_cast<float>(maxModMs * msToSamples);
        reset();
    }

    void reset() noexcept
    {
        for (int i = 0; i < kLines; ++i)
        {
            const auto s = static_cast<std::size_t>(i);
            lines_[s].reset();
            readState_[s] = {};
            hf_[s].y = 0.0f;
            lf_[s].y = 0.0f;
            lenTarget_[s] = base_[s] * size_;
            lenOffset_[s] = 0.0f;
            lfoPhase_[s] = static_cast<float>(i) / kLines;
            gainOffset_[s] = 0.0f;
        }
        modDepth_ = modTarget_;
        snap_ = true;
    }

    /// Call once per control block. size in (0, 1] scales the base lengths; rt60 in seconds;
    /// hfRatio, lfRatio >= 1; modMs is the peak modulation depth. The first call after reset()
    /// jumps straight to these values instead of gliding.
    void setParams(float size, float rt60, float hfRatio, float lfRatio, float modMs, bool freeze) noexcept
    {
        size_ = std::clamp(size, 0.05f, 1.0f);
        modTarget_ =
            freeze ? 0.0f : std::clamp(modMs * static_cast<float>(sampleRate_ / 1000.0), 0.0f, maxMod_);
        rt60 = std::max(rt60, 0.01f);
        for (int i = 0; i < kLines; ++i)
        {
            const auto s = static_cast<std::size_t>(i);
            float target = base_[s] * size_;
            if (freeze)
                target = std::round(target);
            // The glide is kept as an offset from the target that decays to exactly 0: gliding
            // the length itself stalls once a step drops below half a float ulp (~1e-4 here),
            // which would leave frozen lines on fractional lengths.
            lenOffset_[s] = snap_ ? 0.0f : lenOffset_[s] + (lenTarget_[s] - target);
            lenTarget_[s] = target;

            // Uses the current (gliding) length so the decay stays on target while size moves.
            const double len = static_cast<double>(target) + static_cast<double>(lenOffset_[s]);
            const double g = std::pow(10.0, -3.0 * len / (static_cast<double>(rt60) * sampleRate_));
            float gain = 1.0f;
            if (freeze)
            {
                // Not exactly 1: float rounding in the mixing must never make the held loop grow.
                gain = 0.999999f;
                hfG_[s] = 1.0f;
                lfG_[s] = 1.0f;
            }
            else
            {
                gain = static_cast<float>(g);
                hfG_[s] =
                    static_cast<float>(std::pow(g, static_cast<double>(std::max(hfRatio, 1.0f) - 1.0f)));
                lfG_[s] =
                    static_cast<float>(std::pow(g, static_cast<double>(std::max(lfRatio, 1.0f) - 1.0f)));
            }
            gainOffset_[s] = snap_ ? 0.0f : gainOffset_[s] + (gainTarget_[s] - gain);
            gainTarget_[s] = gain;
            // One-pole with unity DC gain and Nyquist gain G: p = (1 - G) / (1 + G).
            hf_[s].p = (1.0f - hfG_[s]) / (1.0f + hfG_[s]);
            lf_[s].p = lowSplit_;
        }
        if (snap_)
            modDepth_ = modTarget_;
        snap_ = false;
    }

    /// One sample: in[8] are added to the lines, out[8] receive the line outputs (after
    /// absorption, before mixing).
    void tick(const float* in, float* out) noexcept
    {
        std::array<float, kLines> y{};
        modDepth_ += glide_ * 8.0f * (modTarget_ - modDepth_);
        for (int i = 0; i < kLines; ++i)
        {
            const auto s = static_cast<std::size_t>(i);
            float offset = lenOffset_[s] * (1.0f - glide_);
            if (std::fabs(offset) < 1e-4f)
                offset = 0.0f; // lands exactly (integer lengths in freeze)
            lenOffset_[s] = offset;
            const float len = lenTarget_[s] + offset;
            // Same offset form as the length: gliding a value near 1 directly would stall
            // ~5e-5 short of the target, an audible loss in freeze.
            gainOffset_[s] *= 1.0f - 8.0f * glide_;

            float phase = lfoPhase_[s] + lfoInc_[s];
            phase -= std::floor(phase);
            lfoPhase_[s] = phase;
            const float mod = modDepth_ < 1e-4f ? 0.0f : modDepth_ * sin2pi(phase);

            float v = lines_[s].readAllpass(len + mod - 1.0f, readState_[s]);
            v = hf_[s].lowpass(v);
            v += (lfG_[s] - 1.0f) * lf_[s].lowpass(v);
            v *= gainTarget_[s] + gainOffset_[s];
            y[s] = v;
            out[i] = v;
        }
        hadamard(y);
        for (int i = 0; i < kLines; ++i)
        {
            const auto s = static_cast<std::size_t>(i);
            lines_[s].write(y[s] + in[i]);
        }
    }

    /// Call once per block: flushes decayed filter states.
    void flush() noexcept
    {
        for (int i = 0; i < kLines; ++i)
        {
            const auto s = static_cast<std::size_t>(i);
            flushDenormal(hf_[s].y);
            flushDenormal(lf_[s].y);
            flushDenormal(gainOffset_[s]);
            flushDenormal(readState_[s].y1);
        }
    }

    [[nodiscard]] float lineLength(int i) const noexcept
    {
        const auto s = static_cast<std::size_t>(i);
        return lenTarget_[s] + lenOffset_[s];
    }

    /// In-place normalised 8-point fast Walsh-Hadamard transform (orthogonal).
    static void hadamard(std::array<float, kLines>& v) noexcept
    {
        for (int h = 1; h < kLines; h <<= 1)
            for (int i = 0; i < kLines; i += 2 * h)
                for (int j = i; j < i + h; ++j)
                {
                    const auto a = static_cast<std::size_t>(j);
                    const auto b = static_cast<std::size_t>(j + h);
                    const float x = v[a];
                    const float z = v[b];
                    v[a] = x + z;
                    v[b] = x - z;
                }
        constexpr float kNorm = 0.35355339059327373f; // 1 / sqrt(8)
        for (auto& x : v)
            x *= kNorm;
    }

private:
    std::array<DelayLine, kLines> lines_{};
    std::array<AllpassReadState, kLines> readState_{};
    std::array<OnePole, kLines> hf_{};
    std::array<OnePole, kLines> lf_{};
    std::array<float, kLines> base_{};
    std::array<float, kLines> lenOffset_{};
    std::array<float, kLines> lenTarget_{};
    std::array<float, kLines> gainOffset_{};
    std::array<float, kLines> gainTarget_{};
    std::array<float, kLines> hfG_{};
    std::array<float, kLines> lfG_{};
    std::array<float, kLines> lfoPhase_{};
    std::array<float, kLines> lfoInc_{};
    double sampleRate_ = 48000.0;
    float size_ = 1.0f;
    float glide_ = 0.0f;
    float lowSplit_ = 0.0f;
    float maxMod_ = 0.0f;
    float modDepth_ = 0.0f;
    float modTarget_ = 0.0f;
    bool snap_ = true;
};

} // namespace alfa::dsp::detail
