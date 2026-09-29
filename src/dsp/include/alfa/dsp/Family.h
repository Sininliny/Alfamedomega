#pragma once

// The processing families a slot can run (ControlFrame.h "Processing slots"). The numeric
// values are part of the control-frame contract: SlotControls::family holds them as floats.
// Append new families at the end; never renumber.

#include "alfa/dsp/Processor.h"

#include <cstdint>
#include <memory>
#include <string_view>

namespace alfa::dsp
{

enum class Family : std::uint8_t
{
    Off = 0,
    Filter,      ///< state-variable multimode LP/BP/HP/notch, 12 or 24 dB, resonant, drive
    Comb,        ///< feedforward / feedback comb, key-tracked
    Formant,     ///< parallel formant resonators with a vowel morph
    Resonator,   ///< modal bank (string, bar, membrane, plate, tube mode sets)
    Nonlinear,   ///< saturation, clipping, waveshaping, folding, rectification, bit/rate reduction
    Waveguide,   ///< Karplus-Strong / digital waveguide string excited by the slot input
    ModDelay,    ///< modulated short delay: chorus, flanger, vibrato
    Phaser,      ///< modulated all-pass chain
    Body,        ///< simple resonant body (fixed-topology biquad bank)
    Delay,       ///< feedback delay, stereo / ping-pong (instance slots only)
    Reverb,      ///< diffusion network / FDN reverb (instance slots only)
    Convolution, ///< partitioned FIR convolution with a prepared IR bank (instance slots only)
    Spectral,    ///< STFT-domain magnitude shaping, masking, freeze, phase ops (instance only)
    PitchShift,  ///< real-time pitch shifter (instance slots only)
    Spatial,     ///< pan, width, mid/side, Haas (instance slots only)
};
inline constexpr int kFamilyCount = 16;

/// Voice slots run per voice (mono, key-tracked); the heavy, long-memory families run only in
/// instance slots.
[[nodiscard]] constexpr bool allowedInVoice(Family f) noexcept
{
    return static_cast<int>(f) <= static_cast<int>(Family::Body);
}

[[nodiscard]] constexpr Family familyFromFloat(float x) noexcept
{
    if (!(x >= 0.5f) || x >= static_cast<float>(kFamilyCount) - 0.5f)
        return Family::Off; // also catches NaN
    return static_cast<Family>(static_cast<int>(x + 0.5f));
}

[[nodiscard]] std::string_view familyName(Family f) noexcept;
[[nodiscard]] const ParamTable& paramTable(Family f) noexcept;
/// Allocates. Returns nullptr for Off.
[[nodiscard]] std::unique_ptr<Processor> makeProcessor(Family f);

} // namespace alfa::dsp
