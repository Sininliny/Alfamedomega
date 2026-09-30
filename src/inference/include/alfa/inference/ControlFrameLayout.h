#pragma once

// How the network's outputs map onto the DSP's control frame.
//
// The layout itself is owned by the DSP component (alfa/dsp/ControlFrame.h). This header only
// adds what the inference host needs on top: frame sizes for all voice slots, the network's
// input sizes (training/METHOD.md section 1), and lookup of a frame field by the ONNX output
// name, which is the field's dotted name there ("osc.re", "fm.index", ...).

#include <alfa/dsp/ControlFrame.h>
#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

namespace alfa::inference
{

/// Upper bound on voice slots per plugin instance. Each tier's budgeted polyphony is lower or equal.
inline constexpr int kMaxVoices = 16;

/// Per-voice inputs to the network each frame (training/METHOD.md section 1):
/// pitch/127, velocity, gate, time since note-on, time since note-off, mod wheel, aftertouch, bend.
inline constexpr int kNoteFeatures = 8;
/// Per-instance modulation inputs each frame.
inline constexpr int kGlobalModInputs = 4;
/// Patch encoder inputs.
inline constexpr int kPatchKnobs = 32;
inline constexpr int kNumMethods = 9;

inline constexpr std::array<std::string_view, kNumMethods> kMethodNames{
    "subtractive", "wavetable", "additive", "spectral", "fm", "physical", "granular", "sampling", "hybrid",
};

/// Floats in one voice's control frame (alfa::dsp::VoiceFrame as a flat array).
inline constexpr std::size_t kFloatsPerVoice = dsp::kFrameFloats;

/// Floats in one whole frame: every voice slot's VoiceFrame, back to back, voice-major.
inline constexpr std::size_t kFloatsPerFrame = kFloatsPerVoice * kMaxVoices;

struct ControlField
{
    std::string_view name; ///< Must equal the ONNX expert output name.
    std::size_t offset;    ///< In floats from the start of the voice's VoiceFrame.
    std::size_t capacity;  ///< Largest size any tier writes; smaller tiers leave the tail zero.
};

/// The FM modulation matrix is 8 x 8 row-major in the frame. A tier with N < 8 operators may
/// output it as N x N, and the host spreads its rows out to the frame's row stride.
inline constexpr std::string_view kFmIndexField = "fm.index";

[[nodiscard]] constexpr std::optional<ControlField> findControlField(std::string_view name) noexcept
{
    for (const auto& f : dsp::kFields)
        if (f.name == name)
            return ControlField{f.name, f.offset, f.count};
    return std::nullopt;
}

} // namespace alfa::inference
