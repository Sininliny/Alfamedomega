#pragma once

// Interface for every processing family that can sit in a voice or instance slot
// (ControlFrame.h "Processing slots", Family.h for the list).
//
// A slot owns one preallocated processor per family, so switching family never allocates.
// Calls, all on the audio thread except prepare():
//   prepare(spec)          allocate every buffer for spec.maxDelaySeconds etc.; may throw
//   reset()                clear state (note-on for voice slots, transport reset for instance)
//   setParams(p, ctx)      new normalised params (kSlotParams floats, already modulated and
//                          clamped to 0..1), once per control block of at most kControlBlock
//                          samples; smooth internally anything that would click
//   process(ch, n)         in place on spec.channels channels, n <= kControlBlock
//   tailSamples()          how long the output can ring after the input stops (for voice end)
//
// Everything after prepare() is noexcept, allocation-free, lock-free and does no I/O.
// Outputs must stay finite for any params in 0..1 and any finite input; slots additionally
// sanitise outputs (a non-finite sample resets the processor and outputs 0).

#include "alfa/dsp/ControlFrame.h"

#include <array>
#include <string_view>

namespace alfa::dsp
{

inline constexpr int kControlBlock = 32;

struct ProcessSpec
{
    double sampleRate = 48000.0;
    int channels = 1;  ///< 1 in voice slots, 2 in instance slots
    TierLimits limits = limitsFor(Tier::Studio);
};

/// Per-control-block context. f0 is the voice fundamental (0 in instance slots, where
/// processors that key-track use their absolute frequency params instead).
struct ControlContext
{
    float f0 = 0.0f;
    float velocity = 0.0f;
};

enum class Curve : unsigned char
{
    Linear,  ///< min + x (max - min)
    Log,     ///< min (max / min)^x, min > 0
    Choice,  ///< round(x (count - 1)) with count = max + 1 choices, min = 0
    Toggle,  ///< x >= 0.5
};

/// Maps a normalised 0..1 slot parameter to its physical value. Exported to layout.json so the
/// training side sees the same ranges.
struct ParamInfo
{
    std::string_view name;  ///< empty for unused params
    float min = 0.0f;
    float max = 1.0f;
    Curve curve = Curve::Linear;
    float defaultNorm = 0.0f;  ///< the normalised value a neutral patch would use
    std::string_view unit;
};

using ParamTable = std::array<ParamInfo, kSlotParams>;

[[nodiscard]] float mapParam(const ParamInfo& info, float x) noexcept;

class Processor
{
public:
    virtual ~Processor() = default;
    virtual void prepare(const ProcessSpec& spec) = 0;
    virtual void reset() noexcept = 0;
    virtual void setParams(const float* normalised, const ControlContext& ctx) noexcept = 0;
    virtual void process(float* const* channels, int n) noexcept = 0;
    [[nodiscard]] virtual int tailSamples() const noexcept { return 0; }
};

} // namespace alfa::dsp
