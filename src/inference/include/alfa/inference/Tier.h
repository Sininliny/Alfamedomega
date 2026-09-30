#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

namespace alfa::inference
{

/// Model size tier (docs/tiers/model-tiers.md, v3).
enum class Tier
{
    Lite,
    Standard,
    Pro,
    Studio,
};

/// Runtime facts about a tier that the inference host needs. Parameter counts and fidelity
/// live in the tier spec; only what changes how the host schedules inference is here.
struct TierSpec
{
    Tier tier;
    std::string_view name;       ///< Manifest and folder name, lower case.
    float maxInferenceMs;        ///< Declared to anira; sets the reported latency.
    unsigned inferenceThreads;   ///< anira worker threads the tier asks for.
    int polyphonyAtBudget;       ///< Voices the tier's CPU budget assumes.
    int smallestBufferSupported; ///< Host block size below which the tier is not supported.
};

// maxInferenceMs is the worst case one frame's inference may take. Because the model is
// recurrent, frames run one after another, so it must stay below the 4 ms frame period or the
// stream falls behind (anira warns "the inference pool cannot keep up"). The tier spec's first
// draft declared 4 / 4 / 8 ms for Standard / Pro / Studio; these are the largest values that keep
// up. Reported latency at 48 kHz: Lite 8 ms, Standard and Pro 12 ms at 128 samples, Studio 20 ms
// at 512 samples (tests/HostTests.cpp prints them).
inline constexpr std::array<TierSpec, 4> kTiers{{
    {Tier::Lite, "lite", 2.0f, 1, 16, 32},
    {Tier::Standard, "standard", 3.0f, 1, 16, 64},
    {Tier::Pro, "pro", 3.0f, 2, 16, 128},
    {Tier::Studio, "studio", 3.5f, 4, 8, 512},
}};

inline constexpr Tier kDefaultTier = Tier::Standard;

[[nodiscard]] constexpr const TierSpec& spec(Tier tier) noexcept
{
    return kTiers[static_cast<std::size_t>(tier)];
}

[[nodiscard]] constexpr std::string_view toString(Tier tier) noexcept
{
    return spec(tier).name;
}

[[nodiscard]] constexpr std::optional<Tier> tierFromString(std::string_view name) noexcept
{
    for (const auto& s : kTiers)
        if (s.name == name)
            return s.tier;
    return std::nullopt;
}

} // namespace alfa::inference
