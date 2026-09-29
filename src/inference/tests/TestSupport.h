#pragma once

#include "alfa/inference/InferenceHost.h"
#include "alfa/inference/ModelLibrary.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

namespace alfa::inference::test
{

inline std::filesystem::path modelsDir()
{
    return ALFA_TEST_MODELS_DIR;
}

/// Loads the dummy models' manifest, or skips the test when they could not be generated.
inline ModelLibrary loadTestLibrary()
{
    if (!ALFA_HAVE_TEST_MODELS)
        SKIP("dummy models need Python 3 with numpy and onnx at configure time");
    ModelLibrary library;
    std::string error;
    REQUIRE(library.load(modelsDir(), error));
    return library;
}

inline const ModelEntry& requireModel(const ModelLibrary& library, Tier tier)
{
    const auto* entry = library.find(tier);
    REQUIRE(entry != nullptr);
    return *entry;
}

/// Offset of the pass-through value in a voice's frame: channel 0 of "mix", which the dummy
/// experts set to feature 0 (pitch) of the same voice and frame.
inline float passThrough(const float* frame, int voice)
{
    return frame[static_cast<std::size_t>(voice) * kFloatsPerVoice + findControlField("mix")->offset];
}

} // namespace alfa::inference::test
