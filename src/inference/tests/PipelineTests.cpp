#include "FramePipeline.h"
#include "TestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

using namespace alfa::inference;
namespace dsp = alfa::dsp;

namespace
{

struct Frames
{
    std::vector<float> in = std::vector<float>(kInputFloatsPerFrame, 0.0f);
    std::vector<float> out = std::vector<float>(kFloatsPerFrame, -1.0f);

    Frames()
    {
        for (int v = 0; v < kMaxVoices; ++v)
            voice(v)[kNoteFeatures] = -1.0f;
    }
    float* voice(int v)
    {
        return in.data() + kGlobalModInputs + static_cast<std::size_t>(v) * kInputFloatsPerVoice;
    }
    const float* frame(int v) const { return out.data() + static_cast<std::size_t>(v) * kFloatsPerVoice; }
};

float sumAbs(const float* p, std::size_t n)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        s += std::abs(p[i]);
    return s;
}

} // namespace

TEST_CASE("pipeline reads its widths from the graphs of every tier", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    for (const auto& s : kTiers)
    {
        FramePipeline pipeline(test::requireModel(library, s.tier));
        REQUIRE(pipeline.embeddingSize() > 0);
        REQUIRE(pipeline.contextSize() > 0);
        REQUIRE(pipeline.hiddenSize() > kNoteFeatures);
    }
}

TEST_CASE("each voice gets its own method's controls and idle voices stay silent", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    const auto tier = GENERATE(Tier::Lite, Tier::Studio);
    FramePipeline pipeline(test::requireModel(library, tier));
    const auto limits = dsp::limitsFor(static_cast<dsp::Tier>(tier));

    Frames f;
    // Voice 0 subtractive, voice 3 FM, voice 5 granular, voice 9 sampling; the rest idle.
    const std::array<std::pair<int, int>, 4> voices{{{0, 0}, {3, 4}, {5, 6}, {9, 7}}};
    for (const auto& [v, method] : voices)
    {
        f.voice(v)[0] = 0.25f + 0.01f * static_cast<float>(v); // pitch
        f.voice(v)[1] = 0.8f;                                  // velocity
        f.voice(v)[2] = 1.0f;                                  // gate
        f.voice(v)[kNoteFeatures] = static_cast<float>(method);
    }
    pipeline.runFrame(f.in.data(), f.out.data());

    for (int v = 0; v < kMaxVoices; ++v)
    {
        const auto it =
            std::find_if(voices.begin(), voices.end(), [v](const auto& p) { return p.first == v; });
        if (it == voices.end())
        {
            REQUIRE(sumAbs(f.frame(v), kFloatsPerVoice) == 0.0f);
            continue;
        }
        // The dummy experts pass pitch through to mix[0] exactly.
        REQUIRE(test::passThrough(f.out.data(), v) == f.voice(v)[0]);
    }

    namespace off = dsp::offsets;
    // Subtractive writes OSC and NOISE, nothing of FM.
    REQUIRE(sumAbs(f.frame(0) + off::oscRe, 2 * dsp::kOscHarmonics) > 0.0f);
    REQUIRE(sumAbs(f.frame(0) + off::noiseBand, dsp::kBands) > 0.0f);
    REQUIRE(sumAbs(f.frame(0) + off::fmRatio, off::noiseBand - off::fmRatio) == 0.0f);
    // FM writes only its operators, and the N x N matrix lands in the frame's 8-wide rows.
    const auto n = static_cast<std::size_t>(limits.fmOperators);
    REQUIRE(sumAbs(f.frame(3) + off::oscRe, 2 * dsp::kOscHarmonics) == 0.0f);
    REQUIRE(sumAbs(f.frame(3) + off::fmRatio, n) > 0.0f);
    REQUIRE(sumAbs(f.frame(3) + off::fmRatio + n, dsp::kMaxFmOperators - n) == 0.0f);
    for (std::size_t row = 0; row < dsp::kMaxFmOperators; ++row)
    {
        const float* r = f.frame(3) + off::fmIndex + row * dsp::kMaxFmOperators;
        if (row < n)
        {
            REQUIRE(sumAbs(r, n) > 0.0f);
            REQUIRE(sumAbs(r + n, dsp::kMaxFmOperators - n) == 0.0f);
        }
        else
        {
            REQUIRE(sumAbs(r, dsp::kMaxFmOperators) == 0.0f);
        }
    }
    // Granular fills the tier's layers only.
    const auto layers = static_cast<std::size_t>(limits.grainLayers);
    REQUIRE(sumAbs(f.frame(5) + off::grainGain, layers) > 0.0f);
    REQUIRE(sumAbs(f.frame(5) + off::grainGain + layers, dsp::kMaxGrainLayers - layers) == 0.0f);
    // Sampling writes the sampler block.
    REQUIRE(sumAbs(f.frame(9) + off::samplerEq, dsp::kBands) > 0.0f);
}

TEST_CASE("recurrent state carries over frames and retrigger clears it", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    FramePipeline pipeline(test::requireModel(library, Tier::Standard));
    Frames f;
    f.voice(2)[0] = 0.5f;
    f.voice(2)[kNoteFeatures] = 1.0f; // wavetable
    const auto oscOf = [&]
    {
        return f.frame(2)[dsp::offsets::oscRe + 1];
    };

    pipeline.runFrame(f.in.data(), f.out.data());
    const float first = oscOf();
    pipeline.runFrame(f.in.data(), f.out.data());
    const float second = oscOf();
    REQUIRE(first != second); // same input, different state

    f.voice(2)[kNoteFeatures + 1] = 1.0f; // retrigger
    pipeline.resetState();                // instance state too, so the frame repeats exactly
    pipeline.runFrame(f.in.data(), f.out.data());
    REQUIRE(oscOf() == first);
}

TEST_CASE("a patch changes the output through the embedding", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    FramePipeline pipeline(test::requireModel(library, Tier::Lite));
    Frames f;
    f.voice(0)[0] = 0.5f;
    f.voice(0)[kNoteFeatures] = 0.0f;
    pipeline.runFrame(f.in.data(), f.out.data());
    const float before = f.frame(0)[dsp::offsets::oscRe + 3];

    std::array<float, kPatchKnobs> knobs{};
    knobs.fill(0.7f);
    pipeline.setPatch(knobs, 0);
    pipeline.resetState();
    pipeline.runFrame(f.in.data(), f.out.data());
    REQUIRE(f.frame(0)[dsp::offsets::oscRe + 3] != before);
}

TEST_CASE("a graph that breaks the contract is rejected with a readable error", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    auto broken = test::requireModel(library, Tier::Lite);
    SECTION("missing graph")
    {
        broken.files.erase("expert_fm.onnx");
        REQUIRE_THROWS_WITH(FramePipeline(broken), Catch::Matchers::ContainsSubstring("expert_fm.onnx"));
    }
    SECTION("wrong widths between graphs")
    {
        // Standard's voice core has a different hidden width than Lite's experts.
        const auto& standard = test::requireModel(library, Tier::Standard);
        const auto dir = std::filesystem::temp_directory_path() / "alfa-inference-mixed";
        std::filesystem::remove_all(dir);
        std::filesystem::copy(broken.folder, dir);
        std::filesystem::copy_file(standard.folder / "voice_core.onnx", dir / "voice_core.onnx",
                                   std::filesystem::copy_options::overwrite_existing);
        broken.folder = dir;
        REQUIRE_THROWS_WITH(FramePipeline(broken), Catch::Matchers::ContainsSubstring("MODEL_CONTRACT"));
    }
}
