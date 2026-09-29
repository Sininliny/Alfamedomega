#include "FramePipeline.h"
#include "LogMel.h"
#include "Npy.h"
#include "TestSupport.h"

#include <catch2/catch_approx.hpp>
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
    pipeline.setPatch(knobs, 0, -1, -1);
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

namespace
{

constexpr int kSubtractive = 0, kAdditive = 2, kFm = 4, kPhysical = 5;

// The dummy exporter's probes (tools/export_dummy_models.py).
constexpr std::size_t kF0Channel = 2;
constexpr float kHybridXf = 0.25f;

const float* field(const float* voiceFrame, std::string_view name)
{
    return voiceFrame + findControlField(name)->offset;
}

} // namespace

TEST_CASE("experts that take f0 get it in Hz from pitch and bend", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    FramePipeline pipeline(test::requireModel(library, Tier::Standard));
    Frames f;
    f.voice(1)[0] = 60.0f / 127.0f; // middle C
    f.voice(1)[7] = 0.5f;           // bend: +1 semitone
    f.voice(1)[kNoteFeatures] = static_cast<float>(kSubtractive);
    pipeline.runFrame(f.in.data(), f.out.data());

    REQUIRE(f0FromFeatures(f.voice(1)) == Catch::Approx(277.18).epsilon(1e-4));
    REQUIRE(field(f.frame(1), "mix")[kF0Channel] == f0FromFeatures(f.voice(1)));
}

TEST_CASE("physical keeps its modal state per voice and clears it on retrigger or release",
          "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    FramePipeline pipeline(test::requireModel(library, Tier::Lite));
    Frames f;
    for (int v : {0, 4})
    {
        f.voice(v)[0] = 0.4f;
        f.voice(v)[kNoteFeatures] = static_cast<float>(kPhysical);
    }
    // The dummy physical expert counts frames in modal_state and shows the count in partials.amp[1].
    const auto count = [&](int v)
    {
        return field(f.frame(v), "partials.amp")[1];
    };

    for (int frame = 0; frame < 3; ++frame)
    {
        pipeline.runFrame(f.in.data(), f.out.data());
        REQUIRE(count(0) == static_cast<float>(frame));
        REQUIRE(count(4) == static_cast<float>(frame));
    }
    f.voice(4)[kNoteFeatures + 1] = 1.0f; // retrigger voice 4 only
    pipeline.runFrame(f.in.data(), f.out.data());
    REQUIRE(count(0) == 3.0f);
    REQUIRE(count(4) == 0.0f);

    f.voice(4)[kNoteFeatures + 1] = 0.0f;
    f.voice(0)[kNoteFeatures] = -1.0f; // voice 0 goes idle
    pipeline.runFrame(f.in.data(), f.out.data());
    f.voice(0)[kNoteFeatures] = static_cast<float>(kPhysical);
    pipeline.runFrame(f.in.data(), f.out.data());
    REQUIRE(count(0) == 0.0f);
    REQUIRE(count(4) == 2.0f);

    pipeline.resetState();
    pipeline.runFrame(f.in.data(), f.out.data());
    REQUIRE(count(4) == 0.0f);
}

TEST_CASE("a hybrid voice merges its two components as training does", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    const auto tier = GENERATE(Tier::Lite, Tier::Studio);
    FramePipeline pipeline(test::requireModel(library, tier));
    std::array<float, kPatchKnobs> knobs{};
    knobs.fill(0.3f);

    // The dummy embedding ignores the method, so each component alone sees the same hidden state.
    const auto render = [&](int method, int a, int b)
    {
        Frames f;
        f.voice(6)[0] = 0.45f;
        f.voice(6)[1] = 0.9f;
        f.voice(6)[kNoteFeatures] = static_cast<float>(method);
        pipeline.setPatch(knobs, method, a, b);
        pipeline.resetState();
        pipeline.runFrame(f.in.data(), f.out.data());
        return std::vector<float>(f.frame(6), f.frame(6) + kFloatsPerVoice);
    };
    const auto A = render(kSubtractive, -1, -1);
    const auto B = render(kFm, -1, -1);
    const auto H = render(kHybridMethod, kSubtractive, kFm);
    const float wa = kHybridXf, wb = 1.0f - kHybridXf;
    namespace off = dsp::offsets;
    constexpr auto osc = static_cast<std::size_t>(dsp::Primitive::Osc);
    constexpr auto noise = static_cast<std::size_t>(dsp::Primitive::Noise);

    for (std::size_t p = 0; p < dsp::kPrimitiveCount; ++p)
        if (p != osc && p != noise)
            REQUIRE(H[off::mix + p] ==
                    Catch::Approx(wa * A[off::mix + p] + wb * B[off::mix + p]).margin(1e-6));
    REQUIRE(H[off::mix + osc] == 1.0f);
    REQUIRE(H[off::mix + noise] == 1.0f);
    REQUIRE(H[off::attackTau] == std::min(A[off::attackTau], B[off::attackTau]));
    for (std::size_t k : {0u, 1u, 500u, 1023u})
    {
        REQUIRE(H[off::oscRe + k] == Catch::Approx(wa * A[off::mix + osc] * A[off::oscRe + k]).margin(1e-6));
        REQUIRE(H[off::oscIm + k] == Catch::Approx(wa * A[off::mix + osc] * A[off::oscIm + k]).margin(1e-6));
    }
    for (std::size_t k = 0; k < dsp::kBands; ++k)
        REQUIRE(H[off::noiseBand + k] ==
                Catch::Approx(std::abs(wa * A[off::mix + noise] * A[off::noiseBand + k])).margin(1e-5));
    // FM belongs to component B alone and is copied as is.
    for (std::size_t k = off::fmRatio; k < off::noiseBand; ++k)
        REQUIRE(H[k] == B[k]);
}

TEST_CASE("a hybrid of two components that drive the same primitive is refused", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    FramePipeline pipeline(test::requireModel(library, Tier::Lite));
    std::array<float, kPatchKnobs> knobs{};
    // Additive and physical both drive PARTIALS, the pair training excludes.
    REQUIRE_THROWS_WITH(pipeline.setPatch(knobs, kHybridMethod, kAdditive, kPhysical),
                        Catch::Matchers::ContainsSubstring("partials"));
    REQUIRE_NOTHROW(pipeline.setPatch(knobs, kHybridMethod, kSubtractive, kPhysical));
}

TEST_CASE("a hybrid voice without components stays silent", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    FramePipeline pipeline(test::requireModel(library, Tier::Lite));
    Frames f;
    f.voice(0)[0] = 0.5f;
    f.voice(0)[kNoteFeatures] = static_cast<float>(kHybridMethod);
    pipeline.runFrame(f.in.data(), f.out.data());
    REQUIRE(sumAbs(f.frame(0), kFloatsPerVoice) == 0.0f);
}

TEST_CASE("a reference sample conditions the patch through the audio encoder", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    FramePipeline pipeline(test::requireModel(library, Tier::Pro));
    REQUIRE(pipeline.hasAudioEncoder());
    Frames f;
    f.voice(0)[0] = 0.5f;
    f.voice(0)[kNoteFeatures] = 7.0f; // sampling
    const auto render = [&]
    {
        pipeline.resetState();
        pipeline.runFrame(f.in.data(), f.out.data());
        return field(f.frame(0), "sampler.eq")[3];
    };
    const float plain = render();

    std::vector<float> sample(48000);
    for (std::size_t i = 0; i < sample.size(); ++i)
        sample[i] = 0.5f * std::sin(2.0f * 3.14159265f * 220.0f * static_cast<float>(i) / 48000.0f);
    REQUIRE(pipeline.setReferenceAudio(sample));
    REQUIRE(render() != plain);

    REQUIRE(pipeline.setReferenceAudio({}));
    REQUIRE(render() == plain);

    REQUIRE_FALSE(pipeline.setReferenceAudio(std::span(sample).first(1024))); // too short
}

TEST_CASE("the grain bank is loaded with the model", "[inference][pipeline]")
{
    const auto library = test::loadTestLibrary();
    FramePipeline pipeline(test::requireModel(library, Tier::Standard));
    REQUIRE(pipeline.grainBankRows() == 8);
    REQUIRE(pipeline.grainBank().size() == 8 * static_cast<std::size_t>(dsp::kGrainSourceLength));
    REQUIRE(sumAbs(pipeline.grainBank().data(), pipeline.grainBank().size()) > 0.0f);
}

TEST_CASE("log-mel matches an independent numpy reference", "[inference][logmel]")
{
    const auto library = test::loadTestLibrary();
    std::vector<float> x, reference;
    std::size_t rows = 0, columns = 0, bands = 0, frames = 0;
    std::string error;
    REQUIRE(readNpyFloat2d(test::modelsDir() / "logmel_input.npy", x, rows, columns, error));
    REQUIRE(rows == 1);
    REQUIRE(readNpyFloat2d(test::modelsDir() / "logmel_reference.npy", reference, bands, frames, error));
    REQUIRE(bands == kMelBands);

    std::size_t gotFrames = 0;
    const auto mel = logMel(x, gotFrames);
    REQUIRE(gotFrames == frames);
    REQUIRE(mel.size() == reference.size());
    float worst = 0.0f;
    for (std::size_t i = 0; i < mel.size(); ++i)
        worst = std::max(worst, std::abs(mel[i] - reference[i]));
    INFO("largest difference " << worst);
    REQUIRE(worst < 1e-3f);
}

TEST_CASE("npy reader refuses what it cannot read", "[inference][npy]")
{
    const auto dir = std::filesystem::temp_directory_path() / "alfa-inference-npy";
    std::filesystem::create_directories(dir);
    const auto write = [&](const std::string& name, const std::string& header, std::size_t dataBytes)
    {
        std::string h = header;
        while ((10 + h.size() + 1) % 64 != 0)
            h += ' ';
        h += '\n';
        std::ofstream out(dir / name, std::ios::binary);
        out << "\x93NUMPY" << '\x01' << '\x00' << static_cast<char>(h.size() & 0xff)
            << static_cast<char>(h.size() >> 8) << h << std::string(dataBytes, '\0');
    };
    std::vector<float> data;
    std::size_t rows = 0, columns = 0;
    std::string error;

    write("ok.npy", "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", 24);
    REQUIRE(readNpyFloat2d(dir / "ok.npy", data, rows, columns, error));
    REQUIRE((rows == 2 && columns == 3 && data.size() == 6));

    write("f8.npy", "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }", 48);
    REQUIRE_FALSE(readNpyFloat2d(dir / "f8.npy", data, rows, columns, error));
    REQUIRE(error.find("float32") != std::string::npos);

    write("short.npy", "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", 20);
    REQUIRE_FALSE(readNpyFloat2d(dir / "short.npy", data, rows, columns, error));

    write("1d.npy", "{'descr': '<f4', 'fortran_order': False, 'shape': (6,), }", 24);
    REQUIRE_FALSE(readNpyFloat2d(dir / "1d.npy", data, rows, columns, error));
}
