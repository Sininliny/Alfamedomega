#include "alfa/tensor/Universe.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <utility>

namespace
{
using namespace alfa::tensor;
using namespace alfa::tensor::universe;
struct Harness
{
    Primitive primitive;
    std::vector<float> state, next, output;
    std::vector<std::vector<float>> assetStorage;
    std::vector<TensorView> assets;
    double rate;
    Harness(std::string_view id, std::size_t count, double sampleRate = 48000)
        : rate(sampleRate)
    {
        std::string error;
        REQUIRE(createPrimitive(id, count, primitive, error));
        state.resize(count, 0.0f);
        next = state;
        output.resize(primitive.outputElements, 0.0f);
    }
    void setAssets(std::vector<std::vector<float>> data)
    {
        assetStorage = std::move(data);
        assets.clear();
        for (const auto& values : assetStorage)
            assets.push_back({{}, {}, values, {}});
    }
    std::span<const float> tick(std::span<const float> params, std::span<const float> audio = {},
                                std::span<const float> latent = {}, std::span<const float> midi = {})
    {
        Step s;
        s.sampleRate = rate;
        s.clock = {1, 0};
        s.activation = 1;
        s.localState = state;
        s.parameters = params;
        s.inputs.audio.values = audio;
        s.inputs.latentSound.values = latent;
        s.inputs.midi.values = midi;
        s.assets = assets;
        primitive.readout(s, output);
        primitive.transition(s, next);
        state.swap(next);
        return output;
    }
    float scalar(std::span<const float> params, float input = 0)
    {
        const std::array<float, 1> a{input};
        return tick(params, a)[0];
    }
};
} // namespace

TEST_CASE("Tensor universe FFT has conventional complex normalization and inverse", "[tensor][universe]")
{
    Harness forward("spectral.fft", 16), inverse("spectral.ifft", 16);
    std::array<float, 16> impulse{};
    impulse[0] = 1.0f;
    forward.tick({}, impulse);
    for (std::size_t i = 0; i < 8; ++i)
    {
        REQUIRE(forward.state[2 * i] == 1.0f);
        REQUIRE(forward.state[2 * i + 1] == 0.0f);
    }
    inverse.tick({}, forward.state);
    for (std::size_t i = 0; i < 16; ++i)
        REQUIRE(inverse.state[i] == Catch::Approx(impulse[i]).margin(0.00001));
    std::array<float, 16> complexTone{};
    for (std::size_t i = 0; i < 8; ++i)
    {
        const double phase = 2 * std::numbers::pi * static_cast<double>(i) / 8;
        complexTone[2 * i] = static_cast<float>(std::cos(phase));
        complexTone[2 * i + 1] = static_cast<float>(std::sin(phase));
    }
    forward.tick({}, complexTone);
    for (std::size_t i = 0; i < 8; ++i)
    {
        REQUIRE(forward.state[2 * i] == Catch::Approx(i == 1 ? 8.0 : 0.0).margin(0.00001));
        REQUIRE(forward.state[2 * i + 1] == Catch::Approx(0).margin(0.00001));
    }
}

TEST_CASE("Tensor spectral transformations retain explicit editable bin semantics", "[tensor][universe]")
{
    const std::array<float, 8> bins{0, 0, 1, 2, 3, 4, 5, 6};
    Harness mask("spectral.mask", 8);
    mask.setAssets({{1, 0, 0.5f, 2}});
    mask.tick({}, bins);
    REQUIRE(mask.state[2] == 0);
    REQUIRE(mask.state[4] == 1.5f);
    REQUIRE(mask.state[7] == 12);
    Harness freeze("spectral.freeze", 8);
    const std::array<float, 1> capture{0}, held{1};
    freeze.tick(capture, bins);
    const auto saved = freeze.state;
    const std::array<float, 8> empty{};
    freeze.tick(held, empty);
    REQUIRE(freeze.state == saved);
    Harness redistribution("spectral.redistribute", 8);
    const std::array<float, 1> ratio{2};
    redistribution.tick(ratio, bins);
    REQUIRE(redistribution.state[4] == 1);
    REQUIRE(redistribution.state[5] == 2);
    REQUIRE(redistribution.state[6] == 0);
    Harness tilt("spectral.tilt", 8, 4000);
    const std::array<float, 2> slope{6.0206f, 1000};
    tilt.tick(slope, bins);
    REQUIRE(tilt.state[2] == Catch::Approx(1).margin(0.00001));
    REQUIRE(tilt.state[4] == Catch::Approx(6).margin(0.00001));
    REQUIRE(tilt.state[6] == Catch::Approx(5).margin(0.00001));
}

TEST_CASE("Tensor wavetable, additive, sample and resampling are actual excitations", "[tensor][universe]")
{
    Harness wave("osc.wavetable", 1);
    wave.setAssets({{0, 1, 0, -1}});
    const std::array<float, 2> oscillator{12000, 1};
    REQUIRE(wave.scalar(oscillator) == 0);
    REQUIRE(wave.scalar(oscillator) == 1);
    REQUIRE(wave.scalar(oscillator) == 0);
    REQUIRE(wave.scalar(oscillator) == -1);
    Harness additive("osc.additive", 2);
    const std::array<float, 4> partials{12000, 1, 6000, 0.5f};
    additive.scalar(partials);
    REQUIRE(additive.scalar(partials) == Catch::Approx(1 + std::sqrt(0.5) * 0.5).margin(0.00001));
    Harness sample("excitation.sample_loop", 1), resample("pitchtime.asset_resample", 1);
    sample.setAssets({{0, 1, 2, 3}});
    resample.setAssets({{0, 1, 2, 3}});
    const std::array<float, 2> speed{48000, 1};
    for (int i = 0; i < 4; ++i)
    {
        REQUIRE(sample.scalar(speed) == static_cast<float>(i));
        REQUIRE(resample.scalar(speed) == static_cast<float>(i));
    }
    REQUIRE(sample.scalar(speed) == 0);
    REQUIRE(resample.scalar(speed) == 0);
    REQUIRE(resample.scalar(speed) == 0);
}

TEST_CASE("Tensor band-limited saw reduces out-of-band harmonic foldover", "[tensor][universe]")
{
    Harness h("osc.polyblep_saw", 1);
    const std::array<float, 2> p{7000, 1};
    std::array<std::complex<double>, 4> corrected{}, naive{};
    double phase = 0;
    for (int i = 0; i < 4800; ++i)
    {
        const double value = h.scalar(p);
        for (std::size_t k = 0; k < 4; ++k)
        {
            const double angle =
                -2 * std::numbers::pi * static_cast<double>(k + 1) * 1000 * static_cast<double>(i) / 48000;
            const std::complex<double> basis{std::cos(angle), std::sin(angle)};
            corrected[k] += value * basis;
            naive[k] += (2 * phase - 1) * basis;
        }
        phase += 7000.0 / 48000;
        phase -= std::floor(phase);
    }
    double a = 0, b = 0;
    for (std::size_t k = 0; k < 4; ++k)
    {
        a += std::norm(corrected[k]);
        b += std::norm(naive[k]);
    }
    REQUIRE(a < b * 0.5);
}

TEST_CASE("Tensor noise is repeatable, finite and statistically bipolar", "[tensor][universe]")
{
    Harness a("excitation.white_noise", 2), b("excitation.white_noise", 2);
    a.state = {123, 456};
    b.state = a.state;
    double mean = 0, power = 0;
    for (int i = 0; i < 10000; ++i)
    {
        const float x = a.scalar({});
        REQUIRE(x == b.scalar({}));
        REQUIRE(x >= -1);
        REQUIRE(x <= 1);
        mean += x;
        power += x * x;
    }
    REQUIRE(std::abs(mean / 10000) < 0.03);
    REQUIRE(power / 10000 == Catch::Approx(1.0 / 3).margin(0.03));
}

TEST_CASE("Tensor granulation windows each independently allocated grain", "[tensor][universe]")
{
    Harness grain("excitation.granular", 7, 1000);
    grain.setAssets({{1, 1, 1, 1}});
    std::array<float, 4> p{0, 0.004f, 1, 1};
    REQUIRE(grain.scalar(p) == 0);
    p[2] = 0;
    REQUIRE(grain.scalar(p) == Catch::Approx(0).margin(0.00001));
    REQUIRE(grain.scalar(p) == Catch::Approx(0.5).margin(0.00001));
    REQUIRE(grain.scalar(p) == Catch::Approx(1).margin(0.00001));
    REQUIRE(grain.scalar(p) == Catch::Approx(0.5).margin(0.00001));
    REQUIRE(grain.scalar(p) == Catch::Approx(0).margin(0.00001));
}

TEST_CASE("Tensor biquads have low/high/band frequency selectivity", "[tensor][universe]")
{
    const std::array<float, 2> p{2000, 0.70710678f};
    const auto power = [&](std::string_view id, double hz)
    {
        Harness h(id, 4);
        double sum = 0;
        for (int i = 0; i < 12000; ++i)
        {
            const float x =
                static_cast<float>(std::sin(2 * std::numbers::pi * hz * static_cast<double>(i) / 48000));
            const double y = h.scalar(p, x);
            if (i > 2000)
                sum += y * y;
        }
        return sum / 9999;
    };
    REQUIRE(power("filter.biquad_lowpass", 200) > 0.45);
    REQUIRE(power("filter.biquad_lowpass", 12000) < 0.01);
    REQUIRE(power("filter.biquad_highpass", 200) < 0.01);
    REQUIRE(power("filter.biquad_highpass", 12000) > 0.45);
    REQUIRE(power("filter.biquad_bandpass", 2000) > 0.45);
    REQUIRE(power("filter.biquad_bandpass", 200) < 0.03);
}

TEST_CASE("Tensor modal and string primitives sustain decaying excitation responses", "[tensor][universe]")
{
    Harness mode("physical.resonator", 2), bank("physical.modal_bank", 4);
    const std::array<float, 3> p{1000, 0.01f, 0.1f};
    const std::array<float, 6> modes{1000, 0.01f, 0.1f, 3000, 0.02f, 0.05f};
    double early = 0, late = 0;
    for (int i = 0; i < 4800; ++i)
    {
        const double y = mode.scalar(p, i == 0 ? 1.0f : 0.0f);
        bank.scalar(modes, i == 0 ? 1.0f : 0.0f);
        if (i < 500)
            early += y * y;
        if (i > 4000)
            late += y * y;
    }
    REQUIRE(early > 0.01);
    REQUIRE(late < early * 0.0001);
    Harness string("physical.string_waveguide", 8);
    const std::array<float, 1> loss{0.98f};
    for (int i = 0; i < 8; ++i)
        REQUIRE(string.scalar(loss, i == 0 ? 1.0f : 0.0f) == 0);
    REQUIRE(string.scalar(loss) == 1);
    REQUIRE(string.scalar(loss) == 0);
    bool continued = false;
    for (int i = 0; i < 64; ++i)
    {
        const auto value = string.scalar(loss);
        continued = continued || std::abs(value) > 0;
    }
    REQUIRE(continued);
}

TEST_CASE("Tensor envelope stages and gesture dynamics follow explicit time parameters", "[tensor][universe]")
{
    Harness adsr("control.adsr", 4, 1000);
    std::array<float, 5> p{1, 0.01f, 0.01f, 0.5f, 0.02f};
    for (int i = 0; i < 10; ++i)
        adsr.scalar(p);
    REQUIRE(adsr.state[0] == Catch::Approx(1).margin(0.00001));
    for (int i = 0; i < 11; ++i)
        adsr.scalar(p);
    REQUIRE(adsr.state[0] == Catch::Approx(0.5).margin(0.00001));
    p[0] = 0;
    for (int i = 0; i < 21; ++i)
        adsr.scalar(p);
    REQUIRE(adsr.state[0] == 0);
    Harness stages("control.multistage_envelope", 2, 1000);
    stages.setAssets({{0.002f, 1, 0.002f, 0.25f}});
    const std::array<float, 1> gate{1};
    stages.scalar(gate);
    REQUIRE(stages.state[1] == Catch::Approx(0.5).margin(0.00001));
    stages.scalar(gate);
    REQUIRE(stages.state[1] == Catch::Approx(1).margin(0.00001));
    for (int i = 0; i < 10; ++i)
        stages.scalar(gate);
    REQUIRE(stages.state[1] == 0.25f);
    Harness follower("control.gesture_follower", 1, 1000);
    const std::array<float, 2> time{0.01f, 0.1f};
    for (int i = 0; i < 10; ++i)
        follower.scalar(time, 1);
    const double high = follower.state[0];
    REQUIRE(high == Catch::Approx(1 - std::exp(-1)).margin(0.00001));
    for (int i = 0; i < 10; ++i)
        follower.scalar(time, 0);
    REQUIRE(follower.state[0] == Catch::Approx(high * std::exp(-0.1)).margin(0.00001));
}

TEST_CASE("Tensor nonlinear operations and recurrent distortion have distinct responses",
          "[tensor][universe]")
{
    Harness saturate("nonlinear.tanh", 0), clip("nonlinear.clip", 0), cubic("nonlinear.cubic_softclip", 0),
        curve("nonlinear.waveshape_table", 0), feedback("nonlinear.feedback_tanh", 1);
    const std::array<float, 1> drive{1}, threshold{0.25f};
    REQUIRE(saturate.scalar(drive, 1) == Catch::Approx(std::tanh(1.0)));
    REQUIRE(clip.scalar(threshold, 1) == 0.25f);
    REQUIRE(cubic.scalar(drive, 0.5f) == Catch::Approx(0.6875));
    curve.setAssets({{-1, 0, 0.5f}});
    REQUIRE(curve.scalar(drive, 1) == 0.5f);
    REQUIRE(curve.scalar(drive, -1) == -1);
    const std::array<float, 2> nonlinear{1, 0.5f};
    REQUIRE(feedback.scalar(nonlinear, 1) == 0);
    REQUIRE(feedback.scalar(nonlinear) == Catch::Approx(std::tanh(1.0)));
    const float previous = feedback.state[0];
    feedback.scalar(nonlinear);
    REQUIRE(feedback.state[0] == Catch::Approx(std::tanh(0.5 * previous)));
}

TEST_CASE("Tensor direct convolution, diffusion and FDN reverb preserve actual temporal behavior",
          "[tensor][universe]")
{
    Harness convolution("time.convolution_fir", 2);
    convolution.setAssets({{1, 0.5f, -0.25f}});
    REQUIRE(convolution.scalar({}, 1) == 1);
    REQUIRE(convolution.scalar({}) == 0.5f);
    REQUIRE(convolution.scalar({}) == -0.25f);
    REQUIRE(convolution.scalar({}) == 0);
    Harness diffusion("time.diffusion_allpass", 3);
    const std::array<float, 1> feedback{0.5f};
    double energy = 0;
    for (int i = 0; i < 100; ++i)
    {
        const double y = diffusion.scalar(feedback, i == 0 ? 1.0f : 0.0f);
        energy += y * y;
    }
    REQUIRE(energy == Catch::Approx(1).margin(0.00001));
    Harness reverb("time.fdn_reverb", 8);
    reverb.setAssets({{3, 5}});
    bool early = false, late = false;
    for (int i = 0; i < 80; ++i)
    {
        const float y = reverb.scalar(feedback, i == 0 ? 1.0f : 0.0f);
        if (i < 10 && std::abs(y) > 0.00001f)
            early = true;
        if (i > 10 && std::abs(y) > 0.00001f)
            late = true;
    }
    REQUIRE(early);
    REQUIRE(late);
}

TEST_CASE("Tensor stereo and learned multichannel transforms are mathematical maps", "[tensor][universe]")
{
    Harness pan("spatial.constant_power_pan", 0);
    const std::array<float, 1> center{0}, mono{1};
    const auto output = pan.tick(center, mono);
    REQUIRE(output[0] * output[0] + output[1] * output[1] == Catch::Approx(1));
    Harness width("spatial.stereo_width", 0);
    const std::array<float, 1> narrow{0};
    const std::array<float, 2> stereo{1, -1};
    const auto collapsed = width.tick(narrow, stereo);
    REQUIRE(collapsed[0] == 0);
    REQUIRE(collapsed[1] == 0);
    Harness matrix("spatial.matrix", 2), latent("control.latent_affine", 2);
    const std::array<float, 4> weights{1, 2, -1, 1};
    const std::array<float, 2> input{0.25f, 0.5f};
    matrix.tick(weights, input);
    latent.tick(weights, {}, input);
    REQUIRE(matrix.state == latent.state);
    REQUIRE(matrix.state[0] == 1.25f);
    REQUIRE(matrix.state[1] == 0.25f);
}

TEST_CASE("Tensor phase vocoder unwraps phase and changes time and pitch explicitly", "[tensor][universe]")
{
    constexpr std::size_t bins = 16;
    Harness vocoder("pitchtime.phase_vocoder", 1 + 4 * bins);
    const std::array<float, 3> timeStretch{4, 6, 1};
    std::array<float, 2 * bins> frame{};
    for (int i = 0; i < 8; ++i)
    {
        const double analysisPhase =
            2 * std::numbers::pi * 2 * 4 * static_cast<double>(i) / static_cast<double>(bins);
        frame[4] = static_cast<float>(std::cos(analysisPhase));
        frame[5] = static_cast<float>(std::sin(analysisPhase));
        vocoder.tick(timeStretch, frame);
        const double expected =
            2 * std::numbers::pi * 2 * 6 * static_cast<double>(i) / static_cast<double>(bins);
        REQUIRE(vocoder.state[1 + 2 * bins + 4] == Catch::Approx(std::cos(expected)).margin(0.00002));
        REQUIRE(vocoder.state[1 + 2 * bins + 5] == Catch::Approx(std::sin(expected)).margin(0.00002));
    }
    Harness pitch("pitchtime.phase_vocoder", 1 + 4 * bins);
    const std::array<float, 3> pitched{4, 4, 2};
    frame.fill(0);
    frame[4] = 1;
    pitch.tick(pitched, frame);
    REQUIRE(pitch.state[1 + 2 * bins + 8] == Catch::Approx(1).margin(0.00001));
    REQUIRE(pitch.state[1 + 2 * bins + 4] == 0);
}

TEST_CASE("Tensor formant compensation retains a supplied envelope after pitching", "[tensor][universe]")
{
    Harness h("pitchtime.formant_compensation", 16);
    // Original envelope peaks at bin1; pitch-up moves the input peak to bin2.
    h.setAssets({{1, 4, 1, 1, 1, 1, 1, 4}});
    std::array<float, 16> pitched{};
    pitched[4] = 4;
    pitched[12] = 4;
    const std::array<float, 1> ratio{2};
    h.tick(ratio, pitched);
    REQUIRE(h.state[4] == 1);
    REQUIRE(h.state[12] == 1);
    // Unity ratio preserves every complex component.
    const std::array<float, 1> unity{1};
    h.tick(unity, pitched);
    REQUIRE(h.state[4] == 4);
    REQUIRE(h.state[12] == 4);
}

TEST_CASE("Tensor binaural FIR uses explicit different ear responses", "[tensor][universe]")
{
    Harness binaural("spatial.binaural_hrtf_fir", 2);
    binaural.setAssets({{1, 0.5f}, {0, 0, 0.25f}});
    const std::array<float, 1> impulse{1}, silence{0};
    auto output = binaural.tick({}, impulse);
    REQUIRE(output[0] == 1);
    REQUIRE(output[1] == 0);
    output = binaural.tick({}, silence);
    REQUIRE(output[0] == 0.5f);
    REQUIRE(output[1] == 0);
    output = binaural.tick({}, silence);
    REQUIRE(output[0] == 0);
    REQUIRE(output[1] == 0.25f);
}

TEST_CASE("Tensor room image sources create distance-delayed direct and reflected arrivals",
          "[tensor][universe]")
{
    Harness room("spatial.room_first_order", 32, 1000);
    // Metres and 1000m/s make one metre equal one sample. Direct distance=2;
    // left-wall image distance=4 with gain0.5. Other wall gains explicitly zero.
    const std::array<float, 16> geometry{10, 10, 10, 1, 5, 5, 3, 5, 5, 1000, 0.5f, 0, 0, 0, 0, 0};
    std::array<float, 7> result{};
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = room.scalar(geometry, i == 0 ? 1.0f : 0.0f);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 0);
    REQUIRE(result[2] == Catch::Approx(0.5));
    REQUIRE(result[3] == 0);
    REQUIRE(result[4] == Catch::Approx(0.125));
    REQUIRE(result[5] == 0);
}

TEST_CASE("Tensor granular stretching separates scan progression from grain playback speed",
          "[tensor][universe]")
{
    Harness h("pitchtime.granular_stretch", 14, 1000);
    h.setAssets({{0, 1, 2, 3, 4, 5, 6, 7}});
    const std::array<float, 5> p{100, 1000, 0.01f, 0.003f, 1};
    for (int i = 0; i < 30; ++i)
        h.scalar(p);
    REQUIRE(h.state[1] == Catch::Approx(0.375).margin(0.00001));
    std::size_t active = 0;
    for (std::size_t i = 2; i < h.state.size(); i += 3)
        if (h.state[i] > 0)
            ++active;
    REQUIRE(active >= 2);
    REQUIRE(active <= 4);
}

TEST_CASE("Tensor registry is extensible and refuses unknown or invalid implementations",
          "[tensor][universe]")
{
    Factory factory;
    Primitive p;
    std::string error;
    REQUIRE_FALSE(factory.create("future.unregistered", 0, p, error));
    REQUIRE_FALSE(error.empty());
    REQUIRE_FALSE(factory.create("spectral.fft", 14, p, error));
    REQUIRE_FALSE(factory.create("control.adsr", 0, p, error));
    REQUIRE(factory.create("osc.wavetable", 1, p, error));
    REQUIRE_FALSE(
        factory.add(catalog()[0], [](std::size_t, Primitive&, std::string&) { return true; }, error));
    auto metadata = catalog()[0];
    metadata.id = "extension.test";
    REQUIRE(factory.add(
        metadata, [](std::size_t n, Primitive& result, std::string& reason)
        { return createPrimitive("osc.wavetable", n, result, reason); }, error));
    REQUIRE(factory.create("extension.test", 1, p, error));
    REQUIRE(catalog().size() >= 40);
}

TEST_CASE("Tensor remaining modulation, filter and time primitives execute their stated equations",
          "[tensor][universe]")
{
    Harness impulse("excitation.impulse", 1);
    std::array<float, 2> trigger{1, 2};
    REQUIRE(impulse.scalar(trigger) == 2);
    REQUIRE(impulse.scalar(trigger) == 0);
    trigger[0] = 0;
    REQUIRE(impulse.scalar(trigger) == 0);
    trigger[0] = 1;
    REQUIRE(impulse.scalar(trigger) == 2);
    Harness mapping("control.midi_affine", 0);
    const std::array<float, 2> mapParameters{2, -0.5f};
    const std::array<float, 1> midi{0.5f};
    REQUIRE(mapping.tick(mapParameters, {}, {}, midi)[0] == 0.5f);
    Harness morph("spectral.morph", 4);
    const std::array<float, 8> frames{1, 0, 2, 0, 3, 0, 6, 0};
    const std::array<float, 1> blend{0.25f};
    morph.tick(blend, frames);
    REQUIRE(morph.state[0] == 1.5f);
    REQUIRE(morph.state[2] == 3);
    Harness svf("filter.state_variable", 2);
    const std::array<float, 2> filter{500, 1.41421356f};
    const std::array<float, 1> dc{1};
    for (int i = 0; i < 5000; ++i)
        svf.tick(filter, dc);
    const auto bands = svf.tick(filter, dc);
    REQUIRE(bands[0] == Catch::Approx(1).margin(0.0001));
    REQUIRE(bands[1] == Catch::Approx(0).margin(0.0001));
    REQUIRE(bands[2] == Catch::Approx(0).margin(0.0001));
    Harness comb("filter.feedforward_comb", 3);
    const std::array<float, 1> gain{0.5f};
    REQUIRE(comb.scalar(gain, 1) == 1);
    REQUIRE(comb.scalar(gain) == 0);
    REQUIRE(comb.scalar(gain) == 0);
    REQUIRE(comb.scalar(gain) == 0.5f);
    REQUIRE(comb.scalar(gain) == 0);
    Harness phaser("time.phaser_one_stage", 2);
    const std::array<float, 2> allpass{0, 1};
    REQUIRE(phaser.scalar(allpass, 1) == 0);
    REQUIRE(phaser.scalar(allpass) == 1);
    REQUIRE(phaser.scalar(allpass) == 0);
    Harness modulated("time.modulated_delay", 9, 1000);
    const std::array<float, 5> delay{0.003f, 0, 1, 1, 0.5f};
    for (int i = 0; i < 10; ++i)
    {
        const auto y = modulated.scalar(delay, i == 0 ? 1.0f : 0.0f);
        const double expected = i == 3 ? 1.0 : i == 6 ? 0.5 : i == 9 ? 0.25 : 0;
        REQUIRE(y == Catch::Approx(expected).margin(0.000001));
    }
}

TEST_CASE("Tensor universe metadata describes individual scalars and repeated layouts", "[tensor][universe]")
{
    const auto lookup = [](std::string_view id) -> const kernels::Metadata&
    {
        const auto list = catalog();
        const auto it = std::find_if(list.begin(), list.end(), [id](const auto& m) { return m.id == id; });
        REQUIRE(it != list.end());
        return *it;
    };
    REQUIRE(lookup("control.adsr").parameters.size() == 5);
    REQUIRE(lookup("control.adsr").parameters[3].name == "sustain");
    REQUIRE(lookup("time.modulated_delay").parameters.size() == 5);
    REQUIRE(lookup("spectral.morph").parameters.size() == 1);
    REQUIRE(lookup("spectral.tilt").parameters.size() == 2);
    REQUIRE(lookup("spectral.fft").parameters.empty());
    REQUIRE(lookup("spatial.room_first_order").parameters.size() == 16);
    REQUIRE(lookup("osc.additive").parameters.size() == 2);
    REQUIRE(lookup("osc.additive").parameters[1].validRange.find("2*i+1") != std::string_view::npos);
    REQUIRE(lookup("physical.modal_bank").parameters.size() == 3);
    REQUIRE(lookup("spatial.matrix").parameters[0].validRange.find("row*N+column") != std::string_view::npos);
    for (const auto& m : catalog())
        for (const auto& p : m.parameters)
        {
            REQUIRE_FALSE(p.name.empty());
            REQUIRE_FALSE(p.units.empty());
            REQUIRE_FALSE(p.validRange.empty());
            REQUIRE(p.trainable == (m.id != "time.partitioned_convolution.v1"));
            REQUIRE(p.controllable);
            REQUIRE(p.modulationTarget == (m.id != "time.partitioned_convolution.v1"));
        }
}

TEST_CASE("Tensor kernels reject an unrepresentable clock interval without state corruption",
          "[tensor][universe]")
{
    Harness oscillator("osc.wavetable", 1, std::numeric_limits<double>::denorm_min());
    oscillator.setAssets({{0, 1, 0, -1}});
    oscillator.state[0] = 0.25f;
    const std::array<float, 2> parameters{1, 1};
    REQUIRE(oscillator.scalar(parameters) == 0);
    REQUIRE(oscillator.state[0] == 0.25f);
}

TEST_CASE("Tensor additive partials at or above Nyquist are muted instead of aliasing", "[tensor][universe]")
{
    // 30 kHz at 48 kHz would fold back to an inharmonic 18 kHz; it must not sound at all.
    Harness above("osc.additive", 1);
    const std::array<float, 2> ultrasonic{30000, 1};
    double energy = 0;
    for (int i = 0; i < 480; ++i)
        energy += std::abs(above.scalar(ultrasonic));
    REQUIRE(energy == 0);
    // Exactly at Nyquist is muted too; a negative frequency is judged by its magnitude.
    Harness edge("osc.additive", 2);
    const std::array<float, 4> limits{24000, 1, -30000, 1};
    for (int i = 0; i < 16; ++i)
        REQUIRE(edge.scalar(limits) == 0);
    // Inside the top tenth of the band a partial fades: halfway through it is at half gain.
    Harness fade("osc.additive", 1);
    const std::array<float, 2> nearLimit{22800, 1};
    fade.state[0] = 0.25f;
    REQUIRE(fade.scalar(nearLimit) == Catch::Approx(0.5).margin(0.0001));
    // Partials below 0.45 of the update rate are untouched.
    Harness audible("osc.additive", 1);
    const std::array<float, 2> tone{21000, 1};
    audible.state[0] = 0.25f;
    REQUIRE(audible.scalar(tone) == Catch::Approx(1).margin(0.00001));
    // The limit follows the primitive's own clock, not the host rate: 2 kHz is above the
    // Nyquist of a 1 kHz update rate.
    Harness slow("osc.additive", 1, 1000);
    const std::array<float, 2> fast{2000, 1};
    slow.state[0] = 0.25f;
    REQUIRE(slow.scalar(fast) == 0);
}
