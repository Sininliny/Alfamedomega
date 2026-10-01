#include "alfa/dsp/detail/DelayLine.h"
#include "alfa/dsp/detail/FiltersBiquad.h"
#include "alfa/tensor/Universe.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

TEST_CASE("Tensor RBJ filters retain legacy impulse response at multiple local clocks", "[tensor][reuse]")
{
    using namespace alfa;
    for (const double rate : {44100.0, 48000.0, 96000.0, 192000.0})
        for (const uint64_t period : {uint64_t{1}, uint64_t{4}})
            for (const auto id :
                 {"filter.biquad_lowpass", "filter.biquad_highpass", "filter.biquad_bandpass"})
            {
                tensor::Primitive primitive;
                std::string error;
                REQUIRE(tensor::universe::createPrimitive(id, 4, primitive, error));
                const double localRate = rate / static_cast<double>(period);
                const std::array<float, 2> params{2000.0f, 0.70710678f};
                const auto design = dsp::detail::biquad::warp(params[0], params[1], localRate);
                const auto coefficients = std::string_view(id).find("lowpass") != std::string_view::npos
                                              ? dsp::detail::biquad::lowpass(design)
                                          : std::string_view(id).find("highpass") != std::string_view::npos
                                              ? dsp::detail::biquad::highpass(design)
                                              : dsp::detail::biquad::bandpass(design);
                dsp::detail::Biquad legacy;
                std::array<float, 4> state{}, next{};
                std::array<float, 1> input{}, output{};
                tensor::Step step;
                step.sampleRate = rate;
                step.clock = {period, 0};
                step.activation = 1;
                step.parameters = params;
                step.inputs.audio.values = input;
                for (size_t i = 0; i < 4096; ++i)
                {
                    input[0] = i == 0 ? 1.0f : 0.0f;
                    step.localState = state;
                    const auto expected = legacy.process(input[0], coefficients);
                    primitive.readout(step, output);
                    primitive.transition(step, next);
                    state = next;
                    REQUIRE(output[0] == Catch::Approx(expected).margin(0.00002));
                }
            }
}
TEST_CASE("Shared RBJ extraction preserves caller frequency and Q policies", "[tensor][reuse]")
{
    using namespace alfa::dsp::detail;
    const auto old = biquad::lowpass(0.1, 0.0001, 48000);
    const auto clamped = biquad::lowpass(biquad::warpUnclamped(1, 0.001, 48000));
    REQUIRE(old.b0 == clamped.b0);
    REQUIRE(old.a1 == clamped.a1);
    const auto raw = biquad::lowpass(biquad::warpUnclamped(0.1, 0.0001, 48000));
    REQUIRE(raw.b0 != old.b0);
}
TEST_CASE("Tensor asset interpolators match legacy delay interpolation with reversed time convention",
          "[tensor][reuse]")
{
    using namespace alfa;
    std::vector<float> asset(128);
    for (size_t i = 0; i < asset.size(); ++i)
        asset[i] = static_cast<float>(std::sin(2 * std::numbers::pi * 0.31 * static_cast<double>(i)));
    dsp::detail::DelayLine line;
    line.prepare(128);
    for (const auto value : asset)
        line.write(value);
    const std::array<tensor::TensorView, 1> assets{{{{}, {}, asset, {}}}};
    const std::array<float, 2> params{0, 1};
    for (const auto id : {"pitchtime.asset_resample_cubic", "pitchtime.asset_resample_sinc8"})
    {
        tensor::Primitive primitive;
        std::string error;
        REQUIRE(tensor::universe::createPrimitive(id, 1, primitive, error));
        std::array<float, 1> state{}, next{}, output{};
        tensor::Step step;
        step.sampleRate = 48000;
        step.clock = {1, 0};
        step.activation = 1;
        step.parameters = params;
        step.assets = assets;
        for (int index = 8; index < 110; ++index)
            for (const float fraction : {0.0f, 0.125f, 0.5f, 0.875f})
            {
                const float position = static_cast<float>(index) + fraction;
                state[0] = position / static_cast<float>(asset.size());
                step.localState = state;
                primitive.readout(step, output);
                primitive.transition(step, next);
                const float delay = static_cast<float>(asset.size() - 1) - position;
                const float expected = std::string_view(id).find("cubic") != std::string_view::npos
                                           ? line.readCubic(delay)
                                           : line.readSinc(delay);
                REQUIRE(output[0] == Catch::Approx(expected).margin(0.000002));
                REQUIRE(next[0] == state[0]);
            }
        state[0] = -0.1f;
        step.localState = state;
        primitive.readout(step, output);
        REQUIRE(output[0] == 0);
        state[0] = 1.0f;
        primitive.readout(step, output);
        REQUIRE(output[0] == 0);
    }
}
TEST_CASE("Legacy sinc tensor interpolation retains high-band fractional tone accuracy", "[tensor][reuse]")
{
    using namespace alfa::tensor;
    std::array<float, 256> data{};
    constexpr double frequency = 0.3;
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>(std::sin(2 * std::numbers::pi * frequency * static_cast<double>(i)));
    const std::array<TensorView, 1> assets{{{{}, {}, data, {}}}};
    const std::array<float, 2> parameters{0, 1};
    std::array<float, 1> state{}, output{};
    Step step;
    step.sampleRate = 96000;
    step.clock = {1, 0};
    step.activation = 1;
    step.parameters = parameters;
    step.assets = assets;
    Primitive sinc, linear;
    std::string error;
    REQUIRE(universe::createPrimitive("pitchtime.asset_resample_sinc8", 1, sinc, error));
    REQUIRE(universe::createPrimitive("pitchtime.asset_resample", 1, linear, error));
    double sincError = 0, linearError = 0;
    for (size_t i = 8; i < 240; ++i)
    {
        state[0] = (static_cast<float>(i) + 0.5f) / static_cast<float>(data.size());
        step.localState = state;
        const double expected = std::sin(2 * std::numbers::pi * frequency * (static_cast<double>(i) + 0.5));
        sinc.readout(step, output);
        sincError += std::pow(output[0] - expected, 2);
        linear.readout(step, output);
        linearError += std::pow(output[0] - expected, 2);
    }
    REQUIRE(sincError < linearError * 0.05);
}
