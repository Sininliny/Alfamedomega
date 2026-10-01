#include "alfa/dsp/detail/NonlinearOversampler.h"
#include "alfa/tensor/NonlinearProduction.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <complex>
#include <iostream>
#include <numbers>

namespace
{
using namespace alfa::tensor;
template <class S>
struct Engine
{
    PrimitiveT<S> primitive;
    ConfigurationT<S> configuration;
    RuntimeT<S> runtime;
    std::array<S, 1> audio{};
    InputsT<S> inputs;
    Engine(std::string_view shape, int factor, Clock clock = {1, 0}, double rate = 48000)
    {
        const auto n = factor == 1 ? 0 : nonlinear::stateElements(24, factor == 4 ? 12 : 0);
        const std::string id = "nonlinear." + std::string(shape) +
                               (factor == 1   ? ".v1"
                                : factor == 2 ? "_polyphase2x.v1"
                                              : "_polyphase4x.v1");
        std::string error;
        REQUIRE(nonlinear::createPrimitive(id, n, primitive, error));
        configuration.activation = {{"instance"}, {1}, {S(1)}};
        configuration.parameters = {{"parameter"}, {3}, {S(4), S(0), S(1)}};
        if (shape == "table")
            configuration.parameters = {{"parameter"}, {5}, {S(1), S(0), S(1), S(-20), S(20)}};
        configuration.routing = {{"weight"}, {0}, {}};
        configuration.modulation = configuration.routing;
        configuration.instances = {
            {&primitive, 0, {0, configuration.parameters.values.size()}, clock, OutputBetweenUpdates::Hold}};
        auto& assets = configuration.instances[0].assets;
        if (factor > 1)
            assets.push_back(nonlinear::designHalfband<S>(11, 7.86, error));
        if (factor == 4)
            assets.push_back(nonlinear::designHalfband<S>(5, 6.76, error));
        if (shape == "table")
            assets.push_back({{"vertex"}, {2}, {S(-20), S(20)}});
        configuration.initialState = {std::vector<S>(n, S(0))};
        std::size_t storage = 1 + configuration.parameters.values.size();
        for (const auto& a : assets)
            storage += a.values.size();
        REQUIRE(runtime.prepare(configuration,
                                Budget{1, n, configuration.parameters.values.size(), 0, 0, 1, storage, 0,
                                       primitive.sparseWriteCapacity},
                                rate, error));
        inputs.audio.values = audio;
    }
    S tick(S x)
    {
        audio[0] = x;
        REQUIRE(runtime.tick(inputs));
        return runtime.output()[0];
    }
};
void fft(std::vector<std::complex<double>>& data)
{
    const auto n = data.size();
    for (std::size_t i = 1, j = 0; i < n; ++i)
    {
        auto bit = n / 2;
        for (; j & bit; bit /= 2)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(data[i], data[j]);
    }
    for (std::size_t length = 2; length <= n; length *= 2)
    {
        const auto rotation = std::polar(1.0, -2 * std::numbers::pi / static_cast<double>(length));
        for (std::size_t base = 0; base < n; base += length)
        {
            std::complex<double> phase = 1;
            for (std::size_t j = 0; j < length / 2; ++j)
            {
                const auto a = data[base + j], b = phase * data[base + j + length / 2];
                data[base + j] = a + b;
                data[base + j + length / 2] = a - b;
                phase *= rotation;
            }
        }
    }
}
template <class S>
double aliasPower(int factor)
{
    Engine<S> e("clip", factor);
    constexpr std::size_t count = 8192, bin = 1536; // exact 9 kHz at 48 kHz
    std::vector<std::complex<double>> spectrum(count);
    for (std::size_t i = 0; i < count * 2; ++i)
    {
        const auto x = static_cast<S>(
            std::sin(2 * std::numbers::pi * static_cast<double>(bin * i) / static_cast<double>(count)));
        const auto y = e.tick(x);
        if (i >= count)
            spectrum[i - count] = static_cast<double>(y);
    }
    fft(spectrum);
    // 3rd,5th,7th harmonics fold to21,3,15kHz. They are absent from the
    // bandlimited ideal output: only the9kHz fundamental lies below Nyquist.
    const double aliases = std::norm(spectrum[3584]) + std::norm(spectrum[512]) + std::norm(spectrum[2560]);
    return aliases / std::norm(spectrum[bin]);
}
template <class S>
void latencyAndSegmentation()
{
    for (const auto factor : {1, 2, 4})
    {
        Engine<S> linear("table", factor);
        double sum = 0, moment = 0;
        for (std::size_t i = 0; i < 256; ++i)
        {
            const double y = static_cast<double>(linear.tick(i == 0 ? S(1) : S(0)));
            sum += y;
            moment += static_cast<double>(i) * y;
        }
        REQUIRE(sum == Catch::Approx(1).margin(0.000002));
        REQUIRE(moment / sum ==
                Catch::Approx(factor == 1 ? 0 : nonlinear::latencyUpdates(24, factor == 4 ? 12 : 0))
                    .margin(0.00001));
        Engine<S> a("tanh", factor), b("tanh", factor);
        std::vector<S> oneBlock;
        for (std::size_t i = 0; i < 512; ++i)
            oneBlock.push_back(a.tick(static_cast<S>(std::sin(static_cast<double>(i) * 0.231))));
        for (std::size_t base = 0; base < 512; base += 17)
            for (std::size_t i = base; i < std::min(base + 17, std::size_t(512)); ++i)
                REQUIRE(b.tick(static_cast<S>(std::sin(static_cast<double>(i) * 0.231))) == oneBlock[i]);
        a.runtime.reset();
        for (std::size_t i = 0; i < 512; ++i)
            REQUIRE(a.tick(static_cast<S>(std::sin(static_cast<double>(i) * 0.231))) == oneBlock[i]);
    }
}
} // namespace
TEST_CASE("Production nonlinear halfband design preserves legacy float coefficients and true double assets",
          "[tensor][nonlinear-production]")
{
    std::string error;
    alfa::dsp::detail::NonlinearHalfband legacy;
    legacy.design(11, 7.86);
    auto f = nonlinear::designHalfband<float>(11, 7.86, error);
    auto d = nonlinear::designHalfband<double>(11, 7.86, error);
    bool morePrecision = false;
    for (std::size_t i = 0; i < f.values.size(); ++i)
    {
        REQUIRE(f.values[i] == legacy.dense[i]);
        morePrecision = morePrecision || static_cast<double>(f.values[i]) != d.values[i];
    }
    REQUIRE(morePrecision);
    REQUIRE(nonlinear::designHalfband<float>(12, 7.86, error).values.empty());
    REQUIRE_FALSE(error.empty());
    REQUIRE(nonlinear::catalog().size() == 9);
}
TEST_CASE("Production polyphase2x agrees with retained legacy resampler equations",
          "[tensor][nonlinear-production]")
{
    Engine<float> e("tanh", 2);
    alfa::dsp::detail::NonlinearHalfband h;
    h.design(11, 7.86);
    alfa::dsp::detail::NonlinearUpsampler2x<11> up;
    alfa::dsp::detail::NonlinearDownsampler2x<11> down;
    up.setKernel(&h);
    down.setKernel(&h);
    for (std::size_t i = 0; i < 4096; ++i)
    {
        const auto input = static_cast<float>(std::sin(static_cast<double>(i) * 0.919));
        float a = 0, b = 0;
        up.process(input, a, b);
        const auto reference = down.process(std::tanh(4.0f * a), std::tanh(4.0f * b));
        REQUIRE(e.tick(input) == Catch::Approx(reference).margin(0.000002));
    }
}
TEST_CASE("Production polyphase4x agrees with explicitly unaligned legacy stage cascade",
          "[tensor][nonlinear-production]")
{
    Engine<float> e("tanh", 4);
    alfa::dsp::detail::NonlinearHalfband h1, h2;
    h1.design(11, 7.86);
    h2.design(5, 6.76);
    alfa::dsp::detail::NonlinearUpsampler2x<11> up1;
    alfa::dsp::detail::NonlinearDownsampler2x<11> down1;
    alfa::dsp::detail::NonlinearUpsampler2x<5> up2;
    alfa::dsp::detail::NonlinearDownsampler2x<5> down2;
    up1.setKernel(&h1);
    down1.setKernel(&h1);
    up2.setKernel(&h2);
    down2.setKernel(&h2);
    for (std::size_t i = 0; i < 2048; ++i)
    {
        const auto input = static_cast<float>(std::sin(static_cast<double>(i) * 1.191));
        float a = 0, b = 0, c = 0, d = 0;
        up1.process(input, a, b);
        up2.process(a, c, d);
        const auto even = down2.process(std::tanh(4.0f * c), std::tanh(4.0f * d));
        up2.process(b, c, d);
        const auto odd = down2.process(std::tanh(4.0f * c), std::tanh(4.0f * d));
        // The old product additionally inserted half an update for dry/wet alignment.
        // This primitive authors no mix path or additional alignment state.
        REQUIRE(e.tick(input) == Catch::Approx(down1.process(even, odd)).margin(0.000003));
    }
}
TEST_CASE("Production oversampling reduces measured folded nonlinear harmonics",
          "[tensor][nonlinear-production]")
{
    for (const bool useDouble : {false, true})
    {
        const double plain = useDouble ? aliasPower<double>(1) : aliasPower<float>(1);
        const double two = useDouble ? aliasPower<double>(2) : aliasPower<float>(2);
        const double four = useDouble ? aliasPower<double>(4) : aliasPower<float>(4);
        const auto db2 = 10 * std::log10(two / plain), db4 = 10 * std::log10(four / plain);
        std::cout << (useDouble ? "float64" : "float32") << " folded alias reduction 2x=" << db2
                  << "dB 4x=" << db4 << "dB\n";
        REQUIRE(db2 < -10);
        REQUIRE(db4 < -10);
    }
}
TEST_CASE("Production nonlinear latency reset and segmentation are explicit in both scalar types",
          "[tensor][nonlinear-production]")
{
    latencyAndSegmentation<float>();
    latencyAndSegmentation<double>();
}
TEST_CASE("Production nonlinear invalid events rollback and independent clocks preserve histories",
          "[tensor][nonlinear-production]")
{
    Engine<double> e("clip", 4, {3, 1}, 96000);
    REQUIRE(e.tick(0.2) == 0); // not due
    e.tick(0.2);               // first scheduled update
    auto state = std::vector<double>(e.runtime.state().begin(), e.runtime.state().end());
    e.tick(0.2);
    e.tick(0.2); // not due
    REQUIRE(std::equal(state.begin(), state.end(), e.runtime.state().begin()));
    const auto sample = e.runtime.sample();
    e.audio[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_FALSE(e.runtime.tick(e.inputs));
    REQUIRE(e.runtime.sample() == sample);
    REQUIRE(std::equal(state.begin(), state.end(), e.runtime.state().begin()));
    e.tick(0.2);
    REQUIRE(e.runtime.sample() == sample + 1);
    Primitive64 p;
    std::string error;
    REQUIRE_FALSE(nonlinear::createPrimitive("nonlinear.clip_polyphase4x.v1", 1, p, error));
    REQUIRE_FALSE(nonlinear::createPrimitive("nonlinear.unknown.v1", 0, p, error));
}
TEST_CASE("Production parameters accept audio-rate drive bias gain and authored table coordinates",
          "[tensor][nonlinear-production]")
{
    Engine<double> e("tanh", 1);
    const auto view = [](const Tensor64& t) -> TensorView64
    {
        return {t.axes, t.shape, t.values, t.presence};
    };
    for (std::size_t i = 0; i < 128; ++i)
    {
        const auto drive = static_cast<double>(i) / 32, bias = std::sin(static_cast<double>(i) * 0.19),
                   gain = 0.3;
        e.configuration.parameters.values = {drive, bias, gain};
        REQUIRE(e.runtime.updateTensors(view(e.configuration.activation), view(e.configuration.parameters),
                                        view(e.configuration.routing), view(e.configuration.modulation)));
        REQUIRE(e.tick(0.2) == Catch::Approx(gain * std::tanh(drive * 0.2 + bias)).margin(1e-14));
    }
    Engine<double> table("table", 1);
    REQUIRE(table.tick(25) == 20); // explicitly clamped table endpoint
    REQUIRE(table.tick(-25) == -20);
    REQUIRE(table.tick(0.25) == Catch::Approx(0.25).margin(1e-14));
    const auto old = table.runtime.sample();
    table.configuration.parameters.values[4] = table.configuration.parameters.values[3];
    REQUIRE(table.runtime.updateTensors(
        view(table.configuration.activation), view(table.configuration.parameters),
        view(table.configuration.routing), view(table.configuration.modulation)));
    table.audio[0] = 0;
    REQUIRE_FALSE(table.runtime.tick(table.inputs));
    REQUIRE(table.runtime.sample() == old);
}
