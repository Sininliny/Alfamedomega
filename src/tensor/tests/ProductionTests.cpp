#include "alfa/tensor/Production.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
using namespace alfa::tensor;
template <typename S>
struct Harness
{
    PrimitiveT<S> primitive;
    std::vector<S> state, output;
    std::vector<StateWriteT<S>> journal;
    std::vector<S> coefficients;
    std::array<TensorViewT<S>, 1> assets{};
    Harness(std::string_view id, std::size_t size)
        : state(size)
        , output(id == "time.fdn_hadamard8.v1" ? 8 : 1)
    {
        std::string error;
        REQUIRE(production::createPrimitive(id, size, primitive, error));
        journal.resize(primitive.sparseWriteCapacity);
    }
    bool tick(std::span<const S> parameters, std::span<const S> audio = {})
    {
        StepT<S> s;
        s.activation = 1;
        s.sampleRate = 192000;
        s.clock = {1, 0};
        s.localState = state;
        s.parameters = parameters;
        s.inputs.audio.values = audio;
        if (!coefficients.empty())
        {
            assets[0].values = coefficients;
            s.assets = assets;
        }
        primitive.readout(s, output);
        const auto count = primitive.sparseTransition(s, journal);
        if (count > journal.size())
            return false;
        for (const auto value : output)
            if (!std::isfinite(value))
                return false;
        for (std::size_t i = 0; i < count; ++i)
        {
            if (journal[i].element >= state.size() || !std::isfinite(journal[i].value))
                return false;
            for (std::size_t j = 0; j < i; ++j)
                if (journal[j].element == journal[i].element)
                    return false;
        }
        for (std::size_t i = 0; i < count; ++i)
            state[journal[i].element] = journal[i].value;
        return true;
    }
    S scalar(std::array<S, 2> parameters, S input = 0)
    {
        const std::array<S, 1> audio{input};
        REQUIRE(tick(parameters, audio));
        return output[0];
    }
};
template <typename S>
void delayCheck()
{
    Harness<S> h("time.ring_delay.v1", 7);
    std::array<S, 9> expected{0, 0, 0, 1, 0, 0, S(0.5), 0, 0};
    for (std::size_t i = 0; i < expected.size(); ++i)
        REQUIRE(h.scalar({3, S(0.5)}, i == 0 ? S(1) : S(0)) == expected[i]);
    REQUIRE(h.primitive.sparseWriteCapacity == 3);
    REQUIRE(h.primitive.transition == nullptr);
    std::fill(h.state.begin(), h.state.end(), S(0));
    REQUIRE(h.scalar({3, 0}, 1) == 0);
    REQUIRE(h.scalar({3, 0}) == 0);
    REQUIRE(h.scalar({3, 0}) == 0);
    REQUIRE(h.scalar({3, 0}) == 1);
}
template <typename S>
void allpassCheck()
{
    Harness<S> h("time.ring_allpass.v1", 7);
    double energy = 0;
    std::array<double, 12> first{};
    for (std::size_t i = 0; i < 1200; ++i)
    {
        const auto sample = h.scalar({3, S(0.5)}, i == 0 ? S(1) : S(0));
        energy += static_cast<double>(sample) * sample;
        if (i < first.size())
            first[i] = sample;
    }
    REQUIRE(first[0] == -0.5);
    REQUIRE(first[3] == 0.75);
    REQUIRE(first[6] == 0.375);
    REQUIRE(first[9] == 0.1875);
    REQUIRE(energy == Catch::Approx(1).margin(0.000001));
}
template <typename S>
void firCheck()
{
    Harness<S> h("time.causal_fir_ring.v1", 5);
    h.coefficients = {S(0.2), S(-0.4), S(0.6), S(0.8)};
    for (std::size_t i = 0; i < 12; ++i)
    {
        const std::array<S, 1> input{i == 0 ? S(1) : S(0)};
        REQUIRE(h.tick({}, input));
        REQUIRE(h.output[0] == (i < h.coefficients.size() ? h.coefficients[i] : S(0)));
    }
}
} // namespace
TEST_CASE("Production ring delay has exact onset recurrence wrap and reset", "[tensor][production]")
{
    delayCheck<float>();
    delayCheck<double>();
}
TEST_CASE("Production allpass preserves impulse energy at fixed integer delay", "[tensor][production]")
{
    allpassCheck<float>();
    allpassCheck<double>();
}
TEST_CASE("Production causal FIR includes current tap without extra latency", "[tensor][production]")
{
    firCheck<float>();
    firCheck<double>();
}
TEST_CASE("Production cursors cross16-bit limb boundary exactly in both precisions", "[tensor][production]")
{
    auto check = []<typename S>()
    {
        Harness<S> h("time.ring_delay.v1", 65540);
        h.state[0] = 65535;
        h.state[65536] = S(0.25);
        REQUIRE(h.scalar({1, 0}, S(0.75)) == S(0.25));
        REQUIRE(h.state[0] == 0);
        REQUIRE(h.state[1] == 1);
        REQUIRE(h.state[65537] == S(0.75));
        REQUIRE(h.scalar({1, 0}) == S(0.75));
        REQUIRE(h.state[0] == 1);
        REQUIRE(h.state[1] == 1);
    };
    check.operator()<float>();
    check.operator()<double>();
}
TEST_CASE("Production cubic modulation follows independent offline reference", "[tensor][production]")
{
    Harness<double> h("time.ring_delay_cubic.v1", 35);
    std::vector<double> history(160, 0);
    const auto reference = [&](int time, int delay)
    {
        const auto index = time - delay;
        return index >= 0 ? history[static_cast<std::size_t>(index)] : 0.0;
    };
    for (int i = 0; i < 160; ++i)
    {
        const auto delay = 5.0 + 0.8 * std::sin(0.07 * i);
        const auto integer = static_cast<int>(std::floor(delay));
        const auto fraction = delay - integer;
        const auto a = reference(i, integer - 1), b = reference(i, integer), c = reference(i, integer + 1),
                   d = reference(i, integer + 2);
        const auto expected =
            b + 0.5 * fraction *
                    (c - a + fraction * (2 * a - 5 * b + 4 * c - d + fraction * (3 * (b - c) + d - a)));
        const auto input = i == 0 ? 1.0 : 0.2 * std::sin(0.13 * i);
        REQUIRE(h.scalar({delay, 0.7}, input) == Catch::Approx(expected).margin(1e-12));
        history[static_cast<std::size_t>(i)] = input + 0.7 * expected;
    }
}
TEST_CASE("Production FDN uses orthogonal feedback with bounded tail in f32 and f64", "[tensor][production]")
{
    auto check = []<typename S>()
    {
        Harness<S> h("time.fdn_hadamard8.v1", 8 * (11 + 3));
        std::array<S, 24> params{};
        for (std::size_t i = 0; i < 8; ++i)
        {
            params[3 * i] = static_cast<S>(i + 1);
            params[3 * i + 1] = S(0.7);
            params[3 * i + 2] = S(0.1);
        }
        double total = 0, last = 0;
        for (std::size_t t = 0; t < 2000; ++t)
        {
            std::array<S, 8> input{};
            if (t == 0)
                input[0] = 1;
            REQUIRE(h.tick(params, input));
            double frame = 0;
            for (const auto value : h.output)
                frame += static_cast<double>(value) * value;
            total += frame;
            if (t > 1900)
                last += frame;
        }
        REQUIRE(total > 0.5);
        REQUIRE(total < 10);
        REQUIRE(last < 1e-12);
        REQUIRE(h.primitive.sparseWriteCapacity == 32);
    };
    check.operator()<float>();
    check.operator()<double>();
}
TEST_CASE("Production invalid controls or cursors leave history intact", "[tensor][production]")
{
    Harness<double> h("time.ring_delay.v1", 9);
    h.scalar({3, 0.5}, 1);
    const auto saved = h.state;
    for (const auto bad : std::array<std::array<double, 2>, 4>{
             {{0, 0}, {3, 1}, {3.5, 0}, {3, std::numeric_limits<double>::quiet_NaN()}}})
    {
        const std::array<double, 1> input{0};
        REQUIRE_FALSE(h.tick(bad, input));
        REQUIRE(h.state == saved);
        REQUIRE(h.output[0] == 0);
    }
    h.state[0] = 0.5;
    const auto invalid = h.state;
    REQUIRE_FALSE(h.tick(std::array<double, 2>{3, 0}));
    REQUIRE(h.state == invalid);
    Primitive64 descriptor;
    std::string error;
    REQUIRE_FALSE(production::createPrimitive("time.ring_delay_cubic.v1", 5, descriptor, error));
    REQUIRE_FALSE(production::createPrimitive("time.fdn_hadamard8.v1", 31, descriptor, error));
}
TEST_CASE("Production ring state work is independent of history size and block segmentation",
          "[tensor][production]")
{
    Harness<double> whole("time.ring_delay.v1", 200002), segmented("time.ring_delay.v1", 200002);
    const std::array<double, 2> params{31, 0.91};
    std::vector<double> reference;
    for (int t = 0; t < 1000; ++t)
        reference.push_back(whole.scalar(params, t == 0 ? 1.0 : 0.0));
    std::size_t offset = 0;
    for (const auto block : {17, 127, 3, 251, 602})
        for (int t = 0; t < block; ++t)
        {
            REQUIRE(segmented.scalar(params, offset == 0 ? 1.0 : 0.0) == reference[offset]);
            ++offset;
        }
    REQUIRE(whole.state == segmented.state);
    REQUIRE(whole.journal.size() == 3);
}
TEST_CASE("Production64 FIR retains scalar distinctions unavailable in f32", "[tensor][production]")
{
    Harness<double> h("time.causal_fir_ring.v1", 3);
    h.coefficients = {1.0 + 1e-12};
    const std::array<double, 1> input{1};
    REQUIRE(h.tick({}, input));
    REQUIRE(h.output[0] > 1.0);
    REQUIRE(h.output[0] == h.coefficients[0]);
    REQUIRE(static_cast<float>(h.output[0]) == 1.0f);
}
TEST_CASE("Production sparse runtime validates budgets commits atomically and resets both precisions",
          "[tensor][production]")
{
    auto check = []<typename S>()
    {
        PrimitiveT<S> primitive;
        std::string error;
        REQUIRE(production::createPrimitive("time.ring_delay.v1", 100002, primitive, error));
        ConfigurationT<S> config;
        config.activation = {{"primitive"}, {1}, {1}};
        config.parameters = {{"parameter"}, {2}, {7, S(0.5)}};
        config.routing = {{"weight"}, {0}, {}};
        config.modulation = {{"weight"}, {0}, {}};
        config.instances.push_back({&primitive, 0, {0, 2}, {1, 0}});
        config.initialState.push_back(std::vector<S>(primitive.stateElements));
        Budget budget{1, primitive.stateElements, 2, 0, 0, 1, 3, 0, 3};
        RuntimeT<S> runtime;
        REQUIRE(runtime.prepare(config, budget, 192000, error));
        const std::array<std::string, 1> axes{"channel"};
        const std::array<std::size_t, 1> shape{1};
        std::array<S, 1> audio{1};
        InputsT<S> inputs;
        inputs.audio = {axes, shape, audio};
        REQUIRE(runtime.tick(inputs));
        audio[0] = 0;
        for (int t = 1; t < 8; ++t)
            REQUIRE(runtime.tick(inputs));
        REQUIRE(runtime.output()[0] == 1);
        const std::vector<S> saved(runtime.state().begin(), runtime.state().end());
        auto invalid = config.parameters;
        invalid.values[1] = 1;
        const auto view = [](const TensorT<S>& tensor)
        {
            return TensorViewT<S>{tensor.axes, tensor.shape, tensor.values, tensor.presence};
        };
        REQUIRE(runtime.updateTensors(view(config.activation), view(invalid), view(config.routing),
                                      view(config.modulation)));
        REQUIRE_FALSE(runtime.tick(inputs));
        REQUIRE(std::equal(saved.begin(), saved.end(), runtime.state().begin()));
        runtime.reset();
        REQUIRE(
            std::all_of(runtime.state().begin(), runtime.state().end(), [](S value) { return value == 0; }));
        REQUIRE(runtime.tick(inputs));
        budget.sparseStateWrites = 2;
        RuntimeT<S> limited;
        REQUIRE_FALSE(limited.prepare(config, budget, 192000, error));
    };
    check.operator()<float>();
    check.operator()<double>();
}
TEST_CASE("Production sparse ring resource measurement fixture", "[.production-benchmark]")
{
    auto measure = []<typename S>()
    {
        Harness<S> h("time.ring_delay.v1", 384002);
        const std::array<S, 2> params{192000, S(0.9)};
        bool valid = true;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t t = 0; t < 192000; ++t)
        {
            const std::array<S, 1> audio{t == 0 ? S(1) : S(0)};
            valid = h.tick(params, audio) && valid;
        }
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        REQUIRE(valid);
        std::cout << "Production ring callbacks " << sizeof(S) * 8
                  << " bits: 192000 updates, state bytes=" << h.state.size() * sizeof(S)
                  << ", journal entries=" << h.journal.size() << ", elapsed_ms=" << elapsed << '\n';
    };
    measure.operator()<float>();
    measure.operator()<double>();
}
