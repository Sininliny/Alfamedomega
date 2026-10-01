#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Kernels.h"
#include "alfa/tensor/Precision.h"
#include "alfa/tensor/StateInteger.h"
#include "alfa/tensor/Universe.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numbers>
#include <sstream>
#include <type_traits>

namespace
{
template <class Scalar>
alfa::tensor::ConfigurationT<Scalar> configuration(alfa::tensor::PrimitiveT<Scalar>& primitive)
{
    using namespace alfa::tensor;
    ConfigurationT<Scalar> c;
    c.activation = {{"instance"}, {1}, {Scalar(1)}};
    c.parameters = {{"parameter"}, {1}, {Scalar(0.8)}};
    c.routing = {{"weight"}, {0}, {}};
    c.modulation = c.routing;
    c.instances = {{&primitive, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero}};
    c.initialState = {{Scalar(0)}};
    return c;
}
template <class Scalar>
double physicalError()
{
    using namespace alfa::tensor;
    PrimitiveT<Scalar> primitive;
    std::string error;
    REQUIRE(universe::createPrimitive("physical.resonator", 2, primitive, error));
    const std::array<Scalar, 3> params{Scalar(40), Scalar(30), Scalar(1)};
    std::array<Scalar, 2> state{}, next{};
    std::array<Scalar, 1> input{}, output{};
    StepT<Scalar> step;
    step.sampleRate = 192000;
    step.clock = {1, 0};
    step.activation = 1;
    step.parameters = params;
    step.inputs.audio.values = input;
    const double r = std::exp(-1 / (192000.0 * 30)), w = 2 * std::numbers::pi * 40 / 192000.0;
    double sumError = 0, sumReference = 0;
    for (size_t i = 0; i < 384000; ++i)
    {
        input[0] = i == 0 ? Scalar(1) : Scalar(0);
        step.localState = state;
        primitive.readout(step, output);
        primitive.transition(step, next);
        state = next;
        const double expected =
            std::pow(r, static_cast<double>(i)) * std::sin((static_cast<double>(i) + 1) * w) / std::sin(w);
        const double difference = output[0] - expected;
        sumError += difference * difference;
        sumReference += expected * expected;
    }
    return std::sqrt(sumError / sumReference);
}
template <class Scalar>
void exactInteger()
{
    using namespace alfa::tensor;
    std::array<Scalar, 4> limbs{};
    uint64_t restored = 0;
    for (const uint64_t value : {uint64_t{0}, uint64_t{0xFFFFFFFF}, std::numeric_limits<uint64_t>::max(),
                                 uint64_t{0xFEDCBA9876543210}})
    {
        REQUIRE(encodeStateInteger<Scalar>(value, limbs));
        REQUIRE(decodeStateInteger<Scalar>(limbs, restored));
        REQUIRE(restored == value);
    }
    limbs[0] = Scalar(0.5);
    REQUIRE_FALSE(decodeStateInteger<Scalar>(limbs, restored));
    limbs[0] = std::numeric_limits<Scalar>::quiet_NaN();
    REQUIRE_FALSE(decodeStateInteger<Scalar>(limbs, restored));
    std::array<Scalar, 2> shortState{};
    REQUIRE_FALSE(encodeStateInteger<Scalar>(std::numeric_limits<uint64_t>::max(), shortState));
}
} // namespace
TEST_CASE("One tier selects one genuine scalar specialization", "[tensor][precision]")
{
    using namespace alfa::tensor;
    PrecisionProfile profile;
    std::string error;
    REQUIRE(resolvePrecisionJSON("{}", profile, error));
    REQUIRE(profile.scalarType == ScalarType::Float32);
    for (const auto tier : {"Lite", "Standard", "Pro", "Studio"})
    {
        REQUIRE(resolvePrecisionJSON(nlohmann::json{{"tier", tier}}.dump(), profile, error));
        REQUIRE(profile.scalarType ==
                (std::string_view(tier) == "Lite" || std::string_view(tier) == "Standard"
                     ? ScalarType::Float32
                     : ScalarType::Float64));
    }
    REQUIRE_FALSE(resolvePrecisionJSON(R"({"tier":"Pro","scalar_type":"float32"})", profile, error));
    REQUIRE_FALSE(resolvePrecisionJSON(R"({"scalar_type":"float16"})", profile, error));
    REQUIRE_FALSE(resolvePrecisionJSON(R"({"scalar_type":"int8"})", profile, error));
    auto p = kernels::feedbackCell<double>();
    auto c = configuration(p);
    const auto serialized = serializeJSON(c);
    Configuration64 restored;
    REQUIRE(parseJSON(serialized, restored, error));
    Configuration narrow;
    REQUIRE_FALSE(parseJSON(serialized, narrow, error));
}
TEST_CASE("Prepared f64 numeric storage is uniformly double and reports actual storage",
          "[tensor][precision]")
{
    using namespace alfa::tensor;
    auto p32 = kernels::feedbackCell<float>();
    auto p64 = kernels::feedbackCell<double>();
    auto c32 = configuration(p32);
    auto c64 = configuration(p64);
    const Budget budget{1, 1, 1, 0, 0, 1, 2, 0};
    Runtime r32;
    Runtime64 r64;
    std::string error;
    REQUIRE(r32.prepare(c32, budget, 48000, error));
    REQUIRE(r64.prepare(c64, budget, 48000, error));
    STATIC_REQUIRE(std::is_same_v<decltype(r64.state())::value_type, double>);
    STATIC_REQUIRE(std::is_same_v<decltype(c64.parameters.values)::value_type, double>);
    REQUIRE(r64.numericStorageBytes() == 2 * r32.numericStorageBytes());
    const std::array<std::string, 1> axes{"channel"};
    const std::array<size_t, 1> shape{1};
    std::array<float, 1> a32{1};
    std::array<double, 1> a64{1};
    Inputs x32;
    Inputs64 x64;
    x32.audio = {axes, shape, a32, {}};
    x64.audio = {axes, shape, a64, {}};
    for (int i = 0; i < 128; ++i)
    {
        REQUIRE(r32.tick(x32));
        REQUIRE(r64.tick(x64));
        REQUIRE(r64.state()[0] == Catch::Approx((1 - std::pow(0.8, i + 1)) / 0.2).epsilon(1e-12));
        a32[0] = 1;
        a64[0] = 1;
    }
}
TEST_CASE("Double assets do not narrow through primitive callbacks or prepared sinc tables",
          "[tensor][precision]")
{
    using namespace alfa::tensor;
    Primitive64 primitive;
    std::string error;
    REQUIRE(universe::createPrimitive("osc.wavetable", 1, primitive, error));
    const double value = 1 + std::ldexp(1.0, -40);
    const std::array<double, 4> data{0, value, 0, -value};
    const std::array<TensorView64, 1> assets{{{{}, {}, data, {}}}};
    const std::array<double, 2> params{0, 1};
    const std::array<double, 1> state{0.25};
    std::array<double, 1> output{};
    Step64 step;
    step.sampleRate = 96000;
    step.clock = {1, 0};
    step.activation = 1;
    step.assets = assets;
    step.parameters = params;
    step.localState = state;
    primitive.readout(step, output);
    REQUIRE(output[0] == value);
    REQUIRE(output[0] != static_cast<double>(static_cast<float>(value)));
    REQUIRE(universe::createPrimitive("pitchtime.asset_resample_sinc8", 1, primitive, error));
    std::array<double, 32> constant;
    constant.fill(value);
    const std::array<TensorView64, 1> sincAssets{{{{}, {}, constant, {}}}};
    const std::array<double, 1> mid{0.515625};
    step.localState = mid;
    step.assets = sincAssets;
    primitive.readout(step, output);
    REQUIRE(output[0] == Catch::Approx(value).margin(1e-14));
}
TEST_CASE("Uniform scalar state preserves full uint64 RNG and cursor identities", "[tensor][precision]")
{
    exactInteger<float>();
    exactInteger<double>();
}
TEST_CASE("Long low-frequency physical memory demonstrates f64 precision policy evidence",
          "[tensor][precision]")
{
    const auto narrow = physicalError<float>(), wide = physicalError<double>();
    REQUIRE(narrow > 0.01);
    REQUIRE(wide < 1e-6);
    REQUIRE(wide < narrow * 0.001);
    // Half precision cannot even encode the near-unit decay distinctly from one.
    // Float16/int8 require a different validated numerical strategy; never pretend
    // quantized neural weights imply a valid universal recurrent DSP scalar type.
}
