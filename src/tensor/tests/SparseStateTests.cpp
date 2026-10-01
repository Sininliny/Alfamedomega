#include "alfa/dsp/detail/DelayFdn.h"
#include "alfa/tensor/Runtime.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <limits>
namespace
{
using namespace alfa::tensor;
template <class T>
size_t sparse(const StepT<T>& step, std::span<StateWriteT<T>> writes) noexcept
{
    if (step.parameters[0] == T(-4))
        return writes.size() + 1;
    if (step.parameters[0] == T(-5))
        return 1; // declared unwritten journal entry must reject
    writes[0] = {0, step.localState[0] + T(1)};
    writes[1] = {step.localState.size() - 1, step.localState.back() + step.parameters[0]};
    if (step.parameters[0] == T(-1))
        writes[1].element = step.localState.size();
    if (step.parameters[0] == T(-2))
        writes[1].element = 0;
    if (step.parameters[0] == T(-3))
        writes[1].value = std::numeric_limits<T>::quiet_NaN();
    return 2;
}
template <class T>
void readLast(const StepT<T>& step, std::span<T> out) noexcept
{
    out[0] = step.localState.back();
}
template <class T>
void readDense(const StepT<T>& step, std::span<T> out) noexcept
{
    out[0] = step.localState[0];
}
template <class T>
void dense(const StepT<T>& step, std::span<T> next) noexcept
{
    const auto source = step.stateAllocation[0];
    next[0] = step.sharedState[source.offset + source.count - 1];
    // Deliberately retain next[1]: dense partial-write compatibility requires old copy.
    if (step.parameters[0] == T(-6))
        next[0] = std::numeric_limits<T>::infinity();
}
template <class T>
ConfigurationT<T> program(PrimitiveT<T>& sparsePrimitive, PrimitiveT<T>& densePrimitive)
{
    ConfigurationT<T> c;
    c.activation = {{"instance"}, {2}, {T(1), T(1)}};
    c.parameters = {{"parameter"}, {1}, {T(2)}};
    c.routing = {{"weight"}, {0}, {}};
    c.modulation = c.routing;
    c.instances = {{&sparsePrimitive, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Hold},
                   {&densePrimitive, 1, {0, 1}, {1, 0}, OutputBetweenUpdates::Hold}};
    c.initialState = {std::vector<T>(65536, T(0)), {T(0), T(17)}};
    c.initialState[0].back() = T(5);
    return c;
}
template <class T>
void behavior()
{
    PrimitiveT<T> a{"test.sparse", 65536, 1, nullptr, readLast<T>, 2, sparse<T>};
    PrimitiveT<T> b{"test.dense", 2, 1, dense<T>, readDense<T>};
    auto c = program(a, b);
    const Budget budget{2, 65538, 1, 0, 0, 2, 3, 0, 2};
    RuntimeT<T> runtime;
    std::string error;
    REQUIRE(runtime.prepare(c, budget, 48000, error));
    const auto storage = runtime.numericStorageBytes();
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.output()[0] == T(5));
    REQUIRE(runtime.state()[65535] == T(7));
    REQUIRE(runtime.state()[65536] == T(5));
    REQUIRE(runtime.state()[65537] == T(17));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65536] == T(7));
    auto stable = std::vector<T>(runtime.state().begin(), runtime.state().end());
    const auto clock = runtime.sample();
    for (const auto invalid : {T(-1), T(-2), T(-3), T(-4), T(-5), T(-6)})
    {
        c.parameters.values[0] = invalid;
        const auto view = [](const TensorT<T>& t) -> TensorViewT<T>
        {
            return {t.axes, t.shape, t.values, t.presence};
        };
        REQUIRE(runtime.updateTensors(view(c.activation), view(c.parameters), view(c.routing),
                                      view(c.modulation)));
        REQUIRE_FALSE(runtime.tick({}));
        REQUIRE(runtime.sample() == clock);
        REQUIRE(std::equal(stable.begin(), stable.end(), runtime.state().begin()));
        REQUIRE(runtime.output()[0] == T(0));
        REQUIRE(runtime.output()[1] == T(0));
    }
    c.parameters.values[0] = T(2);
    const auto view = [](const TensorT<T>& t) -> TensorViewT<T>
    {
        return {t.axes, t.shape, t.values, t.presence};
    };
    REQUIRE(
        runtime.updateTensors(view(c.activation), view(c.parameters), view(c.routing), view(c.modulation)));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65535] == T(11));
    REQUIRE(runtime.state()[65536] == T(9));
    REQUIRE(runtime.state()[1] == T(0)); // untouched huge state remains unchanged
    REQUIRE(runtime.numericStorageBytes() == storage);
    RuntimeT<T> adopted;
    REQUIRE(adopted.prepare(c, budget, 48000, error));
    REQUIRE(adopted.adoptStateFrom(runtime));
    REQUIRE(adopted.sample() == runtime.sample());
    REQUIRE(adopted.tick({}));
    REQUIRE(runtime.tick({}));
    REQUIRE(std::equal(adopted.state().begin(), adopted.state().end(), runtime.state().begin()));
    runtime.reset();
    REQUIRE(runtime.sample() == 0);
    REQUIRE(runtime.state()[65535] == T(5));
    REQUIRE(runtime.state()[65537] == T(17));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65535] == T(7));
    auto insufficient = budget;
    insufficient.sparseStateWrites = 1;
    REQUIRE_FALSE(adopted.prepare(c, insufficient, 48000, error)); // playing program remains intact
    REQUIRE(adopted.tick({}));
    a.sparseWriteCapacity = std::numeric_limits<size_t>::max();
    REQUIRE_FALSE(adopted.prepare(c, budget, 48000, error)); // reject impossible journal before allocation
}
template <class T>
void historyDense(const StepT<T>& step, std::span<T> next) noexcept
{
    next[0] = step.parameters[0] < 0 ? std::numeric_limits<T>::infinity() : step.parameters[0];
}
template <class T>
void history()
{
    PrimitiveT<T> primitive{"test.output-history-rollback", 1, 1, historyDense<T>, readDense<T>};
    ConfigurationT<T> c;
    c.activation = {{"instance"}, {1}, {T(1)}};
    c.parameters = {{"parameter"}, {1}, {T(1)}};
    c.routing = {{"weight"}, {1}, {T(1)}};
    c.modulation = {{"weight"}, {0}, {}};
    c.instances = {{&primitive, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Hold}};
    c.initialState = {{T(3)}};
    c.projections = {{SourceKind::OutputHistory, 0, 0, WeightTensor::Routing, 0, MissingSource::Reject}};
    RuntimeT<T> runtime;
    std::string error;
    REQUIRE(runtime.prepare(c, {1, 1, 1, 1, 0, 1, 3, 1}, 48000, error));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.output()[0] == T(3));
    auto view = [](const TensorT<T>& t) -> TensorViewT<T>
    {
        return {t.axes, t.shape, t.values, t.presence};
    };
    c.parameters.values[0] = T(-10);
    REQUIRE(
        runtime.updateTensors(view(c.activation), view(c.parameters), view(c.routing), view(c.modulation)));
    REQUIRE_FALSE(runtime.tick({}));
    REQUIRE(runtime.output()[0] == T(0));
    REQUIRE(runtime.state()[0] == T(1));
    c.parameters.values[0] = T(1);
    REQUIRE(
        runtime.updateTensors(view(c.activation), view(c.parameters), view(c.routing), view(c.modulation)));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[0] == T(4)); // retained committed output3, not exposed silence0
}
} // namespace
TEST_CASE("Sparse writes and due dense slices commit transactionally with immutable old-state reads",
          "[tensor][sparse]")
{
    behavior<float>();
    behavior<double>();
}
TEST_CASE("Rejected updates silence exposure while retaining committed recurrent output history",
          "[tensor][sparse]")
{
    history<float>();
    history<double>();
}
TEST_CASE("Sparse state storage has no full next-state shadow", "[tensor][sparse]")
{
    using namespace alfa::tensor;
    Primitive a{"test.sparse", 65536, 1, nullptr, readLast<float>, 2, sparse<float>};
    Primitive b{"test.dense", 2, 1, dense<float>, readDense<float>};
    auto c = program(a, b);
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(c, {2, 65538, 1, 0, 0, 2, 3, 0, 2}, 48000, error));
    // Initial config copy + initial runtime state + committed state: three full copies.
    // Dense scratch has only two values; sparse journal two; outputs are tiny.
    REQUIRE(runtime.numericStorageBytes() < 65538 * sizeof(float) * 4);
}

TEST_CASE("Sparse multi-timescale and activation skips never recommit stale journals", "[tensor][sparse]")
{
    using namespace alfa::tensor;
    Primitive a{"test.sparse", 65536, 1, nullptr, readLast<float>, 2, sparse<float>};
    Primitive b{"test.dense", 2, 1, dense<float>, readDense<float>};
    auto c = program(a, b);
    c.instances[0].clock = {4, 1};
    c.instances[1].clock = {2, 0};
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(c, {2, 65538, 1, 0, 0, 2, 3, 0, 2}, 48000, error));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65535] == 5); // only dense due
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65535] == 7); // sparse now due
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65535] == 7);
    REQUIRE(runtime.state()[65536] == 7);
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65535] == 7);
    const auto view = [](const Tensor& t) -> TensorView
    {
        return {t.axes, t.shape, t.values, t.presence};
    };
    c.activation.values[0] = 0;
    REQUIRE(
        runtime.updateTensors(view(c.activation), view(c.parameters), view(c.routing), view(c.modulation)));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65535] == 7); // due but inactive
    c.activation.values[0] = 1;
    REQUIRE(
        runtime.updateTensors(view(c.activation), view(c.parameters), view(c.routing), view(c.modulation)));
    for (int i = 0; i < 4; ++i)
        REQUIRE(runtime.tick({}));
    REQUIRE(runtime.state()[65535] == 9); // exactly one later update
}
TEST_CASE("Shared Hadamard8 retains legacy float arithmetic and true double energy",
          "[tensor][sparse][reuse]")
{
    using namespace alfa::dsp::detail;
    std::array<float, 8> values{0.7f, -0.25f, 0.2f, 1.1f, -0.81f, 0.13f, 0.93f, -0.05f}, oracle = values;
    for (int h = 1; h < 8; h <<= 1)
        for (int i = 0; i < 8; i += 2 * h)
            for (int j = i; j < i + h; ++j)
            {
                const auto a = static_cast<size_t>(j), b = static_cast<size_t>(j + h);
                const float x = oracle[a], z = oracle[b];
                oracle[a] = x + z;
                oracle[b] = x - z;
            }
    for (auto& value : oracle)
        value *= 0.35355339059327373f;
    DelayFdn::hadamard(values);
    REQUIRE(values == oracle);
    for (size_t basis = 0; basis < 8; ++basis)
    {
        std::array<double, 8> column{};
        column[basis] = 1;
        hadamard8(column);
        double energy = 0;
        for (const auto value : column)
            energy += value * value;
        REQUIRE(energy == Catch::Approx(1.0).margin(1e-14));
        hadamard8(column);
        for (size_t i = 0; i < 8; ++i)
            REQUIRE(column[i] == Catch::Approx(i == basis ? 1.0 : 0.0).margin(1e-14));
    }
}
namespace
{
template <class S>
std::size_t largeJournal(const StepT<S>& step, std::span<StateWriteT<S>> writes) noexcept
{
    const auto count = static_cast<std::size_t>(step.parameters[0]);
    const auto mode = static_cast<std::size_t>(step.parameters[1]);
    const auto length = step.localState.size();
    for (std::size_t i = 0; i < count; ++i)
    {
        std::size_t index = i;
        if (mode == 1)
            index = length - 1 - i;
        if (mode >= 2)
            index = (i * 4051 + 997) % length;
        writes[i] = {index, step.localState[index] + S(1)};
    }
    if (mode == 3 && count > 1)
        writes[count - 1].element = writes[0].element;
    return count;
}
template <class S>
std::size_t largeJournalCheck()
{
    constexpr std::size_t length = 8192, capacity = 2048;
    PrimitiveT<S> p{"test.large-journal", length, 1, nullptr, readDense<S>, capacity, largeJournal<S>};
    ConfigurationT<S> c;
    c.activation = {{"instance"}, {1}, {S(1)}};
    c.parameters = {{"parameter"}, {2}, {S(capacity), S(0)}};
    c.routing = {{"weight"}, {0}, {}};
    c.modulation = c.routing;
    c.instances = {{&p, 0, {0, 2}, {1, 0}, OutputBetweenUpdates::Hold}};
    c.initialState = {std::vector<S>(length, S(0))};
    RuntimeT<S> runtime;
    std::string error;
    REQUIRE(runtime.prepare(c, {1, length, 2, 0, 0, 1, 3, 0, capacity}, 48000, error));
    const auto structural = runtime.sparseValidationStorageBytes();
    REQUIRE(structural >= 2 * capacity * sizeof(std::size_t));
    const auto view = [](const TensorT<S>& t) -> TensorViewT<S>
    {
        return {t.axes, t.shape, t.values, t.presence};
    };
    for (const std::size_t count : {capacity, std::size_t(1), std::size_t(0), std::size_t(63), capacity})
        for (std::size_t mode = 0; mode < 4; ++mode)
        {
            c.parameters.values = {static_cast<S>(count), static_cast<S>(mode)};
            REQUIRE(runtime.updateTensors(view(c.activation), view(c.parameters), view(c.routing),
                                          view(c.modulation)));
            const std::vector<S> before(runtime.state().begin(), runtime.state().end());
            const auto sample = runtime.sample();
            const bool valid = mode != 3 || count < 2;
            REQUIRE(runtime.tick({}) == valid);
            if (!valid)
            {
                REQUIRE(runtime.sample() == sample);
                REQUIRE(std::equal(before.begin(), before.end(), runtime.state().begin()));
                REQUIRE(runtime.output()[0] == S(0));
            }
            else
            {
                REQUIRE(runtime.sample() == sample + 1);
                double change = 0;
                for (std::size_t i = 0; i < length; ++i)
                    change += static_cast<double>(runtime.state()[i] - before[i]);
                REQUIRE(change == static_cast<double>(count));
            }
            REQUIRE(runtime.sparseValidationStorageBytes() == structural);
        }
    runtime.reset();
    REQUIRE(runtime.tick({}));
    return structural;
}
} // namespace
TEST_CASE("STFT-sized sparse journals validate arbitrary index orders and duplicate rollback in bounded "
          "structural scratch",
          "[tensor][sparse]")
{
    REQUIRE(largeJournalCheck<float>() == largeJournalCheck<double>());
}
