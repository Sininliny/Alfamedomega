#include "alfa/tensor/Migration.h"
#include "alfa/tensor/Runtime.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>

namespace
{
bool countAllocations = false;
std::size_t allocations = 0;
} // namespace

void* operator new(std::size_t size)
{
    if (countAllocations)
        ++allocations;
    if (auto* result = std::malloc(size == 0 ? 1 : size))
        return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size)
{
    return ::operator new(size);
}
void operator delete(void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}

namespace
{
using namespace alfa::tensor;

Tensor vectorTensor(std::string axis, std::vector<float> values)
{
    const auto size = values.size();
    return {{std::move(axis)}, {size}, std::move(values)};
}

TensorView tensorView(const Tensor& tensor)
{
    return {tensor.axes, tensor.shape, tensor.values, tensor.presence};
}

Inputs audioInput(const std::array<float, 1>& audio)
{
    static const std::array<std::string, 1> axes{"channel"};
    static constexpr std::array<std::size_t, 1> shape{1};
    Inputs result;
    result.audio = {axes, shape, audio};
    return result;
}

void recurrentTransition(const Step& step, std::span<float> next) noexcept
{
    const auto input = step.inputs.audio.values.empty() ? 0.0f : step.inputs.audio.values[0];
    next[0] = input + step.parameters[0] * step.localState[0];
}

void stateReadout(const Step& step, std::span<float> output) noexcept
{
    output[0] = step.activation * step.localState[0];
}

const Primitive recurrent{"test.recurrent-cell", 1, 1, recurrentTransition, stateReadout};

Configuration configuration()
{
    Configuration result;
    result.activation = vectorTensor("primitive", {1.0f});
    result.parameters = vectorTensor("parameter", {0.5f});
    result.routing = {{"source", "destination", "condition", "time"}, {1, 1, 1, 1}, {1.0f}};
    result.instances = {{&recurrent, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero}};
    result.initialState = {{0.0f}};
    return result;
}

Budget budget()
{
    return {16, 128, 128, 128, 128, 128, 1024, 128};
}

void coupledTransition(const Step& step, std::span<float> next) noexcept
{
    // Source selection is explicit in this test primitive's parameter representation.
    const auto source = static_cast<std::size_t>(step.parameters[0]);
    next[0] = step.routing.values[0] * step.sharedState[step.stateAllocation[source].offset];
}

const Primitive coupled{"test.shared-state-reader", 1, 1, coupledTransition, stateReadout};

void invalidTransition(const Step&, std::span<float> next) noexcept
{
    next[0] = std::numeric_limits<float>::quiet_NaN();
}

const Primitive invalid{"test.invalid-state", 1, 1, invalidTransition, stateReadout};

bool translateExample(const LegacySnapshot&, Configuration& output, std::string&)
{
    output = configuration();
    return true;
}
} // namespace

TEST_CASE("Recurrent tensor execution uses old state for output and new state for next sample",
          "[tensor][recurrence]")
{
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(configuration(), budget(), 48000.0, error));
    for (const auto input : std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f})
        REQUIRE(runtime.tick(audioInput({input})));
    CHECK(runtime.output()[0] == Catch::Approx(0.25f));
    CHECK(runtime.state()[0] == Catch::Approx(0.125f));
    CHECK(runtime.sample() == 4);
    runtime.reset();
    CHECK(runtime.state()[0] == 0.0f);
    CHECK(runtime.output()[0] == 0.0f);
    CHECK(runtime.sample() == 0);
}

TEST_CASE("Shared recurrent interaction commits together instead of following execution order",
          "[tensor][routing]")
{
    auto config = configuration();
    config.activation = vectorTensor("primitive", {1.0f, 1.0f});
    config.parameters = vectorTensor("parameter", {1.0f, 0.0f});
    config.instances = {{&coupled, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero},
                        {&coupled, 1, {1, 1}, {1, 0}, OutputBetweenUpdates::Zero}};
    config.initialState = {{2.0f}, {3.0f}};
    Runtime first;
    Runtime reversed;
    std::string error;
    REQUIRE(first.prepare(config, budget(), 48000.0, error));
    std::swap(config.instances[0], config.instances[1]);
    std::swap(config.initialState[0], config.initialState[1]);
    config.parameters.values = {0.0f, 1.0f}; // Same relationships under the new identity order.
    REQUIRE(reversed.prepare(config, budget(), 48000.0, error));
    REQUIRE(first.tick({}));
    REQUIRE(reversed.tick({}));
    CHECK(first.state()[0] == 3.0f);
    CHECK(first.state()[1] == 2.0f);
    CHECK(reversed.state()[0] == first.state()[1]);
    CHECK(reversed.state()[1] == first.state()[0]);
    CHECK(first.output()[0] == 2.0f);
    CHECK(first.output()[1] == 3.0f);
}

TEST_CASE("Independent sample control and slow clocks have explicit hold behavior", "[tensor][clock]")
{
    auto config = configuration();
    config.activation = vectorTensor("primitive", {1.0f, 1.0f, 1.0f});
    config.instances = {{&recurrent, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero},
                        {&recurrent, 1, {0, 1}, {3, 1}, OutputBetweenUpdates::Hold},
                        {&recurrent, 2, {0, 1}, {7, 2}, OutputBetweenUpdates::Zero}};
    config.initialState = {{0.0f}, {2.0f}, {4.0f}};
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(config, budget(), 44100.0, error));
    REQUIRE(runtime.tick(audioInput({0.0f}))); // sample 0: only sample-rate cell.
    CHECK(runtime.state()[1] == 2.0f);
    CHECK(runtime.output()[1] == 0.0f);
    REQUIRE(runtime.tick(audioInput({0.0f}))); // sample 1: control cell.
    CHECK(runtime.state()[1] == 1.0f);
    CHECK(runtime.output()[1] == 2.0f);
    REQUIRE(runtime.tick(audioInput({0.0f}))); // sample 2: slow cell.
    CHECK(runtime.output()[1] == 2.0f);
    CHECK(runtime.state()[2] == 2.0f);
    CHECK(runtime.output()[2] == 4.0f);
    REQUIRE(runtime.tick(audioInput({0.0f})));
    CHECK(runtime.output()[2] == 0.0f);
    CHECK(runtime.state()[2] == 2.0f);
    CHECK(runtime.schedule().size() == 3);
}

TEST_CASE("Tensor compiler rejects malformed shapes clocks and budgets without replacing state",
          "[tensor][compile]")
{
    Runtime runtime;
    std::string error;
    auto config = configuration();
    REQUIRE(runtime.prepare(config, budget(), 48000.0, error));
    REQUIRE(runtime.tick(audioInput({1.0f})));
    SECTION("shape overflow")
    {
        config.routing.shape = {std::numeric_limits<std::size_t>::max(), 2, 1, 1};
    }
    SECTION("duplicate axis")
    {
        config.routing.axes[1] = "source";
    }
    SECTION("zero clock")
    {
        config.instances[0].clock.periodSamples = 0;
    }
    SECTION("invalid phase")
    {
        config.instances[0].clock.phaseSamples = 1;
    }
    SECTION("parameter slice overflow")
    {
        config.instances[0].parameters = {1, std::numeric_limits<std::size_t>::max()};
    }
    SECTION("nonfinite activation")
    {
        config.activation.values[0] = std::numeric_limits<float>::infinity();
    }
    SECTION("missing initial state")
    {
        config.initialState.clear();
    }
    REQUIRE_FALSE(runtime.prepare(config, budget(), 48000.0, error));
    CHECK_FALSE(error.empty());
    CHECK(runtime.sample() == 1);
    CHECK(runtime.state()[0] == 1.0f);
    auto tooSmall = budget();
    tooSmall.activePrimitives = 0;
    REQUIRE_FALSE(runtime.prepare(configuration(), tooSmall, 48000.0, error));
    CHECK(runtime.state()[0] == 1.0f);
}

TEST_CASE("Inactive padded dimensions do not allocate primitive state or consume active budget",
          "[tensor][padding]")
{
    auto config = configuration();
    config.activation = vectorTensor("primitive", {1.0f, 0.0f});
    config.instances.push_back({&recurrent, 1, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero});
    config.initialState.push_back({0.0f});
    auto limits = budget();
    limits.activePrimitives = 1;
    limits.stateElements = 1;
    limits.outputElements = 1;
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(config, limits, 48000.0, error));
    CHECK(runtime.stateAllocation().size() == 2);
    CHECK(runtime.stateAllocation()[1].count == 0);
    CHECK(runtime.schedule().size() == 1);
    CHECK(runtime.state().size() == 1);
}

TEST_CASE("Tiers share padded tensor shapes while budgets count represented dimensions", "[tensor][tier]")
{
    auto config = configuration();
    config.parameters = {{"parameter"}, {4}, {0.5f, 0.0f, 0.0f, 0.0f}, {1, 0, 0, 0}};
    config.routing = {
        {"source", "destination", "condition", "time"}, {2, 2, 1, 1}, {1.0f, 0.0f, 0.0f, 0.0f}, {1, 0, 0, 0}};
    auto limits = budget();
    limits.parameterElements = 1;
    limits.routingElements = 1;
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(config, limits, 48000.0, error));
    REQUIRE(runtime.tick(audioInput({1.0f})));
    CHECK(runtime.state()[0] == 1.0f);
    config.parameters.presence = {1, 1, 1, 1}; // Real zero values still consume dimensionality.
    REQUIRE_FALSE(runtime.prepare(config, limits, 48000.0, error));
    config.parameters.presence = {1, 0, 0, 0};
    limits.storedTensorElements = 1;
    REQUIRE_FALSE(runtime.prepare(config, limits, 48000.0, error));
    limits = budget();
    config.instances[0].parameters = {1, 1};
    REQUIRE_FALSE(runtime.prepare(config, limits, 48000.0, error));
    config.instances[0].parameters = {0, 1};
    config.parameters.values[2] = 0.3f; // A masked nonzero is not silent padding.
    REQUIRE_FALSE(runtime.prepare(config, limits, 48000.0, error));
}

TEST_CASE("Invalid tensor state rejects the entire event and leaves persistent state intact",
          "[tensor][atomic]")
{
    auto config = configuration();
    config.activation = vectorTensor("primitive", {1.0f, 1.0f});
    config.instances.push_back({&invalid, 1, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero});
    config.initialState = {{2.0f}, {3.0f}};
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(config, budget(), 48000.0, error));
    REQUIRE_FALSE(runtime.tick(audioInput({1.0f})));
    CHECK(runtime.state()[0] == 2.0f);
    CHECK(runtime.state()[1] == 3.0f);
    CHECK(runtime.sample() == 0);
    CHECK(runtime.output()[0] == 0.0f);
    CHECK(runtime.output()[1] == 0.0f);
}

TEST_CASE("All six input categories remain independently available to deterministic primitives",
          "[tensor][inputs]")
{
    const Primitive allInputs{"test.all-inputs", 1, 1,
                              [](const Step& step, std::span<float> next) noexcept
                              {
                                  next[0] = step.inputs.audio.values[0] + step.inputs.midi.values[0] +
                                            step.inputs.gesture.values[0] +
                                            step.inputs.userPreference.values[0] +
                                            step.inputs.latentSound.values[0] + step.inputs.context.values[0];
                              },
                              stateReadout};
    auto config = configuration();
    config.instances[0].primitive = &allInputs;
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(config, budget(), 96000.0, error));
    const std::array<float, 1> value{1.0f};
    auto inputs = audioInput(value);
    inputs.midi = inputs.audio;
    inputs.gesture = inputs.audio;
    inputs.userPreference = inputs.audio;
    inputs.latentSound = inputs.audio;
    inputs.context = inputs.audio;
    REQUIRE(runtime.tick(inputs));
    CHECK(runtime.state()[0] == 6.0f);
    const std::array<float, 1> nan{std::numeric_limits<float>::quiet_NaN()};
    inputs.gesture.values = nan;
    REQUIRE_FALSE(runtime.tick(inputs));
    CHECK(runtime.state()[0] == 6.0f);
    CHECK(runtime.sample() == 1);
}

TEST_CASE("State runtime does not allocate after preparation", "[tensor][allocation]")
{
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(configuration(), budget(), 48000.0, error));
    const std::array<float, 1> audio{0.1f};
    const auto inputs = audioInput(audio);
    const auto controls = configuration();
    allocations = 0;
    bool valid = true;
    countAllocations = true;
    for (int index = 0; index < 10000; ++index)
    {
        valid = runtime.updateTensors(tensorView(controls.activation), tensorView(controls.parameters),
                                      tensorView(controls.routing), tensorView(controls.modulation)) &&
                valid;
        valid = runtime.tick(inputs) && valid;
    }
    runtime.reset();
    countAllocations = false;
    CHECK(valid);
    CHECK(allocations == 0);
}

TEST_CASE("Reset restores compiled tensor controls as well as recurrent state", "[tensor][reset]")
{
    Runtime runtime;
    std::string error;
    auto config = configuration();
    REQUIRE(runtime.prepare(config, budget(), 48000.0, error));
    config.parameters.values[0] = 0.9f;
    REQUIRE(runtime.updateTensors(tensorView(config.activation), tensorView(config.parameters),
                                  tensorView(config.routing), tensorView(config.modulation)));
    runtime.reset();
    const std::array<float, 1> impulse{1.0f};
    const std::array<float, 1> silence{0.0f};
    REQUIRE(runtime.tick(audioInput(impulse)));
    REQUIRE(runtime.tick(audioInput(silence)));
    CHECK(runtime.state()[0] == 0.5f);
}

TEST_CASE("Learned tensor values publish atomically without rebuilding the allocation",
          "[tensor][publication]")
{
    auto config = configuration();
    config.activation = vectorTensor("primitive", {1.0f, 0.0f});
    config.instances.push_back({&recurrent, 1, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero});
    config.initialState.push_back({0.0f});
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(config, budget(), 48000.0, error));
    REQUIRE(runtime.tick(audioInput({1.0f})));
    const auto publish = [&]
    {
        return runtime.updateTensors(tensorView(config.activation), tensorView(config.parameters),
                                     tensorView(config.routing), tensorView(config.modulation));
    };
    config.parameters.values[0] = 0.25f;
    REQUIRE(publish());
    REQUIRE(runtime.tick(audioInput({0.0f})));
    CHECK(runtime.state()[0] == 0.25f);
    config.activation.values[1] = 1.0f;
    REQUIRE_FALSE(publish()); // An unallocated primitive requires compilation.
    config.activation.values[1] = 0.0f;
    config.activation.values[0] = 0.0f;
    REQUIRE(publish());
    REQUIRE(runtime.tick(audioInput({1.0f})));
    CHECK(runtime.state()[0] == 0.25f);
    CHECK(runtime.output()[0] == 0.0f);
    config.activation.values[0] = 1.0f;
    config.parameters.values[0] = std::numeric_limits<float>::infinity();
    REQUIRE_FALSE(publish());
    REQUIRE(runtime.tick(audioInput({1.0f})));
    CHECK(runtime.state()[0] == 0.25f); // Failed publication left activation disabled.
    config.parameters.values[0] = 0.25f;
    REQUIRE(publish());
    REQUIRE(runtime.tick(audioInput({0.0f})));
    CHECK(runtime.output()[0] == 0.25f);
    CHECK(runtime.state()[0] == 0.0625f);
}

TEST_CASE("Affine tensor projection shares inputs and recurrent state across parameter destinations",
          "[tensor][projection]")
{
    const Primitive parameterOutput{"test.parameter-output", 1, 1,
                                    [](const Step& step, std::span<float> next) noexcept
                                    { next[0] = step.localState[0] + 1.0f; },
                                    [](const Step& step, std::span<float> out) noexcept
                                    {
                                        out[0] = step.parameters[0];
                                    }};
    auto config = configuration();
    config.activation = vectorTensor("primitive", {1.0f, 1.0f});
    config.parameters = vectorTensor("parameter", {0.0f, 0.0f});
    config.routing = {{"source", "destination", "condition", "time"}, {1, 2, 1, 1}, {1.0f, 0.5f}};
    config.modulation = vectorTensor("relationship", {0.25f});
    config.instances = {{&parameterOutput, 0, {0, 1}, {1, 0}, OutputBetweenUpdates::Zero},
                        {&parameterOutput, 1, {1, 1}, {1, 0}, OutputBetweenUpdates::Zero}};
    config.initialState = {{4.0f}, {8.0f}};
    config.projections = {{SourceKind::Midi, 0, 0, WeightTensor::Routing, 0, MissingSource::Reject},
                          {SourceKind::Midi, 0, 1, WeightTensor::Routing, 1, MissingSource::Reject},
                          {SourceKind::Gesture, 0, 0, WeightTensor::Modulation, 0, MissingSource::Reject},
                          {SourceKind::State, 0, 1, WeightTensor::Routing, 1, MissingSource::Reject}};
    const std::array<float, 1> midi{2.0f}, gesture{3.0f};
    auto inputs = audioInput(midi);
    inputs.midi = inputs.audio;
    inputs.gesture = audioInput(gesture).audio;
    Runtime runtime;
    std::string error;
    REQUIRE(runtime.prepare(config, budget(), 48000.0, error));
    REQUIRE(runtime.tick(inputs));
    CHECK(runtime.output()[0] == 2.75f);
    CHECK(runtime.output()[1] == 3.0f); // Uses old shared state 4, not newly computed 5.
    REQUIRE(runtime.tick(inputs));
    CHECK(runtime.output()[1] == 3.5f);
    REQUIRE_FALSE(runtime.tick({})); // Missing bound input is not silently replaced.
    CHECK(runtime.sample() == 2);
    CHECK(runtime.state()[0] == 6.0f);

    config.parameterLocks = {0, 1};
    REQUIRE(runtime.prepare(config, budget(), 48000.0, error));
    REQUIRE(runtime.tick(inputs));
    CHECK(runtime.output()[1] == 0.0f); // User lock takes precedence over both projections.
    config.parameters.values[1] = 0.5f;
    REQUIRE_FALSE(runtime.updateTensors(tensorView(config.activation), tensorView(config.parameters),
                                        tensorView(config.routing), tensorView(config.modulation)));
    config.parameters.values[1] = 0.0f;
    config.projections[0].weightElement = 999;
    REQUIRE_FALSE(runtime.prepare(config, budget(), 48000.0, error));
    config.projections[0].weightElement = 0;
    const std::array<float, 1> zero{0.0f};
    const std::array<std::uint8_t, 1> absent{0};
    inputs.midi.values = zero;
    inputs.midi.presence = absent;
    REQUIRE_FALSE(runtime.tick(inputs)); // Padding obeys explicit missing-source policy.
    config.projections[0].missingSource = MissingSource::Zero;
    config.projections[1].missingSource = MissingSource::Zero;
    REQUIRE(runtime.prepare(config, budget(), 48000.0, error));
    REQUIRE(runtime.tick(inputs));
    CHECK(runtime.output()[0] == 0.75f); // Gesture still contributes, padded MIDI does not.
}

TEST_CASE("Migration retains exact legacy project bytes and requires explicit semantic translation",
          "[tensor][migration]")
{
    const LegacySnapshot original{"legacy.juce-project", {std::byte{0}, std::byte{255}, std::byte{17}}};
    Migration migration(original);
    std::string error;
    REQUIRE_FALSE(migration.translate(nullptr, budget(), 48000.0, error));
    CHECK(migration.status() == MigrationStatus::LegacyOnly);
    CHECK(migration.original().format == original.format);
    CHECK(migration.original().bytes == original.bytes);
    REQUIRE(migration.translate(translateExample, budget(), 48000.0, error));
    CHECK(migration.status() == MigrationStatus::TensorCandidate);
    CHECK(migration.original().bytes == original.bytes);
    auto limits = budget();
    limits.stateElements = 0;
    REQUIRE_FALSE(migration.translate(translateExample, limits, 48000.0, error));
    CHECK(migration.status() == MigrationStatus::TensorCandidate);
    CHECK(migration.candidate().state().size() == 1);
    CHECK(migration.original().bytes == original.bytes);
}
