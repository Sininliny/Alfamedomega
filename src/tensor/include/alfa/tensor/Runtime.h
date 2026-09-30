#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace alfa::tensor
{

// An initial CPU representation, not a fixed ontology shape or a serialized model ABI.
// Axes are named by the producer; no method, slot or parameter-count enum is imposed.
struct Tensor
{
    std::vector<std::string> axes;
    std::vector<std::size_t> shape;
    std::vector<float> values;
    // Optional binary validity mask: 0 is padding, 1 is represented capacity. Padding
    // values must be zero. A genuine zero-valued parameter may still be represented.
    std::vector<std::uint8_t> presence{};
};

struct TensorView
{
    std::span<const std::string> axes;
    std::span<const std::size_t> shape;
    std::span<const float> values;
    std::span<const std::uint8_t> presence{};
};

// Empty views explicitly represent absent input categories; interpretation is primitive-
// specific. No pitch/bend encoding or fixed MIDI/performance width is imposed here.
struct Inputs
{
    TensorView audio;
    TensorView midi;
    TensorView gesture;
    TensorView userPreference;
    TensorView latentSound;
    TensorView context;
};

struct Slice
{
    std::size_t offset = 0;
    std::size_t count = 0;
};

struct Clock
{
    // Chosen by the configuration producer. 1 permits sample/audio-rate execution;
    // larger periods support control and slow state. No global control rate is inferred.
    std::uint64_t periodSamples = 0;
    std::uint64_t phaseSamples = 0;
};

enum class OutputBetweenUpdates
{
    Hold,
    Zero
};

struct Step
{
    std::uint64_t sample = 0;
    double sampleRate = 0.0;
    Clock clock;
    float activation = 0.0f;
    Inputs inputs;
    std::span<const float> sharedState;
    std::span<const float> localState;
    std::span<const Slice> stateAllocation;
    std::span<const float> parameters;
    TensorView routing;
    TensorView modulation;
    std::span<const TensorView> assets{};
};

// Deterministic primitive implementations own mathematical interpretation of their
// parameter/routing tensors. Both functions read S_t; neither can read another
// primitive's uncommitted S_(t+1) or output. All reachable code must be allocation-free,
// lock-free and noexcept. The runtime copies descriptors during compilation.
using Transition = void (*)(const Step&, std::span<float> nextState) noexcept;
using Readout = void (*)(const Step&, std::span<float> output) noexcept;

struct Primitive
{
    std::string id;
    std::size_t stateElements = 0;
    std::size_t outputElements = 0;
    Transition transition = nullptr;
    Readout readout = nullptr;
};

struct Instance
{
    const Primitive* primitive = nullptr;
    std::size_t activationIndex = 0;
    Slice parameters;
    Clock clock;
    OutputBetweenUpdates outputBetweenUpdates = OutputBetweenUpdates::Zero;
    std::vector<Tensor> assets{};
    // Serialized programs own factory descriptors through this optional pointer.
    std::shared_ptr<const Primitive> descriptor{};
};

enum class SourceKind
{
    Audio,
    Midi,
    Gesture,
    UserPreference,
    LatentSound,
    Context,
    State
};

enum class WeightTensor
{
    Routing,
    Modulation
};

enum class MissingSource
{
    Reject,
    Zero
};

// Explicit lowering for the mathematical affine projection operation:
// parameter[d] = base[d] + sum(weight[w] * source[s]). It is optional, not a
// mandated interpretation of all future routing tensors. The producer selects
// every source, destination and flattened condition/time weight; no automatic
// velocity, pitch, condition or time encoding is inferred.
struct Projection
{
    SourceKind source = SourceKind::Audio;
    std::size_t sourceElement = 0;
    std::size_t parameterElement = 0;
    WeightTensor weightTensor = WeightTensor::Routing;
    std::size_t weightElement = 0;
    MissingSource missingSource = MissingSource::Reject;
};

struct Configuration
{
    Tensor activation;
    Tensor parameters;
    Tensor routing;
    Tensor modulation;
    std::vector<Instance> instances;
    std::vector<Projection> projections;
    // A binary user lock mask. Locked base parameter values cannot be changed by
    // predicted tensor publication or affine projection. Recompilation is explicit.
    std::vector<std::uint8_t> parameterLocks;
    // Explicit shared initial S_0. Local slices follow this prefix in the allocation map.
    std::vector<float> sharedInitialState;
    // Initial values for each local primitive state, including inactive instances.
    std::vector<std::vector<float>> initialState;
};

// Resource limits supplied by a tier/hardware policy, not hard-coded tier numbers.
// Zero means zero allowed capacity. No automatic pruning or tier conversion occurs here.
struct Budget
{
    std::size_t activePrimitives = 0;
    std::size_t stateElements = 0;
    std::size_t parameterElements = 0;
    std::size_t routingElements = 0;
    std::size_t modulationElements = 0;
    std::size_t outputElements = 0;
    // Full padded storage is bounded separately from represented dimensionality.
    std::size_t storedTensorElements = 0;
    std::size_t projectionTerms = 0;
};

struct ScheduledUpdate
{
    std::size_t instance = 0;
    Clock clock;
};

// Off-thread compilation validates tensors and allocates a schedule and persistent state.
// This is deterministic lowering of compiler predictions, not a trained neural model.
class Runtime
{
public:
    Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept = default;
    Runtime& operator=(Runtime&&) noexcept = default;

    bool prepare(const Configuration& configuration, const Budget& budget, double sampleRate,
                 std::string& error);

    // X_t is supplied at the host sample clock. Due primitives read one immutable S_t,
    // write disjoint next-state slices, and commit together. Not-due states persist.
    // Invalid input/output/state rejects the event atomically, silences outputs and
    // does not advance time. No allocation or logging occurs in this call.
    bool tick(const Inputs& inputs) noexcept;
    // Publish new predicted values into the existing allocation without allocation.
    // Shapes/masks must match the prepared program. Activating an unallocated primitive
    // or changing capacity requires off-thread recompilation, not a silent approximation.
    bool updateTensors(TensorView activation, TensorView parameters, TensorView routing,
                       TensorView modulation) noexcept;
    void reset() noexcept;

    [[nodiscard]] std::span<const float> state() const noexcept { return state_; }
    [[nodiscard]] std::span<const float> output() const noexcept { return output_; }
    [[nodiscard]] std::span<const Slice> stateAllocation() const noexcept { return states_; }
    [[nodiscard]] std::span<const Slice> outputAllocation() const noexcept { return outputs_; }
    [[nodiscard]] std::span<const ScheduledUpdate> schedule() const noexcept { return schedule_; }
    [[nodiscard]] std::uint64_t sample() const noexcept { return sample_; }

private:
    Configuration configuration_;
    std::vector<float> initialActivation_, initialParameters_, initialRouting_, initialModulation_;
    std::vector<Primitive> primitives_;
    std::vector<Slice> states_;
    std::vector<Slice> outputs_;
    std::vector<ScheduledUpdate> schedule_;
    std::vector<std::uint8_t> allocated_;
    std::vector<std::vector<TensorView>> assets_;
    std::vector<float> initial_;
    std::vector<float> state_;
    std::vector<float> next_;
    std::vector<float> output_;
    std::vector<float> candidateOutput_;
    std::vector<float> effectiveParameters_;
    std::uint64_t sample_ = 0;
    double sampleRate_ = 0.0;
    bool prepared_ = false;
};

} // namespace alfa::tensor
