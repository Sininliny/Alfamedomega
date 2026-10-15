#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace alfa::tensor
{

// An initial CPU representation, not a fixed ontology shape or a serialized model ABI.
// Axes are named by the producer; no method, slot or parameter-count enum is imposed.
template <typename Scalar>
struct TensorT
{
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);
    std::vector<std::string> axes;
    std::vector<std::size_t> shape;
    std::vector<Scalar> values;
    // Optional binary validity mask: 0 is padding, 1 is represented capacity. Padding
    // values must be zero. A genuine zero-valued parameter may still be represented.
    std::vector<std::uint8_t> presence{};
};

template <typename Scalar>
struct TensorViewT
{
    std::span<const std::string> axes;
    std::span<const std::size_t> shape;
    std::span<const Scalar> values;
    std::span<const std::uint8_t> presence{};
};

// Empty views explicitly represent absent input categories; interpretation is primitive-
// specific. No pitch/bend encoding or fixed MIDI/performance width is imposed here.
template <typename Scalar>
struct InputsT
{
    TensorViewT<Scalar> audio;
    TensorViewT<Scalar> midi;
    TensorViewT<Scalar> gesture;
    TensorViewT<Scalar> userPreference;
    TensorViewT<Scalar> latentSound;
    TensorViewT<Scalar> context;
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

template <typename Scalar>
struct StepT
{
    std::uint64_t sample = 0;
    double sampleRate = 0.0;
    Clock clock;
    Scalar activation = 0.0f;
    InputsT<Scalar> inputs;
    std::span<const Scalar> sharedState;
    std::span<const Scalar> localState;
    std::span<const Slice> stateAllocation;
    std::span<const Scalar> parameters;
    TensorViewT<Scalar> routing;
    TensorViewT<Scalar> modulation;
    std::span<const TensorViewT<Scalar>> assets{};
    // Ephemeral typed workspace; callbacks initialize every element they read.
    // Contents are unspecified between calls and never represent persistent S_t.
    std::span<Scalar> scratch{};
    // Set by the runtime, which already guarantees finite parameters and state (every state
    // write and every effective parameter is checked), so primitives need not rescan them.
    bool finiteVerified = false;
};

// Deterministic primitive implementations own mathematical interpretation of their
// parameter/routing tensors. Both functions read S_t; neither can read another
// primitive's uncommitted S_(t+1) or output. All reachable code must be allocation-free,
// lock-free and noexcept. The runtime copies descriptors during compilation.
template <typename Scalar>
using TransitionT = void (*)(const StepT<Scalar>&, std::span<Scalar> nextState) noexcept;
template <typename Scalar>
using ReadoutT = void (*)(const StepT<Scalar>&, std::span<Scalar> output) noexcept;
// Optional fused update: must equal readout(output) followed by transition(nextState) on the
// same step. It lets a primitive validate its step once instead of once per callback.
template <typename Scalar>
using AdvanceT = void (*)(const StepT<Scalar>&, std::span<Scalar> output,
                          std::span<Scalar> nextState) noexcept;

template <typename Scalar>
struct StateWriteT
{
    std::size_t element = 0; // local to the primitive state slice
    Scalar value = 0;
};
// Return the used journal length. A return above capacity rejects the whole tick.
// Writes are validated then committed together; all callbacks observe old S_t.
template <typename Scalar>
using SparseTransitionT = std::size_t (*)(const StepT<Scalar>&, std::span<StateWriteT<Scalar>>) noexcept;

template <typename Scalar>
struct PrimitiveT
{
    std::string id;
    std::size_t stateElements = 0;
    std::size_t outputElements = 0;
    TransitionT<Scalar> transition = nullptr;
    ReadoutT<Scalar> readout = nullptr;
    std::size_t sparseWriteCapacity = 0;
    SparseTransitionT<Scalar> sparseTransition = nullptr;
    std::size_t scratchElements = 0;
    AdvanceT<Scalar> advance = nullptr;
};

enum class SourceKind
{
    Audio,
    Midi,
    Gesture,
    UserPreference,
    LatentSound,
    Context,
    State,
    OutputHistory
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
struct AudioBinding
{
    SourceKind source = SourceKind::Audio;
    std::size_t sourceElement = 0, destinationElement = 0;
    WeightTensor weightTensor = WeightTensor::Routing;
    std::size_t weightElement = 0;
    MissingSource missingSource = MissingSource::Reject;
};

template <typename Scalar>
struct InstanceT
{
    const PrimitiveT<Scalar>* primitive = nullptr;
    std::size_t activationIndex = 0;
    Slice parameters;
    Clock clock;
    OutputBetweenUpdates outputBetweenUpdates = OutputBetweenUpdates::Zero;
    std::vector<TensorT<Scalar>> assets{};
    // Serialized programs own factory descriptors through this optional pointer.
    std::shared_ptr<const PrimitiveT<Scalar>> descriptor{};
    std::size_t inputChannels = 0;
    std::vector<AudioBinding> audioBindings{};
    std::string instanceId{};
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

template <typename Scalar>
struct ConfigurationT
{
    TensorT<Scalar> activation;
    TensorT<Scalar> parameters;
    TensorT<Scalar> routing;
    TensorT<Scalar> modulation;
    std::vector<InstanceT<Scalar>> instances;
    std::vector<Projection> projections;
    // A binary user lock mask. Locked base parameter values cannot be changed by
    // predicted tensor publication or affine projection. Recompilation is explicit.
    std::vector<std::uint8_t> parameterLocks;
    // Explicit shared initial S_0. Local slices follow this prefix in the allocation map.
    std::vector<Scalar> sharedInitialState;
    // Initial values for each local primitive state, including inactive instances.
    std::vector<std::vector<Scalar>> initialState;
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
    // Preallocated sparse journal entries; one Scalar value and structural index each.
    std::size_t sparseStateWrites = 0;
    // Reused numerical workspace: maximum required by one active primitive.
    std::size_t scratchElements = 0;
};

struct ScheduledUpdate
{
    std::size_t instance = 0;
    Clock clock;
};

// Off-thread compilation validates tensors and allocates a schedule and persistent state.
// This is deterministic lowering of compiler predictions, not a trained neural model.
template <typename Scalar>
class RuntimeT
{
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);

public:
    using Tensor = TensorT<Scalar>;
    using TensorView = TensorViewT<Scalar>;
    using Inputs = InputsT<Scalar>;
    using Step = StepT<Scalar>;
    using Primitive = PrimitiveT<Scalar>;
    using Instance = InstanceT<Scalar>;
    using Configuration = ConfigurationT<Scalar>;
    RuntimeT() = default;
    RuntimeT(const RuntimeT<Scalar>&) = delete;
    RuntimeT<Scalar>& operator=(const RuntimeT<Scalar>&) = delete;
    RuntimeT(RuntimeT<Scalar>&&) noexcept = default;
    RuntimeT<Scalar>& operator=(RuntimeT<Scalar>&&) noexcept = default;

    bool prepare(const ConfigurationT<Scalar>& configuration, const Budget& budget, double sampleRate,
                 std::string& error);

    // X_t is supplied at the host sample clock. Due primitives read one immutable S_t,
    // write disjoint next-state slices, and commit together. Not-due states persist.
    // Invalid input/output/state rejects the event atomically, silences outputs and
    // does not advance time. No allocation or logging occurs in this call.
    bool tick(const InputsT<Scalar>& inputs) noexcept;
    // Publish new predicted values into the existing allocation without allocation.
    // Shapes/masks must match the prepared program. Activating an unallocated primitive
    // or changing capacity requires off-thread recompilation, not a silent approximation.
    bool updateTensors(TensorViewT<Scalar> activation, TensorViewT<Scalar> parameters,
                       TensorViewT<Scalar> routing, TensorViewT<Scalar> modulation) noexcept;
    void reset() noexcept;
    bool adoptStateFrom(const RuntimeT<Scalar>&) noexcept;
    // Compatible live publication transfers buffer ownership in constant time.
    bool takeStateFrom(RuntimeT<Scalar>&) noexcept;

    [[nodiscard]] std::span<const Scalar> state() const noexcept { return state_; }
    [[nodiscard]] std::span<const Scalar> output() const noexcept { return output_; }
    [[nodiscard]] std::span<const Slice> stateAllocation() const noexcept { return states_; }
    [[nodiscard]] std::span<const Slice> outputAllocation() const noexcept { return outputs_; }
    [[nodiscard]] std::span<const ScheduledUpdate> schedule() const noexcept { return schedule_; }
    [[nodiscard]] std::uint64_t sample() const noexcept { return sample_; }
    // Numeric backing storage only; structural indices/masks/container overhead excluded.
    [[nodiscard]] std::size_t numericStorageBytes() const noexcept;
    // Structural scratch, excluding numeric journal values and container overhead.
    [[nodiscard]] std::size_t sparseValidationStorageBytes() const noexcept
    {
        return sparseIndices_.size() * sizeof(SparseIndexNode);
    }

private:
    ConfigurationT<Scalar> configuration_;
    std::vector<Scalar> initialActivation_, initialParameters_, initialRouting_, initialModulation_;
    std::vector<PrimitiveT<Scalar>> primitives_;
    std::vector<Slice> states_;
    std::vector<Slice> outputs_;
    std::vector<ScheduledUpdate> schedule_;
    std::vector<std::uint8_t> allocated_;
    std::vector<std::vector<TensorViewT<Scalar>>> assets_;
    std::vector<std::vector<Scalar>> audioOperands_;
    std::vector<std::array<std::size_t, 1>> audioShapes_;
    std::array<std::string, 1> audioAxes_{"channel"};
    std::vector<Scalar> initial_;
    std::vector<Scalar> state_;
    std::vector<Scalar> next_;
    std::vector<Scalar> output_;
    std::vector<Scalar> candidateOutput_;
    std::vector<Scalar> committedOutput_;
    std::vector<Slice> denseScratch_, journalSlices_;
    std::vector<StateWriteT<Scalar>> journal_;
    struct SparseIndexNode
    {
        std::size_t key = 0;
        std::array<std::size_t, 2> children{};
        unsigned bit = 0;
        bool leaf = true;
    };
    std::vector<SparseIndexNode> sparseIndices_;
    std::vector<std::size_t> journalCounts_;
    std::vector<std::uint8_t> denseUpdated_;
    std::vector<Scalar> effectiveParameters_;
    std::vector<Scalar> scratch_;
    std::uint64_t sample_ = 0;
    double sampleRate_ = 0.0;
    bool prepared_ = false;
    std::array<std::uint64_t, 2> stateIdentity_{};
};

using Transition = TransitionT<float>;
using Readout = ReadoutT<float>;
using Tensor = TensorT<float>;
using Tensor64 = TensorT<double>;
using TensorView = TensorViewT<float>;
using TensorView64 = TensorViewT<double>;
using Inputs = InputsT<float>;
using Inputs64 = InputsT<double>;
using Step = StepT<float>;
using Step64 = StepT<double>;
using Primitive = PrimitiveT<float>;
using Primitive64 = PrimitiveT<double>;
using Instance = InstanceT<float>;
using Instance64 = InstanceT<double>;
using Configuration = ConfigurationT<float>;
using Configuration64 = ConfigurationT<double>;
using Runtime = RuntimeT<float>;
using Runtime64 = RuntimeT<double>;

} // namespace alfa::tensor
