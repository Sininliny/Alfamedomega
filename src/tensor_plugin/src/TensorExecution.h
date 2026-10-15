#pragma once
#include "alfa/tensor/Precision.h"
#include "alfa/tensor/Runtime.h"

#include <atomic>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace alfa::tensorhost
{
struct NativePolicy
{
    size_t voiceCapacity = 0;
    double releaseSeconds = 0, tailSeconds = 0;
    float bendSemitones = 0;
    int latencySamples = 0;
};
// One policy parser for library, execution and Processor publication. Throws on
// malformed policies; callers prepare everything before publishing a new sound.
NativePolicy readNativePolicy(const nlohmann::json&);

// Type erasure is only at the native host boundary. A prepared execution owns
// uniformly typed tensors, model, layers, assets and recurrent state.
class TensorExecution
{
public:
    struct TickResult
    {
        bool audioRendered;
        bool modelAccepted;
    };
    virtual ~TensorExecution() = default;
    virtual bool updateControls(std::span<const float>) noexcept = 0;
    // Eight wave feature knobs in [-1, 1]. Stored for the linear fallback matrix, which the next
    // updateControls applies; a compiler model reads them from Inputs::waveFeature instead.
    virtual bool updateWaveFeatures(std::span<const float>) noexcept = 0;
    virtual TickResult tick(const alfa::tensor::Inputs&, double& modelMilliseconds) noexcept = 0;
    virtual float mixedOutput(int channel) const noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual bool adopt(TensorExecution&) noexcept = 0;
    virtual void snapshot(std::span<std::atomic<float>>) const noexcept = 0;
    // Where each instance's outputs and state live in the compiled program. Message thread, after prepare.
    struct Layout
    {
        std::vector<alfa::tensor::Slice> outputs, states;
    };
    virtual Layout layout() const = 0;
    // Live values for the editor, published once per block on the audio thread: the peak |output| of
    // each instance since the last call (then reset), the parameters the runtime used (after projections)
    // and every control layer's values after its matrix and clamps, concatenated layer by layer.
    virtual void telemetry(std::span<std::atomic<float>> instanceLevels,
                           std::span<std::atomic<float>> parameters,
                           std::span<std::atomic<float>> layerValues) noexcept = 0;
    virtual std::vector<float> highestDefaults() const = 0;
    virtual std::string inspection() const = 0;
    virtual alfa::tensor::ScalarType scalarType() const noexcept = 0;
};
std::unique_ptr<TensorExecution> prepareExecution(std::string_view, double sampleRate, std::string& error);
} // namespace alfa::tensorhost
