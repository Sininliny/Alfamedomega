#pragma once
#include "alfa/tensor/Precision.h"
#include "alfa/tensor/Runtime.h"

#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace alfa::tensorhost
{
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
    virtual TickResult tick(const alfa::tensor::Inputs&, double& modelMilliseconds) noexcept = 0;
    virtual float mixedOutput(int channel) const noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual bool adopt(const TensorExecution&) noexcept = 0;
    virtual void snapshot(std::span<std::atomic<float>>) const noexcept = 0;
    virtual std::vector<float> highestDefaults() const = 0;
    virtual std::string inspection() const = 0;
    virtual alfa::tensor::ScalarType scalarType() const noexcept = 0;
};
std::unique_ptr<TensorExecution> prepareExecution(std::string_view, double sampleRate, std::string& error);
} // namespace alfa::tensorhost
