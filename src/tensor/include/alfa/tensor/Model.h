#pragma once

#include "alfa/tensor/Runtime.h"

#include <array>
#include <string_view>

namespace alfa::tensor
{
// CPU GRU inference for the independently budgeted compiler model. It predicts
// tensor values/presence; deterministic DSP remains responsible for all samples.
class TensorModel
{
public:
    bool prepare(std::string_view json, const Configuration& controls, std::string& error);
    bool infer(const Inputs& inputs) noexcept;
    bool apply(const Inputs& inputs, Runtime& runtime) noexcept;
    void reset() noexcept;
    [[nodiscard]] std::span<const float> state() const noexcept { return state_; }
    [[nodiscard]] std::span<const float> values() const noexcept { return values_; }
    [[nodiscard]] std::span<const float> presenceLogits() const noexcept { return logits_; }

private:
    std::array<std::size_t, 6> widths_{};
    std::size_t hidden_ = 0;
    Clock clock_{};
    float threshold_ = 0.0f;
    std::vector<float> inputWeights_, stateWeights_, inputBias_, stateBias_;
    std::vector<float> valueWeights_, valueBias_, maskWeights_, maskBias_;
    std::vector<float> input_, state_, nextState_, inputGates_, stateGates_, values_, logits_;
    std::vector<float> initialState_;
    std::vector<std::array<float, 2>> bounds_;
    std::array<Tensor, 4> controls_;
    std::vector<std::uint8_t> locks_;
    std::vector<float> lockedValues_;
    std::vector<std::pair<std::size_t, Slice>> heads_;
};
} // namespace alfa::tensor
