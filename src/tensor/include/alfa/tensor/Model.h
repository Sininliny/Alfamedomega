#pragma once

#include "alfa/tensor/Runtime.h"

#include <array>
#include <string_view>

namespace alfa::tensor
{
// CPU GRU inference for the independently budgeted compiler model. It predicts
// tensor values/presence; deterministic DSP remains responsible for all samples.
template <typename Scalar>
class TensorModelT
{
public:
    bool prepare(std::string_view json, const ConfigurationT<Scalar>& controls, std::string& error);
    bool infer(const InputsT<Scalar>& inputs) noexcept;
    bool apply(const InputsT<Scalar>& inputs, RuntimeT<Scalar>& runtime) noexcept;
    void reset() noexcept;
    [[nodiscard]] std::span<const Scalar> state() const noexcept { return state_; }
    [[nodiscard]] std::span<const Scalar> values() const noexcept { return values_; }
    [[nodiscard]] std::span<const Scalar> presenceLogits() const noexcept { return logits_; }

private:
    std::array<std::size_t, 6> widths_{};
    std::size_t hidden_ = 0;
    Clock clock_{};
    Scalar threshold_ = 0.0f;
    std::vector<Scalar> inputWeights_, stateWeights_, inputBias_, stateBias_;
    std::vector<Scalar> valueWeights_, valueBias_, maskWeights_, maskBias_;
    std::vector<Scalar> input_, state_, nextState_, inputGates_, stateGates_, values_, logits_;
    std::vector<Scalar> initialState_;
    std::vector<std::array<Scalar, 2>> bounds_;
    std::array<TensorT<Scalar>, 4> controls_;
    std::vector<std::uint8_t> locks_;
    std::vector<Scalar> lockedValues_;
    std::vector<std::pair<std::size_t, Slice>> heads_;
};
using TensorModel = TensorModelT<float>;
using TensorModel64 = TensorModelT<double>;
extern template class TensorModelT<float>;
extern template class TensorModelT<double>;
} // namespace alfa::tensor
