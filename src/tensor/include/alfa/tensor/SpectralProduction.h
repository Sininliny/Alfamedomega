#pragma once
#include "alfa/tensor/Kernels.h"

namespace alfa::tensor::spectralProduction
{
// Streaming mathematical operations, not a fixed spectral processing chain.
// Analysis state size=N+4, window asset0[N], parameter hop[1..N]. It captures
// audio[0] at each explicit instance update; the first hop emits a zero-extended
// startup frame. Output[0..2*N) is full interleaved complex FFT, output[2*N] is
// a binary fresh-frame flag. Other updates output zero, including the flag.
// Window frequency bins correspond to sampleRate/(clock.periodSamples*N).
//
// Synthesis state size=N+3, windows asset0[N] and reciprocal COLA asset1[N].
// Its input is the complete frame and flag above. Frames schedule normalized
// inverse FFT contributions into a ring beginning the NEXT update; readout
// always emits the old OLA cursor. Asset normalization is authored, not inferred.
// Connecting via committed output history at matching clocks therefore gives
// N+1 update latency for the zero-extended analysis/synthesis roundtrip.
// At periods above one host sample, select Hold for analysis between updates
// so its frame survives until the next due synthesis update. Differing clocks
// require an explicit frame transport/resampling policy, not inferred bridging.
// N must be a power of two >=2. Window/hop/FFT choices and resource capacities
// are explicit, and every persistent/scratch numeric value uses tier Scalar.
std::span<const kernels::Metadata> catalog() noexcept;
template <typename Scalar>
bool createPrimitive(std::string_view id, std::size_t stateElements, PrimitiveT<Scalar>& result,
                     std::string& error);
} // namespace alfa::tensor::spectralProduction
