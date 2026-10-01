#include "alfa/tensor/Universe.h"

#include "alfa/dsp/detail/DelayInterpolation.h"
#include "alfa/dsp/detail/FiltersBiquad.h"
#include "alfa/dsp/detail/FiltersSvf.h"
#include "alfa/tensor/NonlinearProduction.h"
#include "alfa/tensor/Production.h"
#include "alfa/tensor/SpectralProduction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <utility>

namespace alfa::tensor::universe
{
namespace
{
constexpr double tau = 2 * std::numbers::pi;
enum class Op
{
    Saw,
    Wavetable,
    Additive,
    Noise,
    Impulse,
    Sample,
    Grain,
    Fft,
    Ifft,
    SpectralMask,
    SpectralMorph,
    SpectralTilt,
    SpectralFreeze,
    Redistribution,
    Lowpass,
    Highpass,
    Bandpass,
    Svf,
    Comb,
    Resonator,
    Modal,
    String,
    Adsr,
    Multistage,
    MidiAffine,
    GestureFollower,
    LatentAffine,
    Saturation,
    Waveshape,
    Clip,
    Cubic,
    NonlinearFeedback,
    ModDelay,
    Phaser,
    Diffusion,
    Reverb,
    Convolution,
    StereoWidth,
    Pan,
    Matrix,
    Resample,
    ResampleCubic,
    ResampleSinc8,
    PhaseVocoder,
    Formant,
    Binaural,
    Room,
    GrainStretch
};
constexpr kernels::Parameter oscillatorParams[]{
    {"frequency", "Hz", "finite signed frequency", true, true, true},
    {"amplitude", "linear", "finite", true, true, true}};
constexpr kernels::Parameter sawParams[]{
    {"frequency", "Hz", "finite; abs(frequency) < updateRate/2", true, true, true},
    {"amplitude", "linear", "finite", true, true, true}};
constexpr kernels::Parameter additiveParams[]{
    {"frequency[i]", "Hz", "finite; repeated pair at parameter[2*i], one pair per phase state", true, true,
     true},
    {"amplitude[i]", "linear", "finite; repeated pair at parameter[2*i+1]", true, true, true}};
constexpr std::array<kernels::Parameter, 0> none{};
constexpr kernels::Parameter impulseParams[]{
    {"gate", "linear", "finite; positive rising gate triggers impulse", true, true, true},
    {"amplitude", "linear", "finite", true, true, true}};
constexpr kernels::Parameter sampleParams[]{
    {"speed", "asset frames/sec", "finite signed playback speed", true, true, true},
    {"amplitude", "linear", "finite", true, true, true}};
constexpr kernels::Parameter grainParams[]{
    {"speed", "asset frames/sec", "finite signed grain playback speed", true, true, true},
    {"duration", "seconds", "finite > 0", true, true, true},
    {"launch_gate", "linear", "finite; positive rising gate launches one available grain", true, true, true},
    {"amplitude", "linear", "finite", true, true, true}};
constexpr kernels::Parameter morphParams[]{
    {"blend", "linear", "[0,1]; Cartesian complex interpolation", true, true, true}};
constexpr kernels::Parameter tiltParams[]{
    {"slope", "dB/octave", "finite; resultant bin gain dB clamped to [-600,600] before exponentiation", true,
     true, true},
    {"anchor", "Hz", "finite > 0", true, true, true}};
constexpr kernels::Parameter freezeParams[]{
    {"freeze", "linear", "finite; > 0 holds spectrum, <= 0 captures input", true, true, true}};
constexpr kernels::Parameter redistributionParams[]{
    {"frequency_ratio", "ratio", "finite > 0; bins outside output range discarded", true, true, true}};
constexpr kernels::Parameter filterParams[]{{"cutoff", "Hz", "0 < value < updateRate/2", true, true, true},
                                            {"Q", "ratio", "finite > 0", true, true, true}};
constexpr kernels::Parameter svfParams[]{{"cutoff", "Hz", "0 < value < updateRate/2", true, true, true},
                                         {"damping", "linear", "finite > 0", true, true, true}};
constexpr kernels::Parameter feedbackParams[]{{"feedback", "linear", "abs(value) < 1", true, true, true}};
constexpr kernels::Parameter combParams[]{{"delayed_gain", "linear", "abs(value) < 1", true, true, true}};
constexpr kernels::Parameter diffusionParams[]{{"coefficient", "linear", "abs(value) < 1", true, true, true}};
constexpr kernels::Parameter resonatorParams[]{
    {"frequency[i]", "Hz", "0 <= value < updateRate/2; parameter[3*i], repeated triplet per mode", true, true,
     true},
    {"decay[i]", "seconds", "finite > 0; amplitude e-folding time; parameter[3*i+1]", true, true, true},
    {"gain[i]", "linear", "finite excitation gain; parameter[3*i+2]", true, true, true}};
constexpr kernels::Parameter adsrParams[]{
    {"gate", "linear", "finite; > 0 means on", true, true, true},
    {"attack", "seconds", "finite >= 0", true, true, true},
    {"decay", "seconds", "finite >= 0", true, true, true},
    {"sustain", "linear", "[0,1]", true, true, true},
    {"release", "seconds", "finite >= 0; measured from captured gate-off level", true, true, true}};
constexpr kernels::Parameter gateParams[]{
    {"gate", "linear", "finite; <= 0 resets timeline", true, true, true}};
constexpr kernels::Parameter affineParams[]{{"scale", "linear", "finite", true, true, true},
                                            {"offset", "linear", "finite", true, true, true}};
constexpr kernels::Parameter followerParams[]{
    {"attack", "seconds", "finite >= 0; exponential e-folding rise time", true, true, true},
    {"release", "seconds", "finite >= 0; exponential e-folding fall time", true, true, true}};
constexpr kernels::Parameter matrixParams[]{
    {"weight[row,column]", "linear",
     "finite; row-major parameter[row*N+column], M*N values; M=state_size,N=input width", true, true, true}};
constexpr kernels::Parameter driveParams[]{{"drive", "linear", "finite >= 0", true, true, true}};
constexpr kernels::Parameter clipParams[]{{"threshold", "linear", "finite > 0", true, true, true}};
constexpr kernels::Parameter nonlinearParams[]{
    {"drive", "linear", "finite >= 0", true, true, true},
    {"feedback", "linear", "finite; tanh bounds recurrent state for any finite feedback", true, true, true}};
constexpr kernels::Parameter modDelayParams[]{
    {"base_delay", "seconds", "base +/- depth must lie within [one update,allocated delay duration]", true,
     true, true},
    {"depth", "seconds", "finite >= 0; complete modulation range must fit allocated delay", true, true, true},
    {"rate", "Hz", "finite signed modulation frequency", true, true, true},
    {"mix", "linear", "[0,1] wet fraction", true, true, true},
    {"feedback", "linear", "abs(value) < 1", true, true, true}};
constexpr kernels::Parameter phaserParams[]{
    {"coefficient", "linear", "abs(value) < 1", true, true, true},
    {"mix", "linear", "[0,1] allpass wet fraction", true, true, true}};
constexpr kernels::Parameter widthParams[]{{"width", "linear", "finite >= 0", true, true, true}};
constexpr kernels::Parameter panParams[]{
    {"position", "linear", "[-1,1] constant-power stereo pan", true, true, true}};
constexpr kernels::Parameter vocoderParams[]{
    {"analysis_hop", "samples", "finite > 0; elapsed samples between source analysis frames", true, true,
     true},
    {"synthesis_hop", "samples", "finite > 0; target overlap-add hop supplied by producer", true, true, true},
    {"pitch_ratio", "ratio", "finite > 0; pitch-bin redistribution and instantaneous-frequency scaling", true,
     true, true}};
constexpr kernels::Parameter formantParams[]{
    {"pitch ratio", "ratio", "positive; supplied envelope compensates a previously pitched spectrum", true,
     true, true}};
constexpr kernels::Parameter roomParams[]{
    {"length_x", "metres", "finite > 0", true, true, true},
    {"length_y", "metres", "finite > 0", true, true, true},
    {"length_z", "metres", "finite > 0", true, true, true},
    {"source_x", "metres", "0 < value < length_x; source and receiver must differ", true, true, true},
    {"source_y", "metres", "0 < value < length_y", true, true, true},
    {"source_z", "metres", "0 < value < length_z", true, true, true},
    {"receiver_x", "metres", "0 < value < length_x; delay memory must cover all seven propagation paths",
     true, true, true},
    {"receiver_y", "metres", "0 < value < length_y", true, true, true},
    {"receiver_z", "metres", "0 < value < length_z", true, true, true},
    {"sound_speed", "metres/sec", "finite > 0", true, true, true},
    {"wall_gain_x_min", "linear", "[-1,1]", true, true, true},
    {"wall_gain_x_max", "linear", "[-1,1]", true, true, true},
    {"wall_gain_y_min", "linear", "[-1,1]", true, true, true},
    {"wall_gain_y_max", "linear", "[-1,1]", true, true, true},
    {"wall_gain_z_min", "linear", "[-1,1]", true, true, true},
    {"wall_gain_z_max", "linear", "[-1,1]", true, true, true}};
constexpr kernels::Parameter stretchParams[]{
    {"scan_speed", "asset frames/sec", "finite signed source scan speed", true, true, true},
    {"grain_speed", "asset frames/sec", "finite signed playback speed inside each grain", true, true, true},
    {"duration", "seconds", "finite > 0", true, true, true},
    {"launch_interval", "seconds", "finite > 0; exhausted allocated grain capacity skips launch", true, true,
     true},
    {"amplitude", "linear", "finite", true, true, true}};

// Rate descriptions use the explicitly chosen per-instance update rate. No global
// control clock, automatic wiring, or algorithm selection is hidden in the catalog.
constexpr std::string_view rates = "explicit positive clock; updateRate = sampleRate / periodSamples";
constexpr std::string_view reuse = "bounded only by explicit configuration/hardware capacity";
constexpr kernels::Metadata records[]{
    {"osc.polyblep_saw", "none; params frequency,amplitude; phase initial [0,1)", "scalar", sawParams,
     "one phase cycle", "O(1)", rates, reuse},
    {"osc.wavetable", "asset[0] periodic scalar table, >=2 values; frequency,amplitude", "scalar",
     oscillatorParams, "one phase cycle [0,1)", "O(1) linear interpolation", rates, reuse},
    {"osc.additive", "frequency/amplitude pair per state phase", "scalar partial sum", additiveParams,
     "one phase cycle per partial", "O(partials)", rates, reuse},
    {"excitation.white_noise", "none; initial state two nonzero uint16 seed words", "scalar [-1,1]", none,
     "two exact uint16 words for xorshift32", "O(1)", rates, reuse},
    {"excitation.impulse", "gate,amplitude parameters", "rising gate impulse", impulseParams, "previous gate",
     "O(1)", rates, reuse},
    {"excitation.sample_loop", "asset[0] mono sample; speed frames/sec,amplitude",
     "scalar interpolated sample", sampleParams, "one normalized loop position [0,1)", "O(1)", rates, reuse},
    {"excitation.granular", "asset[0] mono sample; speed,duration,launchGate,amplitude",
     "overlapping Hann-windowed grains", grainParams,
     "previous gate plus 3 values per grain: active,ageSec,normalizedPosition", "O(grain capacity)", rates,
     reuse},
    {"spectral.fft", "audio interleaved complex N, N power-of-two; no params",
     "interleaved complex N, forward unnormalized", none, "2*N transform output; readout prior frame",
     "O(N log N) radix-2 CPU FFT", rates, reuse},
    {"spectral.ifft", "audio interleaved complex N, N power-of-two; no params",
     "interleaved complex N, inverse /N", none, "2*N transform output; readout prior frame",
     "O(N log N) radix-2 CPU FFT", rates, reuse},
    {"spectral.mask", "audio complex N; asset[0] real bin gains N; no params", "complex N weighted bins",
     none, "2*N prior processed frame", "O(N)", rates, reuse},
    {"spectral.morph", "audio two concatenated complex N frames; param blend [0,1]",
     "complex Cartesian interpolation (phase cancellation possible)", morphParams,
     "2*N prior processed frame", "O(N)", rates, reuse},
    {"spectral.tilt", "audio full complex N; params slope dB/oct,anchor Hz>0",
     "complex symmetric frequency tilt; DC anchored at first nonzero bin", tiltParams,
     "2*N prior processed frame", "O(N)", rates, reuse},
    {"spectral.freeze", "audio complex N; param freeze (>0 holds state)", "complex N frozen/captured frame",
     freezeParams, "2*N prior processed frame", "O(N)", rates, reuse},
    {"spectral.redistribute", "audio positive-frequency complex N; param frequency ratio>0",
     "complex bins linearly redistributed; out-of-range discarded", redistributionParams,
     "2*N prior processed frame", "O(N)", rates, reuse},
    {"filter.biquad_lowpass", "audio scalar; params cutoff,Q", "scalar low-pass", filterParams, "x1,x2,y1,y2",
     "O(1)", rates, reuse},
    {"filter.biquad_highpass", "audio scalar; params cutoff,Q", "scalar high-pass", filterParams,
     "x1,x2,y1,y2", "O(1)", rates, reuse},
    {"filter.biquad_bandpass", "audio scalar; params cutoff,Q", "scalar constant-peak band-pass",
     filterParams, "x1,x2,y1,y2", "O(1)", rates, reuse},
    {"filter.state_variable", "audio scalar; params cutoff,damping", "three scalar outputs low,band,high",
     svfParams, "two trapezoidal integrator memories", "O(1)", rates, reuse},
    {"filter.feedforward_comb", "audio scalar; param gain abs<1", "input + gain*delayedInput", combParams,
     "explicit length shift register", "O(length) reference kernel", rates, reuse},
    {"physical.resonator", "audio scalar excitation; params frequency,decay,gain",
     "scalar damped second-order response", resonatorParams, "two response memories", "O(1)", rates, reuse},
    {"physical.modal_bank", "audio scalar excitation; repeated frequency,decay,gain",
     "summed independent acoustic modes", resonatorParams, "two response memories per mode", "O(modes)",
     rates, reuse},
    {"physical.string_waveguide", "audio scalar excitation; param loop feedback",
     "Karplus-Strong waveguide with two-tap loss averaging", feedbackParams, "explicit delay length >=2",
     "O(length) reference kernel", rates, reuse},
    {"control.adsr", "params gate,attack,decay,sustain,release; no audio", "linear ADSR level", adsrParams,
     "level,stage,previousGate,releaseStart", "O(1)", rates, reuse},
    {"control.multistage_envelope", "asset[0] pairs durationSeconds>0,target; param gate",
     "piecewise-linear timeline; final target held", gateParams, "elapsedSeconds,level", "O(segments)", rates,
     reuse},
    {"control.midi_affine", "inputs.midi.values[0] explicitly bound scalar; scale,offset",
     "scalar mapped MIDI control", affineParams, "none", "O(1)", rates, reuse},
    {"control.gesture_follower", "audio scalar; params attackSeconds,releaseSeconds",
     "absolute-amplitude gesture follower", followerParams, "one smoothed magnitude", "O(1)", rates, reuse},
    {"control.latent_affine", "latentSound vector N; parameter weights row-major M*N",
     "M learned linear modulation values", matrixParams, "M prior projection outputs", "O(M*N)", rates,
     reuse},
    {"nonlinear.tanh", "audio scalar; param drive", "tanh(drive*input)", driveParams, "none", "O(1)", rates,
     reuse},
    {"nonlinear.waveshape_table", "audio scalar; asset[0] transfer curve over [-1,1]; param drive",
     "linearly interpolated transfer function", driveParams, "none", "O(1)", rates, reuse},
    {"nonlinear.clip", "audio scalar; param threshold", "hard-clipped scalar", clipParams, "none", "O(1)",
     rates, reuse},
    {"nonlinear.cubic_softclip", "audio scalar; param drive", "1.5*x-0.5*x^3 on clamped driven [-1,1]",
     driveParams, "none", "O(1)", rates, reuse},
    {"nonlinear.feedback_tanh", "audio scalar; params drive,feedback",
     "previous recurrent nonlinear response", nonlinearParams,
     "one response: next=tanh(drive*x+feedback*state)", "O(1)", rates, reuse},
    {"time.modulated_delay", "audio scalar; params base,depth,rate,mix,feedback",
     "fractional sine-modulated delay; supports chorus/flanger constructions", modDelayParams,
     "phase cycle plus explicit delay samples >=2", "O(length) reference kernel", rates, reuse},
    {"time.phaser_one_stage", "audio scalar; params allpass coefficient,mix",
     "dry/allpass mix creates phase cancellation", phaserParams, "previous input and allpass output", "O(1)",
     rates, reuse},
    {"time.diffusion_allpass", "audio scalar; param coefficient abs<1", "Schroeder delay allpass response",
     diffusionParams, "explicit delay length", "O(length) reference kernel", rates, reuse},
    {"time.fdn_reverb",
     "audio scalar; asset[0] positive integer delay lengths summing to state_size; param feedback",
     "Householder feedback delay network wet output", feedbackParams,
     "concatenated delay memories; no automatic room inference", "O(total state + delay lines)", rates,
     reuse},
    {"time.convolution_fir", "audio scalar; asset[0] real impulse response length state_size+1; no params",
     "causal direct FIR convolution", none, "past input samples", "O(IR length) reference kernel", rates,
     reuse},
    {"spatial.stereo_width", "audio stereo L,R; param width", "mid/side width-transformed stereo",
     widthParams, "none", "O(1)", rates, reuse},
    {"spatial.constant_power_pan", "audio mono; param position [-1,1]", "stereo L,R", panParams, "none",
     "O(1)", rates, reuse},
    {"spatial.matrix", "audio vector N; parameter row-major weights M*N",
     "M-channel arbitrary spatial transform", matrixParams, "M prior transformed outputs", "O(M*N)", rates,
     reuse},
    {"pitchtime.asset_resample", "asset[0] mono sample; params speed frames/sec,amplitude",
     "nonlooping linear resampling; out-of-range silence", sampleParams,
     "one absolute normalized sample position (zero start)", "O(1)", rates, reuse},
    {"pitchtime.asset_resample_cubic", "asset[0] mono sample; params speed frames/sec,amplitude",
     "nonlooping four-point legacy Catmull-Rom interpolation; zero-extended boundaries", sampleParams,
     "one normalized sample position", "O(4); interpolation, no downsample low-pass", rates, reuse},
    {"pitchtime.asset_resample_sinc8", "asset[0] mono sample; params speed frames/sec,amplitude",
     "nonlooping eight-tap legacy normalized Kaiser-sinc interpolation; zero-extended boundaries",
     sampleParams, "one normalized sample position; immutable prepared coefficient table",
     "O(8); fractional interpolation, no ratio-dependent downsample low-pass", rates, reuse},
    {"pitchtime.phase_vocoder", "audio full complex FFT N; params analysisHop,synthesisHop,pitchRatio",
     "complex N phase-unwrapped spectral frames with pitch-bin redistribution", vocoderParams,
     "1+4*N: primed flag, previous analysis phases N, synthesis phases N, prior output complex N",
     "O(N); frame kernel requires explicit window/overlap-add framing by producer", rates, reuse},
    {"pitchtime.formant_compensation",
     "audio already-pitched full complex N; asset[0] strictly positive original spectral-envelope gains N; "
     "param pitchRatio",
     "complex spectrum scaled envelope[k]/envelope[k/pitchRatio] (signed bins)", formantParams,
     "2*N prior compensated frame; envelope estimation is explicit external analysis", "O(N)", rates, reuse},
    {"spatial.binaural_hrtf_fir",
     "audio mono; assets[0,1] supplied left/right HRTF impulse responses; max length=state_size+1; no params",
     "stereo actual HRTF FIR response", none, "past mono samples; no invented HRTF dataset",
     "O(leftIR+rightIR) reference direct convolution", rates, reuse},
    {"spatial.room_first_order",
     "audio mono; explicit box geometry/source/receiver/speed/wall gains parameters",
     "direct plus six image-source first-order room reflections with fractional propagation delays",
     roomParams, "past mono samples; distance attenuation 1/r; no inferred geometry or high-order claim",
     "O(state length+7 paths) reference kernel", rates, reuse},
    {"pitchtime.granular_stretch",
     "asset[0] mono sample; independent scan/grain speeds, duration, launch interval, amplitude",
     "overlapping Hann grains decouple scan time and playback pitch", stretchParams,
     "2+3*grainCapacity: launch phase, normalized scan position, then active/age/position per grain",
     "O(grain capacity)", rates, reuse}};

template <typename Scalar>
Scalar safe(double v) noexcept
{
    if (!std::isfinite(v))
        return 0.0f;
    const double limit = std::numeric_limits<Scalar>::max();
    return static_cast<Scalar>(std::clamp(v, -limit, limit));
}
template <typename Scalar>
void zero(std::span<Scalar> values) noexcept
{
    std::fill(values.begin(), values.end(), 0.0f);
}
template <typename Scalar>
void hold(const StepT<Scalar>& s, std::span<Scalar> next) noexcept
{
    zero<Scalar>(next);
    std::copy_n(s.localState.begin(), std::min(s.localState.size(), next.size()), next.begin());
}
template <typename Scalar>
bool allFinite(std::span<const Scalar> a) noexcept
{
    return std::all_of(a.begin(), a.end(), [](Scalar v) { return std::isfinite(v); });
}
template <typename Scalar>
bool base(const StepT<Scalar>& s) noexcept
{
    if (!(s.sampleRate > 0) || !std::isfinite(s.sampleRate) || s.clock.periodSamples == 0)
        return false;
    const double interval = static_cast<double>(s.clock.periodSamples) / s.sampleRate;
    return interval > 0 && std::isfinite(interval) && std::isfinite(s.activation) && s.activation >= 0 &&
           allFinite<Scalar>(s.parameters) && allFinite<Scalar>(s.localState);
}
template <typename Scalar>
double dt(const StepT<Scalar>& s) noexcept
{
    return static_cast<double>(s.clock.periodSamples) / s.sampleRate;
}
template <typename Scalar>
Scalar audio(const StepT<Scalar>& s, std::size_t i = 0) noexcept
{
    return i < s.inputs.audio.values.size() && std::isfinite(s.inputs.audio.values[i])
               ? s.inputs.audio.values[i]
               : 0.0f;
}
template <typename Scalar>
std::span<const Scalar> asset(const StepT<Scalar>& s, std::size_t i) noexcept
{
    return i < s.assets.size() ? s.assets[i].values : std::span<const Scalar>{};
}
double wrap(double x) noexcept
{
    return std::isfinite(x) ? x - std::floor(x) : 0;
}
template <typename Scalar>
Scalar interpolate(std::span<const Scalar> a, double position, bool looping) noexcept
{
    if (a.empty() || !std::isfinite(position))
        return 0.0f;
    if (looping)
        position = wrap(position);
    else if (position < 0 || position >= 1)
        return 0.0f;
    const double index = position * static_cast<double>(a.size());
    const auto first = std::min(static_cast<std::size_t>(index), a.size() - 1);
    const auto second = first + 1 < a.size() ? first + 1 : looping ? 0 : first;
    return safe<Scalar>(a[first] +
                        (index - static_cast<double>(first)) * (static_cast<double>(a[second]) - a[first]));
}
template <typename Scalar>
Scalar interpolatedAsset(std::span<const Scalar> table, double position, Op op) noexcept
{
    if (table.empty() || !std::isfinite(position) || position < 0 || position >= 1)
        return 0.0f;
    const double index = position * static_cast<double>(table.size());
    const auto baseIndex = static_cast<int64_t>(index);
    const auto sample = [&](int64_t offset) -> Scalar
    {
        const auto i = baseIndex + offset;
        return i >= 0 && static_cast<uint64_t>(i) < table.size() ? table[static_cast<size_t>(i)] : 0.0f;
    };
    const Scalar fraction = static_cast<Scalar>(index - static_cast<double>(baseIndex));
    if (op == Op::ResampleCubic)
        return alfa::dsp::detail::hermite(sample(-1), sample(0), sample(1), sample(2), fraction);
    // Legacy delay fraction goes toward older samples; asset position goes forward.
    // Symmetric taps -3..4 apply the same immutable normalized Kaiser-sinc kernel.
    std::array<Scalar, 8> weights{};
    alfa::dsp::detail::typedSincTable<Scalar>().weights(fraction, weights.data());
    double sum = 0;
    for (size_t tap = 0; tap < weights.size(); ++tap)
        sum += static_cast<double>(weights[tap]) * sample(static_cast<int64_t>(tap) - 3);
    return safe<Scalar>(sum);
}
template <typename Scalar>
bool phaseValid(const StepT<Scalar>& s) noexcept
{
    return s.localState.size() == 1 && s.localState[0] >= 0 && s.localState[0] < 1;
}
bool powerOfTwo(std::size_t n) noexcept
{
    return n && (n & (n - 1)) == 0;
}
template <typename Scalar>
double modalValue(const StepT<Scalar>& s, std::size_t mode) noexcept
{
    const auto p = mode * 3;
    const auto q = mode * 2;
    const double decay = std::exp(-dt<Scalar>(s) / s.parameters[p + 1]);
    return static_cast<double>(audio<Scalar>(s)) * s.parameters[p + 2] +
           2 * decay * std::cos(tau * s.parameters[p] * dt<Scalar>(s)) * s.localState[q] -
           decay * decay * s.localState[q + 1];
}
template <typename Scalar>
double biquadValue(const StepT<Scalar>& s, Op op) noexcept
{
    // Effective sample rate follows this primitive's own declared clock.
    const auto design =
        alfa::dsp::detail::biquad::warpUnclamped(s.parameters[0], s.parameters[1], 1.0 / dt<Scalar>(s));
    const auto coefficients = op == Op::Lowpass    ? alfa::dsp::detail::biquad::lowpass(design)
                              : op == Op::Highpass ? alfa::dsp::detail::biquad::highpass(design)
                                                   : alfa::dsp::detail::biquad::bandpass(design);
    // Preserve the existing x1,x2,y1,y2 tensor coordinates; no hidden Biquad object.
    return coefficients.b0 * audio<Scalar>(s) + coefficients.b1 * s.localState[0] +
           coefficients.b2 * s.localState[1] - coefficients.a1 * s.localState[2] -
           coefficients.a2 * s.localState[3];
}
template <typename Scalar>
double delayValue(const StepT<Scalar>& s) noexcept
{
    const auto memory = s.localState.subspan(1);
    const double samples =
        (s.parameters[0] + s.parameters[1] * std::sin(tau * s.localState[0])) / dt<Scalar>(s);
    const double index = std::clamp(samples - 1, 0.0, static_cast<double>(memory.size() - 1));
    const auto i = static_cast<std::size_t>(index);
    const auto j = std::min(i + 1, memory.size() - 1);
    return memory[i] + (index - static_cast<double>(i)) * (static_cast<double>(memory[j]) - memory[i]);
}
template <typename Scalar>
double roomDistance(const StepT<Scalar>& s, std::size_t path) noexcept
{
    double source[3]{s.parameters[3], s.parameters[4], s.parameters[5]};
    if (path)
    {
        const auto axis = (path - 1) / 2;
        source[axis] = (path % 2) ? -source[axis] : 2 * s.parameters[axis] - source[axis];
    }
    double squared = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const double d = source[axis] - s.parameters[6 + axis];
        squared += d * d;
    }
    return std::sqrt(squared);
}
template <typename Scalar>
double pastInput(const StepT<Scalar>& s, double delaySamples) noexcept
{
    if (delaySamples < 1)
        return (1 - delaySamples) * audio<Scalar>(s) + delaySamples * s.localState[0];
    const double index = delaySamples - 1;
    const auto i = static_cast<std::size_t>(index);
    const auto j = std::min(i + 1, s.localState.size() - 1);
    return s.localState[i] +
           (index - static_cast<double>(i)) * (static_cast<double>(s.localState[j]) - s.localState[i]);
}
template <typename Scalar>
bool valid(const StepT<Scalar>& s, Op op) noexcept
{
    if (!base<Scalar>(s))
        return false;
    const auto p = s.parameters;
    const auto n = s.localState.size();
    const auto table = asset<Scalar>(s, 0);
    switch (op)
    {
    case Op::Saw:
        return p.size() == 2 && phaseValid<Scalar>(s) && std::abs(p[0]) * dt<Scalar>(s) < 0.5;
    case Op::Wavetable:
        return p.size() == 2 && phaseValid<Scalar>(s) && table.size() >= 2;
    case Op::Additive:
        return n > 0 && p.size() == 2 * n &&
               std::all_of(s.localState.begin(), s.localState.end(),
                           [](Scalar x) { return x >= 0 && x < 1; });
    case Op::Noise:
        return p.empty() && n == 2 && s.localState[0] >= 0 && s.localState[0] <= 65535 &&
               s.localState[1] >= 0 && s.localState[1] <= 65535 &&
               std::floor(s.localState[0]) == s.localState[0] &&
               std::floor(s.localState[1]) == s.localState[1] &&
               (s.localState[0] != 0 || s.localState[1] != 0);
    case Op::Impulse:
        return p.size() == 2 && n == 1;
    case Op::Sample:
        return p.size() == 2 && phaseValid<Scalar>(s) && !table.empty();
    case Op::Resample:
    case Op::ResampleCubic:
    case Op::ResampleSinc8:
        return p.size() == 2 && n == 1 && !table.empty();
    case Op::Grain:
        return p.size() == 4 && p[1] > 0 && n >= 4 && (n - 1) % 3 == 0 && !table.empty();
    case Op::Fft:
    case Op::Ifft:
        return p.empty() && n >= 2 && n % 2 == 0 && powerOfTwo(n / 2) && s.inputs.audio.values.size() == n &&
               allFinite<Scalar>(s.inputs.audio.values);
    case Op::SpectralMask:
        return p.empty() && n >= 2 && n % 2 == 0 && s.inputs.audio.values.size() == n &&
               table.size() == n / 2 && allFinite<Scalar>(table);
    case Op::SpectralMorph:
        return p.size() == 1 && p[0] >= 0 && p[0] <= 1 && n >= 2 && n % 2 == 0 &&
               s.inputs.audio.values.size() == 2 * n;
    case Op::SpectralTilt:
        return p.size() == 2 && p[1] > 0 && n >= 2 && n % 2 == 0 && s.inputs.audio.values.size() == n;
    case Op::SpectralFreeze:
        return p.size() == 1 && n >= 2 && n % 2 == 0 && s.inputs.audio.values.size() == n;
    case Op::Redistribution:
        return p.size() == 1 && p[0] > 0 && n >= 2 && n % 2 == 0 && s.inputs.audio.values.size() == n;
    case Op::Lowpass:
    case Op::Highpass:
    case Op::Bandpass:
        return p.size() == 2 && n == 4 && p[0] > 0 && p[0] * dt<Scalar>(s) < 0.5 && p[1] > 0;
    case Op::Svf:
        return p.size() == 2 && n == 2 && p[0] > 0 && p[0] * dt<Scalar>(s) < 0.5 && p[1] > 0;
    case Op::Comb:
    case Op::Diffusion:
        return p.size() == 1 && n > 0 && std::abs(p[0]) < 1;
    case Op::String:
        return p.size() == 1 && n >= 2 && std::abs(p[0]) < 1;
    case Op::Resonator:
    case Op::Modal:
        if (n == 0 || n % 2 || p.size() != 3 * (n / 2))
            return false;
        for (std::size_t i = 0; i < p.size(); i += 3)
            if (p[i] < 0 || p[i] * dt<Scalar>(s) >= 0.5 || p[i + 1] <= 0)
                return false;
        return true;
    case Op::Adsr:
        return p.size() == 5 && n == 4 && p[1] >= 0 && p[2] >= 0 && p[3] >= 0 && p[3] <= 1 && p[4] >= 0 &&
               s.localState[0] >= 0 && s.localState[0] <= 1 && s.localState[1] >= 0 && s.localState[1] <= 4;
    case Op::Multistage:
        if (p.size() != 1 || n != 2 || table.empty() || table.size() % 2 || !allFinite<Scalar>(table))
            return false;
        for (std::size_t i = 0; i < table.size(); i += 2)
            if (table[i] <= 0)
                return false;
        return true;
    case Op::MidiAffine:
        return p.size() == 2 && n == 0;
    case Op::GestureFollower:
        return p.size() == 2 && p[0] >= 0 && p[1] >= 0 && n == 1;
    case Op::LatentAffine:
        return n > 0 && !s.inputs.latentSound.values.empty() &&
               p.size() / n == s.inputs.latentSound.values.size() && p.size() % n == 0 &&
               allFinite<Scalar>(s.inputs.latentSound.values);
    case Op::Matrix:
        return n > 0 && !s.inputs.audio.values.empty() && p.size() / n == s.inputs.audio.values.size() &&
               p.size() % n == 0 && allFinite<Scalar>(s.inputs.audio.values);
    case Op::Saturation:
    case Op::Cubic:
        return p.size() == 1 && p[0] >= 0 && n == 0;
    case Op::Waveshape:
        return p.size() == 1 && p[0] >= 0 && n == 0 && table.size() >= 2;
    case Op::Clip:
        return p.size() == 1 && p[0] > 0 && n == 0;
    case Op::NonlinearFeedback:
        return p.size() == 2 && p[0] >= 0 && n == 1;
    case Op::ModDelay:
        return p.size() == 5 && n >= 3 && s.localState[0] >= 0 && s.localState[0] < 1 && p[1] >= 0 &&
               (p[0] - p[1]) / dt<Scalar>(s) >= 1 &&
               (p[0] + p[1]) / dt<Scalar>(s) <= static_cast<double>(n - 1) && p[3] >= 0 && p[3] <= 1 &&
               std::abs(p[4]) < 1;
    case Op::Phaser:
        return p.size() == 2 && n == 2 && std::abs(p[0]) < 1 && p[1] >= 0 && p[1] <= 1;
    case Op::Reverb:
        if (p.size() != 1 || std::abs(p[0]) >= 1 || table.empty() || !allFinite<Scalar>(table))
            return false;
        {
            std::size_t sum = 0;
            for (Scalar length : table)
            {
                if (length < 1 || std::floor(length) != length ||
                    static_cast<double>(length) > static_cast<double>(n - sum))
                    return false;
                sum += static_cast<std::size_t>(length);
            }
            return sum == n;
        }
    case Op::Convolution:
        return p.empty() && table.size() == n + 1 && allFinite<Scalar>(table);
    case Op::StereoWidth:
        return p.size() == 1 && p[0] >= 0 && n == 0;
    case Op::Pan:
        return p.size() == 1 && p[0] >= -1 && p[0] <= 1 && n == 0;
    case Op::PhaseVocoder:
        return p.size() == 3 && p[0] > 0 && p[1] > 0 && p[2] > 0 && n >= 5 && (n - 1) % 4 == 0 &&
               s.inputs.audio.values.size() == (n - 1) / 2 && allFinite<Scalar>(s.inputs.audio.values);
    case Op::Formant:
        return p.size() == 1 && p[0] > 0 && n >= 2 && n % 2 == 0 && s.inputs.audio.values.size() == n &&
               table.size() == n / 2 && allFinite<Scalar>(table) &&
               std::all_of(table.begin(), table.end(), [](Scalar x) { return x > 0; });
    case Op::Binaural:
        return p.empty() && !table.empty() && !asset<Scalar>(s, 1).empty() && table.size() <= n + 1 &&
               asset<Scalar>(s, 1).size() <= n + 1 &&
               std::max(table.size(), asset<Scalar>(s, 1).size()) == n + 1 && allFinite<Scalar>(table) &&
               allFinite<Scalar>(asset<Scalar>(s, 1));
    case Op::Room:
        if (p.size() != 16 || n == 0 || p[9] <= 0)
            return false;
        for (std::size_t i = 0; i < 3; ++i)
            if (p[i] <= 0 || p[3 + i] <= 0 || p[3 + i] >= p[i] || p[6 + i] <= 0 || p[6 + i] >= p[i])
                return false;
        for (std::size_t i = 10; i < 16; ++i)
            if (std::abs(p[i]) > 1)
                return false;
        for (std::size_t path = 0; path < 7; ++path)
        {
            const double distance = roomDistance<Scalar>(s, path);
            if (distance <= 0 || distance / (p[9] * dt<Scalar>(s)) > static_cast<double>(n))
                return false;
        }
        return true;
    case Op::GrainStretch:
        return p.size() == 5 && p[2] > 0 && p[3] > 0 && n >= 5 && (n - 2) % 3 == 0 && !table.empty() &&
               s.localState[0] >= 0 && s.localState[1] >= 0 && s.localState[1] < 1;
    }
    return false;
}

template <typename Scalar>
void fft(std::span<Scalar> z, bool inverse) noexcept
{
    const auto n = z.size() / 2;
    for (std::size_t i = 1, j = 0; i < n; ++i)
    {
        auto bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
        {
            std::swap(z[2 * i], z[2 * j]);
            std::swap(z[2 * i + 1], z[2 * j + 1]);
        }
    }
    for (std::size_t length = 2; length <= n; length *= 2)
    {
        const double angle = (inverse ? tau : -tau) / static_cast<double>(length);
        const std::complex<double> step{std::cos(angle), std::sin(angle)};
        for (std::size_t start = 0; start < n; start += length)
        {
            std::complex<double> w{1, 0};
            for (std::size_t j = 0; j < length / 2; ++j)
            {
                const auto a = start + j, b = a + length / 2;
                const std::complex<double> u{z[2 * a], z[2 * a + 1]};
                const auto v = std::complex<double>{z[2 * b], z[2 * b + 1]} * w;
                z[2 * a] = safe<Scalar>((u + v).real());
                z[2 * a + 1] = safe<Scalar>((u + v).imag());
                z[2 * b] = safe<Scalar>((u - v).real());
                z[2 * b + 1] = safe<Scalar>((u - v).imag());
                w *= step;
            }
        }
        if (length == n)
            break;
    }
    if (inverse)
        for (Scalar& x : z)
            x = safe<Scalar>(x / static_cast<double>(n));
}

template <Op op, typename Scalar>
void transition(const StepT<Scalar>& s, std::span<Scalar> next) noexcept
{
    hold<Scalar>(s, next);
    if (next.size() != s.localState.size() || !valid<Scalar>(s, op))
        return;
    const auto p = s.parameters;
    const auto n = next.size();
    const auto table = asset<Scalar>(s, 0);
    if constexpr (op == Op::Saw || op == Op::Wavetable)
    {
        next[0] = safe<Scalar>(wrap(s.localState[0] + p[0] * dt<Scalar>(s)));
        if (next[0] >= 1)
            next[0] = 0.0f;
    }
    if constexpr (op == Op::Additive)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            next[i] = safe<Scalar>(wrap(s.localState[i] + p[2 * i] * dt<Scalar>(s)));
            if (next[i] >= 1)
                next[i] = 0.0f;
        }
    }
    if constexpr (op == Op::Noise)
    {
        auto seed =
            static_cast<std::uint32_t>(s.localState[0]) | (static_cast<std::uint32_t>(s.localState[1]) << 16);
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        next[0] = static_cast<Scalar>(seed & 65535u);
        next[1] = static_cast<Scalar>(seed >> 16);
    }
    if constexpr (op == Op::Impulse)
    {
        next[0] = p[0];
    }
    if constexpr (op == Op::Sample || op == Op::Resample || op == Op::ResampleCubic ||
                  op == Op::ResampleSinc8)
    {
        const double position = s.localState[0] + p[0] * dt<Scalar>(s) / static_cast<double>(table.size());
        next[0] = safe<Scalar>(op == Op::Sample ? wrap(position) : position);
        if constexpr (op == Op::Sample)
        {
            if (next[0] >= 1)
                next[0] = 0.0f;
        }
    }
    if constexpr (op == Op::Grain)
    {
        for (std::size_t i = 1; i < n; i += 3)
            if (s.localState[i] > 0)
            {
                next[i + 1] = safe<Scalar>(s.localState[i + 1] + dt<Scalar>(s));
                next[i + 2] = safe<Scalar>(
                    wrap(s.localState[i + 2] + p[0] * dt<Scalar>(s) / static_cast<double>(table.size())));
                if (next[i + 1] >= p[1])
                    next[i] = 0.0f;
            }
        if (p[2] > 0 && s.localState[0] <= 0)
            for (std::size_t i = 1; i < n; i += 3)
                if (next[i] <= 0)
                {
                    next[i] = 1.0f;
                    next[i + 1] = 0.0f;
                    next[i + 2] = 0.0f;
                    break;
                }
        next[0] = p[2];
    }
    if constexpr (op == Op::Fft || op == Op::Ifft)
    {
        std::copy(s.inputs.audio.values.begin(), s.inputs.audio.values.end(), next.begin());
        fft<Scalar>(next, op == Op::Ifft);
    }
    if constexpr (op == Op::SpectralMask)
    {
        for (std::size_t i = 0; i < n; ++i)
            next[i] = safe<Scalar>(static_cast<double>(audio<Scalar>(s, i)) * table[i / 2]);
    }
    if constexpr (op == Op::SpectralMorph)
    {
        for (std::size_t i = 0; i < n; ++i)
            next[i] = safe<Scalar>((1 - static_cast<double>(p[0])) * audio<Scalar>(s, i) +
                                   static_cast<double>(p[0]) * audio<Scalar>(s, n + i));
    }
    if constexpr (op == Op::SpectralTilt)
    {
        for (std::size_t i = 0; i < n / 2; ++i)
        {
            const auto bin = std::max(std::min(i, n / 2 - i), std::size_t{1});
            const double frequency = static_cast<double>(bin) / (static_cast<double>(n / 2) * dt<Scalar>(s));
            const double db =
                std::clamp(static_cast<double>(p[0]) * std::log2(frequency / p[1]), -600.0, 600.0);
            const double gain = std::pow(10.0, db / 20);
            next[2 * i] = safe<Scalar>(audio<Scalar>(s, 2 * i) * gain);
            next[2 * i + 1] = safe<Scalar>(audio<Scalar>(s, 2 * i + 1) * gain);
        }
    }
    if constexpr (op == Op::SpectralFreeze)
    {
        if (p[0] <= 0)
            for (std::size_t i = 0; i < n; ++i)
                next[i] = audio<Scalar>(s, i);
    }
    if constexpr (op == Op::Redistribution)
    {
        zero<Scalar>(next);
        for (std::size_t i = 0; i < n / 2; ++i)
        {
            const double position = static_cast<double>(i) * p[0];
            if (position >= static_cast<double>(n / 2))
                continue;
            const auto j = static_cast<std::size_t>(position);
            const double mix = position - static_cast<double>(j);
            for (std::size_t c = 0; c < 2; ++c)
            {
                next[2 * j + c] = safe<Scalar>(next[2 * j + c] + audio<Scalar>(s, 2 * i + c) * (1 - mix));
                if (j + 1 < n / 2)
                    next[2 * (j + 1) + c] =
                        safe<Scalar>(next[2 * (j + 1) + c] + audio<Scalar>(s, 2 * i + c) * mix);
            }
        }
    }
    if constexpr (op == Op::Lowpass || op == Op::Highpass || op == Op::Bandpass)
    {
        next[0] = audio<Scalar>(s);
        next[1] = s.localState[0];
        next[2] = safe<Scalar>(biquadValue<Scalar>(s, op));
        next[3] = s.localState[2];
    }
    if constexpr (op == Op::Svf)
    {
        const auto step = alfa::dsp::detail::svfStep(
            static_cast<double>(audio<Scalar>(s)), std::tan(std::numbers::pi * p[0] * dt<Scalar>(s)),
            static_cast<double>(p[1]), static_cast<double>(s.localState[0]),
            static_cast<double>(s.localState[1]));
        next[0] = safe<Scalar>(step.ic1);
        next[1] = safe<Scalar>(step.ic2);
    }
    if constexpr (op == Op::Comb || op == Op::String || op == Op::Diffusion)
    {
        double x = audio<Scalar>(s);
        if constexpr (op == Op::String)
            x += 0.5 * p[0] * (static_cast<double>(s.localState[n - 1]) + s.localState[n - 2]);
        if constexpr (op == Op::Diffusion)
            x += p[0] * (-static_cast<double>(p[0]) * audio<Scalar>(s) + s.localState.back());
        next[0] = safe<Scalar>(x);
        for (std::size_t i = 1; i < n; ++i)
            next[i] = s.localState[i - 1];
    }
    if constexpr (op == Op::Resonator || op == Op::Modal)
    {
        for (std::size_t i = 0; i < n / 2; ++i)
        {
            next[2 * i] = safe<Scalar>(modalValue<Scalar>(s, i));
            next[2 * i + 1] = s.localState[2 * i];
        }
    }
    if constexpr (op == Op::Adsr)
    {
        double level = s.localState[0];
        int stage = static_cast<int>(s.localState[1]);
        double releaseStart = s.localState[3];
        if (p[0] > 0 && s.localState[2] <= 0)
            stage = 1;
        if (p[0] <= 0 && s.localState[2] > 0)
        {
            stage = 4;
            releaseStart = level;
        }
        if (stage == 1)
        {
            level = p[1] == 0 ? 1 : std::min(1.0, level + dt<Scalar>(s) / p[1]);
            if (level >= 1)
                stage = 2;
        }
        else if (stage == 2)
        {
            level = p[2] == 0
                        ? p[3]
                        : std::max(static_cast<double>(p[3]), level - (1 - p[3]) * dt<Scalar>(s) / p[2]);
            if (level <= p[3])
                stage = 3;
        }
        else if (stage == 3)
            level = p[3];
        else if (stage == 4)
        {
            level = p[4] == 0 ? 0 : std::max(0.0, level - releaseStart * dt<Scalar>(s) / p[4]);
            if (level <= 0)
                stage = 0;
        }
        next[0] = safe<Scalar>(level);
        next[1] = static_cast<Scalar>(stage);
        next[2] = p[0];
        next[3] = safe<Scalar>(releaseStart);
    }
    if constexpr (op == Op::Multistage)
    {
        if (p[0] <= 0)
        {
            zero<Scalar>(next);
            return;
        }
        const double elapsed = std::max(0.0, static_cast<double>(s.localState[0])) + dt<Scalar>(s);
        double end = 0, start = 0, from = 0, level = table.back();
        for (std::size_t i = 0; i < table.size(); i += 2)
        {
            end += table[i];
            if (elapsed <= end)
            {
                level = from + (table[i + 1] - from) * (elapsed - start) / table[i];
                break;
            }
            start = end;
            from = table[i + 1];
        }
        next[0] = safe<Scalar>(elapsed);
        next[1] = safe<Scalar>(level);
    }
    if constexpr (op == Op::GestureFollower)
    {
        const double target = std::abs(audio<Scalar>(s));
        const double time = target > s.localState[0] ? p[0] : p[1];
        const double alpha = time == 0 ? 1 : -std::expm1(-dt<Scalar>(s) / time);
        next[0] = safe<Scalar>(s.localState[0] + alpha * (target - s.localState[0]));
    }
    if constexpr (op == Op::LatentAffine || op == Op::Matrix)
    {
        const auto input = op == Op::LatentAffine ? s.inputs.latentSound.values : s.inputs.audio.values;
        for (std::size_t row = 0; row < n; ++row)
        {
            double y = 0;
            for (std::size_t col = 0; col < input.size(); ++col)
                y += static_cast<double>(p[row * input.size() + col]) * input[col];
            next[row] = safe<Scalar>(y);
        }
    }
    if constexpr (op == Op::NonlinearFeedback)
    {
        next[0] = safe<Scalar>(std::tanh(static_cast<double>(p[0]) * audio<Scalar>(s) +
                                         static_cast<double>(p[1]) * s.localState[0]));
    }
    if constexpr (op == Op::ModDelay)
    {
        const double delayed = delayValue<Scalar>(s);
        next[0] = safe<Scalar>(wrap(s.localState[0] + p[2] * dt<Scalar>(s)));
        if (next[0] >= 1)
            next[0] = 0.0f;
        next[1] = safe<Scalar>(audio<Scalar>(s) + p[4] * delayed);
        for (std::size_t i = 2; i < n; ++i)
            next[i] = s.localState[i - 1];
    }
    if constexpr (op == Op::Phaser)
    {
        next[0] = audio<Scalar>(s);
        next[1] = safe<Scalar>(-static_cast<double>(p[0]) * audio<Scalar>(s) + s.localState[0] +
                               static_cast<double>(p[0]) * s.localState[1]);
    }
    if constexpr (op == Op::Reverb)
    {
        double total = 0;
        std::size_t offset = 0;
        for (Scalar value : table)
        {
            offset += static_cast<std::size_t>(value);
            total += s.localState[offset - 1];
        }
        const double mean = total / static_cast<double>(table.size()),
                     norm = std::sqrt(static_cast<double>(table.size()));
        offset = 0;
        for (Scalar value : table)
        {
            const auto length = static_cast<std::size_t>(value);
            next[offset] =
                safe<Scalar>(audio<Scalar>(s) / norm + p[0] * (s.localState[offset + length - 1] - 2 * mean));
            for (std::size_t i = 1; i < length; ++i)
                next[offset + i] = s.localState[offset + i - 1];
            offset += length;
        }
    }
    if constexpr (op == Op::Convolution)
    {
        if (n)
        {
            next[0] = audio<Scalar>(s);
            for (std::size_t i = 1; i < n; ++i)
                next[i] = s.localState[i - 1];
        }
    }
    if constexpr (op == Op::PhaseVocoder)
    {
        const auto bins = (n - 1) / 4;
        zero<Scalar>(next.subspan(1 + 2 * bins));
        for (std::size_t k = 0; k < bins; ++k)
        {
            const double real = audio<Scalar>(s, 2 * k), imag = audio<Scalar>(s, 2 * k + 1),
                         phase = std::atan2(imag, real);
            const double signedBin =
                k <= bins / 2 ? static_cast<double>(k) : static_cast<double>(k) - static_cast<double>(bins);
            const double expected = tau * signedBin * p[0] / static_cast<double>(bins);
            const double deviation = std::remainder(phase - s.localState[1 + k] - expected, tau);
            const double advance = (expected + deviation) * p[1] / p[0] * p[2];
            const double synthesis =
                s.localState[0] <= 0 ? phase : std::remainder(s.localState[1 + bins + k] + advance, tau);
            next[1 + k] = safe<Scalar>(phase);
            next[1 + bins + k] = safe<Scalar>(synthesis);
            const double mapped = signedBin * p[2];
            if (std::abs(mapped) > static_cast<double>(bins) / 2)
                continue;
            const double magnitude = std::hypot(real, imag);
            const double position = mapped < 0 ? mapped + static_cast<double>(bins) : mapped;
            const auto first = static_cast<std::size_t>(position) % bins, second = (first + 1) % bins;
            const double mix = position - std::floor(position);
            for (std::size_t c = 0; c < 2; ++c)
            {
                const double component = magnitude * (c == 0 ? std::cos(synthesis) : std::sin(synthesis));
                const auto offset = 1 + 2 * bins;
                next[offset + 2 * first + c] =
                    safe<Scalar>(next[offset + 2 * first + c] + (1 - mix) * component);
                next[offset + 2 * second + c] = safe<Scalar>(next[offset + 2 * second + c] + mix * component);
            }
        }
        next[0] = 1.0f;
    }
    if constexpr (op == Op::Formant)
    {
        const auto bins = n / 2;
        for (std::size_t k = 0; k < bins; ++k)
        {
            const double signedBin =
                k <= bins / 2 ? static_cast<double>(k) : static_cast<double>(k) - static_cast<double>(bins);
            const double source = signedBin / p[0];
            if (std::abs(source) > static_cast<double>(bins) / 2)
            {
                next[2 * k] = 0.0f;
                next[2 * k + 1] = 0.0f;
                continue;
            }
            const double position = source < 0 ? source + static_cast<double>(bins) : source;
            const auto first = static_cast<std::size_t>(position) % bins, second = (first + 1) % bins;
            const double mix = position - std::floor(position);
            const double envelope = (1 - mix) * table[first] + mix * table[second],
                         gain = table[k] / envelope;
            next[2 * k] = safe<Scalar>(audio<Scalar>(s, 2 * k) * gain);
            next[2 * k + 1] = safe<Scalar>(audio<Scalar>(s, 2 * k + 1) * gain);
        }
    }
    if constexpr (op == Op::Binaural || op == Op::Room)
    {
        if (n)
        {
            next[0] = audio<Scalar>(s);
            for (std::size_t i = 1; i < n; ++i)
                next[i] = s.localState[i - 1];
        }
    }
    if constexpr (op == Op::GrainStretch)
    {
        for (std::size_t i = 2; i < n; i += 3)
            if (s.localState[i] > 0)
            {
                next[i + 1] = safe<Scalar>(s.localState[i + 1] + dt<Scalar>(s));
                next[i + 2] = safe<Scalar>(
                    wrap(s.localState[i + 2] + p[1] * dt<Scalar>(s) / static_cast<double>(table.size())));
                if (next[i + 1] >= p[2])
                    next[i] = 0.0f;
            }
        const double launch = s.localState[0] + dt<Scalar>(s) / p[3];
        next[0] = safe<Scalar>(wrap(launch));
        next[1] =
            safe<Scalar>(wrap(s.localState[1] + p[0] * dt<Scalar>(s) / static_cast<double>(table.size())));
        if (next[1] >= 1)
            next[1] = 0.0f;
        if (launch >= 1)
            for (std::size_t i = 2; i < n; i += 3)
                if (next[i] <= 0)
                {
                    next[i] = 1.0f;
                    next[i + 1] = 0.0f;
                    next[i + 2] = s.localState[1];
                    break;
                }
    }
}

template <Op op, typename Scalar>
void readout(const StepT<Scalar>& s, std::span<Scalar> out) noexcept
{
    zero<Scalar>(out);
    if (!valid<Scalar>(s, op))
        return;
    const auto p = s.parameters;
    const auto n = s.localState.size();
    const auto table = asset<Scalar>(s, 0);
    if constexpr (op == Op::Fft || op == Op::Ifft || op == Op::SpectralMask || op == Op::SpectralMorph ||
                  op == Op::SpectralTilt || op == Op::SpectralFreeze || op == Op::Redistribution ||
                  op == Op::LatentAffine || op == Op::Matrix)
    {
        if (out.size() == n)
            for (std::size_t i = 0; i < n; ++i)
                out[i] = safe<Scalar>(static_cast<double>(s.localState[i]) * s.activation);
        return;
    }
    else if constexpr (op == Op::StereoWidth)
    {
        if (out.size() != 2)
            return;
        const double mid = (static_cast<double>(audio<Scalar>(s)) + audio<Scalar>(s, 1)) / 2,
                     side = (static_cast<double>(audio<Scalar>(s)) - audio<Scalar>(s, 1)) / 2;
        out[0] = safe<Scalar>((mid + p[0] * side) * s.activation);
        out[1] = safe<Scalar>((mid - p[0] * side) * s.activation);
        return;
    }
    else if constexpr (op == Op::Pan)
    {
        if (out.size() != 2)
            return;
        const double angle = (p[0] + 1.0) * std::numbers::pi / 4;
        out[0] = safe<Scalar>(audio<Scalar>(s) * std::cos(angle) * s.activation);
        out[1] = safe<Scalar>(audio<Scalar>(s) * std::sin(angle) * s.activation);
        return;
    }
    else if constexpr (op == Op::Svf)
    {
        if (out.size() != 3)
            return;
        const auto step = alfa::dsp::detail::svfStep(
            static_cast<double>(audio<Scalar>(s)), std::tan(std::numbers::pi * p[0] * dt<Scalar>(s)),
            static_cast<double>(p[1]), static_cast<double>(s.localState[0]),
            static_cast<double>(s.localState[1]));
        out[0] = safe<Scalar>(step.low * s.activation);
        out[1] = safe<Scalar>(step.band * s.activation);
        out[2] = safe<Scalar>((audio<Scalar>(s) - p[1] * step.band - step.low) * s.activation);
        return;
    }
    else if constexpr (op == Op::PhaseVocoder)
    {
        const auto bins = (n - 1) / 4;
        if (out.size() != 2 * bins)
            return;
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = safe<Scalar>(static_cast<double>(s.localState[1 + 2 * bins + i]) * s.activation);
        return;
    }
    else if constexpr (op == Op::Formant)
    {
        if (out.size() != n)
            return;
        for (std::size_t i = 0; i < n; ++i)
            out[i] = safe<Scalar>(static_cast<double>(s.localState[i]) * s.activation);
        return;
    }
    else if constexpr (op == Op::Binaural)
    {
        if (out.size() != 2)
            return;
        for (std::size_t channel = 0; channel < 2; ++channel)
        {
            const auto ir = asset<Scalar>(s, channel);
            double value = static_cast<double>(audio<Scalar>(s)) * ir[0];
            for (std::size_t i = 1; i < ir.size(); ++i)
                value += static_cast<double>(s.localState[i - 1]) * ir[i];
            out[channel] = safe<Scalar>(value * s.activation);
        }
        return;
    }
    else
    {
        double y = 0;
        if (out.size() != 1)
            return;
        if constexpr (op == Op::Saw)
        {
            double phase = s.localState[0], d = std::abs(p[0]) * dt<Scalar>(s), blep = 0;
            if (d > 0)
            {
                if (phase < d)
                {
                    const double x = phase / d;
                    blep = x + x - x * x - 1;
                }
                else if (phase > 1 - d)
                {
                    const double x = (phase - 1) / d;
                    blep = x * x + x + x + 1;
                }
            }
            y = p[1] * (2 * phase - 1 - blep);
        }
        if constexpr (op == Op::Wavetable)
        {
            y = static_cast<double>(interpolate<Scalar>(table, s.localState[0], true)) * p[1];
        }
        if constexpr (op == Op::Additive)
        {
            for (std::size_t i = 0; i < n; ++i)
                y += p[2 * i + 1] * std::sin(tau * s.localState[i]);
        }
        if constexpr (op == Op::Noise)
        {
            const auto seed = static_cast<std::uint32_t>(s.localState[0]) |
                              (static_cast<std::uint32_t>(s.localState[1]) << 16);
            y = static_cast<double>(seed) / 2147483648.0 - 1;
        }
        if constexpr (op == Op::Impulse)
        {
            y = p[0] > 0 && s.localState[0] <= 0 ? p[1] : 0;
        }
        if constexpr (op == Op::Sample || op == Op::Resample || op == Op::ResampleCubic ||
                      op == Op::ResampleSinc8)
        {
            y = static_cast<double>(op == Op::ResampleCubic || op == Op::ResampleSinc8
                                        ? interpolatedAsset<Scalar>(table, s.localState[0], op)
                                        : interpolate<Scalar>(table, s.localState[0], op == Op::Sample)) *
                p[1];
        }
        if constexpr (op == Op::Grain)
        {
            for (std::size_t i = 1; i < n; i += 3)
                if (s.localState[i] > 0 && s.localState[i + 1] >= 0 && s.localState[i + 1] < p[1])
                    y += 0.5 * (1 - std::cos(tau * s.localState[i + 1] / p[1])) *
                         interpolate<Scalar>(table, s.localState[i + 2], true) * p[3];
        }
        if constexpr (op == Op::Lowpass || op == Op::Highpass || op == Op::Bandpass)
            y = biquadValue<Scalar>(s, op);
        if constexpr (op == Op::Comb)
            y = audio<Scalar>(s) + static_cast<double>(p[0]) * s.localState.back();
        if constexpr (op == Op::String)
            y = s.localState.back();
        if constexpr (op == Op::Diffusion)
            y = -static_cast<double>(p[0]) * audio<Scalar>(s) + s.localState.back();
        if constexpr (op == Op::Resonator || op == Op::Modal)
        {
            for (std::size_t i = 0; i < n / 2; ++i)
                y += modalValue<Scalar>(s, i);
        }
        if constexpr (op == Op::Adsr || op == Op::GestureFollower || op == Op::NonlinearFeedback)
            y = s.localState[0];
        if constexpr (op == Op::Multistage)
            y = s.localState[1];
        if constexpr (op == Op::MidiAffine)
        {
            const double control = s.inputs.midi.values.empty() ? 0 : s.inputs.midi.values[0];
            y = control * p[0] + p[1];
        }
        if constexpr (op == Op::Saturation)
            y = std::tanh(static_cast<double>(p[0]) * audio<Scalar>(s));
        if constexpr (op == Op::Waveshape)
        {
            const double x = std::clamp(static_cast<double>(p[0]) * audio<Scalar>(s), -1.0, 1.0);
            const double index = (x + 1) * 0.5 * static_cast<double>(table.size() - 1);
            const auto i = static_cast<std::size_t>(index), j = std::min(i + 1, table.size() - 1);
            y = table[i] + (index - static_cast<double>(i)) * (static_cast<double>(table[j]) - table[i]);
        }
        if constexpr (op == Op::Clip)
            y = std::clamp(audio<Scalar>(s), -p[0], p[0]);
        if constexpr (op == Op::Cubic)
        {
            const double x = std::clamp(static_cast<double>(p[0]) * audio<Scalar>(s), -1.0, 1.0);
            y = 1.5 * x - 0.5 * x * x * x;
        }
        if constexpr (op == Op::ModDelay)
            y = (1 - static_cast<double>(p[3])) * audio<Scalar>(s) + p[3] * delayValue<Scalar>(s);
        if constexpr (op == Op::Phaser)
        {
            const double wet = -static_cast<double>(p[0]) * audio<Scalar>(s) + s.localState[0] +
                               static_cast<double>(p[0]) * s.localState[1];
            y = (1 - static_cast<double>(p[1])) * audio<Scalar>(s) + p[1] * wet;
        }
        if constexpr (op == Op::Reverb)
        {
            std::size_t offset = 0;
            for (Scalar length : table)
            {
                offset += static_cast<std::size_t>(length);
                y += s.localState[offset - 1];
            }
            y /= std::sqrt(static_cast<double>(table.size()));
        }
        if constexpr (op == Op::Convolution)
        {
            y = static_cast<double>(audio<Scalar>(s)) * table[0];
            for (std::size_t i = 0; i < n; ++i)
                y += static_cast<double>(s.localState[i]) * table[i + 1];
        }
        if constexpr (op == Op::Room)
        {
            for (std::size_t path = 0; path < 7; ++path)
            {
                const double distance = roomDistance<Scalar>(s, path), gain = path == 0 ? 1.0 : p[9 + path];
                y += gain * pastInput<Scalar>(s, distance / (p[9] * dt<Scalar>(s))) / distance;
            }
        }
        if constexpr (op == Op::GrainStretch)
        {
            for (std::size_t i = 2; i < n; i += 3)
                if (s.localState[i] > 0 && s.localState[i + 1] >= 0 && s.localState[i + 1] < p[2])
                    y += 0.5 * (1 - std::cos(tau * s.localState[i + 1] / p[2])) *
                         interpolate<Scalar>(table, s.localState[i + 2], true) * p[4];
        }
        out[0] = safe<Scalar>(y * s.activation);
    }
}

template <Op op, typename Scalar>
PrimitiveT<Scalar> descriptor(std::size_t states, std::size_t outputs, std::string_view id)
{
    return {std::string(id), states, outputs, transition<op, Scalar>, readout<op, Scalar>};
}
} // namespace

std::span<const kernels::Metadata> catalog() noexcept
{
    // Metadata is prepared off thread; callbacks never enumerate or allocate it.
    static const auto combined = []
    {
        std::vector<kernels::Metadata> result(std::begin(records), std::end(records));
        const auto added = production::catalog();
        result.insert(result.end(), added.begin(), added.end());
        const auto spectral = spectralProduction::catalog();
        result.insert(result.end(), spectral.begin(), spectral.end());
        const auto nonlinear = nonlinear::catalog();
        result.insert(result.end(), nonlinear.begin(), nonlinear.end());
        return result;
    }();
    return combined;
}
template <typename Scalar>
bool createPrimitive(std::string_view id, std::size_t n, PrimitiveT<Scalar>& result, std::string& error)
{
    for (const auto& metadata : production::catalog())
        if (metadata.id == id)
            return production::createPrimitive(id, n, result, error);
    for (const auto& metadata : spectralProduction::catalog())
        if (metadata.id == id)
            return spectralProduction::createPrimitive(id, n, result, error);
    for (const auto& metadata : nonlinear::catalog())
        if (metadata.id == id)
            return nonlinear::createPrimitive(id, n, result, error);
    error.clear();
    const auto fail = [&](const char* text)
    {
        error = text;
        return false;
    };
    const auto fixed = [&](std::size_t required)
    {
        return n == required;
    };
    // Each dispatch has an actual mathematical transition; unsupported IDs fail.
#define FIXED(ID, OP, N, OUT)                                                                                \
    if (id == ID)                                                                                            \
    {                                                                                                        \
        if (!fixed(N))                                                                                       \
            return fail("Incorrect intrinsic state_size for primitive");                                     \
        result = descriptor<Op::OP, Scalar>(n, OUT, id);                                                     \
        return true;                                                                                         \
    }
#define VARIABLE(ID, OP, VALID, OUT)                                                                         \
    if (id == ID)                                                                                            \
    {                                                                                                        \
        if (!(VALID))                                                                                        \
            return fail("Invalid state_size for primitive");                                                 \
        result = descriptor<Op::OP, Scalar>(n, OUT, id);                                                     \
        return true;                                                                                         \
    }
    FIXED("osc.polyblep_saw", Saw, 1, 1)
    FIXED("osc.wavetable", Wavetable, 1, 1)
    VARIABLE("osc.additive", Additive, n > 0 && n <= std::numeric_limits<std::size_t>::max() / 2, 1)
    FIXED("excitation.white_noise", Noise, 2, 1)
    FIXED("excitation.impulse", Impulse, 1, 1)
    FIXED("excitation.sample_loop", Sample, 1, 1)
    VARIABLE("excitation.granular", Grain, n >= 4 && (n - 1) % 3 == 0, 1)
    VARIABLE("spectral.fft", Fft, n >= 2 && n % 2 == 0 && powerOfTwo(n / 2), n)
    VARIABLE("spectral.ifft", Ifft, n >= 2 && n % 2 == 0 && powerOfTwo(n / 2), n)
    VARIABLE("spectral.mask", SpectralMask, n >= 2 && n % 2 == 0, n)
    VARIABLE("spectral.morph", SpectralMorph,
             n >= 2 && n % 2 == 0 && n <= std::numeric_limits<std::size_t>::max() / 2, n)
    VARIABLE("spectral.tilt", SpectralTilt, n >= 2 && n % 2 == 0, n)
    VARIABLE("spectral.freeze", SpectralFreeze, n >= 2 && n % 2 == 0, n)
    VARIABLE("spectral.redistribute", Redistribution, n >= 2 && n % 2 == 0, n)
    FIXED("filter.biquad_lowpass", Lowpass, 4, 1)
    FIXED("filter.biquad_highpass", Highpass, 4, 1)
    FIXED("filter.biquad_bandpass", Bandpass, 4, 1)
    FIXED("filter.state_variable", Svf, 2, 3)
    VARIABLE("filter.feedforward_comb", Comb, n > 0, 1)
    FIXED("physical.resonator", Resonator, 2, 1)
    VARIABLE("physical.modal_bank", Modal,
             n > 0 && n % 2 == 0 && n / 2 <= std::numeric_limits<std::size_t>::max() / 3, 1)
    VARIABLE("physical.string_waveguide", String, n >= 2, 1)
    FIXED("control.adsr", Adsr, 4, 1)
    FIXED("control.multistage_envelope", Multistage, 2, 1)
    FIXED("control.midi_affine", MidiAffine, 0, 1)
    FIXED("control.gesture_follower", GestureFollower, 1, 1)
    VARIABLE("control.latent_affine", LatentAffine, n > 0, n)
    FIXED("nonlinear.tanh", Saturation, 0, 1)
    FIXED("nonlinear.waveshape_table", Waveshape, 0, 1)
    FIXED("nonlinear.clip", Clip, 0, 1)
    FIXED("nonlinear.cubic_softclip", Cubic, 0, 1)
    FIXED("nonlinear.feedback_tanh", NonlinearFeedback, 1, 1)
    VARIABLE("time.modulated_delay", ModDelay, n >= 3, 1)
    FIXED("time.phaser_one_stage", Phaser, 2, 1)
    VARIABLE("time.diffusion_allpass", Diffusion, n > 0, 1)
    VARIABLE("time.fdn_reverb", Reverb, n > 0, 1)
    VARIABLE("time.convolution_fir", Convolution, true, 1)
    FIXED("spatial.stereo_width", StereoWidth, 0, 2)
    FIXED("spatial.constant_power_pan", Pan, 0, 2)
    VARIABLE("spatial.matrix", Matrix, n > 0, n)
    FIXED("pitchtime.asset_resample", Resample, 1, 1)
    FIXED("pitchtime.asset_resample_cubic", ResampleCubic, 1, 1)
    if (id == "pitchtime.asset_resample_sinc8")
    {
        (void)alfa::dsp::detail::typedSincTable<Scalar>(); // immutable preparation before callbacks
        if (n != 1)
            return fail("State size must be one");
        result = descriptor<Op::ResampleSinc8, Scalar>(n, 1, id);
        return true;
    }
    VARIABLE("pitchtime.phase_vocoder", PhaseVocoder, n >= 5 && (n - 1) % 4 == 0, (n - 1) / 2)
    VARIABLE("pitchtime.formant_compensation", Formant, n >= 2 && n % 2 == 0, n)
    VARIABLE("spatial.binaural_hrtf_fir", Binaural, true, 2)
    VARIABLE("spatial.room_first_order", Room, n > 0, 1)
    VARIABLE("pitchtime.granular_stretch", GrainStretch, n >= 5 && (n - 2) % 3 == 0, 1)
#undef FIXED
#undef VARIABLE
    return fail("Unknown primitive; no placeholder implementation is substituted");
}
template <typename Scalar>
FactoryT<Scalar>::FactoryT()
{
    entries_.reserve(catalog().size());
    for (const auto& metadata : catalog())
    {
        const auto id = metadata.id;
        entries_.push_back({metadata, [id](std::size_t n, PrimitiveT<Scalar>& p, std::string& error)
                            { return createPrimitive(id, n, p, error); }});
    }
}
template <typename Scalar>
bool FactoryT<Scalar>::add(kernels::Metadata metadata, Creator creator, std::string& error)
{
    error.clear();
    if (metadata.id.empty() || !creator)
    {
        error = "Extension requires identity and creator";
        return false;
    }
    if (std::any_of(entries_.begin(), entries_.end(),
                    [&](const Entry& e) { return e.metadata.id == metadata.id; }))
    {
        error = "Primitive identity already registered";
        return false;
    }
    entries_.push_back({metadata, std::move(creator)});
    return true;
}
template <typename Scalar>
bool FactoryT<Scalar>::create(std::string_view id, std::size_t n, PrimitiveT<Scalar>& p,
                              std::string& error) const
{
    const auto it =
        std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.metadata.id == id; });
    if (it == entries_.end())
    {
        error = "Unknown primitive";
        return false;
    }
    return it->creator(n, p, error);
}
template bool createPrimitive<float>(std::string_view, std::size_t, PrimitiveT<float>&, std::string&);
template bool createPrimitive<double>(std::string_view, std::size_t, PrimitiveT<double>&, std::string&);
template class FactoryT<float>;
template class FactoryT<double>;
} // namespace alfa::tensor::universe
