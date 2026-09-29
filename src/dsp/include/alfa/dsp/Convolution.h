#pragma once

// Partitioned FIR convolution and the CONVOLUTION processing family (Family::Convolution,
// instance slots).
//
// Building blocks, smallest first:
//   DirectFir                    time-domain FIR up to 64 taps (no allocation at all)
//   UniformPartitionedConvolver  uniformly partitioned overlap-save FFT convolution with a
//                                frequency-domain delay line; latency = one partition
//   PreparedIr                   an IR resampled, normalised and cut into partition spectra,
//                                immutable and shared; built off the audio thread
//   PartitionedConvolver         zero-latency non-uniform convolver running a PreparedIr on one
//                                or two channels (mono, stereo or true-stereo IRs)
//   IrBank                       owner-side list of up to 16 PreparedIrs, published to the
//                                processor by an atomic pointer switch
//   ConvolutionProcessor         the slot processor
//   makeSyntheticIr              offline generator of decaying-noise, room, plate and spring IRs
//
// Non-uniform partitioning (PartitionedConvolver). The IR is split into
//   [0, 64)          direct FIR (the "head"), no latency
//   [64, 1024)       64-sample partitions, FFT 128, computed at each 64-sample boundary
//   [1024, 8192)     512-sample partitions, FFT 1024
//   [8192, end)      4096-sample partitions, FFT 8192
// Each stage starts at least one of its own blocks into the IR, so its overlap-save latency is
// absorbed and the whole convolver has zero latency. The two large stages start two blocks in,
// which leaves one block of slack: their frequency-domain multiply-adds are spread evenly over
// the block instead of landing on one callback, so the worst-case callback costs little more
// than the average (only the FFTs remain at the boundaries). Cost grows with IR length L as about
// L / 4096 complex multiply-adds per sample per path, against L / 128 for 128-sample uniform
// partitions. Measured (Release, x86-64, 48 kHz, 128-sample blocks, stereo IR, whole processor):
//   2 s IR: 16 us per block (0.6 % of the block), 5 s IR: 23 us (0.9 %);
//   two UniformPartitionedConvolvers(128) alone: 180 us (6.8 %) and 363 us (13.6 %).
// So uniform partitions are fine for IRs up to a second or two; the non-uniform scheme is what
// makes long IRs cheap (benchmark: ConvolutionTests.cpp, tag [.bench]).
//
// Threading and lifetime. Everything that allocates (PreparedIr::create, IrBank edits,
// IrBank::publish, ConvolutionProcessor::prepare) runs off the audio thread. The audio thread
// only ever sees an `IrBankView` (a generation number and up to 16 raw PreparedIr pointers)
// loaded from an atomic. It acknowledges a generation once it no longer touches anything from
// older views, and IrBank::collect frees only views older than that acknowledgement, so the
// owner keeps old IRs alive exactly as long as the audio thread may still use them. The IrBank
// must outlive the processor's use of it: destroy the processor first, or publish an empty bank
// and wait for its acknowledgement.

#include "alfa/dsp/Processor.h"
#include "alfa/dsp/detail/ConvspatialUtil.h"
#include "alfa/dsp/detail/Fft.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace alfa::dsp
{

// Direct FIR -----------------------------------------------------------------------------------

/// Input history for direct FIRs of up to kMaxTaps taps, processed in chunks of at most kChunk.
/// Shared by DirectFir and the convolver head (one history per input, one kernel per path).
class FirHistory
{
public:
    static constexpr int kMaxTaps = 64;
    static constexpr int kChunk = 64;

    void reset() noexcept { buf_.fill(0.0f); }

    /// Appends n <= kChunk input samples after the history.
    void load(const float* in, int n) noexcept;
    /// out[i] (+)= sum_k h[k] x[t - k] for the loaded chunk, with `reversed` the kernel stored
    /// back to front and zero-padded to kMaxTaps (reversed[kMaxTaps - 1] = h[0]).
    void apply(const float* reversed, float* out, int n, bool accumulate) const noexcept;
    /// Keeps the last kMaxTaps - 1 samples as history for the next chunk.
    void advance(int n) noexcept;

private:
    alignas(64) std::array<float, kMaxTaps - 1 + kChunk> buf_{};
};

/// Time-domain FIR for short kernels (1..64 taps). The dot product runs over a contiguous,
/// reversed kernel so it vectorises. No allocation anywhere, so setKernel is real-time safe.
class DirectFir
{
public:
    static constexpr int kMaxTaps = FirHistory::kMaxTaps;

    /// Copies up to kMaxTaps taps; extra taps are ignored.
    void setKernel(const float* h, int taps) noexcept;
    void reset() noexcept { history_.reset(); }
    /// Any n; in and out may alias.
    void process(const float* in, float* out, int n) noexcept;
    [[nodiscard]] int taps() const noexcept { return taps_; }

private:
    FirHistory history_;
    alignas(64) std::array<float, kMaxTaps> reversed_{};
    int taps_ = 0;
};

// Partition stage ------------------------------------------------------------------------------

namespace conv
{
inline constexpr int kHeadTaps = FirHistory::kMaxTaps;
inline constexpr int kStages = 3;
inline constexpr std::array<int, kStages> kStageBlock{64, 512, 4096};
/// IR sample where each stage starts; a stage ends where the next starts (the last never ends).
inline constexpr std::array<int, kStages> kStageStart{64, 1024, 8192};
inline constexpr int kMaxPaths = 4;

/// One input -> output path through a stage: which input FDL, which output accumulator, and
/// the path's partition spectra (partitions x 2 block floats, split re | im, prescaled 1/N).
struct Route
{
    int in = 0;
    int out = 0;
    const float* spectra = nullptr;
};

struct StageFilter
{
    std::array<Route, kMaxPaths> routes{};
    int routeCount = 0;
    int partitions = 0;
    const float* gains = nullptr; ///< per-partition gain, nullptr = all 1
};

/// Overlap-save engine for one partition size: the input frequency-domain delay line (one per
/// input channel) and the output accumulators (one per output channel). Partition j of a route
/// is delayed by (firstDelay + j) blocks, so with firstDelay >= 1 the stage adds no latency of
/// its own. With firstDelay >= 2 the multiply-adds for the next block are spread over the
/// current one.
class PartitionStage
{
public:
    /// Allocates.
    void prepare(int block, int firstDelay, int maxPartitions, int inputs, int outputs);
    /// Clears the time-domain state; the delay line is invalidated lazily (O(block), not O(IR)).
    void reset() noexcept;
    /// n <= remaining(). Adds this stage's output to out[]. in[] and out[] must not alias.
    void process(const float* const* in, float* const* out, int n, const StageFilter& filter) noexcept;
    [[nodiscard]] int remaining() const noexcept { return block_ - pos_; }
    [[nodiscard]] int block() const noexcept { return block_; }
    [[nodiscard]] int maxPartitions() const noexcept { return maxPartitions_; }

private:
    void boundary(const StageFilter& filter) noexcept;
    void macUpTo(int target, const StageFilter& filter) noexcept;
    [[nodiscard]] int usable(const StageFilter& filter) const noexcept;

    detail::RealFft fft_;
    int block_ = 0;
    int firstDelay_ = 1;
    int lag_ = 0;
    int maxPartitions_ = 0;
    int ring_ = 1;
    int inputs_ = 0;
    int outputs_ = 0;
    std::array<detail::AlignedBuffer, 2> time_; // 2 blocks per input: previous | current
    std::array<detail::AlignedBuffer, 2> fdl_;  // ring_ spectra per input, split re | im
    std::array<detail::AlignedBuffer, 2> acc_;  // accumulator per output, split re | im
    std::array<detail::AlignedBuffer, 2> out_;  // the block being emitted, per output
    detail::AlignedBuffer scratch_;
    int pos_ = 0;
    int newest_ = -1;       // ring slot of the newest input spectrum
    long long spectra_ = 0; // input spectra computed since reset
    int macDone_ = 0;
};
} // namespace conv

/// Uniformly partitioned overlap-save convolution (one channel) with a frequency-domain delay
/// line. Output = input * IR delayed by latencySamples() = the partition size; exact against
/// direct convolution to float rounding. Use PartitionedConvolver when latency matters.
class UniformPartitionedConvolver
{
public:
    /// Allocates. partitionSize is a power of two >= kControlBlock (64 or 128 are typical).
    void prepare(const float* ir, int length, int partitionSize);
    void reset() noexcept { stage_.reset(); }
    /// Any n; in and out may alias.
    void process(const float* in, float* out, int n) noexcept;
    [[nodiscard]] int latencySamples() const noexcept { return stage_.block(); }

private:
    conv::PartitionStage stage_;
    conv::StageFilter filter_;
    detail::AlignedBuffer spectra_;
    std::array<float, 64> scratchIn_{};
    std::array<float, 64> scratchOut_{};
};

// Prepared IRs ---------------------------------------------------------------------------------

/// An impulse response as loaded (from a file or makeSyntheticIr). channels: 1 (mono, applied
/// to every channel), 2 (stereo: channel c filtered by IR c) or 4 (true stereo, in -> out:
/// LL, LR, RL, RR).
struct IrSource
{
    std::vector<std::vector<float>> channels;
    double sampleRate = 48000.0;
    std::string name;
};

struct IrPrepareOptions
{
    /// Scale to unit energy (the louder output of a stereo / true-stereo IR), so white noise
    /// keeps its level. A unit impulse is left unchanged.
    bool normalise = true;
    float gainDb = 0.0f;      ///< applied after normalisation
    double maxSeconds = 10.0; ///< longer IRs are truncated (with a 10 ms fade)
};

/// Resamples an IR offline with a Blackman-windowed sinc (32 zero crossings, band-limited to the
/// lower of the two Nyquists). Allocates. Identity when the rates are equal.
[[nodiscard]] std::vector<float> resampleIr(const std::vector<float>& ir, double fromRate, double toRate);

/// Immutable, shareable IR ready for PartitionedConvolver at one host sample rate.
class PreparedIr
{
public:
    /// Allocates; call off the audio thread. Throws std::invalid_argument for an empty source,
    /// a channel count other than 1, 2, 4, or non-positive rates.
    [[nodiscard]] static std::shared_ptr<const PreparedIr> create(const IrSource& source, double hostRate,
                                                                  const IrPrepareOptions& options = {});

    [[nodiscard]] double sampleRate() const noexcept { return sampleRate_; }
    [[nodiscard]] int paths() const noexcept { return paths_; }
    [[nodiscard]] int length() const noexcept { return length_; }
    [[nodiscard]] float normalisationGain() const noexcept { return gain_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// Head kernel of a path: kHeadTaps floats, reversed (see FirHistory::apply).
    [[nodiscard]] const float* head(int path) const noexcept;
    [[nodiscard]] int partitions(int stage) const noexcept
    {
        return partitions_[static_cast<std::size_t>(stage)];
    }
    [[nodiscard]] const float* spectra(int stage, int path) const noexcept;

private:
    double sampleRate_ = 0.0;
    int paths_ = 0;
    int length_ = 0;
    float gain_ = 1.0f;
    std::string name_;
    detail::AlignedBuffer head_; // paths x kHeadTaps
    std::array<int, conv::kStages> partitions_{};
    std::array<detail::AlignedBuffer, conv::kStages> spectra_; // paths x partitions x 2 block
};

/// Zero-latency convolver running one PreparedIr on 1 or 2 channels (see the header comment for
/// the partitioning). setIr() is real-time safe: it swaps a pointer and clears state in
/// O(largest block). With one channel a stereo IR uses its first channel and a true-stereo IR its
/// LL path.
class PartitionedConvolver
{
public:
    /// Allocates for IRs up to maxIrLength samples (longer ones are truncated).
    void prepare(int channels, int maxIrLength);
    void reset() noexcept;
    /// Also resets. nullptr = silence.
    void setIr(const PreparedIr* ir) noexcept;
    [[nodiscard]] const PreparedIr* ir() const noexcept { return ir_; }
    /// Trims the IR: gain 1 up to keep - fade, a raised-cosine fade to 0 at keep (in samples).
    /// Applied per partition (64 samples early on, 4096 in the late tail), so it is a smooth
    /// staircase envelope; partitions beyond `keep` are skipped entirely and cost nothing.
    void setTrim(float keepSamples, float fadeSamples) noexcept;
    /// Overwrites out[c] with the convolution of in[c]. Any n. in and out must not alias.
    void process(const float* const* in, float* const* out, int n) noexcept;
    /// Samples of IR actually played (after truncation and trim).
    [[nodiscard]] int activeLength() const noexcept;

private:
    int channels_ = 1;
    int maxIrLength_ = 0;
    const PreparedIr* ir_ = nullptr;
    std::array<FirHistory, 2> head_;
    std::array<conv::PartitionStage, conv::kStages> stages_;
    std::array<conv::StageFilter, conv::kStages> filters_{};
    std::array<std::vector<float>, conv::kStages> gains_;
    std::array<const float*, conv::kMaxPaths> headKernel_{};
    std::array<conv::Route, conv::kMaxPaths> headRoutes_{};
    int headRouteCount_ = 0;
    float keep_ = -1.0f;
    float fade_ = 0.0f;
    bool trimmed_ = false;
};

// Bank -----------------------------------------------------------------------------------------

inline constexpr int kMaxBankIrs = 16;

/// What the audio thread sees of an IrBank: immutable once published.
struct IrBankView
{
    std::uint64_t generation = 0;
    int count = 0;
    std::array<const PreparedIr*, kMaxBankIrs> irs{};
};

class ConvolutionProcessor;

/// Owner-side IR list (not thread-safe; use it from one non-audio thread). Edits take effect on
/// the audio thread only at publish().
class IrBank
{
public:
    IrBank() = default;
    IrBank(const IrBank&) = delete;
    IrBank& operator=(const IrBank&) = delete;

    /// Prepares `source` at the current host rate and appends it; returns its index, or -1 when
    /// the bank is full. Allocates.
    int add(IrSource source, const IrPrepareOptions& options = {});
    /// Replaces entry `index` (0 <= index < size()).
    void replace(int index, IrSource source, const IrPrepareOptions& options = {});
    void remove(int index);
    void clear();
    /// Re-prepares every IR for a new host rate (call before or after the processor's prepare,
    /// then publish). IRs whose rate does not match the processor's are ignored (dry).
    void setHostRate(double hostRate);
    [[nodiscard]] double hostRate() const noexcept { return hostRate_; }
    [[nodiscard]] int size() const noexcept { return static_cast<int>(entries_.size()); }
    [[nodiscard]] std::shared_ptr<const PreparedIr> get(int index) const;

    /// Hands the current contents to the processor (an atomic pointer store). Allocates the view.
    void publish(ConvolutionProcessor& processor);
    /// Frees published views, and IRs referenced only by them, that the processor has
    /// acknowledged it no longer uses. With several processors, pass the smallest
    /// acknowledgedGeneration() to collectBefore.
    void collect(const ConvolutionProcessor& processor);
    void collectBefore(std::uint64_t generation);
    [[nodiscard]] std::size_t publishedViews() const noexcept { return published_.size(); }

private:
    struct Entry
    {
        IrSource source;
        IrPrepareOptions options;
        std::shared_ptr<const PreparedIr> prepared;
    };
    struct Published
    {
        std::unique_ptr<IrBankView> view;
        std::vector<std::shared_ptr<const PreparedIr>> keep;
    };
    std::vector<Entry> entries_;
    std::vector<Published> published_;
    double hostRate_ = 48000.0;
    std::uint64_t nextGeneration_ = 1;
};

// Synthetic IRs --------------------------------------------------------------------------------

enum class SyntheticIrKind : unsigned char
{
    Decay,  ///< exponentially decaying noise, high frequencies decaying faster
    Room,   ///< sparse early reflections, then a short, darker diffuse tail
    Plate,  ///< dense and bright from the first sample
    Spring, ///< dispersive chirp echoes every ~33 ms over a band-limited decay
};

struct SyntheticIrSpec
{
    SyntheticIrKind kind = SyntheticIrKind::Decay;
    double sampleRate = 48000.0;
    double rt60 = 1.0;    ///< seconds to decay by 60 dB
    double seconds = 0.0; ///< IR length; 0 = rt60 (clamped to 0.05..10 s)
    int channels = 2;     ///< 1 or 2 (decorrelated)
    std::uint32_t seed = 1;
};

/// Offline and allocating; deterministic for a given spec. The result is not normalised
/// (PreparedIr does that).
[[nodiscard]] IrSource makeSyntheticIr(const SyntheticIrSpec& spec);

// CONVOLUTION family ---------------------------------------------------------------------------

/// Convolution reverb / cabinet for instance slots. Signal flow:
///   input -> pre-delay -> two PartitionedConvolvers (old and new IR, crossfaded on a swap)
///   -> low-cut, high-cut -> width -> gain -> mix with the dry input.
/// With no bank published, or an empty one, or IRs prepared for another sample rate, it passes
/// the input through unchanged. Latency 0.
///
/// IR swap: when the selected IR changes, the idle convolver is reset (O(block) work), given
/// the new IR, and the output crossfades linearly from the old to the new over 20 ms. The new
/// IR starts from silence; the old one is cut after the fade. A further change during a fade
/// waits for it to finish. Nothing allocates.
///
/// Params (normalised 0..1, order is the contract):
///   0 ir            0..15 Choice   bank index; clamped to the bank size
///   1 predelay      0..250 ms      Linear
///   2 length        5..100 %       IR trim (early cut), Linear
///   3 fade          0..100 %       part of the kept length that fades out, Linear
///   4 low_cut       20..1000 Hz    12 dB/oct high-pass on the wet signal, Log
///   5 high_cut      1k..22k Hz     12 dB/oct low-pass on the wet signal, Log
///   6 width         0..2           wet stereo width, 1 unchanged, Linear
///   7 gain          -30..10 dB     wet gain, Linear
///   8 mix           0..1           dry / wet, Linear
class ConvolutionProcessor final : public Processor
{
public:
    static constexpr double kMaxPredelayMs = 250.0;
    static constexpr double kCrossfadeMs = 20.0;
    static constexpr double kDefaultMaxIrSeconds = 6.0;

    ConvolutionProcessor() = default;
    ConvolutionProcessor(const ConvolutionProcessor&) = delete;
    ConvolutionProcessor& operator=(const ConvolutionProcessor&) = delete;

    [[nodiscard]] static const ParamTable& params() noexcept;

    /// Longest IR the convolvers are sized for. Their state costs about 8 bytes per IR sample per
    /// channel and convolver (two convolvers): 9 MB for 6 s stereo at 48 kHz. Call before
    /// prepare().
    void setMaxIrSeconds(double seconds) noexcept { maxIrSeconds_ = seconds; }

    void prepare(const ProcessSpec& spec) override;
    void reset() noexcept override;
    void setParams(const float* normalised, const ControlContext& ctx) noexcept override;
    void process(float* const* channels, int n) noexcept override;
    [[nodiscard]] int tailSamples() const noexcept override;

    /// Any thread. Normally called by IrBank::publish. The view must stay alive until
    /// acknowledgedGeneration() >= a later view's generation (IrBank takes care of that).
    void setBankView(const IrBankView* view) noexcept { pending_.store(view, std::memory_order_release); }
    /// Any thread: the newest generation the audio thread has fully switched to.
    [[nodiscard]] std::uint64_t acknowledgedGeneration() const noexcept
    {
        return acknowledged_.load(std::memory_order_acquire);
    }
    /// Audio thread (or tests): the IR currently faded in, nullptr when passing dry.
    [[nodiscard]] const PreparedIr* currentIr() const noexcept
    {
        return engines_[static_cast<std::size_t>(active_)].ir();
    }
    [[nodiscard]] bool crossfading() const noexcept { return fading_; }

private:
    void processChunk(float* const* ch, int n) noexcept;
    void clearState() noexcept;
    void updateTrim() noexcept;
    [[nodiscard]] bool viewHolds(const PreparedIr* ir) const noexcept;

    double sampleRate_ = 48000.0;
    int channels_ = 2;
    double maxIrSeconds_ = kDefaultMaxIrSeconds;
    bool first_ = true;

    std::atomic<const IrBankView*> pending_{nullptr};
    std::atomic<std::uint64_t> acknowledged_{0};
    const IrBankView* view_ = nullptr;
    int index_ = 0;

    std::array<PartitionedConvolver, 2> engines_;
    int active_ = 0; // engine faded in (or fading in)
    bool fading_ = false;
    float fadePos_ = 1.0f;
    float fadeStep_ = 0.0f;

    // Pre-delay: one line per channel, read with linear interpolation at a smoothed delay.
    std::vector<float> predelay_;
    int predelaySize_ = 0;
    int predelayMask_ = 0;
    int writePos_ = 0;
    float predelaySamples_ = 0.0f;
    float predelayTarget_ = 0.0f;
    float predelaySmooth_ = 1.0f;

    float keepFrac_ = 1.0f, keepTarget_ = 1.0f;
    float fadeFrac_ = 0.0f, fadeTarget_ = 0.0f;
    float blockSmooth_ = 1.0f;
    float lowCutHz_ = 20.0f, lowCutTarget_ = 20.0f;
    float highCutHz_ = 22000.0f, highCutTarget_ = 22000.0f;
    std::array<detail::TptSvf, 2> lowCut_;
    std::array<detail::TptSvf, 2> highCut_;
    detail::LinearRamp width_;
    detail::LinearRamp gain_;
    detail::LinearRamp mix_;

    static constexpr int kChunk = 64;
    std::array<std::array<float, kChunk>, 2> dry_{};
    std::array<std::array<float, kChunk>, 2> delayed_{};
    std::array<std::array<std::array<float, kChunk>, 2>, 2> wet_{}; // [engine][channel]
};

} // namespace alfa::dsp
