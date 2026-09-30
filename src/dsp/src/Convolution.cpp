#include "alfa/dsp/Convolution.h"

#include "alfa/dsp/Spatial.h"
#include "alfa/dsp/detail/Denormals.h"
#include "alfa/dsp/detail/FastMath.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace alfa::dsp
{

namespace
{
constexpr double kPiD = 3.14159265358979323846;

/// acc += g * x * h for split spectra (re[0..B) | im[0..B)) of a 2B-point real FFT, where re[0]
/// holds DC and im[0] Nyquist (pffft's ordered layout, deinterleaved). Bin 0 is fixed up
/// afterwards so the main loop stays a plain complex multiply that vectorises.
void multiplyAdd(float* acc, const float* x, const float* h, int bins, float g) noexcept
{
    float* ar = acc;
    float* ai = acc + bins;
    const float* xr = x;
    const float* xi = x + bins;
    const float* hr = h;
    const float* hi = h + bins;
    const float dc = ar[0] + g * xr[0] * hr[0];
    const float nyquist = ai[0] + g * xi[0] * hi[0];
    if (g == 1.0f)
    {
#pragma omp simd
        for (int k = 0; k < bins; ++k)
        {
            const float re = xr[k] * hr[k] - xi[k] * hi[k];
            const float im = xr[k] * hi[k] + xi[k] * hr[k];
            ar[k] += re;
            ai[k] += im;
        }
    }
    else
    {
#pragma omp simd
        for (int k = 0; k < bins; ++k)
        {
            const float re = xr[k] * hr[k] - xi[k] * hi[k];
            const float im = xr[k] * hi[k] + xi[k] * hr[k];
            ar[k] += g * re;
            ai[k] += g * im;
        }
    }
    ar[0] = dc;
    ai[0] = nyquist;
}

void deinterleave(const float* ordered, float* split, int bins) noexcept
{
    for (int k = 0; k < bins; ++k)
    {
        split[k] = ordered[2 * k];
        split[bins + k] = ordered[2 * k + 1];
    }
}

void interleave(const float* split, float* ordered, int bins) noexcept
{
    for (int k = 0; k < bins; ++k)
    {
        ordered[2 * k] = split[k];
        ordered[2 * k + 1] = split[bins + k];
    }
}

/// Spectrum of one partition (block samples of h, zero-padded to 2 block), prescaled by 1/N so
/// the unscaled inverse FFT returns the convolution directly.
void partitionSpectrum(detail::RealFft& fft, detail::AlignedBuffer& work, const float* h, int count,
                       int block, float gain, float* split)
{
    work.clear();
    const float scale = gain / static_cast<float>(2 * block);
    for (int t = 0; t < count; ++t)
        work[static_cast<std::size_t>(t)] = h[t] * scale;
    fft.forward(work.data(), work.data());
    deinterleave(work.data(), split, block);
}

bool allFinite(const float* x, int n) noexcept
{
    float acc = 0.0f;
    for (int i = 0; i < n; ++i)
        acc += x[i] - x[i];
    return acc == 0.0f;
}

std::size_t idx(int i) noexcept
{
    return static_cast<std::size_t>(i);
}
} // namespace

// FirHistory / DirectFir -----------------------------------------------------------------------

void FirHistory::load(const float* in, int n) noexcept
{
    std::copy_n(in, n, buf_.data() + (kMaxTaps - 1));
}

void FirHistory::apply(const float* reversed, float* out, int n, bool accumulate) const noexcept
{
    const float* b = buf_.data();
    for (int i = 0; i < n; ++i)
    {
        float s = 0.0f;
#pragma omp simd reduction(+ : s)
        for (int k = 0; k < kMaxTaps; ++k)
            s += reversed[k] * b[i + k];
        out[i] = accumulate ? out[i] + s : s;
    }
}

void FirHistory::advance(int n) noexcept
{
    std::copy_n(buf_.data() + n, kMaxTaps - 1, buf_.data());
}

void DirectFir::setKernel(const float* h, int taps) noexcept
{
    taps_ = std::clamp(taps, 0, kMaxTaps);
    reversed_.fill(0.0f);
    for (int k = 0; k < taps_; ++k)
        reversed_[idx(kMaxTaps - 1 - k)] = h[k];
}

void DirectFir::process(const float* in, float* out, int n) noexcept
{
    for (int done = 0; done < n;)
    {
        const int k = std::min(FirHistory::kChunk, n - done);
        history_.load(in + done, k); // copied before out is written, so in == out is fine
        history_.apply(reversed_.data(), out + done, k, false);
        history_.advance(k);
        done += k;
    }
}

// PartitionStage -------------------------------------------------------------------------------

namespace conv
{

void PartitionStage::prepare(int block, int firstDelay, int maxPartitions, int inputs, int outputs)
{
    if (block < 16 || (block & (block - 1)) != 0)
        throw std::invalid_argument("partition size must be a power of two >= 16");
    block_ = block;
    firstDelay_ = std::max(1, firstDelay);
    lag_ = firstDelay_ >= 2 ? firstDelay_ - 2 : 0;
    maxPartitions_ = std::max(0, maxPartitions);
    ring_ = lag_ + maxPartitions_ + 1;
    inputs_ = std::clamp(inputs, 1, 2);
    outputs_ = std::clamp(outputs, 1, 2);
    fft_.prepare(2 * block_);
    const auto two = static_cast<std::size_t>(2 * block_);
    for (int i = 0; i < 2; ++i)
    {
        const bool in = i < inputs_;
        const bool out = i < outputs_;
        time_[idx(i)].resize(in ? two : 0);
        fdl_[idx(i)].resize(in ? two * static_cast<std::size_t>(ring_) : 0);
        acc_[idx(i)].resize(out ? two : 0);
        out_[idx(i)].resize(out ? static_cast<std::size_t>(block_) : 0);
    }
    scratch_.resize(two);
    reset();
}

void PartitionStage::reset() noexcept
{
    for (int i = 0; i < 2; ++i)
    {
        time_[idx(i)].clear();
        acc_[idx(i)].clear();
        out_[idx(i)].clear();
    }
    // The delay line is not cleared: spectra_ counts what was written since, and older slots
    // are never read (usable()).
    pos_ = 0;
    newest_ = ring_ - 1;
    spectra_ = 0;
    macDone_ = 0;
}

int PartitionStage::usable(const StageFilter& filter) const noexcept
{
    const int p = std::min(filter.partitions, maxPartitions_);
    const long long available = spectra_ - lag_;
    return static_cast<int>(std::clamp<long long>(available, 0, p));
}

void PartitionStage::macUpTo(int target, const StageFilter& filter) noexcept
{
    const int end = std::min(target, usable(filter));
    const auto span = static_cast<std::size_t>(2 * block_);
    for (int j = macDone_; j < end; ++j)
    {
        const float g = filter.gains != nullptr ? filter.gains[j] : 1.0f;
        if (g == 0.0f)
            continue;
        int slot = (newest_ - lag_ - j) % ring_;
        if (slot < 0)
            slot += ring_;
        for (int r = 0; r < filter.routeCount; ++r)
        {
            const Route& route = filter.routes[idx(r)];
            const float* x = fdl_[idx(route.in)].data() + static_cast<std::size_t>(slot) * span;
            const float* h = route.spectra + static_cast<std::size_t>(j) * span;
            multiplyAdd(acc_[idx(route.out)].data(), x, h, block_, g);
        }
    }
    macDone_ = std::max(macDone_, end);
}

void PartitionStage::boundary(const StageFilter& filter) noexcept
{
    const auto b = static_cast<std::size_t>(block_);
    const auto span = 2 * b;
    // With a delay of two or more blocks the pending block needs nothing from this one, so it
    // is finished first, before the ring slot it may still read is overwritten.
    if (firstDelay_ >= 2)
        macUpTo(INT_MAX, filter);

    newest_ = (newest_ + 1) % ring_;
    for (int i = 0; i < inputs_; ++i)
    {
        float* t = time_[idx(i)].data();
        fft_.forward(t, scratch_.data());
        deinterleave(scratch_.data(), fdl_[idx(i)].data() + static_cast<std::size_t>(newest_) * span, block_);
        std::copy_n(t + b, b, t);
    }
    ++spectra_;

    if (firstDelay_ == 1)
        macUpTo(INT_MAX, filter);

    for (int o = 0; o < outputs_; ++o)
    {
        float* out = out_[idx(o)].data();
        if (macDone_ == 0)
        {
            std::fill_n(out, b, 0.0f);
            continue;
        }
        float* acc = acc_[idx(o)].data();
        interleave(acc, scratch_.data(), block_);
        fft_.inverse(scratch_.data(), scratch_.data());
        std::copy_n(scratch_.data() + b, b, out);
        std::fill_n(acc, span, 0.0f);
    }
    macDone_ = 0;
    pos_ = 0;
}

void PartitionStage::process(const float* const* in, float* const* out, int n,
                             const StageFilter& filter) noexcept
{
    for (int o = 0; o < outputs_; ++o)
    {
        const float* src = out_[idx(o)].data() + pos_;
        float* dst = out[o];
        for (int i = 0; i < n; ++i)
            dst[i] += src[i];
    }
    for (int i = 0; i < inputs_; ++i)
        std::copy_n(in[i], n, time_[idx(i)].data() + block_ + pos_);
    pos_ += n;
    if (pos_ >= block_)
    {
        boundary(filter);
        return;
    }
    if (firstDelay_ >= 2)
    {
        // Spread the next block's multiply-adds evenly over this one.
        const long long p = usable(filter);
        const auto target = static_cast<int>((p * pos_ + block_ - 1) / block_);
        macUpTo(target, filter);
    }
}

} // namespace conv

// UniformPartitionedConvolver ------------------------------------------------------------------

void UniformPartitionedConvolver::prepare(const float* ir, int length, int partitionSize)
{
    if (partitionSize < kControlBlock || (partitionSize & (partitionSize - 1)) != 0)
        throw std::invalid_argument("partition size must be a power of two >= kControlBlock");
    if (ir == nullptr || length <= 0)
        throw std::invalid_argument("empty impulse response");
    const int b = partitionSize;
    const int partitions = (length + b - 1) / b;
    detail::RealFft fft;
    fft.prepare(2 * b);
    detail::AlignedBuffer work(static_cast<std::size_t>(2 * b));
    spectra_.resize(static_cast<std::size_t>(partitions) * static_cast<std::size_t>(2 * b));
    for (int j = 0; j < partitions; ++j)
        partitionSpectrum(fft, work, ir + static_cast<std::ptrdiff_t>(j) * b, std::min(b, length - j * b), b,
                          1.0f,
                          spectra_.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(2 * b));
    // Delay 1 block: partition 0 is heard one partition late, which is the reported latency.
    stage_.prepare(b, 1, partitions, 1, 1);
    filter_ = {};
    filter_.routes[0] = {0, 0, spectra_.data()};
    filter_.routeCount = 1;
    filter_.partitions = partitions;
}

void UniformPartitionedConvolver::process(const float* in, float* out, int n) noexcept
{
    for (int done = 0; done < n;)
    {
        const int k = std::min({n - done, stage_.remaining(), static_cast<int>(scratchIn_.size())});
        std::copy_n(in + done, k, scratchIn_.data());
        std::fill_n(scratchOut_.data(), k, 0.0f);
        const float* ins[1] = {scratchIn_.data()};
        float* outs[1] = {scratchOut_.data()};
        stage_.process(ins, outs, k, filter_);
        std::copy_n(scratchOut_.data(), k, out + done);
        done += k;
    }
}

// Resampling and PreparedIr --------------------------------------------------------------------

std::vector<float> resampleIr(const std::vector<float>& ir, double fromRate, double toRate)
{
    if (!(fromRate > 0.0) || !(toRate > 0.0))
        throw std::invalid_argument("sample rates must be positive");
    if (ir.empty() || fromRate == toRate)
        return ir;
    const double ratio = toRate / fromRate;
    const double cutoff = std::min(1.0, ratio); // relative to the input Nyquist
    constexpr double kZeroCrossings = 16.0;
    const double half = kZeroCrossings / cutoff; // window half-width in input samples
    const auto n = static_cast<long long>(ir.size());
    const auto outLength = static_cast<long long>(std::ceil(static_cast<double>(n) * ratio - 1e-9));
    // A continuous-time IR sampled faster has more taps of the same height, so scale by the
    // rate ratio to keep the filter's gain.
    const double scale = cutoff / ratio;
    std::vector<float> out(static_cast<std::size_t>(outLength));
    for (long long i = 0; i < outLength; ++i)
    {
        const double t = static_cast<double>(i) / ratio;
        const auto k0 = std::max(0LL, static_cast<long long>(std::ceil(t - half)));
        const auto k1 = std::min(n - 1, static_cast<long long>(std::floor(t + half)));
        double acc = 0.0;
        for (long long k = k0; k <= k1; ++k)
        {
            const double x = t - static_cast<double>(k);
            const double u = (x + half) / (2.0 * half);
            const double w = 0.42 - 0.5 * std::cos(2.0 * kPiD * u) + 0.08 * std::cos(4.0 * kPiD * u);
            const double a = kPiD * cutoff * x;
            const double sinc = std::abs(a) < 1e-9 ? 1.0 : std::sin(a) / a;
            acc += static_cast<double>(ir[static_cast<std::size_t>(k)]) * sinc * w;
        }
        out[static_cast<std::size_t>(i)] = static_cast<float>(acc * scale);
    }
    return out;
}

std::shared_ptr<const PreparedIr> PreparedIr::create(const IrSource& source, double hostRate,
                                                     const IrPrepareOptions& options)
{
    const auto paths = static_cast<int>(source.channels.size());
    if (paths != 1 && paths != 2 && paths != 4)
        throw std::invalid_argument("an IR needs 1, 2 or 4 channels");
    if (!(hostRate > 0.0) || !(source.sampleRate > 0.0))
        throw std::invalid_argument("sample rates must be positive");

    std::vector<std::vector<float>> h;
    h.reserve(static_cast<std::size_t>(paths));
    std::size_t length = 0;
    for (const auto& c : source.channels)
    {
        h.push_back(resampleIr(c, source.sampleRate, hostRate));
        length = std::max(length, h.back().size());
    }
    if (length == 0)
        throw std::invalid_argument("empty impulse response");
    const auto maxLength = static_cast<std::size_t>(std::max(1.0, options.maxSeconds * hostRate));
    const bool truncated = length > maxLength;
    length = std::min(length, maxLength);
    for (auto& c : h)
    {
        for (auto& v : c)
            v = std::isfinite(v) ? v : 0.0f;
        c.resize(length, 0.0f);
        if (truncated)
        {
            const auto fade = std::min(length, static_cast<std::size_t>(0.01 * hostRate) + 1);
            for (std::size_t i = 0; i < fade; ++i)
                c[length - 1 - i] *= static_cast<float>(
                    0.5 - 0.5 * std::cos(kPiD * static_cast<double>(i) / static_cast<double>(fade)));
        }
    }

    auto energy = [&](int p)
    {
        double e = 0.0;
        for (float v : h[static_cast<std::size_t>(p)])
            e += static_cast<double>(v) * v;
        return e;
    };
    double loudest = 0.0;
    if (paths == 4)
        loudest = std::max(energy(0) + energy(2), energy(1) + energy(3));
    else
        for (int p = 0; p < paths; ++p)
            loudest = std::max(loudest, energy(p));
    double gain = std::pow(10.0, static_cast<double>(options.gainDb) / 20.0);
    if (options.normalise && loudest > 0.0)
        gain /= std::sqrt(loudest);

    auto ir = std::make_shared<PreparedIr>();
    ir->sampleRate_ = hostRate;
    ir->paths_ = paths;
    ir->length_ = static_cast<int>(length);
    ir->gain_ = static_cast<float>(gain);
    ir->name_ = source.name;
    const float g = ir->gain_;
    const int len = ir->length_;

    ir->head_.resize(static_cast<std::size_t>(paths * conv::kHeadTaps));
    for (int p = 0; p < paths; ++p)
    {
        const auto& c = h[static_cast<std::size_t>(p)];
        float* dst = ir->head_.data() + static_cast<std::size_t>(p * conv::kHeadTaps);
        for (int k = 0; k < std::min(len, conv::kHeadTaps); ++k)
            dst[conv::kHeadTaps - 1 - k] = g * c[static_cast<std::size_t>(k)];
    }

    for (int s = 0; s < conv::kStages; ++s)
    {
        const int b = conv::kStageBlock[idx(s)];
        const int start = conv::kStageStart[idx(s)];
        const int end = s + 1 < conv::kStages ? std::min(len, conv::kStageStart[idx(s + 1)]) : len;
        const int count = end > start ? (end - start + b - 1) / b : 0;
        ir->partitions_[idx(s)] = count;
        if (count == 0)
            continue;
        detail::RealFft fft;
        fft.prepare(2 * b);
        detail::AlignedBuffer work(static_cast<std::size_t>(2 * b));
        const auto span = static_cast<std::size_t>(2 * b);
        ir->spectra_[idx(s)].resize(static_cast<std::size_t>(paths) * static_cast<std::size_t>(count) * span);
        for (int p = 0; p < paths; ++p)
        {
            const float* c = h[static_cast<std::size_t>(p)].data();
            for (int j = 0; j < count; ++j)
            {
                const int from = start + j * b;
                const int taps = std::min(b, end - from);
                float* dst = ir->spectra_[idx(s)].data() +
                             (static_cast<std::size_t>(p) * static_cast<std::size_t>(count) +
                              static_cast<std::size_t>(j)) *
                                 span;
                partitionSpectrum(fft, work, c + from, taps, b, g, dst);
            }
        }
    }
    return ir;
}

const float* PreparedIr::head(int path) const noexcept
{
    return head_.data() + static_cast<std::size_t>(path * conv::kHeadTaps);
}

const float* PreparedIr::spectra(int stage, int path) const noexcept
{
    const int count = partitions_[idx(stage)];
    if (count == 0)
        return nullptr;
    return spectra_[idx(stage)].data() + static_cast<std::size_t>(path) * static_cast<std::size_t>(count) *
                                             static_cast<std::size_t>(2 * conv::kStageBlock[idx(stage)]);
}

// PartitionedConvolver -------------------------------------------------------------------------

void PartitionedConvolver::prepare(int channels, int maxIrLength)
{
    channels_ = std::clamp(channels, 1, 2);
    maxIrLength_ = std::max(1, maxIrLength);
    for (int s = 0; s < conv::kStages; ++s)
    {
        const int b = conv::kStageBlock[idx(s)];
        const int start = conv::kStageStart[idx(s)];
        const int end =
            s + 1 < conv::kStages ? std::min(maxIrLength_, conv::kStageStart[idx(s + 1)]) : maxIrLength_;
        const int count = end > start ? (end - start + b - 1) / b : 0;
        stages_[idx(s)].prepare(b, start / b, count, channels_, channels_);
        gains_[idx(s)].assign(static_cast<std::size_t>(count), 1.0f);
    }
    setIr(nullptr);
}

void PartitionedConvolver::reset() noexcept
{
    for (auto& h : head_)
        h.reset();
    for (auto& s : stages_)
        s.reset();
}

void PartitionedConvolver::setIr(const PreparedIr* ir) noexcept
{
    ir_ = ir;
    headRouteCount_ = 0;
    for (auto& f : filters_)
        f = {};
    keep_ = -1.0f;
    fade_ = 0.0f;
    trimmed_ = false;
    reset();
    if (ir == nullptr)
        return;

    // (input, output, path) triples; see PartitionedConvolver's comment for mono processors.
    std::array<std::array<int, 3>, conv::kMaxPaths> routes{};
    int count = 0;
    if (ir->paths() == 4 && channels_ == 2)
    {
        routes = {{{0, 0, 0}, {0, 1, 1}, {1, 0, 2}, {1, 1, 3}}};
        count = 4;
    }
    else
    {
        for (int c = 0; c < channels_; ++c)
            routes[idx(c)] = {c, c, ir->paths() == 2 ? c : 0};
        count = channels_;
    }

    headRouteCount_ = count;
    for (int r = 0; r < count; ++r)
    {
        const auto& t = routes[idx(r)];
        headRoutes_[idx(r)] = {t[0], t[1], nullptr};
        headKernel_[idx(r)] = ir->head(t[2]);
    }
    for (int s = 0; s < conv::kStages; ++s)
    {
        auto& f = filters_[idx(s)];
        f.partitions = std::min(ir->partitions(s), stages_[idx(s)].maxPartitions());
        f.routeCount = f.partitions > 0 ? count : 0;
        for (int r = 0; r < f.routeCount; ++r)
        {
            const auto& t = routes[idx(r)];
            f.routes[idx(r)] = {t[0], t[1], ir->spectra(s, t[2])};
        }
    }
}

void PartitionedConvolver::setTrim(float keepSamples, float fadeSamples) noexcept
{
    if (ir_ == nullptr || (keepSamples == keep_ && fadeSamples == fade_))
        return;
    keep_ = keepSamples;
    fade_ = fadeSamples;
    const auto length = static_cast<float>(std::min(ir_->length(), maxIrLength_));
    trimmed_ = keep_ < length || fade_ > 0.0f;
    const float fadeStart = keep_ - std::max(fade_, 0.0f);
    for (int s = 0; s < conv::kStages; ++s)
    {
        auto& f = filters_[idx(s)];
        if (!trimmed_)
        {
            f.gains = nullptr;
            continue;
        }
        auto& g = gains_[idx(s)];
        const int b = conv::kStageBlock[idx(s)];
        const int start = conv::kStageStart[idx(s)];
        for (int j = 0; j < f.partitions; ++j)
        {
            const float t = static_cast<float>(start + j * b) + 0.5f * static_cast<float>(b);
            float v = 1.0f;
            if (t >= keep_)
                v = 0.0f;
            else if (t > fadeStart && fade_ > 0.0f)
                v = 0.5f + 0.5f * std::cos(detail::kPi * (t - fadeStart) / fade_);
            g[idx(j)] = v;
        }
        f.gains = g.data();
    }
}

int PartitionedConvolver::activeLength() const noexcept
{
    if (ir_ == nullptr)
        return 0;
    int length = std::min(ir_->length(), maxIrLength_);
    if (trimmed_)
        length = std::min(length, static_cast<int>(std::ceil(std::max(keep_, 0.0f))) +
                                      conv::kStageBlock[conv::kStages - 1]);
    return length;
}

void PartitionedConvolver::process(const float* const* in, float* const* out, int n) noexcept
{
    for (int c = 0; c < channels_; ++c)
        std::fill_n(out[c], n, 0.0f);
    if (ir_ == nullptr)
        return;
    for (int done = 0; done < n;)
    {
        // Every stage block is a multiple of the first and all start together, so the first
        // stage's boundary is the only one a chunk can cross.
        const int k = std::min({n - done, stages_[0].remaining(), FirHistory::kChunk});
        const float* ins[2] = {in[0] + done, channels_ > 1 ? in[1] + done : nullptr};
        float* outs[2] = {out[0] + done, channels_ > 1 ? out[1] + done : nullptr};
        for (int c = 0; c < channels_; ++c)
            head_[idx(c)].load(ins[c], k);
        for (int r = 0; r < headRouteCount_; ++r)
        {
            const auto& route = headRoutes_[idx(r)];
            head_[idx(route.in)].apply(headKernel_[idx(r)], outs[route.out], k, true);
        }
        for (int c = 0; c < channels_; ++c)
            head_[idx(c)].advance(k);
        for (int s = 0; s < conv::kStages; ++s)
            if (filters_[idx(s)].partitions > 0)
                stages_[idx(s)].process(ins, outs, k, filters_[idx(s)]);
        done += k;
    }
}

// IrBank ---------------------------------------------------------------------------------------

int IrBank::add(IrSource source, const IrPrepareOptions& options)
{
    if (size() >= kMaxBankIrs)
        return -1;
    auto prepared = PreparedIr::create(source, hostRate_, options);
    entries_.push_back({std::move(source), options, std::move(prepared)});
    return size() - 1;
}

void IrBank::replace(int index, IrSource source, const IrPrepareOptions& options)
{
    if (index < 0 || index >= size())
        throw std::out_of_range("IrBank index");
    auto prepared = PreparedIr::create(source, hostRate_, options);
    entries_[idx(index)] = {std::move(source), options, std::move(prepared)};
}

void IrBank::remove(int index)
{
    if (index < 0 || index >= size())
        throw std::out_of_range("IrBank index");
    entries_.erase(entries_.begin() + index);
}

void IrBank::clear()
{
    entries_.clear();
}

void IrBank::setHostRate(double hostRate)
{
    if (!(hostRate > 0.0))
        throw std::invalid_argument("sample rate must be positive");
    if (hostRate == hostRate_)
        return;
    // Prepare everything first so a throw leaves the bank unchanged.
    std::vector<std::shared_ptr<const PreparedIr>> prepared;
    prepared.reserve(entries_.size());
    for (const auto& e : entries_)
        prepared.push_back(PreparedIr::create(e.source, hostRate, e.options));
    for (std::size_t i = 0; i < entries_.size(); ++i)
        entries_[i].prepared = std::move(prepared[i]);
    hostRate_ = hostRate;
}

std::shared_ptr<const PreparedIr> IrBank::get(int index) const
{
    if (index < 0 || index >= size())
        return nullptr;
    return entries_[idx(index)].prepared;
}

void IrBank::publish(ConvolutionProcessor& processor)
{
    Published p;
    p.view = std::make_unique<IrBankView>();
    p.view->generation = nextGeneration_++;
    p.view->count = size();
    p.keep.reserve(entries_.size());
    for (std::size_t i = 0; i < entries_.size(); ++i)
    {
        p.view->irs[i] = entries_[i].prepared.get();
        p.keep.push_back(entries_[i].prepared);
    }
    const IrBankView* view = p.view.get();
    published_.push_back(std::move(p));
    processor.setBankView(view);
}

void IrBank::collect(const ConvolutionProcessor& processor)
{
    collectBefore(processor.acknowledgedGeneration());
}

void IrBank::collectBefore(std::uint64_t generation)
{
    std::erase_if(published_, [generation](const Published& p) { return p.view->generation < generation; });
}

// Synthetic IRs --------------------------------------------------------------------------------

IrSource makeSyntheticIr(const SyntheticIrSpec& spec)
{
    const double sr = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    const double rt60 = std::clamp(spec.rt60, 0.05, 20.0);
    const double seconds = std::clamp(spec.seconds > 0.0 ? spec.seconds : rt60, 0.05, 10.0);
    const auto n = static_cast<std::size_t>(seconds * sr);
    const int channels = std::clamp(spec.channels, 1, 2);

    IrSource src;
    src.sampleRate = sr;
    const char* names[] = {"decay", "room", "plate", "spring"};
    src.name = names[static_cast<int>(spec.kind)];

    // Per kind: starting and final brightness of the tail (Hz), how much faster than the
    // broadband decay the brightness falls, and when the diffuse tail starts.
    double brightStart = 12000.0, brightEnd = 800.0, darkening = 1.0, diffuseStart = 0.0;
    switch (spec.kind)
    {
    case SyntheticIrKind::Decay:
        break;
    case SyntheticIrKind::Room:
        brightStart = 9000.0;
        brightEnd = 500.0;
        darkening = 1.5;
        diffuseStart = 0.008;
        break;
    case SyntheticIrKind::Plate:
        brightStart = 16000.0;
        brightEnd = 2500.0;
        darkening = 0.6;
        break;
    case SyntheticIrKind::Spring:
        brightStart = 4500.0;
        brightEnd = 1200.0;
        darkening = 0.5;
        break;
    }

    for (int c = 0; c < channels; ++c)
    {
        detail::Random rng(spec.seed * 2654435761u + static_cast<std::uint32_t>(c) * 97u + 1u);
        std::vector<float> h(n, 0.0f);
        float lp = 0.0f, hpState = 0.0f;
        double a = 1.0, comp = 1.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(i) / sr;
            if (i % 64 == 0)
            {
                // Brightness falls exponentially; the gain compensation keeps the broadband
                // energy on the exp decay, so the RT60 holds while the tail darkens.
                const double fc = std::max(brightEnd, brightStart * std::exp(-3.0 * darkening * t / rt60));
                a = 1.0 - std::exp(-2.0 * kPiD * std::min(fc, 0.45 * sr) / sr);
                comp = std::sqrt((2.0 - a) / a);
            }
            const double env = std::pow(10.0, -3.0 * t / rt60);
            lp += static_cast<float>(a) * (rng.nextBipolar() - lp);
            double v = static_cast<double>(lp) * comp * 1.7320508; // unit-variance noise
            if (spec.kind == SyntheticIrKind::Spring)
            {
                // Remove the lows too: springs have little below ~200 Hz.
                hpState += 0.026f * (static_cast<float>(v) - hpState);
                v -= static_cast<double>(hpState);
                v *= 0.35;
            }
            if (t < diffuseStart)
                v = 0.0;
            else if (diffuseStart > 0.0 && t < diffuseStart + 0.015)
                v *= (t - diffuseStart) / 0.015;
            h[i] = static_cast<float>(v * env);
        }

        if (spec.kind == SyntheticIrKind::Room)
        {
            // Sparse early reflections, 3..45 ms, falling in level.
            h[0] = 1.0f;
            for (int r = 0; r < 14; ++r)
            {
                const double t = 0.003 + 0.042 * (0.5 + 0.5 * static_cast<double>(rng.nextBipolar()));
                const auto i = static_cast<std::size_t>(t * sr);
                if (i < n)
                    h[i] += static_cast<float>((rng.nextBipolar() < 0.0f ? -0.7 : 0.7) * (1.0 - t / 0.06));
            }
        }
        else if (spec.kind == SyntheticIrKind::Spring)
        {
            // Dispersive echoes: each round trip is a falling chirp, 5 kHz to 300 Hz in 20 ms.
            const double period = 0.033 + 0.0015 * c;
            const double chirp = 0.02;
            for (int k = 0;; ++k)
            {
                const double t0 = 0.004 + period * k;
                if (t0 >= seconds)
                    break;
                const double amp = 0.8 * std::pow(10.0, -3.0 * t0 / rt60);
                double phase = 0.0;
                for (std::size_t i = static_cast<std::size_t>(t0 * sr); i < n; ++i)
                {
                    const double u = (static_cast<double>(i) / sr - t0) / chirp;
                    if (u >= 1.0)
                        break;
                    const double f = 5000.0 * std::pow(300.0 / 5000.0, u);
                    phase += f / sr;
                    const double w = std::sin(kPiD * u);
                    h[i] += static_cast<float>(amp * w * w * std::sin(2.0 * kPiD * phase));
                }
            }
        }

        // Fade the last 5% so a truncated tail does not end in a step.
        const std::size_t fade = std::max<std::size_t>(1, n / 20);
        for (std::size_t i = 0; i < fade && i < n; ++i)
            h[n - 1 - i] *= static_cast<float>(
                0.5 - 0.5 * std::cos(kPiD * static_cast<double>(i) / static_cast<double>(fade)));
        src.channels.push_back(std::move(h));
    }
    return src;
}

// ConvolutionProcessor -------------------------------------------------------------------------

const ParamTable& ConvolutionProcessor::params() noexcept
{
    static const ParamTable table = []
    {
        ParamTable t{};
        t[0] = {"ir", 0.0f, static_cast<float>(kMaxBankIrs - 1), Curve::Choice, 0.0f, ""};
        t[1] = {"predelay", 0.0f, static_cast<float>(kMaxPredelayMs), Curve::Linear, 0.0f, "ms"};
        t[2] = {"length", 5.0f, 100.0f, Curve::Linear, 1.0f, "%"};
        t[3] = {"fade", 0.0f, 100.0f, Curve::Linear, 0.0f, "%"};
        t[4] = {"low_cut", 20.0f, 1000.0f, Curve::Log, 0.0f, "Hz"};
        t[5] = {"high_cut", 1000.0f, 22000.0f, Curve::Log, 1.0f, "Hz"};
        t[6] = {"width", 0.0f, 2.0f, Curve::Linear, 0.5f, ""};
        t[7] = {"gain", -30.0f, 10.0f, Curve::Linear, 0.75f, "dB"};
        t[8] = {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""};
        return t;
    }();
    return table;
}

void ConvolutionProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate > 0.0 ? spec.sampleRate : 48000.0;
    channels_ = std::clamp(spec.channels, 1, 2);
    const int maxIr = static_cast<int>(std::max(0.05, maxIrSeconds_) * sampleRate_);
    for (auto& e : engines_)
        e.prepare(channels_, maxIr);
    const int maxDelay = static_cast<int>(std::ceil(kMaxPredelayMs * 1e-3 * sampleRate_)) + 4;
    int size = 1;
    while (size < maxDelay)
        size <<= 1;
    predelaySize_ = size;
    predelayMask_ = size - 1;
    predelay_.assign(static_cast<std::size_t>(2 * size), 0.0f);
    predelaySmooth_ = detail::smoothingCoeff(0.03, sampleRate_);
    blockSmooth_ = detail::smoothingCoeff(0.03, sampleRate_, kControlBlock);
    fadeStep_ = static_cast<float>(1.0 / (kCrossfadeMs * 1e-3 * sampleRate_));
    view_ = nullptr;
    active_ = 0;
    fading_ = false;
    fadePos_ = 1.0f;
    reset();
}

void ConvolutionProcessor::clearState() noexcept
{
    for (auto& e : engines_)
        e.reset();
    std::fill(predelay_.begin(), predelay_.end(), 0.0f);
    writePos_ = 0;
    for (auto& f : lowCut_)
        f.reset();
    for (auto& f : highCut_)
        f.reset();
}

void ConvolutionProcessor::reset() noexcept
{
    // Keep the selected IR; drop a half-finished swap.
    if (fading_)
    {
        engines_[static_cast<std::size_t>(1 - active_)].setIr(nullptr);
        fading_ = false;
        fadePos_ = 1.0f;
    }
    clearState();
    width_.reset(1.0f);
    gain_.reset(1.0f);
    mix_.reset(1.0f);
    first_ = true;
}

bool ConvolutionProcessor::viewHolds(const PreparedIr* ir) const noexcept
{
    if (ir == nullptr)
        return true;
    if (view_ == nullptr)
        return false;
    for (int i = 0; i < view_->count; ++i)
        if (view_->irs[idx(i)] == ir)
            return true;
    return false;
}

void ConvolutionProcessor::updateTrim() noexcept
{
    for (auto& e : engines_)
    {
        const PreparedIr* ir = e.ir();
        if (ir == nullptr)
            continue;
        const float length = static_cast<float>(ir->length());
        const float keep = keepFrac_ >= 1.0f ? length : keepFrac_ * length;
        e.setTrim(keep, fadeFrac_ * keep);
    }
}

void ConvolutionProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const ParamTable& t = params();

    view_ = pending_.load(std::memory_order_acquire);
    index_ = static_cast<int>(mapParam(t[0], p[0]));
    const PreparedIr* target = nullptr;
    if (view_ != nullptr && view_->count > 0)
    {
        target = view_->irs[idx(std::clamp(index_, 0, std::min(view_->count, kMaxBankIrs) - 1))];
        if (target != nullptr && std::abs(target->sampleRate() - sampleRate_) > 1e-6 * sampleRate_)
            target = nullptr; // prepared for another rate: pass dry until the bank is re-prepared
    }

    auto& current = engines_[static_cast<std::size_t>(active_)];
    if (!fading_ && target != current.ir())
    {
        const int other = 1 - active_;
        engines_[static_cast<std::size_t>(other)].setIr(target);
        active_ = other;
        if (first_)
        {
            engines_[static_cast<std::size_t>(1 - active_)].setIr(nullptr);
            fadePos_ = 1.0f;
        }
        else
        {
            fading_ = true;
            fadePos_ = 0.0f;
        }
    }
    if (view_ != nullptr && viewHolds(engines_[0].ir()) && viewHolds(engines_[1].ir()))
        acknowledged_.store(view_->generation, std::memory_order_release);

    const float predelayMs = mapParam(t[1], p[1]);
    predelayTarget_ = std::clamp(predelayMs * 1e-3f * static_cast<float>(sampleRate_), 0.0f,
                                 static_cast<float>(predelaySize_ - 2));
    keepTarget_ = mapParam(t[2], p[2]) * 0.01f;
    fadeTarget_ = mapParam(t[3], p[3]) * 0.01f;
    lowCutTarget_ = mapParam(t[4], p[4]);
    highCutTarget_ = mapParam(t[5], p[5]);
    const float width = mapParam(t[6], p[6]);
    const float gainDb = mapParam(t[7], p[7]);
    const float mix = mapParam(t[8], p[8]);

    const int ramp = first_ ? 0 : kControlBlock;
    if (first_)
    {
        predelaySamples_ = predelayTarget_;
        keepFrac_ = keepTarget_;
        fadeFrac_ = fadeTarget_;
        lowCutHz_ = lowCutTarget_;
        highCutHz_ = highCutTarget_;
    }
    else
    {
        keepFrac_ += (keepTarget_ - keepFrac_) * blockSmooth_;
        fadeFrac_ += (fadeTarget_ - fadeFrac_) * blockSmooth_;
        if (std::abs(keepTarget_ - keepFrac_) < 1e-4f)
            keepFrac_ = keepTarget_;
        if (std::abs(fadeTarget_ - fadeFrac_) < 1e-4f)
            fadeFrac_ = fadeTarget_;
        // Cutoffs glide in the log domain.
        lowCutHz_ *= std::pow(lowCutTarget_ / lowCutHz_, blockSmooth_);
        highCutHz_ *= std::pow(highCutTarget_ / highCutHz_, blockSmooth_);
    }
    for (auto& f : lowCut_)
        f.setup(lowCutHz_, 0.70710678, sampleRate_);
    for (auto& f : highCut_)
        f.setup(highCutHz_, 0.70710678, sampleRate_);
    updateTrim();

    width_.setTarget(width, ramp);
    gain_.setTarget(std::pow(10.0f, gainDb / 20.0f), ramp);
    mix_.setTarget(mix, ramp);
    first_ = false;
}

void ConvolutionProcessor::processChunk(float* const* ch, int n) noexcept
{
    const bool stereo = channels_ > 1;
    for (int c = 0; c < channels_; ++c)
        std::copy_n(ch[c], n, dry_[idx(c)].data());

    // Pre-delay.
    float* lines[2] = {predelay_.data(), predelay_.data() + predelaySize_};
    for (int i = 0; i < n; ++i)
    {
        predelaySamples_ += (predelayTarget_ - predelaySamples_) * predelaySmooth_;
        if (std::abs(predelayTarget_ - predelaySamples_) < 1e-3f)
            predelaySamples_ = predelayTarget_;
        const auto whole = static_cast<int>(predelaySamples_);
        const float frac = predelaySamples_ - static_cast<float>(whole);
        for (int c = 0; c < channels_; ++c)
        {
            float* line = lines[c];
            line[writePos_] = dry_[idx(c)][idx(i)];
            const float a = line[(writePos_ - whole) & predelayMask_];
            const float b = line[(writePos_ - whole - 1) & predelayMask_];
            delayed_[idx(c)][idx(i)] = a + frac * (b - a);
        }
        writePos_ = (writePos_ + 1) & predelayMask_;
    }

    const float* in[2] = {delayed_[0].data(), delayed_[1].data()};
    for (int e = 0; e < 2; ++e)
    {
        float* out[2] = {wet_[idx(e)][0].data(), wet_[idx(e)][1].data()};
        engines_[idx(e)].process(in, out, n); // zeros when the engine has no IR
    }

    const auto now = static_cast<std::size_t>(active_);
    const auto old = static_cast<std::size_t>(1 - active_);
    const bool nowWet = engines_[now].ir() != nullptr;
    const bool oldWet = engines_[old].ir() != nullptr;
    for (int i = 0; i < n; ++i)
    {
        const auto s = idx(i);
        if (fading_)
            fadePos_ = std::min(1.0f, fadePos_ + fadeStep_);
        const float wNow = fadePos_;
        const float wOld = 1.0f - fadePos_;
        // Share of the output that is plain dry signal because its engine has no IR.
        const float pass = (nowWet ? 0.0f : wNow) + (oldWet ? 0.0f : wOld);
        float wet[2] = {0.0f, 0.0f};
        for (int c = 0; c < channels_; ++c)
        {
            float w = wNow * wet_[now][idx(c)][s] + wOld * wet_[old][idx(c)][s];
            w = highCut_[idx(c)].tick(lowCut_[idx(c)].tick(w).hp).lp;
            wet[c] = w;
        }
        const float width = width_.next();
        if (stereo)
            applyWidth(wet[0], wet[1], width);
        const float g = gain_.next();
        const float m = mix_.next();
        const float dryGain = pass + (1.0f - pass) * (1.0f - m);
        const float wetGain = m * g;
        for (int c = 0; c < channels_; ++c)
            ch[c][i] = dryGain * dry_[idx(c)][s] + wetGain * wet[c];
    }
    if (fading_ && fadePos_ >= 1.0f)
    {
        fading_ = false;
        engines_[old].setIr(nullptr);
    }
}

void ConvolutionProcessor::process(float* const* ch, int n) noexcept
{
    detail::ScopedNoDenormals noDenormals;
    for (int done = 0; done < n;)
    {
        const int k = std::min(kChunk, n - done);
        float* chunk[2] = {ch[0] + done, channels_ > 1 ? ch[1] + done : nullptr};
        processChunk(chunk, k);
        bool ok = true;
        for (int c = 0; c < channels_; ++c)
            ok = ok && allFinite(chunk[c], k);
        if (!ok)
        {
            clearState();
            for (int c = 0; c < channels_; ++c)
                std::fill_n(chunk[c], k, 0.0f);
        }
        done += k;
    }
}

int ConvolutionProcessor::tailSamples() const noexcept
{
    const int ir = std::max(engines_[0].activeLength(), engines_[1].activeLength());
    const float pre = std::max(predelaySamples_, predelayTarget_);
    // Plus a little for the wet filters to ring down.
    return ir + static_cast<int>(std::ceil(pre)) + static_cast<int>(0.02 * sampleRate_);
}

} // namespace alfa::dsp
