#include "alfa/dsp/Voice.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace alfa::dsp
{
namespace
{
constexpr float kMinAttackTau = 1.0e-4f;
constexpr float kSilenceThreshold = 1.0e-5f; // -100 dBFS peak
constexpr int kSilentHopsToEnd = 25;         // 100 ms after release
constexpr double kMaxReleaseSeconds = 30.0;
} // namespace

Voice::Voice()
    : latest_(std::make_unique<VoiceFrame>())
    , frameA_(std::make_unique<VoiceFrame>())
    , frameB_(std::make_unique<VoiceFrame>())
{
    prev_ = frameA_.get();
    cur_ = frameB_.get();
}

Voice::~Voice() = default;

void Voice::prepare(const PrepareSpec& spec, std::uint32_t seed)
{
    spec_ = spec;
    hop_ = spec.hop;
    osc_.prepare(spec);
    partials_.prepare(spec);
    fm_.prepare(spec);
    noise_.prepare(spec);
    noise_.setSeed(seed);
    grain_.prepare(spec);
    sampler_.prepare(spec);
    scratch_.assign(static_cast<std::size_t>(2 * hop_), 0.0f);
    active_ = false;
}

void Voice::setGrainBank(const GrainBank* bank) noexcept
{
    grain_.setBank(bank);
}

void Voice::setSample(const SampleData* sample) noexcept
{
    sampler_.setSample(sample);
}

void Voice::start(int note, float velocity, float f0Hz, const VoiceFrame* firstFrame) noexcept
{
    note_ = note;
    velocity_ = velocity;
    f0Target_ = f0Hz;
    f0Last_ = f0Hz;
    active_ = true;
    releasing_ = false;
    age_ = 0;
    releaseAge_ = 0;
    silentHops_ = 0;
    hopPeak_ = 0.0f;
    firstHop_ = true;
    hopPos_ = hop_; // the first process call starts hop 0 immediately
    haveFrame_ = firstFrame != nullptr;
    if (firstFrame != nullptr)
        *latest_ = *firstFrame;

    osc_.reset();
    partials_.reset();
    fm_.reset();
    noise_.reset();
    grain_.reset();
    sampler_.reset();
}

void Voice::setFrame(const VoiceFrame& frame) noexcept
{
    std::memcpy(static_cast<void*>(latest_.get()), &frame, sizeof(VoiceFrame));
    haveFrame_ = true;
}

void Voice::beginHop() noexcept
{
    // Frame j-1 <- frame j, frame j <- latest.
    std::swap(prev_, cur_);
    std::memcpy(static_cast<void*>(cur_), latest_.get(), sizeof(VoiceFrame));
    if (firstHop_)
        std::memcpy(static_cast<void*>(prev_), cur_, sizeof(VoiceFrame));

    HopContext ctx;
    ctx.prev = prev_;
    ctx.cur = cur_;
    ctx.f0Start = firstHop_ ? f0Target_ : f0Last_;
    ctx.f0End = f0Target_;
    ctx.firstHop = firstHop_;
    f0Last_ = f0Target_;

    if (firstHop_)
    {
        const float tau = std::max(cur_->attackTau, kMinAttackTau);
        attackDecay_ = static_cast<float>(std::exp(-1.0 / (static_cast<double>(tau) * spec_.sampleRate)));
        attackEnv_ = 1.0f;
    }

    const float invHop = 1.0f / static_cast<float>(hop_);
    for (int p = 0; p < kPrimitiveCount; ++p)
    {
        const auto i = static_cast<std::size_t>(p);
        const float a = prev_->mix[i];
        const float b = cur_->mix[i];
        rendering_[i] = a != 0.0f || b != 0.0f;
        mixFrom_[i] = a;
        mixStep_[i] = (b - a) * invHop;
    }

    const auto dispatch = [&](auto& prim, Primitive p)
    {
        if (rendering_[static_cast<std::size_t>(p)])
            prim.beginHop(ctx);
        else
            prim.skipHop(ctx);
    };
    dispatch(osc_, Primitive::Osc);
    dispatch(partials_, Primitive::Partials);
    dispatch(fm_, Primitive::Fm);
    dispatch(noise_, Primitive::Noise);
    dispatch(grain_, Primitive::Grain);
    dispatch(sampler_, Primitive::Sampler);

    firstHop_ = false;
    hopPos_ = 0;
}

void Voice::renderChunk(float* out, int n) noexcept
{
    // Chunks never cross a hop boundary, so n <= hop_ and both scratch halves fit.
    float* tmp = scratch_.data();
    float* acc = scratch_.data() + hop_;
    std::fill_n(acc, n, 0.0f);

    const auto mixIn = [&](auto& prim, Primitive p)
    {
        const auto i = static_cast<std::size_t>(p);
        if (!rendering_[i])
            return;
        prim.render(tmp, n);
        const float g0 = mixFrom_[i] + mixStep_[i] * static_cast<float>(hopPos_);
        const float dg = mixStep_[i];
        for (int t = 0; t < n; ++t)
            acc[t] += tmp[t] * (g0 + dg * static_cast<float>(t));
    };
    mixIn(osc_, Primitive::Osc);
    mixIn(partials_, Primitive::Partials);
    mixIn(fm_, Primitive::Fm);
    mixIn(noise_, Primitive::Noise);
    mixIn(grain_, Primitive::Grain);
    mixIn(sampler_, Primitive::Sampler);

    float peak = hopPeak_;
    if (attackEnv_ > 1.0e-7f)
    {
        float e = attackEnv_;
        for (int t = 0; t < n; ++t)
        {
            const float y = acc[t] * (1.0f - e);
            e *= attackDecay_;
            out[t] += y;
            peak = std::max(peak, std::fabs(y));
        }
        attackEnv_ = e;
    }
    else
    {
        for (int t = 0; t < n; ++t)
        {
            out[t] += acc[t];
            peak = std::max(peak, std::fabs(acc[t]));
        }
    }
    hopPeak_ = peak;
}

void Voice::process(float* out, int n) noexcept
{
    if (!active_)
        return;
    int done = 0;
    while (done < n)
    {
        if (hopPos_ >= hop_)
        {
            if (!haveFrame_)
                return; // waiting for the note-on frame
            if (!firstHop_ && releasing_)
            {
                const bool allSilent =
                    std::all_of(cur_->mix.begin(), cur_->mix.end(), [](float g) { return g == 0.0f; }) &&
                    std::all_of(latest_->mix.begin(), latest_->mix.end(), [](float g) { return g == 0.0f; });
                silentHops_ = hopPeak_ < kSilenceThreshold ? silentHops_ + 1 : 0;
                const bool tooLong = static_cast<double>(releaseAge_) > kMaxReleaseSeconds * spec_.sampleRate;
                if (allSilent || silentHops_ >= kSilentHopsToEnd || tooLong)
                {
                    active_ = false;
                    return;
                }
            }
            hopPeak_ = 0.0f;
            beginHop();
        }
        const int chunk = std::min(n - done, hop_ - hopPos_);
        renderChunk(out + done, chunk);
        done += chunk;
        hopPos_ += chunk;
        age_ += chunk;
        if (releasing_)
            releaseAge_ += chunk;
    }
}

} // namespace alfa::dsp
