#include "alfa/dsp/PitchTime.h"

#include "alfa/dsp/detail/FastMath.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace alfa::dsp
{

namespace
{

constexpr double kPiD = 3.14159265358979323846;

std::size_t u(long long i) noexcept
{
    return static_cast<std::size_t>(i);
}

float dbToGain(float db) noexcept
{
    return std::pow(10.0f, db * 0.05f);
}

} // namespace

// PitchShiftProcessor -------------------------------------------------------------------------

const ParamTable& PitchShiftProcessor::params() noexcept
{
    static const ParamTable table = []
    {
        ParamTable t{};
        t[0] = {"shift", -24.0f, 24.0f, Curve::Linear, 0.5f, "st"};
        t[1] = {"fine", -100.0f, 100.0f, Curve::Linear, 0.5f, "cents"};
        t[2] = {"grain", 10.0f, 100.0f, Curve::Log, 0.69897f, "ms"}; // 50 ms
        t[3] = {"feedback", 0.0f, 0.9f, Curve::Linear, 0.0f, ""};
        t[4] = {"feedback tone", 500.0f, 20000.0f, Curve::Log, 1.0f, "Hz"};
        t[5] = {"spread", 0.0f, 50.0f, Curve::Linear, 0.0f, "cents"};
        t[6] = {"align", 0.0f, 1.0f, Curve::Toggle, 1.0f, ""};
        t[7] = {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""};
        t[8] = {"output gain", -24.0f, 24.0f, Curve::Linear, 0.5f, "dB"};
        return t;
    }();
    return table;
}

void PitchShiftProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    channels_ = std::clamp(spec.channels, 1, 2);
    margin_ = 4;
    searchRange_ = static_cast<int>(std::lround(0.012 * sampleRate_));
    corrLength_ = static_cast<int>(std::lround(0.005 * sampleRate_));
    corrStride_ = std::max(1, static_cast<int>(std::lround(sampleRate_ / 16000.0)));

    // Longest delay: a 100 ms grain at +25.5 semitones starts (p - 1) * 100 ms back, plus the
    // search window and the correlation segment behind it.
    const auto size = detail::nextPowerOfTwo(static_cast<int>(std::ceil(0.5 * sampleRate_)) + 1024);
    mask_ = size - 1;
    for (auto& c : ch_)
        c.buffer.assign(static_cast<std::size_t>(size), 0.0f);

    std::array<float, kSlotParams> defaults{};
    for (std::size_t i = 0; i < defaults.size(); ++i)
        defaults[i] = params()[i].defaultNorm;
    setParams(defaults.data(), {});
    reset();
}

void PitchShiftProcessor::reset() noexcept
{
    time_ = 0;
    for (int i = 0; i < 2; ++i)
    {
        auto& c = ch_[static_cast<std::size_t>(i)];
        std::fill(c.buffer.begin(), c.buffer.end(), 0.0f);
        for (auto& g : c.grains)
            g = Grain{};
        c.next = 0;
        c.untilNext = 0;
        c.semitones = shiftTarget_ + (channels_ == 2 ? (i == 0 ? -spreadTarget_ : spreadTarget_) : 0.0);
        c.feedback = 0.0f;
    }
    grainSeconds_ = grainTarget_;
    feedbackAmount_ = feedbackTarget_;
    mix_ = mixTarget_;
    gain_ = gainTarget_;
}

void PitchShiftProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const auto& t = params();
    auto get = [&](int i)
    {
        return mapParam(t[static_cast<std::size_t>(i)], p[i]);
    };
    shiftTarget_ = static_cast<double>(get(0)) + static_cast<double>(get(1)) * 0.01;
    grainTarget_ = static_cast<double>(get(2)) * 0.001;
    feedbackTarget_ = std::clamp(get(3), 0.0f, 0.9f);
    toneCoeff_ = static_cast<float>(1.0 - std::exp(-2.0 * kPiD * static_cast<double>(get(4)) / sampleRate_));
    spreadTarget_ = static_cast<double>(get(5)) * 0.01;
    align_ = get(6) >= 0.5f;
    mixTarget_ = get(7);
    gainTarget_ = dbToGain(get(8));
}

float PitchShiftProcessor::read(const Channel& c, double position) const noexcept
{
    // 4-point cubic Hermite (Catmull-Rom).
    const auto base = static_cast<long long>(std::floor(position));
    const auto f = static_cast<float>(position - static_cast<double>(base));
    const float* b = c.buffer.data();
    const float xm = b[u((base - 1) & mask_)];
    const float x0 = b[u(base & mask_)];
    const float x1 = b[u((base + 1) & mask_)];
    const float x2 = b[u((base + 2) & mask_)];
    const float c1 = 0.5f * (x1 - xm);
    const float c2 = xm - 2.5f * x0 + 2.0f * x1 - 0.5f * x2;
    const float c3 = 0.5f * (x2 - xm) + 1.5f * (x0 - x1);
    return ((c3 * f + c2) * f + c1) * f + x0;
}

double PitchShiftProcessor::alignedStart(const Channel& c, const Grain& other, double minDelay) const noexcept
{
    const auto now = static_cast<double>(time_);
    const double fallback = now - (minDelay + 0.5 * searchRange_);
    const double oldDelay = now - other.read;
    const int lo = static_cast<int>(std::ceil(minDelay - oldDelay));
    const int hi = lo + searchRange_;
    const int len = corrLength_;
    const int stride = corrStride_;
    const float* b = c.buffer.data();
    const long long refStart = static_cast<long long>(std::floor(other.read)) - len;
    auto at = [&](long long i)
    {
        return b[u(i & mask_)];
    };

    double refEnergy = 0.0;
    for (int i = 0; i < len; i += stride)
        refEnergy += static_cast<double>(at(refStart + i)) * at(refStart + i);
    if (refEnergy < 1e-12)
        return fallback;

    // Normalised cross-correlation of the segment the fading grain just played with the
    // segment that would precede each candidate start: the best candidate continues the
    // waveform in phase.
    auto score = [&](int m, int step)
    {
        double num = 0.0;
        double den = 1e-20;
        const long long s = refStart - m;
        for (int i = 0; i < len; i += step)
        {
            const double x = at(s + i);
            num += x * at(refStart + i);
            den += x * x;
        }
        return num / std::sqrt(den);
    };

    int best = lo;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int m = lo; m <= hi; m += stride)
    {
        const double sc = score(m, stride);
        if (sc > bestScore)
        {
            bestScore = sc;
            best = m;
        }
    }
    const int coarse = best;
    bestScore = -std::numeric_limits<double>::infinity();
    for (int m = std::max(lo, coarse - stride + 1); m <= std::min(hi, coarse + stride - 1); ++m)
    {
        const double sc = score(m, 1);
        if (sc > bestScore)
        {
            bestScore = sc;
            best = m;
        }
    }
    double delta = 0.0;
    if (best > lo && best < hi)
    {
        const double y0 = score(best - 1, 1);
        const double y2 = score(best + 1, 1);
        const double den = y0 - 2.0 * bestScore + y2;
        if (den < 0.0)
            delta = std::clamp(0.5 * (y0 - y2) / den, -0.5, 0.5);
    }
    return other.read - (best + delta);
}

void PitchShiftProcessor::startGrain(Channel& c, double ratio) noexcept
{
    const int length = std::max(16, static_cast<int>(std::lround(grainSeconds_ * sampleRate_)));
    Grain& g = c.grains[static_cast<std::size_t>(c.next)];
    const Grain& other = c.grains[static_cast<std::size_t>(c.next ^ 1)];
    // Delay at the grain start such that the read head never passes the write head (p > 1),
    // with a little slack for the ratio gliding upwards during the grain.
    const double minDelay = margin_ + (ratio > 1.0 ? (ratio - 1.0) * length * 1.02 : 0.0);
    const double search = align_ ? searchRange_ : 0.0;
    const auto now = static_cast<double>(time_);

    g.age = 0;
    g.length = length;
    g.active = true;
    if (!other.active)
    {
        // First grain after reset: start it at its peak so the output is at full level at once;
        // the next grain starts on the following sample.
        g.age = length / 2;
        g.read = now - (minDelay + 0.5 * search - (ratio - 1.0) * g.age);
        c.untilNext = 1;
    }
    else
    {
        g.read = align_ ? alignedStart(c, other, minDelay) : now - minDelay;
        c.untilNext = length / 2;
    }
    c.next ^= 1;
}

void PitchShiftProcessor::process(float* const* channels, int n) noexcept
{
    if (n <= 0)
        return;
    bool finite = true;
    for (int c = 0; c < channels_; ++c)
        for (int t = 0; t < n; ++t)
            finite = finite && std::isfinite(channels[c][t]);
    if (!finite)
    {
        reset();
        for (int c = 0; c < channels_; ++c)
            std::fill_n(channels[c], n, 0.0f);
        return;
    }

    const double glide = 1.0 - std::exp(-n / (0.02 * sampleRate_));
    const double grainGlide = 1.0 - std::exp(-n / (0.1 * sampleRate_));
    grainSeconds_ += grainGlide * (grainTarget_ - grainSeconds_);
    const float inv = 1.0f / static_cast<float>(n);
    const long long start = time_;

    for (int ci = 0; ci < channels_; ++ci)
    {
        Channel& c = ch_[static_cast<std::size_t>(ci)];
        float* x = channels[ci];
        const double target =
            shiftTarget_ + (channels_ == 2 ? (ci == 0 ? -spreadTarget_ : spreadTarget_) : 0.0);
        c.semitones += glide * (target - c.semitones);
        const double ratio = std::exp2(c.semitones / 12.0);
        time_ = start;
        for (int t = 0; t < n; ++t, ++time_)
        {
            const float a = static_cast<float>(t + 1) * inv;
            const float fb = feedbackAmount_ + a * (feedbackTarget_ - feedbackAmount_);
            const float dry = x[t];
            c.buffer[u(time_ & mask_)] = dry + fb * c.feedback;

            if (c.untilNext <= 0)
                startGrain(c, ratio);
            --c.untilNext;

            const auto newest = static_cast<double>(time_) - 2.0;
            const double oldest = static_cast<double>(time_ - mask_) + 4.0;
            float wet = 0.0f;
            float weights = 0.0f;
            for (auto& g : c.grains)
            {
                if (!g.active)
                    continue;
                const float s =
                    detail::sin2pi(0.5f * static_cast<float>(g.age) / static_cast<float>(g.length));
                const float w = s * s;
                g.read = std::clamp(g.read, oldest, newest);
                wet += w * read(c, g.read);
                weights += w;
                g.read += ratio;
                if (++g.age >= g.length)
                    g.active = false;
            }
            // The two windows sum to 1 in steady state; dividing covers grain-length changes.
            wet /= std::max(weights, 0.25f);

            c.feedback += toneCoeff_ * (std::tanh(wet) - c.feedback);
            if (std::fabs(c.feedback) < 1e-20f)
                c.feedback = 0.0f;
            const float mix = mix_ + a * (mixTarget_ - mix_);
            const float gain = gain_ + a * (gainTarget_ - gain_);
            x[t] = gain * (mix * wet + (1.0f - mix) * dry);
        }
    }
    time_ = start + n;
    feedbackAmount_ = feedbackTarget_;
    mix_ = mixTarget_;
    gain_ = gainTarget_;

    for (int c = 0; c < channels_; ++c)
        for (int t = 0; t < n; ++t)
            finite = finite && std::isfinite(channels[c][t]);
    if (!finite)
    {
        reset();
        for (int c = 0; c < channels_; ++c)
            std::fill_n(channels[c], n, 0.0f);
    }
}

int PitchShiftProcessor::latencySamples() const noexcept
{
    const double ratio = std::exp2(shiftTarget_ / 12.0);
    const double length = grainSeconds_ * sampleRate_;
    return static_cast<int>(margin_ + (align_ ? 0.5 * searchRange_ : 0.0) +
                            0.5 * std::fabs(ratio - 1.0) * length);
}

int PitchShiftProcessor::tailSamples() const noexcept
{
    const int latency = latencySamples();
    const auto length = static_cast<int>(grainSeconds_ * sampleRate_);
    if (feedbackTarget_ <= 0.0f)
        return latency + length;
    // Each pass round the loop is attenuated by at least the feedback amount.
    const double passes = std::ceil(std::log(1e-3) / std::log(static_cast<double>(feedbackTarget_)));
    return static_cast<int>((passes + 1.0) * (latency + length));
}

// Offline -------------------------------------------------------------------------------------

namespace offline
{

namespace
{

/// Modified Bessel function of the first kind, order 0 (power series; converges fast for the
/// Kaiser betas used here). Written out because std::cyl_bessel_i is missing on some platforms.
double besselI0(double x)
{
    double sum = 1.0;
    double term = 1.0;
    const double q = 0.25 * x * x;
    for (int k = 1; k < 64; ++k)
    {
        term *= q / (static_cast<double>(k) * k);
        sum += term;
        if (term < 1e-17 * sum)
            break;
    }
    return sum;
}

} // namespace

std::vector<float> timeStretch(const float* in, std::size_t n, double ratio, int fftSize)
{
    ratio = std::isfinite(ratio) ? std::clamp(ratio, 0.25, 8.0) : 1.0;
    const int size = std::clamp(detail::nextPowerOfTwo(std::max(fftSize, 1)), 256, 16384);
    const auto outLength = static_cast<std::size_t>(std::llround(static_cast<double>(n) * ratio));
    std::vector<float> out(outLength, 0.0f);
    if (n == 0 || outLength == 0)
        return out;

    // Synthesis hop n/4 (n/8 when compressing, so the analysis hop stays <= n/2 and the phase
    // unwrapping range stays at least +-1 bin).
    const int hopOut = ratio >= 1.0 ? size / 4 : size / 8;
    const double hopIn = hopOut / ratio;
    const int half = size / 2;

    detail::RealFft fft;
    fft.prepare(size);
    detail::PhaseVocoderFrame pv;
    pv.prepare(size);
    detail::AlignedBuffer frame(static_cast<std::size_t>(size));
    std::vector<float> window(static_cast<std::size_t>(size));
    double overlap = 0.0;
    for (int t = 0; t < size; ++t)
    {
        const double w = 0.5 - 0.5 * std::cos(2.0 * kPiD * t / size);
        window[static_cast<std::size_t>(t)] = static_cast<float>(w);
        overlap += w * w;
    }
    const auto scale = static_cast<float>(1.0 / (overlap / hopOut * size));

    const auto frames = static_cast<long long>((static_cast<double>(outLength) + half) / hopOut) + 2;
    long long prevStart = 0;
    for (long long j = 0; j < frames; ++j)
    {
        // Frames are centred: analysis centre j * hopIn, synthesis centre j * hopOut.
        const long long inStart = std::llround(static_cast<double>(j) * hopIn) - half;
        const long long outStart = j * hopOut - half;
        for (int t = 0; t < size; ++t)
        {
            const long long i = inStart + t;
            const float x = (i >= 0 && i < static_cast<long long>(n)) ? in[i] : 0.0f;
            frame[static_cast<std::size_t>(t)] = x * window[static_cast<std::size_t>(t)];
        }
        fft.forward(frame.data(), frame.data());
        pv.analyse(frame.data(), size, j == 0 ? 0.0 : static_cast<double>(inStart - prevStart));
        pv.synthesise(frame.data(), hopOut, true);
        fft.inverse(frame.data(), frame.data());
        for (int t = 0; t < size; ++t)
        {
            const long long o = outStart + t;
            if (o >= 0 && o < static_cast<long long>(outLength))
                out[static_cast<std::size_t>(o)] +=
                    frame[static_cast<std::size_t>(t)] * window[static_cast<std::size_t>(t)] * scale;
        }
        prevStart = inStart;
    }
    return out;
}

std::vector<float> resample(const float* in, std::size_t n, double step, std::size_t outLength)
{
    std::vector<float> out(outLength, 0.0f);
    if (n == 0 || !(step > 0.0) || !std::isfinite(step))
        return out;
    constexpr double kZeroCrossings = 24.0;
    constexpr double kBeta = 8.6;
    const double cutoff = std::min(1.0, 1.0 / step) * 0.97; // a little guard band below Nyquist
    const double halfWidth = kZeroCrossings / cutoff;
    const double norm = 1.0 / besselI0(kBeta);
    for (std::size_t i = 0; i < outLength; ++i)
    {
        const double pos = static_cast<double>(i) * step;
        const auto first = static_cast<long long>(std::ceil(pos - halfWidth));
        const auto last = static_cast<long long>(std::floor(pos + halfWidth));
        double acc = 0.0;
        for (long long j = std::max(0LL, first); j <= std::min(static_cast<long long>(n) - 1, last); ++j)
        {
            const double x = pos - static_cast<double>(j);
            const double r = x / halfWidth;
            const double window = besselI0(kBeta * std::sqrt(std::max(0.0, 1.0 - r * r))) * norm;
            const double arg = kPiD * cutoff * x;
            const double sinc = std::fabs(arg) < 1e-12 ? 1.0 : std::sin(arg) / arg;
            acc += static_cast<double>(in[j]) * cutoff * sinc * window;
        }
        out[i] = static_cast<float>(acc);
    }
    return out;
}

std::vector<float> pitchShiftOffline(const float* in, std::size_t n, double semitones, int fftSize)
{
    const double ratio =
        std::exp2(std::isfinite(semitones) ? std::clamp(semitones, -24.0, 24.0) / 12.0 : 0.0);
    const auto stretched = timeStretch(in, n, ratio, fftSize);
    return resample(stretched.data(), stretched.size(), ratio, n);
}

} // namespace offline

} // namespace alfa::dsp
