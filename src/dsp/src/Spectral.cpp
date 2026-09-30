#include "alfa/dsp/Spectral.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace alfa::dsp
{

namespace
{

constexpr double kPiD = 3.14159265358979323846;
constexpr float kTiny = 1e-20f; // below this a magnitude counts as silence

std::size_t u(int i) noexcept
{
    return static_cast<std::size_t>(i);
}

int orderOf(int n) noexcept
{
    int order = 0;
    while ((1 << order) < n)
        ++order;
    return order;
}

float dbToGain(float db) noexcept
{
    return std::pow(10.0f, db * 0.05f);
}

} // namespace

namespace spectral
{

void magnitudes(const float* s, int fftSize, float* mag) noexcept
{
    const int half = fftSize / 2;
    mag[0] = std::fabs(s[0]);
    mag[half] = std::fabs(s[1]);
    for (int k = 1; k < half; ++k)
        mag[k] = std::sqrt(s[2 * k] * s[2 * k] + s[2 * k + 1] * s[2 * k + 1]);
}

// Freeze --------------------------------------------------------------------------------------

void Freeze::prepare(int maxFftSize)
{
    const auto bins = u(maxFftSize / 2 + 1);
    prev_.assign(u(maxFftSize), 0.0f);
    mag_.assign(bins, 0.0f);
    phase_.assign(bins, 0.0f);
    advance_.assign(bins, 0.0f);
    reset();
}

void Freeze::reset() noexcept
{
    havePrev_ = false;
    captured_ = false;
}

void Freeze::setAmount(float amount) noexcept
{
    amount_ = std::isfinite(amount) ? std::clamp(amount, 0.0f, 1.0f) : 0.0f;
}

void Freeze::apply(float* s, const FrameInfo& info) noexcept
{
    const int n = info.fftSize;
    const int half = n / 2;
    if (!active())
    {
        captured_ = false;
        std::copy_n(s, n, prev_.data());
        havePrev_ = true;
        return;
    }

    if (!captured_)
    {
        // Capture: magnitude and phase now, frequency from the phase advance since the previous
        // frame, so the held partials keep their exact pitch.
        const double perBin = 2.0 * kPiD * info.hop / n;
        mag_[0] = s[0];
        mag_[u(half)] = s[1];
        advance_[0] = 0.0f;
        advance_[u(half)] = 0.0f; // pi * hop, a whole number of turns since the hop is n / 4
        for (int k = 1; k < half; ++k)
        {
            const auto i = u(k);
            const float re = s[2 * k];
            const float im = s[2 * k + 1];
            mag_[i] = std::sqrt(re * re + im * im);
            phase_[i] = std::atan2(im, re);
            const auto expected = static_cast<float>(std::fmod(perBin * k, 2.0 * kPiD));
            float advance = expected;
            if (havePrev_)
            {
                const float prevPhase = std::atan2(prev_[u(2 * k + 1)], prev_[u(2 * k)]);
                advance += detail::wrapPhase(phase_[i] - prevPhase - expected);
            }
            advance_[i] = advance;
        }
        captured_ = true;
    }
    else
    {
        for (int k = 1; k < half; ++k)
            phase_[u(k)] = detail::wrapPhase(phase_[u(k)] + advance_[u(k)]);
    }

    std::copy_n(s, n, prev_.data());
    havePrev_ = true;

    const float a = amount_;
    const float b = 1.0f - a;
    s[0] = b * s[0] + a * mag_[0];
    s[1] = b * s[1] + a * mag_[u(half)];
    for (int k = 1; k < half; ++k)
    {
        const auto i = u(k);
        s[2 * k] = b * s[2 * k] + a * mag_[i] * std::cos(phase_[i]);
        s[2 * k + 1] = b * s[2 * k + 1] + a * mag_[i] * std::sin(phase_[i]);
    }
}

// Smear ---------------------------------------------------------------------------------------

void Smear::prepare(int maxFftSize)
{
    state_.assign(u(maxFftSize / 2 + 1), 0.0f);
    reset();
}

void Smear::setTime(float seconds) noexcept
{
    seconds_ = std::isfinite(seconds) ? std::max(seconds, 0.0f) : 0.0f;
}

void Smear::apply(float* mag, const FrameInfo& info) noexcept
{
    const int bins = info.bins();
    if (!active())
    {
        primed_ = false;
        return;
    }
    if (!primed_)
    {
        std::copy_n(mag, bins, state_.data());
        primed_ = true;
        return;
    }
    const auto c =
        static_cast<float>(1.0 - std::exp(-info.hop / (static_cast<double>(seconds_) * info.sampleRate)));
    for (int k = 0; k < bins; ++k)
    {
        float& st = state_[u(k)];
        st += c * (mag[k] - st);
        if (st < kTiny)
            st = 0.0f; // decaying state must not reach denormals
        mag[k] = st;
    }
}

// Blur ----------------------------------------------------------------------------------------

void Blur::prepare(int maxFftSize)
{
    prefix_.assign(u(maxFftSize / 2 + 2), 0.0);
}

void Blur::setRadius(float bins) noexcept
{
    radius_ = std::isfinite(bins) ? std::clamp(bins, 0.0f, 256.0f) : 0.0f;
}

void Blur::apply(float* mag, const FrameInfo& info) noexcept
{
    if (!active())
        return;
    const int bins = info.bins();
    prefix_[0] = 0.0;
    for (int k = 0; k < bins; ++k)
        prefix_[u(k + 1)] = prefix_[u(k)] + mag[k];

    const int r0 = static_cast<int>(radius_);
    const float frac = radius_ - static_cast<float>(r0);
    auto boxMean = [&](int k, int r)
    {
        const int lo = std::max(0, k - r);
        const int hi = std::min(bins - 1, k + r);
        return (prefix_[u(hi + 1)] - prefix_[u(lo)]) / (hi - lo + 1);
    };
    // The sum is double, so the running prefix does not lose the quiet bins next to loud ones.
    for (int k = 0; k < bins; ++k)
    {
        const double a = boxMean(k, r0);
        const double b = frac > 0.0f ? boxMean(k, r0 + 1) : a;
        mag[k] = static_cast<float>(a + frac * (b - a));
    }
}

// BandShape -----------------------------------------------------------------------------------

void BandShape::prepare(double sampleRate, int minFftSize, int maxFftSize)
{
    sampleRate_ = sampleRate;
    for (int order = orderOf(minFftSize); order <= orderOf(maxFftSize); ++order)
        maps_[u(order)].prepare(sampleRate, 1 << order);
    db_.fill(0.0f);
    tilt_ = 0.0f;
    update();
}

double BandShape::bandHz(int b) const noexcept
{
    return kBandLowHz * std::pow(0.5 * sampleRate_ / kBandLowHz, static_cast<double>(b) / (kBands - 1));
}

void BandShape::bump(float* db, float gainDb, float centreHz, float widthOctaves) const noexcept
{
    const double sigma = std::max(0.05, 0.5 * static_cast<double>(widthOctaves));
    for (int b = 0; b < kBands; ++b)
    {
        const double d = std::log2(bandHz(b) / std::max(1.0, static_cast<double>(centreHz))) / sigma;
        db[b] = static_cast<float>(gainDb * std::exp(-0.5 * d * d));
    }
}

void BandShape::setBandsDb(const float* db) noexcept
{
    for (int b = 0; b < kBands; ++b)
        db_[u(b)] = std::isfinite(db[b]) ? std::clamp(db[b], -96.0f, 48.0f) : 0.0f;
    update();
}

void BandShape::setTilt(float dbPerOctave) noexcept
{
    tilt_ = std::isfinite(dbPerOctave) ? std::clamp(dbPerOctave, -24.0f, 24.0f) : 0.0f;
    update();
}

void BandShape::update() noexcept
{
    active_ = tilt_ != 0.0f || std::any_of(db_.begin(), db_.end(), [](float d) { return d != 0.0f; });
    // The tilt is folded into the band gains, so a frame costs one interpolation per bin.
    for (int b = 0; b < kBands; ++b)
    {
        const auto octaves = static_cast<float>(std::log2(bandHz(b) / 1000.0));
        // Capped at +36 dB so band gain and a steep tilt cannot stack without bound.
        gain_[u(b)] = dbToGain(std::min(db_[u(b)] + tilt_ * octaves, 36.0f));
    }
}

void BandShape::apply(float* mag, const FrameInfo& info) const noexcept
{
    if (!active_)
        return;
    const auto& map = maps_[u(orderOf(info.fftSize))];
    if (map.bins() != info.bins())
        return; // size not prepared
    const int bins = info.bins();
    for (int k = 0; k < bins; ++k)
        mag[k] *= map.gain(gain_.data(), k);
}

// Gate ----------------------------------------------------------------------------------------

void Gate::setThreshold(float db) noexcept
{
    thresholdDb_ = std::isfinite(db) ? std::clamp(db, kOffDb, 0.0f) : kOffDb;
}

void Gate::setKnee(float db) noexcept
{
    kneeDb_ = std::isfinite(db) ? std::clamp(db, 0.0f, 48.0f) : 0.0f;
}

void Gate::apply(float* mag, const FrameInfo& info) const noexcept
{
    if (!active())
        return;
    const int bins = info.bins();
    const float peak = *std::max_element(mag, mag + bins);
    if (!(peak > kTiny))
        return;
    const float lowDb = thresholdDb_ - 0.5f * kneeDb_;
    const float lo = peak * dbToGain(lowDb);
    const float hi = peak * dbToGain(thresholdDb_ + 0.5f * kneeDb_);
    const float invPeak = 1.0f / peak;
    for (int k = 0; k < bins; ++k)
    {
        const float m = mag[k];
        float g = 1.0f;
        if (m < lo)
            g = 0.0f;
        else if (m < hi)
        {
            // Inside the knee: smoothstep in dB, so only these few bins pay for the log.
            const float x = std::clamp((20.0f * std::log10(m * invPeak) - lowDb) / kneeDb_, 0.0f, 1.0f);
            g = x * x * (3.0f - 2.0f * x);
        }
        mag[k] = m * (invert_ ? 1.0f - g : g);
    }
}

// Sparsify ------------------------------------------------------------------------------------

void Sparsify::prepare(int maxFftSize)
{
    scratch_.assign(u(maxFftSize / 2 + 1), 0.0f);
}

void Sparsify::setKeep(float fraction) noexcept
{
    keep_ = std::isfinite(fraction) ? std::clamp(fraction, 0.0f, 1.0f) : 1.0f;
}

void Sparsify::apply(float* mag, const FrameInfo& info) noexcept
{
    if (!active())
        return;
    const int bins = info.bins();
    const int count = std::clamp(static_cast<int>(std::ceil(keep_ * static_cast<float>(bins))), 1, bins);
    if (count >= bins)
        return;
    std::copy_n(mag, bins, scratch_.data());
    auto nth = scratch_.begin() + (count - 1);
    std::nth_element(scratch_.begin(), nth, scratch_.begin() + bins, std::greater<float>());
    const float threshold = *nth;
    for (int k = 0; k < bins; ++k)
        if (mag[k] < threshold)
            mag[k] = 0.0f;
}

// FrequencyShift ------------------------------------------------------------------------------

void FrequencyShift::setShift(float hz) noexcept
{
    hz_ = std::isfinite(hz) ? std::clamp(hz, -24000.0f, 24000.0f) : 0.0f;
    if (hz_ == 0.0f)
        rotation_ = 0.0;
}

void FrequencyShift::apply(float* s, const FrameInfo& info) noexcept
{
    if (!active())
        return;
    const int n = info.fftSize;
    const int half = n / 2;
    const double shiftBins = static_cast<double>(hz_) * n / info.sampleRate;
    const auto move = static_cast<int>(std::lround(shiftBins));

    // Bin k at frame j carries phase advance 2 pi k hop / n per frame; adding the shift's own
    // advance as a common rotation makes consecutive frames agree on f + shift even when the
    // shift is not a whole number of bins.
    const auto c = static_cast<float>(std::cos(rotation_));
    const auto sn = static_cast<float>(std::sin(rotation_));
    rotation_ = std::fmod(rotation_ + 2.0 * kPiD * shiftBins * info.hop / n, 2.0 * kPiD);

    auto put = [&](int dst, int src)
    {
        const float re = s[2 * src];
        const float im = s[2 * src + 1];
        s[2 * dst] = re * c - im * sn;
        s[2 * dst + 1] = re * sn + im * c;
    };
    auto clear = [&](int dst)
    {
        s[2 * dst] = 0.0f;
        s[2 * dst + 1] = 0.0f;
    };
    // DC and Nyquist are real and cannot carry a rotation: they are dropped.
    s[0] = 0.0f;
    s[1] = 0.0f;
    if (move >= 0)
    {
        for (int k = half - 1; k >= 1; --k)
        {
            if (k - move >= 1)
                put(k, k - move);
            else
                clear(k);
        }
    }
    else
    {
        for (int k = 1; k < half; ++k)
        {
            if (k - move < half)
                put(k, k - move);
            else
                clear(k);
        }
    }
}

// PhaseMorph ----------------------------------------------------------------------------------

void PhaseMorph::setAmount(float amount) noexcept
{
    amount_ = std::isfinite(amount) ? std::clamp(amount, -1.0f, 1.0f) : 0.0f;
}

void PhaseMorph::apply(float* s, const FrameInfo& info) noexcept
{
    if (!active())
        return;
    const int half = info.fftSize / 2;
    if (amount_ < 0.0f)
    {
        // Robotise: blend towards |X| with linear phase (-1)^k, which puts each frame's energy
        // at the frame centre where the synthesis window is 1 (zero phase would sit at the
        // frame edge, where it is windowed away).
        const float a = -amount_;
        const float b = 1.0f - a;
        s[0] = b * s[0] + a * std::fabs(s[0]);
        s[1] = b * s[1] + a * std::fabs(s[1]); // half is even, so (-1)^half = 1
        for (int k = 1; k < half; ++k)
        {
            const float re = s[2 * k];
            const float im = s[2 * k + 1];
            const float target = (k & 1) != 0 ? -std::sqrt(re * re + im * im) : std::sqrt(re * re + im * im);
            s[2 * k] = b * re + a * target;
            s[2 * k + 1] = b * im;
        }
        return;
    }
    // Whisperise: a fresh random phase offset of up to +-pi * amount per bin and frame.
    const float spread = 0.5f * amount_;
    for (int k = 1; k < half; ++k)
    {
        const float turns = spread * random_.nextBipolar();
        const float c = detail::sin2pi(turns + 0.25f);
        const float sn = detail::sin2pi(turns);
        const float re = s[2 * k];
        const float im = s[2 * k + 1];
        s[2 * k] = re * c - im * sn;
        s[2 * k + 1] = re * sn + im * c;
    }
}

} // namespace spectral

// SpectralProcessor ---------------------------------------------------------------------------

const ParamTable& SpectralProcessor::params() noexcept
{
    static const ParamTable table = []
    {
        ParamTable t{};
        t[0] = {"fft size", 0.0f, 2.0f, Curve::Choice, 0.5f, "512/1024/2048"};
        t[1] = {"band gain", -24.0f, 24.0f, Curve::Linear, 0.5f, "dB"};
        t[2] = {"band centre", 40.0f, 16000.0f, Curve::Log, 0.53724f, "Hz"}; // 1 kHz
        t[3] = {"band width", 0.25f, 8.0f, Curve::Log, 0.6f, "oct"};         // 2 octaves
        t[4] = {"tilt", -6.0f, 6.0f, Curve::Linear, 0.5f, "dB/oct"};
        t[5] = {"gate threshold", -96.0f, 0.0f, Curve::Linear, 0.0f, "dB"};
        t[6] = {"gate knee", 0.0f, 24.0f, Curve::Linear, 0.25f, "dB"};
        t[7] = {"gate invert", 0.0f, 1.0f, Curve::Toggle, 0.0f, ""};
        t[8] = {"sparsify keep", 1.0f, 0.01f, Curve::Log, 0.0f, "fraction"};
        t[9] = {"freeze", 0.0f, 1.0f, Curve::Linear, 0.0f, ""};
        t[10] = {"smear", 0.0f, 2.0f, Curve::Linear, 0.0f, "s"};
        t[11] = {"blur", 0.0f, 32.0f, Curve::Linear, 0.0f, "bins"};
        t[12] = {"frequency shift", -2000.0f, 2000.0f, Curve::Linear, 0.5f, "Hz"};
        t[13] = {"phase", -1.0f, 1.0f, Curve::Linear, 0.5f, "robot..whisper"};
        t[14] = {"mix", 0.0f, 1.0f, Curve::Linear, 1.0f, ""};
        t[15] = {"output gain", -24.0f, 24.0f, Curve::Linear, 0.5f, "dB"};
        return t;
    }();
    return table;
}

void SpectralProcessor::prepare(const ProcessSpec& spec)
{
    sampleRate_ = spec.sampleRate;
    channels_ = std::clamp(spec.channels, 1, 2);
    unsigned int seed = 0x5EC7A1u;
    for (auto& c : ch_)
    {
        c.stft.prepare(kMinFftSize, kMaxFftSize);
        c.freeze.prepare(kMaxFftSize);
        c.smear.prepare(kMaxFftSize);
        c.phase.setSeed(seed++);
        c.random.setSeed(seed++);
        c.dry.assign(u(2 * kMaxFftSize), 0.0f);
    }
    blur_.prepare(kMaxFftSize);
    band_.prepare(sampleRate_, kMinFftSize, kMaxFftSize);
    sparse_.prepare(kMaxFftSize);
    mag_.assign(u(kMaxFftSize / 2 + 1), 0.0f);
    mag0_.assign(u(kMaxFftSize / 2 + 1), 0.0f);

    std::array<float, kSlotParams> defaults{};
    for (std::size_t i = 0; i < defaults.size(); ++i)
        defaults[i] = params()[i].defaultNorm;
    bandValid_ = false;
    setParams(defaults.data(), {});
    mix_ = mixTarget_;
    gain_ = gainTarget_;
    reset();
}

void SpectralProcessor::reset() noexcept
{
    fftSize_ = pendingFftSize_;
    for (auto& c : ch_)
    {
        c.stft.setFftSize(fftSize_);
        c.freeze.reset();
        c.smear.reset();
        c.shift.reset();
        c.phase.reset();
        std::fill(c.dry.begin(), c.dry.end(), 0.0f);
        c.dryPos = 0;
    }
    mix_ = mixTarget_;
    gain_ = gainTarget_;
}

void SpectralProcessor::setParams(const float* p, const ControlContext&) noexcept
{
    const auto& t = params();
    auto get = [&](int i)
    {
        return mapParam(t[u(i)], p[i]);
    };

    pendingFftSize_ = kMinFftSize << static_cast<int>(get(0));

    const std::array<float, 4> band{get(1), get(2), get(3), get(4)};
    if (!bandValid_ || band != bandParams_)
    {
        std::array<float, kBands> db{};
        if (band[0] != 0.0f)
            band_.bump(db.data(), band[0], band[1], band[2]);
        band_.setBandsDb(db.data());
        band_.setTilt(band[3]);
        bandParams_ = band;
        bandValid_ = true;
    }

    gate_.setThreshold(get(5));
    gate_.setKnee(get(6));
    gate_.setInvert(get(7) >= 0.5f);
    // x = 0 must be exactly "keep everything", which the log curve gives (1 * 0.01^0).
    sparse_.setKeep(get(8));
    const float freeze = get(9);
    smearSeconds_ = get(10);
    blur_.setRadius(get(11));
    const float shift = get(12);
    const float phase = get(13);
    freezeOn_ = freeze > 0.0f;
    for (auto& c : ch_)
    {
        c.freeze.setAmount(freeze);
        c.smear.setTime(smearSeconds_);
        c.shift.setShift(shift);
        c.phase.setAmount(phase);
    }
    mixTarget_ = get(14);
    gainTarget_ = dbToGain(get(15));
}

void SpectralProcessor::processFrame(Channel& c, float* s) noexcept
{
    spectral::FrameInfo fi;
    fi.fftSize = fftSize_;
    fi.hop = c.stft.hop();
    fi.sampleRate = sampleRate_;
    const int bins = fi.bins();

    c.freeze.apply(s, fi);

    const bool magOps =
        c.smear.active() || blur_.active() || band_.active() || gate_.active() || sparse_.active();
    if (magOps)
    {
        float* m = mag_.data();
        float* m0 = mag0_.data();
        spectral::magnitudes(s, fftSize_, m);
        std::copy_n(m, bins, m0);
        c.smear.apply(m, fi);
        blur_.apply(m, fi);
        band_.apply(m, fi);
        gate_.apply(m, fi);
        sparse_.apply(m, fi);

        // Rescale the complex bins, keeping the analysed phase. A bin that was silent but gained
        // energy (smear tail, blur) gets a random phase, like a noise tail.
        const int half = fftSize_ / 2;
        s[0] = m0[0] > kTiny ? s[0] * (m[0] / m0[0]) : m[0];
        s[1] = m0[half] > kTiny ? s[1] * (m[half] / m0[half]) : m[half];
        for (int k = 1; k < half; ++k)
        {
            if (m0[k] > kTiny)
            {
                const float g = m[k] / m0[k];
                s[2 * k] *= g;
                s[2 * k + 1] *= g;
            }
            else if (m[k] > kTiny)
            {
                const float turns = 0.5f * c.random.nextBipolar();
                s[2 * k] = m[k] * detail::sin2pi(turns + 0.25f);
                s[2 * k + 1] = m[k] * detail::sin2pi(turns);
            }
            else
            {
                s[2 * k] = 0.0f;
                s[2 * k + 1] = 0.0f;
            }
        }
    }
    else
    {
        c.smear.reset();
    }

    c.shift.apply(s, fi);
    c.phase.apply(s, fi);
}

void SpectralProcessor::processChannel(Channel& c, float* x, int n, float mixFrom, float mixTo,
                                       float gainFrom, float gainTo) noexcept
{
    // Latency-compensated dry path: the delay equals the STFT latency.
    const int mask = static_cast<int>(c.dry.size()) - 1;
    for (int t = 0; t < n; ++t)
    {
        c.dry[u(c.dryPos)] = x[t];
        dryBlock_[u(t)] = c.dry[u((c.dryPos - fftSize_) & mask)];
        c.dryPos = (c.dryPos + 1) & mask;
    }

    c.stft.process(x, x, n, [this, &c](float* spectrum) { processFrame(c, spectrum); });

    const float inv = 1.0f / static_cast<float>(n);
    for (int t = 0; t < n; ++t)
    {
        const float a = static_cast<float>(t + 1) * inv;
        const float mix = mixFrom + a * (mixTo - mixFrom);
        const float gain = gainFrom + a * (gainTo - gainFrom);
        x[t] = gain * (mix * x[t] + (1.0f - mix) * dryBlock_[u(t)]);
    }
}

void SpectralProcessor::process(float* const* channels, int n) noexcept
{
    const float mixStart = mix_;
    const float gainStart = gain_;
    int offset = 0;
    while (offset < n)
    {
        const int len = std::min(kControlBlock, n - offset);
        bool finite = true;
        for (int c = 0; c < channels_; ++c)
        {
            const float* x = channels[c] + offset;
            for (int t = 0; t < len; ++t)
                finite = finite && std::isfinite(x[t]);
        }
        if (!finite)
        {
            reset();
            for (int c = 0; c < channels_; ++c)
                std::fill_n(channels[c] + offset, len, 0.0f);
            offset += len;
            continue;
        }

        // Mix and gain ramp linearly across the whole call.
        const float from = static_cast<float>(offset) / static_cast<float>(n);
        const float to = static_cast<float>(offset + len) / static_cast<float>(n);
        const float mixFrom = mixStart + from * (mixTarget_ - mixStart);
        const float mixTo = mixStart + to * (mixTarget_ - mixStart);
        const float gainFrom = gainStart + from * (gainTarget_ - gainStart);
        const float gainTo = gainStart + to * (gainTarget_ - gainStart);
        for (int c = 0; c < channels_; ++c)
            processChannel(ch_[u(c)], channels[c] + offset, len, mixFrom, mixTo, gainFrom, gainTo);

        for (int c = 0; c < channels_; ++c)
        {
            const float* y = channels[c] + offset;
            for (int t = 0; t < len; ++t)
                finite = finite && std::isfinite(y[t]);
        }
        if (!finite)
        {
            reset();
            for (int c = 0; c < channels_; ++c)
                std::fill_n(channels[c] + offset, len, 0.0f);
        }
        offset += len;
    }
    mix_ = mixTarget_;
    gain_ = gainTarget_;
}

int SpectralProcessor::tailSamples() const noexcept
{
    if (freezeOn_)
        return static_cast<int>(60.0 * sampleRate_); // held indefinitely; report a minute
    // Smear decays by 60 dB in about 6.9 time constants.
    return fftSize_ + static_cast<int>(7.0 * smearSeconds_ * sampleRate_);
}

} // namespace alfa::dsp
