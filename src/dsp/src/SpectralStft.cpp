#include "alfa/dsp/detail/SpectralStft.h"

#include <stdexcept>

namespace alfa::dsp::detail
{

namespace
{

constexpr double kPiD = 3.14159265358979323846;

int orderOf(int n) noexcept
{
    int order = 0;
    while ((1 << order) < n)
        ++order;
    return order;
}

bool isPowerOfTwo(int n) noexcept
{
    return n > 0 && (n & (n - 1)) == 0;
}

} // namespace

// SpectralStft --------------------------------------------------------------------------------

void SpectralStft::prepare(int minFftSize, int maxFftSize, StftWindow window)
{
    if (!isPowerOfTwo(minFftSize) || !isPowerOfTwo(maxFftSize) || minFftSize < 32 ||
        minFftSize > maxFftSize || maxFftSize > (1 << kMaxOrder))
        throw std::invalid_argument("SpectralStft sizes must be powers of two in [32, 65536]");
    for (int order = orderOf(minFftSize); order <= orderOf(maxFftSize); ++order)
        ffts_[static_cast<std::size_t>(order)].prepare(1 << order);
    window_ = window;
    minFftSize_ = minFftSize;
    maxFftSize_ = maxFftSize;
    const auto size = static_cast<std::size_t>(maxFftSize);
    analysisWindow_.resize(size);
    synthesisWindow_.resize(size);
    input_.resize(size);
    frame_.resize(size);
    acc_.resize(size);
    ready_.resize(size / kOverlap);
    fftSize_ = 0;
    setFftSize(maxFftSize);
}

void SpectralStft::setFftSize(int n) noexcept
{
    if (maxFftSize_ == 0)
        return;
    n = std::clamp(nextPowerOfTwo(std::max(n, 1)), minFftSize_, maxFftSize_);
    if (n != fftSize_)
    {
        fftSize_ = n;
        hop_ = n / kOverlap;
        fft_ = &ffts_[static_cast<std::size_t>(orderOf(n))];

        // Periodic Hann, so four copies at a hop of n/4 sum to a constant (and so do their
        // squares); the constant is measured rather than assumed so both window types share
        // one code path.
        for (int t = 0; t < n; ++t)
        {
            const double h = 0.5 - 0.5 * std::cos(2.0 * kPiD * t / n);
            const double w = window_ == StftWindow::SqrtHann ? std::sqrt(h) : h;
            analysisWindow_[static_cast<std::size_t>(t)] = static_cast<float>(w);
        }
        double overlapSum = 0.0;
        for (int t = 0; t < n; ++t)
        {
            const double w = analysisWindow_[static_cast<std::size_t>(t)];
            overlapSum += w * w;
        }
        overlapSum /= hop_; // average over one hop of the sum of the four overlapping frames
        const double scale = 1.0 / (overlapSum * n);
        for (int t = 0; t < n; ++t)
        {
            const auto i = static_cast<std::size_t>(t);
            synthesisWindow_[i] = static_cast<float>(analysisWindow_[i] * scale);
        }
    }
    reset();
}

void SpectralStft::reset() noexcept
{
    input_.clear();
    frame_.clear();
    acc_.clear();
    ready_.clear();
    fill_ = 0;
}

void SpectralStft::analyse() noexcept
{
    float* f = frame_.data();
    float* x = input_.data();
    const float* w = analysisWindow_.data();
    for (int t = 0; t < fftSize_; ++t)
        f[t] = x[t] * w[t];
    std::copy(x + hop_, x + fftSize_, x);
    fft_->forward(f, f);
}

void SpectralStft::synthesise() noexcept
{
    float* f = frame_.data();
    float* a = acc_.data();
    const float* w = synthesisWindow_.data();
    fft_->inverse(f, f);
    for (int t = 0; t < fftSize_; ++t)
        a[t] += f[t] * w[t];
    std::copy_n(a, hop_, ready_.data());
    std::copy(a + hop_, a + fftSize_, a);
    std::fill(a + fftSize_ - hop_, a + fftSize_, 0.0f);
}

// PhaseVocoderFrame ---------------------------------------------------------------------------

void PhaseVocoderFrame::prepare(int maxFftSize)
{
    const auto bins = static_cast<std::size_t>(maxFftSize / 2 + 1);
    mag_.assign(bins, 0.0f);
    freq_.assign(bins, 0.0f);
    phase_.assign(bins, 0.0f);
    prevPhase_.assign(bins, 0.0f);
    outPhase_.assign(bins, 0.0f);
    peaks_.assign(bins, 0);
    fftSize_ = 0;
    reset();
}

void PhaseVocoderFrame::reset() noexcept
{
    haveAnalysis_ = false;
    haveSynthesis_ = false;
}

void PhaseVocoderFrame::analyse(const float* s, int fftSize, double analysisHop) noexcept
{
    if (fftSize != fftSize_)
    {
        fftSize_ = std::min(fftSize, static_cast<int>(mag_.size() - 1) * 2);
        reset();
    }
    const int half = fftSize_ / 2;
    std::swap(phase_, prevPhase_); // no allocation: swaps the buffers
    for (int k = 0; k <= half; ++k)
    {
        float re = 0.0f;
        float im = 0.0f;
        if (k == 0)
            re = s[0];
        else if (k == half)
            re = s[1];
        else
        {
            re = s[2 * k];
            im = s[2 * k + 1];
        }
        const auto i = static_cast<std::size_t>(k);
        mag_[i] = std::sqrt(re * re + im * im);
        phase_[i] = std::atan2(im, re);
    }

    if (!haveAnalysis_ || !(analysisHop > 0.0))
    {
        for (int k = 0; k <= half; ++k)
            freq_[static_cast<std::size_t>(k)] = static_cast<float>(k);
        haveAnalysis_ = true;
        return;
    }
    // Deviation from the bin centre's expected advance, unwrapped, gives the frequency. It is
    // unambiguous within +-fftSize / (2 hop) bins of the centre (+-2 bins at 75% overlap).
    const double expectedPerBin = 2.0 * kPiD * analysisHop / fftSize_;
    const double binsPerRadian = fftSize_ / (2.0 * kPiD * analysisHop);
    for (int k = 0; k <= half; ++k)
    {
        const auto i = static_cast<std::size_t>(k);
        const double expected = std::fmod(expectedPerBin * k, 2.0 * kPiD);
        const float dev = wrapPhase(static_cast<float>(phase_[i] - prevPhase_[i] - expected));
        freq_[i] = static_cast<float>(k + dev * binsPerRadian);
    }
}

void PhaseVocoderFrame::synthesise(float* s, double synthesisHop, bool lockPhases) noexcept
{
    const int half = fftSize_ / 2;
    if (half == 0)
        return;
    if (!haveSynthesis_)
    {
        std::copy_n(phase_.data(), half + 1, outPhase_.data());
        haveSynthesis_ = true;
    }
    else
    {
        const auto radiansPerBin = static_cast<float>(2.0 * kPiD * synthesisHop / fftSize_);
        if (!lockPhases)
        {
            for (int k = 0; k <= half; ++k)
            {
                const auto i = static_cast<std::size_t>(k);
                outPhase_[i] = wrapPhase(outPhase_[i] + wrapPhase(radiansPerBin * freq_[i]));
            }
        }
        else
        {
            // Peaks accumulate phase; every other bin keeps its analysed offset to the peak of
            // its region (regions split halfway between neighbouring peaks).
            int count = 0;
            for (int k = 1; k < half; ++k)
            {
                const auto i = static_cast<std::size_t>(k);
                if (mag_[i] > mag_[i - 1] && mag_[i] >= mag_[i + 1])
                    peaks_[static_cast<std::size_t>(count++)] = k;
            }
            if (count == 0)
            {
                for (int k = 0; k <= half; ++k)
                {
                    const auto i = static_cast<std::size_t>(k);
                    outPhase_[i] = wrapPhase(outPhase_[i] + wrapPhase(radiansPerBin * freq_[i]));
                }
            }
            else
            {
                for (int p = 0; p < count; ++p)
                {
                    const auto i = static_cast<std::size_t>(peaks_[static_cast<std::size_t>(p)]);
                    outPhase_[i] = wrapPhase(outPhase_[i] + wrapPhase(radiansPerBin * freq_[i]));
                }
                int p = 0;
                for (int k = 0; k <= half; ++k)
                {
                    while (p + 1 < count && 2 * k >= peaks_[static_cast<std::size_t>(p)] +
                                                         peaks_[static_cast<std::size_t>(p + 1)])
                        ++p;
                    const auto peak = static_cast<std::size_t>(peaks_[static_cast<std::size_t>(p)]);
                    const auto i = static_cast<std::size_t>(k);
                    if (i != peak)
                        outPhase_[i] = wrapPhase(outPhase_[peak] + phase_[i] - phase_[peak]);
                }
            }
        }
    }

    s[0] = mag_[0] * std::cos(outPhase_[0]);
    s[1] = mag_[static_cast<std::size_t>(half)] * std::cos(outPhase_[static_cast<std::size_t>(half)]);
    for (int k = 1; k < half; ++k)
    {
        const auto i = static_cast<std::size_t>(k);
        s[2 * k] = mag_[i] * std::cos(outPhase_[i]);
        s[2 * k + 1] = mag_[i] * std::sin(outPhase_[i]);
    }
}

} // namespace alfa::dsp::detail
