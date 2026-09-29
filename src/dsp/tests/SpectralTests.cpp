#include "TestHelpers.h"
#include "alfa/dsp/Spectral.h"
#include "alfa/dsp/detail/SpectralStft.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace alfa::dsp;
using alfa::dsp::test::AllocationGuard;
using alfa::dsp::test::rms;
using alfa::dsp::test::toneAmplitude;
using alfa::dsp::test::zeroCrossingHz;

namespace
{

constexpr double kPiD = 3.14159265358979323846;
constexpr std::array<double, 3> kRates{44100.0, 48000.0, 96000.0};

std::size_t u(long long i)
{
    return static_cast<std::size_t>(i);
}

std::vector<float> sine(double hz, double sr, long long n, float amp = 0.5f)
{
    std::vector<float> x(u(n));
    for (long long i = 0; i < n; ++i)
        x[u(i)] = amp * static_cast<float>(std::sin(2.0 * kPiD * hz * static_cast<double>(i) / sr));
    return x;
}

std::vector<float> noise(long long n, std::uint32_t seed = 1, float amp = 0.5f)
{
    detail::Random r(seed);
    std::vector<float> x(u(n));
    for (auto& v : x)
        v = amp * r.nextBipolar();
    return x;
}

using Params = std::array<float, kSlotParams>;

Params defaults()
{
    Params p{};
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = SpectralProcessor::params()[i].defaultNorm;
    return p;
}

float fftNorm(int fftSize)
{
    return fftSize == 512 ? 0.0f : fftSize == 1024 ? 0.5f : 1.0f;
}

ProcessSpec spec(double sr, int channels = 2)
{
    ProcessSpec s;
    s.sampleRate = sr;
    s.channels = channels;
    return s;
}

/// Runs mono or stereo audio through the processor in varying block sizes (<= 32), calling
/// setParams before every block.
std::vector<std::vector<float>> run(Processor& p, const Params& params, std::vector<std::vector<float>> x,
                                    int blockSeed = 7)
{
    detail::Random r(static_cast<std::uint32_t>(blockSeed));
    const auto n = static_cast<long long>(x[0].size());
    long long pos = 0;
    std::array<float*, 2> ptrs{};
    while (pos < n)
    {
        const int len = static_cast<int>(std::min<long long>(n - pos, 1 + (r.nextU32() % kControlBlock)));
        p.setParams(params.data(), {});
        for (std::size_t c = 0; c < x.size(); ++c)
            ptrs[c] = x[c].data() + pos;
        p.process(ptrs.data(), len);
        pos += len;
    }
    return x;
}

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b, long long offset, long long from)
{
    // max |a[i] - b[i - offset]| for i >= from
    double m = 0.0;
    for (long long i = from; i < static_cast<long long>(a.size()); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[u(i)]) - b[u(i - offset)]));
    return m;
}

std::vector<float> tail(const std::vector<float>& x, long long from)
{
    return {x.begin() + from, x.end()};
}

double dB(double ratio)
{
    return 20.0 * std::log10(std::max(ratio, 1e-12));
}

} // namespace

// STFT engine ---------------------------------------------------------------------------------

TEST_CASE("STFT reconstructs the input after its latency at every size, rate and window", "[dsp][spectral]")
{
    for (const auto window : {detail::StftWindow::SqrtHann, detail::StftWindow::Hann})
        for (const double sr : kRates)
            for (int size = 256; size <= 4096; size *= 2)
            {
                detail::SpectralStft stft;
                stft.prepare(256, 4096, window);
                stft.setFftSize(size);
                REQUIRE(stft.fftSize() == size);
                REQUIRE(stft.latency() == size);
                auto x = noise(size * 8, 3);
                const auto s = sine(997.0, sr, size * 8, 0.4f);
                for (std::size_t i = 0; i < x.size(); ++i)
                    x[i] += s[i];
                std::vector<float> y(x.size());
                detail::Random r(5);
                std::size_t pos = 0;
                int frames = 0;
                while (pos < x.size())
                {
                    const auto len = std::min<std::size_t>(x.size() - pos, 1 + r.nextU32() % kControlBlock);
                    stft.process(x.data() + pos, y.data() + pos, static_cast<int>(len),
                                 [&](float*) { ++frames; });
                    pos += len;
                }
                REQUIRE(frames == static_cast<int>(x.size()) / stft.hop());
                REQUIRE(maxAbsDiff(y, x, size, size) < 1e-4);
                // Nothing leaks out before the latency.
                REQUIRE(rms(y.data(), u(size)) < 1e-6);
            }
}

TEST_CASE("STFT works in place", "[dsp][spectral]")
{
    detail::SpectralStft stft;
    stft.prepare(512, 512);
    const auto x = noise(4096, 9);
    auto y = x;
    for (std::size_t pos = 0; pos < y.size(); pos += 32)
        stft.process(y.data() + pos, y.data() + pos, 32, [](float*) {});
    REQUIRE(maxAbsDiff(y, x, 512, 512) < 1e-4);
}

TEST_CASE("Phase-vocoder frame round trip is transparent and measures frequency", "[dsp][spectral]")
{
    constexpr double sr = 48000.0;
    constexpr int size = 1024;
    detail::SpectralStft stft;
    stft.prepare(size, size);
    detail::PhaseVocoderFrame pv;
    pv.prepare(size);
    const double hz = 1234.5;
    const auto x = sine(hz, sr, 16384);
    std::vector<float> y(x.size());
    double measured = 0.0;
    for (std::size_t pos = 0; pos < x.size(); pos += 32)
        stft.process(x.data() + pos, y.data() + pos, 32,
                     [&](float* s)
                     {
                         pv.analyse(s, size, stft.hop());
                         const float* m = pv.magnitude();
                         const auto peak = std::max_element(m, m + pv.bins()) - m;
                         measured = pv.frequency()[peak] * sr / size;
                         pv.synthesise(s, stft.hop());
                     });
    REQUIRE(std::fabs(measured - hz) < 0.05);
    REQUIRE(maxAbsDiff(y, x, size, size) < 1e-4);
}

// Individual bin operations -------------------------------------------------------------------

TEST_CASE("Every bin operation is transparent at its neutral setting", "[dsp][spectral]")
{
    constexpr int size = 1024;
    spectral::FrameInfo info;
    info.fftSize = size;
    info.hop = size / 4;
    info.sampleRate = 48000.0;
    const auto spectrum = noise(size, 11);
    std::vector<float> mag(u(info.bins()));
    spectral::magnitudes(spectrum.data(), size, mag.data());
    const auto mag0 = mag;

    spectral::Freeze freeze;
    freeze.prepare(size);
    spectral::Smear smear;
    smear.prepare(size);
    spectral::Blur blur;
    blur.prepare(size);
    spectral::BandShape band;
    band.prepare(48000.0, 512, 2048);
    spectral::Gate gate;
    spectral::Sparsify sparse;
    sparse.prepare(size);
    spectral::FrequencyShift shift;
    spectral::PhaseMorph phase;

    freeze.setAmount(0.0f);
    smear.setTime(0.0f);
    blur.setRadius(0.0f);
    std::array<float, kBands> zeros{};
    band.setBandsDb(zeros.data());
    band.setTilt(0.0f);
    gate.setThreshold(spectral::Gate::kOffDb);
    sparse.setKeep(1.0f);
    shift.setShift(0.0f);
    phase.setAmount(0.0f);
    REQUIRE_FALSE(freeze.active());
    REQUIRE_FALSE(smear.active());
    REQUIRE_FALSE(blur.active());
    REQUIRE_FALSE(band.active());
    REQUIRE_FALSE(gate.active());
    REQUIRE_FALSE(sparse.active());
    REQUIRE_FALSE(shift.active());
    REQUIRE_FALSE(phase.active());

    auto s = spectrum;
    for (int frame = 0; frame < 3; ++frame)
    {
        freeze.apply(s.data(), info);
        smear.apply(mag.data(), info);
        blur.apply(mag.data(), info);
        band.apply(mag.data(), info);
        gate.apply(mag.data(), info);
        sparse.apply(mag.data(), info);
        shift.apply(s.data(), info);
        phase.apply(s.data(), info);
    }
    REQUIRE(s == spectrum);
    REQUIRE(mag == mag0);

    // Smear switched on primes itself from the current frame, so switching it on is seamless.
    smear.setTime(0.5f);
    auto m = mag0;
    smear.apply(m.data(), info);
    REQUIRE(m == mag0);
}

TEST_CASE("Band shape follows the 64-band curve and tilt", "[dsp][spectral]")
{
    constexpr int size = 2048;
    spectral::BandShape band;
    band.prepare(48000.0, 512, 2048);
    spectral::FrameInfo info{size, size / 4, 48000.0};
    std::vector<float> mag(u(info.bins()), 1.0f);
    band.setTilt(6.0f);
    band.apply(mag.data(), info);
    auto binOf = [&](double hz)
    {
        return u(std::llround(hz * size / 48000.0));
    };
    REQUIRE(std::fabs(dB(mag[binOf(1000.0)])) < 0.3);
    REQUIRE(std::fabs(dB(mag[binOf(4000.0)]) - 12.0) < 0.5);
    REQUIRE(std::fabs(dB(mag[binOf(250.0)]) + 12.0) < 0.5);

    std::array<float, kBands> db{};
    band.bump(db.data(), -12.0f, 2000.0f, 1.0f);
    band.setBandsDb(db.data());
    band.setTilt(0.0f);
    std::fill(mag.begin(), mag.end(), 1.0f);
    band.apply(mag.data(), info);
    REQUIRE(std::fabs(dB(mag[binOf(2000.0)]) + 12.0) < 0.5);
    REQUIRE(std::fabs(dB(mag[binOf(200.0)])) < 0.1);
}

TEST_CASE("Sparsify keeps the strongest fraction of bins", "[dsp][spectral]")
{
    constexpr int size = 512;
    spectral::Sparsify sparse;
    sparse.prepare(size);
    spectral::FrameInfo info{size, size / 4, 48000.0};
    std::vector<float> mag(u(info.bins()));
    for (int k = 0; k < info.bins(); ++k)
        mag[u(k)] = static_cast<float>((k * 37) % info.bins()); // a permutation of 0 .. 256
    sparse.setKeep(0.1f);
    sparse.apply(mag.data(), info);
    const auto kept = std::count_if(mag.begin(), mag.end(), [](float m) { return m > 0.0f; });
    REQUIRE(kept == static_cast<long>(std::ceil(0.1 * info.bins())));
    REQUIRE(*std::max_element(mag.begin(), mag.end()) == static_cast<float>(info.bins() - 1));
}

// SpectralProcessor ---------------------------------------------------------------------------

TEST_CASE("SPECTRAL param table is complete", "[dsp][spectral]")
{
    const auto& t = SpectralProcessor::params();
    for (const auto& info : t)
    {
        REQUIRE_FALSE(info.name.empty());
        REQUIRE(info.defaultNorm >= 0.0f);
        REQUIRE(info.defaultNorm <= 1.0f);
    }
    REQUIRE(mapParam(t[2], t[2].defaultNorm) > 990.0f);
    REQUIRE(mapParam(t[2], t[2].defaultNorm) < 1010.0f);
    REQUIRE(mapParam(t[8], 0.0f) == 1.0f);
    REQUIRE(mapParam(t[12], 0.5f) == 0.0f);
    REQUIRE(mapParam(t[13], 0.5f) == 0.0f);
    REQUIRE(mapParam(t[15], 0.5f) == 0.0f);
}

TEST_CASE("SPECTRAL at neutral params is a pure delay at every size and rate", "[dsp][spectral]")
{
    for (const double sr : kRates)
        for (const int size : {512, 1024, 2048})
        {
            SpectralProcessor p;
            auto params = defaults();
            params[0] = fftNorm(size);
            p.prepare(spec(sr));
            p.setParams(params.data(), {});
            p.reset(); // the FFT size takes effect at reset
            REQUIRE(p.latencySamples() == size);
            const auto l = noise(size * 6, 21);
            const auto r = sine(440.0, sr, size * 6);
            const auto y = run(p, params, {l, r});
            REQUIRE(maxAbsDiff(y[0], l, size, size) < 1e-4);
            REQUIRE(maxAbsDiff(y[1], r, size, size) < 1e-4);
        }
}

TEST_CASE("SPECTRAL dry mix is latency compensated", "[dsp][spectral]")
{
    SpectralProcessor p;
    p.prepare(spec(48000.0, 1));
    auto params = defaults();
    params[14] = 0.3f; // mix
    params[4] = 0.5f;  // neutral tilt
    const auto x = noise(8192, 4);
    const auto y = run(p, params, {x});
    REQUIRE(maxAbsDiff(y[0], x, 1024, 2048) < 1e-4);
}

TEST_CASE("SPECTRAL gate removes a quiet tone and keeps a loud one", "[dsp][spectral]")
{
    constexpr double sr = 48000.0;
    SpectralProcessor p;
    p.prepare(spec(sr, 1));
    auto params = defaults();
    params[0] = fftNorm(2048);
    params[5] = 1.0f - 30.0f / 96.0f; // -30 dB
    params[6] = 0.25f;                // 6 dB knee
    p.setParams(params.data(), {});
    p.reset();
    const long long n = 48000;
    auto x = sine(1000.0, sr, n, 0.5f);
    const auto quiet = sine(3000.0, sr, n, 0.0025f); // -46 dB
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] += quiet[i];
    const auto y = tail(run(p, params, {x})[0], 8192);
    REQUIRE(std::fabs(dB(toneAmplitude(y, 1000.0, sr) / 0.5)) < 0.5);
    REQUIRE(toneAmplitude(y, 3000.0, sr) < 0.0025 * 0.01); // at least 40 dB down

    // Inverted, the loud tone goes and the quiet one stays.
    params[7] = 1.0f;
    p.reset();
    const auto z = tail(run(p, params, {x})[0], 8192);
    REQUIRE(toneAmplitude(z, 1000.0, sr) < 0.05);
    REQUIRE(std::fabs(dB(toneAmplitude(z, 3000.0, sr) / 0.0025)) < 1.0);
}

TEST_CASE("SPECTRAL freeze sustains a tone after the input stops", "[dsp][spectral]")
{
    constexpr double sr = 48000.0;
    SpectralProcessor p;
    p.prepare(spec(sr, 1));
    auto params = defaults();
    const long long n = 24000;
    auto x = sine(660.0, sr, n);
    x.resize(u(3 * n), 0.0f); // then 1 s of silence
    // Live for 0.4 s, then freeze.
    std::vector<float> y = x;
    std::array<float*, 1> ptr{};
    for (long long pos = 0; pos < 3 * n; pos += 32)
    {
        params[9] = pos < 19200 ? 0.0f : 1.0f;
        p.setParams(params.data(), {});
        ptr[0] = y.data() + pos;
        p.process(ptr.data(), 32);
    }
    const std::vector<float> late(y.begin() + 2 * n, y.end()); // 0.5 .. 1 s after the input stopped
    REQUIRE(std::fabs(dB(toneAmplitude(late, 660.0, sr) / 0.5)) < 1.5);
    REQUIRE(std::fabs(zeroCrossingHz(late, sr) - 660.0) < 0.5);
    REQUIRE(p.tailSamples() > static_cast<int>(sr));

    // Releasing the freeze lets it go silent.
    params[9] = 0.0f;
    std::vector<float> z(u(8192), 0.0f);
    for (long long pos = 0; pos < 8192; pos += 32)
    {
        p.setParams(params.data(), {});
        ptr[0] = z.data() + pos;
        p.process(ptr.data(), 32);
    }
    REQUIRE(rms(z.data() + 4096, 4096) < 1e-6);
}

TEST_CASE("SPECTRAL frequency shift moves a 1 kHz tone by the requested Hz", "[dsp][spectral]")
{
    for (const double sr : kRates)
        for (const float shiftHz : {250.0f, -137.0f, 61.3f})
        {
            SpectralProcessor p;
            p.prepare(spec(sr, 1));
            auto params = defaults();
            params[12] = 0.5f + shiftHz / 4000.0f;
            const auto x = sine(1000.0, sr, static_cast<long long>(sr));
            const auto y = tail(run(p, params, {x})[0], 4096);
            const double target = 1000.0 + mapParam(SpectralProcessor::params()[12], params[12]);
            INFO("sr " << sr << " shift " << shiftHz);
            REQUIRE(std::fabs(zeroCrossingHz(y, sr) - target) < 0.5);
            REQUIRE(toneAmplitude(y, target, sr) > 0.4);
            REQUIRE(toneAmplitude(y, 1000.0, sr) < 0.01);
        }
}

TEST_CASE("SPECTRAL robot and whisper keep the level and stay finite", "[dsp][spectral]")
{
    constexpr double sr = 48000.0;
    for (const float phase : {0.0f, 1.0f})
    {
        SpectralProcessor p;
        p.prepare(spec(sr, 1));
        auto params = defaults();
        params[13] = phase;
        const auto x = noise(48000, 8, 0.3f);
        const auto y = tail(run(p, params, {x})[0], 4096);
        const double ratio = rms(y) / rms(x);
        INFO("phase " << phase << " ratio " << ratio);
        REQUIRE(ratio > 0.5);
        REQUIRE(ratio < 1.5);
    }
}

TEST_CASE("SPECTRAL smear leaves a decaying tail", "[dsp][spectral]")
{
    constexpr double sr = 48000.0;
    SpectralProcessor p;
    p.prepare(spec(sr, 1));
    auto params = defaults();
    params[10] = 0.25f; // 0.5 s
    auto x = sine(500.0, sr, 24000);
    x.resize(u(96000), 0.0f);
    const auto y = run(p, params, {x})[0];
    const double justAfter = rms(y.data() + 24000 + 2048, 4800);
    const double later = rms(y.data() + 72000, 4800);
    REQUIRE(justAfter > 0.05);
    REQUIRE(later < justAfter);
    REQUIRE(p.tailSamples() > static_cast<int>(3.0 * sr));
}

TEST_CASE("SPECTRAL silence in gives silence out", "[dsp][spectral]")
{
    SpectralProcessor p;
    p.prepare(spec(48000.0));
    auto params = defaults();
    params[1] = 1.0f; // +24 dB band
    params[4] = 1.0f;
    params[5] = 0.5f;
    params[8] = 0.5f;
    params[10] = 0.5f;
    params[11] = 0.5f;
    params[12] = 0.8f;
    params[13] = 0.0f;
    std::vector<float> z(u(8192), 0.0f);
    const auto y = run(p, params, {z, z});
    REQUIRE(rms(y[0]) == 0.0);
    REQUIRE(rms(y[1]) == 0.0);
}

TEST_CASE("SPECTRAL survives NaN and Inf input and NaN params", "[dsp][spectral]")
{
    SpectralProcessor p;
    p.prepare(spec(48000.0));
    auto params = defaults();
    params[1] = 0.8f;
    params[10] = 0.3f;
    auto x = noise(8192, 2);
    x[1000] = std::numeric_limits<float>::quiet_NaN();
    x[3000] = std::numeric_limits<float>::infinity();
    auto y = run(p, params, {x, x});
    for (const auto& ch : y)
        for (const float v : ch)
            REQUIRE(std::isfinite(v));

    Params nanParams;
    nanParams.fill(std::numeric_limits<float>::quiet_NaN());
    y = run(p, nanParams, {noise(8192, 3), noise(8192, 4)});
    for (const auto& ch : y)
        for (const float v : ch)
            REQUIRE(std::isfinite(v));

    // Recovers: back to transparent after a reset delay.
    const auto clean = noise(8192, 5);
    y = run(p, defaults(), {clean, clean});
    REQUIRE(maxAbsDiff(y[0], clean, 1024, 2048) < 1e-4);
}

TEST_CASE("SPECTRAL extreme and random params stay finite and bounded", "[dsp][spectral]")
{
    for (const double sr : kRates)
    {
        SpectralProcessor p;
        p.prepare(spec(sr));
        detail::Random r(99);
        const auto x = noise(static_cast<long long>(sr / 2), 6, 0.9f);
        std::vector<std::vector<float>> y{x, x};
        std::array<float*, 2> ptr{};
        Params params{};
        for (std::size_t pos = 0; pos < x.size(); pos += 32)
        {
            const auto block = pos / 32;
            for (auto& v : params)
            {
                if (block % 400 < 100)
                    v = 0.0f;
                else if (block % 400 < 200)
                    v = 1.0f;
                else
                    v = 0.5f + 0.5f * r.nextBipolar();
            }
            p.setParams(params.data(), {});
            if (block % 97 == 0)
                p.reset();
            ptr[0] = y[0].data() + pos;
            ptr[1] = y[1].data() + pos;
            p.process(ptr.data(), static_cast<int>(std::min<std::size_t>(32, x.size() - pos)));
        }
        for (const auto& ch : y)
            for (const float v : ch)
            {
                REQUIRE(std::isfinite(v));
                REQUIRE(std::fabs(v) < 2000.0f); // band curve capped at +36 dB, +24 dB output gain
            }
    }
}

TEST_CASE("SPECTRAL does not allocate in setParams or process", "[dsp][spectral]")
{
    SpectralProcessor p;
    p.prepare(spec(48000.0));
    auto params = defaults();
    for (std::size_t i = 1; i < params.size(); ++i)
        params[i] = 0.7f;
    auto x = noise(8192, 1);
    auto y = x;
    std::array<float*, 2> ptr{};
    AllocationGuard guard;
    for (std::size_t pos = 0; pos < x.size(); pos += 32)
    {
        params[0] = (pos / 32) % 50 == 0 ? 0.0f : 1.0f;
        p.setParams(params.data(), {});
        if ((pos / 32) % 50 == 0)
            p.reset();
        ptr[0] = x.data() + pos;
        ptr[1] = y.data() + pos;
        p.process(ptr.data(), 32);
    }
    REQUIRE(guard.count() == 0);
}

TEST_CASE("SPECTRAL cost, 1 stereo instance, all ops on", "[.bench][dsp][spectral]")
{
    for (const int size : {512, 1024, 2048})
    {
        SpectralProcessor p;
        p.prepare(spec(48000.0));
        Params params{};
        params[0] = fftNorm(size);
        params[1] = 0.8f;
        params[2] = 0.5f;
        params[3] = 0.5f;
        params[4] = 0.7f;
        params[5] = 0.5f;
        params[6] = 0.25f;
        params[8] = 0.3f;
        params[9] = 0.3f;
        params[10] = 0.1f;
        params[11] = 0.1f;
        params[12] = 0.55f;
        params[13] = 0.8f;
        params[14] = 0.8f;
        params[15] = 0.5f;
        p.setParams(params.data(), {});
        p.reset();
        auto l = noise(48000, 1);
        auto r = noise(48000, 2);
        std::array<float*, 2> ptr{};
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t pos = 0; pos < l.size(); pos += 32)
        {
            p.setParams(params.data(), {});
            ptr[0] = l.data() + pos;
            ptr[1] = r.data() + pos;
            p.process(ptr.data(), 32);
        }
        const auto us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        std::printf("SPECTRAL fft %d: %.2f us per 128-sample stereo block (sink %g)\n", size,
                    us / (48000.0 / 128.0), static_cast<double>(l[100] + r[200]));
    }
}
