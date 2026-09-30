#include "TestHelpers.h"
#include "alfa/dsp/Resample.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{
constexpr double kPi = 3.14159265358979323846;

std::size_t u(long long i)
{
    return static_cast<std::size_t>(i);
}

std::vector<float> sine(double hz, double sr, std::size_t n, double amp = 0.5)
{
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = static_cast<float>(amp * std::sin(2.0 * kPi * hz * static_cast<double>(i) / sr));
    return x;
}

/// Streams x through the resampler in uneven chunks (as an audio callback would).
std::vector<float> stream(PolyphaseResampler& r, const std::vector<float>& x)
{
    std::vector<float> out;
    std::vector<float> buf(256);
    std::size_t pos = 0;
    int chunk = 37;
    while (pos < x.size())
    {
        const int n = static_cast<int>(std::min<std::size_t>(u(chunk), x.size() - pos));
        int used = 0;
        int got = 0;
        do
        {
            int consumed = 0;
            got = r.process(x.data() + pos + u(used), n - used, buf.data(), static_cast<int>(buf.size()),
                            consumed);
            used += consumed;
            out.insert(out.end(), buf.begin(), buf.begin() + got);
        } while (used < n);
        pos += u(n);
        chunk = chunk == 37 ? 64 : 37;
    }
    return out;
}

struct Fit
{
    double amplitude;
    double snrDb;
};

/// Least-squares sine fit at hz over [from, to), and the SNR of the residual.
Fit fitSine(const std::vector<float>& y, std::size_t from, std::size_t to, double hz, double sr)
{
    double ss = 0, cc = 0, sc = 0, ys = 0, yc = 0;
    for (std::size_t i = from; i < to; ++i)
    {
        const double w = 2.0 * kPi * hz * static_cast<double>(i) / sr;
        const double s = std::sin(w), c = std::cos(w);
        ss += s * s;
        cc += c * c;
        sc += s * c;
        ys += y[i] * s;
        yc += y[i] * c;
    }
    const double det = ss * cc - sc * sc;
    const double a = (ys * cc - yc * sc) / det;
    const double b = (yc * ss - ys * sc) / det;
    double sig = 0, err = 0;
    for (std::size_t i = from; i < to; ++i)
    {
        const double w = 2.0 * kPi * hz * static_cast<double>(i) / sr;
        const double model = a * std::sin(w) + b * std::cos(w);
        sig += model * model;
        err += (y[i] - model) * (y[i] - model);
    }
    return {std::sqrt(a * a + b * b), 10.0 * std::log10(sig / std::max(err, 1e-300))};
}
} // namespace

// Real-time resampler ---------------------------------------------------------------------------

TEST_CASE("PolyphaseResampler passband is flat", "[dsp][delay][resample]")
{
    const double inRate = 44100.0, outRate = 48000.0;
    for (double hz : {100.0, 1000.0, 5000.0, 10000.0, 15000.0, 17500.0})
    {
        PolyphaseResampler r;
        r.prepare();
        r.setRatio(inRate / outRate);
        const auto y = stream(r, sine(hz, inRate, 44100));
        const auto fit = fitSine(y, 1000, y.size() - 1000, hz, outRate);
        INFO(hz << " Hz: " << 20.0 * std::log10(fit.amplitude / 0.5) << " dB");
        REQUIRE(std::fabs(20.0 * std::log10(fit.amplitude / 0.5)) < 0.01);
    }
}

TEST_CASE("PolyphaseResampler rejects the stopband when downsampling", "[dsp][delay][resample]")
{
    // 96 -> 48 kHz: everything above 24 kHz must not alias back.
    for (double hz : {25000.0, 30000.0, 40000.0})
    {
        PolyphaseResampler r;
        r.prepare();
        r.setRatio(2.0);
        const auto y = stream(r, sine(hz, 96000.0, 96000));
        const double level = rms(y.data() + 500, y.size() - 1000) / (0.5 / std::sqrt(2.0));
        INFO(hz << " Hz: " << 20.0 * std::log10(level) << " dB");
        REQUIRE(20.0 * std::log10(level) < -75.0);
    }
}

TEST_CASE("PolyphaseResampler sine SNR", "[dsp][delay][resample]")
{
    struct Case
    {
        double from, to, hz;
    };
    for (const auto& c : {Case{44100.0, 48000.0, 1000.0}, Case{48000.0, 44100.0, 1000.0},
                          Case{48000.0, 96000.0, 7000.0}, Case{44100.0, 48000.0, 12000.0}})
    {
        PolyphaseResampler r;
        r.prepare();
        r.setRatio(c.from / c.to);
        const auto y = stream(r, sine(c.hz, c.from, static_cast<std::size_t>(c.from)));
        const auto fit = fitSine(y, 500, y.size() - 500, c.hz, c.to);
        INFO(c.from << " -> " << c.to << " at " << c.hz << " Hz: " << fit.snrDb << " dB");
        REQUIRE(fit.snrDb > 80.0);
    }
}

TEST_CASE("PolyphaseResampler latency and output count", "[dsp][delay][resample]")
{
    PolyphaseResampler r;
    r.prepare(64);
    r.setRatio(1.0);
    // Output m is aligned with input time m * ratio: an impulse at input 0 peaks at output 0...
    std::vector<float> x(200, 0.0f);
    x[0] = 1.0f;
    const auto y = stream(r, x);
    const auto peakAt =
        std::max_element(y.begin(), y.end(), [](float a, float b) { return std::fabs(a) < std::fabs(b); }) -
        y.begin();
    REQUIRE(peakAt == 0);
    // ...but it can only be produced once taps / 2 further inputs (the latency) have arrived.
    r.reset();
    std::vector<float> out(8);
    int consumed = 0;
    const auto look = static_cast<int>(r.latency());
    REQUIRE(r.process(x.data(), look, out.data(), 8, consumed) == 0);
    REQUIRE(consumed == look);
    REQUIRE(r.process(x.data() + look, 1, out.data(), 8, consumed) == 1);
    REQUIRE(out[0] > 0.9f);

    // Producing N outputs at ratio q consumes about N q inputs.
    PolyphaseResampler q;
    q.prepare();
    q.setRatio(1.5);
    std::vector<float> in(30000, 0.1f);
    std::vector<float> many(10000);
    const int got = q.process(in.data(), static_cast<int>(in.size()), many.data(),
                              static_cast<int>(many.size()), consumed);
    REQUIRE(got == 10000);
    REQUIRE(std::fabs(static_cast<double>(consumed) - (10000.0 * 1.5 + q.latency())) < 3.0);
}

TEST_CASE("PolyphaseResampler variable ratio is continuous and allocation-free", "[dsp][delay][resample]")
{
    PolyphaseResampler r;
    r.prepare(64, 4.0);
    const auto x = sine(440.0, 48000.0, 96000, 0.5);
    std::vector<float> out(64);
    std::vector<float> all;
    all.reserve(400000);
    std::size_t pos = 0;
    int block = 0;
    AllocationGuard guard;
    while (pos + 32 <= x.size())
    {
        // Glide between 0.5x and 2x speed (vibrato-like pitch sweep for sample playback).
        r.setRatio(1.25 + 0.75 * std::sin(0.01 * block++));
        int used = 0;
        while (used < 32)
        {
            int consumed = 0;
            const int got = r.process(x.data() + pos + u(used), 32 - used, out.data(),
                                      static_cast<int>(out.size()), consumed);
            used += consumed;
            for (int i = 0; i < got; ++i)
                all.push_back(out[u(i)]);
        }
        pos += 32;
    }
    REQUIRE(guard.count() == 0);
    float maxStep = 0.0f;
    for (std::size_t i = 100; i < all.size(); ++i)
        maxStep = std::max(maxStep, std::fabs(all[i] - all[i - 1]));
    // 440 Hz at up to 2x speed: slope <= 0.5 * 2 pi 880 / 48000 = 0.058 per output sample.
    REQUIRE(maxStep < 0.065f);
}

TEST_CASE("PolyphaseResampler random-access reads match the band-limited signal", "[dsp][delay][resample]")
{
    PolyphaseResampler r;
    r.prepare();
    const double sr = 48000.0;
    const auto x = sine(3000.0, sr, 4096, 0.5);
    double maxErr = 0.0;
    for (double pos = 200.0; pos < 3800.0; pos += 7.37)
    {
        const double ideal = 0.5 * std::sin(2.0 * kPi * 3000.0 * pos / sr);
        maxErr = std::max(
            maxErr,
            std::fabs(static_cast<double>(r.sampleAt(x.data(), static_cast<std::int64_t>(x.size()), pos)) -
                      ideal));
    }
    REQUIRE(maxErr < 1e-4);
    // Outside the buffer the signal is zero; non-finite positions read silence.
    REQUIRE(r.sampleAt(x.data(), static_cast<std::int64_t>(x.size()), -1000.0) == 0.0f);
    REQUIRE(r.sampleAt(x.data(), static_cast<std::int64_t>(x.size()), std::nan("")) == 0.0f);
}

TEST_CASE("PolyphaseResampler survives non-finite input", "[dsp][delay][resample]")
{
    PolyphaseResampler r;
    r.prepare();
    r.setRatio(0.9);
    auto x = sine(1000.0, 48000.0, 4800);
    x[100] = std::nanf("");
    x[200] = INFINITY;
    const auto y = stream(r, x);
    for (float v : y)
        REQUIRE(std::isfinite(v));
    r.setRatio(std::nan("")); // ignored
    REQUIRE(r.ratio() == 0.9);
}

// Offline ---------------------------------------------------------------------------------------

TEST_CASE("resampleOffline high quality reaches > 90 dB SNR", "[dsp][delay][resample]")
{
    struct Case
    {
        double from, to, hz;
    };
    for (const auto& c :
         {Case{44100.0, 48000.0, 1000.0}, Case{44100.0, 48000.0, 15000.0}, Case{48000.0, 44100.0, 997.0},
          Case{48000.0, 44100.0, 19000.0}, Case{96000.0, 44100.0, 5000.0}, Case{22050.0, 48000.0, 8000.0}})
    {
        const auto n = static_cast<std::size_t>(c.from / 2.0);
        const auto x = sine(c.hz, c.from, n, 0.5);
        const auto y = resampleOffline(x.data(), x.size(), c.from, c.to, 3);
        REQUIRE(y.size() == static_cast<std::size_t>(std::llround(static_cast<double>(n) * c.to / c.from)));
        // Output m is aligned with input time m * from / to, so the ideal is known exactly.
        double sig = 0.0, err = 0.0;
        for (std::size_t m = 400; m + 400 < y.size(); ++m)
        {
            const double ideal = 0.5 * std::sin(2.0 * kPi * c.hz * static_cast<double>(m) / c.to);
            sig += ideal * ideal;
            err += (y[m] - ideal) * (y[m] - ideal);
        }
        const double snr = 10.0 * std::log10(sig / err);
        INFO(c.from << " -> " << c.to << " at " << c.hz << " Hz: " << snr << " dB");
        REQUIRE(snr > 90.0);
    }
}

TEST_CASE("resampleOffline rejects content above the output Nyquist", "[dsp][delay][resample]")
{
    for (int q : {1, 3})
    {
        const auto x = sine(23000.0, 48000.0, 48000, 0.5);
        const auto y = resampleOffline(x.data(), x.size(), 48000.0, 44100.0, q);
        const double level = rms(y.data() + 1000, y.size() - 2000) / (0.5 / std::sqrt(2.0));
        INFO("quality " << q << ": " << 20.0 * std::log10(level) << " dB");
        REQUIRE(20.0 * std::log10(level) < (q == 3 ? -100.0 : -70.0));
    }
}

TEST_CASE("resampleOffline handles every quality and bad arguments", "[dsp][delay][resample]")
{
    const auto x = sine(440.0, 48000.0, 4800);
    for (int q : {-5, 0, 1, 2, 3, 9})
    {
        const auto y = resampleOffline(x.data(), x.size(), 48000.0, 32000.0, q);
        REQUIRE(y.size() == 3200);
        REQUIRE(fitSine(y, 200, 3000, 440.0, 32000.0).amplitude == Catch::Approx(0.5).epsilon(0.001));
    }
    REQUIRE(resampleOffline(nullptr, 10, 48000.0, 44100.0).empty());
    REQUIRE(resampleOffline(x.data(), x.size(), 0.0, 44100.0).empty());
    REQUIRE(resampleOffline(x.data(), x.size(), 48000.0, std::nan("")).empty());
}

TEST_CASE("PolyphaseResampler cost", "[.bench][dsp][delay][resample]")
{
    PolyphaseResampler r;
    r.prepare();
    r.setRatio(44100.0 / 48000.0);
    const auto x = sine(1000.0, 44100.0, 128);
    std::vector<float> out(160);
    const int blocks = 48000 / 128;
    float sink = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < blocks; ++b)
    {
        int consumed = 0;
        const int got = r.process(x.data(), 128, out.data(), 128, consumed);
        sink += out[u(got - 1)];
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("PolyphaseResampler 64 taps: %.2f us per 128 output samples  [%g]\n", us / blocks,
                static_cast<double>(sink));
}
