#include "TestHelpers.h"
#include "alfa/dsp/Convolution.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace alfa::dsp;
using namespace alfa::dsp::test;

namespace
{
constexpr double kPi = 3.14159265358979323846;

std::vector<float> randomSignal(std::size_t n, std::uint32_t seed, float scale = 1.0f)
{
    detail::Random rng(seed);
    std::vector<float> x(n);
    for (auto& v : x)
        v = scale * rng.nextBipolar();
    return x;
}

std::vector<double> directConvolution(const std::vector<float>& x, const std::vector<float>& h)
{
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t t = 0; t < x.size(); ++t)
    {
        double s = 0.0;
        const std::size_t kMax = std::min(h.size(), t + 1);
        for (std::size_t k = 0; k < kMax; ++k)
            s += static_cast<double>(h[k]) * x[t - k];
        y[t] = s;
    }
    return y;
}

/// Uneven chunk sizes, cycling, so block boundaries land everywhere.
constexpr int kChunks[] = {1, 7, 32, 63, 100, 257, 5, 64, 31, 129};

IrSource sourceOf(std::vector<std::vector<float>> channels, double rate = 48000.0)
{
    IrSource s;
    s.channels = std::move(channels);
    s.sampleRate = rate;
    return s;
}

IrPrepareOptions raw()
{
    IrPrepareOptions o;
    o.normalise = false;
    return o;
}

/// Runs a PartitionedConvolver over whole signals with uneven chunks.
std::vector<std::vector<float>> runConvolver(PartitionedConvolver& conv,
                                             const std::vector<std::vector<float>>& in)
{
    const std::size_t n = in[0].size();
    std::vector<std::vector<float>> out(in.size(), std::vector<float>(n));
    std::size_t pos = 0;
    int c = 0;
    while (pos < n)
    {
        const auto k = std::min(static_cast<std::size_t>(kChunks[c++ % 10]), n - pos);
        const float* ins[2] = {in[0].data() + pos, in.size() > 1 ? in[1].data() + pos : nullptr};
        float* outs[2] = {out[0].data() + pos, out.size() > 1 ? out[1].data() + pos : nullptr};
        conv.process(ins, outs, static_cast<int>(k));
        pos += k;
    }
    return out;
}

double maxError(const std::vector<float>& a, const std::vector<double>& b)
{
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(static_cast<double>(a[i]) - b[i]));
    return e;
}

std::array<float, kSlotParams> defaults()
{
    std::array<float, kSlotParams> p{};
    const auto& t = ConvolutionProcessor::params();
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = t[i].defaultNorm;
    return p;
}

float irNorm(int index)
{
    return static_cast<float>(index) / static_cast<float>(kMaxBankIrs - 1);
}

/// Runs the processor in 32-sample control blocks, calling setParams before each.
void run(ConvolutionProcessor& p, const std::array<float, kSlotParams>& params, std::vector<float>& l,
         std::vector<float>& r, int block = kControlBlock)
{
    ControlContext ctx;
    for (std::size_t pos = 0; pos < l.size(); pos += static_cast<std::size_t>(block))
    {
        const int n = static_cast<int>(std::min(static_cast<std::size_t>(block), l.size() - pos));
        p.setParams(params.data(), ctx);
        float* ch[2] = {l.data() + pos, r.data() + pos};
        p.process(ch, n);
    }
}

ProcessSpec stereoSpec(double rate = 48000.0)
{
    ProcessSpec s;
    s.sampleRate = rate;
    s.channels = 2;
    return s;
}

std::vector<float> sine(double hz, double rate, std::size_t n, float amp = 0.5f)
{
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = amp * static_cast<float>(std::sin(2.0 * kPi * hz * static_cast<double>(i) / rate));
    return x;
}

float maxAbs(const std::vector<float>& x, std::size_t from = 0, std::size_t to = SIZE_MAX)
{
    float m = 0.0f;
    for (std::size_t i = from; i < std::min(to, x.size()); ++i)
        m = std::max(m, std::abs(x[i]));
    return m;
}

float maxStep(const std::vector<float>& x, std::size_t from, std::size_t to)
{
    float m = 0.0f;
    for (std::size_t i = std::max<std::size_t>(from, 1); i < std::min(to, x.size()); ++i)
        m = std::max(m, std::abs(x[i] - x[i - 1]));
    return m;
}

bool finite(const std::vector<float>& x)
{
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
} // namespace

// Building blocks ------------------------------------------------------------------------------

TEST_CASE("DirectFir matches direct convolution, in place and in uneven chunks", "[dsp][convspatial]")
{
    for (int taps : {1, 5, 17, 64})
    {
        const auto h = randomSignal(static_cast<std::size_t>(taps), 11u + static_cast<std::uint32_t>(taps));
        auto x = randomSignal(3000, 3);
        const auto ref = directConvolution(x, h);
        DirectFir fir;
        fir.setKernel(h.data(), taps);
        std::size_t pos = 0;
        int c = 0;
        while (pos < x.size())
        {
            const auto k = std::min(static_cast<std::size_t>(kChunks[c++ % 10]), x.size() - pos);
            fir.process(x.data() + pos, x.data() + pos, static_cast<int>(k));
            pos += k;
        }
        INFO("taps " << taps);
        REQUIRE(maxError(x, ref) < 1e-5);
    }
}

TEST_CASE("UniformPartitionedConvolver is exact and reports its latency", "[dsp][convspatial]")
{
    const auto x = randomSignal(12000, 5);
    for (int b : {32, 64, 128})
    {
        const auto h = randomSignal(3001, 7u + static_cast<std::uint32_t>(b), 1.0f / std::sqrt(3001.0f));
        const auto ref = directConvolution(x, h);
        UniformPartitionedConvolver conv;
        conv.prepare(h.data(), static_cast<int>(h.size()), b);
        REQUIRE(conv.latencySamples() == b);
        std::vector<float> y(x.size());
        std::size_t pos = 0;
        int c = 0;
        while (pos < x.size())
        {
            const auto k = std::min(static_cast<std::size_t>(kChunks[c++ % 10]), x.size() - pos);
            conv.process(x.data() + pos, y.data() + pos, static_cast<int>(k));
            pos += k;
        }
        double err = 0.0;
        for (std::size_t t = 0; t < y.size(); ++t)
        {
            const double want = t >= static_cast<std::size_t>(b) ? ref[t - static_cast<std::size_t>(b)] : 0.0;
            err = std::max(err, std::abs(static_cast<double>(y[t]) - want));
        }
        INFO("partition " << b);
        REQUIRE(err < 1e-5);
    }
}

TEST_CASE("PartitionedConvolver is exact with zero latency for every stage boundary", "[dsp][convspatial]")
{
    const auto x = randomSignal(14000, 21);
    // Lengths straddling the head and each stage start.
    for (int length : {1, 63, 64, 65, 1000, 1024, 1025, 8192, 8193, 12000})
    {
        const auto h =
            randomSignal(static_cast<std::size_t>(length), 100u + static_cast<std::uint32_t>(length),
                         1.0f / std::sqrt(static_cast<float>(length)));
        const auto ir = PreparedIr::create(sourceOf({h}), 48000.0, raw());
        PartitionedConvolver conv;
        conv.prepare(1, 48000);
        conv.setIr(ir.get());
        const auto y = runConvolver(conv, {x});
        const auto ref = directConvolution(x, h);
        INFO("length " << length);
        REQUIRE(maxError(y[0], ref) < 1e-5);
    }
}

TEST_CASE("PartitionedConvolver true-stereo and stereo IRs route correctly", "[dsp][convspatial]")
{
    const std::size_t len = 9000;
    std::vector<std::vector<float>> h;
    for (std::uint32_t p = 0; p < 4; ++p)
        h.push_back(randomSignal(len, 40 + p, 0.01f));
    const std::vector<std::vector<float>> x = {randomSignal(12000, 1), randomSignal(12000, 2)};

    SECTION("true stereo")
    {
        const auto ir = PreparedIr::create(sourceOf(h), 48000.0, raw());
        PartitionedConvolver conv;
        conv.prepare(2, 48000);
        conv.setIr(ir.get());
        const auto y = runConvolver(conv, x);
        auto ll = directConvolution(x[0], h[0]);
        const auto lr = directConvolution(x[0], h[1]);
        const auto rl = directConvolution(x[1], h[2]);
        auto rr = directConvolution(x[1], h[3]);
        for (std::size_t i = 0; i < ll.size(); ++i)
        {
            ll[i] += rl[i];
            rr[i] += lr[i];
        }
        REQUIRE(maxError(y[0], ll) < 1e-5);
        REQUIRE(maxError(y[1], rr) < 1e-5);
    }
    SECTION("stereo, and a mono convolver uses the first channel")
    {
        const auto ir = PreparedIr::create(sourceOf({h[0], h[1]}), 48000.0, raw());
        PartitionedConvolver conv;
        conv.prepare(2, 48000);
        conv.setIr(ir.get());
        const auto y = runConvolver(conv, x);
        REQUIRE(maxError(y[0], directConvolution(x[0], h[0])) < 1e-5);
        REQUIRE(maxError(y[1], directConvolution(x[1], h[1])) < 1e-5);

        PartitionedConvolver mono;
        mono.prepare(1, 48000);
        mono.setIr(ir.get());
        const auto m = runConvolver(mono, {x[0]});
        REQUIRE(maxError(m[0], directConvolution(x[0], h[0])) < 1e-5);
    }
}

TEST_CASE("A Dirac IR is the identity", "[dsp][convspatial]")
{
    const auto ir = PreparedIr::create(sourceOf({{1.0f}}), 48000.0); // normalised: still 1
    REQUIRE(ir->normalisationGain() == 1.0f);
    PartitionedConvolver conv;
    conv.prepare(2, 48000);
    conv.setIr(ir.get());
    const std::vector<std::vector<float>> x = {randomSignal(5000, 8), randomSignal(5000, 9)};
    const auto y = runConvolver(conv, x);
    REQUIRE(y[0] == x[0]);
    REQUIRE(y[1] == x[1]);

    // Through the processor with neutral settings the only change is the wet filters at their
    // extremes (20 Hz high-pass, 22 kHz low-pass): a 1 kHz tone keeps its level.
    IrBank bank;
    bank.add(sourceOf({{1.0f}}));
    ConvolutionProcessor p;
    p.prepare(stereoSpec());
    bank.publish(p);
    auto l = sine(1000.0, 48000.0, 24000);
    auto r = l;
    const auto in = l;
    run(p, defaults(), l, r);
    const std::vector<float> tail(l.begin() + 4800, l.end());
    const std::vector<float> ref(in.begin() + 4800, in.end());
    REQUIRE(std::abs(toneAmplitude(tail, 1000.0, 48000.0) / toneAmplitude(ref, 1000.0, 48000.0) - 1.0) <
            0.01);
}

// Processor ------------------------------------------------------------------------------------

TEST_CASE("CONVOLUTION passes dry with no bank, an empty bank or a wrong-rate IR", "[dsp][convspatial]")
{
    ConvolutionProcessor p;
    p.prepare(stereoSpec());
    auto params = defaults();
    auto l = randomSignal(4000, 1);
    auto r = randomSignal(4000, 2);
    const auto l0 = l, r0 = r;
    run(p, params, l, r);
    REQUIRE(l == l0);
    REQUIRE(r == r0);

    IrBank empty;
    empty.publish(p);
    run(p, params, l, r);
    REQUIRE(l == l0);

    IrBank other;
    other.setHostRate(44100.0);
    other.add(makeSyntheticIr({SyntheticIrKind::Room, 44100.0, 0.3}));
    other.publish(p);
    run(p, params, l, r);
    REQUIRE(l == l0);
    REQUIRE(r == r0);
    REQUIRE(p.currentIr() == nullptr);
}

TEST_CASE("CONVOLUTION pre-delay is sample accurate at 44.1, 48 and 96 kHz", "[dsp][convspatial]")
{
    for (double rate : {44100.0, 48000.0, 96000.0})
    {
        IrBank bank;
        bank.setHostRate(rate);
        bank.add(sourceOf({{1.0f}}, rate));
        ConvolutionProcessor p;
        p.prepare(stereoSpec(rate));
        bank.publish(p);
        auto params = defaults();
        params[1] = 0.2f; // 50 ms
        const auto expected = static_cast<std::size_t>(std::lround(0.05 * rate));
        std::vector<float> l(static_cast<std::size_t>(0.2 * rate), 0.0f), r = l;
        l[0] = 1.0f;
        r[0] = 1.0f;
        run(p, params, l, r);
        const auto peak = static_cast<std::size_t>(
            std::max_element(l.begin(), l.end(), [](float a, float b) { return std::abs(a) < std::abs(b); }) -
            l.begin());
        INFO("rate " << rate);
        // Nothing before the delay, the onset exactly on it; the wet low-pass may move the peak
        // by a sample at 96 kHz, where 22 kHz is well below Nyquist.
        REQUIRE(maxAbs(l, 0, expected) < 1e-6f);
        REQUIRE(std::abs(l[expected]) > 0.1f);
        REQUIRE(peak >= expected);
        REQUIRE(peak <= expected + 1);
    }
}

TEST_CASE("CONVOLUTION IR swap crossfades without clicks or allocation", "[dsp][convspatial]")
{
    IrBank bank;
    bank.add(makeSyntheticIr({SyntheticIrKind::Decay, 48000.0, 0.6, 0.0, 2, 1}));
    bank.add(makeSyntheticIr({SyntheticIrKind::Plate, 48000.0, 1.2, 0.0, 2, 2}));
    ConvolutionProcessor p;
    p.prepare(stereoSpec());
    bank.publish(p);
    auto params = defaults();

    const std::size_t n = 48000 * 3;
    auto l = sine(300.0, 48000.0, n);
    auto r = sine(301.0, 48000.0, n);
    const std::size_t swapAt = 48000;
    ControlContext ctx;
    long allocations = 0;
    for (std::size_t pos = 0; pos < n; pos += kControlBlock)
    {
        params[0] = pos < swapAt ? irNorm(0) : irNorm(1);
        float* ch[2] = {l.data() + pos, r.data() + pos};
        AllocationGuard guard;
        p.setParams(params.data(), ctx);
        p.process(ch, kControlBlock);
        allocations += guard.count();
    }
    REQUIRE(allocations == 0);
    REQUIRE(p.currentIr() == bank.get(1).get());
    REQUIRE_FALSE(p.crossfading());

    const float before = maxStep(l, swapAt - 12000, swapAt);
    const float after = maxStep(l, n - 12000, n);
    const float during = maxStep(l, swapAt, swapAt + 4800);
    INFO("before " << before << " during " << during << " after " << after);
    // An instant switch would step by about the output amplitude (~0.5); the crossfade keeps the
    // sample-to-sample change within the steady-state range of either IR.
    REQUIRE(during < 1.5f * std::max(before, after));
    REQUIRE(finite(l));
}

TEST_CASE("CONVOLUTION bank publication is acknowledged and old views are collected", "[dsp][convspatial]")
{
    IrBank bank;
    bank.add(makeSyntheticIr({SyntheticIrKind::Room, 48000.0, 0.3}));
    ConvolutionProcessor p;
    p.prepare(stereoSpec());
    bank.publish(p);
    auto params = defaults();
    std::vector<float> l(4096, 0.0f), r = l;
    run(p, params, l, r);
    REQUIRE(p.acknowledgedGeneration() == 1);

    // Replace the IR: the processor keeps playing the old one through a crossfade and only
    // then acknowledges the new view.
    bank.replace(0, makeSyntheticIr({SyntheticIrKind::Plate, 48000.0, 0.5}));
    bank.publish(p);
    REQUIRE(bank.publishedViews() == 2);
    bank.collect(p);
    REQUIRE(bank.publishedViews() == 2); // nothing acknowledged yet
    std::vector<float> one(kControlBlock, 0.0f), two = one;
    run(p, params, one, two);
    REQUIRE(p.crossfading());
    REQUIRE(p.acknowledgedGeneration() == 1); // the old IR is still audible
    run(p, params, l, r);                     // 85 ms, longer than the fade
    REQUIRE(p.acknowledgedGeneration() == 2);
    bank.collect(p);
    REQUIRE(bank.publishedViews() == 1);
    REQUIRE(p.currentIr() == bank.get(0).get());
}

TEST_CASE("CONVOLUTION silence in gives silence out, and the tail ends", "[dsp][convspatial]")
{
    IrBank bank;
    bank.add(makeSyntheticIr({SyntheticIrKind::Decay, 48000.0, 0.4}));
    ConvolutionProcessor p;
    p.prepare(stereoSpec());
    bank.publish(p);
    auto params = defaults();
    params[1] = 0.1f; // some pre-delay

    std::vector<float> l(48000, 0.0f), r = l;
    run(p, params, l, r);
    REQUIRE(maxAbs(l) == 0.0f);
    REQUIRE(maxAbs(r) == 0.0f);

    std::fill(l.begin(), l.end(), 0.0f);
    std::fill(r.begin(), r.end(), 0.0f);
    l[0] = 1.0f;
    r[0] = 1.0f;
    run(p, params, l, r);
    const auto tail = static_cast<std::size_t>(p.tailSamples());
    REQUIRE(tail < l.size());
    REQUIRE(maxAbs(l, 0, 4800) > 0.01f);
    REQUIRE(maxAbs(l, tail) < 1e-5f);
    REQUIRE(maxAbs(r, tail) < 1e-5f);
}

TEST_CASE("CONVOLUTION length trims the tail", "[dsp][convspatial]")
{
    IrBank bank;
    bank.add(makeSyntheticIr({SyntheticIrKind::Decay, 48000.0, 3.0, 1.0, 2}));
    ConvolutionProcessor p;
    p.prepare(stereoSpec());
    bank.publish(p);
    auto params = defaults();
    params[2] = (30.0f - 5.0f) / 95.0f; // keep 30 % = 300 ms
    params[3] = 0.2f;                   // fade over the last 20 % of it
    std::vector<float> l(48000, 0.0f), r = l;
    l[0] = 1.0f;
    r[0] = 1.0f;
    run(p, params, l, r);
    // Partition granularity in the late stage is 4096 samples (85 ms).
    REQUIRE(maxAbs(l, 0, 9600) > 0.005f);
    REQUIRE(maxAbs(l, 14400 + 4096 + 1000) < 1e-5f);
    REQUIRE(p.tailSamples() < 48000 * 7 / 10);
}

TEST_CASE("CONVOLUTION wet filters and width act on the wet signal", "[dsp][convspatial]")
{
    IrBank bank;
    bank.add(sourceOf({{1.0f}}));
    bank.add(makeSyntheticIr({SyntheticIrKind::Plate, 48000.0, 0.5}));
    ConvolutionProcessor p;
    p.prepare(stereoSpec());
    bank.publish(p);

    auto level = [&](std::array<float, kSlotParams> params, double hz)
    {
        p.reset();
        auto l = sine(hz, 48000.0, 24000);
        auto r = l;
        run(p, params, l, r);
        return toneAmplitude(std::vector<float>(l.begin() + 9600, l.end()), hz, 48000.0);
    };
    auto params = defaults();
    params[4] = std::log(1000.0f / 20.0f) / std::log(50.0f); // low cut 1 kHz
    REQUIRE(level(params, 100.0) < 0.5 * 0.03);
    REQUIRE(level(params, 8000.0) > 0.5 * 0.9);
    params = defaults();
    params[5] = 0.0f; // high cut 1 kHz
    REQUIRE(level(params, 8000.0) < 0.5 * 0.03);
    REQUIRE(level(params, 100.0) > 0.5 * 0.9);

    params = defaults();
    params[0] = irNorm(1);
    params[6] = 0.0f; // width 0: the decorrelated stereo plate becomes mono
    p.reset();
    auto l = randomSignal(12000, 3);
    auto r = randomSignal(12000, 4);
    run(p, params, l, r);
    REQUIRE(maxAbs(l, 6000) > 0.01f);
    for (std::size_t i = 6000; i < l.size(); ++i)
        REQUIRE(l[i] == r[i]);
}

TEST_CASE("CONVOLUTION survives NaN and Inf input and NaN params", "[dsp][convspatial]")
{
    IrBank bank;
    bank.add(makeSyntheticIr({SyntheticIrKind::Room, 48000.0, 0.3}));
    ConvolutionProcessor p;
    p.prepare(stereoSpec());
    bank.publish(p);
    auto params = defaults();
    auto l = randomSignal(9600, 5, 0.5f);
    auto r = randomSignal(9600, 6, 0.5f);
    l[1000] = std::nanf("");
    r[2000] = INFINITY;
    run(p, params, l, r);
    REQUIRE(finite(l));
    REQUIRE(finite(r));

    std::array<float, kSlotParams> bad;
    bad.fill(std::nanf(""));
    auto l2 = randomSignal(9600, 7, 0.5f);
    auto r2 = randomSignal(9600, 8, 0.5f);
    run(p, bad, l2, r2);
    REQUIRE(finite(l2));
    // Recovers: normal input after the bad block gives normal output.
    auto l3 = randomSignal(9600, 9, 0.5f);
    auto r3 = l3;
    run(p, params, l3, r3);
    REQUIRE(finite(l3));
    REQUIRE(maxAbs(l3, 4800) > 0.01f);
}

TEST_CASE("CONVOLUTION extreme and randomly jumping params stay bounded", "[dsp][convspatial]")
{
    for (double rate : {44100.0, 48000.0, 96000.0})
    {
        IrBank bank;
        bank.setHostRate(rate);
        bank.add(makeSyntheticIr({SyntheticIrKind::Decay, 48000.0, 0.8})); // resampled to rate
        bank.add(makeSyntheticIr({SyntheticIrKind::Spring, 48000.0, 1.5}));
        bank.add(makeSyntheticIr({SyntheticIrKind::Room, 48000.0, 0.4, 0.0, 1}));
        ConvolutionProcessor p;
        p.prepare(stereoSpec(rate));
        bank.publish(p);
        std::array<float, kSlotParams> params{};
        detail::Random rng(77);
        ControlContext ctx;
        float peak = 0.0f;
        long allocations = 0;
        for (int b = 0; b < 3000; ++b)
        {
            const int mode = b / 1000; // all 0, all 1, random jumps every block
            for (auto& v : params)
                v = mode == 0 ? 0.0f : mode == 1 ? 1.0f : 0.5f + 0.5f * rng.nextBipolar();
            std::array<float, kControlBlock> l{}, r{};
            for (int i = 0; i < kControlBlock; ++i)
            {
                l[static_cast<std::size_t>(i)] = 0.5f * rng.nextBipolar();
                r[static_cast<std::size_t>(i)] = 0.5f * rng.nextBipolar();
            }
            float* ch[2] = {l.data(), r.data()};
            AllocationGuard guard;
            p.setParams(params.data(), ctx);
            p.process(ch, kControlBlock);
            allocations += guard.count();
            for (int i = 0; i < kControlBlock; ++i)
            {
                REQUIRE(std::isfinite(l[static_cast<std::size_t>(i)]));
                peak = std::max({peak, std::abs(l[static_cast<std::size_t>(i)]),
                                 std::abs(r[static_cast<std::size_t>(i)])});
            }
        }
        INFO("rate " << rate << " peak " << peak);
        REQUIRE(allocations == 0);
        REQUIRE(peak < 50.0f);
    }
}

TEST_CASE("CONVOLUTION mono slot works", "[dsp][convspatial]")
{
    IrBank bank;
    bank.add(makeSyntheticIr({SyntheticIrKind::Plate, 48000.0, 0.5}));
    ConvolutionProcessor p;
    ProcessSpec spec;
    spec.channels = 1;
    p.prepare(spec);
    bank.publish(p);
    auto params = defaults();
    std::vector<float> x(9600, 0.0f);
    x[0] = 1.0f;
    ControlContext ctx;
    for (std::size_t pos = 0; pos < x.size(); pos += kControlBlock)
    {
        p.setParams(params.data(), ctx);
        float* ch[1] = {x.data() + pos};
        p.process(ch, kControlBlock);
    }
    REQUIRE(finite(x));
    REQUIRE(rms(x) > 1e-3);
}

// IR preparation -------------------------------------------------------------------------------

TEST_CASE("resampleIr keeps frequency and gain", "[dsp][convspatial]")
{
    const auto x = sine(1000.0, 44100.0, 44100);
    const auto y = resampleIr(x, 44100.0, 48000.0);
    REQUIRE(y.size() == 48000);
    const std::vector<float> mid(y.begin() + 1000, y.end() - 1000);
    REQUIRE(std::abs(zeroCrossingHz(mid, 48000.0) - 1000.0) < 0.5);
    // The taps are scaled by the rate ratio so the IR keeps its filter gain.
    REQUIRE(std::abs(toneAmplitude(mid, 1000.0, 48000.0) / (0.5 * 44100.0 / 48000.0) - 1.0) < 0.01);

    // A unit impulse keeps its DC gain (sum of taps) either way.
    for (double to : {32000.0, 96000.0})
    {
        std::vector<float> d(400, 0.0f);
        d[200] = 1.0f;
        const auto e = resampleIr(d, 48000.0, to);
        double sum = 0.0;
        for (float v : e)
            sum += v;
        INFO("to " << to);
        REQUIRE(std::abs(sum - 1.0) < 0.01);
    }
    REQUIRE(resampleIr(x, 44100.0, 44100.0) == x);
}

TEST_CASE("Synthetic IRs have the requested RT60 and are decorrelated", "[dsp][convspatial]")
{
    for (double rt60 : {0.5, 1.5})
    {
        const auto src = makeSyntheticIr({SyntheticIrKind::Decay, 48000.0, rt60, rt60 * 1.2, 2, 3});
        REQUIRE(src.channels.size() == 2);
        const auto& h = src.channels[0];
        // Schroeder backward integration, T20 fit between -5 and -25 dB.
        std::vector<double> edc(h.size() + 1, 0.0);
        for (std::size_t i = h.size(); i-- > 0;)
            edc[i] = edc[i + 1] + static_cast<double>(h[i]) * h[i];
        auto timeAt = [&](double db)
        {
            for (std::size_t i = 0; i < h.size(); ++i)
                if (10.0 * std::log10(edc[i] / edc[0]) <= db)
                    return static_cast<double>(i) / 48000.0;
            return 0.0;
        };
        const double measured = 3.0 * (timeAt(-25.0) - timeAt(-5.0));
        INFO("rt60 " << rt60 << " measured " << measured);
        REQUIRE(std::abs(measured / rt60 - 1.0) < 0.15);

        double lr = 0.0, ll = 0.0, rr = 0.0;
        for (std::size_t i = 0; i < h.size(); ++i)
        {
            lr += static_cast<double>(h[i]) * src.channels[1][i];
            ll += static_cast<double>(h[i]) * h[i];
            rr += static_cast<double>(src.channels[1][i]) * src.channels[1][i];
        }
        REQUIRE(std::abs(lr) / std::sqrt(ll * rr) < 0.1);
    }
    for (auto kind : {SyntheticIrKind::Room, SyntheticIrKind::Plate, SyntheticIrKind::Spring})
    {
        const auto src = makeSyntheticIr({kind, 48000.0, 1.0});
        REQUIRE(src.channels.size() == 2);
        REQUIRE(finite(src.channels[0]));
        REQUIRE(rms(src.channels[0]) > 1e-3);
        const auto ir = PreparedIr::create(src, 48000.0);
        REQUIRE(ir->length() == 48000);
    }
}

TEST_CASE("PreparedIr rejects bad sources and normalises to unit energy", "[dsp][convspatial]")
{
    REQUIRE_THROWS(PreparedIr::create(sourceOf({}), 48000.0));
    REQUIRE_THROWS(PreparedIr::create(sourceOf({{1.0f}, {1.0f}, {1.0f}}), 48000.0));
    REQUIRE_THROWS(PreparedIr::create(sourceOf({{}}), 48000.0));
    REQUIRE_THROWS(PreparedIr::create(sourceOf({{1.0f}}), 0.0));

    const auto ir = PreparedIr::create(sourceOf({randomSignal(20000, 1, 0.3f)}), 48000.0);
    PartitionedConvolver conv;
    conv.prepare(1, 48000);
    conv.setIr(ir.get());
    const auto x = randomSignal(96000, 2);
    const auto y = runConvolver(conv, {x});
    const std::vector<float> steady(y[0].begin() + 30000, y[0].end());
    const std::vector<float> in(x.begin() + 30000, x.end());
    REQUIRE(std::abs(rms(steady) / rms(in) - 1.0) < 0.05);
}

// Benchmark ------------------------------------------------------------------------------------

namespace
{
template <typename Fn>
void timeBlocks(const char* label, int blocks, Fn&& fn)
{
    double total = 0.0, worst = 0.0;
    for (int b = 0; b < blocks; ++b)
    {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const double us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        total += us;
        worst = std::max(worst, us);
    }
    std::printf("%-58s %8.2f us/128 avg  %8.2f us worst  (%.2f%% of a 2667 us block)\n", label,
                total / blocks, worst, 100.0 * total / blocks / 2666.7);
}
} // namespace

TEST_CASE("CONVOLUTION cost, 2 s stereo IR at 48 kHz, 128-sample blocks", "[.bench][dsp][convspatial]")
{
    const int blocks = 48000 * 10 / 128;
    for (double seconds : {2.0, 5.0})
    {
        IrBank bank;
        bank.add(makeSyntheticIr({SyntheticIrKind::Decay, 48000.0, seconds, seconds, 2}));
        ConvolutionProcessor p;
        p.prepare(stereoSpec());
        bank.publish(p);
        auto params = defaults();
        params[1] = 0.08f; // 20 ms pre-delay
        params[2] = 0.95f; // keep ~95 %
        params[3] = 0.2f;  // fade
        params[4] = 0.3f;  // low cut ~65 Hz
        params[5] = 0.8f;  // high cut ~12 kHz
        params[6] = 0.6f;  // width 1.2
        params[8] = 0.5f;  // mix
        std::vector<float> l = randomSignal(128, 1, 0.3f), r = randomSignal(128, 2, 0.3f);
        ControlContext ctx;
        char label[96];
        std::snprintf(label, sizeof label, "ConvolutionProcessor, %.0f s stereo IR (non-uniform)", seconds);
        timeBlocks(label, blocks,
                   [&]
                   {
                       for (int s = 0; s < 128; s += kControlBlock)
                       {
                           p.setParams(params.data(), ctx);
                           float* ch[2] = {l.data() + s, r.data() + s};
                           p.process(ch, kControlBlock);
                       }
                   });

        // Reference: two uniform 128-sample partitioned convolvers (one per channel).
        const auto src = makeSyntheticIr({SyntheticIrKind::Decay, 48000.0, seconds, seconds, 2});
        std::array<UniformPartitionedConvolver, 2> uni;
        for (int c = 0; c < 2; ++c)
            uni[static_cast<std::size_t>(c)].prepare(src.channels[static_cast<std::size_t>(c)].data(),
                                                     static_cast<int>(src.channels[0].size()), 128);
        std::snprintf(label, sizeof label, "2 x UniformPartitionedConvolver(128), %.0f s IR", seconds);
        std::vector<float> ol(128), orr(128);
        timeBlocks(label, blocks,
                   [&]
                   {
                       uni[0].process(l.data(), ol.data(), 128);
                       uni[1].process(r.data(), orr.data(), 128);
                   });
    }
}
