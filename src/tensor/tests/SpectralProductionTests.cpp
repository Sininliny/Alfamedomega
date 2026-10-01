#include "alfa/tensor/SpectralProduction.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace
{
using namespace alfa::tensor;
template <class S>
struct Stream
{
    PrimitiveT<S> a, b;
    std::vector<S> as, bs, frame, oldFrame, out, window, normal;
    std::vector<StateWriteT<S>> aj, bj;
    std::vector<S> scratch;
    std::array<TensorViewT<S>, 2> assets{};
    std::array<S, 1> hop;
    explicit Stream(std::size_t n, std::size_t h)
        : as(n + 4)
        , bs(n + 3)
        , frame(2 * n + 1)
        , oldFrame(2 * n + 1)
        , out(1)
        , window(n)
        , normal(n)
        , hop{static_cast<S>(h)}
    {
        std::string error;
        REQUIRE(spectralProduction::createPrimitive("spectral.stft_analysis.v1", n + 4, a, error));
        REQUIRE(spectralProduction::createPrimitive("spectral.stft_synthesis.v1", n + 3, b, error));
        aj.resize(a.sparseWriteCapacity);
        bj.resize(b.sparseWriteCapacity);
        scratch.resize(b.scratchElements);
        for (std::size_t i = 0; i < n; ++i)
            window[i] = std::sqrt(S(.5) - S(.5) * std::cos(S(2) * std::numbers::pi_v<S> * static_cast<S>(i) /
                                                           static_cast<S>(n)));
        for (std::size_t i = 0; i < n; ++i)
        {
            S sum = 0;
            for (std::size_t j = i % h; j < n; j += h)
                sum += window[j] * window[j];
            normal[i] = 1 / sum;
        }
        assets[0].values = window;
        assets[1].values = normal;
    }
    S tick(S x, S gain = 1)
    {
        StepT<S> s;
        s.clock = {1, 0};
        s.sampleRate = 48000;
        s.activation = 1;
        s.assets = assets;
        s.localState = bs;
        s.inputs.audio.values = oldFrame;
        s.scratch = scratch;
        b.readout(s, out);
        auto count = b.sparseTransition(s, bj);
        REQUIRE(count <= bj.size());
        for (std::size_t i = 0; i < count; ++i)
        {
            REQUIRE(std::isfinite(bj[i].value));
            bs[bj[i].element] = bj[i].value;
        }
        const std::array<S, 1> input{x};
        s.localState = as;
        s.inputs.audio.values = input;
        s.assets = {assets.data(), 1};
        s.parameters = hop;
        a.readout(s, frame);
        count = a.sparseTransition(s, aj);
        REQUIRE(count == 5);
        for (std::size_t i = 0; i < count; ++i)
            as[aj[i].element] = aj[i].value;
        for (std::size_t i = 0; i + 1 < frame.size(); ++i)
            frame[i] *= gain;
        oldFrame = frame;
        return out[0];
    }
    void reset()
    {
        std::fill(as.begin(), as.end(), S(0));
        std::fill(bs.begin(), bs.end(), S(0));
        std::fill(oldFrame.begin(), oldFrame.end(), S(0));
    }
};
template <class S>
void roundtrip()
{
    for (const std::size_t n : {8u, 32u, 128u})
        for (const auto h : {n / 2, n / 4})
        {
            Stream<S> stream(n, h);
            std::vector<S> input(1000), output;
            for (std::size_t i = 0; i < 500; ++i)
                input[i] = S(.3) * std::sin(S(.071) * static_cast<S>(i)) +
                           S(.2) * std::cos(S(.17) * static_cast<S>(i));
            for (const S x : input)
                output.push_back(stream.tick(x));
            const auto latency = n + 1;
            for (std::size_t i = 0; i < 500; ++i)
                REQUIRE(std::abs(output[i + latency] - input[i]) <
                        (std::is_same_v<S, float> ? S(3e-6) : S(3e-14)));
            stream.reset();
            for (std::size_t i = 0; i < input.size(); ++i)
                REQUIRE(stream.tick(input[i]) == output[i]);
            stream.reset();
            for (std::size_t i = 0; i < input.size(); ++i)
            {
                const S y = stream.tick(input[i], S(.25));
                if (i >= latency && i < 500 + latency)
                    REQUIRE(std::abs(y - S(.25) * input[i - latency]) <
                            (std::is_same_v<S, float> ? S(1e-6) : S(1e-14)));
            }
        }
}
} // namespace
TEST_CASE("streaming tensor STFT COLA roundtrip and spectrum gain float32", "[spectral-production]")
{
    roundtrip<float>();
}
TEST_CASE("streaming tensor STFT COLA roundtrip and spectrum gain float64", "[spectral-production]")
{
    roundtrip<double>();
}
TEST_CASE("streaming spectral allocation contract rejects malformed sizes", "[spectral-production]")
{
    PrimitiveT<double> p;
    std::string error;
    REQUIRE_FALSE(spectralProduction::createPrimitive("spectral.stft_analysis.v1", 17, p, error));
    REQUIRE(spectralProduction::createPrimitive("spectral.stft_analysis.v1", 1028, p, error));
    REQUIRE(p.sparseWriteCapacity == 5);
    REQUIRE(p.outputElements == 2049);
    REQUIRE(spectralProduction::createPrimitive("spectral.stft_synthesis.v1", 1027, p, error));
    REQUIRE(p.sparseWriteCapacity == 1027);
    REQUIRE(p.scratchElements == 2048);
    REQUIRE(p.outputElements == 1);
    REQUIRE(p.transition == nullptr);
}
TEST_CASE("STFT recurrent output history composes transactionally with independent clocks and reset",
          "[spectral-production]")
{
    auto run = []<class S>()
    {
        constexpr std::size_t n = 32;
        Stream<S> reference(n, 8);
        ConfigurationT<S> config;
        config.activation = {{"primitive"}, {2}, {1, 1}};
        config.parameters = {{"hop"}, {1}, {8}};
        config.routing = {{"coefficient"}, {1}, {1}};
        config.modulation = {{"relationship"}, {0}, {}};
        InstanceT<S> a{&reference.a, 0, {0, 1}, {1, 0}}, b{&reference.b, 1, {1, 0}, {1, 0}};
        TensorT<S> window{{"sample"}, {n}, reference.window}, normal{{"sample"}, {n}, reference.normal};
        a.assets = {window};
        b.assets = {window, normal};
        a.inputChannels = 1;
        b.inputChannels = 2 * n + 1;
        a.audioBindings.push_back({SourceKind::Audio, 0, 0, WeightTensor::Routing, 0, MissingSource::Zero});
        for (std::size_t i = 0; i < 2 * n + 1; ++i)
            b.audioBindings.push_back(
                {SourceKind::OutputHistory, i, i, WeightTensor::Routing, 0, MissingSource::Reject});
        config.instances = {a, b};
        config.initialState = {std::vector<S>(n + 4), std::vector<S>(n + 3)};
        Budget budget{2, 2 * n + 7, 1, 1, 0, 2 * n + 2, 5 * n + 6, 2 * n + 2, 5 + n + 3, 2 * n};
        RuntimeT<S> engine;
        std::string error;
        const bool prepared = engine.prepare(config, budget, 48000, error);
        INFO(error);
        REQUIRE(prepared);
        const std::array<std::string, 1> axes{"channel"};
        const std::array<std::size_t, 1> shape{1};
        std::array<S, 1> sample{0};
        InputsT<S> inputs;
        inputs.audio = {axes, shape, sample};
        std::vector<S> result;
        // Segment boundaries affect neither clock phase nor capture/OLA state.
        std::size_t t = 0;
        for (const std::size_t block : {1u, 19u, 64u, 7u, 128u, 37u})
            for (std::size_t i = 0; i < block; ++i, ++t)
            {
                sample[0] = t < 100 ? S(.2) * std::sin(S(.13) * static_cast<S>(t)) : S(0);
                REQUIRE(engine.tick(inputs));
                result.push_back(engine.output()[2 * n + 1]);
                REQUIRE(result.back() == reference.tick(sample[0]));
            }
        const auto storage = engine.numericStorageBytes();
        REQUIRE(storage > sizeof(S) * (2 * n + 7));
        engine.reset();
        reference.reset();
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            sample[0] = i < 100 ? S(.2) * std::sin(S(.13) * static_cast<S>(i)) : S(0);
            REQUIRE(engine.tick(inputs));
            REQUIRE(engine.output()[2 * n + 1] == result[i]);
        }
        const std::vector<S> saved(engine.state().begin(), engine.state().end());
        sample[0] = std::numeric_limits<S>::infinity();
        REQUIRE_FALSE(engine.tick(inputs));
        REQUIRE(std::equal(saved.begin(), saved.end(), engine.state().begin()));
        config.instances[0].clock = {2, 1};
        config.instances[0].outputBetweenUpdates = OutputBetweenUpdates::Hold;
        config.instances[1].clock = {2, 1};
        RuntimeT<S> slower;
        REQUIRE(slower.prepare(config, budget, 48000, error));
        reference.reset();
        for (std::size_t i = 0; i < 256; ++i)
        {
            sample[0] = S(.3) * std::sin(S(.11) * static_cast<S>(i));
            REQUIRE(slower.tick(inputs));
            if (i % 2 == 1)
                REQUIRE(slower.output()[2 * n + 1] == reference.tick(sample[0]));
            else
                REQUIRE(slower.output()[2 * n + 1] == 0);
        }
        --budget.sparseStateWrites;
        RuntimeT<S> limited;
        REQUIRE_FALSE(limited.prepare(config, budget, 48000, error));
        ++budget.sparseStateWrites;
        --budget.scratchElements;
        REQUIRE_FALSE(limited.prepare(config, budget, 48000, error));
        return storage;
    };
    const auto f32 = run.operator()<float>(), f64 = run.operator()<double>();
    REQUIRE(f64 == 2 * f32);
}
TEST_CASE("STFT double frame retains coefficient distinctions and matches independent DFT",
          "[spectral-production]")
{
    Stream<double> stream(8, 8);
    std::fill(stream.window.begin(), stream.window.end(), 1.0 + 1e-12);
    for (std::size_t i = 0; i < 8; ++i)
        stream.tick(i == 0 ? 1.0 : 0.0);
    REQUIRE(stream.frame[16] == 1);
    for (std::size_t k = 0; k < 8; ++k)
    {
        REQUIRE(stream.frame[2 * k] > 1.0);
        REQUIRE(stream.frame[2 * k] == 1.0 + 1e-12);
        REQUIRE(stream.frame[2 * k + 1] == 0);
    }
    stream.reset();
    for (std::size_t i = 0; i < 8; ++i)
        stream.tick(static_cast<double>(i + 1));
    for (std::size_t k = 0; k < 8; ++k)
    {
        double re = 0, im = 0;
        for (std::size_t i = 0; i < 8; ++i)
        {
            const double angle = -2 * std::numbers::pi * static_cast<double>(k * i) / 8;
            re += static_cast<double>(i + 1) * stream.window[i] * std::cos(angle);
            im += static_cast<double>(i + 1) * stream.window[i] * std::sin(angle);
        }
        REQUIRE(std::abs(stream.frame[2 * k] - re) < 1e-13);
        REQUIRE(std::abs(stream.frame[2 * k + 1] - im) < 1e-13);
    }
}
TEST_CASE("STFT invalid frame and hop reject without changing persistent state", "[spectral-production]")
{
    Stream<double> stream(8, 2);
    std::array<double, 17> frame{};
    frame[16] = .5;
    Step64 s;
    s.clock = {1, 0};
    s.sampleRate = 48000;
    s.activation = 1;
    s.assets = stream.assets;
    s.localState = stream.bs;
    s.inputs.audio.values = frame;
    REQUIRE(stream.b.sparseTransition(s, stream.bj) > stream.bj.size());
    frame[16] = 1;
    frame[2] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(stream.b.sparseTransition(s, stream.bj) > stream.bj.size());
    const std::array<double, 1> badHop{2.5};
    s.localState = stream.as;
    s.assets = {stream.assets.data(), 1};
    s.parameters = badHop;
    REQUIRE(stream.a.sparseTransition(s, stream.aj) > stream.aj.size());
    REQUIRE(std::all_of(stream.as.begin(), stream.as.end(), [](double x) { return x == 0; }));
}
