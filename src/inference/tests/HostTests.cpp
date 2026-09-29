#include "TestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace alfa::inference;
namespace dsp = alfa::dsp;

namespace
{

struct Run
{
    std::vector<float> in;  // pass-through input value used at each frame boundary
    std::vector<float> out; // pass-through value of the frame received at that boundary
    std::vector<int> offsets;
    std::uint64_t late = 0;
};

// Drives voice 0 with a pitch value that changes every block, and records what comes back.
Run drive(InferenceHost& host, int blockSize, int numBlocks, double sampleRate, bool paceRealtime)
{
    Run run;
    VoiceInput voice;
    voice.method = 0;
    voice.features[2] = 1.0f;
    const auto blockDuration = std::chrono::duration<double>(blockSize / sampleRate);
    auto next = std::chrono::steady_clock::now();
    const auto lateBefore = host.stats().framesLate;
    for (int b = 0; b < numBlocks; ++b)
    {
        const float value = static_cast<float>(b + 1) * 1e-3f;
        voice.features[0] = value;
        host.setVoice(0, voice);
        for (const auto& e : host.process(blockSize))
        {
            run.in.push_back(value);
            run.out.push_back(test::passThrough(e.frame, 0));
            run.offsets.push_back(e.sampleOffset);
        }
        if (paceRealtime)
        {
            next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(blockDuration);
            std::this_thread::sleep_until(next);
        }
    }
    run.late = host.stats().framesLate - lateBefore;
    return run;
}

// Fraction of frames n >= lag whose value is the input of frame n - lag.
double matchRate(const Run& run, int lag)
{
    int matches = 0, total = 0;
    for (std::size_t n = static_cast<std::size_t>(lag); n < run.out.size(); ++n, ++total)
        matches += run.out[n] == run.in[n - static_cast<std::size_t>(lag)] ? 1 : 0;
    return total ? static_cast<double>(matches) / total : 0.0;
}

} // namespace

TEST_CASE("reported latency is exactly the delay the frames arrive with", "[inference][latency]")
{
    const auto library = test::loadTestLibrary();
    const auto tier = GENERATE(Tier::Lite, Tier::Standard, Tier::Pro, Tier::Studio);
    const auto sampleRate = GENERATE(44100.0, 48000.0, 96000.0);
    const auto blockSize = GENERATE(64, 128, 256, 512);
    if (blockSize < spec(tier).smallestBufferSupported)
        return; // below the tier's smallest supported buffer

    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, tier), error));
    host.setNonRealtime(true); // deterministic: wait for every frame
    host.prepare(sampleRate, blockSize);

    const int hop = dsp::hopSamples(sampleRate);
    const int lag = host.latencyFrames();
    REQUIRE(lag >= 1);
    REQUIRE(host.latencySamples() == lag * hop);

    const auto run = drive(host, blockSize, 300, sampleRate, false);
    CAPTURE(toString(tier), sampleRate, blockSize, lag);
    REQUIRE(run.late == 0);
    // The first `lag` frames are the latency padding, then every frame is exactly `lag` late.
    for (int n = 0; n < lag; ++n)
        REQUIRE(run.out[static_cast<std::size_t>(n)] == 0.0f);
    REQUIRE(matchRate(run, lag) == 1.0);
}

TEST_CASE("latency per tier stays within the tier spec", "[inference][latency]")
{
    const auto library = test::loadTestLibrary();
    // Measured with anira 2.3.0; the tier spec's estimate was 6 / 8 / 8 / 12 ms, which leaves
    // out that latency comes in whole 4 ms frames and that a host block can bring several frames.
    const std::array<double, 4> specMs{8.0, 12.0, 12.0, 20.0};
    for (const auto& s : kTiers)
    {
        InferenceHost host;
        std::string error;
        REQUIRE(host.loadTier(test::requireModel(library, s.tier), error));
        const int block = std::max(128, s.smallestBufferSupported);
        host.prepare(48000.0, block);
        const double ms = host.latencySamples() / 48.0;
        std::printf("latency %-8s block %4d: %d frames, %d samples, %.1f ms\n", std::string(s.name).c_str(),
                    block, host.latencyFrames(), host.latencySamples(), ms);
        CAPTURE(s.name, ms);
        REQUIRE(ms <= specMs[static_cast<std::size_t>(s.tier)]);
    }
}

TEST_CASE("paced like a real host, frames keep up and stay aligned", "[inference][latency]")
{
    const auto library = test::loadTestLibrary();
    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, Tier::Lite), error));
    host.prepare(48000.0, 128);
    const auto run = drive(host, 128, 190, 48000.0, true); // about half a second
    CAPTURE(run.late, run.out.size());
    REQUIRE(static_cast<double>(run.late) <= 0.05 * static_cast<double>(run.out.size()));
    REQUIRE(matchRate(run, host.latencyFrames()) >= 0.9);
}

TEST_CASE("without a model the host still ticks and hands out silent frames", "[inference][host]")
{
    InferenceHost host;
    REQUIRE_FALSE(host.hasModel());
    host.prepare(48000.0, 512);
    const auto events = host.process(512);
    REQUIRE(events.size() == 3);
    for (const auto& e : events)
        for (std::size_t i = 0; i < kFloatsPerFrame; ++i)
            REQUIRE(e.frame[i] == 0.0f);
    REQUIRE(host.latencySamples() == 0);
}

TEST_CASE("switching tier takes effect at the next block and changes latency", "[inference][host]")
{
    const auto library = test::loadTestLibrary();
    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, Tier::Lite), error));
    host.setNonRealtime(true);
    host.prepare(48000.0, 512);
    const int liteLatency = host.latencySamples();
    drive(host, 512, 20, 48000.0, false);

    REQUIRE(host.loadTier(test::requireModel(library, Tier::Studio), error));
    REQUIRE(host.tier() == Tier::Studio);
    const int studioLatency = host.latencySamples();
    CAPTURE(liteLatency, studioLatency);
    REQUIRE(studioLatency >= liteLatency);

    const auto run = drive(host, 512, 40, 48000.0, false);
    host.releaseRetired();
    REQUIRE(matchRate(run, host.latencyFrames()) > 0.9);
}

TEST_CASE("a model whose files were modified is refused and the old tier keeps playing", "[inference][host]")
{
    const auto library = test::loadTestLibrary();
    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, Tier::Lite), error));

    auto tampered = test::requireModel(library, Tier::Standard);
    tampered.files["voice_core.onnx"].sha256 = std::string(64, '0');
    REQUIRE_FALSE(host.loadTier(tampered, error));
    REQUIRE(error.find("checksum") != std::string::npos);
    REQUIRE(host.tier() == Tier::Lite);
}

TEST_CASE("reset restarts the stream with the latency padding", "[inference][host]")
{
    const auto library = test::loadTestLibrary();
    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, Tier::Standard), error));
    host.setNonRealtime(true);
    host.prepare(48000.0, 192);
    drive(host, 192, 50, 48000.0, false);
    host.reset();
    const auto run = drive(host, 192, 50, 48000.0, false);
    REQUIRE(run.offsets.front() == 0);
    REQUIRE(run.out.front() == 0.0f);
    REQUIRE(matchRate(run, host.latencyFrames()) == 1.0);
}

TEST_CASE("audio-thread cost per frame", "[inference][.bench]")
{
    const auto library = test::loadTestLibrary();
    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, Tier::Standard), error));
    host.prepare(48000.0, 192);
    VoiceInput voice;
    voice.method = 0;
    for (int v = 0; v < kMaxVoices; ++v)
        host.setVoice(v, voice);

    // One frame per call, paced at the frame rate; time only the process() call.
    std::vector<double> micros;
    auto next = std::chrono::steady_clock::now();
    for (int b = 0; b < 500; ++b)
    {
        const auto t0 = std::chrono::steady_clock::now();
        host.process(192);
        micros.push_back(
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        next += std::chrono::microseconds(4000);
        std::this_thread::sleep_until(next);
    }
    std::sort(micros.begin(), micros.end());
    std::printf(
        "process() per frame, 16 voices: median %.1f us, p99 %.1f us, max %.1f us (frame period 4000 us)\n",
        micros[micros.size() / 2], micros[micros.size() * 99 / 100], micros.back());
}
