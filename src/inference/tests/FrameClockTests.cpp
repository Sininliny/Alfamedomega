#include "alfa/inference/FrameClock.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <vector>

using alfa::inference::FrameClock;

TEST_CASE("frame boundaries land every hop samples across blocks", "[inference][clock]")
{
    const double sampleRate = GENERATE(44100.0, 48000.0, 96000.0);
    const int block = GENERATE(1, 32, 64, 128, 176, 441, 512, 1024);
    FrameClock clock;
    clock.prepare(sampleRate);
    REQUIRE(clock.hop() == alfa::dsp::hopSamples(sampleRate));

    std::vector<long long> boundaries;
    long long blockStart = 0;
    for (int b = 0; b < 400; ++b)
    {
        const int count = clock.advance(block,
                                        [&](int offset)
                                        {
                                            REQUIRE(offset >= 0);
                                            REQUIRE(offset < block);
                                            boundaries.push_back(blockStart + offset);
                                        });
        REQUIRE(count <= clock.maxFramesPerBlock(block));
        blockStart += block;
    }

    REQUIRE(!boundaries.empty());
    for (std::size_t n = 0; n < boundaries.size(); ++n)
        REQUIRE(boundaries[n] == static_cast<long long>(n) * clock.hop());
    REQUIRE(static_cast<long long>(boundaries.size()) == (blockStart - 1) / clock.hop() + 1);
    REQUIRE(clock.framesElapsed() == static_cast<long long>(boundaries.size()));
}

TEST_CASE("hop matches the DSP at common rates", "[inference][clock]")
{
    FrameClock clock;
    clock.prepare(48000.0);
    REQUIRE(clock.hop() == 192);
    REQUIRE(clock.maxFramesPerBlock(128) == 1);
    REQUIRE(clock.maxFramesPerBlock(192) == 1);
    REQUIRE(clock.maxFramesPerBlock(193) == 2);
    REQUIRE(clock.maxFramesPerBlock(512) == 3);
    clock.prepare(44100.0);
    REQUIRE(clock.hop() == 176);
}

TEST_CASE("reset restarts the frame grid", "[inference][clock]")
{
    FrameClock clock;
    clock.prepare(48000.0);
    clock.advance(1000, [](int) {});
    clock.reset();
    std::vector<int> offsets;
    clock.advance(400, [&](int o) { offsets.push_back(o); });
    REQUIRE(offsets == std::vector<int>{0, 192, 384});
}
