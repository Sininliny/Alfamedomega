// Everything the audio thread calls must not allocate. This binary replaces malloc and operator
// new to count allocations made on the calling thread while a guard is alive.

#include "AllocationGuard.h"
#include "TestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace alfa::inference;
using alfa::inference::test::AllocationGuard;

namespace
{

// One audio callback: what a plugin's processBlock does with the host.
void audioBlock(InferenceHost& host, int blockSize, int block)
{
    VoiceInput voice;
    voice.method = block % kNumMethods;
    voice.features[0] = 0.5f;
    voice.features[2] = 1.0f;
    for (int v = 0; v < kMaxVoices; ++v)
    {
        voice.method = (block + v) % (kNumMethods + 1) - 1; // include idle slots
        host.setVoice(v, voice);
    }
    if (block % 7 == 0)
        host.retrigger(block % kMaxVoices);
    const std::array<float, kGlobalModInputs> mod{0.1f, 0.2f, 0.3f, 0.4f};
    host.setGlobalMod(mod);
    for (const auto& e : host.process(blockSize))
        (void)e;
}

} // namespace

TEST_CASE("the guard sees allocations", "[inference][realtime]")
{
    AllocationGuard guard;
    auto* p = new int(3);
    delete p;
    REQUIRE(guard.count() >= 1);
}

TEST_CASE("audio-thread calls never allocate", "[inference][realtime]")
{
    const auto library = test::loadTestLibrary();
    const auto tier = GENERATE(Tier::Lite, Tier::Standard, Tier::Studio);
    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, tier), error));
    host.prepare(48000.0, 512);

    long allocations = 0;
    for (int block = 0; block < 400; ++block)
    {
        AllocationGuard guard;
        audioBlock(host, 64 + (block * 37) % 449, block); // varying block sizes up to 512
        allocations += guard.count();
    }
    {
        AllocationGuard guard;
        host.reset();
        allocations += guard.count();
    }
    REQUIRE(allocations == 0);
}

TEST_CASE("picking up a new tier on the audio thread does not allocate", "[inference][realtime]")
{
    const auto library = test::loadTestLibrary();
    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, Tier::Lite), error));
    host.prepare(48000.0, 256);
    for (int b = 0; b < 50; ++b)
        audioBlock(host, 256, b);

    const auto next = GENERATE(Tier::Standard, Tier::Pro);
    REQUIRE(host.loadTier(test::requireModel(library, next), error)); // message thread
    long allocations = 0;
    for (int b = 0; b < 50; ++b)
    {
        AllocationGuard guard;
        audioBlock(host, 256, b);
        allocations += guard.count();
    }
    REQUIRE(allocations == 0);
    host.releaseRetired(); // the old tier is freed here, on the message thread
}
