// RealtimeSanitizer (Clang 20+) run of the audio-thread API. Any allocation, lock, blocking call
// or system call made while inside a [[clang::nonblocking]] function aborts the test with a
// stack trace, including calls made inside anira, tanh-lib and ONNX Runtime.

#include "TestSupport.h"

#include <catch2/catch_test_macros.hpp>

using namespace alfa::inference;

namespace
{

void audioBlock(InferenceHost& host, int blockSize, int block) noexcept [[clang::nonblocking]]
{
    VoiceInput voice;
    voice.features[0] = 0.5f;
    voice.features[2] = 1.0f;
    for (int v = 0; v < kMaxVoices; ++v)
    {
        voice.method = (block + v) % (kNumMethods + 1) - 1;
        host.setVoice(v, voice);
    }
    if (block % 5 == 0)
        host.retrigger(block % kMaxVoices);
    const std::array<float, kGlobalModInputs> mod{0.1f, 0.2f, 0.3f, 0.4f};
    host.setGlobalMod(mod);
    for (const auto& e : host.process(blockSize))
        (void)e;
    if (block % 97 == 96)
        host.reset();
}

} // namespace

TEST_CASE("audio-thread API passes RealtimeSanitizer", "[inference][realtime][rtsan]")
{
    const auto library = test::loadTestLibrary();
    InferenceHost host;
    std::string error;
    REQUIRE(host.loadTier(test::requireModel(library, Tier::Lite), error));
    host.prepare(48000.0, 512);

    for (int block = 0; block < 300; ++block)
    {
        audioBlock(host, 64 + (block * 37) % 449, block);
        if (block == 150)
            REQUIRE(host.loadTier(test::requireModel(library, Tier::Standard), error)); // switch mid-stream
    }
    host.releaseRetired();
    SUCCEED("no real-time violations");
}
