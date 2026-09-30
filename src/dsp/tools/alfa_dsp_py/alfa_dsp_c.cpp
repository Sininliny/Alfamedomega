// C API over alfa::dsp::Synth for offline rendering from Python (ctypes; see alfa_dsp.py).
// Used by training (METHOD.md section 2b) to put the real DSP in the loss. Not for real time:
// render() takes whole frame sequences and may be called from any one thread per handle.

#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Synth.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#if defined(_WIN32)
#define ALFA_EXPORT extern "C" __declspec(dllexport)
#else
#define ALFA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

using namespace alfa::dsp;

namespace
{
struct Handle
{
    Synth synth;
    std::vector<float> grainData;
    GrainBank grains;
    std::vector<float> sampleData;
    SampleData sample;
    std::unique_ptr<VoiceFrame> frame = std::make_unique<VoiceFrame>();
    std::unique_ptr<InstanceFrame> instance = std::make_unique<InstanceFrame>();
};
} // namespace

ALFA_EXPORT int alfa_layout_version()
{
    return kLayoutVersion;
}
ALFA_EXPORT int alfa_frame_floats()
{
    return static_cast<int>(kFrameFloats);
}
ALFA_EXPORT int alfa_instance_frame_floats()
{
    return static_cast<int>(kInstanceFrameFloats);
}
ALFA_EXPORT int alfa_hop(double sampleRate)
{
    return hopSamples(sampleRate);
}

/// tier: 0 Lite, 1 Standard, 2 Pro, 3 Studio. Returns null on bad arguments.
ALFA_EXPORT void* alfa_create(double sampleRate, int tier)
{
    if (!(sampleRate >= 8000.0 && sampleRate <= 384000.0) || tier < 0 || tier > 3)
        return nullptr;
    auto* h = new Handle();
    h->synth.prepare(sampleRate, 1024, static_cast<Tier>(tier));
    return h;
}

ALFA_EXPORT void alfa_destroy(void* handle)
{
    delete static_cast<Handle*>(handle);
}

/// Copies count sources of length samples each (row-major). count 0 clears the bank.
ALFA_EXPORT void alfa_set_grain_bank(void* handle, const float* data, int count, int length)
{
    auto* h = static_cast<Handle*>(handle);
    h->grainData.assign(data, data + static_cast<std::size_t>(std::max(0, count) * std::max(0, length)));
    h->grains = GrainBank{h->grainData.data(), count, length};
    h->synth.setGrainBank(count > 0 ? &h->grains : nullptr);
}

/// Copies a mono sample. loopStart/loopEnd -1 for no loop.
ALFA_EXPORT void alfa_set_sample(void* handle, const float* data, long long length, double sampleRate,
                                 double rootHz, long long loopStart, long long loopEnd)
{
    auto* h = static_cast<Handle*>(handle);
    h->sampleData.assign(data, data + std::max(0LL, length));
    h->sample = SampleData{h->sampleData.data(), length, sampleRate, rootHz, loopStart, loopEnd};
    h->synth.setSample(length > 0 ? &h->sample : nullptr);
}

/// Renders one note from frame 0 for frames * hop samples into outLeft/outRight (each that long).
///   voiceFrames:    [frames, alfa_frame_floats()] float32, one per hop
///   instanceFrames: [frames, alfa_instance_frame_floats()] or null (default instance frame)
///   f0:             [frames] Hz per frame, or null (MIDI note pitch). Applied from that hop.
///   noteOffFrame:   frame at which the note is released, or -1 to hold
/// The synth is reset first, so every call is independent and deterministic. Returns the
/// number of samples written, or -1 on bad arguments.
ALFA_EXPORT long long alfa_render(void* handle, const float* voiceFrames, const float* instanceFrames,
                                  const float* f0, int frames, int note, float velocity, int noteOffFrame,
                                  float* outLeft, float* outRight)
{
    auto* h = static_cast<Handle*>(handle);
    if (h == nullptr || voiceFrames == nullptr || frames <= 0 || outLeft == nullptr || outRight == nullptr)
        return -1;
    Synth& s = h->synth;
    s.reset();
    const int hop = s.hop();
    const double noteHz = 440.0 * std::pow(2.0, (note - 69) / 12.0);

    int voice = -1;
    long long written = 0;
    for (int t = 0; t < frames; ++t)
    {
        std::memcpy(static_cast<void*>(h->frame.get()),
                    voiceFrames + static_cast<std::size_t>(t) * kFrameFloats, sizeof(VoiceFrame));
        if (instanceFrames != nullptr)
        {
            std::memcpy(static_cast<void*>(h->instance.get()),
                        instanceFrames + static_cast<std::size_t>(t) * kInstanceFrameFloats,
                        sizeof(InstanceFrame));
            s.setInstanceFrame(*h->instance);
        }
        if (f0 != nullptr && std::isfinite(f0[t]) && f0[t] > 0.0f)
            s.setPitchBend(static_cast<float>(12.0 * std::log2(static_cast<double>(f0[t]) / noteHz)));
        if (t == 0)
            voice = s.noteOn(note, velocity, h->frame.get());
        else
            s.setFrame(voice, *h->frame);
        if (t == noteOffFrame)
            s.noteOff(note);
        s.process(outLeft + written, outRight + written, hop);
        written += hop;
    }
    return written;
}
