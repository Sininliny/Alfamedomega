# Alfa::inference

Runs the neural model for one plugin instance and hands the DSP a control frame every 4 ms,
with a fixed latency the plugin reports to the host. JUCE-free static library.

Stack (ADR 0001): ONNX Runtime 1.26 (CPU, statically linked) inside anira 2.3.0, which owns the
inference thread pool, buffering and latency. Only the ONNX Runtime backend is built.

## How a frame is made

```
audio thread                     anira worker thread
------------                     -------------------
process(numSamples)
  at each 250 Hz boundary:
    push packed input frame ───► PipelineBackend::process
    (global mod + 16 voices x      instance_step           (once)
     8 features, method, retrigger) voice_core             (active voices batched)
                                    expert_<method>        (once per method in use)
    pop control frame ◄──────────  scatter outputs into 16 x VoiceFrame
  returns FrameEvent{offset, frame}
```

The frame the DSP receives at boundary n was computed from the inputs given at boundary
n - latencyFrames(). If a frame is not back in time, the previous one is held, `stats()`
counts it, and anira keeps the stream aligned.

`patch_encoder` runs on the message thread in `setPatch()`; its embedding reaches the worker
through a lock-free triple buffer.

## Using it from the plugin

```cpp
alfa::inference::ModelLibrary library;                       // message thread
library.load(alfa::inference::ModelLibrary::defaultModelsDir(), error);
const auto* model = library.findNearest(savedTier);           // tell the user if tier differs
host.loadTier(*model, error);
host.setPatch(knobs, method, error);

void prepareToPlay(double sr, int block)
{
    host.prepare(sr, block);
    setLatencySamples(host.latencySamples());
}

void processBlock(...)
{
    // Delay note events given to the DSP by host.latencySamples() so notes line up with frames.
    host.setVoice(v, {features, method});    // per voice
    host.retrigger(v);                        // on note-on / steal
    for (auto e : host.process(numSamples))
        dsp.setFrame(e.sampleOffset, e.frame); // 16 x alfa::dsp::VoiceFrame, voice-major
}

void setNonRealtime(bool offline) { host.setNonRealtime(offline); } // bounce: never hold frames
```

Tier switching: `loadTier()` builds and prepares the new tier on the message thread; the audio
thread swaps to it at its next block without allocating, and `releaseRetired()` frees the old
one. Latency changes with the tier, so switch with the transport stopped and call
`setLatencySamples` again (packaging plan section 3).

## Threading rules

| Thread | Calls |
|---|---|
| Message | `loadTier`, `prepare`, `setPatch`, `setNonRealtime`, `releaseRetired`, queries, destruction |
| Audio | `setVoice`, `retrigger`, `setGlobalMod`, `process`, `reset` (wait-free, no allocation, locks or syscalls) |
| anira workers | the ONNX graphs |

`prepare` and `setNonRealtime` must not run while `process` runs (JUCE already guarantees this).

## Latency (48 kHz, measured with the dummy models)

| Tier | Max inference / frame | Buffer | Reported latency |
|---|---|---|---|
| Lite | 2 ms | 128 | 2 frames, 8 ms |
| Standard | 3 ms | 128 | 3 frames, 12 ms |
| Pro | 3 ms | 128 | 3 frames, 12 ms |
| Studio | 3.5 ms | 512 | 5 frames, 20 ms |

Latency comes in whole frames. The declared maximum must stay under the 4 ms frame period: a
recurrent model runs its frames one after another, so a tier that needs longer than a frame per
frame can never keep up, whatever the buffering.

## Tests

```bash
cmake --preset core -B /tmp/alfa-inference && cmake --build /tmp/alfa-inference
ctest --test-dir /tmp/alfa-inference -R inference
```

The tests export dummy models at build time, which needs Python 3 with `numpy` and `onnx`
(`pip install numpy onnx`); without them the model tests are skipped.

- `alfa_inference_tests`: frame clock, manifest and SHA-256, graph contract checks, the
  pipeline's routing into VoiceFrame, and latency: for every tier, sample rate and block size,
  the reported latency equals the delay frames actually arrive with, frame for frame.
- `alfa_inference_alloc_tests`: counts heap allocations (malloc and operator new) on the audio
  thread across varying block sizes, retriggers, reset and a tier switch. Must be zero.
- `alfa_inference_rtsan_tests`: built when the compiler supports `-fsanitize=realtime`
  (Clang 20+, e.g. `CC=clang-20 CXX=clang++-20 cmake --preset core -B /tmp/alfa-rtsan`).
  RealtimeSanitizer aborts on any allocation, lock or system call inside the audio-thread calls,
  including inside anira and ONNX Runtime.
- `alfa_inference_tests "[.bench]"`: audio-thread cost per frame.

## Files

- `include/alfa/inference/InferenceHost.h`: the API above.
- `include/alfa/inference/ModelLibrary.h`: shared models folder, manifest, SHA-256, tier fallback.
- `include/alfa/inference/Tier.h`: per-tier inference budget, threads, smallest buffer.
- `include/alfa/inference/ControlFrameLayout.h`: maps ONNX output names onto `alfa/dsp/ControlFrame.h`.
- `include/alfa/inference/FrameClock.h`: the 250 Hz frame grid on the host's sample clock.
- `src/FramePipeline.*`: the ONNX Runtime sessions, recurrent state and output scatter.
- `MODEL_CONTRACT.md`: what the training export must produce.
- `tools/export_dummy_models.py`: toy models following that contract.
