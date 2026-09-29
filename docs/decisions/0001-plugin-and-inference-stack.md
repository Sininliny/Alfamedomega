# ADR 0001 — Plugin framework and inference stack

- **Status:** Proposed
- **Date:** 2026-09-29
- **Context:** Alfamedomega, a neural synthesiser shipping as VST3/AU (and CLAP), with several model-size tiers

## Decision

Build the plugin on **JUCE 8**, and run the neural model with **ONNX Runtime (CPU execution provider) wrapped in [anira](https://github.com/anira-project/anira)**.

Keep **RTNeural** as an optional second backend for the smallest tier, and only if that tier's graph stays inside the layer types RTNeural supports.

## Why

### The real-time problem comes first

Neither ONNX Runtime nor LibTorch is safe to call from the audio callback: both allocate, lock, and manage their own thread pools, so a plain `session->Run()` inside `processBlock` will eventually produce a dropout that no amount of tuning removes. Any stack we pick has to answer this, not just "which engine is fastest".

anira is the answer we would otherwise have to write ourselves. It puts inference on a static, pre-allocated thread pool sized at `prepare()` time, converts the resulting delay into a fixed, reportable latency in samples (you declare a maximum inference time in ms and it derives the buffering), and its real-time safety is validated with the RealtimeSanitizer rather than asserted. It is Apache-2.0, ships JUCE VST3 and CLAP examples, builds for macOS/Windows/Linux/iOS/Android/WASM, and includes a benchmarking harness we can point at each model tier. See the [ANIRA paper](https://arxiv.org/abs/2506.12665) for the architecture and measurements.

### Why ONNX Runtime as the engine

- **The architecture is not settled yet.** A synth spanning subtractive, FM, wavetable, granular, spectral and physical-modelling behaviour will very likely grow convolutional, recurrent and attention blocks before it stabilises. ONNX takes whatever PyTorch exports.
- **Model tiers are a runtime concern, not a compile-time one.** ONNX models are data files: a tier becomes a file we load, and adding or re-training a tier ships without a rebuild.
- **Binary size is shippable.** A CPU-only ONNX Runtime build is tens of megabytes; LibTorch is in the hundreds, which is painful inside a plugin bundle that has to be codesigned and notarised.
- **Escape hatches exist.** Graph optimisation, quantisation, and later a CoreML or DirectML execution provider are configuration rather than a rewrite.

### Why not the alternatives

| Option | Verdict |
|---|---|
| **LibTorch** | Rejected. No inference advantage over ONNX Runtime, much larger bundle, heavier dependency and codesigning burden. Its value is training parity, and training does not ship. |
| **RTNeural** | Not the primary. It is the only option here that is genuinely real-time-safe by construction — no allocation, no thread pool, templated and static — and it is the right tool for a small, fixed dense/GRU/LSTM graph. But it supports Dense, GRU, LSTM, Conv1D/2D and BatchNorm only: no attention or transformer layers, and its own docs flag 2-D I/O and stateless Conv1D as still developing. Models arrive as JSON and the fast path wants shapes known at compile time, which fights against swappable tiers. Committing to it now means letting it constrain the model design. |
| **iPlug2** | Rejected, with regret. Permissively licensed and lighter than JUCE, which is real money saved, but a smaller community, a thinner AUv3/AAX story, and much less prior art for the GUI surface a synth of this scope needs. |
| **TFLite / LiteRT, ExecuTorch** | Not now. Both are anira backends already, so either stays available if mobile or a smaller runtime becomes a requirement. Neither beats ONNX for desktop CPU today. |

### On JUCE's cost

JUCE is dual-licensed: AGPLv3, or a commercial licence — Starter is free under $20k revenue, Indie is $800 perpetual up to $300k, Pro is $3,500 perpetual. That is a known, bounded cost, and it buys VST3 + AU + AAX + standalone from one codebase plus the widest body of examples and hiring pool in the field. iPlug2's saving is not worth the time it costs us elsewhere.

## Consequences

- Inference runs off the audio thread with a declared, fixed latency. This is acceptable for a synth but must be reported to the host, and note-on response has to be designed around it rather than discovered late.
- **Strong recommendation for the model design, not yet a decision:** have the network emit control or spectral frames at a low rate (roughly 50–250 Hz) and render audio on the audio thread with lightweight hand-written DSP — oscillators, filters, noise, convolution — in the DDSP manner. This moves the per-sample cost out of the network entirely and makes the real-time budget a question of frame rate rather than sample rate. It changes the inference burden by orders of magnitude and should be settled before the model is scaled up. Worth its own ADR.
- CPU-only for v1. GPU execution providers add driver-dependent, non-deterministic latency; revisit once the CPU path is measured.
- Training stays in PyTorch, with ONNX export as a required, tested step in the pipeline — including an exportability check in CI, because an op that does not export is discovered too late otherwise.
- anira is a young library and a dependency on the audio-thread critical path. Mitigation: it is Apache-2.0 and small enough to fork, and its backend abstraction means dropping to raw ONNX Runtime is a contained change.
- Build system is CMake across all three platforms, so anira, ONNX Runtime and JUCE compose without a bespoke build layer.

## Open questions

1. Frame-rate architecture (above) — settle before scaling the model.
2. What maximum inference time in ms each tier declares, which sets the plugin's reported latency per tier.
3. Whether tiers are separate ONNX files or one graph with a width parameter.
4. CLAP from day one, or after VST3/AU.
