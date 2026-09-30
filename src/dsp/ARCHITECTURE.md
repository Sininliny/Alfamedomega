# Alfa::dsp architecture

`src/dsp` is the JUCE-free real-time engine. A neural network writes control frames at 250 Hz, and
this library turns them into audio. The network decides everything musical: which primitives
sound, which processing families run, in what order, how much of each reaches the output, and how
parameters move between frames. The DSP has no fixed signal chain.

## Signal path

```
per voice (mono inside, up to limits.polyphony voices)
  six primitives: OSC (unison reads), PARTIALS, FM, NOISE, GRAIN, SAMPLER
        │  × attack envelope 1 − exp(−t/τ)
        ├──────────────► dry = Σ mix[p] · primitive_p
        ▼
  voice slots 0..voiceSlots−1   (any of FILTER…BODY, or Off)
        input_k = Σ_p input[p] · primitive_p + Σ_{i<k} input[6+i] · slot_i
  voice out = pan( dryGain · dry + Σ_k outGain_k · slot_k )     equal-power, stereo
        │
        ▼  Σ voices → stereo bus
instance chain (its own hop grid, from prepare)
  instance slots 0..instanceSlots−1   (any family)
        input_k = input[0] · bus + Σ_{i<k} input[1+i] · slot_i
  out = safety( masterGain · width( dryGain · bus + Σ_k outGain_k · slot_k ) )
```

The routing is lower-triangular, so any DAG over the slot indices can be expressed. A family can
appear in any position, more than once, or not at all. Slots whose family is Off cost nothing.

- **Hop and control blocks.** Frames arrive every hop (`round(sr/250)` samples). Inside a hop the
  voice works in control blocks of at most 32 samples (`kControlBlock`). For each block it
  interpolates frame j−1 → j to the block's end, applies the modulation matrix, calls
  `setParams` on each running slot, then renders. Gains ramp per sample within the block.
- **Family switching.** When a slot's family changes, the new processor is reset, and the old one
  keeps running and crossfades out across the hop.
- **Modulation.** Each voice has two DAHDSR envelopes, two LFOs, sample-and-hold, smoothed random,
  an envelope follower, velocity and key. The instance has only the LFO and random sources. Up to
  `limits.modRoutes` routes add `depth · source` to any slot param, slot out-gain, primitive mix,
  pitch (applied per hop), pan or dry gain. Every destination is clamped and sanitised
  afterwards.
- **Safety stage.** Output is linear up to +6 dBFS and saturates with tanh to at most +12 dBFS.
  Stacked high-Q slots can legitimately reach levels far above 0 dBFS, and this stage keeps a bad
  frame from reaching speakers at that level. Training renders through the same `Synth`, so the
  stage is part of the instrument.

## Families

| id | family | where | summary |
|---|---|---|---|
| 1 | filter | voice, instance | TPT SVF, LP/BP/HP/notch morph, 12/24 dB, bell and shelves, drive, keytrack |
| 2 | comb | voice, instance | feedforward/feedback comb, keytracked, damped loop |
| 3 | formant | voice, instance | 5 parallel band-passes, vowel and voice morph |
| 4 | resonator | voice, instance | modal bank: string, stiff string, bar, membrane, plate, tube |
| 5 | nonlinear | voice, instance | 8 shapes (soft…fold…rectify), bias, bits, rate, 1/2/4× oversampling |
| 6 | waveguide | voice, instance | Karplus-Strong/waveguide string: dispersion, pickup, tension |
| 7 | mod_delay | voice, instance | chorus, flanger, vibrato, 1–4 voices |
| 8 | phaser | voice, instance | 2–12 all-pass stages, feedback, stereo phase |
| 9 | body | voice, instance | guitar, violin, drum-shell or box resonance (8 peaks + shelf) |
| 10 | delay | instance | stereo/ping-pong, filtered feedback, wow, diffusion |
| 11 | reverb | instance | 8-line FDN with diffusers, damping, freeze |
| 12 | convolution | instance | zero-latency non-uniform partitioned convolution, IR bank |
| 13 | spectral | instance | STFT: band shape, gate, sparsify, freeze, smear, blur, shift, phase |
| 14 | pitch_shift | instance | grain pitch shifter with correlation alignment |
| 15 | spatial | instance | pan laws, width, M/S, rotation, Haas, mono-maker, distance |

Each family implements `Processor` (`Processor.h`) and publishes a 16-entry `ParamTable`: a
name, range, curve, neutral default and unit for each param. Param i of a slot is entry i. The
network always writes normalised 0..1 values, and `mapParam` maps them to physical units.
Everything is exported to `layout.json`. Voice slots run mono with `ctx.f0` set to the voice
pitch. Instance slots run stereo with `ctx.f0 = 0`, so key-tracked families fall back to their
frequency param.

Offline-only tools live beside the families and never run on the audio thread:
`offline::timeStretch`, `offline::resample` and `offline::pitchShiftOffline` (in PitchTime.h),
the offline sinc resampler (Resample.h), `IrBank` preparation and `makeSyntheticIr`
(Convolution.h). The real-time polyphase resampler is also in Resample.h.

### Extending

- **New family.** Add it to the end of `Family` (never renumber), implement `Processor`, and
  register it in `Family.cpp`. Voice families must sit at or below `Body` in the enum, or voice
  slots will not accept them; otherwise it is instance-only. Add tests in the style of the
  existing `tests/*Tests.cpp` and a `[.bench]` case.
- **Membranes and plates** go through a new mode table or class that reuses
  `detail::ModalBank` (Physical.h).
- **Ambisonics and other multichannel spatial renderers** need a new family (see Spatial.h).
- **Frame layout.** Changes only append, and bump `kLayoutVersion`. Then regenerate
  `layout.json` and tell the inference and training threads.

## Capacity and tiers

The DSP holds no capacity weights. Per-tier limits (polyphony, partials, FM operators, grain
layers, voice and instance slots, modulation routes, resonator modes, OSC unison) come from
`docs/tiers/allocation_by_tier.json` (`dsp_limits`), owned by the tier spec. They are generated
into `TierLimitsGenerated.h` by `tools/gen_tier_limits.py`, and a CTest fails when the header is
stale.

## Real-time rules

After `prepare()`, everything on the audio thread is `noexcept` and makes no allocation, lock,
file access, model load or graph build. The allocation tests override global `operator new`
(`tests/AllocationGuard.cpp`) and check `setParams`/`process` for every family, plus the whole
synth with random family switching at 44.1, 48 and 96 kHz. `prepare()` builds every allowed
family for every slot up front, so a family switch only calls `reset()`.

Non-finite values never propagate:
- Frames are sanitised by the modulation matrix.
- Each family resets itself if its state goes non-finite.
- `SlotGraph` silences any processor that outputs NaN or Inf for one block and resets it.

Denormals are flushed (`ScopedNoDenormals`, FTZ/DAZ) for the whole `Synth::process` call.

## Host responsibilities

- Call `Synth::prepare(sr, maxBlock, tier)`, `setGrainBank` and `setSample` off the audio
  thread, while processing is suspended.
- Convolution IRs: the host owns an `IrBank`, calls `synth.publishIrs(bank)` after prepare and
  after each change, and calls `synth.collectIrs(bank)` from time to time on a non-audio thread.
- Latency: nonlinear oversampling (0/23/29 samples), spectral (FFT size) and pitch shift report
  `latencySamples()`, but the slot graph does not yet compensate for them. Each family's own dry
  path is latency-matched internally. Graph-level compensation is future work.
- Memory: every slot pre-builds all its families. The instance chain is the large part, about
  11 MB per instance slot at 48 kHz, mostly the 6 s stereo convolution state and the 2 s delay
  lines.

## Tests and benchmarks

- `ctest` runs every family's tests: deterministic response, silence, NaN/Inf, extreme and
  random params, 44.1/48/96 kHz, no allocation, stability at maximum resonance and feedback, and
  fast modulation. `tests/IntegrationTests.cpp` covers the whole path.
- `alfa_dsp_bench` (Release) prints µs per 128-sample block for each primitive scenario and each
  family, per tier. The `[.bench]` Catch2 cases time each processor on its own.

## Python binding (offline)

`tools/alfa_dsp_py` wraps `Synth` in a small C API (`alfa_dsp_c`, shared library) and a ctypes
module (`alfa_dsp.py`, needs numpy). `Synth(sr, tier).render(voice_frames[T, 2627],
instance_frames[T, 164] | None, f0[T] | None, note, velocity, note_off_frame)` returns stereo
float32 of T·hop samples. Each call resets the synth, so renders are deterministic and can be
batched with one `Synth` per worker. Build it with `cmake --build <dir> --target alfa_dsp_c`.

## Audio budget guard

`BudgetGuard.h`. The user chose it on 2026-09-30, and the tier spec's `audio_budget_guard` configures it. When the load is over the tier's `max_block_share` of real time, the guard degrades voice slots for their next hop:
- it takes the heaviest slot first, then the lowest family capacity weight (generated from `family_share_pct`), then the newest note;
- it first moves a slot to reduced quality, where the family has a cheaper mode (`Processor::hasReducedQuality`: NONLINEAR drops to 1× oversampling, RESONATOR halves its modes);
- after that it bypasses the slot, which crossfades out like switching to Off.

Instance slots and primitives are never degraded. Shares are per second of audio, so the budget holds at every sample rate.

There are two triggers (`Synth::setBudgetGuardTrigger`):
- **Measured**, for real-time hosts. `process()` times itself with a steady clock and the guard keeps an exponential average of the load (0.2 s time constant). The predicted load is that average plus the change in modelled cost since then, so a note-on burst is caught in its first block. Above budget it degrades; below 80% of budget (20% for a 0.25 budget) it restores one step per block, provided the step keeps it under that line. Levels persist between blocks and reset on a new note or a family change.
- **Model**, the default, for offline renders and tests. The modelled cost alone decides, recomputed every block. No clock is read, so a given input always gets the same decisions.

The cost model, `CostTableGenerated.h`, is written by `alfa_dsp_calibrate`. It holds the ns per sample measured at 48 and 96 kHz on the calibration machine. Slot costs are the median over several typical param sets. A factor per rate (`rateScale`) rescales the model so its total matches the measured total over a set of whole-synth patches, and the guard multiplies every cost by a 1.2× safety margin (`BudgetGuard::kSafetyMargin`), as the tier spec asked. Hosts can rescale the model with `Synth::setCostScale`. In the Measured trigger the model only picks victims and predicts bursts.
