# Model contract (what the training export must produce)

The inference host loads one folder per tier and checks every graph against this contract when
the tier is loaded. A mismatch is refused with a message naming the graph and tensor, and the
previous tier keeps playing. `tools/export_dummy_models.py` writes toy models that follow it.

This file restates `training/METHOD.md` section 7 in exact tensor terms. If the two disagree,
fix one of them in the same change.

## Folder and manifest

```
Models/
  manifest.json
  <tier>-<version>/          e.g. standard-1.0.0
    patch_encoder.onnx
    instance_step.onnx
    voice_core.onnx
    expert_subtractive.onnx  expert_wavetable.onnx  expert_additive.onnx
    expert_spectral.onnx     expert_fm.onnx         expert_physical.onnx
    expert_granular.onnx     expert_sampling.onnx   expert_hybrid.onnx
```

```json
{
  "schema": 1,
  "models": [
    {
      "tier": "standard",              // lite | standard | pro | studio
      "version": "1.0.0",
      "folder": "standard-1.0.0",
      "opset": 17,
      "frame_rate_hz": 250,
      "files": { "voice_core.onnx": "<sha-256, hex>", "...": "..." }
    }
  ]
}
```

Every file in the folder that the host opens must be listed with its SHA-256. The newest
`version` of a tier wins.

## Graphs

All tensors are float32. `V` is a dynamic first dimension (the number of voices in the call,
1 to 16); name it with a `dim_param`, not a fixed size. E, C and H are the tier's widths and may
be anything; the host reads them from the graphs.

| Graph | When | Inputs | Outputs |
|---|---|---|---|
| `patch_encoder` | patch or knob change (message thread) | `knobs` [1, 32], `method` [1, 9] one-hot | `embedding` [1, E] |
| `instance_step` | every frame | `embedding` [1, E], `global_mod` [1, 4], `state` [1, C] | `context` [1, C], `state_out` [1, C] |
| `voice_core` | every frame, active voices batched | `features` [V, 8], `context` [1, C], `state` [V, H] | `hidden` [V, H], `state_out` [V, H] |
| `expert_<method>` | every frame, the voices of that method | `hidden` [V, H] | one output per control-frame field it drives, [V, width] |

- `state_out` is fed back as `state` on the next frame. The host zeroes a voice's state on
  note-on (retrigger) and when the voice goes idle, and zeroes everything on transport reset.
- Note features, in order: pitch/127, velocity, gate, log1p(time since note-on)/log1p(4 s),
  log1p(time since note-off)/log1p(4 s), mod wheel, aftertouch, pitch bend.
- Method order (for the one-hot and for voice routing): subtractive, wavetable, additive,
  spectral, fm, physical, granular, sampling, hybrid.

## Expert outputs

Each output is named after a field of `alfa::dsp::VoiceFrame`
(`src/dsp/include/alfa/dsp/ControlFrame.h`, `kFields`):

`mix`, `attack_tau`, `osc.re`, `osc.im`, `partials.ratio`, `partials.amp`, `fm.ratio`,
`fm.level`, `fm.index`, `noise.band`, `grain.source`, `grain.weight`, `grain.position`,
`grain.rate`, `grain.gain`, `sampler.rate`, `sampler.start`, `sampler.gain`, `sampler.eq`.

- The width may be smaller than the field (a Lite model's `partials.ratio` has 16 of 64). The
  host writes it at the start of the field and leaves the rest zero.
- `fm.index` is the one exception: a tier with N < 8 operators outputs it as N x N row-major
  (width N²), and the host spreads the rows to the frame's 8-wide rows.
- Fields an expert does not output stay zero for that voice, including `mix`, so every expert
  should output `mix` and `attack_tau`.
- Values are in the physical units ControlFrame.h states (linear gains, ratios, radians,
  seconds). The graph does any scaling; the host copies values as they are.
- `expert_hybrid` is run like any other expert and must output primitive controls directly.
  If hybrid needs to run two component experts on transformed hidden states (METHOD.md
  section 3), put that inside `expert_hybrid.onnx`, or tell the inference owner so the host can
  chain graphs.

## Timing

The host runs one frame per 250 Hz hop (`alfa::dsp::hopSamples`, 192 samples at 48 kHz). A
frame's whole pipeline must finish within the tier's declared maximum inference time
(`include/alfa/inference/Tier.h`), which must stay below the 4 ms frame period because frames
of a recurrent model cannot overlap.
