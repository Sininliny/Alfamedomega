# Model contract (what the training export must produce)

The inference host loads one folder per tier and checks every graph against this contract when
the tier is loaded. A mismatch is refused with a message naming the graph and tensor, and the
previous tier keeps playing. `tools/export_dummy_models.py` writes toy models that follow it.

This file restates `training/METHOD.md` section 7 in exact tensor terms. If the two disagree,
fix one of them in the same change.

## Folder and manifest

```
Models/                      the control center's model store (docs/control-center/model-store.md)
  manifest.json
  <dir>/                     e.g. standard-1.0.0; folders starting with "." are never models
    patch_encoder.onnx
    audio_encoder.onnx       optional: sampler reference conditioning
    instance_step.onnx
    voice_core.onnx
    expert_subtractive.onnx  expert_wavetable.onnx  expert_additive.onnx
    expert_spectral.onnx     expert_fm.onnx         expert_physical.onnx
    expert_granular.onnx     expert_sampling.onnx   expert_hybrid.onnx
    grain_bank.npy           optional: GRAIN's learned sources, float32 [N, 8192]
    export_info.json         optional: training's export record; the host checks its
                             layout_version (= kLayoutVersion, 2) and voiceframe_floats (2627)
```

The host reads the control center's schema 1 as written:

```json
{
  "schema": 1,
  "default_tier": "standard",
  "models": [
    {
      "id": "standard-1.0.0",
      "tier": "standard",              // lite | standard | pro | studio
      "version": "1.0.0",
      "dir": "standard-1.0.0",
      "files": [{"path": "voice_core.onnx", "sha256": "<hex>", "bytes": 123456}],
      "bytes": 123456, "sr": 48000, "frame_hz": 250, "hop": 192
    }
  ]
}
```

- A higher `schema` is refused. Keys the host does not use (`source`, `installed_at`, ...) are
  ignored.
- It also reads the training export's first form: `folder` for `dir`, `frame_rate_hz` for
  `frame_hz`, and `files` as a map of path to SHA-256. `id` defaults to `<tier>-<version>`.
- Every file the host opens must be listed. On load it checks each file's SHA-256 (any case)
  and, when given, its size. The newest `version` of a tier wins.
- A `dir` or file path that is absolute or contains `..` is refused.
- Where the host looks: `ALFA_MODELS_DIR` if set, else `/Library/Application Support`
  (macOS), `%ProgramData%` (Windows) or `$XDG_DATA_HOME` (default `~/.local/share`) (Linux),
  each followed by `Alfamedomega/Models`.

## Graphs

All tensors are float32. `V` is a dynamic first dimension (the number of voices in the call,
1 to 16); name it with a `dim_param`, not a fixed size. E, S, C, H and M are the tier's widths
and may be anything; the host reads them from the graphs.

| Graph | When | Inputs | Outputs |
|---|---|---|---|
| `patch_encoder` | patch or knob change (message thread) | `knobs` [1, 32], `method` [1, 9] one-hot, optional `hybrid_parts` [1, 18] | `embedding` [1, E] |
| `audio_encoder` (optional) | reference sample change (message thread) | `logmel` [1, 64, frames] | `e_audio` [1, E] |
| `instance_step` | every frame | `embedding` [1, E], `global_mod` [1, 4], `state` [1, S] | `context` [1, C], `state_out` [1, S] |
| `voice_core` | every frame, active voices batched | `features` [V, 8], `context` [1, C], `state` [V, H] | `hidden` [V, H], `state_out` [V, H] |
| `expert_<method>` | every frame, the voices of that method | `hidden` [V, H], optional `f0` [V], physical only `modal_state` [V, M] | one output per control-frame field it drives, [V, width]; physical also `modal_state_out` [V, M] |
| `expert_hybrid` | every frame, hybrid voices | `hidden` [V, H] | `hidden_a` [V, H], `hidden_b` [V, H], `xf` [V] |

- Every `state_out` is fed back as its `state` on the next frame, and `modal_state_out` as
  `modal_state`. The host zeroes a voice's states on note-on (retrigger) and when the voice
  goes idle, and zeroes everything on transport reset.
- Note features, in order: pitch/127, velocity, gate, log1p(time since note-on)/log1p(4 s),
  log1p(time since note-off)/log1p(4 s), mod wheel, aftertouch, pitch bend (±1 = ±2 semitones).
- `f0` is in Hz: `440 * 2^((pitch * 127 + 2 * bend - 69) / 12)`. Experts that do not use it may
  leave it out (the training export prunes it).
- Method order (for the one-hot and for voice routing): subtractive, wavetable, additive,
  spectral, fm, physical, granular, sampling, hybrid.
- `hybrid_parts` is two one-hots over the same order: component A in 0..8, component B in 9..17.
  Zero for a non-hybrid patch.
- With a reference sample the host feeds `patch embedding + e_audio` to `instance_step`.
  `logmel` is computed as training's `logmel()`: 48 kHz mono, n_fft 2048, hop 512, periodic
  Hann, centred frames with reflect padding, magnitude, 64 HTK-mel bands from 20 Hz to 16 kHz,
  `log(mel + 1e-5)`.

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
- Every voice starts each frame as a default-constructed `alfa::dsp::VoiceFrame` (layout v2:
  `dry_gain` 1, unused `osc.detune_ratio` entries 1, `attack_tau` 1 ms, ...), and each expert
  output overwrites a prefix of its field. `mix` defaults to 0, so every expert should output
  `mix` and `attack_tau`. Idle voices keep the default frame.
- Layout v2 fields (`pan`, `dry_gain`, `osc.detune_ratio`, `osc.detune_level`, `slots`, `mod`)
  are written like any other when an expert outputs them by name.
- Values are in the physical units ControlFrame.h states (linear gains, ratios, radians,
  seconds). The graph does any scaling; the host copies values as they are.

## Hybrid

For a hybrid voice the host runs `expert_hybrid`, then component A's expert on `hidden_a` and
component B's on `hidden_b` (with the voice's f0 and modal state), and merges the two frames
exactly as training's `merge_hybrid()`, with `wa = xf` and `wb = 1 - xf`:

- `mix = wa * mix_A + wb * mix_B`, and `attack_tau = min(A, B)`.
- If either component drives OSC: `osc = Σ w * mix_c[osc] * osc_c`, and `mix[osc] = 1`.
- If either drives NOISE: `noise.band = sqrt(Σ (w * mix_c[noise] * band_c)² + 1e-12)`, and
  `mix[noise] = 1`.
- If either drives OSC unison (`osc.detune_level`): with `g_c = w_c * mix_c[osc]`,
  `detune_level = Σ g·level / (Σ g + 1e-6)` and `detune_ratio = Σ g·level·ratio / (Σ g·level + 1e-12)`,
  or 1 where `Σ g·level ≤ 1e-9`. A component without unison counts as level 0, ratio 1.
- Every other field is copied from the one component that outputs it. A pair whose experts
  share such a field (additive and physical both drive PARTIALS) is refused in `setPatch`.

The components come from the patch (`InferenceHost::setPatch(knobs, method, a, b, error)`).
A hybrid voice without two components stays silent.

## Timing

The host runs one frame per 250 Hz hop (`alfa::dsp::hopSamples`, 192 samples at 48 kHz). A
frame's whole pipeline must finish within the tier's declared maximum inference time
(`include/alfa/inference/Tier.h`), which must stay below the 4 ms frame period because frames
of a recurrent model cannot overlap.
