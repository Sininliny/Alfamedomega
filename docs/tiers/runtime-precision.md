# Uniform executable tensor precision

This is the active runtime precision policy. The earlier model size estimates in
`model-tiers.md` describe a historical weight format and do not select DSP types,
execution clocks, fixed slots or operation families for the tensor product.

| Tier | Numerical execution type | Bytes per scalar |
|---|---|---|
| Lite | float32 | 4 |
| Standard | float32 | 4 |
| Pro | float64 | 8 |
| Studio | float64 | 8 |

Within a configuration, activation, routing, parameters, modulation, latent and
control values, decoded compiler weights, assets, coefficient tables, persistent
DSP/model state and audio operands use this one type. Structural indices, tensor
shapes, presence masks and scheduling metadata are not numerical signal tensors.
Integer-like persistent signal state is encoded in exact 16-bit limbs of the
selected scalar type. Wider temporary arithmetic is allowed for numerical stability.

JUCE host automation and the current native audio/C interface use float32. These
are explicit conversion boundaries, not float32 internal processing in Pro/Studio.
Conversion storage is prepared before rendering and requires a declared
`budget.boundary_conversion_bytes`. Native input capacities describe an interface
allocation bound, not fixed primitive slots or tensor ontology limits. Raw editor
and JSON values preserve double precision; the eight highest host automation ports
retain their host float32 boundary. The UI state visualization is also a float32
diagnostic snapshot, not authoritative state storage.

## Measurement supporting the choice

The regression fixture evaluates a 40 Hz resonator at 192 kHz over two seconds,
with an exponential decay time constant of 30 seconds (not RT60), against an
analytic response. Float32 relative RMS error was 0.052077; float64 was 4.72e-9.
This tests a numerically difficult long decay, not every primitive or all sounds.

On the local Windows machine, an optimized build with 16 simultaneous resonators,
192 kHz and 4096 callbacks of 256 samples measured 1344 bytes of numeric runtime
backing for float32 and 2688 bytes for float64. One run measured 344.877 and
330.240 microseconds per callback respectively. Timings are machine dependent.
The final integrated benchmark run measured 360.864 and 343.523 microseconds
respectively, with the same storage counts. The result supports a storage/precision
choice and does not establish a float32 CPU advantage.
The recurrent output-history rollback buffer introduced during sparse-state
migration increases this fixture's numerical backing to 1408 and 2816 bytes.
Earlier measurements above describe the preceding runtime layout. The sparse
state benchmark and its allocation accounting are documented in
[sparse state execution](../runtime/sparse-state.md).
`alfa_tensor_precision_bench` reproduces the resource/timing fixture;
`PrecisionTests.cpp` verifies response accuracy and exact integer-state encoding.
After building `alfa_tensor_c`, `python src/tensor/tools/precision_smoke.py`
checks native float32/float64 rendering, exact reset replay and conversion-budget
refusal without running a training job (NumPy is required).

All tiers retain the same ontology, editable coordinates and scheduling contract.
Actual configuration capacity is enforced by its explicit budgets. This policy
does not import the historical model parameter counts as measured runtime limits.
Preset transfer retains the same computation and coordinates; changing precision
can change numerical output and starts fresh state if the scalar type changes.

JSON uses `"tier": "Pro"` and optionally `"scalar_type": "float64"`. Contradictory
types, unknown tiers and unsupported scalar formats are rejected transactionally.
Legacy documents without either field retain float32. Unnamed documents may
explicitly request a supported scalar type. Settings tier changes add the native
boundary allocation budget and preserve the rest of the tensor configuration.
