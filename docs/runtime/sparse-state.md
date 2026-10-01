# Transactional sparse DSP state

The runtime implements `S_(t+1) = F(S_t, X_t, R_t)` with simultaneous old-state
reads. Scheduling does not imply a graph execution order: a callback cannot read
another callback's uncommitted state or candidate output. Each primitive has its
own explicit clock. Skipped updates preserve state, and output policy explicitly
selects hold or zero.

## Dense and sparse transitions

A dense transition receives a preallocated candidate span for its local state.
Only a scheduled, active dense slice is copied from old state and finite-checked.
This preserves compatibility with callbacks that deliberately leave some elements
unchanged. Dense scratch is allocated for dense slices only; sparse ring histories
have no dense next-state shadow.

A sparse transition has this typed interface:

```cpp
template<class Scalar>
struct StateWriteT { std::size_t element; Scalar value; };

template<class Scalar>
using SparseTransitionT = std::size_t (*)(
    const StepT<Scalar>&, std::span<StateWriteT<Scalar>>) noexcept;
```

`element` is relative to the primitive's local slice. `value` uses the runtime's
uniform scalar type; indices remain structural metadata. The descriptor declares
`sparseWriteCapacity` and `sparseTransition`. When both transition forms exist,
sparse takes precedence. The return value is the number of journal entries used;
a value above the supplied capacity rejects the tick.

Preparation allocates the journals. The sum of capacities must fit
`Budget.sparseStateWrites`, serialized as `budget.sparse_state_writes`. A capacity
larger than the primitive's state slice is rejected before allocation. Counted
entries must all be written, address valid local elements, contain finite values,
and use distinct indices. Duplicate detection costs O(k²) for a journal of k
entries; it does not inspect the rest of the state history. A production ring
delay uses three entries: two scalar cursor limbs and one overwritten sample.
The cursor is represented exactly with base-65536 limbs in either float32 or
float64 state; no second persistent numeric datatype is introduced.

After all due callbacks and candidate outputs pass validation, the runtime commits
dense candidates and sparse journals together. A failed tick commits no state,
does not advance its sample counter, and exposes zero output. Its last committed
output history remains available to future recurrent routing; exposed silence
does not replace that history. Reset and compatible state adoption clear scratch
metadata. There are no callback-time allocations in this implementation; this
benchmark measures timing and correctness rather than instrumenting the allocator.

The const-state and bounded-span callback contract remains mandatory. Validation
does not protect against a callback performing an illegal out-of-bounds write or
casting away constness.

## Full-runtime scalability measurement

Measured on 2026-10-01, Windows x64, AMD Ryzen 9 6900HS with Radeon Graphics,
MSVC 14.51, `/std:c++20 /EHsc /MD /O2 /W4 /WX`. This is a machine-specific
microbenchmark, not a host/plugin latency guarantee or a tier performance rating.

The reproducible fixture is `src/tensor/bench/SparseRuntimeBench.cpp`. It links
the actual `Runtime.cpp` and `Production.cpp`, calls `RuntimeT::tick` rather than
calling the primitive directly, and uses one `time.ring_delay.v1` with a
sample-clock period of 1 at 192,000 Hz. Its explicitly authored delay is 127
updates, feedback is 0.5, and input is 0.25. Keeping these parameters identical
isolates the cost of allocated history length. At 384,000 history samples the
measured run does not traverse the entire history; it still writes 192,000
consecutive samples, while the allocated state and its accounting cover the full
capacity. This does not characterize very long tap/cache patterns or many voices.

Each cell reports the median of seven runs of 192,000 attempted ticks, following
reset and 4,096 warm-up ticks. Preparation, reset, warm-up, and full-buffer rollback
comparison are outside the timed interval. A consumed output checksum and checked
commit counter prevent treating the work as unused. Normal runs finish with
192,000 additional committed ticks. Rejection runs inject NaN input once every
64 attempts, expect 3,000 rejected ticks, and finish with 189,000 additional
committed ticks. All configurations independently verify that an invalid tick
leaves the full state and sample counter unchanged, exposes zero output, and is
followed successfully by a valid tick.

| Type | History samples | Numeric backing bytes | Valid median ns/tick | Valid min–max ns/tick | Reject 1/64 median ns/attempt |
|---|---:|---:|---:|---:|---:|
| float32 | 128 | 1,616 | 90.210 | 85.896–110.928 | 84.683 |
| float64 | 128 | 3,232 | 87.345 | 85.311–88.811 | 85.802 |
| float32 | 65,536 | 786,512 | 88.973 | 86.794–96.424 | 86.771 |
| float64 | 65,536 | 1,573,024 | 86.930 | 84.425–90.223 | 86.423 |
| float32 | 384,000 | 4,608,080 | 88.556 | 84.581–95.238 | 90.410 |
| float64 | 384,000 | 9,216,160 | 92.089 | 86.210–111.593 | 87.759 |

`numericStorageBytes()` counts owned numeric values, including initial
configuration/state copies, journal values, and committed output history. It
excludes structural indices, masks, journal padding, container overhead, and the
fixture's external input/configuration. Here storage is exactly
`sizeof(Scalar) * (3 * history + 20)`: persistent state, its reset baseline, and
the retained configuration's initial state dominate the three history copies.
There is no fourth full-history candidate state. Float64 doubles this numeric
storage, but this fixture does not show a consistent float32 CPU speed advantage.

Both source inspection and the observed nearly constant tick costs support that
this sparse path neither copies nor finite-scans the entire history every sample.
Preparation still validates and copies initial state, and reset/state adoption
remain whole-state operations outside the tick path. Long FIR evaluation can
still scale with the number of coefficients; sparse writes optimize state
publication, not the mathematical cost of every primitive.

`src/tensor/tests/SparseStateTests.cpp` separately covers mixed dense/sparse old
state reads, clocks, inactive updates, malformed journals, downstream failure,
reset/adoption, and committed output-history rollback. The independent sparse
test run passed 5 cases and 243 assertions. Production primitive tests independently
passed 10 cases and 9,932 assertions; the benchmark above adds full-runtime
integration and does not replace those behavioral checks.

To reproduce without touching central build outputs, create `build/runtime-audit`
and use a Visual Studio x64 developer command prompt from the repository root:

```bat
cl /nologo /std:c++20 /EHsc /MD /O2 /W4 /WX /I"src/tensor/include" /I"src/dsp/include" /Fobuild/runtime-audit/ /Febuild/runtime-audit/sparse-runtime-bench.exe src/tensor/src/Runtime.cpp src/tensor/src/Production.cpp src/tensor/bench/SparseRuntimeBench.cpp
build\runtime-audit\sparse-runtime-bench.exe
```
