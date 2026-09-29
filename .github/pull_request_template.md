## What changes

Before:

After:

## How

## Checks

- [ ] `cmake --build` and `ctest` pass locally (preset: `core` or `dev`)
- [ ] `bash scripts/format.sh --check` passes
- [ ] Nothing on the audio thread allocates, locks, or does I/O
- [ ] Touches only the component directory this branch owns, or says why not
