# preference/

Reference implementation of adaptive preference learning
([spec](../docs/preference/adaptive-preference-learning.md), [design](../docs/preference/design.md)).
Python 3.9+ and NumPy only, no torch, so it runs on any machine and on Colab.

| Module | Design section | What it does |
|---|---|---|
| `alfapref/descriptor.py` | 2 | 134-number timbre descriptor from audio, same log-mel front end as training |
| `alfapref/events.py` | 3.1 | plugin behaviour log schema, exposures, weak labels and pairs |
| `alfapref/library.py` | 3.2, 7 | WAV reader, incremental library scan that stores no audio and no paths |
| `alfapref/tokens.py` | 5 | words from file and folder names, query parsing |
| `alfapref/model.py` | 2, 4, 5 | basis, contextual preference scorer (Newton + Laplace), word map |
| `alfapref/generate.py` | 6 | training prior, "randomize nearby", Gumbel top-k, cross-entropy refinement |
| `alfapref/profile.py` | 4–7 | a user's local profile folder, fit and generate |
| `alfapref/renderers.py` | 6.2 | preview renderer interface and a NumPy stand-in synth |
| `alfapref/simulate.py` | 9 | simulated user with a hidden taste |

## Use

From this folder:

```bash
python -m unittest discover tests              # ~5 s; ALFA_SLOW=1 adds the simulated user
python -m alfapref init --profile ~/amo-profile
python -m alfapref scan --profile ~/amo-profile ~/Samples/MyPads
python -m alfapref fit --profile ~/amo-profile                     # add --events log.jsonl --behaviour on
python -m alfapref words --profile ~/amo-profile                   # the vocabulary it learned
python -m alfapref generate --profile ~/amo-profile --text "warm not dark" --context bass -k 8
python -m alfapref simulate                                         # learning curve, ~1 min
python learner.py batch --index IDX.jsonl --events EVENTS_DIR --out OUT   # control-center batch
```

`learner.py` runs the same commands from any working directory; it is the command the control
center calls (design section 8.3).

On Colab: upload or clone the repo, `%cd Alfamedomega/preference`, and run the same commands.
The default preview renderer is a NumPy stand-in, so its knob settings only test the loop. For the
real sound, put `training/` on the path and pass the training script's renderer (needs torch):

```bash
PYTHONPATH=../training python -m alfapref generate --profile ~/amo-profile --text warm \
    --renderer alfamedomega_train:render_patch
```

Nothing here opens a network connection. The profile folder is the only thing written.
