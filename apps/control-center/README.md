# Alfamedomega control center

A standalone app for the work that isn't playing music: training model sizes, installing and
moving model weights, running Adaptive Preference Learning in batch over your sample
libraries, and learning the timbres of the instrument plugins you already own. The plugin (VST3/AAX) stays focused on performance and reads what this app installs.

Why Python and Qt rather than JUCE: [CC-0001](../../docs/control-center/0001-app-stack.md).

## Run it

```bash
cd apps/control-center
python -m pip install -e ".[gui,test]"
alfa-cc-gui                 # the window
alfa-cc --help              # the same actions from a terminal
python -m pytest            # tests (no PyTorch needed)
```

In the packaged app, set **Settings > Repository folder** to your Alfamedomega checkout once.
Training, the preference learner and a `.venv` with PyTorch inside it are found from there
(`alfa-cc settings --set repo_dir=~/src/Alfamedomega`). Run from source, nothing needs setting.

Development without touching system folders:

```bash
export ALFA_MODELS_DIR=/tmp/alfa/Models ALFA_CC_HOME=/tmp/alfa/home
```

## What it does

| Area | Actions |
|---|---|
| **Models** | See which sizes are installed; import a `.alfamodel` bundle, a training export folder or a zipped export from Colab; export a bundle; verify SHA-256; remove; set the default size. Folder layout and manifest: [model-store.md](../../docs/control-center/model-store.md). |
| **Training** | Run `training/alfamedomega_train.py` with a Python that has PyTorch, for any tier and stage (pretrain, polish, fine-tune, distil), then export and install the result in the same job. Or write a self-contained Colab notebook, and import the export zip when it is done. |
| **Preference learning** | Choose sample and preset folders, index them into weak text labels from file and folder names, and run the preference learner over them. Forget everything with one button. [preference-batch.md](../../docs/control-center/preference-batch.md). |
| **Plugin assimilation** | Find installed VST3 (and AU on macOS) instruments, open the ones you pick in a separate render host, find each preset's playable range, render it at Fast, Normal, Detailed or Unreasonable density and analyse the notes into a local timbre knowledge base. AAX can't be loaded outside Pro Tools; the page points to the VST3 or AU twin. [plugin-assimilation.md](../../docs/control-center/plugin-assimilation.md). |
| **Jobs** | One queue for all of the above, with progress, a live log and cancel. History survives restarts. |

## CLI examples

```bash
alfa-cc info                                          # paths, and whether training can run here
alfa-cc settings --set training_python=~/venvs/torch/bin/python
alfa-cc train --tier lite --smoke                     # tiny CPU run: train, export, install
alfa-cc train --tier studio --steps 200000 --colab .  # write a Colab notebook instead
alfa-cc export --tier pro --ckpt pro_distill.pt       # export and install a checkpoint
alfa-cc models import ~/Downloads/export.zip --version 1.0.0
alfa-cc models export standard-1.0.0 ~/Desktop
alfa-cc models verify
alfa-cc pref add ~/Samples && alfa-cc pref batch
alfa-cc plugins discover && alfa-cc plugins scan       # find and open installed instruments
alfa-cc plugins assimilate "Pigments" --density normal
alfa-cc jobs list
```

Plugin assimilation needs the render host, a small JUCE program in `render-host/`:

```bash
cmake -S render-host -B build-rh -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-rh
export ALFA_RENDER_HOST=$PWD/build-rh/alfa_render_host_artefacts/Release/alfa-render-host
export ALFA_TEST_SYNTH=$PWD/build-rh/test-synth/alfa_test_synth_artefacts/Release/VST3/AlfaTestSynth.vst3
python -m pytest tests/test_assimilate.py    # the real-host test runs when both are set
```

## Layout

```
src/alfacc/
  settings.py    paths and settings.json
  models.py      the Models folder: manifest, import, bundles, verify
  jobs.py        background job queue with logs and progress
  training.py    training/export jobs and Colab notebooks
  preference.py  libraries, index, labels, learner batches
  assimilate/    plugin discovery, range finding, densities, harvest jobs, analysis worker
  cli.py         alfa-cc
  gui/           alfa-cc-gui (PySide6), Graphite theme from docs/ui/ui-design.md
render-host/     alfa-render-host (JUCE console app) and AlfaTestSynth, a VST3 used in tests
tests/           pytest; fake_train.py and fake_render_host.py stand in for the real programs
```

The core uses only the standard library. PySide6 is needed for the window only. The analysis
worker runs in the training Python and needs NumPy.
