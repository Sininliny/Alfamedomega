# Alfamedomega control center

A standalone app for the work that isn't playing music: training model sizes, installing and
moving model weights, and running Adaptive Preference Learning in batch over your sample
libraries. The plugin (VST3/AAX) stays focused on performance and reads what this app installs.

Why Python and Qt rather than JUCE: [CC-0001](../../docs/control-center/0001-app-stack.md).

## Run it

```bash
cd apps/control-center
python -m pip install -e ".[gui,test]"
alfa-cc-gui                 # the window
alfa-cc --help              # the same actions from a terminal
python -m pytest            # tests (no PyTorch needed)
```

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
alfa-cc jobs list
```

## Layout

```
src/alfacc/
  settings.py    paths and settings.json
  models.py      the Models folder: manifest, import, bundles, verify
  jobs.py        background job queue with logs and progress
  training.py    training/export jobs and Colab notebooks
  preference.py  libraries, index, labels, learner batches
  cli.py         alfa-cc
  gui/           alfa-cc-gui (PySide6), Graphite theme from docs/ui/ui-design.md
tests/           pytest; fake_train.py stands in for the training script
```

The core uses only the standard library. PySide6 is needed for the window only.
