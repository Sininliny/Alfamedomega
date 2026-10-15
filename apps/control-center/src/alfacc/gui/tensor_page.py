"""AMO Centre pages for the Audio Tensor Space: one page per task, each asking only for its own inputs.

All work runs as durable jobs in a separate process (see ..amo). Results chain forward through
`win.amo_recent`: a built dataset fills the manifest fields, a trained WAV model fills Evaluate and
Import sound. Fields the user already filled are never overwritten.
"""
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QPlainTextEdit, QProgressBar, QSpinBox)

from ..amo import create_tensor_job
from ..jobs import FINISHED
from .widgets import Page, PathField, button, card, label

STATUS = {"queued": "Queued", "running": "Working", "succeeded": "Completed", "failed": "Failed",
          "cancelled": "Cancelled", "interrupted": "Interrupted"}

# Four separate model roles with separate data and budgets (docs/control-center/amo-training-system.md).
ROLES = [
    ("train", "WAV recognition", "wav_preset_learning", "model.pt", "PyTorch model (*.pt)",
     "Learns how a recording was constructed: the original preset before perturbation and the "
     "rendered variation. Needs a WAV manifest from Datasets."),
    ("train-runtime", "Runtime compiler", "runtime", "runtime-model.json", "JSON (*.json)",
     "Predicts tensor controls from MIDI, audio and context while playing. Exported as native "
     "weights the plugin can import."),
    ("train-preference", "User preference", "user_preference", "preference-model.json", "JSON (*.json)",
     "Learns local taste from pairwise feedback. Stays on this computer."),
    ("train-importance", "Context importance", "context_importance", "importance-model.json", "JSON (*.json)",
     "Learns which tensor details matter in which musical context, for tier conversion and the "
     "order of controls in the plugin's Play view (see Rank controls)."),
]


NAMES = {"wav_model": "WAV model", "recording": "Recording", "tensor_template": "Tensor template",
         "parameter_bounds": "Parameter bounds", "output": "Save as", "dataset_plan": "Dataset plan",
         "output_folder": "Output folder", "manifest": "Manifest", "report": "Save report as",
         "sound": "Sound", "importance_model": "Importance model", "context": "Context features"}


def default_repository(win) -> str:
    return win.ctx.settings.repo_dir or str(Path(__file__).resolve().parents[5])


class JobPanel:
    """Status, progress and log tail of the latest job a page started."""

    def __init__(self, page: "AmoPage"):
        self.page = page
        self.frame, lay = card()
        top = QHBoxLayout()
        self.status = label("No job started from this page yet.", "muted", wrap=True)
        top.addWidget(self.status, 1)
        self.cancel_btn = button("Cancel", self.cancel, tip="Stop cooperatively; training saves a checkpoint")
        self.cancel_btn.setEnabled(False)
        top.addWidget(self.cancel_btn)
        top.addWidget(button("Show in Jobs", self.show_in_jobs))
        lay.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        lay.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMinimumHeight(140)
        lay.addWidget(self.log)
        self.progress.hide()
        self.log.hide()
        self.job_id = None
        self._last_log = ""

    def track(self, job_id):
        self.job_id = job_id
        self._last_log = ""
        self.log.clear()
        self.progress.show()
        self.log.show()
        self.refresh()

    def message(self, text: str, warn: bool = True):
        self.status.setObjectName("warn" if warn else "muted")
        self.status.setText(text)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def refresh(self):
        if not self.job_id:
            return
        jobs = self.page.ctx.jobs
        job = jobs.get(self.job_id)
        if not job:
            return
        text = jobs.read_log(self.job_id, tail=1000)
        if text != self._last_log:
            self.log.setPlainText(text)
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            self._last_log = text
        state = job["status"]
        self.message(f"{job['title']}: {STATUS.get(state, state)}" + (f". {job['error']}" if job.get("error") else ""),
                     warn=state in ("failed", "interrupted"))
        self.cancel_btn.setEnabled(state not in FINISHED)
        progress = job.get("progress")
        if progress and progress[1] > 0:
            self.progress.setRange(0, progress[1])
            self.progress.setValue(progress[1] if state == "succeeded" else progress[0])
        elif state not in FINISHED:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100 if state == "succeeded" else 0)
        if state == "succeeded" and job["id"] not in self.page.win.amo_finished:
            self.page.win.amo_finished.add(job["id"])
            self.page.finished(job)

    def cancel(self):
        if self.job_id:
            self.page.ctx.jobs.cancel(self.job_id)
            self.message("Stopping. Training saves its last complete step to the checkpoint.", warn=False)

    def show_in_jobs(self):
        self.page.win.jobs_changed(self.job_id)


class AmoPage(Page):
    """A task page: a form card, one primary action, and the job panel underneath."""

    def __init__(self, win, title: str, subtitle: str):
        super().__init__(win, title, subtitle)
        self.panel = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.panel and self.panel.refresh())
        self.timer.start(500)

    def refresh(self):
        self.fill()
        if self.panel:
            self.panel.refresh()

    def fill(self):
        """Fill empty fields from results of earlier steps."""

    def form_card(self, heading: str, note: str = ""):
        frame, lay = card()
        lay.addWidget(label(heading, "h2"))
        if note:
            lay.addWidget(label(note, "muted", wrap=True))
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        lay.addLayout(form)
        self.lay.addWidget(frame)
        return form, lay

    def actions(self, lay, *buttons):
        row = QHBoxLayout()
        row.addStretch(1)
        for b in buttons:
            row.addWidget(b)
        lay.addLayout(row)

    def finish(self):
        self.panel = JobPanel(self)
        self.lay.addWidget(self.panel.frame)
        self.lay.addStretch(1)

    @property
    def job_id(self):
        return self.panel.job_id

    @property
    def progress(self):
        return self.panel.progress

    @property
    def log(self):
        return self.panel.log

    def recent(self, key: str, field):
        """Fill an empty field from the latest result of an earlier step."""
        value = self.win.amo_recent.get(key)
        if value and not field.text():
            field.setText(value)

    def require(self, **fields) -> dict | None:
        """Resolved paths by key, or None after naming the empty fields the way the form labels them."""
        missing = [NAMES.get(name, name.replace("_", " ")) for name, field in fields.items() if not field.text()]
        if missing:
            self.panel.message("Fill in: " + ", ".join(missing) + ".")
            return None
        return {name: str(Path(field.text()).expanduser().resolve()) for name, field in fields.items()}

    def run_task(self, action: str, arguments: list, **options) -> bool:
        try:
            job = create_tensor_job(self.ctx.jobs, default_repository(self.win), action, arguments,
                                    python=self.ctx.settings.resolved_training_python(), **options)
        except Exception as error:  # the service validates paths, roles and checkpoints
            self.panel.message(str(error))
            return False
        self.ctx.jobs.enqueue(job["id"])
        self.panel.track(job["id"])
        self.win.poll()
        return True

    def finished(self, job: dict):
        pass


class ImportSoundPage(AmoPage):
    def __init__(self, win):
        super().__init__(win, "Import sound",
                         "Turn a recording into an editable tensor sound the plugin can open. The WAV model "
                         "predicts which primitives the template uses and their parameters, within the bounds you give.")
        form, lay = self.form_card("Recording to sound")
        self.model = PathField(filter="PyTorch model (*.pt)", placeholder="trained WAV recognition model")
        self.wav = PathField(filter="Audio (*.wav)", placeholder="the recording to import")
        self.template = PathField(filter="JSON (*.json)", placeholder="tensor template that names the primitives")
        self.bounds = PathField(filter="JSON (*.json)", placeholder="allowed parameter ranges")
        self.output = PathField(filter="JSON (*.json)", placeholder="new sound file", save=True)
        form.addRow("WAV model", self.model)
        form.addRow("Recording", self.wav)
        form.addRow("Tensor template", self.template)
        form.addRow("Parameter bounds", self.bounds)
        form.addRow("Save sound as", self.output)
        self.head = QComboBox()
        self.head.addItem("Original preset (before perturbation)", "primary_canonical")
        self.head.addItem("Rendered variation (as heard)", "secondary_rendered_instance")
        form.addRow("Recover", self.head)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(.01, .99)
        self.threshold.setSingleStep(.05)
        self.threshold.setValue(.5)
        self.threshold.setToolTip("A primitive is kept when its predicted presence is above this value")
        form.addRow("Presence threshold", self.threshold)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(1000, 384000)
        self.rate.setDecimals(0)
        self.rate.setValue(48000)
        self.rate.setSuffix(" Hz")
        form.addRow("Render sample rate", self.rate)
        self.actions(lay, button("Import", self.start_import, "primary"))
        self.finish()

    def fill(self):
        self.recent("wav_model", self.model)

    def start_import(self):
        paths = self.require(wav_model=self.model, recording=self.wav, tensor_template=self.template,
                             parameter_bounds=self.bounds, output=self.output)
        if paths:
            self.run_task("import-wav", [*paths.values(), "--head", self.head.currentData(),
                                         "--mask-threshold", str(self.threshold.value()),
                                         "--sample-rate", str(self.rate.value())])


class DatasetsPage(AmoPage):
    def __init__(self, win):
        super().__init__(win, "Datasets",
                         "Render a controlled dataset through the native DSP, then check a manifest before training. "
                         "Checking reads the data and estimates memory; it starts no training.")
        form, lay = self.form_card("Build a controlled dataset",
                                   "The plan names original sounds, bounded perturbations and performances. "
                                   "Pick a new folder: existing datasets are never overwritten.")
        self.plan = PathField(filter="JSON (*.json)", placeholder="dataset plan, e.g. training/amo/examples/controlled-dataset-plan.json")
        self.folder = PathField(folder=True, placeholder="new dataset folder")
        form.addRow("Dataset plan", self.plan)
        form.addRow("Output folder", self.folder)
        self.actions(lay, button("Build dataset", self.build, "primary"))
        form, lay = self.form_card("Check a manifest",
                                   "Verifies hashes, splits by original sound, shapes and memory for one model role.")
        self.manifest = PathField(filter="JSON (*.json)", placeholder="manifest.json")
        self.category = QComboBox()
        for _, title, category, *_ in ROLES:
            self.category.addItem(title, category)
        form.addRow("Manifest", self.manifest)
        form.addRow("Model role", self.category)
        self.actions(lay, button("Check manifest", self.check, "primary"))
        self.finish()

    def fill(self):
        self.recent("manifest", self.manifest)

    def build(self):
        paths = self.require(dataset_plan=self.plan, output_folder=self.folder)
        if paths and self.run_task("build-dataset", [paths["dataset_plan"], paths["output_folder"]]):
            self.win.amo_recent["pending_manifest"] = str(Path(paths["output_folder"]) / "manifest.json")

    def finished(self, job: dict):
        pending = self.win.amo_recent.pop("pending_manifest", None)
        if pending and Path(pending).is_file():
            self.win.amo_recent["manifest"] = pending
            self.manifest.setText(pending)

    def check(self):
        paths = self.require(manifest=self.manifest)
        if paths:
            self.run_task("preflight", [paths["manifest"], "--category", self.category.currentData()])


class TrainPage(AmoPage):
    def __init__(self, win):
        super().__init__(win, "Train",
                         "Four separate models with separate data and budgets. Training runs in its own process "
                         "with the Python chosen in Settings, and can be cancelled and resumed.")
        form, lay = self.form_card("Model")
        self.role = QComboBox()
        for action, title, *_ in ROLES:
            self.role.addItem(title, action)
        self.role.currentIndexChanged.connect(self.role_changed)
        form.addRow("Role", self.role)
        self.about = label("", "muted", wrap=True)
        form.addRow("", self.about)
        self.manifest = PathField(filter="JSON (*.json)", placeholder="training manifest")
        self.output = PathField(placeholder="where to save the trained model", save=True)
        self.checkpoint = PathField(filter="Checkpoint (*.pt)", placeholder="optional; defaults to OUTPUT.checkpoint.pt",
                                    save=True)
        self.resume = QCheckBox("Resume from the checkpoint")
        form.addRow("Manifest", self.manifest)
        form.addRow("Save model as", self.output)
        form.addRow("Checkpoint", self.checkpoint)
        form.addRow("", self.resume)
        self.start_btn = button("Start training", self.train, "primary")
        self.actions(lay, self.start_btn)
        self.finish()
        self.role_changed()

    def role_changed(self, *_):
        action, title, category, filename, filt, about = ROLES[self.role.currentIndex()]
        self.about.setText(about)
        self.output.filter = filt
        self.output.edit.setPlaceholderText(f"where to save the trained model, e.g. {filename}")

    def set_role(self, action: str):
        self.role.setCurrentIndex(next(i for i, r in enumerate(ROLES) if r[0] == action))

    def fill(self):
        self.recent("manifest", self.manifest)

    def train(self):
        paths = self.require(manifest=self.manifest, output=self.output)
        if not paths:
            return
        action = ROLES[self.role.currentIndex()][0]
        checkpoint = self.checkpoint.text() or None
        if self.run_task(action, [paths["manifest"], paths["output"]], checkpoint=checkpoint,
                         resume=self.resume.isChecked()):
            self.pending = (action, paths["output"])

    def finished(self, job: dict):
        action, output = getattr(self, "pending", (None, None))
        if action == "train" and output:
            self.win.amo_recent["wav_model"] = output
        elif action == "train-importance" and output:
            self.win.amo_recent["importance_model"] = output


class EvaluatePage(AmoPage):
    def __init__(self, win):
        super().__init__(win, "Evaluate",
                         "Measure how well a WAV model reconstructs held-out sounds through the native DSP. "
                         "The report lists canonical and rendered-instance errors per sound.")
        form, lay = self.form_card("Native reconstruction")
        self.model = PathField(filter="PyTorch model (*.pt)", placeholder="trained WAV recognition model")
        self.manifest = PathField(filter="JSON (*.json)", placeholder="manifest with a validation split")
        self.report = PathField(filter="JSON (*.json)", placeholder="report file to write", save=True)
        form.addRow("WAV model", self.model)
        form.addRow("Manifest", self.manifest)
        form.addRow("Save report as", self.report)
        self.actions(lay, button("Evaluate", self.evaluate, "primary"))
        self.finish()

    def fill(self):
        self.recent("wav_model", self.model)
        self.recent("manifest", self.manifest)

    def evaluate(self):
        paths = self.require(wav_model=self.model, manifest=self.manifest, report=self.report)
        if paths:
            self.run_task("evaluate-wav", list(paths.values()))


class RankControlsPage(AmoPage):
    def __init__(self, win):
        super().__init__(win, "Rank controls",
                         "Order a sound's X controls by learned importance. A context importance model scores "
                         "each control of one layer for the context you describe; the plugin's Play view can then "
                         "show the most important controls first and names the model it came from.")
        form, lay = self.form_card("Learned control order",
                                   "The model must rank exactly as many dimensions as the layer has controls. "
                                   "The context file holds the model's named features (state, preference, "
                                   "context, relationship). The original sound is not changed.")
        self.sound = PathField(filter="JSON (*.json)", placeholder="sound saved from the plugin")
        self.model = PathField(filter="JSON (*.json)", placeholder="trained context importance model")
        self.context = PathField(filter="JSON (*.json)", placeholder="features describing the musical context")
        self.layer = QSpinBox()
        self.layer.setRange(0, 63)
        self.layer.setToolTip("X layer to rank; 0 is the top layer shown first in the plugin")
        self.output = PathField(filter="JSON (*.json)", placeholder="ranked copy of the sound", save=True)
        form.addRow("Sound", self.sound)
        form.addRow("Importance model", self.model)
        form.addRow("Context features", self.context)
        form.addRow("Layer", self.layer)
        form.addRow("Save sound as", self.output)
        self.actions(lay, button("Rank controls", self.rank, "primary"))
        self.finish()

    def fill(self):
        self.recent("importance_model", self.model)

    def rank(self):
        paths = self.require(sound=self.sound, importance_model=self.model, context=self.context,
                             output=self.output)
        if paths:
            self.run_task("rank-controls", [*paths.values(), "--layer", str(self.layer.value())])
