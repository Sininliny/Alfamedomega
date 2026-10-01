"""Offline tensor tools run in a separate process, away from the audio host."""
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QLineEdit,
                               QLabel, QPushButton, QPlainTextEdit,
                               QComboBox, QDoubleSpinBox, QCheckBox, QProgressBar, QGridLayout)
from ..aom import create_tensor_job
from ..jobs import FINISHED


class TensorPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Audio Tensor Space"))
        note = QLabel("Train models, build labelled recordings, check reconstruction, or turn a WAV into an editable preset. "
                      "Choose the training Python in Settings. Every run is recorded in Jobs.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.fields = {}
        for key, title in (("repository", "Repository"), ("manifest", "Training manifest / dataset plan"),
                           ("model", "Model package"), ("checkpoint", "Checkpoint (optional)"), ("wav", "WAV recording"),
                           ("template", "Tensor template"), ("bounds", "Parameter bounds"),
                           ("output", "Output model / file / dataset folder")):
            edit = QLineEdit()
            self.fields[key] = edit
            form.addRow(title, edit)
        self.fields["repository"].setText(win.ctx.settings.repo_dir or
                                         str(Path(__file__).resolve().parents[5]))
        self.resume = QCheckBox("Resume from checkpoint")
        form.addRow("Training", self.resume)
        self.category = QComboBox()
        for title, value in (("WAV learning", "wav_preset_learning"), ("Runtime compiler", "runtime"),
                             ("User preference", "user_preference"), ("Context importance", "context_importance")):
            self.category.addItem(title, value)
        form.addRow("Preflight model category", self.category)
        self.head = QComboBox()
        self.head.addItem("Original preset", "primary_canonical")
        self.head.addItem("Rendered variation", "secondary_rendered_instance")
        form.addRow("Recovery target", self.head)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(.01, .99)
        self.threshold.setValue(.5)
        form.addRow("Presence threshold", self.threshold)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(1000, 384000)
        self.rate.setValue(48000)
        form.addRow("Render sample rate", self.rate)
        layout.addLayout(form)
        actions = QGridLayout()
        self.buttons = []
        for index, (title, action) in enumerate((("Train WAV model", "train"),
                              ("Train runtime compiler", "train-runtime"),
                              ("Train preference", "train-preference"),
                              ("Train importance", "train-importance"),
                              ("Check manifest", "preflight"),
                              ("Build dataset", "build-dataset"),
                              ("Evaluate reconstruction", "evaluate-wav"),
                              ("Import WAV", "import-wav"))):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, action=action: self.run(action))
            actions.addWidget(button, index // 3, index % 3)
            self.buttons.append(button)
        stop = QPushButton("Cancel")
        stop.clicked.connect(self.cancel)
        actions.addWidget(stop, 3, 0)
        layout.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.status = QLabel("Ready. Training uses the Python interpreter selected in Settings.")
        layout.addWidget(self.status)
        history = QPushButton("Open job history")
        history.clicked.connect(lambda: self.win.jobs_changed(self.job_id))
        layout.addWidget(history)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        layout.addWidget(self.log)
        self.job_id = None
        self._last_log = ""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(500)

    def refresh(self):
        if not self.job_id:
            return
        job = self.win.ctx.jobs.get(self.job_id)
        if not job:
            return
        text = self.win.ctx.jobs.read_log(self.job_id, tail=1000)
        if text != self._last_log:
            self.log.setPlainText(text)
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            self._last_log = text
        status = {"queued": "Queued", "running": "Working", "succeeded": "Completed",
                  "failed": "Failed", "cancelled": "Cancelled", "interrupted": "Interrupted"}.get(job["status"], job["status"])
        self.status.setText(status + (": " + job["error"] if job.get("error") else ""))
        progress = job.get("progress")
        if progress and progress[1] > 0:
            self.progress.setRange(0, progress[1])
            self.progress.setValue(progress[0])
        elif job["status"] not in FINISHED:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100 if job["status"] == "succeeded" else 0)

    def run(self, action):
        required = ["repository", "manifest"] if action == "preflight" else ["repository", "output", "manifest"]
        if action == "evaluate-wav":
            required = ["repository", "model", "manifest", "output"]
        if action == "import-wav":
            required = ["repository", "model", "wav", "template", "bounds", "output"]
        if any(not self.fields[key].text().strip() for key in required):
            self.log.appendPlainText("Fill the required paths before starting.")
            return
        paths = {key: str(Path(edit.text().strip()).expanduser().resolve())
                 for key, edit in self.fields.items() if edit.text().strip()}
        if action == "import-wav":
            args = [paths[key] for key in ("model", "wav", "template", "bounds", "output")]
            args += ["--head", self.head.currentData(), "--mask-threshold", str(self.threshold.value()),
                     "--sample-rate", str(self.rate.value())]
        elif action == "evaluate-wav":
            args = [paths["model"], paths["manifest"], paths["output"]]
        elif action == "preflight":
            args = [paths["manifest"], "--category", self.category.currentData()]
        else:
            args = [paths["manifest"], paths["output"]]
        try:
            job = create_tensor_job(self.win.ctx.jobs, paths["repository"], action, args,
                                    python=self.win.ctx.settings.resolved_training_python(),
                                    checkpoint=paths.get("checkpoint"), resume=self.resume.isChecked())
            self.job_id = job["id"]
            self._last_log = ""
            self.win.ctx.jobs.enqueue(self.job_id)
            self.refresh()
        except Exception as error:
            self.log.appendPlainText(str(error))

    def cancel(self):
        if self.job_id:
            self.win.ctx.jobs.cancel(self.job_id)
            self.status.setText("Stopping; completed training steps will be checkpointed.")
