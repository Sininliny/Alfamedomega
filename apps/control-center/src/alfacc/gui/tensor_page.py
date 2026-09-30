"""Offline tensor tools run in a separate process, away from the audio host."""
import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QLineEdit,
                               QLabel, QPushButton, QHBoxLayout, QPlainTextEdit,
                               QComboBox, QDoubleSpinBox)


class TensorPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Audio Tensor Space"))
        note = QLabel("Train construction representations or import a WAV into an editable "
                      "tensor program. Offline learning runs separately from audio playback. "
                      "Models and structural templates must be supplied explicitly.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.fields = {}
        for key, title in (("repository", "Repository"), ("manifest", "Training manifest"),
                           ("model", "Model package"), ("wav", "WAV recording"),
                           ("template", "Tensor template"), ("bounds", "Parameter bounds"),
                           ("output", "Output file")):
            edit = QLineEdit()
            self.fields[key] = edit
            form.addRow(title, edit)
        self.fields["repository"].setText(win.ctx.settings.repo_dir or
                                         str(Path(__file__).resolve().parents[5]))
        self.head = QComboBox()
        self.head.addItems(["primary_canonical", "secondary_rendered_instance"])
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
        actions = QHBoxLayout()
        self.buttons = []
        for title, action in (("Train WAV model", "train"),
                              ("Train runtime compiler", "train-runtime"),
                              ("Import WAV", "import-wav")):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, action=action: self.run(action))
            actions.addWidget(button)
            self.buttons.append(button)
        stop = QPushButton("Cancel")
        stop.clicked.connect(self.cancel)
        actions.addWidget(stop)
        layout.addLayout(actions)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        layout.addWidget(self.log)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.failed)

    def refresh(self):
        pass

    def run(self, action):
        if self.process.state() != QProcess.NotRunning:
            return
        required = ["repository", "output", "manifest"] if action != "import-wav" else [
            "repository", "model", "wav", "template", "bounds", "output"]
        if any(not self.fields[key].text().strip() for key in required):
            self.log.appendPlainText("Fill the required paths before starting.")
            return
        paths = {key: str(Path(edit.text().strip()).expanduser().resolve())
                 for key, edit in self.fields.items() if edit.text().strip()}
        if action == "import-wav":
            args = [paths[key] for key in ("model", "wav", "template", "bounds", "output")]
            args += ["--head", self.head.currentText(), "--mask-threshold", str(self.threshold.value()),
                     "--sample-rate", str(self.rate.value())]
        else:
            args = [paths["manifest"], paths["output"]]
        self.process.setWorkingDirectory(paths["repository"])
        for button in self.buttons:
            button.setEnabled(False)
        self.log.appendPlainText("Started " + action)
        self.process.start(self.win.ctx.settings.training_python or sys.executable,
                           ["-m", "training.aom", action, *args])

    def read_output(self):
        self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace"))

    def finished(self, code, status):
        self.read_output()
        self.log.appendPlainText("Completed" if code == 0 and status == QProcess.NormalExit else
                                 "Stopped or failed; inspect the diagnostic above.")
        for button in self.buttons:
            button.setEnabled(True)

    def failed(self, error):
        self.log.appendPlainText(self.process.errorString())
        for button in self.buttons:
            button.setEnabled(True)

    def cancel(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
