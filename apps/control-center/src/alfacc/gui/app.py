"""alfa-cc-gui: the control center window (PySide6).

A thin layer over the core: every button creates a job or calls the model store, so the GUI and
the CLI always agree. Jobs run one at a time in the background; the window polls their records.
"""

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import __version__, jobs
from ..context import Context
from ..models import BUNDLE_SUFFIX, ModelStoreError
from ..preference import create_batch_job
from ..tiers import DEFAULT_TIER, STAGES, TIERS
from ..training import (TrainSpec, check_environment, create_export_job, create_training_job,
                        write_colab_notebook)
from .theme import TOKENS, stylesheet

STATUS_TEXT = {jobs.QUEUED: "Queued", jobs.RUNNING: "Running", jobs.SUCCEEDED: "Done",
               jobs.FAILED: "Failed", jobs.CANCELLED: "Cancelled", jobs.INTERRUPTED: "Interrupted"}


# ---------------------------------------------------------------------- small widgets

def label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def button(text: str, slot, name: str = "", tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    if tip:
        b.setToolTip(tip)
    b.clicked.connect(slot)
    return b


def card() -> tuple:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(6)
    return f, lay


def item(text, align_right=False, mono=False) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if align_right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if mono:
        it.setFont(QFont("IBM Plex Mono"))
    return it


def table(headers: list) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setAlternatingRowColors(True)
    t.setShowGrid(False)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setHighlightSections(False)
    return t


def ago(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso).astimezone()
    except ValueError:
        return iso
    return dt.strftime("%Y-%m-%d %H:%M")


class PathField(QWidget):
    """A line edit with a Browse button, for files or folders."""

    def __init__(self, folder: bool = False, filter: str = "", placeholder: str = ""):
        super().__init__()
        self.folder, self.filter = folder, filter
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        lay.addWidget(self.edit, 1)
        lay.addWidget(button("Browse…", self.browse))

    def browse(self):
        if self.folder:
            path = QFileDialog.getExistingDirectory(self, "Choose a folder", self.edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose a file", self.edit.text(), self.filter)
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str):
        self.edit.setText(t)


class Page(QWidget):
    def __init__(self, win: "MainWindow", title: str, subtitle: str):
        super().__init__()
        self.setObjectName("page")
        self.win, self.ctx = win, win.ctx
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("page")
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(28, 24, 28, 24)
        self.lay.setSpacing(14)
        self.lay.addWidget(label(title, "h1"))
        self.lay.addWidget(label(subtitle, "muted", wrap=True))
        scroll.setWidget(body)
        outer.addWidget(scroll)

    def refresh(self):
        pass

    def start(self, job: dict):
        self.ctx.jobs.enqueue(job["id"])
        self.win.jobs_changed(job["id"])


# ---------------------------------------------------------------------- Models

class ModelsPage(Page):
    def __init__(self, win):
        super().__init__(win, "Models",
                         "Weights for each model size, in the folder the plugin reads. Lite and "
                         "Standard come with the base install; Pro and Studio are add-ons.")
        grid = QGridLayout()
        grid.setSpacing(12)
        self.tier_cards = {}
        for i, (name, t) in enumerate(TIERS.items()):
            f, lay = card()
            head = QHBoxLayout()
            head.addWidget(label(t.label, "h2"))
            head.addStretch(1)
            badge = label("base" if t.base_install else "add-on", "muted")
            head.addWidget(badge)
            lay.addLayout(head)
            lay.addWidget(label(f"{t.params // 1_000_000} M parameters\nabout {t.shipped_mb} MB on disk", "muted"))
            state = label("", "", wrap=True)
            lay.addWidget(state)
            grid.addWidget(f, 0, i)
            self.tier_cards[name] = (f, state)
        self.lay.addLayout(grid)
        row = QHBoxLayout()
        row.addWidget(label("Default size for new instances", "muted"))
        self.default_combo = QComboBox()
        for name, t in TIERS.items():
            self.default_combo.addItem(t.label, name)
        self.default_combo.activated.connect(self.set_default)
        row.addWidget(self.default_combo)
        row.addStretch(1)
        self.lay.addLayout(row)

        bar = QHBoxLayout()
        bar.addWidget(button("Import…", self.import_model, "primary",
                             "Install a .alfamodel bundle, a training export folder or a zipped export"))
        bar.addWidget(button("Import folder…", self.import_folder))
        self.export_btn = button("Export bundle…", self.export_bundle)
        self.verify_btn = button("Verify all", self.verify)
        self.remove_btn = button("Remove", self.remove, "danger")
        for b in (self.export_btn, self.verify_btn, self.remove_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        self.lay.addLayout(bar)

        self.table = table(["Model", "Size", "Version", "On disk", "Files", "Installed", "Notes"])
        self.table.itemSelectionChanged.connect(self.update_buttons)
        self.table.setMinimumHeight(220)
        self.lay.addWidget(self.table, 1)
        self.folder_label = label("", "muted", wrap=True)
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lay.addWidget(self.folder_label)
        self.refresh()

    def selected_id(self):
        rows = self.table.selectionModel().selectedRows()
        return self.table.item(rows[0].row(), 0).text() if rows else None

    def update_buttons(self):
        has = self.selected_id() is not None
        self.export_btn.setEnabled(has)
        self.remove_btn.setEnabled(has)

    def refresh(self):
        store = self.ctx.store
        self.folder_label.setText(f"Models folder: {store.root}")
        try:
            man = store.manifest()
        except (ModelStoreError, ValueError) as e:
            self.folder_label.setText(f"Can't read {store.root}: {e}")
            return
        models = man["models"]
        keep = self.selected_id()
        self.table.setRowCount(len(models))
        for r, m in enumerate(models):
            vals = [item(m["id"], mono=True), item(TIERS[m["tier"]].label), item(m["version"], mono=True),
                    item(f"{m['bytes'] / 1e6:.1f} MB", True, True), item(len(m["files"]), True),
                    item(ago(m.get("installed_at"))), item(m.get("notes", ""))]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, v)
            if m["id"] == keep:
                self.table.selectRow(r)
        self.table.resizeColumnsToContents()
        for name, (f, state) in self.tier_cards.items():
            versions = [m["version"] for m in models if m["tier"] == name]
            default = man["default_tier"] == name
            if versions:
                state.setText(("Default · " if default else "") + "installed " + ", ".join(versions))
                state.setObjectName("ok")
            else:
                state.setText("not installed" + (" · default" if default else ""))
                state.setObjectName("warn" if TIERS[name].base_install else "muted")
            f.setProperty("installed", bool(versions))
            for w in (f, state):
                w.style().unpolish(w)
                w.style().polish(w)
        self.default_combo.setCurrentIndex(list(TIERS).index(man.get("default_tier", DEFAULT_TIER)))
        self.verify_btn.setEnabled(bool(models))
        self.update_buttons()

    def _import(self, src: str):
        job = self.ctx.jobs.create("import", f"Import {Path(src).name}",
                                   [jobs.action_step("import_model", label="Install model",
                                                     models_dir=str(self.ctx.models_dir), src=src)])
        self.start(job)

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import model", str(Path.home()),
            f"Alfamedomega models (*{BUNDLE_SUFFIX} *.zip manifest.json);;All files (*)")
        if path:
            self._import(str(Path(path).parent) if Path(path).name == "manifest.json" else path)

    def import_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a training export folder", str(Path.home()))
        if path:
            self._import(path)

    def export_bundle(self):
        mid = self.selected_id()
        if not mid:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Export model",
                                              str(Path.home() / f"alfamedomega-{mid}{BUNDLE_SUFFIX}"),
                                              f"Alfamedomega model (*{BUNDLE_SUFFIX})")
        if dest:
            self.start(self.ctx.jobs.create("export-bundle", f"Export {mid}", [jobs.action_step(
                "export_bundle", label="Pack bundle", models_dir=str(self.ctx.models_dir), id=mid, dest=dest)]))

    def verify(self):
        self.start(self.ctx.jobs.create("verify", "Verify installed models", [jobs.action_step(
            "verify_models", label="Check SHA-256", models_dir=str(self.ctx.models_dir))]))

    def remove(self):
        mid = self.selected_id()
        if mid and QMessageBox.question(self, "Remove model",
                                        f"Remove {mid}? Projects that use it will fall back to the "
                                        "nearest installed size.") == QMessageBox.Yes:
            try:
                self.ctx.store.remove(mid)
            except (ModelStoreError, OSError) as e:
                self.win.error(str(e))
            self.refresh()

    def set_default(self):
        try:
            self.ctx.store.set_default_tier(self.default_combo.currentData())
        except (ModelStoreError, OSError) as e:
            self.win.error(str(e))
        self.refresh()


# ---------------------------------------------------------------------- Training

class TrainingPage(Page):
    def __init__(self, win):
        super().__init__(win, "Training",
                         "Train a model size with training/alfamedomega_train.py, then export it and "
                         "install it into the Models folder. Train here with a local PyTorch, or make a "
                         "Colab notebook and import the export when it finishes.")
        f, lay = card()
        self.env_label = label("Checking the training environment…", "muted", wrap=True)
        lay.addWidget(self.env_label)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Check again", self.check_env))
        lay.addLayout(row)
        self.lay.addWidget(f)

        f, lay = card()
        lay.addWidget(label("Run", "h2"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(16)
        self.tier = QComboBox()
        for name, t in TIERS.items():
            self.tier.addItem(f"{t.label} ({t.params // 1_000_000} M)", name)
        self.stage = QComboBox()
        stage_help = {"pretrain": "Pretrain on teacher programs", "polish": "Polish (hybrids, lower rate)",
                      "finetune": "Fine-tune on real notes", "distill": "Distil from Studio"}
        for s in STAGES:
            self.stage.addItem(stage_help[s], s)
        self.stage.currentIndexChanged.connect(self.update_fields)
        self.steps = QSpinBox()
        self.steps.setRange(1, 10_000_000)
        self.steps.setSingleStep(1000)
        self.steps.setValue(200_000)
        self.batch = QSpinBox()
        self.batch.setRange(1, 1024)
        self.batch.setValue(16)
        self.accum = QSpinBox()
        self.accum.setRange(1, 64)
        self.accum.setValue(4)
        self.clip = QDoubleSpinBox()
        self.clip.setRange(0.1, 10)
        self.clip.setValue(1.0)
        self.clip.setSuffix(" s")
        self.init = PathField(filter="Checkpoints (*.pt)", placeholder="optional: continue from a checkpoint")
        self.teacher = PathField(filter="Checkpoints (*.pt)", placeholder="Studio checkpoint")
        self.data = PathField(folder=True, placeholder="NSynth-layout folder of labelled notes")
        self.version = QLineEdit()
        self.version.setObjectName("mono")
        self.version.setPlaceholderText("next patch version")
        self.smoke = QCheckBox("Smoke test (a few tiny steps on the CPU, to check the pipeline)")
        self.export = QCheckBox("Export ONNX graphs when training finishes")
        self.export.setChecked(True)
        self.install = QCheckBox("Install the export into the Models folder")
        self.install.setChecked(True)
        self.export.toggled.connect(self.install.setEnabled)
        self.int8 = QCheckBox("Also write int8 copies (for Lite)")
        form.addRow("Model size", self.tier)
        form.addRow("Stage", self.stage)
        form.addRow("Steps", self.steps)
        form.addRow("Batch × accumulation", self._pair(self.batch, self.accum))
        form.addRow("Clip length", self.clip)
        form.addRow("Checkpoint", self.init)
        self.teacher_row = ("Teacher", self.teacher)
        form.addRow(*self.teacher_row)
        form.addRow("Notes folder", self.data)
        form.addRow("Install as version", self.version)
        form.addRow("", self.smoke)
        form.addRow("", self.export)
        form.addRow("", self.install)
        form.addRow("", self.int8)
        self.form = form
        lay.addLayout(form)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Make Colab notebook…", self.colab,
                             tip="A notebook with the training script inside; results go to your Drive"))
        self.start_btn = button("Start training", self.train, "primary")
        row.addWidget(self.start_btn)
        lay.addLayout(row)
        self.lay.addWidget(f)

        f, lay = card()
        lay.addWidget(label("Export a checkpoint", "h2"))
        lay.addWidget(label("For a checkpoint trained on Colab or elsewhere: export its ONNX graphs "
                            "and install them.", "muted", wrap=True))
        row = QHBoxLayout()
        self.ckpt_tier = QComboBox()
        for name, t in TIERS.items():
            self.ckpt_tier.addItem(t.label, name)
        self.ckpt = PathField(filter="Checkpoints (*.pt)", placeholder="checkpoint (.pt)")
        row.addWidget(self.ckpt_tier)
        row.addWidget(self.ckpt, 1)
        self.export_btn = button("Export and install", self.export_ckpt)
        row.addWidget(self.export_btn)
        lay.addLayout(row)
        self.lay.addWidget(f)
        self.lay.addStretch(1)
        self.env = None
        self.update_fields()
        QTimer.singleShot(0, self.check_env)

    @staticmethod
    def _pair(a, b):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(QLabel("×"))
        h.addWidget(b)
        h.addStretch(1)
        return w

    def update_fields(self):
        stage = self.stage.currentData()
        self.teacher.setEnabled(stage == "distill")
        self.data.setEnabled(stage in ("finetune", "distill"))

    def check_env(self):
        self.env = check_environment(self.ctx.python, self.ctx.script)
        e = self.env
        if e.get("ok"):
            gpu = "CUDA GPU available" if e["cuda"] else "no CUDA GPU, so only smoke tests are practical here"
            self.env_label.setText(f"Ready: Python {e['python_version']}, PyTorch {e['torch']}, "
                                   f"ONNX Runtime {e['onnxruntime']}, {gpu}.\nScript: {e['script']}")
            self.env_label.setObjectName("ok")
        else:
            self.env_label.setText(
                f"Local training isn't available: {e.get('error')}.\nPoint Settings at a Python "
                "with PyTorch and ONNX Runtime, or use a Colab notebook.")
            self.env_label.setObjectName("warn")
        self.env_label.style().unpolish(self.env_label)
        self.env_label.style().polish(self.env_label)
        self.start_btn.setEnabled(bool(e.get("ok")))
        self.export_btn.setEnabled(bool(e.get("ok")))

    def spec(self) -> TrainSpec:
        return TrainSpec(tier=self.tier.currentData(), stage=self.stage.currentData(),
                         steps=self.steps.value(), batch=self.batch.value(), accum=self.accum.value(),
                         clip=self.clip.value(), init=self.init.text(), teacher=self.teacher.text(),
                         data=self.data.text(), smoke=self.smoke.isChecked(),
                         export=self.export.isChecked(),
                         install=self.export.isChecked() and self.install.isChecked(),
                         version=self.version.text().strip(), int8=self.int8.isChecked())

    def train(self):
        try:
            job = create_training_job(self.ctx.jobs, self.spec(), self.ctx.python, self.ctx.script,
                                      self.ctx.runs_dir, self.ctx.models_dir)
        except (ValueError, OSError) as e:
            return self.win.error(str(e))
        self.start(job)

    def colab(self):
        try:
            spec = self.spec()
            spec.validate()
        except ValueError as e:
            return self.win.error(str(e))
        if not self.ctx.script:
            return self.win.error("The training script wasn't found. Set it in Settings.")
        dest, _ = QFileDialog.getSaveFileName(self, "Save Colab notebook",
                                              str(Path.home() / f"alfamedomega-{spec.run_name()}.ipynb"),
                                              "Notebooks (*.ipynb)")
        if dest:
            path = write_colab_notebook(spec, self.ctx.script, Path(dest))
            self.win.statusBar().showMessage(f"Wrote {path}. Open it in Colab (File > Upload notebook).", 10000)

    def export_ckpt(self):
        try:
            job = create_export_job(self.ctx.jobs, self.ckpt_tier.currentData(), self.ckpt.text(),
                                    self.ctx.python, self.ctx.script, self.ctx.runs_dir,
                                    self.ctx.models_dir)
        except (ValueError, OSError) as e:
            return self.win.error(str(e))
        self.start(job)


# ---------------------------------------------------------------------- Preference learning

class PreferencePage(Page):
    def __init__(self, win):
        super().__init__(win, "Preference learning",
                         "Teach Alfamedomega your taste from sample and preset folders you choose. "
                         "File and folder names become labels such as “warm”, “airy” or “dark”, and "
                         "the learner maps them to sounds. Everything stays on this computer.")
        f, lay = card()
        head = QHBoxLayout()
        head.addWidget(label("Libraries", "h2"))
        head.addStretch(1)
        head.addWidget(button("Add folder…", self.add))
        self.remove_btn = button("Remove", self.remove)
        head.addWidget(self.remove_btn)
        lay.addLayout(head)
        self.libs = table(["Use", "Name", "Folder"])
        self.libs.setMinimumHeight(150)
        self.libs.itemChanged.connect(self.toggled)
        self.libs.itemSelectionChanged.connect(
            lambda: self.remove_btn.setEnabled(bool(self.libs.selectionModel().selectedRows())))
        lay.addWidget(self.libs)
        self.lay.addWidget(f)

        f, lay = card()
        head = QHBoxLayout()
        head.addWidget(label("What it has learned from", "h2"))
        head.addStretch(1)
        head.addWidget(button("Forget everything…", self.forget, "danger",
                              "Delete the index, behaviour log and learned profiles"))
        self.batch_btn = button("Run batch", self.batch, "primary",
                                "Index the libraries, then run the preference learner")
        head.addWidget(self.batch_btn)
        lay.addLayout(head)
        self.summary = label("", "muted", wrap=True)
        lay.addWidget(self.summary)
        self.labels = label("", "", wrap=True)
        self.labels.setTextFormat(Qt.RichText)
        lay.addWidget(self.labels)
        self.learner = label("", "muted", wrap=True)
        lay.addWidget(self.learner)
        self.lay.addWidget(f)
        self.lay.addStretch(1)
        self._filling = False
        self.refresh()

    def refresh(self):
        p = self.ctx.prefs
        libs = p.libraries()
        self._filling = True
        self.libs.setRowCount(len(libs))
        for r, lib in enumerate(libs):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            use.setCheckState(Qt.Checked if lib.get("enabled", True) else Qt.Unchecked)
            use.setData(Qt.UserRole, lib["path"])
            self.libs.setItem(r, 0, use)
            self.libs.setItem(r, 1, item(lib["label"]))
            self.libs.setItem(r, 2, item(lib["path"]))
        self.libs.resizeColumnsToContents()
        self._filling = False
        self.remove_btn.setEnabled(bool(self.libs.selectionModel().selectedRows()))
        self.batch_btn.setEnabled(any(lib.get("enabled", True) for lib in libs))

        voc = p.vocabulary()
        s = voc["summary"]
        if s:
            self.summary.setText(
                f"{s['files']} sounds in {s['libraries']} "
                f"{'library' if s['libraries'] == 1 else 'libraries'}, {s['labelled_files']} with name "
                f"labels, {s['labels']} distinct labels. Indexed {ago(s['scanned_at'])}. "
                f"Learned profiles: {len(p.profiles())}.")
            chips = " ".join(
                f"<span style='color:{TOKENS['accent']}'>{k}</span>"
                f"<span style='color:{TOKENS['text2']}'>&nbsp;{v}</span>&nbsp;&nbsp;"
                for k, v in list(voc["labels"].items())[:40])
            self.labels.setText(chips)
        else:
            self.summary.setText("Nothing indexed yet. Add a folder and run a batch.")
            self.labels.setText("")
        learner = self.ctx.settings.preference_learner
        self.learner.setText(f"Learner: {learner}" if learner else
                             "No preference learner is set yet, so a batch indexes your libraries "
                             "and stops there. Set it in Settings once preference/ provides one.")

    def toggled(self, it: QTableWidgetItem):
        if self._filling or it.column() != 0:
            return
        self.ctx.prefs.set_enabled(Path(it.data(Qt.UserRole)), it.checkState() == Qt.Checked)
        self.refresh()

    def add(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a sample or preset folder", str(Path.home()))
        if path:
            try:
                self.ctx.prefs.add_library(Path(path))
            except ValueError as e:
                self.win.error(str(e))
            self.refresh()

    def remove(self):
        rows = self.libs.selectionModel().selectedRows()
        if rows:
            self.ctx.prefs.remove_library(Path(self.libs.item(rows[0].row(), 0).data(Qt.UserRole)))
            self.refresh()

    def batch(self):
        try:
            job = create_batch_job(self.ctx.jobs, self.ctx.home, self.ctx.settings.preference_learner)
        except ValueError as e:
            return self.win.error(str(e))
        self.start(job)

    def forget(self):
        if QMessageBox.question(self, "Forget everything",
                                "Delete the library index, the behaviour log and every learned "
                                "profile? Your sample folders are not touched.") == QMessageBox.Yes:
            self.ctx.prefs.forget()
            self.refresh()


# ---------------------------------------------------------------------- Jobs

class JobsPage(Page):
    def __init__(self, win):
        super().__init__(win, "Jobs", "Training, exports, imports and preference batches, one at a time.")
        self.table = table(["Job", "Status", "Progress", "Started", "Finished"])
        self.table.setMinimumHeight(200)
        self.table.itemSelectionChanged.connect(self.show_log)
        self.lay.addWidget(self.table)
        row = QHBoxLayout()
        self.detail = label("", "muted", wrap=True)
        row.addWidget(self.detail, 1)
        self.cancel_btn = button("Cancel", self.cancel, "danger")
        row.addWidget(self.cancel_btn)
        self.lay.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(240)
        self.log.setMaximumBlockCount(5000)
        self.lay.addWidget(self.log, 1)
        self.ids = []
        self.refresh()

    def selected(self):
        rows = self.table.selectionModel().selectedRows()
        return self.ids[rows[0].row()] if rows and rows[0].row() < len(self.ids) else None

    def select(self, job_id):
        if job_id in self.ids:
            self.table.selectRow(self.ids.index(job_id))

    def refresh(self):
        keep = self.selected()
        all_jobs = self.ctx.jobs.list()[:200]
        self.ids = [j["id"] for j in all_jobs]
        self.table.setRowCount(len(all_jobs))
        for r, j in enumerate(all_jobs):
            self.table.removeCellWidget(r, 2)
            self.table.setItem(r, 0, item(j["title"]))
            st = item(STATUS_TEXT.get(j["status"], j["status"]))
            color = {jobs.RUNNING: TOKENS["mod"], jobs.FAILED: TOKENS["warn"],
                     jobs.INTERRUPTED: TOKENS["warn"], jobs.SUCCEEDED: TOKENS["ok"]}.get(j["status"])
            if color:
                st.setForeground(QColor(color))
            self.table.setItem(r, 1, st)
            prog = j.get("progress")
            n_steps = max(1, len(j["steps"]))
            if j["status"] == jobs.RUNNING:
                bar = QProgressBar()
                if prog:
                    bar.setRange(0, prog[1])
                    bar.setValue(prog[0])
                    bar.setFormat(f"{prog[0]}/{prog[1]}  · step {j['current_step'] + 1}/{n_steps}")
                else:
                    bar.setRange(0, 0)  # busy, no count yet
                self.table.setItem(r, 2, item(""))
                self.table.setCellWidget(r, 2, bar)
            elif j["status"] == jobs.SUCCEEDED:
                self.table.setItem(r, 2, item(f"{n_steps} of {n_steps} steps", mono=True))
            elif j["status"] == jobs.QUEUED:
                self.table.setItem(r, 2, item("waiting"))
            else:
                self.table.setItem(r, 2, item(f"stopped at step {j['current_step'] + 1} of {n_steps}"))
            self.table.setItem(r, 3, item(ago(j.get("started"))))
            self.table.setItem(r, 4, item(ago(j.get("finished"))))
        self.table.setColumnWidth(0, 300)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 220)
        self.table.setColumnWidth(3, 140)
        if keep:
            self.select(keep)
        elif self.ids:
            self.table.selectRow(0)
        self.show_log()

    def show_log(self):
        jid = self.selected()
        if not jid:
            self.detail.setText("")
            self.log.setPlainText("")
            self.cancel_btn.setEnabled(False)
            return
        j = self.ctx.jobs.get(jid)
        steps = " → ".join(s["label"] for s in j["steps"])
        err = f"   Error: {j['error']}" if j.get("error") else ""
        res = "   " + ", ".join(f"{k}: {v}" for k, v in j["result"].items()
                                 if isinstance(v, str)) if j.get("result") else ""
        self.detail.setText(f"{steps}{err}{res}")
        self.cancel_btn.setEnabled(j["status"] in (jobs.QUEUED, jobs.RUNNING))
        text = self.ctx.jobs.read_log(jid, tail=2000)
        if text != self.log.toPlainText():
            at_end = self.log.verticalScrollBar().value() >= self.log.verticalScrollBar().maximum() - 4
            self.log.setPlainText(text)
            if at_end:
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def cancel(self):
        jid = self.selected()
        if jid:
            self.ctx.jobs.cancel(jid)
            self.refresh()


# ---------------------------------------------------------------------- Settings

class SettingsPage(Page):
    def __init__(self, win):
        super().__init__(win, "Settings", f"Control center {__version__}. Data folder: {win.ctx.home}")
        f, lay = card()
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(16)
        self.models_dir = PathField(folder=True)
        self.python = PathField(placeholder="Python with PyTorch and ONNX Runtime")
        self.script = PathField(filter="Python (*.py)", placeholder="training/alfamedomega_train.py")
        self.runs = PathField(folder=True)
        self.learner = QLineEdit()
        self.learner.setObjectName("mono")
        self.learner.setPlaceholderText("for example: python -m alfa_preference")
        form.addRow("Models folder", self.models_dir)
        form.addRow("", label("Where the plugin looks for weights. Leave empty for the standard "
                              "system folder.", "muted", wrap=True))
        form.addRow("Training Python", self.python)
        form.addRow("Training script", self.script)
        form.addRow("Training runs", self.runs)
        form.addRow("Preference learner", self.learner)
        lay.addLayout(form)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Save", self.save, "primary"))
        lay.addLayout(row)
        self.lay.addWidget(f)
        self.lay.addStretch(1)
        self.refresh()

    def refresh(self):
        s = self.ctx.settings
        self.models_dir.setText(s.models_dir)
        self.models_dir.edit.setPlaceholderText(str(self.ctx.models_dir))
        self.python.setText(s.training_python)
        self.python.edit.setPlaceholderText(self.ctx.python)
        self.script.setText(s.training_script)
        self.script.edit.setPlaceholderText(self.ctx.script or "not found")
        self.runs.setText(s.runs_dir)
        self.runs.edit.setPlaceholderText(str(self.ctx.runs_dir))
        self.learner.setText(s.preference_learner)

    def save(self):
        s = self.ctx.settings
        s.models_dir, s.training_python = self.models_dir.text(), self.python.text()
        s.training_script, s.runs_dir = self.script.text(), self.runs.text()
        s.preference_learner = self.learner.text().strip()
        self.ctx.save_settings()
        self.win.statusBar().showMessage("Settings saved", 4000)
        self.win.refresh_all()


# ---------------------------------------------------------------------- window

class MainWindow(QMainWindow):
    def __init__(self, ctx: Context):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("Alfamedomega Control Center")
        self.resize(1180, 800)
        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(200)
        self.stack = QStackedWidget()
        self.pages = [ModelsPage(self), TrainingPage(self), PreferencePage(self), JobsPage(self),
                      SettingsPage(self)]
        for name, page in zip(["Models", "Training", "Preference learning", "Jobs", "Settings"], self.pages):
            self.nav.addItem(QListWidgetItem(name))
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.show_page)
        brand = QWidget()
        brand.setStyleSheet(f"background:{TOKENS['panel']}")
        bl = QVBoxLayout(brand)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        title = label("  Alfamedomega", "h2")
        title.setContentsMargins(12, 18, 0, 0)
        bl.addWidget(title)
        sub = label("   Control center", "muted")
        sub.setContentsMargins(12, 0, 0, 8)
        bl.addWidget(sub)
        bl.addWidget(self.nav, 1)
        brand.setFixedWidth(200)
        h.addWidget(brand)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.job_status = QLabel("")
        self.statusBar().addPermanentWidget(self.job_status)
        self.nav.setCurrentRow(0)
        for i in range(len(self.pages)):
            act = QAction(self)
            act.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            act.triggered.connect(lambda _=False, i=i: self.nav.setCurrentRow(i))
            self.addAction(act)
        self._last_state = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1000)

    def show_page(self, i):
        self.stack.setCurrentIndex(i)
        self.pages[i].refresh()

    def refresh_all(self):
        for p in self.pages:
            p.refresh()

    def jobs_changed(self, job_id=None):
        self.nav.setCurrentRow(3)
        if job_id:
            self.pages[3].select(job_id)
        self.poll()

    def poll(self):
        live = [j for j in self.ctx.jobs.list()[:50] if j["status"] in (jobs.RUNNING, jobs.QUEUED)]
        running = next((j for j in live if j["status"] == jobs.RUNNING), None)
        if running:
            p = running.get("progress")
            self.job_status.setText(f"● {running['title']}" + (f"  {p[0]}/{p[1]}" if p else "")
                                    + (f"  (+{len(live) - 1} queued)" if len(live) > 1 else ""))
            self.job_status.setStyleSheet(f"color:{TOKENS['mod']}")
        else:
            self.job_status.setText("No jobs running")
            self.job_status.setStyleSheet("")
        state = tuple((j["id"], j["status"], tuple(j.get("progress") or ())) for j in live)
        if state != self._last_state or live:
            if self.stack.currentIndex() == 3:
                self.pages[3].refresh()
            if state != self._last_state:
                # A job just finished or started: models or the index may have changed.
                for i in (0, 2):
                    self.pages[i].refresh()
        self._last_state = state

    def error(self, text: str):
        QMessageBox.warning(self, "Alfamedomega", text)


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("Alfamedomega Control Center")
    app.setStyleSheet(stylesheet())
    home = models = None
    args = argv[1:]
    for flag in ("--home", "--models-dir"):
        if flag in args and args.index(flag) + 1 < len(args):
            val = args[args.index(flag) + 1]
            home, models = (val, models) if flag == "--home" else (home, val)
    win = MainWindow(Context(home, models))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
