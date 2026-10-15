"""Shared widgets for AMO Centre pages: labels, buttons, cards, path fields and the page frame."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)


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


class PathField(QWidget):
    """A line edit with a Browse button, for files or folders. `save` picks a file to create."""

    def __init__(self, folder: bool = False, filter: str = "", placeholder: str = "", save: bool = False):
        super().__init__()
        self.folder, self.filter, self.save = folder, filter, save
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        lay.addWidget(self.edit, 1)
        lay.addWidget(button("Browse…", self.browse))

    def browse(self):
        if self.folder:
            path = QFileDialog.getExistingDirectory(self, "Choose a folder", self.edit.text())
        elif self.save:
            path, _ = QFileDialog.getSaveFileName(self, "Save as", self.edit.text(), self.filter)
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
