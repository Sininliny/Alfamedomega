import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)


@pytest.fixture(scope="module")
def qapp():
    from alfacc.gui.theme import stylesheet
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Windows offscreen sessions can lack the system font database. Load one
    # explicit system font so rendered QA also verifies readable labels.
    if os.name == "nt":
        from PySide6.QtGui import QFontDatabase
        if not QFontDatabase.families():
            QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    app.setStyleSheet(stylesheet())
    return app


def test_window_runs_a_training_job_and_shows_the_model(qapp, ctx):
    from alfacc.gui.app import MainWindow
    win = MainWindow(ctx)
    win.show()
    for i in range(win.nav.count()):
        win.nav.setCurrentRow(i)
        qapp.processEvents()
    training = win.legacy_training_page
    training.check_env()  # the test interpreter has no PyTorch, so local training is off
    assert not training.start_btn.isEnabled()
    training.steps.setValue(4)
    training.version.setText("3.0.0")
    training.train()  # the button is disabled, but the action itself still works
    assert win.stack.currentWidget() is win.jobs_page
    deadline = time.time() + 20
    while time.time() < deadline and not ctx.jobs.wait(0.05):
        qapp.processEvents()
    win.poll()
    assert ctx.jobs.list()[0]["status"] == "succeeded"
    assert win.models_page.table.rowCount() == 1
    assert win.models_page.table.item(0, 0).text() == "lite-3.0.0"
    win.jobs_page.refresh()
    assert "step 4/4" in win.jobs_page.log.toPlainText()
    win.close()


def test_tensor_page_uses_durable_training_jobs(qapp, ctx, tmp_path):
    python = os.environ.get("AOM_TRAINING_TEST_PYTHON")
    if not python:
        pytest.skip("Set AOM_TRAINING_TEST_PYTHON for the offline AOM integration")
    import json
    from pathlib import Path
    from alfacc.gui.app import MainWindow
    ctx.settings.training_python = python
    manifest = tmp_path / "feedback.json"
    manifest.write_text(json.dumps({"schema": "aom.training-manifest/1", "category": "user_preference",
        "feature_widths": {"state": 1}, "hidden_width": 4, "steps": 3, "learning_rate": .01,
        "available_ram_bytes": 10_000_000, "seed": 2, "observations": [{"group_id": "local",
        "first": {"state": [.8]}, "second": {"state": [.2]}, "preferred_first": True}]}), encoding="utf-8")
    output = tmp_path / "local-model.json"
    win = MainWindow(ctx)
    win.show()
    ctx.settings.repo_dir = str(Path(__file__).resolve().parents[3])
    page = win.train_page
    page.set_role("train-preference")
    page.manifest.setText(str(manifest))
    page.output.setText(str(output))
    page.train()
    first_job_id = page.job_id
    second_output = tmp_path / "second-local-model.json"
    page.output.setText(str(second_output))
    page.train()
    assert page.job_id != first_job_id
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline and not ctx.jobs.wait(.03):
        qapp.processEvents()
    page.refresh()
    assert ctx.jobs.get(page.job_id)["status"] == "succeeded", page.log.toPlainText()
    assert ctx.jobs.get(first_job_id)["status"] == "succeeded"
    assert page.progress.value() == page.progress.maximum()
    assert "saved " in page.log.toPlainText()
    assert ctx.store.models() == []
    assert json.loads(output.read_text())["category"] == "user_preference"
    win.grab().save(str(tmp_path / "aom-centre.png"))
    win.close()


def test_aom_centre_puts_tensor_work_first_and_checks_inputs(qapp, ctx):
    from alfacc.gui.app import MainWindow
    win = MainWindow(ctx)
    win.show()
    assert win.windowTitle() == "AOM Centre"
    assert win.stack.currentWidget() is win.import_page
    names = [win.nav.item(i).text() for i in range(win.nav.count())]
    assert names.index("Import sound") < names.index("Jobs") < names.index("LEGACY") < names.index("Models")
    # Section headings cannot be selected and do not change the page.
    win.nav.setCurrentRow(names.index("LEGACY"))
    assert win.stack.currentWidget() is win.import_page
    # A task with missing inputs says which ones instead of queuing a job.
    win.show_page(win.evaluate_page)
    win.evaluate_page.evaluate()
    assert win.evaluate_page.job_id is None
    assert "WAV model" in win.evaluate_page.panel.status.text()
    assert ctx.jobs.list() == []
    # Results chain forward into empty fields only.
    win.aom_recent.update(manifest="/data/manifest.json", wav_model="/data/model.pt")
    win.evaluate_page.manifest.setText("/mine/manifest.json")
    win.show_page(win.datasets_page)
    win.show_page(win.evaluate_page)
    assert win.evaluate_page.model.text() == "/data/model.pt"
    assert win.evaluate_page.manifest.text() == "/mine/manifest.json"
    win.close()
