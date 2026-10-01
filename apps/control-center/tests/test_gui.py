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
    for i in range(len(win.pages)):
        win.nav.setCurrentRow(i)
        qapp.processEvents()
    training = win.pages[1]
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
    assert win.pages[0].table.rowCount() == 1
    assert win.pages[0].table.item(0, 0).text() == "lite-3.0.0"
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
    page = win.pages[6]
    page.fields["repository"].setText(str(Path(__file__).resolve().parents[3]))
    page.fields["manifest"].setText(str(manifest))
    page.fields["output"].setText(str(output))
    page.run("train-preference")
    first_job_id = page.job_id
    second_output = tmp_path / "second-local-model.json"
    page.fields["output"].setText(str(second_output))
    page.run("train-preference")
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
