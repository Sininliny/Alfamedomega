import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)


@pytest.fixture(scope="module")
def qapp():
    from alfacc.gui.theme import stylesheet
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
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
