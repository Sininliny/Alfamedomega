import json
import sys
import time

import pytest

from alfacc import jobs
from alfacc.cli import main
from alfacc.training import TrainSpec, colab_notebook, create_training_job


def test_training_job_trains_exports_and_installs(ctx):
    spec = TrainSpec(tier="lite", steps=5, version="0.9.0")
    job = create_training_job(ctx.jobs, spec, ctx.python, ctx.script, ctx.runs_dir, ctx.models_dir)
    done = ctx.jobs.run(job["id"])
    assert done["status"] == jobs.SUCCEEDED, ctx.jobs.read_log(job["id"])
    assert done["result"]["installed"] == "lite-0.9.0"
    assert done["artifacts"][0].endswith("lite_pretrain.pt")
    assert done["progress"] is None or done["progress"] == [5, 5]
    assert [m["folder"] for m in ctx.store.models()] == ["lite-0.9.0"]
    log = ctx.jobs.read_log(job["id"])
    assert "step 5/5" in log and "installed lite-0.9.0" in log


def test_failed_step_stops_the_job(ctx, monkeypatch):
    monkeypatch.setenv("FAKE_TRAIN_FAIL", "1")
    job = create_training_job(ctx.jobs, TrainSpec(steps=3), ctx.python, ctx.script, ctx.runs_dir,
                              ctx.models_dir)
    done = ctx.jobs.run(job["id"])
    assert done["status"] == jobs.FAILED and "exited with code 3" in done["error"]
    assert done["current_step"] == 0 and ctx.store.models() == []


def test_background_queue_and_cancel(ctx, tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text("import time\nfor i in range(100):\n    print(f'progress {i}/100', flush=True)\n"
                    "    time.sleep(0.05)\n")
    a = ctx.jobs.create("test", "slow", [jobs.cmd_step([sys.executable, slow])])
    b = ctx.jobs.create("test", "queued", [jobs.cmd_step([sys.executable, "-c", "print(1)"])])
    ctx.jobs.enqueue(a["id"])
    ctx.jobs.enqueue(b["id"])
    deadline = time.time() + 10
    while ctx.jobs.get(a["id"])["status"] != jobs.RUNNING and time.time() < deadline:
        time.sleep(0.05)
    ctx.jobs.cancel(b["id"])
    time.sleep(0.4)
    ctx.jobs.cancel(a["id"])
    assert ctx.jobs.wait(10)
    assert ctx.jobs.get(a["id"])["status"] == jobs.CANCELLED
    assert ctx.jobs.get(b["id"])["status"] == jobs.CANCELLED
    assert ctx.jobs.get(a["id"])["progress"][0] > 0


def test_stale_running_job_is_marked_interrupted(ctx):
    job = ctx.jobs.create("test", "orphan", [])
    job.update(status=jobs.RUNNING, heartbeat=time.time() - 3600)
    ctx.jobs._save(job)
    fresh = ctx.jobs.create("test", "someone else's live job", [])
    fresh.update(status=jobs.RUNNING, heartbeat=time.time())
    ctx.jobs._save(fresh)
    manager = jobs.JobManager(ctx.home)
    assert manager.get(job["id"])["status"] == jobs.INTERRUPTED
    assert manager.get(fresh["id"])["status"] == jobs.RUNNING


@pytest.mark.parametrize("kwargs,msg", [
    ({"stage": "distill"}, "teacher"), ({"stage": "finetune"}, "labelled notes"),
    ({"tier": "huge"}, "unknown tier"), ({"export": False, "install": True}, "export step"),
])
def test_spec_validation(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        TrainSpec(**kwargs).validate()


def test_colab_notebook_embeds_script_and_commands():
    nb = colab_notebook(TrainSpec(tier="studio", steps=1000), "print('hi')\n")
    src = ["".join(c["source"]) for c in nb["cells"]]
    assert any(s.startswith("%%writefile alfamedomega_train.py\nprint('hi')") for s in src)
    assert any("train --tier studio --stage pretrain --steps 1000" in s for s in src)
    assert any("export --ckpt /content/drive/MyDrive/alfamedomega/runs/studio-pretrain/studio_pretrain.pt"
               " --tier studio --out /content/drive/MyDrive/alfamedomega/runs/studio-pretrain/export"
               " --version 0.1.0" in s for s in src)
    json.dumps(nb)


def test_cli_end_to_end(ctx, tmp_path, capsys, fake_export):
    home, models = str(ctx.home), str(ctx.models_dir)
    base = ["--home", home, "--models-dir", models]
    assert main(base + ["settings", "--set", f"training_script={ctx.script}",
                        "--set", f"training_python={sys.executable}"]) == 0
    assert main(base + ["train", "--tier", "standard", "--steps", "3", "--quiet"]) == 0
    assert main(base + ["models", "import", str(fake_export / "lite-0.1.0"), "--version", "1.0.0"]) == 0
    assert main(base + ["models", "verify"]) == 0
    assert main(base + ["models", "export", "lite-1.0.0", str(tmp_path)]) == 0
    assert main(base + ["models", "remove", "lite-1.0.0"]) == 0
    assert main(base + ["models", "import", str(tmp_path / "alfamedomega-lite-1.0.0.alfamodel")]) == 0
    assert main(base + ["train", "--tier", "pro", "--colab", str(tmp_path)]) == 0
    assert (tmp_path / "alfamedomega-pro-pretrain.ipynb").exists()
    capsys.readouterr()
    assert main(base + ["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "standard-0.1.0" in out and "lite-1.0.0" in out
    assert main(base + ["models", "remove", "nope-1.0.0"]) == 1
