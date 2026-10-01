"""AOM queue integration without opening the legacy Models store or a DAW."""
import json
import os
from pathlib import Path
import sys
import time

import pytest
from alfacc.aom import create_tensor_job
from alfacc.jobs import JobManager, cmd_step, SUCCEEDED, CANCELLED
from alfacc.cli import build_parser


REPOSITORY = Path(__file__).resolve().parents[3]


def test_aom_jobs_use_selected_interpreter_and_private_checkpoint(tmp_path):
    manager = JobManager(tmp_path)
    model, manifest = tmp_path / "model.json", tmp_path / "manifest.json"
    job = create_tensor_job(manager, REPOSITORY, "train-preference", [manifest, model], python="chosen-python", resume=True)
    assert job["steps"][0]["argv"][0] == "chosen-python"
    assert job["params"]["category"] == "user_preference"
    assert job["params"]["checkpoint"] != str(model)
    assert "--resume" in job["steps"][0]["argv"]
    assert Path(job["steps"][0]["cancel_file"]).is_absolute()
    assert len(job["steps"]) == 1  # no legacy install/export step
    with pytest.raises(ValueError, match="separate"):
        create_tensor_job(manager, REPOSITORY, "train", [manifest, model], checkpoint=model)
    parser = build_parser()
    args = parser.parse_args(["aom", "--repository", str(REPOSITORY), "train-importance", str(manifest), str(model), "--resume", "--python", "torch-python"])
    assert args.resume and args.python == "torch-python"


def test_cancelling_unrelated_job_does_not_terminate_current_process(tmp_path):
    manager = JobManager(tmp_path)
    running = manager.create("test", "running", [cmd_step([sys.executable, "-u", "-c",
        "import time; print('progress 1/1', flush=True); time.sleep(.7)"])])
    unrelated = manager.create("test", "not enqueued", [])
    manager.enqueue(running["id"])
    deadline = time.monotonic() + 10
    while not manager.read_log(running["id"]) and time.monotonic() < deadline:
        time.sleep(.02)
    manager.cancel(unrelated["id"])
    assert manager.wait(10)
    assert manager.get(running["id"])["status"] == SUCCEEDED
    assert manager.run(unrelated["id"])["status"] == CANCELLED


def test_external_cancel_bounds_a_hung_process(tmp_path):
    manager = JobManager(tmp_path)
    flag = tmp_path / "cancel.flag"
    step = cmd_step([sys.executable, "-u", "-c", "import time; print('progress 1/2',flush=True); time.sleep(60)"])
    step["cancel_file"] = str(flag)
    job = manager.create("aom", "hung native-call fixture", [step])
    manager.enqueue(job["id"])
    deadline = time.monotonic() + 10
    while "progress 1/2" not in manager.read_log(job["id"]) and time.monotonic() < deadline:
        time.sleep(.02)
    assert "progress 1/2" in manager.read_log(job["id"])
    JobManager(tmp_path).cancel(job["id"])
    assert manager.wait(18)
    assert manager.get(job["id"])["status"] == CANCELLED


def test_log_tail_reads_unicode_across_chunk_boundaries(tmp_path):
    manager = JobManager(tmp_path)
    job = manager.create("test", "large log", [])
    lines = [f"记录 {i}: 音色与时间保持" for i in range(6000)]
    manager.log_path(job["id"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert manager.read_log(job["id"], tail=3) == "\n".join(lines[-3:])
    assert manager.read_log(job["id"], tail=0) == ""


def test_real_training_job_cancel_from_another_manager_and_resume(tmp_path):
    python = os.environ.get("AOM_TRAINING_TEST_PYTHON")
    if not python:
        pytest.skip("Set AOM_TRAINING_TEST_PYTHON to an installed PyTorch interpreter")
    manager = JobManager(tmp_path / "home")
    manifest, model = tmp_path / "manifest.json", tmp_path / "model.json"
    data = {"schema": "aom.training-manifest/1", "category": "user_preference", "feature_widths": {"state": 1},
            "hidden_width": 4, "steps": 100000, "learning_rate": .01, "available_ram_bytes": 1_000_000_000,
            "seed": 2, "require_validation": True, "observations": [
            {"group_id": "train", "first": {"state": [.8]}, "second": {"state": [.2]}, "preferred_first": True},
            {"group_id": "held", "split": "validation", "first": {"state": [.7]}, "second": {"state": [.3]}, "preferred_first": True}]}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    job = create_tensor_job(manager, REPOSITORY, "train-preference", [manifest, model], python=python, checkpoint_every=2)
    manager.enqueue(job["id"])
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        record = manager.get(job["id"])
        if record.get("progress") or record["status"] in ("failed", "succeeded"):
            break
        time.sleep(.03)
    assert manager.get(job["id"]).get("progress"), manager.read_log(job["id"])
    JobManager(tmp_path / "home").cancel(job["id"])
    assert manager.wait(25)
    assert manager.get(job["id"])["status"] == CANCELLED, manager.read_log(job["id"])
    assert not model.exists() and Path(job["params"]["checkpoint"]).is_file()
    # Finish three more optimizer steps after the exact saved boundary.
    import subprocess
    result = subprocess.run([python, "-c", "import torch,sys; print(torch.load(sys.argv[1],weights_only=True)['step'])",
                             job["params"]["checkpoint"]], capture_output=True, text=True, check=True)
    data["steps"] = int(result.stdout.strip()) + 3
    manifest.write_text(json.dumps(data), encoding="utf-8")
    resumed = create_tensor_job(manager, REPOSITORY, "train-preference", [manifest, model], python=python, resume=True)
    done = manager.run(resumed["id"])
    assert done["status"] == SUCCEEDED, manager.read_log(resumed["id"])
    assert done["progress"] == [data["steps"], data["steps"]]
    assert str(model) in done["artifacts"]
    assert json.loads(model.read_text())["category"] == "user_preference"
