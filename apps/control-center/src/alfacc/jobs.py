"""Long-running jobs (training, export, preference batches) with logs, progress and a queue.

A job is a list of steps. A step is either an external command (a training run in the user's
PyTorch interpreter) or a named action in this process (install an export into the Models
folder). Each job keeps its state in <home>/jobs/<id>/job.json and its output in log.txt, so the
GUI, the CLI and a later session all see the same history.

Jobs run one at a time: training saturates the machine, and running two would only slow both.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .settings import read_json, write_json_atomic

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED, INTERRUPTED = (
    "queued", "running", "succeeded", "failed", "cancelled", "interrupted")
FINISHED = {SUCCEEDED, FAILED, CANCELLED, INTERRUPTED}
HEARTBEAT_S = 5      # how often a live job's record is touched
STALE_S = 60         # a queued or running job untouched this long belongs to an exited session
SAVE_EVERY_S = 1.0   # progress is written at most this often

# "step 120/2000 ..." from training/alfamedomega_train.py, "progress 3/40" from anything else.
PROGRESS_RE = re.compile(r"^(?:step|progress) (\d+)/(\d+)")
SAVED_RE = re.compile(r"^saved (.+)$")

_ACTIONS = {}


def action(name):
    """Register an in-process step: fn(args: dict, log: callable) -> dict | None."""
    def deco(fn):
        _ACTIONS[name] = fn
        return fn
    return deco


def cmd_step(argv, cwd=None, label=None, env=None) -> dict:
    return {"type": "cmd", "argv": [str(a) for a in argv], "cwd": str(cwd) if cwd else None,
            "label": label or Path(str(argv[0])).name, "env": env or {}}


def action_step(name, label=None, **args) -> dict:
    return {"type": "action", "name": name, "args": args, "label": label or name}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class JobCancelled(Exception):
    pass


class StepTimeout(RuntimeError):
    pass


_current = threading.local()


def spawn(argv, log, cwd=None, timeout: float | None = None, label: str | None = None) -> None:
    """Run a command from inside a job action, with the job's log, cancel and an optional timeout.

    Outside a job (tests, the CLI calling an action directly) it just runs the command.
    """
    step = cmd_step(argv, cwd=cwd, label=label)
    ctx = getattr(_current, "job", None)
    if ctx is None:
        JobManager._run_standalone(step, log, timeout)
    else:
        manager, job_id = ctx
        manager._run_cmd(job_id, step, log, timeout)


class JobManager:
    def __init__(self, home: Path):
        self.dir = Path(home) / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue = []
        self._worker = None
        self._proc = None
        self._cancel = set()
        self._save_lock = threading.Lock()
        self._live = {}          # job id -> in-memory record, for jobs this manager owns
        self._beat = None
        self._reap_stale()

    # ------------------------------------------------------------------ records

    def _path(self, job_id: str) -> Path:
        return self.dir / job_id / "job.json"

    def log_path(self, job_id: str) -> Path:
        return self.dir / job_id / "log.txt"

    def get(self, job_id: str) -> dict:
        return read_json(self._path(job_id))

    def _save(self, job: dict) -> None:
        with self._save_lock:
            write_json_atomic(self._path(job["id"]), job)

    def _own(self, job: dict) -> None:
        """Keep a queued or running job's heartbeat fresh so other sessions leave it alone."""
        with self._lock:
            self._live[job["id"]] = job
            if self._beat is None or not self._beat.is_alive():
                self._beat = threading.Thread(target=self._heartbeat, daemon=True)
                self._beat.start()

    def _release(self, job_id: str) -> None:
        with self._lock:
            self._live.pop(job_id, None)

    def _heartbeat(self) -> None:
        while True:
            with self._lock:
                live = list(self._live.values())
            if not live:
                return
            for job in live:
                job["heartbeat"] = time.time()
                self._save(job)
            time.sleep(HEARTBEAT_S)

    def list(self) -> list:
        jobs = []
        for p in self.dir.glob("*/job.json"):
            try:
                jobs.append(read_json(p))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def read_log(self, job_id: str, tail: int | None = None) -> str:
        path = self.log_path(job_id)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if tail:
            text = "\n".join(text.splitlines()[-tail:])
        return text

    def _reap_stale(self) -> None:
        """Close out jobs left behind by a session that has since exited."""
        now = time.time()
        for job in self.list():
            if job["status"] in (RUNNING, QUEUED) and now - job.get("heartbeat", 0) > STALE_S:
                was_running = job["status"] == RUNNING
                job["status"] = INTERRUPTED if was_running else CANCELLED
                job["finished"] = _now()
                job["error"] = ("the control center exited while this job was running" if was_running
                                else "the control center exited before this job started")
                self._save(job)

    # ------------------------------------------------------------------ create / run

    def create(self, kind: str, title: str, steps: list, params: dict | None = None) -> dict:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"{stamp}-{kind}-{uuid.uuid4().hex[:6]}"
        job = {"id": job_id, "kind": kind, "title": title, "params": params or {},
               "steps": steps, "status": QUEUED, "created": _now(), "started": None,
               "finished": None, "current_step": 0, "progress": None, "artifacts": [],
               "result": {}, "error": None, "heartbeat": time.time()}
        (self.dir / job_id).mkdir(parents=True)
        self._save(job)
        return job

    def enqueue(self, job_id: str) -> None:
        """Run in the background after any jobs already queued."""
        self._own(self.get(job_id))
        with self._lock:
            self._queue.append(job_id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drain, daemon=True)
                self._worker.start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    return
                job_id = self._queue.pop(0)
            self.run(job_id)

    def wait(self, timeout: float | None = None) -> bool:
        worker = self._worker
        if worker is not None:
            worker.join(timeout)
            return not worker.is_alive()
        return True

    def cancel(self, job_id: str) -> None:
        with self._lock:
            queued = job_id in self._queue
            if queued:
                self._queue.remove(job_id)
                job = self._live.pop(job_id, None) or self.get(job_id)
        if queued:
            job["status"], job["finished"] = CANCELLED, _now()
            self._save(job)
            return
        with self._lock:
            self._cancel.add(job_id)
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def run(self, job_id: str, echo=None) -> dict:
        """Run a job in the calling thread and return its final record. echo(line) mirrors the log."""
        with self._lock:
            job = self._live.get(job_id) or self.get(job_id)
        if job["status"] in FINISHED:
            self._release(job_id)
            return job
        job.update(status=RUNNING, started=_now(), heartbeat=time.time())
        self._save(job)
        self._own(job)
        last_save = [0.0]
        log_file = open(self.log_path(job_id), "a", encoding="utf-8", buffering=1)

        def log(line: str) -> None:
            log_file.write(line.rstrip("\n") + "\n")
            if echo:
                echo(line.rstrip("\n"))
            m = PROGRESS_RE.match(line)
            if m:
                job["progress"] = [int(m.group(1)), int(m.group(2))]
            m = SAVED_RE.match(line)
            if m:
                with self._save_lock:
                    job["artifacts"].append(m.group(1).strip())
            if time.time() - last_save[0] > SAVE_EVERY_S:
                last_save[0] = time.time()
                self._save(job)

        _current.job = (self, job_id)
        try:
            for i, step in enumerate(job["steps"]):
                if job_id in self._cancel:
                    raise JobCancelled()
                job["current_step"] = i
                job["progress"] = None
                self._save(job)
                log(f"== {step['label']}")
                if step["type"] == "cmd":
                    self._run_cmd(job_id, step, log)
                else:
                    fn = _ACTIONS.get(step["name"])
                    if fn is None:
                        raise RuntimeError(f"unknown action {step['name']!r}")
                    out = fn(step["args"], log)
                    if out:
                        with self._save_lock:
                            job["result"].update(out)
            job["status"] = SUCCEEDED
        except JobCancelled:
            job["status"] = CANCELLED
            log("== cancelled")
        except Exception as e:  # noqa: BLE001 - a job failure is reported, not raised
            job["status"] = FAILED
            job["error"] = str(e) or type(e).__name__
            log(traceback.format_exc())
        finally:
            _current.job = None
            self._release(job_id)
            job["finished"] = _now()
            job["heartbeat"] = time.time()
            self._save(job)
            log_file.close()
            self._cancel.discard(job_id)
        return job

    @staticmethod
    def _popen(step: dict):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **step.get("env", {})}
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        return subprocess.Popen(step["argv"], cwd=step.get("cwd"), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                creationflags=flags)

    @staticmethod
    def _pump(proc, step: dict, log, timeout: float | None) -> None:
        timed_out = threading.Event()
        timer = None
        if timeout:
            def kill():
                timed_out.set()
                proc.kill()
            timer = threading.Timer(timeout, kill)
            timer.daemon = True
            timer.start()
        try:
            for line in proc.stdout:
                log(line)
            rc = proc.wait()
        finally:
            if timer:
                timer.cancel()
        if timed_out.is_set():
            raise StepTimeout(f"{step['label']} did not finish within {timeout:.0f} s and was stopped")
        if rc != 0:
            raise RuntimeError(f"{step['label']} exited with code {rc}")

    @classmethod
    def _run_standalone(cls, step: dict, log, timeout: float | None) -> None:
        log("$ " + subprocess.list2cmdline(step["argv"]))
        cls._pump(cls._popen(step), step, log, timeout)

    def _run_cmd(self, job_id: str, step: dict, log, timeout: float | None = None) -> None:
        log("$ " + subprocess.list2cmdline(step["argv"]))
        with self._lock:
            if job_id in self._cancel:
                raise JobCancelled()
            proc = self._popen(step)
            self._proc = proc
        try:
            self._pump(proc, step, log, timeout)
        except RuntimeError:
            if job_id in self._cancel:
                raise JobCancelled() from None
            raise
        finally:
            with self._lock:
                self._proc = None
        if job_id in self._cancel:
            raise JobCancelled()
