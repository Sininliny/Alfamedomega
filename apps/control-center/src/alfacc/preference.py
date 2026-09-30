"""Batch Adaptive Preference Learning: the user's libraries, their index, and the learner run.

The control center's part (docs/control-center/preference-batch.md):

1. Keep the list of sample and preset folders the user has chosen to learn from.
2. Scan them into a local index: one record per audio file with its format, length and the
   weak text labels taken from its file and folder names ("Pads/warm_pad_03.wav" gives
   folder label "pads" and name labels "warm", "pad"). Rescans only reread changed files.
3. Hand the index and the plugin's behaviour log to the preference learner from preference/,
   which computes embeddings, the preference model and the text-to-timbre map.

Everything stays on this machine. The index holds paths and labels, never audio, and
forget() deletes all of it.
"""

import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import jobs
from .settings import find_in_checkout, split_command, write_json_atomic

AUDIO_SUFFIXES = {".wav", ".wave", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}

# Words in sample names that say nothing about timbre.
STOPWORDS = {
    "wav", "wave", "aif", "aiff", "flac", "mp3", "ogg", "sample", "samples", "sound", "sounds",
    "one", "shot", "oneshot", "loop", "loops", "the", "and", "of", "for", "with", "new", "copy",
    "final", "edit", "mix", "master", "version", "take", "stereo", "mono", "st", "bpm", "hz",
    "khz", "bit", "untitled", "audio", "export", "bounce", "rendered", "render",
}
_NOTE_RE = re.compile(r"^[a-g](?:#|b|s|sharp|flat)?-?\d?$")  # "F#1", "Bb", and "G#" of "G#-1"
_VERSION_RE = re.compile(r"^v\d+$|^\d+(?:bpm|k|hz|khz|bit)?$")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_RE = re.compile(r"[^0-9A-Za-z#]+")


# The kinds of behaviour the user can let the learner use (preference design, section 7.1).
CATEGORIES = {
    "performance": "Notes you play: pitch range, note lengths, velocity and expression",
    "parameters": "Knob and parameter movements after a sound is loaded",
    "suggestions": "Which generated or suggested sounds you keep, like or reject",
    "workflow": "How you use sounds: loads, play time, presets saved, sounds kept and reused",
}


def _shared_tokenizer(repo: str = ""):
    """preference/alfapref/tokens.py, so labels shown here match what the learner computes.

    Loaded by file path: the alfapref package itself pulls in numpy, which the control center
    doesn't need. Falls back to the copy below when the app runs outside a checkout.
    """
    path = find_in_checkout("preference", "alfapref", "tokens.py", repo=repo)
    if not path:
        return None
    try:
        spec = importlib.util.spec_from_file_location("_alfapref_tokens", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.words
    except Exception:  # noqa: BLE001 - a broken sibling checkout must not stop the app
        return None


def use_repo(repo: str) -> None:
    """Switch to the tokenizer in the repository folder chosen in Settings."""
    global _SHARED_WORDS
    _SHARED_WORDS = _shared_tokenizer(repo) or _SHARED_WORDS


def tokens(text: str) -> list:
    """Split a file or folder name into lower-case timbre words."""
    if _SHARED_WORDS is not None:
        return list(dict.fromkeys(_SHARED_WORDS(text)))
    return fallback_tokens(text)


def fallback_tokens(text: str) -> list:
    """Local copy of the rules in preference/alfapref/tokens.py, for standalone installs."""
    out = []
    for part in _SPLIT_RE.split(text):
        for word in _CAMEL_RE.split(part):
            w = word.lower()
            if len(w) < 2 or w in STOPWORDS or _NOTE_RE.match(w) or _VERSION_RE.match(w):
                continue
            if w.isdigit():
                continue
            if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
                w = w[:-1]  # "Pads" folder and "pad" file name are one label
            out.append(w)
    return out


_SHARED_WORDS = _shared_tokenizer()


def audio_info(path: Path) -> dict:
    """Sample rate, channels and length for WAV files (RIFF/RF64 headers), without decoding."""
    info = {}
    if path.suffix.lower() not in (".wav", ".wave"):
        return info
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
                return info
            fmt = None
            data_size = None
            while True:
                ch = f.read(8)
                if len(ch) < 8:
                    break
                cid, size = ch[:4], struct.unpack("<I", ch[4:])[0]
                if cid == b"fmt ":
                    body = f.read(size)
                    fmt = struct.unpack("<HHIIHH", body[:16])
                elif cid == b"data":
                    data_size = size
                    break
                else:
                    f.seek(size + (size & 1), os.SEEK_CUR)
            if fmt:
                _tag, ch_n, sr, byte_rate, _align, bits = fmt
                info.update(sr=sr, channels=ch_n, bits=bits)
                if data_size is not None and byte_rate and data_size != 0xFFFFFFFF:
                    info["duration_s"] = round(data_size / byte_rate, 3)
    except (OSError, struct.error):
        pass
    return info


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PreferenceStore:
    def __init__(self, home: Path):
        self.dir = Path(home) / "preference"
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.dir / "library_index.jsonl"

    @property
    def vocab_path(self) -> Path:
        return self.dir / "vocabulary.json"

    @property
    def profiles_dir(self) -> Path:
        return self.dir / "profiles"

    @property
    def events_dir(self) -> Path:
        return self.dir / "events"

    @property
    def consent_path(self) -> Path:
        """The learning switches, shared with the plugin (preference design, section 7.1)."""
        return self.dir / "plugin.json"

    @property
    def annotations_path(self) -> Path:
        return self.dir / "annotations.json"

    @property
    def labels_path(self) -> Path:
        return self.dir / "labels.jsonl"

    # ------------------------------------------------------------------ consent

    def consent(self) -> dict:
        """The switches as the learner reads them: anything not explicitly true is off."""
        try:
            raw = json.loads(self.consent_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        raw = raw if isinstance(raw, dict) else {}
        learn = raw.get("learn") is True
        cats = raw.get("categories")
        if isinstance(cats, dict):
            cats = {c: learn and cats.get(c) is True for c in CATEGORIES}
        else:
            cats = {c: learn for c in CATEGORIES}
        try:
            pers = min(1.0, max(0.0, float(raw.get("personalization", 1.0))))
        except (TypeError, ValueError):
            pers = 1.0
        return {"learn": learn, "categories": cats, "personalization": pers}

    def set_consent(self, learn: bool | None = None, categories: dict | None = None,
                    personalization: float | None = None) -> dict:
        """Change some switches. Keys the plugin or learner added are kept."""
        try:
            raw = json.loads(self.consent_path.read_text(encoding="utf-8"))
            raw = raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            raw = {}
        now = self.consent()
        raw["learn"] = now["learn"] if learn is None else bool(learn)
        cats = dict(now["categories"]) if now["learn"] else {c: True for c in CATEGORIES}
        for c, on in (categories or {}).items():
            if c not in CATEGORIES:
                raise ValueError(f"unknown category {c!r}; known: {', '.join(CATEGORIES)}")
            cats[c] = bool(on)
        raw["categories"] = cats
        if personalization is not None:
            raw["personalization"] = round(min(1.0, max(0.0, float(personalization))), 3)
        raw.setdefault("personalization", now["personalization"])
        write_json_atomic(self.consent_path, raw)
        return self.consent()

    # ------------------------------------------------------------------ the user's own words

    def annotations(self) -> dict:
        """{path: {"tags": [...], "description": str, "labels": [...]}} the user wrote."""
        try:
            return json.loads(self.annotations_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def annotate(self, path: Path, tags=None, description: str | None = None, labels=None) -> dict:
        """Tag, describe or hand-label one file. Empty values remove that field."""
        notes = self.annotations()
        key = str(Path(path).expanduser().resolve())
        entry = dict(notes.get(key, {}))
        for field_, value in (("tags", tags), ("labels", labels)):
            if value is not None:
                words = [w.strip() for w in value if w and w.strip()]
                entry[field_] = list(dict.fromkeys(words))
        if description is not None:
            entry["description"] = description.strip()
        entry = {k: v for k, v in entry.items() if v}
        if entry:
            notes[key] = entry
        else:
            notes.pop(key, None)
        write_json_atomic(self.annotations_path, notes)
        self._write_labels(notes)
        return entry

    def _write_labels(self, notes: dict) -> None:
        """labels.jsonl for the learner's --labels: one {"path", "labels"} per hand-labelled file."""
        with open(self.labels_path, "w", encoding="utf-8") as f:
            for path, entry in sorted(notes.items()):
                if entry.get("labels"):
                    f.write(json.dumps({"path": path, "labels": entry["labels"]}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ what was learned

    def latest_profile(self) -> Path | None:
        names = self.profiles()
        return self.profiles_dir / names[0] if names else None

    def explain(self) -> list:
        """Plain-language lines about what the latest profile learned, from its summary.json."""
        prof = self.latest_profile()
        try:
            summary = json.loads((prof / "summary.json").read_text(encoding="utf-8")) if prof else None
        except (OSError, ValueError):
            summary = None
        if not summary:
            return []
        if summary.get("explain"):
            return [str(line) for line in summary["explain"]]
        # Learners before the consent change only wrote counts.
        files = summary.get("files", {})
        lines = []
        if isinstance(files, dict):
            lines += [f"{v} {k.replace('_', ' ')}" for k, v in files.items() if isinstance(v, (int, float))]
        if summary.get("vocabulary"):
            lines.append("Words: " + ", ".join(list(summary["vocabulary"])[:20]))
        return lines

    # ------------------------------------------------------------------ libraries

    def libraries(self) -> list:
        p = self.dir / "libraries.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def _save_libraries(self, libs: list) -> None:
        write_json_atomic(self.dir / "libraries.json", libs)

    def add_library(self, path: Path, label: str = "") -> dict:
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"{path} is not a folder")
        libs = self.libraries()
        for lib in libs:
            if Path(lib["path"]) == path:
                return lib
        lib = {"path": str(path), "label": label or path.name, "enabled": True, "added": _now()}
        self._save_libraries(libs + [lib])
        return lib

    def remove_library(self, path: Path) -> None:
        path = str(Path(path).expanduser().resolve())
        self._save_libraries([lib for lib in self.libraries() if lib["path"] != path])

    def set_enabled(self, path: Path, enabled: bool) -> None:
        path = str(Path(path).expanduser().resolve())
        libs = self.libraries()
        for lib in libs:
            if lib["path"] == path:
                lib["enabled"] = enabled
        self._save_libraries(libs)

    # ------------------------------------------------------------------ index

    def load_index(self) -> dict:
        out = {}
        if self.index_path.exists():
            with open(self.index_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        out[rec["path"]] = rec
        return out

    def scan(self, log=print) -> dict:
        """Index every enabled library. Unchanged files are carried over without rereading."""
        old = self.load_index()
        notes = self.annotations()
        libs = [lib for lib in self.libraries() if lib.get("enabled", True)]
        records, reused, reread = [], 0, 0
        files = []
        for lib in libs:
            root = Path(lib["path"])
            if not root.is_dir():
                log(f"skipping {root}: folder not found")
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                for name in sorted(filenames):
                    if Path(name).suffix.lower() in AUDIO_SUFFIXES and not name.startswith("."):
                        files.append((lib, root, Path(dirpath) / name))
        total = len(files)
        for i, (lib, root, p) in enumerate(files, 1):
            try:
                st = p.stat()
            except OSError:
                continue
            key = str(p)
            prev = old.get(key)
            if prev and prev["bytes"] == st.st_size and prev["mtime"] == int(st.st_mtime):
                rec = dict(prev)
                reused += 1
            else:
                rel = p.relative_to(root)
                folders = [t for part in rel.parts[:-1] for t in tokens(part)]
                rec = {"path": key, "library": lib["label"], "rel": rel.as_posix(),
                       "format": p.suffix.lower().lstrip("."), "bytes": st.st_size,
                       "mtime": int(st.st_mtime), "name_labels": tokens(p.stem),
                       "folder_labels": list(dict.fromkeys(folders)), **audio_info(p)}
                reread += 1
            for field_ in ("tags", "description", "labels"):
                rec.pop(field_, None)
            rec.update(notes.get(str(p.resolve()), {}))
            records.append(rec)
            if i % 200 == 0 or i == total:
                log(f"progress {i}/{total}")

        tmp = self.index_path.with_name(self.index_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, self.index_path)

        vocab = Counter()
        for rec in records:
            vocab.update(set(rec["name_labels"]) | set(rec["folder_labels"]))
        summary = {"files": len(records), "reused": reused, "reread": reread,
                   "removed": len(set(old) - {r["path"] for r in records}),
                   "libraries": len(libs), "labels": len(vocab),
                   "labelled_files": sum(1 for r in records if r["name_labels"] or r["folder_labels"]),
                   "scanned_at": _now()}
        write_json_atomic(self.vocab_path, {"summary": summary, "labels": dict(vocab.most_common())})
        log(f"indexed {summary['files']} files ({reread} read, {reused} unchanged, "
            f"{summary['removed']} gone), {summary['labels']} distinct labels")
        return summary

    def vocabulary(self) -> dict:
        if not self.vocab_path.exists():
            return {"summary": None, "labels": {}}
        return json.loads(self.vocab_path.read_text(encoding="utf-8"))

    def profiles(self) -> list:
        if not self.profiles_dir.exists():
            return []
        return sorted((p.name for p in self.profiles_dir.iterdir() if p.is_dir()), reverse=True)

    def forget(self) -> None:
        """Delete the index, vocabulary, behaviour log and learned profiles. Libraries stay listed."""
        for p in (self.index_path, self.vocab_path):
            p.unlink(missing_ok=True)
        for d in (self.profiles_dir, self.events_dir):
            shutil.rmtree(d, ignore_errors=True)


def learner_argv(learner: str, store: PreferenceStore, out: Path, assimilation: Path | None = None,
                 extra: list | None = None) -> list:
    """The call to the preference learner (contract in docs/control-center/preference-batch.md)."""
    argv = [*split_command(learner), "batch",
            "--index", str(store.index_path), "--events", str(store.events_dir)]
    if assimilation is not None:
        argv += ["--assimilation", str(assimilation)]
    return argv + list(extra or []) + ["--out", str(out)]


def learner_supports(learner: str, text: str) -> bool:
    """Whether the learner's source mentions a flag or command.

    Older learners reject unknown flags, so optional ones are passed only when learner.py or the
    alfapref/cli.py it launches mentions them.
    """
    scripts = [t for t in split_command(learner) if t.endswith(".py")]
    if not scripts:
        return False
    script = Path(scripts[-1])
    for source in (script, script.parent / "alfapref" / "cli.py"):
        try:
            if text in source.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


def assimilation_for(home: Path, learner: str) -> Path | None:
    """<home>/assimilation when it holds analysed plugin notes and the learner reads them.

    Older learners reject unknown flags, so the flag is passed only when the learner script
    or its alfapref/cli.py mentions it (docs/control-center/plugin-assimilation.md, section 10).
    """
    folder = Path(home) / "assimilation"
    if not any(folder.glob("*/features.jsonl")):
        return None
    return folder if learner_supports(learner, "--assimilation") else None


def create_batch_job(manager: jobs.JobManager, home: Path, learner: str) -> dict:
    """Scan the libraries, then run the preference learner on the index, as one job."""
    store = PreferenceStore(home)
    assimilation = assimilation_for(home, learner) if learner else None
    if not any(lib.get("enabled", True) for lib in store.libraries()) and assimilation is None:
        raise ValueError("add at least one library folder first")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = store.profiles_dir / stamp
    steps = [jobs.action_step("preference_scan", label="Index libraries", home=str(home))]
    if learner:
        extra = []
        if learner_supports(learner, "--consent"):
            extra += ["--consent", str(store.consent_path)]
        if learner_supports(learner, "--labels"):
            store._write_labels(store.annotations())
            if store.labels_path.stat().st_size:
                extra += ["--labels", str(store.labels_path)]
        steps.append(jobs.cmd_step(learner_argv(learner, store, out, assimilation, extra),
                                   label="Learn preferences"))
    else:
        steps.append(jobs.action_step("preference_no_learner", label="Learn preferences"))
    return manager.create("preference", "Adaptive Preference Learning batch", steps,
                          params={"profile": str(out), "learner": learner})


@jobs.action("preference_scan")
def _scan(args: dict, log) -> dict:
    return {"scan": PreferenceStore(Path(args["home"])).scan(log)}


@jobs.action("preference_no_learner")
def _no_learner(args: dict, log) -> dict:
    log("The preference learner wasn't found, so this batch stopped after indexing. Set the "
        "repository folder in Settings (it uses preference/learner.py there), or set the learner "
        "command directly.")
    return {"learner": "not configured"}


def reset_learning(store: PreferenceStore, learner: str, everything: bool = False) -> str:
    """Forget what was learned from behaviour: the learned model and the plugin's event log.

    With everything, also drop every example (the learner's reset --all). Libraries, the index
    and the user's own tags and labels stay. Returns what was done, for the user.
    """
    prof = store.latest_profile()
    done = []
    if prof and learner and learner_supports(learner, '"reset"'):
        argv = [*split_command(learner), "reset", "--profile", str(prof)] + (["--all"] if everything else [])
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError((r.stderr.strip().splitlines() or ["the learner's reset failed"])[-1])
        done.append("reset the learned model")
        others = [p for p in store.profiles_dir.iterdir() if p.is_dir() and p != prof]
    else:
        others = [p for p in store.profiles_dir.iterdir() if p.is_dir()] if store.profiles_dir.exists() else []
    for p in others:
        shutil.rmtree(p, ignore_errors=True)
    if others:
        done.append(f"deleted {len(others)} older profile{'s' if len(others) > 1 else ''}")
    if store.events_dir.exists():
        shutil.rmtree(store.events_dir, ignore_errors=True)
        done.append("deleted the behaviour log")
    return ", ".join(done).capitalize() + "." if done else "Nothing was learned yet."


def run_learner_check(learner: str) -> str | None:
    """Return None if the learner command starts, else the reason it doesn't."""
    try:
        r = subprocess.run([*split_command(learner), "--help"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        return str(e)
    return None if r.returncode == 0 else (r.stderr.strip().splitlines() or ["failed"])[-1]
