"""The shared Models folder: install, verify, export and remove tier weights.

The folder and its manifest follow the inference host's contract (src/inference/MODEL_CONTRACT.md,
summarised in docs/control-center/model-store.md):

    Models/
      manifest.json              # {"schema": 1, "models": [{tier, version, folder, opset,
                                 #   frame_rate_hz, files: {name: sha256}}]}
      standard-1.0.0/            # one folder per tier and model version
        export_info.json         # the training export's description of the graphs
        patch_encoder.onnx ...

The control center adds keys the host ignores: default_tier at the top, and sizes, bytes,
source, installed_at and notes per model.

The plugin reads Models/manifest.json and checks hashes before loading, so every write here
is staged in a hidden folder and published with a rename, and the manifest is replaced
atomically. A reader sees the old state or the new one, never a partial model.
"""

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .settings import write_json_atomic
from .tiers import DEFAULT_TIER, REQUIRED_GRAPHS, TIERS, check_tier

SCHEMA = 1
MANIFEST = "manifest.json"
EXPORT_INFO = "export_info.json"
BUNDLE_META = "alfamodel.json"
BUNDLE_SUFFIX = ".alfamodel"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
FOLDER_RE = re.compile(r"^(lite|standard|pro|studio)-(\d+\.\d+\.\d+)$")
DEFAULT_OPSET = 17
DEFAULT_FRAME_HZ = 250

# Files a training export contains that the plugin needs at run time.
RUNTIME_SUFFIXES = {".onnx", ".npy", ".json"}
# Developer-only outputs left out of an install unless asked for (C++ DSP test vectors).
DEV_FILES = {"dsp_golden.npz"}


class ModelStoreError(Exception):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _version_key(v: str):
    return tuple(int(x) for x in v.split("."))


@dataclass
class VerifyResult:
    model_id: str
    ok: bool
    problems: list = field(default_factory=list)


@dataclass
class ExportFolder:
    """One tier folder from a training export, ready to install."""
    path: Path
    tier: str
    version: str | None      # from the folder name or the export's manifest, if it says
    opset: int
    frame_rate_hz: int
    missing: list


class ModelStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ manifest

    def manifest(self) -> dict:
        path = self.root / MANIFEST
        if not path.exists():
            return {"schema": SCHEMA, "default_tier": DEFAULT_TIER, "models": []}
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema", 0) > SCHEMA:
            raise ModelStoreError(
                f"{path} uses manifest schema {data['schema']}; this control center reads "
                f"schema {SCHEMA}. Update the control center.")
        data.setdefault("models", [])
        data.setdefault("default_tier", DEFAULT_TIER)
        return data

    def _write_manifest(self, data: dict) -> None:
        data["schema"] = SCHEMA
        data["models"].sort(key=lambda m: (list(TIERS).index(m["tier"]), _version_key(m["version"])))
        write_json_atomic(self.root / MANIFEST, data)

    def models(self) -> list:
        return self.manifest()["models"]

    def get(self, model_id: str) -> dict:
        for m in self.models():
            if m["folder"] == model_id:
                return m
        raise ModelStoreError(f"no installed model {model_id!r}")

    def next_version(self, tier: str) -> str:
        versions = [m["version"] for m in self.models() if m["tier"] == tier]
        if not versions:
            return "0.1.0"
        major, minor, patch = max(versions, key=_version_key).split(".")
        return f"{major}.{minor}.{int(patch) + 1}"

    # ------------------------------------------------------------------ inspect

    @staticmethod
    def find_exports(src: Path) -> list:
        """Tier folders in what the user picked: a Models folder written by the training export,
        or one tier folder from it."""
        src = Path(src)
        man = src / MANIFEST
        if man.is_file():
            data = json.loads(man.read_text(encoding="utf-8"))
            if isinstance(data.get("models"), list):  # a Models folder
                out = []
                for m in data["models"]:
                    folder = src / m.get("folder", "")
                    if folder.is_dir() and m.get("folder"):
                        out.append(ModelStore.inspect_folder(folder, m))
                if not out:
                    raise ModelStoreError(f"{man} lists no model folders that exist next to it")
                return out
        if (src / EXPORT_INFO).is_file() or FOLDER_RE.match(src.name):
            return [ModelStore.inspect_folder(src)]
        subs = [p for p in sorted(src.iterdir()) if p.is_dir() and (p / EXPORT_INFO).is_file()]
        if subs:
            return [ModelStore.inspect_folder(p) for p in subs]
        raise ModelStoreError(f"{src} is not a training export (no {MANIFEST} or {EXPORT_INFO})")

    @staticmethod
    def inspect_folder(folder: Path, entry: dict | None = None) -> ExportFolder:
        entry = entry or {}
        info = {}
        if (folder / EXPORT_INFO).is_file():
            info = json.loads((folder / EXPORT_INFO).read_text(encoding="utf-8"))
        m = FOLDER_RE.match(folder.name)
        tier = entry.get("tier") or info.get("tier") or (m.group(1) if m else None)
        if tier not in TIERS:
            raise ModelStoreError(f"can't tell which model size {folder} is")
        version = entry.get("version") or (m.group(2) if m else None)
        graphs = {p.stem for p in folder.glob("*.onnx") if not p.stem.endswith(".int8")}
        return ExportFolder(folder, tier, version, int(entry.get("opset", DEFAULT_OPSET)),
                            int(entry.get("frame_rate_hz", info.get("frame_hz", DEFAULT_FRAME_HZ))),
                            [g for g in REQUIRED_GRAPHS if g not in graphs])

    # ------------------------------------------------------------------ install

    def import_export(self, src: Path, version: str | None = None, notes: str = "",
                      include_dev_files: bool = False, replace: bool = False) -> list:
        """Install every tier folder in a training export. Returns the new manifest entries."""
        exports = self.find_exports(Path(src))
        if version and len(exports) > 1:
            raise ModelStoreError("this export holds several models; import them without a version")
        for e in exports:
            if e.missing:
                raise ModelStoreError(f"{e.path} is missing graphs: {', '.join(e.missing)}")
        entries = []
        for e in exports:
            files = [(p.name, p) for p in sorted(e.path.iterdir()) if p.is_file() and (
                (p.suffix in RUNTIME_SUFFIXES and p.name not in DEV_FILES)
                or (include_dev_files and p.name in DEV_FILES))]
            entries.append(self._install(
                e.tier, version or e.version, files, source=f"training export {Path(src).name}",
                notes=notes, meta={"opset": e.opset, "frame_rate_hz": e.frame_rate_hz}, replace=replace))
        return entries

    def import_any(self, src: Path, version: str | None = None, notes: str = "",
                   replace: bool = False) -> list:
        """Install whatever the user picked: a bundle, an export folder, or a zipped export."""
        src = Path(src)
        if src.suffix == BUNDLE_SUFFIX:
            return [self.import_bundle(src, replace=replace)]
        if src.is_dir():
            return self.import_export(src, version=version, notes=notes, replace=replace)
        if src.suffix == ".zip":
            with tempfile.TemporaryDirectory(prefix="alfa-import-") as tmp:
                staging = Path(tmp)
                with zipfile.ZipFile(src) as zf:
                    for info in zf.infolist():
                        parts = PurePosixPath(info.filename).parts
                        if info.is_dir() or not parts:
                            continue
                        target = staging.joinpath(*[_safe_name(x) for x in parts])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as fin, open(target, "wb") as fout:
                            shutil.copyfileobj(fin, fout, 1 << 20)
                # The zip may hold the export folder itself or wrap it in one more folder.
                roots = [staging] + sorted(p for p in staging.iterdir() if p.is_dir())
                for root in roots:
                    try:
                        self.find_exports(root)
                    except ModelStoreError:
                        continue
                    return self.import_export(root, version=version,
                                              notes=notes or f"from {src.name}", replace=replace)
                raise ModelStoreError(f"{src.name} has no training export inside")
        raise ModelStoreError(f"{src} is not a model bundle, export folder or zipped export")

    def import_bundle(self, bundle: Path, replace: bool = False) -> dict:
        """Install a .alfamodel bundle made by export_bundle, checking every hash."""
        bundle = Path(bundle)
        with zipfile.ZipFile(bundle) as zf:
            try:
                meta = json.loads(zf.read(BUNDLE_META))
            except KeyError:
                raise ModelStoreError(f"{bundle} is not an Alfamedomega model bundle") from None
            if meta.get("schema", 0) > SCHEMA:
                raise ModelStoreError(f"{bundle} was made by a newer control center")
            tier = check_tier(meta["tier"])
            with tempfile.TemporaryDirectory(prefix="alfa-bundle-") as tmp:
                files = []
                for name, sha in meta["files"].items():
                    name = _safe_name(name)
                    target = Path(tmp) / name
                    with zf.open(f"files/{name}") as fin, open(target, "wb") as fout:
                        shutil.copyfileobj(fin, fout, 1 << 20)
                    if sha256_file(target) != sha:
                        raise ModelStoreError(f"{name} in {bundle.name} is corrupt (hash mismatch)")
                    files.append((name, target))
                extra = {k: meta[k] for k in ("opset", "frame_rate_hz") if k in meta}
                return self._install(tier, meta["version"], files, source=f"bundle {bundle.name}",
                                     notes=meta.get("notes", ""), meta=extra, replace=replace)

    def _staging_dir(self, tag: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f".staging-{tag}"
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir()
        return path

    def _install(self, tier, version, files, source, notes, meta, replace) -> dict:
        check_tier(tier)
        version = version or self.next_version(tier)
        if not VERSION_RE.match(version):
            raise ModelStoreError(f"version {version!r} is not MAJOR.MINOR.PATCH")
        folder = f"{tier}-{version}"
        data = self.manifest()
        if any(m["folder"] == folder for m in data["models"]) and not replace:
            raise ModelStoreError(f"{folder} is already installed (use replace to overwrite)")

        staging = self._staging_dir(folder)
        try:
            hashes, sizes = {}, {}
            for name, path in files:
                target = staging / name
                shutil.copyfile(path, target)
                hashes[name] = sha256_file(target)
                sizes[name] = target.stat().st_size
            final = self.root / folder
            if final.exists():
                old = self.root / f".old-{folder}"
                shutil.rmtree(old, ignore_errors=True)
                final.rename(old)
                staging.rename(final)
                shutil.rmtree(old, ignore_errors=True)
            else:
                staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        entry = {"tier": tier, "version": version, "folder": folder,
                 "opset": meta.get("opset", DEFAULT_OPSET),
                 "frame_rate_hz": meta.get("frame_rate_hz", DEFAULT_FRAME_HZ),
                 "files": hashes, "sizes": sizes, "bytes": sum(sizes.values()),
                 "source": source, "installed_at": _now(), "notes": notes}
        data["models"] = [m for m in data["models"] if m["folder"] != folder] + [entry]
        self._write_manifest(data)
        return entry

    # ------------------------------------------------------------------ verify / export / remove

    def verify(self, model_id: str | None = None) -> list:
        results = []
        for m in self.models():
            if model_id and m["folder"] != model_id:
                continue
            problems = []
            base = self.root / m["folder"]
            for name, sha in m["files"].items():
                p = base / name
                if not p.is_file():
                    problems.append(f"missing {name}")
                elif sha256_file(p) != sha:
                    problems.append(f"changed {name}")
            results.append(VerifyResult(m["folder"], not problems, problems))
        if model_id and not results:
            raise ModelStoreError(f"no installed model {model_id!r}")
        return results

    def export_bundle(self, model_id: str, dest: Path) -> Path:
        """Pack an installed model into one .alfamodel file to move it to another machine."""
        m = self.get(model_id)
        check = self.verify(model_id)[0]
        if not check.ok:
            raise ModelStoreError(f"{model_id} failed verification: {'; '.join(check.problems)}")
        dest = Path(dest)
        if dest.is_dir():
            dest = dest / f"alfamedomega-{model_id}{BUNDLE_SUFFIX}"
        meta = {k: v for k, v in m.items() if k not in ("folder", "installed_at", "source")}
        meta["schema"] = SCHEMA
        meta["exported_at"] = _now()
        tmp = dest.with_name(dest.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            zf.writestr(BUNDLE_META, json.dumps(meta, indent=2))
            for name in m["files"]:
                zf.write(self.root / m["folder"] / name, f"files/{name}")
        tmp.replace(dest)
        return dest

    def remove(self, model_id: str) -> None:
        data = self.manifest()
        entry = next((m for m in data["models"] if m["folder"] == model_id), None)
        if entry is None:
            raise ModelStoreError(f"no installed model {model_id!r}")
        # Unlist first, so the plugin never picks a model whose files are being deleted.
        data["models"] = [m for m in data["models"] if m["folder"] != model_id]
        self._write_manifest(data)
        shutil.rmtree(self.root / entry["folder"], ignore_errors=True)

    def set_default_tier(self, tier: str) -> None:
        data = self.manifest()
        data["default_tier"] = check_tier(tier)
        self._write_manifest(data)


def model_bytes(m: dict) -> int:
    """Size on disk; 0 for entries written by the training export, which records no sizes."""
    return m.get("bytes", 0)


def _safe_name(name: str) -> str:
    """Reject archive entries that would write outside the target folder."""
    p = PurePosixPath(name)
    if p.is_absolute() or len(p.parts) != 1 or p.name in ("", ".", "..") or "\\" in name:
        raise ModelStoreError(f"unsafe file name in archive: {name!r}")
    return p.name


# Job actions, so large imports and exports run in the background with a log.
from . import jobs  # noqa: E402


@jobs.action("import_model")
def _import_model(args: dict, log) -> dict:
    store = ModelStore(Path(args["models_dir"]))
    log(f"importing {args['src']}")
    entries = store.import_any(Path(args["src"]), version=args.get("version") or None,
                               notes=args.get("notes", ""), replace=args.get("replace", False))
    for e in entries:
        log(f"installed {e['folder']} ({e['bytes'] / 1e6:.1f} MB, {len(e['files'])} files)")
    return {"installed": ", ".join(e["folder"] for e in entries)}


@jobs.action("export_bundle")
def _export_bundle(args: dict, log) -> dict:
    path = ModelStore(Path(args["models_dir"])).export_bundle(args["id"], Path(args["dest"]))
    log(f"wrote {path}")
    return {"bundle": str(path)}


@jobs.action("verify_models")
def _verify(args: dict, log) -> dict:
    results = ModelStore(Path(args["models_dir"])).verify(args.get("id"))
    for r in results:
        log(f"{r.model_id}: {'ok' if r.ok else '; '.join(r.problems)}")
    bad = [r.model_id for r in results if not r.ok]
    if bad:
        raise ModelStoreError(f"failed verification: {', '.join(bad)}")
    return {"verified": [r.model_id for r in results]}
