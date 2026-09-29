"""The shared Models folder: install, verify, export and remove tier weights.

Layout (docs/control-center/model-store.md, packaging plan section 3):

    Models/
      manifest.json              # every installed model, its files and their SHA-256
      standard-1.0.0/            # one folder per tier and model version
        manifest.json            # the training export's own manifest, copied as is
        patch_encoder.onnx ...

The plugin reads Models/manifest.json and checks hashes before loading, so every write here
is staged in a hidden folder and published with a rename, and the manifest is replaced
atomically. A reader sees the old state or the new one, never a partial model.
"""

import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .settings import write_json_atomic
from .tiers import DEFAULT_TIER, REQUIRED_GRAPHS, TIERS, check_tier

SCHEMA = 1
MANIFEST = "manifest.json"
BUNDLE_META = "alfamodel.json"
BUNDLE_SUFFIX = ".alfamodel"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

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
            if m["id"] == model_id:
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
    def inspect_export(src: Path) -> dict:
        """Read a training export folder (METHOD.md section 7) and say what is in it."""
        src = Path(src)
        man_path = src / MANIFEST
        if not man_path.is_file():
            raise ModelStoreError(f"{src} has no {MANIFEST}; is it a training export folder?")
        export_man = json.loads(man_path.read_text(encoding="utf-8"))
        tier = export_man.get("tier")
        if tier not in TIERS:
            raise ModelStoreError(f"{man_path} names tier {tier!r}, which is not a known tier")
        graphs = {p.stem for p in src.glob("*.onnx") if not p.stem.endswith(".int8")}
        missing = [g for g in REQUIRED_GRAPHS if g not in graphs]
        return {"tier": tier, "manifest": export_man, "graphs": sorted(graphs), "missing": missing}

    # ------------------------------------------------------------------ install

    def import_export(self, src: Path, version: str | None = None, notes: str = "",
                      include_dev_files: bool = False, replace: bool = False) -> dict:
        """Install a training export folder as <tier>-<version>."""
        src = Path(src)
        info = self.inspect_export(src)
        if info["missing"]:
            raise ModelStoreError(f"{src} is missing graphs: {', '.join(info['missing'])}")
        files = []
        for p in sorted(src.iterdir()):
            if not p.is_file():
                continue
            if p.name in DEV_FILES and not include_dev_files:
                continue
            if p.suffix in RUNTIME_SUFFIXES or p.name in DEV_FILES:
                files.append((p.name, p))
        export_man = info["manifest"]
        meta = {k: export_man[k] for k in ("sr", "frame_hz", "hop") if k in export_man}
        return self._install(info["tier"], version, files, source=f"training export {src.name}",
                             notes=notes, meta=meta, replace=replace)

    def import_any(self, src: Path, version: str | None = None, notes: str = "",
                   replace: bool = False) -> dict:
        """Install whatever the user picked: a bundle, an export folder, or a zipped export."""
        src = Path(src)
        if src.suffix == BUNDLE_SUFFIX:
            return self.import_bundle(src, replace=replace)
        if src.is_dir():
            return self.import_export(src, version=version, notes=notes, replace=replace)
        if src.suffix == ".zip":
            staging = self._staging_dir(f"zip-{src.stem}")
            try:
                with zipfile.ZipFile(src) as zf:
                    for info in zf.infolist():
                        parts = PurePosixPath(info.filename).parts
                        if info.is_dir() or not parts:
                            continue
                        target = staging.joinpath(*[_safe_name(x) for x in parts])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as fin, open(target, "wb") as fout:
                            shutil.copyfileobj(fin, fout, 1 << 20)
                found = sorted(p.parent for p in staging.rglob(MANIFEST))
                if not found:
                    raise ModelStoreError(f"{src.name} has no training export inside")
                return self.import_export(found[0], version=version,
                                          notes=notes or f"from {src.name}", replace=replace)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
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
            staging = self._staging_dir(f"bundle-{bundle.stem}")
            try:
                files = []
                for entry in meta["files"]:
                    name = _safe_name(entry["path"])
                    target = staging / name
                    with zf.open(f"files/{name}") as fin, open(target, "wb") as fout:
                        shutil.copyfileobj(fin, fout, 1 << 20)
                    got = sha256_file(target)
                    if got != entry["sha256"]:
                        raise ModelStoreError(f"{name} in {bundle.name} is corrupt (hash mismatch)")
                    files.append((name, target))
                extra = {k: meta[k] for k in ("sr", "frame_hz", "hop") if k in meta}
                return self._install(tier, meta["version"], files, source=f"bundle {bundle.name}",
                                     notes=meta.get("notes", ""), meta=extra, replace=replace)
            finally:
                shutil.rmtree(staging, ignore_errors=True)

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
        model_id = f"{tier}-{version}"
        data = self.manifest()
        existing = [m for m in data["models"] if m["id"] == model_id]
        if existing and not replace:
            raise ModelStoreError(f"{model_id} is already installed (use replace to overwrite)")

        staging = self._staging_dir(model_id)
        try:
            entries = []
            for name, path in files:
                target = staging / name
                shutil.copyfile(path, target)
                entries.append({"path": name, "sha256": sha256_file(target),
                                "bytes": target.stat().st_size})
            final = self.root / model_id
            if final.exists():
                old = self.root / f".old-{model_id}"
                shutil.rmtree(old, ignore_errors=True)
                final.rename(old)
                staging.rename(final)
                shutil.rmtree(old, ignore_errors=True)
            else:
                staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        entry = {"id": model_id, "tier": tier, "version": version, "dir": model_id,
                 "files": entries, "bytes": sum(e["bytes"] for e in entries), **meta,
                 "source": source, "installed_at": _now(), "notes": notes}
        data["models"] = [m for m in data["models"] if m["id"] != model_id] + [entry]
        self._write_manifest(data)
        return entry

    # ------------------------------------------------------------------ verify / export / remove

    def verify(self, model_id: str | None = None) -> list:
        results = []
        for m in self.models():
            if model_id and m["id"] != model_id:
                continue
            problems = []
            base = self.root / m["dir"]
            for f in m["files"]:
                p = base / f["path"]
                if not p.is_file():
                    problems.append(f"missing {f['path']}")
                elif p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
                    problems.append(f"changed {f['path']}")
            results.append(VerifyResult(m["id"], not problems, problems))
        if model_id and not results:
            raise ModelStoreError(f"no installed model {model_id!r}")
        return results

    def export_bundle(self, model_id: str, dest: Path) -> Path:
        """Pack an installed model into one .alfamodel file to move it to another machine."""
        m = self.get(model_id)
        bad = self.verify(model_id)[0]
        if not bad.ok:
            raise ModelStoreError(f"{model_id} failed verification: {'; '.join(bad.problems)}")
        dest = Path(dest)
        if dest.is_dir():
            dest = dest / f"alfamedomega-{model_id}{BUNDLE_SUFFIX}"
        meta = {k: v for k, v in m.items() if k not in ("dir", "installed_at", "source")}
        meta["schema"] = SCHEMA
        meta["exported_at"] = _now()
        tmp = dest.with_name(dest.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            zf.writestr(BUNDLE_META, json.dumps(meta, indent=2))
            for f in m["files"]:
                zf.write(self.root / m["dir"] / f["path"], f"files/{f['path']}")
        tmp.replace(dest)
        return dest

    def remove(self, model_id: str) -> None:
        data = self.manifest()
        entry = next((m for m in data["models"] if m["id"] == model_id), None)
        if entry is None:
            raise ModelStoreError(f"no installed model {model_id!r}")
        # Unlist first, so the plugin never picks a model whose files are being deleted.
        data["models"] = [m for m in data["models"] if m["id"] != model_id]
        self._write_manifest(data)
        shutil.rmtree(self.root / entry["dir"], ignore_errors=True)

    def set_default_tier(self, tier: str) -> None:
        data = self.manifest()
        data["default_tier"] = check_tier(tier)
        self._write_manifest(data)


def _safe_name(name: str) -> str:
    """Reject bundle entries that would write outside the model folder."""
    p = PurePosixPath(name)
    if p.is_absolute() or len(p.parts) != 1 or p.name in ("", ".", "..") or "\\" in name:
        raise ModelStoreError(f"unsafe file name in bundle: {name!r}")
    return p.name


# Job actions, so large imports and exports run in the background with a log.
from . import jobs  # noqa: E402


@jobs.action("import_model")
def _import_model(args: dict, log) -> dict:
    store = ModelStore(Path(args["models_dir"]))
    log(f"importing {args['src']}")
    e = store.import_any(Path(args["src"]), version=args.get("version") or None,
                         notes=args.get("notes", ""), replace=args.get("replace", False))
    log(f"installed {e['id']} ({e['bytes'] / 1e6:.1f} MB, {len(e['files'])} files)")
    return {"installed": e["id"]}


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
