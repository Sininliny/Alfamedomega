import json
import zipfile

import pytest

from alfacc.models import ModelStore, ModelStoreError


def test_import_export_folder_writes_host_manifest(tmp_path, fake_export):
    store = ModelStore(tmp_path / "Models")
    [entry] = store.import_export(fake_export, version="0.1.0")
    assert entry["folder"] == "lite-0.1.0"
    assert "voice_core.onnx" in entry["files"] and "export_info.json" in entry["files"]
    assert "dsp_golden.npz" not in entry["files"]  # developer test vectors stay out of installs
    man = json.loads((tmp_path / "Models" / "manifest.json").read_text())
    m = man["models"][0]
    # The inference host's contract (src/inference/MODEL_CONTRACT.md).
    assert man["schema"] == 1
    assert {k: m[k] for k in ("tier", "version", "folder", "opset", "frame_rate_hz")} == \
        {"tier": "lite", "version": "0.1.0", "folder": "lite-0.1.0", "opset": 17, "frame_rate_hz": 250}
    assert all(isinstance(v, str) and len(v) == 64 for v in m["files"].values())
    assert all(r.ok for r in store.verify())
    assert not list((tmp_path / "Models").glob(".staging-*"))


def test_versions_increment_and_duplicates_are_refused(tmp_path, fake_export):
    store = ModelStore(tmp_path / "Models")
    store.import_export(fake_export)
    assert store.import_export(fake_export, version="0.1.1")[0]["version"] == "0.1.1"
    with pytest.raises(ModelStoreError, match="already installed"):
        store.import_export(fake_export, version="0.1.0")
    assert store.import_export(fake_export, version="0.1.0", replace=True)[0]["folder"] == "lite-0.1.0"
    assert [m["folder"] for m in store.models()] == ["lite-0.1.0", "lite-0.1.1"]
    with pytest.raises(ModelStoreError, match="MAJOR.MINOR.PATCH"):
        store.import_export(fake_export, version="1.0")


def test_incomplete_export_is_rejected(tmp_path, fake_export):
    (fake_export / "lite-0.1.0" / "voice_core.onnx").unlink()
    with pytest.raises(ModelStoreError, match="voice_core"):
        ModelStore(tmp_path / "Models").import_export(fake_export)


def test_verify_detects_tampering_and_missing_files(tmp_path, fake_export):
    store = ModelStore(tmp_path / "Models")
    store.import_export(fake_export)
    d = tmp_path / "Models" / "lite-0.1.0"
    (d / "expert_fm.onnx").write_bytes(b"tampered")
    (d / "voice_core.onnx").unlink()
    [r] = store.verify()
    assert not r.ok
    assert set(r.problems) == {"changed expert_fm.onnx", "missing voice_core.onnx"}


def test_bundle_round_trip_between_machines(tmp_path, fake_export):
    a = ModelStore(tmp_path / "A")
    a.import_export(fake_export, version="1.2.3", notes="first")
    bundle = a.export_bundle("lite-1.2.3", tmp_path)
    assert bundle.name == "alfamedomega-lite-1.2.3.alfamodel"
    b = ModelStore(tmp_path / "B")
    [entry] = b.import_any(bundle)
    assert entry["folder"] == "lite-1.2.3" and entry["notes"] == "first"
    assert entry["files"] == a.get("lite-1.2.3")["files"]
    assert all(r.ok for r in b.verify())


def test_corrupt_bundle_is_rejected(tmp_path, fake_export):
    a = ModelStore(tmp_path / "A")
    a.import_export(fake_export)
    bundle = a.export_bundle("lite-0.1.0", tmp_path)
    bad = tmp_path / "bad.alfamodel"
    with zipfile.ZipFile(bundle) as zin, zipfile.ZipFile(bad, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item)
            zout.writestr(item, b"evil" if item.filename == "files/voice_core.onnx" else data)
    b = ModelStore(tmp_path / "B")
    with pytest.raises(ModelStoreError, match="corrupt"):
        b.import_bundle(bad)
    assert b.models() == []


def test_bundle_with_path_traversal_is_rejected(tmp_path):
    bad = tmp_path / "evil.alfamodel"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("alfamodel.json", json.dumps({"schema": 1, "tier": "lite", "version": "1.0.0",
                                                 "files": {"../../x": "0"}}))
        z.writestr("files/../../x", b"x")
    with pytest.raises(ModelStoreError, match="unsafe"):
        ModelStore(tmp_path / "M").import_bundle(bad)
    assert not (tmp_path / "x").exists()


def test_zipped_export_from_colab(tmp_path, fake_export):
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:  # what the Colab notebook's last cell makes
        for p in fake_export.rglob("*"):
            if p.is_file():
                zf.write(p, f"export/{p.relative_to(fake_export).as_posix()}")
    [entry] = ModelStore(tmp_path / "M").import_any(z)
    assert entry["folder"] == "lite-0.1.0"


def test_single_tier_folder_and_training_written_models_folder(tmp_path, fake_export):
    """A lone <tier>-<version> folder imports too, and a Models folder written by the training
    export directly (no sizes, no default_tier) is readable."""
    [e] = ModelStore(tmp_path / "M").import_any(fake_export / "lite-0.1.0")
    assert e["folder"] == "lite-0.1.0"
    direct = ModelStore(fake_export)
    assert [m["folder"] for m in direct.models()] == ["lite-0.1.0"]
    assert direct.manifest()["default_tier"] == "standard"
    [r] = direct.verify()
    assert r.ok


def test_remove_and_default_tier(tmp_path, fake_export):
    store = ModelStore(tmp_path / "Models")
    store.import_export(fake_export)
    store.set_default_tier("lite")
    store.remove("lite-0.1.0")
    assert store.models() == [] and store.manifest()["default_tier"] == "lite"
    assert not (tmp_path / "Models" / "lite-0.1.0").exists()
    with pytest.raises(ModelStoreError):
        store.remove("lite-0.1.0")


def test_newer_manifest_schema_is_not_overwritten(tmp_path):
    root = tmp_path / "Models"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": 99, "models": []}))
    with pytest.raises(ModelStoreError, match="schema 99"):
        ModelStore(root).models()
