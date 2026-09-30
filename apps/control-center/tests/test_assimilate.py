import json
import os
import stat
import sys
from pathlib import Path

import pytest

from alfacc.assimilate import discovery, harvest, plans
from alfacc.cli import main as cli

HERE = Path(__file__).parent


# ---------------------------------------------------------------------- plans

def test_grid_is_anchored_on_c_and_keeps_both_ends():
    assert plans.grid_keys(24, 96, 12) == [24, 36, 48, 60, 72, 84, 96]
    assert plans.grid_keys(21, 108, 12) == [21, 24, 36, 48, 60, 72, 84, 96, 108]
    assert plans.grid_keys(60, 62, 1) == [60, 61, 62]


def test_densities_match_the_spec():
    d = plans.DENSITIES
    assert (d["fast"].step, len(d["fast"].velocities)) == (12, 1)
    assert 4 <= d["normal"].step <= 6 and len(d["normal"].velocities) == 3
    assert d["detailed"].step == 1 and 4 <= len(d["detailed"].velocities) <= 8
    u = d["unreasonable"]
    assert u.step == 1 and len(u.velocities) > 1 and len(u.durations) > 1 and u.release_analysis
    notes = plans.grid_notes(24, 36, u)
    assert len(notes) == 13 * 8 * 3 and len({n.id for n in notes}) == len(notes)


def test_custom_density_overrides_only_what_is_given():
    c = plans.custom_density("normal", step=4)
    assert c.step == 4 and c.velocities == plans.DENSITIES["normal"].velocities and c.name == "custom"


# ---------------------------------------------------------------------- discovery

def test_discovery_marks_what_the_host_can_open(tmp_path):
    vst3, aax, vst2 = tmp_path / "VST3", tmp_path / "AAX", tmp_path / "VST"
    (vst3 / "Pads.vst3" / "Contents" / "x86_64-linux").mkdir(parents=True)
    (vst3 / "Pads.vst3" / "Contents" / "x86_64-linux" / "Pads.so").write_bytes(b"")
    (aax / "Pads.aaxplugin" / "Contents").mkdir(parents=True)
    (aax / "Lonely.aaxplugin").mkdir(parents=True)
    vst2.mkdir()
    (vst2 / "Old.so").write_bytes(b"")
    found = discovery.discover(host_formats=["VST3"], folders={"VST3": [vst3], "AAX": [aax], "VST2": [vst2]})
    by = {(p.name, p.format): p for p in found}
    assert by[("Pads", "VST3")].loadable
    assert not by[("Pads", "AAX")].loadable and by[("Pads", "AAX")].twin.endswith("Pads.vst3")
    assert not by[("Lonely", "AAX")].loadable and not by[("Lonely", "AAX")].twin
    if sys.platform == "linux":
        assert "VST2 SDK" in by[("Old", "VST2")].reason
    # The .so inside the VST3 bundle isn't reported as a separate VST2 plugin.
    assert not any(p.path.endswith("Pads.so") for p in found)


# ---------------------------------------------------------------------- range finding

def _rec(midi, peak=-10.0, pitched=True, audible=True):
    return {"midi": midi, "peak_db": peak, "pitched": pitched, "audible": audible}


def test_range_is_the_longest_run_of_keys_that_track_pitch():
    # Few keys track pitch: an unpitched preset (a drum, a noise) uses every audible key.
    recs = [_rec(m, pitched=24 <= m <= 60) for m in range(12, 121, 6)]
    rng = harvest.decide_range(recs)
    assert (rng["status"], rng["pitched"], rng["low"], rng["high"]) == ("ranged", False, 12, 120)
    recs = [_rec(m, pitched=36 <= m <= 108) for m in range(12, 121, 6)]
    rng = harvest.decide_range(recs)
    assert rng["pitched"] and (rng["low"], rng["high"]) == (36, 108)


def test_range_ignores_silent_and_far_too_quiet_keys():
    recs = [_rec(m, peak=-240, audible=False) for m in (12, 18)]
    recs += [_rec(m) for m in range(24, 61, 6)]
    recs += [_rec(66, peak=-55.0)]
    rng = harvest.decide_range(recs)
    assert (rng["low"], rng["high"]) == (24, 60)
    assert harvest.decide_range([_rec(m, peak=-240, audible=False) for m in (12, 18)])["status"] == "silent"


def test_edges_are_refined_key_by_key_inside_the_probed_span():
    rng = {"status": "ranged", "pitched": True, "low": 24, "high": 60}
    assert harvest.refine_keys(rng) == [19, 20, 21, 22, 23, 61, 62, 63, 64, 65]
    assert harvest.refine_keys({**rng, "low": 12, "high": 120}) == []
    refined = [_rec(m, pitched=m >= 21) if m < 24 else _rec(m, audible=m <= 62, peak=-10 if m <= 62 else -240)
               for m in harvest.refine_keys(rng)]
    out = harvest.extend_range(rng, refined, [_rec(30)])
    assert (out["low"], out["high"], out["range"]) == (21, 62, "A0–D4")


# ---------------------------------------------------------------------- analysis worker

np = pytest.importorskip("numpy")


def _wav(path, x, sr=48000):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def test_worker_finds_pitch_level_and_release(tmp_path):
    sr = 48000
    t = np.arange(int(1.5 * sr)) / sr
    env = np.where(t < 1.0, 1.0, np.exp(-(t - 1.0) / 0.05))
    _wav(tmp_path / "a.wav", 0.5 * env * np.sin(2 * np.pi * 220 * t))
    _wav(tmp_path / "n.wav", 0.5 * np.random.default_rng(0).uniform(-1, 1, len(t)))
    renders = {"renders": [{"preset": "p", "note": "a", "wav": str(tmp_path / "a.wav")},
                           {"preset": "p", "note": "n", "wav": str(tmp_path / "n.wav")}]}
    notes = {k: {"midi": 57, "velocity": 100, "dur_s": 1.0} for k in ("a", "n")}
    (tmp_path / "r.json").write_text(json.dumps(renders))
    (tmp_path / "n.json").write_text(json.dumps(notes))
    sys.path.insert(0, str(Path(harvest.WORKER).parent))
    import analyze_worker
    analyze_worker.main(["--renders", str(tmp_path / "r.json"), "--notes", str(tmp_path / "n.json"),
                         "--out", str(tmp_path / "f.jsonl"), "--delete-audio"])
    a, n = [json.loads(line) for line in (tmp_path / "f.jsonl").read_text().splitlines()]
    assert a["pitched"] and abs(a["cents"]) < 5 and a["octave"] == 0
    assert abs(a["peak_db"] + 6.0) < 0.5 and 200 < a["release_ms"] < 500
    assert not n["pitched"] and n["flatness"] > 0.5
    assert not (tmp_path / "a.wav").exists()


# ---------------------------------------------------------------------- whole pipeline

@pytest.fixture
def fake_host(tmp_path):
    if sys.platform == "win32":
        pytest.skip("the fake host wrapper is a shell script")
    host = tmp_path / "alfa-render-host"
    host.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE / "fake_render_host.py"}" "$@"\n')
    host.chmod(host.stat().st_mode | stat.S_IEXEC)
    return str(host)


def _assimilate(ctx, host, density="fast"):
    ctx.settings.render_host = host
    ctx.save_settings()
    plug = Path(ctx.home) / "FakeSynth.vst3"
    plug.mkdir(exist_ok=True)
    assert cli(["--home", str(ctx.home), "plugins", "scan", str(plug), "--quiet"]) == 0
    store = ctx.assimilation
    plugin = store.instruments()[0]
    job = harvest.create_assimilation_job(ctx.jobs, ctx.home, plugin["uid"], plans.DENSITIES[density],
                                          host, sys.executable)
    return store, plugin, ctx.jobs.run(job["id"])


def test_assimilation_finds_ranges_and_writes_the_knowledge_base(ctx, fake_host):
    store, plugin, done = _assimilate(ctx, fake_host)
    assert done["status"] == "succeeded", ctx.jobs.read_log(done["id"])
    res = store.results(plugin)
    assert res["p0001"]["range"] == "C1–C4" and res["p0001"]["pitched"]
    assert not res["p0002"]["pitched"]
    assert res["p0000"]["low"] <= 24 and res["p0000"]["high"] >= 96
    feats = store.features(plugin, "p0001")
    assert [f["midi"] for f in feats] == [24, 36, 48, 60]
    assert all(f["pitched"] for f in feats)
    assert not (store.plugin_dir(plugin) / "work").exists()  # audio deleted after analysis
    rows = store.summary()
    assert rows[0]["harvested"] == 3 and rows[0]["renders"] == len(store.features(plugin))


def test_a_preset_that_crashes_the_plugin_is_recorded_and_skipped(ctx, fake_host, monkeypatch):
    store, plugin, _ = _assimilate(ctx, fake_host)
    monkeypatch.setenv("FAKE_HOST_CRASH", "p0002")
    done = ctx.jobs.run(harvest.create_assimilation_job(
        ctx.jobs, ctx.home, plugin["uid"], plans.DENSITIES["fast"], fake_host, sys.executable)["id"])
    assert done["status"] == "succeeded", ctx.jobs.read_log(done["id"])
    res = store.results(plugin)
    assert res["p0002"]["status"] == "failed" and "error" in res["p0002"]
    assert res["p0001"]["status"] == "harvested" and res["p0001"]["range"] == "C1–C4"
    assert "retrying each preset" in ctx.jobs.read_log(done["id"])


def test_scanning_survives_a_plugin_that_crashes(ctx, fake_host):
    good, bad = Path(ctx.home) / "Good.vst3", Path(ctx.home) / "Crash.vst3"
    good.mkdir(), bad.mkdir()
    job = harvest.create_scan_job(ctx.jobs, ctx.home, fake_host, [str(good), str(bad)])
    done = ctx.jobs.run(job["id"])
    assert done["status"] == "succeeded"
    cat = ctx.assimilation.catalog()
    assert str(bad) in cat["failed"] and len(cat["plugins"]) == 1


def test_cli_lists_estimates_and_forgets(ctx, fake_host, capsys):
    _assimilate(ctx, fake_host)
    home = str(ctx.home)
    assert cli(["--home", home, "plugins", "list"]) == 0
    assert "3/3 presets" in capsys.readouterr().out
    assert cli(["--home", home, "plugins", "estimate", "Fake"]) == 0
    assert "Unreasonable" in capsys.readouterr().out
    assert cli(["--home", home, "plugins", "show", "Fake", "--preset", "p0001"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 4
    assert cli(["--home", home, "plugins", "forget", "Fake"]) == 0
    assert ctx.assimilation.results(ctx.assimilation.instruments()[0]) == {}


# ---------------------------------------------------------------------- the real render host

REAL_HOST = os.environ.get("ALFA_RENDER_HOST", "")
TEST_SYNTH = os.environ.get("ALFA_TEST_SYNTH", "")


@pytest.mark.skipif(not (REAL_HOST and TEST_SYNTH), reason="set ALFA_RENDER_HOST and ALFA_TEST_SYNTH "
                    "to a built render host and AlfaTestSynth.vst3")
def test_real_render_host_with_the_test_synth(ctx):
    ctx.settings.render_host = REAL_HOST
    assert harvest.host_formats(REAL_HOST)[0] == "VST3"
    ctx.jobs.run(harvest.create_scan_job(ctx.jobs, ctx.home, REAL_HOST, [TEST_SYNTH])["id"])
    plugin = ctx.assimilation.instruments()[0]
    assert plugin["name"] == "AlfaTestSynth" and plugin["outputs"] == 2
    done = ctx.jobs.run(harvest.create_assimilation_job(
        ctx.jobs, ctx.home, plugin["uid"], plans.DENSITIES["fast"], REAL_HOST, sys.executable)["id"])
    assert done["status"] == "succeeded", ctx.jobs.read_log(done["id"])
    res = ctx.assimilation.results(plugin)
    assert res["p0001"]["range"] == "C1–C4" and not res["p0002"]["pitched"]
