import struct
import json
import sys

import pytest

from alfacc import jobs, preference
from alfacc.preference import PreferenceStore, audio_info, create_batch_job, fallback_tokens, tokens


def write_wav(path, seconds=0.5, sr=48000, ch=2, bits=24):
    n = int(seconds * sr)
    block = ch * bits // 8
    data = b"\0" * (n * block)
    fmt = struct.pack("<HHIIHH", 1, ch, sr, sr * block, block, bits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + 8 + 16 + 8 + len(data) + 8) + b"WAVE"
                     + b"LIST" + struct.pack("<I", 0)
                     + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data)


def test_fallback_tokens_keep_timbre_words_only():
    # The control center's own rules, used when no preference/ checkout is around.
    assert fallback_tokens("warm_pad_03") == ["warm", "pad"]
    assert fallback_tokens("AiryVocal C3 120bpm v2 (final)") == ["airy", "vocal"]
    assert fallback_tokens("GlassBell-OneShot") == ["glass", "bell"]
    assert fallback_tokens("dark bass F#1.wav") == ["dark", "bass"]
    assert fallback_tokens("G#-1 bell Bb pluck") == ["bell", "pluck"]
    assert fallback_tokens("808 Kick") == ["kick"]
    assert fallback_tokens("Pads Leads Keys Brass Chorus") == ["pad", "lead", "key", "brass", "chorus"]


def test_tokens_follow_the_preference_tokenizer_in_a_checkout():
    """In a checkout, labels come from preference/alfapref/tokens.py so they match what the
    learner computes. Its exact rules belong to preference/ and are tested there."""
    if preference._SHARED_WORDS is None:
        pytest.skip("no preference/ checkout next to the control center")
    for text in ("warm_pad_03", "dark bass F#1.wav", "GlassBell-OneShot"):
        assert tokens(text) == list(dict.fromkeys(preference._SHARED_WORDS(text)))
    assert tokens("warm_pad_03") == ["warm", "pad"]


def test_wav_header_is_read_without_decoding(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p, seconds=0.25, sr=44100, ch=1, bits=16)
    assert audio_info(p) == {"sr": 44100, "channels": 1, "bits": 16, "duration_s": 0.25}
    (tmp_path / "junk.wav").write_bytes(b"not a wav")
    assert audio_info(tmp_path / "junk.wav") == {}


def test_scan_indexes_labels_and_is_incremental(tmp_path):
    lib = tmp_path / "MyLib"
    write_wav(lib / "Pads" / "warm_pad_03.wav")
    write_wav(lib / "Leads" / "aggressive_lead.wav")
    (lib / "Keys").mkdir()
    (lib / "Keys" / "glass_bell.flac").write_bytes(b"flac")
    (lib / "notes.txt").write_text("ignored")
    store = PreferenceStore(tmp_path / "home")
    store.add_library(lib)
    s = store.scan(lambda _: None)
    assert s["files"] == 3 and s["reread"] == 3
    idx = store.load_index()
    rec = idx[str(lib / "Pads" / "warm_pad_03.wav")]
    assert rec["name_labels"] == ["warm", "pad"] and rec["folder_labels"] == ["pad"]
    assert rec["duration_s"] == 0.5
    voc = store.vocabulary()["labels"]
    assert voc["glass"] == 1 and voc["aggressive"] == 1

    (lib / "Leads" / "aggressive_lead.wav").unlink()
    write_wav(lib / "Pads" / "dark_pad.wav")
    s = store.scan(lambda _: None)
    assert (s["files"], s["reused"], s["reread"], s["removed"]) == (3, 2, 1, 1)

    store.set_enabled(lib, False)
    assert store.scan(lambda _: None)["files"] == 0


def test_batch_runs_learner_with_index_and_forget_clears(tmp_path, ctx):
    lib = tmp_path / "Lib"
    write_wav(lib / "warm_pad.wav")
    ctx.prefs.add_library(lib)
    learner = tmp_path / "learner.py"
    learner.write_text(
        "import sys, os, json\n"
        "a = sys.argv[1:]\n"
        "assert a[0] == 'batch'\n"
        "out = a[a.index('--out') + 1]\n"
        "os.makedirs(out)\n"
        "n = sum(1 for _ in open(a[a.index('--index') + 1]))\n"
        "json.dump({'files': n}, open(os.path.join(out, 'profile.json'), 'w'))\n"
        "print('progress 1/1')\n")
    job = create_batch_job(ctx.jobs, ctx.home, f'"{sys.executable}" "{learner}"')
    done = ctx.jobs.run(job["id"])
    assert done["status"] == jobs.SUCCEEDED, ctx.jobs.read_log(job["id"])
    assert done["result"]["scan"]["files"] == 1
    assert len(ctx.prefs.profiles()) == 1

    ctx.prefs.forget()
    assert ctx.prefs.profiles() == [] and not ctx.prefs.index_path.exists()
    assert ctx.prefs.libraries()  # the folder list is kept


def test_batch_without_learner_still_indexes(tmp_path, ctx):
    write_wav(tmp_path / "Lib" / "airy_vocal.wav")
    ctx.prefs.add_library(tmp_path / "Lib")
    done = ctx.jobs.run(create_batch_job(ctx.jobs, ctx.home, "")["id"])
    assert done["status"] == jobs.SUCCEEDED
    assert done["result"]["learner"] == "not configured"
    assert ctx.prefs.vocabulary()["labels"]["airy"] == 1


def test_batch_passes_assimilated_plugins_only_to_a_learner_that_reads_them(tmp_path):
    from alfacc import jobs as jobs_mod
    from alfacc.preference import create_batch_job
    home = tmp_path / "home"
    (home / "assimilation" / "vendor-synth-vst3").mkdir(parents=True)
    (home / "assimilation" / "vendor-synth-vst3" / "features.jsonl").write_text("{}\n")
    new, old = tmp_path / "learner.py", tmp_path / "old_learner.py"
    new.write_text('p.add_argument("--assimilation")\n')
    old.write_text("# no such flag\n")
    manager = jobs_mod.JobManager(home)
    job = create_batch_job(manager, home, f"python {new}")  # no libraries: plugin notes are enough
    argv = job["steps"][1]["argv"]
    assert argv[argv.index("--assimilation") + 1] == str(home / "assimilation")
    with pytest.raises(ValueError):
        create_batch_job(manager, home, f"python {old}")
    # The real learner.py is a launcher; the flag lives in alfapref/cli.py beside it.
    launcher = tmp_path / "repo" / "learner.py"
    (launcher.parent / "alfapref").mkdir(parents=True)
    launcher.write_text("from alfapref.cli import main\n")
    (launcher.parent / "alfapref" / "cli.py").write_text('p.add_argument("--assimilation")\n')
    assert "--assimilation" in create_batch_job(manager, home, f"python {launcher}")["steps"][1]["argv"]


# ---------------------------------------------------------------------- consent, words, reset

def test_learning_is_off_until_turned_on_and_keeps_the_plugins_keys(tmp_path):
    from alfacc.preference import PreferenceStore
    p = PreferenceStore(tmp_path)
    c = p.consent()
    assert not c["learn"] and not any(c["categories"].values())
    p.consent_path.write_text('{"learn": true, "plugin_extra": 7}')
    assert all(p.consent()["categories"].values())  # {"learn": true} alone means every category
    c = p.set_consent(categories={"workflow": False}, personalization=1.7)
    assert c["categories"] == {"performance": True, "parameters": True, "suggestions": True,
                               "workflow": False}
    assert c["personalization"] == 1.0
    raw = json.loads(p.consent_path.read_text())
    assert raw["plugin_extra"] == 7 and raw["learn"] is True
    assert not any(p.set_consent(learn=False)["categories"].values())
    with pytest.raises(ValueError):
        p.set_consent(categories={"spying": True})


def test_tags_and_labels_reach_the_index_and_the_learner(tmp_path):
    from alfacc import jobs as jobs_mod
    from alfacc.preference import PreferenceStore, create_batch_job
    home, lib = tmp_path / "home", tmp_path / "lib"
    lib.mkdir()
    wav = lib / "Pad.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    p = PreferenceStore(home)
    p.add_library(lib)
    p.annotate(wav, tags=["pad", " "], description="slow swell", labels=["warm", "warm", "airy"])
    p.scan(lambda _l: None)
    rec = next(iter(p.load_index().values()))
    assert rec["tags"] == ["pad"] and rec["labels"] == ["warm", "airy"] and rec["description"] == "slow swell"
    assert json.loads(p.labels_path.read_text())["labels"] == ["warm", "airy"]
    learner = tmp_path / "learner.py"
    learner.write_text('ap.add_argument("--consent")\nap.add_argument("--labels")\n')
    argv = create_batch_job(jobs_mod.JobManager(home), home, f"python {learner}")["steps"][1]["argv"]
    assert argv[argv.index("--consent") + 1] == str(p.consent_path)
    assert argv[argv.index("--labels") + 1] == str(p.labels_path)
    assert argv[-2] == "--out"
    # An older learner gets neither flag.
    old = tmp_path / "old.py"
    old.write_text("# batch only\n")
    argv = create_batch_job(jobs_mod.JobManager(home), home, f"python {old}")["steps"][1]["argv"]
    assert "--consent" not in argv and "--labels" not in argv
    # Clearing every field removes the file's entry, and a rescan drops the words.
    p.annotate(wav, tags=[], description="", labels=[])
    p.scan(lambda _l: None)
    assert "tags" not in next(iter(p.load_index().values())) and p.annotations() == {}


def test_explain_and_reset(tmp_path):
    from alfacc.preference import PreferenceStore, reset_learning
    p = PreferenceStore(tmp_path)
    assert p.explain() == [] and reset_learning(p, "") == "Nothing was learned yet."
    for stamp in ("20260930-100000", "20260930-110000"):
        (p.profiles_dir / stamp).mkdir(parents=True)
    latest = p.profiles_dir / "20260930-110000"
    (latest / "summary.json").write_text(json.dumps({"explain": ["Heard 12 sounds.", "Liked 3."]}))
    (p.events_dir).mkdir()
    (p.events_dir / "2026-09-30.jsonl").write_text("{}\n")
    assert p.explain() == ["Heard 12 sounds.", "Liked 3."]
    learner = tmp_path / "learner.py"
    learner.write_text(
        'import sys, pathlib\n# choices=["reset"]\n'
        'a = sys.argv\nprof = pathlib.Path(a[a.index("--profile") + 1])\n'
        '(prof / "reset.txt").write_text("all" if "--all" in a else "model")\n')
    msg = reset_learning(p, f"{sys.executable} {learner}", everything=True)
    assert (latest / "reset.txt").read_text() == "all"
    assert not (p.profiles_dir / "20260930-100000").exists() and not p.events_dir.exists()
    assert msg.startswith("Reset the learned model")
