import struct
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
