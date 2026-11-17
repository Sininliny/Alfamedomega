"""Mix export (MIX-DESIGN.md §15): MIDI files, windows, cross-window linking and the preset folder."""

import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path

from training.amo.corpus import mixexport, mixscene

try:
    from training.amo.native import _library_path

    _library_path(str(mixscene.prepare.LIB) if mixscene.prepare.LIB.is_file() else None)
    NATIVE = True
except (FileNotFoundError, OSError):
    NATIVE = False


def read_notes(data):
    """Minimal SMF reader: [(start tick, end tick, pitch, velocity)] of track 2."""
    assert data[:4] == b"MThd"
    offset = 14
    tracks = []
    while offset < len(data):
        length = struct.unpack(">I", data[offset + 4:offset + 8])[0]
        tracks.append(data[offset + 8:offset + 8 + length])
        offset += 8 + length
    track, i, now, open_, notes = tracks[1], 0, 0, {}, []
    while i < len(track):
        delta = 0
        while True:
            byte = track[i]
            i += 1
            delta = (delta << 7) | (byte & 0x7F)
            if byte < 0x80:
                break
        now += delta
        status = track[i]
        if status == 0xFF:
            length = track[i + 2]
            i += 3 + length
            continue
        pitch, velocity = track[i + 1], track[i + 2]
        i += 3
        if status == 0x90 and velocity:
            open_[pitch] = (now, velocity)
        else:
            start, v = open_.pop(pitch)
            notes.append((start, now, pitch, v))
    return sorted(notes)


class MidiTests(unittest.TestCase):
    def test_round_trip(self):
        notes = [[0.0, 0.5, 60, 1.0], [0.25, 0.25, 64, 0.5], [1.0, 0.1, 67, 0.01]]
        parsed = read_notes(mixexport.midi_bytes(notes, "x"))
        self.assertEqual([(0, 480, 60, 127), (240, 480, 64, 64), (960, 1056, 67, 1)], parsed)

    def test_echo_taps_are_baked(self):
        inst = {"notes": [[0.0, 0.2, 60, 0.8]], "note_echo": {"taps": 2, "delay": 0.3, "transpose": 12,
                                                             "decay": 0.5}}
        notes = mixexport.played_notes(inst)
        self.assertEqual([[0.0, 0.2, 60, 0.8], [0.3, 0.2, 72, 0.4], [0.6, 0.2, 84, 0.2]],
                         [[round(x, 6) for x in n] for n in notes])
        self.assertEqual(1, len(mixexport.played_notes(inst, bake_echo=False)))


class WindowTests(unittest.TestCase):
    def test_windows_cover_and_fold_short_tail(self):
        sr = mixscene.SAMPLE_RATE
        self.assertEqual([(0, 100)], mixexport.windows(100))
        self.assertEqual([(0, 8 * sr), (8 * sr, 16 * sr + sr // 2)], mixexport.windows(16 * sr + sr // 2))
        self.assertEqual([(0, 8 * sr), (8 * sr, 16 * sr), (16 * sr, 18 * sr)], mixexport.windows(18 * sr))

    def test_link_keeps_instruments_across_windows(self):
        scene = mixscene.sample_scene(mixscene.SEED, 3)
        later = copy.deepcopy(scene)
        later["instruments"] = later["instruments"][::-1]
        linked = mixexport.link_windows([(0.0, scene), (8.0, later)])
        self.assertEqual(len(scene["instruments"]), len(linked))
        for inst in linked:
            self.assertEqual(2, inst["windows"])
            first = [n for n in inst["notes"] if n[0] < 8.0]
            self.assertEqual(len(first) * 2, len(inst["notes"]))


@unittest.skipUnless(NATIVE, "native tensor library not built")
class FolderTests(unittest.TestCase):
    def test_folder_has_presets_midi_bus_and_report(self):
        scene = next(s for s in (mixscene.sample_scene(mixscene.SEED, i) for i in range(50)) if s["bus_fx"])
        with tempfile.TemporaryDirectory() as folder:
            report = mixexport.write_export(scene, folder, "mix.wav")
            names = {p.name for p in Path(folder).iterdir()}
            self.assertIn("report.json", names)
            self.assertIn("bus.amo.json", names)
            self.assertEqual(len(scene["instruments"]), len(report["instruments"]))
            for entry in report["instruments"]:
                self.assertIn(entry["preset"], names)
                self.assertIn(entry["midi"], names)
                preset = json.loads((Path(folder) / entry["preset"]).read_text(encoding="utf-8"))
                self.assertEqual("voice_indexed", preset["performance"]["midi_layout"])
            bus = json.loads((Path(folder) / "bus.amo.json").read_text(encoding="utf-8"))
            self.assertEqual({"mode": "fx"}, bus["performance"])
            self.assertEqual(1, bus["preferred_mode"])


if __name__ == "__main__":
    unittest.main()
