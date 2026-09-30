"""Smoke test for the ctypes binding. Exit 77 (skip) when numpy or the library is missing."""

import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit(77)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alfa_dsp  # noqa: E402

lib = sys.argv[1] if len(sys.argv) > 1 else None
if lib is None or not os.path.exists(lib):
    sys.exit(77)

s = alfa_dsp.Synth(48000, "standard", lib_path=lib)
assert s.layout_version == 2 and s.hop == 192
frames = np.zeros((250, s.frame_floats), np.float32)
frames[:, 0] = 1.0            # mix[osc]
frames[:, 6] = 1e-3           # attackTau
frames[:, 8] = 0.5            # osc.re[0]: harmonic 1
frames[:, 2452] = 1.0         # dry_gain (v2 default)
left, right = s.render(frames, note=69)
assert left.shape == (250 * 192,)
assert np.all(np.isfinite(left)) and np.allclose(left, right)


def peak_hz(x):
    w = x[4800:]
    return np.argmax(np.abs(np.fft.rfft(w * np.hanning(w.size)))) * 48000 / w.size


assert abs(peak_hz(left) - 440) <= 2, peak_hz(left)
again, _ = s.render(frames, note=69)
assert np.array_equal(left, again), "renders must be deterministic"
# f0 per frame overrides the note pitch.
up, _ = s.render(frames, f0=np.full(250, 880.0, np.float32), note=69)
assert abs(peak_hz(up) - 880) <= 2, peak_hz(up)
print("ok")
