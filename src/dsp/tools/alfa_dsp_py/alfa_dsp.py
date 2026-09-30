"""ctypes binding for Alfa::dsp offline rendering (tools/alfa_dsp_py/alfa_dsp_c.cpp).

Build: cmake --build <dir> --target alfa_dsp_c, then point ALFA_DSP_LIB at the library or pass
its path to Synth(). Batching: one Synth per worker, render() per item; each call resets the
synth, so results are deterministic and independent of order.

    s = Synth(48000, tier="standard")
    left, right = s.render(voice_frames, note=60)          # voice_frames: [T, frame_floats]
"""

import ctypes
import os
import sys

import numpy as np

_TIERS = {"lite": 0, "standard": 1, "pro": 2, "studio": 3}
_F32P = ctypes.POINTER(ctypes.c_float)


def _default_lib():
    env = os.environ.get("ALFA_DSP_LIB")
    if env:
        return env
    name = {"win32": "alfa_dsp_c.dll", "darwin": "libalfa_dsp_c.dylib"}.get(sys.platform, "libalfa_dsp_c.so")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def _load(path):
    lib = ctypes.CDLL(path)
    lib.alfa_layout_version.restype = ctypes.c_int
    lib.alfa_frame_floats.restype = ctypes.c_int
    lib.alfa_instance_frame_floats.restype = ctypes.c_int
    lib.alfa_hop.argtypes = [ctypes.c_double]
    lib.alfa_hop.restype = ctypes.c_int
    lib.alfa_create.argtypes = [ctypes.c_double, ctypes.c_int]
    lib.alfa_create.restype = ctypes.c_void_p
    lib.alfa_destroy.argtypes = [ctypes.c_void_p]
    lib.alfa_set_grain_bank.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_int, ctypes.c_int]
    lib.alfa_set_sample.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_longlong, ctypes.c_double, ctypes.c_double,
                                    ctypes.c_longlong, ctypes.c_longlong]
    lib.alfa_render.argtypes = [ctypes.c_void_p, _F32P, _F32P, _F32P, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                ctypes.c_int, _F32P, _F32P]
    lib.alfa_render.restype = ctypes.c_longlong
    return lib


def _ptr(a):
    return None if a is None else a.ctypes.data_as(_F32P)


def _f32(a, shape_tail=None):
    if a is None:
        return None
    a = np.ascontiguousarray(a, dtype=np.float32)
    if shape_tail is not None and (a.ndim != 2 or a.shape[1] != shape_tail):
        raise ValueError(f"expected [T, {shape_tail}] frames, got {a.shape}")
    return a


class Synth:
    def __init__(self, sample_rate=48000.0, tier="standard", lib_path=None):
        self._lib = _load(lib_path or _default_lib())
        self.layout_version = self._lib.alfa_layout_version()
        self.frame_floats = self._lib.alfa_frame_floats()
        self.instance_frame_floats = self._lib.alfa_instance_frame_floats()
        self.sample_rate = float(sample_rate)
        self.hop = self._lib.alfa_hop(self.sample_rate)
        tier_id = _TIERS[tier] if isinstance(tier, str) else int(tier)
        self._h = self._lib.alfa_create(self.sample_rate, tier_id)
        if not self._h:
            raise ValueError("alfa_create rejected the sample rate or tier")
        self._keep = []  # arrays the C side copies anyway; kept only for clarity of lifetime

    def close(self):
        if self._h:
            self._lib.alfa_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def set_grain_bank(self, sources):
        """sources: [count, length] float32 (length = grain_source_length in layout.json)."""
        s = _f32(np.atleast_2d(sources))
        self._lib.alfa_set_grain_bank(self._h, _ptr(s), s.shape[0], s.shape[1])

    def set_sample(self, data, sample_rate, root_hz=261.6255653, loop=None):
        d = _f32(np.ravel(data))
        start, end = loop if loop is not None else (-1, -1)
        self._lib.alfa_set_sample(self._h, _ptr(d), d.size, float(sample_rate), float(root_hz), int(start), int(end))

    def render(self, voice_frames, instance_frames=None, f0=None, note=60, velocity=1.0, note_off_frame=-1):
        """Returns (left, right), each float32 of len(voice_frames) * hop samples."""
        v = _f32(voice_frames, self.frame_floats)
        i = _f32(instance_frames, self.instance_frame_floats)
        f = _f32(None if f0 is None else np.ravel(f0))
        frames = v.shape[0]
        if i is not None and i.shape[0] != frames or f is not None and f.size != frames:
            raise ValueError("instance_frames and f0 need one entry per voice frame")
        n = frames * self.hop
        left = np.zeros(n, np.float32)
        right = np.zeros(n, np.float32)
        got = self._lib.alfa_render(self._h, _ptr(v), _ptr(i), _ptr(f), frames, int(note), float(velocity),
                                    int(note_off_frame), _ptr(left), _ptr(right))
        if got != n:
            raise RuntimeError("alfa_render failed")
        return left, right
