"""Native deterministic tensor execution; no neural waveform generation."""

import ctypes
import json
import os
from pathlib import Path

import numpy as np


def _library_path(path=None):
    if path:
        return Path(path)
    configured = os.environ.get("AOM_TENSOR_LIBRARY")
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[2]
    names = ("alfa_tensor_c.dll", "libalfa_tensor_c.so", "libalfa_tensor_c.dylib")
    for directory in (root / "build/src/tensor/Release", root / "build/src/tensor",
                      root / "build/core/src/tensor", root / "build/release/src/tensor"):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("Build alfa_tensor_c or supply AOM_TENSOR_LIBRARY")


class NativeRuntime:
    def __init__(self, document, sample_rate=48000.0, dll_path=None):
        self._dll = ctypes.CDLL(str(_library_path(dll_path)))
        self._dll.aom_create.argtypes = (ctypes.c_char_p, ctypes.c_double,
                                        ctypes.c_char_p, ctypes.c_size_t)
        self._dll.aom_create.restype = ctypes.c_void_p
        self._dll.aom_destroy.argtypes = (ctypes.c_void_p,)
        self._dll.aom_output_count.argtypes = (ctypes.c_void_p,)
        self._dll.aom_output_count.restype = ctypes.c_size_t
        self._dll.aom_reset.argtypes = (ctypes.c_void_p,)
        pointer = ctypes.POINTER(ctypes.c_float)
        self._dll.aom_render.argtypes = (ctypes.c_void_p, ctypes.c_size_t,
                                        ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t),
                                        pointer, ctypes.c_size_t)
        self._dll.aom_render.restype = ctypes.c_int
        encoded = document if isinstance(document, str) else json.dumps(document, allow_nan=False)
        error = ctypes.create_string_buffer(1024)
        self._handle = self._dll.aom_create(encoded.encode("utf-8"), sample_rate, error, len(error))
        if not self._handle:
            raise ValueError(error.value.decode("utf-8", errors="replace"))
        self.output_count = self._dll.aom_output_count(self._handle)

    def close(self):
        if getattr(self, "_handle", None):
            self._dll.aom_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    def reset(self):
        if not self._handle:
            raise ValueError("Runtime is closed")
        self._dll.aom_reset(self._handle)

    def render(self, audio=None, midi=None, gesture=None, preference=None, latent=None,
               context=None, frames=None, wave_feature=None):
        if not self._handle:
            raise ValueError("Runtime is closed")
        arrays = []
        for value in (audio, midi, gesture, preference, latent, context, wave_feature):
            if value is None:
                arrays.append(None)
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError("Inputs require finite [sample, feature] arrays")
            if frames is None:
                frames = array.shape[0]
            if array.shape[0] != frames:
                raise ValueError("Input sample counts disagree")
            arrays.append(np.ascontiguousarray(array))
        if frames is None or type(frames) is not int or frames < 0:
            raise ValueError("An explicit nonnegative frame count or input array is required")
        widths = (ctypes.c_size_t * 7)(*(0 if value is None else value.shape[1] for value in arrays))
        pointer = ctypes.POINTER(ctypes.c_float)
        pointers = (pointer * 7)(*(pointer() if value is None else value.ctypes.data_as(pointer)
                                  for value in arrays))
        output = np.empty((frames, self.output_count), dtype=np.float32)
        if not self._dll.aom_render(self._handle, frames, pointers, widths,
                                   output.ctypes.data_as(pointer), output.size):
            raise ValueError("Tensor execution rejected input, projection or state update")
        return output


def render_json(document, frames, sample_rate=48000.0, **inputs):
    dll_path = inputs.pop("dll_path", None)
    with NativeRuntime(document, sample_rate, dll_path) as runtime:
        return runtime.render(frames=frames, **inputs)
