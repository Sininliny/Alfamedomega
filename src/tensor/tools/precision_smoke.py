import ctypes
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from training.aom.native import NativeRuntime

program = json.loads((Path(__file__).resolve().parents[3] / 'programs/instrument.aom.json').read_text())
midi = np.tile(np.array([440., .8], dtype=np.float32), (1024, 1))
for tier, scalar, size in [('Standard', 'float32', 4), ('Pro', 'float64', 8)]:
    program['tier'], program['scalar_type'] = tier, scalar
    program['input_capacity'] = [0, 2, 0, 0, 0, 0]
    program['budget']['boundary_conversion_bytes'] = 2 * size
    with NativeRuntime(program) as runtime:
        runtime._dll.aom_scalar_bytes.argtypes = (ctypes.c_void_p,)
        runtime._dll.aom_scalar_bytes.restype = ctypes.c_size_t
        assert runtime._dll.aom_scalar_bytes(runtime._handle) == size
        first = runtime.render(midi=midi)
        assert np.isfinite(first).all() and np.max(np.abs(first)) > .01
        runtime.reset()
        assert np.array_equal(first, runtime.render(midi=midi))
        try:
            runtime.render(midi=np.ones((2, 3), dtype=np.float32))
            raise AssertionError('oversized boundary accepted')
        except ValueError:
            pass
    print(tier, 'typed render/reset/capacity passed')
program['budget']['boundary_conversion_bytes'] = 1
try:
    NativeRuntime(program)
    raise AssertionError('insufficient conversion budget accepted')
except ValueError:
    print('insufficient float64 boundary budget refused')
