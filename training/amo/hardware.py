"""Read current hardware capacity; reservations are explicit producer policy."""
from dataclasses import dataclass
import ctypes
import os


@dataclass(frozen=True)
class HardwareCapacity:
    cpu_cores: int
    available_ram_bytes: int
    available_vram_bytes: int
    gpu_device: int | None

    def allowance(self, *, reserve_ram_bytes: int, reserve_vram_bytes: int):
        if any(type(value) is not int or value < 0 for value in (reserve_ram_bytes, reserve_vram_bytes)):
            raise ValueError("Explicit nonnegative host/device reserves required")
        ram = self.available_ram_bytes - reserve_ram_bytes
        vram = self.available_vram_bytes - reserve_vram_bytes
        if ram < 0 or vram < 0:
            raise ValueError("Requested reserve exceeds available hardware memory")
        return {"available_ram_bytes": ram, "available_vram_bytes": vram}


def detect(*, gpu_device: int | None):
    """Snapshot availability without allocating model weights.

    CPU-only use explicitly passes None. A GPU query may initialize its driver
    context; it never assumes that installed CUDA implies usable VRAM.
    """
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_pagefile", ctypes.c_ulonglong), ("available_pagefile", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended_virtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Windows physical memory query failed")
        ram = status.available_physical
    elif os.path.isfile("/proc/meminfo"):
        with open("/proc/meminfo", encoding="ascii") as stream:
            values = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in stream}
        ram = values["MemAvailable"]
    else:
        # Explicit failure is preferable to counting swap or inventing free RAM.
        raise OSError("No supported available-physical-memory query; supply measured allowance")
    vram = 0
    if gpu_device is not None:
        import torch
        if type(gpu_device) is not int or gpu_device < 0 or not torch.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable")
        vram, _ = torch.cuda.mem_get_info(gpu_device)
    return HardwareCapacity(os.cpu_count() or 1, int(ram), int(vram), gpu_device)
