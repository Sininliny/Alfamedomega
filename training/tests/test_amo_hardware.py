import unittest
from training.amo.hardware import HardwareCapacity, detect


class HardwareTests(unittest.TestCase):
    def test_independent_explicit_reserve_and_unavailable_memory_rejection(self):
        hardware = HardwareCapacity(8, 1000, 500, 0)
        self.assertEqual(hardware.allowance(reserve_ram_bytes=100, reserve_vram_bytes=50),
                         {"available_ram_bytes": 900, "available_vram_bytes": 450})
        with self.assertRaises(ValueError):
            hardware.allowance(reserve_ram_bytes=1001, reserve_vram_bytes=0)

    def test_local_cpu_memory_probe_does_not_assume_gpu(self):
        hardware = detect(gpu_device=None)
        self.assertGreater(hardware.available_ram_bytes, 0)
        self.assertGreater(hardware.cpu_cores, 0)
        self.assertEqual(hardware.available_vram_bytes, 0)


if __name__ == "__main__":
    unittest.main()
