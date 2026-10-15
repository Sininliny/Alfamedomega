"""Wave feature descriptors measure what they claim (numpy; the loss test needs PyTorch)."""
import unittest
import numpy as np
from training.amo import wave_features as wf

SR = 22050
T = np.arange(int(SR * 3.0)) / SR
NOTE = [wf.Segment(0.0, 1.5)]


def note(signal, release=0.05, stereo=None):
    env = np.where(T < 1.5, 1.0, np.exp(-(T - 1.5) / release))
    left = signal * env
    right = left if stereo is None else stereo * env
    y = np.stack([left, right])
    return y / np.abs(y).max()


def saw(f0, detune=1.0):
    return sum(np.sin(2 * np.pi * f0 * detune * k * T) / k for k in range(1, 80) if f0 * k < 10000)


def measure(y, segments=NOTE):
    values, valid = wf.measure(y, SR, segments)
    return dict(zip(wf.DESCRIPTORS, values)), dict(zip(wf.DESCRIPTORS, valid))


class DescriptorTests(unittest.TestCase):
    def test_tilt_reads_spectral_slope(self):
        self.assertAlmostEqual(measure(note(saw(110)))[0]["tilt"], -6.0, delta=1.5)
        noise = np.random.default_rng(0).standard_normal(T.size)
        self.assertAlmostEqual(measure(note(noise))[0]["tilt"], 0.0, delta=1.0)

    def test_attack_sustain_and_tail_follow_the_envelope(self):
        fast = measure(note(np.sin(2 * np.pi * 440 * T) * np.minimum(T / 0.002, 1)))[0]
        slow = measure(note(np.sin(2 * np.pi * 440 * T) * np.minimum(T / 0.3, 1)))[0]
        self.assertGreater(fast["attack"], slow["attack"] + 1)
        held = measure(note(saw(220)))[0]
        decaying = measure(note(saw(220) * np.exp(-T / 0.2)))[0]
        self.assertGreater(held["sustain"], decaying["sustain"] + 10)
        short = measure(note(saw(220), release=0.02))[0]
        long = measure(note(saw(220), release=0.4))[0]
        self.assertGreater(long["tail"], short["tail"] + 0.5)

    def test_noise_roughness_and_motion_are_told_apart(self):
        rng = np.random.default_rng(1)
        clean = measure(note(saw(110)))[0]
        noisy = measure(note(saw(110) / 3 + 0.3 * rng.standard_normal(T.size)))[0]
        beating = measure(note(np.sin(2 * np.pi * 440 * T) + np.sin(2 * np.pi * 470 * T)))[0]
        tremolo = measure(note(np.sin(2 * np.pi * 440 * T) * (1 + 0.8 * np.sin(2 * np.pi * 4 * T))))[0]
        self.assertGreater(noisy["noisiness"], clean["noisiness"] + 0.02)
        self.assertLess(tremolo["noisiness"], 0.02)            # amplitude motion is not noise
        self.assertGreater(beating["roughness"], 0.05)         # 30 Hz beating is roughness
        self.assertLess(tremolo["roughness"], 0.02)            # 4 Hz is motion, not roughness
        self.assertGreater(tremolo["motion"], clean["motion"] + 2)

    def test_width_and_validity(self):
        rng = np.random.default_rng(2)
        mono = measure(note(saw(220)))
        wide = measure(note(saw(220), stereo=saw(220, 1.004) + 0.2 * rng.standard_normal(T.size)))
        self.assertGreater(wide[0]["width"], mono[0]["width"] + 20)
        values, valid = wf.measure(note(saw(220))[0], SR, [wf.Segment(0.0, None)])
        self.assertFalse(valid[wf.DESCRIPTORS.index("width")])   # one channel
        self.assertFalse(valid[wf.DESCRIPTORS.index("tail")])    # never released

    def test_segments_from_audio(self):
        y = note(saw(220))
        segments = wf.detect_segments(y, SR)
        self.assertEqual(len(segments), 1)
        self.assertLess(segments[0].onset, 0.01)
        self.assertGreater(segments[0].offset, 1.5)            # gate closes during the release
        self.assertLess(segments[0].offset, 1.9)


class KnobMathTests(unittest.TestCase):
    def test_calibration_and_targets_saturate(self):
        rng = np.random.default_rng(3)
        calibration = wf.Calibration.fit(rng.standard_normal((200, 8)))
        p = calibration.percentile(np.zeros(8))
        np.testing.assert_allclose(p, 0.5, atol=0.08)
        restored = wf.Calibration.from_json(calibration.to_json())
        np.testing.assert_allclose(restored.percentile(np.ones(8)), calibration.percentile(np.ones(8)))
        target = wf.knob_target(np.array([0.9] * 8), np.array([1, -1, 0, 0, 0, 0, 0, 0.5]))
        self.assertEqual(target[0], 1.0)
        self.assertAlmostEqual(target[1], 0.55)
        with self.assertRaises(ValueError):
            wf.knob_target(np.zeros(8), np.full(8, 2.0))

    def test_teacher_moves_its_descriptor_by_one_at_least_cost(self):
        rng = np.random.default_rng(4)
        K, r = 12, 3
        Jd = rng.standard_normal((8, K))
        U, S = np.linalg.qr(rng.standard_normal((K, r)))[0], np.array([3.0, 2.0, 1.0])
        lam = np.full(K, 0.1)
        free = np.ones(K, dtype=bool)
        free[0] = False
        directions, active = wf.teacher_directions(Jd, U, S, lam, free)
        self.assertTrue(active.all())
        np.testing.assert_allclose(np.diag(Jd @ directions), 1.0, atol=1e-9)
        np.testing.assert_array_equal(directions[0], 0.0)       # locked coordinate untouched
        M = np.diag(lam) + U @ np.diag(S ** 2) @ U.T
        cost = directions[:, 2] @ M @ directions[:, 2]
        for _ in range(20):                                       # any other unit move costs more
            other = directions[:, 2] + rng.standard_normal(K) * 0.05
            other[0] = 0
            other /= Jd[2] @ other
            self.assertGreaterEqual(other @ M @ other, cost - 1e-9)
        held, _ = wf.teacher_directions(Jd, U, S, lam, free, hold_others=1e6)
        cross = np.delete(Jd @ held[:, 2], 2)
        self.assertLess(np.abs(cross).max(), 1e-3)
        Jd[5] = 0
        _, active = wf.teacher_directions(Jd, U, S, lam, free)
        self.assertFalse(active[5])

    def test_fallback_block_matches_runtime_contract(self):
        directions = np.ones((3, 8))
        block = wf.fallback_matrix(directions, np.array([True] * 7 + [False]), np.array([1.0, 2.0, 3.0]))
        self.assertEqual(block["schema"], "amo.wave-features/1")
        self.assertEqual(block["matrix"]["shape"], [3, 8])
        matrix = np.array(block["matrix"]["values"]).reshape(3, 8)
        self.assertEqual(matrix[2, 0], 3 * wf.DEFAULT_GAIN)
        self.assertEqual(matrix[0, 7], 0.0)

    def test_knob_curves_stay_in_range(self):
        curves = wf.sample_knob_curves(np.random.default_rng(5), 400, move_probability=1.0)
        self.assertEqual(curves.shape, (400, 8))
        self.assertLessEqual(np.abs(curves).max(), 1.0)
        self.assertTrue((np.abs(curves).max(axis=0) > 0).all())

    def test_losses_reward_the_main_direction(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        base = torch.full((1, 8), 0.5)
        knobs = torch.zeros(1, 8)
        knobs[0, 0] = 1.0
        valid = torch.ones(1, 8, dtype=torch.bool)
        good = base.clone()
        good[0, 0] = 0.85
        good[0, 3] = 0.6                     # others may follow the main move
        bad = base.clone()
        bad[0, 0] = 0.55
        bad[0, 3] = 0.95                     # another descriptor moved more than the turned one
        g = wf.knob_losses(base, good, knobs, valid)
        b = wf.knob_losses(base, bad, knobs, valid)
        self.assertLess(g["total"].item(), 1e-6)
        self.assertGreater(b["dominance"].item(), 0)
        backwards = base.clone()
        backwards[0, 0] = 0.3
        self.assertGreater(wf.knob_losses(base, backwards, knobs, valid)["monotone"].item(), 0)


if __name__ == "__main__":
    unittest.main()
