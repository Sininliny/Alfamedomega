"""Regression tests for the legacy Colab trainer, training/alfamedomega_train.py.

Named test_aom_* so the training suite command runs them:
    python -m unittest discover -s training/tests -p 'test_aom*.py'
They need torch and are skipped without it.
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import alfamedomega_train as at


def _ctrl(tier, B=2, T=5, seed=0):
    """Random positive control frames with the field sizes a model of this tier emits."""
    t = at.TIERS[tier]
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g) + 0.1
    N, M, G, R = t.n_ops, t.n_modes, t.n_grains, t.unison - 1
    return {"mix": r(B, T, 6), "attack_tau": r(B, T), "osc.re": r(B, T, at.N_HARM), "osc.im": r(B, T, at.N_HARM),
            "osc.detune_ratio": r(B, T, R), "osc.detune_level": r(B, T, R), "noise.band": r(B, T, at.N_BANDS),
            "fm.ratio": r(B, T, N), "fm.level": r(B, T, N), "fm.index": r(B, T, N, N),
            "partials.ratio": r(B, T, M), "partials.amp": r(B, T, M), "grain.rate": r(B, T, G),
            "grain.gain": r(B, T, G), "sampler.rate": r(B, T), "sampler.gain": r(B, T), "sampler.eq": r(B, T, at.N_BANDS)}


@unittest.skipIf(torch is None, "torch not installed")
class CrossTierDistillTests(unittest.TestCase):
    def test_student_and_studio_teacher_frames_compare(self):
        # Distillation compares a smaller tier's frames with Studio's, which has more partials,
        # FM operators, grain layers and unison reads. This used to fail with a shape error.
        for student in ("lite", "standard", "pro"):
            d = at.ctrl_distance(_ctrl(student), _ctrl("studio", seed=1))
            self.assertTrue(torch.isfinite(d), student)

    def test_extra_teacher_entries_are_ignored(self):
        student, teacher = _ctrl("pro"), _ctrl("studio")
        for k, v in student.items():          # teacher = student, plus the entries Pro does not have
            if v.dim() >= 3 and v.shape[-1] != teacher[k].shape[-1]:
                w = teacher[k].clone()
                if k == "fm.index":
                    w[..., :v.shape[-2], :v.shape[-1]] = v
                else:
                    w[..., :v.shape[-1]] = v
                teacher[k] = w
            else:
                teacher[k] = v.clone()
        self.assertEqual(float(at.ctrl_distance(student, teacher)), 0.0)


def _train(out, steps, *extra):
    args = at.argparse.Namespace(
        tier="lite", stage="pretrain", steps=steps, batch=2, accum=1, clip=0.12, lr=None, init=None,
        teacher=None, data=None, ft_methods="sampling,physical,additive,subtractive,wavetable", out=out,
        device="cpu", seed=0, log_every=100, ckpt_every=3, audio_every=10 ** 9, no_amp=True, smoke=False,
        resume=False)
    for flag in extra:
        setattr(args, flag, True)
    return at.train(args)


@unittest.skipIf(torch is None, "torch not installed")
class ResumeTests(unittest.TestCase):
    def test_resumed_run_matches_an_uninterrupted_one(self):
        root = tempfile.mkdtemp()
        try:
            straight, interrupted = os.path.join(root, "a"), os.path.join(root, "b")
            full = _train(straight, 6)
            # Keep the step-3 resume state of a run, as if the runtime died before step 6.
            saved = at._save_atomic
            def keep_step_3(obj, path):
                saved(obj, path)
                if path.endswith(".resume.pt") and obj["step"] == 3:
                    shutil.copy(path, path + ".step3")
            at._save_atomic = keep_step_3
            try:
                _train(interrupted, 6)
            finally:
                at._save_atomic = saved
            resume = os.path.join(interrupted, "lite_pretrain.resume.pt")
            os.replace(resume + ".step3", resume)
            resumed = _train(interrupted, 6, "resume")
            self.assertEqual(torch.load(os.path.join(interrupted, "lite_pretrain.pt"), weights_only=False)["step"], 6)
            for (name, a), b in zip(full.state_dict().items(), resumed.state_dict().values()):
                self.assertTrue(torch.allclose(a, b, atol=1e-6), name)
        finally:
            shutil.rmtree(root)

    def test_resume_without_state_fails_clearly(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(FileNotFoundError):
                _train(out, 3, "resume")


@unittest.skipIf(torch is None, "torch not installed")
class FactorySampleTests(unittest.TestCase):
    def test_factory_samples_do_not_depend_on_or_shift_the_data_stream(self):
        a = at.Teachers(at.TIERS["lite"], torch.device("cpu"), 3)
        b = at.Teachers(at.TIERS["lite"], torch.device("cpu"), 3)
        b.batch("additive", 2, 20)                 # b's stream has moved on before samples exist
        a._factory_samples()
        b._factory_samples()
        self.assertTrue(torch.equal(a.samples, b.samples))
        c = at.Teachers(at.TIERS["lite"], torch.device("cpu"), 3)
        c._factory_samples()
        x = at.Teachers(at.TIERS["lite"], torch.device("cpu"), 3)
        self.assertTrue(torch.equal(c.gen.get_state(), x.gen.get_state()))


if __name__ == "__main__":
    unittest.main()
