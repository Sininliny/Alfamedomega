"""Lifecycle regressions: exact resume, publication isolation and held-out groups."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch

from training.aom.learner import train, LearnSpec
from training.aom.session import TrainingSession, TrainingCancelled, atomic_write, atomic_json
from training.aom.workflow import preflight, train_local, acceptance
from training.tests import test_aom_contracts


def fixture():
    torch.set_num_threads(1)
    value = test_aom_contracts.TrainingContractTests()
    value.setUp()
    examples = tuple(replace(value.example, example_id=f"example-{i}") for i in range(3))
    return value, examples, [[1., 0.], [0., 1.], [.5, .5]]


def test_minibatch_resume_restores_adam_and_rng_exactly(tmp_path):
    f, examples, features = fixture()
    spec = LearnSpec(6, 12, .01, 2., 1., .1, 10_000_000, 7, batch_size=2)
    complete, report = train(examples, features, f.canonical.ontology, f.schemas, spec)
    checkpoint = tmp_path / "checkpoint.pt"
    train(examples, features, f.canonical.ontology, f.schemas, replace(spec, steps=5),
          session=TrainingSession("producer", checkpoint=checkpoint, every=2))
    resumed, resumed_report = train(examples, features, f.canonical.ontology, f.schemas, spec,
          session=TrainingSession("producer", checkpoint=checkpoint, resume=True))
    assert report["losses"] == resumed_report["losses"]
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in complete.state_dict().items())
    with pytest.raises(ValueError, match="identity differs"):
        train(examples, [[2., 0.], *features[1:]], f.canonical.ontology, f.schemas, spec,
              session=TrainingSession("producer", checkpoint=checkpoint, resume=True))
    with pytest.raises(ValueError, match="identity differs"):
        train(examples, features, f.canonical.ontology, dict(reversed(list(f.schemas.items()))), spec,
              session=TrainingSession("producer", checkpoint=checkpoint, resume=True))


def test_cancellation_saves_last_complete_step_and_can_resume(tmp_path):
    f, examples, features = fixture()
    checkpoint, stop = tmp_path / "checkpoint.pt", [False]
    spec = LearnSpec(6, 8, .01, 2., 1., .1, 10_000_000, 7)
    session = TrainingSession("producer", checkpoint=checkpoint, every=100,
                              cancelled=lambda: stop[0], progress=lambda step, *_: stop.__setitem__(0, step == 3))
    with pytest.raises(TrainingCancelled):
        train(examples, features, f.canonical.ontology, f.schemas, spec, session=session)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["step"] == 3 and len(saved["history"]) == 3
    resumed, report = train(examples, features, f.canonical.ontology, f.schemas, spec,
                           session=TrainingSession("producer", checkpoint=checkpoint, resume=True))
    uninterrupted, reference = train(examples, features, f.canonical.ontology, f.schemas, spec)
    assert report["losses"] == reference["losses"]


def test_minibatched_validation_matches_full_masked_objective():
    from training.aom.learner import masked_loss
    from training.aom.contracts import dual_targets
    f, examples, features = fixture()
    held = tuple(replace(f.example, example_id=f"held-{i}", canonical_id="held-group", canonical=f.instance) for i in range(7))
    held_features = [[i / 7, 1 - i / 7] for i in range(7)]
    spec = LearnSpec(6, 2, .01, 2., 1., .1, 10_000_000, 7, batch_size=2)
    model, report = train(examples, features, f.canonical.ontology, f.schemas, spec, validation=(held, held_features))
    target = dual_targets(held[0], f.canonical.ontology, f.schemas)
    arrays = []
    for head in ("primary_canonical", "secondary_rendered_instance"):
        values = torch.tensor([[v for name in f.schemas for v in target[head][name].tensor.values]] * 7)
        masks = torch.tensor([[float(v) for name in f.schemas for v in target[head][name].mask]] * 7)
        scales = report["target_normalization"][head]
        arrays.extend(((values - torch.tensor(scales["mean"])) / torch.tensor(scales["scale"]), masks))
    with torch.no_grad():
        p, r, pm, rm = model(torch.tensor(held_features))
        primary = masked_loss(p, arrays[0], arrays[1])
        secondary = masked_loss(r, arrays[2], arrays[3])
        mask = torch.nn.functional.binary_cross_entropy_with_logits(pm, arrays[1]) + torch.nn.functional.binary_cross_entropy_with_logits(rm, arrays[3])
    actual = report["losses"][-1]
    assert actual["validation_primary"] == pytest.approx(float(primary), rel=1e-6)
    assert actual["validation_secondary"] == pytest.approx(float(secondary), rel=1e-6)
    assert actual["validation_mask"] == pytest.approx(float(mask), rel=1e-6)


def test_atomic_write_preserves_published_artifact_on_failure(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"published model")
    def failing(stream):
        stream.write(b"partial candidate")
        raise RuntimeError("disk failure")
    with pytest.raises(RuntimeError):
        atomic_write(model, failing)
    assert model.read_bytes() == b"published model"
    assert list(tmp_path.iterdir()) == [model]


def preference_manifest():
    return {"schema": "aom.training-manifest/1", "category": "user_preference",
            "feature_widths": {"state": 1}, "hidden_width": 4, "steps": 6,
            "learning_rate": .02, "available_ram_bytes": 10_000_000, "seed": 2,
            "require_validation": True,
            "observations": [{"group_id": "train", "first": {"state": [.8]}, "second": {"state": [.2]}, "preferred_first": True},
                             {"group_id": "held", "split": "validation", "first": {"state": [.7]}, "second": {"state": [.3]}, "preferred_first": True}]}


def test_local_training_has_separate_package_validation_and_quality_gate(tmp_path):
    path, output, checkpoint = tmp_path / "manifest.json", tmp_path / "model.json", tmp_path / "checkpoint.pt"
    manifest = preference_manifest()
    atomic_json(path, manifest)
    assert preflight(path, "user_preference")["validation_observations"] == 1
    with pytest.raises(ValueError, match="overwrite a training source"):
        train_local(path, path, "user_preference")
    report = train_local(path, output, "user_preference", session=TrainingSession("local", checkpoint=checkpoint))
    assert "validation_total" in report["metrics"][-1]
    assert json.loads(output.read_text())["category"] == "user_preference"
    published = json.loads(output.read_text())["training"]
    assert "losses" not in published and "metrics" not in published
    assert published["training_summary"]["completed_steps"] == 6
    original = output.read_bytes()
    manifest["steps"] = 9
    manifest["acceptance"] = {"validation_total_max": 0.0}
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="Publication refused"):
        train_local(path, output, "user_preference", session=TrainingSession("local", checkpoint=checkpoint, resume=True))
    assert output.read_bytes() == original
    assert torch.load(checkpoint, weights_only=True)["step"] == 9
    manifest["observations"][1]["group_id"] = "train"
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="leaks"):
        preflight(path, "user_preference")


def test_quality_claim_requires_held_out_metrics():
    with pytest.raises(ValueError, match="held-out"):
        acceptance({"acceptance": {"total_max": 1.0}}, {"total": .1})
    with pytest.raises(ValueError, match="held-out"):
        acceptance({"acceptance": {"validation_total_max": 1.0}}, {"total": .1})


def test_declared_performance_trajectories_preserve_order_and_memory_budget():
    from training.aom.producer import performance, INPUTS
    values = [[200., .2], [400., .5], [100., .1]]
    spec = {"frames": 3, "sample_rate": 16000, "inputs": {k: [] for k in INPUTS}}
    spec["inputs"]["midi"] = {"frames": values}
    arrays = performance(spec, 10000)
    assert arrays["midi"].shape == (3, 2)
    assert arrays["midi"][1].tolist() == pytest.approx([400., .5])
    assert arrays["audio"].shape == (3, 0)
    with pytest.raises(ValueError, match="allowance"):
        performance(spec, 1)
    spec["inputs"]["midi"]["frames"] = values[:2]
    with pytest.raises(ValueError, match="per frame"):
        performance(spec, 10000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Explicit CUDA device unavailable")
def test_small_explicit_cuda_training_and_resume(tmp_path):
    f, examples, features = fixture()
    spec = LearnSpec(6, 4, .01, 2., 1., .1, 10_000_000, 7, batch_size=2,
                     device="cuda:0", available_vram_bytes=10_000_000)
    checkpoint = tmp_path / "gpu.pt"
    train(examples, features, f.canonical.ontology, f.schemas, replace(spec, steps=2),
          session=TrainingSession("gpu", checkpoint=checkpoint))
    resumed, report = train(examples, features, f.canonical.ontology, f.schemas, spec,
                           session=TrainingSession("gpu", checkpoint=checkpoint, resume=True))
    complete, expected = train(examples, features, f.canonical.ontology, f.schemas, spec)
    assert report["losses"] == expected["losses"]
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in complete.state_dict().items())
    assert next(resumed.parameters()).device.type == "cpu"


def test_runtime_manifest_validation_resume_and_native_export(tmp_path):
    from training.aom.pipeline import train_runtime_manifest
    from training.aom.runtime_model import INPUT_CATEGORIES
    widths = {k: 0 for k in INPUT_CATEGORIES}
    widths["midi"] = 1
    groups = []
    for index, value in enumerate((.2, .8)):
        atomic_json(tmp_path / f"target-{index}.json", {"ontology": "test", "tensors": {
            "parameters": {"axes": ["parameter"], "shape": [1], "values": [value]}}})
        inputs = {k: [] for k in INPUT_CATEGORIES}
        inputs["midi"] = [value]
        groups.append({"canonical_id": f"group-{index}", "split": "train" if index == 0 else "validation",
                       "events": [{"inputs": inputs, "target_json": f"target-{index}.json"}]})
    manifest = {"schema": "aom.training-manifest/1", "category": "runtime", "ontology": "test",
                "input_widths": widths, "hidden_width": 4, "schemas": {"parameters": {"axes": ["parameter"], "padded_shape": [1]}},
                "steps": 3, "learning_rate": .02, "mask_weight": .1, "available_ram_bytes": 10_000_000,
                "clock": {"period_samples": 7, "phase_samples": 2}, "mask_threshold": .5,
                "bounds": {"parameters": [[0., 1.]]}, "seed": 7, "sequences": groups, "require_validation": True}
    path, checkpoint = tmp_path / "events.json", tmp_path / "runtime.pt"
    atomic_json(path, manifest)
    report = train_runtime_manifest(path, tmp_path / "runtime.json", session=TrainingSession("runtime", checkpoint=checkpoint))
    assert "validation_total" in report["metrics"][-1]
    manifest["steps"] = 6
    atomic_json(path, manifest)
    resumed = train_runtime_manifest(path, tmp_path / "resumed.json", session=TrainingSession("runtime", checkpoint=checkpoint, resume=True))
    expected = train_runtime_manifest(path, tmp_path / "full.json")
    assert resumed["losses"] == expected["losses"]
    first, second = json.loads((tmp_path / "resumed.json").read_text()), json.loads((tmp_path / "full.json").read_text())
    assert first["weights"] == second["weights"] and first["golden"] == second["golden"]
    assert first["clock"] == {"period_samples": 7, "phase_samples": 2}
    assert "losses" not in first["training"] and "sources" not in first["training"]["dataset"]


def test_native_producer_preflight_training_and_reconstruction(tmp_path):
    from training.aom.native import _library_path
    from training.aom.smoke import fixture as native_fixture
    from training.aom.producer import build_dataset, evaluate_wav, INPUTS
    from training.aom.pipeline import train_manifest
    from training.aom.dataset import prepare_wav
    try:
        library = _library_path()
    except FileNotFoundError:
        pytest.skip("Native compiler build required for producer integration")
    template = native_fixture(.15)
    template["tensors"]["parameters"]["values"][1] = .35
    schemas = {k: {"axes": t["axes"], "padded_shape": t["shape"]} for k, t in template["tensors"].items()}
    bounds = {"activation": [[0., 1.], [0., 1.]], "parameters": [[200., 200.], [.01, .5], [400., 400.], [.01, .5]], "routing": [], "modulation": []}
    groups = []
    for index, amplitude in enumerate((.15, .25)):
        document = native_fixture(amplitude)
        document["tensors"]["parameters"]["values"][1] = .35
        atomic_json(tmp_path / f"source-{index}.json", document)
        groups.append({"id": f"group-{index}", "split": "train" if index == 0 else "validation",
                       "canonical_json": f"source-{index}.json", "variants": [{"id": f"example-{index}", "perturbation_id": "explicit-amplitude",
                       "updates": {"parameters": [{"index": 3, "value": amplitude + .02}]},
                       "render": {"frames": 800, "sample_rate": 16000, "inputs": {k: [] for k in INPUTS}}}]})
    plan = {"schema": "aom.dataset-plan/1", "ontology": "aom.math/1", "groups": groups,
            "bounds": bounds, "schemas": schemas,
            "analysis": {"frames": 2, "harmonic_ratios": [1., 2.], "minimum_fundamental_hz": 100., "maximum_fundamental_hz": 300., "recording_policy": "controlled_render"},
            "normalization": {"policy_id": "controlled-test", "removed": ["pitch", "loudness", "recording_condition"], "preserved": ["timbre", "articulation", "temporal_behavior"]},
            "learn": {"hidden_width": 5, "steps": 80, "learning_rate": .01, "primary_weight": 2., "secondary_weight": 1., "mask_weight": .1, "available_ram_bytes": 10_000_000, "seed": 7},
            "reconstruction_mask_threshold": .5}
    atomic_json(tmp_path / "plan.json", plan)
    result = build_dataset(tmp_path / "plan.json", tmp_path / "dataset", dll_path=library)
    manifest = Path(result["manifest"])
    _, _, split, audit, _ = prepare_wav(manifest)
    assert audit["validation_examples"] == 1
    assert not hasattr(split["train"][0][0], "normalized_audio")
    output = tmp_path / "model.pt"
    metadata = train_manifest(manifest, output)
    assert "validation_total" in metadata["losses"][-1]
    report = evaluate_wav(output, manifest, dll_path=library)
    assert len(report["metrics"]) == 2
    assert all(v >= 0 for v in report["metrics"].values())
    # The optional DSP objective supplies a real native gradient, then participates
    # in both supervision heads, held-out metrics and reproducible resume.
    from training.aom.reconstruction import NativeReconstruction, _NativeLoss
    document = json.loads(manifest.read_text())
    document["reconstruction"] = {"weight": .1, "primary_weight": 2., "secondary_weight": 1.,
                                   "finite_difference_step": .0001, "parameter_indices": [3]}
    from training.aom.contracts import TensorSchema
    schema_objects = {k: TensorSchema(tuple(v["axes"]), tuple(v["padded_shape"])) for k, v in schemas.items()}
    objective = NativeReconstruction(manifest, document, schema_objects, available_ram_bytes=10_000_000)
    values = torch.tensor([[200., .35, 400., .2]], requires_grad=True)
    native_loss = _NativeLoss.apply(values, objective, objective.samples[False], 0, True)
    native_loss.backward()
    expected_gradient = 2 * (.2 - .15) / (.35 ** 2 + .15 ** 2)
    assert values.grad[0, 3].item() == pytest.approx(expected_gradient, rel=.003)
    assert values.grad[0, :3].tolist() == [0., 0., 0.]
    document["learn"]["steps"] = 3
    atomic_json(manifest, document)
    checkpoint = tmp_path / "native-checkpoint.pt"
    native_report = train_manifest(manifest, tmp_path / "native-model.pt",
                                  session=TrainingSession("native", checkpoint=checkpoint))
    assert "reconstruction" in native_report["losses"][-1]
    assert "validation_reconstruction" in native_report["losses"][-1]
    document["learn"]["steps"] = 5
    atomic_json(manifest, document)
    resumed = train_manifest(manifest, tmp_path / "native-resumed.pt",
                            session=TrainingSession("native", checkpoint=checkpoint, resume=True))
    uninterrupted = train_manifest(manifest, tmp_path / "native-full.pt")
    assert resumed["losses"] == uninterrupted["losses"]
    previous = (tmp_path / "native-resumed.pt").read_bytes()
    document["reconstruction_acceptance"] = {"validation_primary_canonical_relative_rmse_max": 0.0}
    atomic_json(manifest, document)
    with pytest.raises(ValueError, match="Publication refused|Reconstructed|Active primitive"):
        train_manifest(manifest, tmp_path / "native-resumed.pt",
                       session=TrainingSession("native", checkpoint=checkpoint, resume=True))
    assert (tmp_path / "native-resumed.pt").read_bytes() == previous
    document = json.loads(manifest.read_text())
    original_path = manifest.parent / document["examples"][0]["canonical_json"]
    alias = json.loads(original_path.read_text())
    alias["tensors"] = dict(reversed(list(alias["tensors"].items())))
    for tensor in alias["tensors"].values():
        tensor["presence"] = [True] * len(tensor["values"])
    atomic_json(manifest.parent / "renamed-original.json", alias)
    previous_original = document["examples"][1]["canonical_json"]
    document["examples"][1]["canonical_json"] = "renamed-original.json"
    atomic_json(manifest, document)
    with pytest.raises(ValueError, match="leak"):
        preflight(manifest, "wav_preset_learning")
    document["examples"][1]["canonical_json"] = previous_original
    document["examples"][1]["canonical_id"] = document["examples"][0]["canonical_id"]
    atomic_json(manifest, document)
    with pytest.raises(ValueError, match="conflicting|leak"):
        preflight(manifest, "wav_preset_learning")
