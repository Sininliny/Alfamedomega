"""Measured candidate search within one ontology; no blind tensor pruning."""
from dataclasses import dataclass
import copy
import math
import torch
from .schema import same_id


class CannotFitTier(ValueError):
    pass


@dataclass(frozen=True)
class TierPolicy:
    capacity: dict[str, int]
    maximum_errors: dict[str, float]
    objective_weights: dict[str, float]

    def validate(self):
        if not self.capacity or any(type(v) is not int or v < 0 for v in self.capacity.values()):
            raise ValueError("Explicit nonnegative capacity budgets required")
        if not {"pitch", "timing"} <= set(self.maximum_errors):
            raise ValueError("Pitch and timing identity constraints are required")
        if any(not math.isfinite(v) or v < 0 for v in self.maximum_errors.values()):
            raise ValueError("Error tolerances must be finite and nonnegative")
        if not self.objective_weights or any(not math.isfinite(v) or v <= 0 for v in self.objective_weights.values()):
            raise ValueError("Explicit positive objective weights required")


def _changes(reference, candidate):
    if not same_id(reference["ontology"], candidate["ontology"]) or set(reference["tensors"]) != set(candidate["tensors"]):
        raise ValueError("Cross-tier candidates must retain the same ontology")
    result = []
    for name in reference["tensors"]:
        old, new = reference["tensors"][name], candidate["tensors"][name]
        if old["axes"] != new["axes"] or old["shape"] != new["shape"] or len(old["values"]) != len(new["values"]):
            raise ValueError("Cross-tier candidate must retain shared padded tensor coordinates")
        result.extend(abs(float(a) - float(b)) for a, b in zip(old["values"], new["values"]))
    return result


def fit_importance(reference: dict, observations: list[tuple[dict, float]], *, ridge: float):
    """Learn dimension sensitivity from producer experiments, not assumed rankings.

    This linear local estimator is diagnostic. Search acceptance always uses the
    actual caller evaluator, because a sensitivity fit cannot prove preservation.
    """
    if not observations or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Measured observations and explicit positive regularization required")
    x = torch.tensor([_changes(reference, candidate) for candidate, _ in observations], dtype=torch.float64)
    y = torch.tensor([loss for _, loss in observations], dtype=torch.float64)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (y < 0).any():
        raise ValueError("Importance experiments require finite nonnegative measured losses")
    # Dual ridge solve is sized by observation count instead of tensor universe;
    # use the same BLAS backend as learner to avoid duplicate Windows OpenMP DLLs.
    weights = x.T @ torch.linalg.solve(x @ x.T + ridge * torch.eye(x.shape[0], dtype=torch.float64), y)
    return {"ontology": reference["ontology"], "weights": weights.tolist(), "observations": len(observations), "ridge": ridge}


def optimize_candidates(reference: dict, candidates: list[dict], policy: TierPolicy,
                        *, validate_runtime, evaluate, importance=None):
    """Compile and measure explicit candidate configurations under target budget.

    `validate_runtime(document)` must invoke compiler capacity validation.
    `evaluate(reference, document)` returns nonnegative measured errors, including
    pitch and timing. Candidates are producer proposals; no automatic removal or
    rerouting changes an unknown sound's identity.
    """
    policy.validate()
    locks = reference.get("parameter_locks", [])
    best = None
    rejected = []
    ordered = list(enumerate(candidates))
    if importance is not None:
        if not same_id(importance["ontology"], reference["ontology"]):
            raise ValueError("Importance ontology mismatch")
        def predicted(item):
            delta = _changes(reference, item[1])
            if len(delta) != len(importance["weights"]):
                raise ValueError("Importance dimension mismatch")
            return sum(a * b for a, b in zip(delta, importance["weights"]))
        ordered.sort(key=predicted)
    for index, proposal in ordered:
        candidate = copy.deepcopy(proposal)
        try:
            _changes(reference, candidate)
            if locks and any(lock and reference["tensors"]["parameters"]["values"][i] != candidate["tensors"]["parameters"]["values"][i] for i, lock in enumerate(locks)):
                raise ValueError("Candidate modifies a locked parameter")
            candidate["parameter_locks"] = copy.deepcopy(locks)
            candidate["budget"] = copy.deepcopy(policy.capacity)
            validate_runtime(candidate)
            metrics = evaluate(reference, candidate)
            required = set(policy.maximum_errors) | set(policy.objective_weights)
            if not required <= set(metrics):
                raise ValueError("Evaluator omitted required identity/objective metrics")
            if any(not math.isfinite(metrics[key]) or metrics[key] < 0 for key in required):
                raise ValueError("Evaluator errors must be finite and nonnegative")
            if any(metrics[key] > maximum for key, maximum in policy.maximum_errors.items()):
                raise ValueError("Candidate cannot preserve required identity")
            score = sum(metrics[key] * weight for key, weight in policy.objective_weights.items())
            if best is None or score < best["score"]:
                best = {"document": candidate, "score": score, "errors": metrics, "candidate": index}
        except ValueError as error:
            rejected.append({"candidate": index, "reason": str(error)})
    if best is None:
        raise CannotFitTier(f"No proposed configuration fits capacity and identity constraints: {rejected}")
    best["rejected"] = rejected
    return best
