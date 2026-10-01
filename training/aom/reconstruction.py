"""Optional native DSP loss for explicitly selected continuous parameter coordinates.

Topology/presence remain supervised by the tensor heads. Native finite differences
train declared unlocked parameters against original and instance renderings; this
does not pretend the discrete compiler is differentiable.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch

from .native import NativeRuntime
from .producer import performance
from .contracts import ModelBudget
from .session import fingerprint, TrainingCancelled


class NativeReconstruction:
    def __init__(self, manifest_path, manifest, schemas, *, available_ram_bytes, cancelled=None):
        self.config = manifest["reconstruction"]
        required = ("weight", "primary_weight", "secondary_weight", "finite_difference_step")
        if any(not math.isfinite(self.config[k]) or self.config[k] <= 0 for k in required):
            raise ValueError("Explicit positive native reconstruction weights and difference step required")
        self.indices = self.config["parameter_indices"]
        self.width = math.prod(schemas["parameters"].padded_shape)
        if not self.indices or len(set(self.indices)) != len(self.indices) or any(type(i) is not int or not 0 <= i < self.width for i in self.indices):
            raise ValueError("Explicit unique continuous parameter coordinates required")
        self.offset = 0
        for name, schema in schemas.items():
            if name == "parameters":
                break
            self.offset += math.prod(schema.padded_shape)
        self.bounds = manifest["reconstruction_bounds"]["parameters"]
        if len(self.bounds) != self.width or any(not math.isfinite(v) for pair in self.bounds for v in pair) or any(len(pair) != 2 or pair[0] > pair[1] for pair in self.bounds):
            raise ValueError("Explicit finite parameter bounds matching native templates required")
        self.cancelled = cancelled or (lambda: False)
        self.samples = {False: [], True: []}
        self.estimated_peak_bytes = 0
        base = Path(manifest_path).parent
        for row in manifest["examples"]:
            render = row["render"]
            inputs = performance(render, available_ram_bytes)
            sample = {"render": render, "inputs": inputs, "heads": []}
            for key in ("canonical_json", "instance_json"):
                self.check_cancel()
                document = json.loads((base / row[key]).read_text(encoding="utf-8"))
                parameters = document["tensors"]["parameters"]
                if parameters["axes"] != list(schemas["parameters"].axes) or parameters["shape"] != list(schemas["parameters"].padded_shape):
                    raise ValueError("Native loss needs parameter schemas matching each explicit structural template")
                if any(document.get("parameter_locks", [False] * self.width)[i] for i in self.indices):
                    raise ValueError("Native reconstruction cannot optimize locked parameters")
                if parameters.get("presence") and any(not parameters["presence"][i] for i in self.indices):
                    raise ValueError("Native reconstruction cannot optimize absent parameters")
                with NativeRuntime(document, render["sample_rate"]) as runtime:
                    self.estimated_peak_bytes += render["frames"] * (24 * runtime.output_count + 96 + 4 * sum(v.shape[1] for v in inputs.values()))
                    ModelBudget(self.estimated_peak_bytes, available_ram_bytes, 0, 0)
                    expected = runtime.render(frames=render["frames"], **inputs).astype(np.float64)
                power = float(np.mean(expected ** 2)) if expected.size else 0.
                if not math.isfinite(power) or power <= 1e-24:
                    raise ValueError("Non-silent finite native reference required for reconstruction loss")
                sample["heads"].append((document, expected, power))
            self.samples[row.get("split", "train") == "validation"].append(sample)
        self.identity = fingerprint({"config": self.config, "bounds": self.bounds,
            "samples": [{"render": s["render"], "heads": [d for d, _, _ in s["heads"]]} for split in self.samples.values() for s in split]})

    def check_cancel(self):
        if self.cancelled():
            raise TrainingCancelled("Cancelled during native reconstruction; last checkpoint retained")

    def loss(self, values, sample, head, *, check_cancel=True):
        if check_cancel:
            self.check_cancel()
        template, expected, power = sample["heads"][head]
        document = copy.deepcopy(template)
        for index in self.indices:
            low, high = self.bounds[index]
            document["tensors"]["parameters"]["values"][index] = float(np.clip(values[index], low, high))
        with NativeRuntime(document, sample["render"]["sample_rate"]) as runtime:
            actual = runtime.render(frames=sample["render"]["frames"], **sample["inputs"])
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError("Native reconstruction candidate is invalid")
        return float(np.mean((actual.astype(np.float64) - expected) ** 2) / power)

    def __call__(self, primary, secondary, indices, *, validation=False):
        samples = [self.samples[validation][i] for i in indices.tolist()]
        p = primary[:, self.offset:self.offset + self.width]
        r = secondary[:, self.offset:self.offset + self.width]
        return (self.config["primary_weight"] * _NativeLoss.apply(p, self, samples, 0, not validation)
                + self.config["secondary_weight"] * _NativeLoss.apply(r, self, samples, 1, not validation))


class _NativeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, objective, samples, head, check_cancel):
        array = values.detach().cpu().double().numpy().copy()
        ctx.objective, ctx.samples, ctx.head, ctx.array = objective, samples, head, array
        ctx.device, ctx.dtype = values.device, values.dtype
        losses = [objective.loss(row, sample, head, check_cancel=check_cancel) for row, sample in zip(array, samples)]
        return values.new_tensor(sum(losses) / len(losses))

    @staticmethod
    def backward(ctx, gradient):
        result = np.zeros_like(ctx.array)
        epsilon = ctx.objective.config["finite_difference_step"]
        for row_index, (row, sample) in enumerate(zip(ctx.array, ctx.samples)):
            for index in ctx.objective.indices:
                plus, minus = row.copy(), row.copy()
                plus[index] += epsilon
                minus[index] -= epsilon
                result[row_index, index] = (ctx.objective.loss(plus, sample, ctx.head) - ctx.objective.loss(minus, sample, ctx.head)) / (2 * epsilon * len(ctx.samples))
        return torch.as_tensor(result, dtype=ctx.dtype, device=ctx.device) * gradient, None, None, None, None
