"""Model size tiers (docs/tiers/model-tiers.md v3, packaging plan section 3)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TierInfo:
    name: str
    label: str
    params: int          # total parameters
    base_install: bool   # shipped in the base installer; the others are optional add-ons
    shipped_mb: int      # approximate size on disk (packaging plan section 3)


TIERS = {
    "lite": TierInfo("lite", "Lite", 4_000_000, True, 2),
    "standard": TierInfo("standard", "Standard", 16_000_000, True, 24),
    "pro": TierInfo("pro", "Pro", 64_000_000, False, 80),
    "studio": TierInfo("studio", "Studio", 256_000_000, False, 320),
}

DEFAULT_TIER = "standard"

# Training stages from training/METHOD.md section 5.
STAGES = ["pretrain", "polish", "finetune", "distill"]

# Graphs a training export must contain (METHOD.md section 7).
METHODS = ["subtractive", "wavetable", "additive", "spectral", "fm", "physical", "granular",
           "sampling"]
REQUIRED_GRAPHS = (["patch_encoder", "instance_step", "voice_core"]
                   + [f"expert_{m}" for m in METHODS] + ["expert_hybrid"])


def check_tier(name: str) -> str:
    if name not in TIERS:
        raise ValueError(f"unknown tier {name!r}; expected one of {', '.join(TIERS)}")
    return name
