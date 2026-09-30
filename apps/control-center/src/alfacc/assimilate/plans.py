"""What to render: sampling densities, probe notes for range finding, and the note grid."""

from dataclasses import asdict, dataclass, field

C1 = 24  # MIDI note of C1 (C4 = 60); grids are anchored on C so octaves line up across presets
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass(frozen=True)
class Density:
    name: str
    label: str
    step: int                    # semitones between rendered keys
    velocities: tuple
    durations: tuple             # seconds the key is held
    max_tail_s: float            # longest release captured after note-off
    release_analysis: bool       # measure the release tail in detail
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


DENSITIES = {
    "fast": Density("fast", "Fast", 12, (100,), (1.0,), 2.0, False,
                    "every 12 semitones, 1 velocity"),
    "normal": Density("normal", "Normal", 5, (40, 80, 120), (1.0,), 3.0, False,
                      "every 5 semitones, 3 velocities"),
    "detailed": Density("detailed", "Detailed", 1, (24, 48, 72, 96, 112, 127), (1.5,), 4.0, False,
                        "every semitone, 6 velocities"),
    "unreasonable": Density("unreasonable", "Unreasonable", 1, (16, 32, 48, 64, 80, 96, 112, 127),
                            (0.25, 1.0, 4.0), 10.0, True,
                            "every playable key, 8 velocities, 3 note lengths, release tails"),
}


def custom_density(base: str, step: int | None = None, velocities=None, durations=None,
                   max_tail_s: float | None = None) -> Density:
    """A density built from a preset, for the user's own settings (e.g. Normal at 4 or 6)."""
    d = DENSITIES[base]
    return Density(d.name if step is None and velocities is None and durations is None else "custom",
                   d.label, step or d.step, tuple(velocities or d.velocities),
                   tuple(durations or d.durations), max_tail_s or d.max_tail_s, d.release_analysis)


@dataclass
class Note:
    id: str
    midi: int
    velocity: int
    dur_s: float
    tail_s: float
    tail_mode: str = "auto"      # stop early once the release has decayed below -90 dBFS

    def to_dict(self) -> dict:
        return asdict(self)


def note_id(midi: int, velocity: int, dur_s: float) -> str:
    return f"m{midi:03d}_v{velocity:03d}_d{int(round(dur_s * 1000)):05d}"


PROBE_LOW, PROBE_HIGH, PROBE_STEP = 12, 120, 6
PROBE_VELOCITY, PROBE_DUR, PROBE_TAIL = 100, 0.4, 1.0


def probe_notes(lo: int = PROBE_LOW, hi: int = PROBE_HIGH, step: int = PROBE_STEP) -> list:
    """Coarse probes across the MIDI range, to find where a preset makes a usable sound."""
    return [Note(note_id(m, PROBE_VELOCITY, PROBE_DUR), m, PROBE_VELOCITY, PROBE_DUR, PROBE_TAIL)
            for m in range(lo, hi + 1, step)]


def refine_notes(keys) -> list:
    return [Note(note_id(m, PROBE_VELOCITY, PROBE_DUR), m, PROBE_VELOCITY, PROBE_DUR, PROBE_TAIL)
            for m in sorted(set(keys))]


def grid_keys(lo: int, hi: int, step: int) -> list:
    """Keys on a C-anchored grid inside [lo, hi], always including both ends of the range."""
    keys = [m for m in range(lo, hi + 1) if (m - C1) % step == 0]
    return sorted(set(keys) | {lo, hi})


def grid_notes(lo: int, hi: int, density: Density) -> list:
    notes = []
    for m in grid_keys(lo, hi, density.step):
        for v in density.velocities:
            for d in density.durations:
                notes.append(Note(note_id(m, v, d), m, v, d, density.max_tail_s))
    return notes


@dataclass
class Estimate:
    renders: int
    audio_seconds: float

    def text(self) -> str:
        mins = self.audio_seconds / 60
        dur = f"{mins:.0f} min" if mins < 120 else f"{mins / 60:.1f} h"
        return f"{self.renders:,} renders, about {dur} of audio per preset set"


def estimate(presets: int, density: Density, keys: int = 61, settle_s: float = 0.3) -> Estimate:
    """Rough size of a harvest before the ranges are known (61 keys = a 5-octave keyboard).

    Releases usually end well before max_tail_s, so each render counts about 1 s of tail.
    """
    n_keys = len(grid_keys(C1 + 12, C1 + 12 + keys - 1, density.step))
    per_preset = n_keys * len(density.velocities) * len(density.durations)
    tail = min(density.max_tail_s, 1.0 if not density.release_analysis else 3.0)
    seconds = presets * (settle_s + per_preset * (sum(density.durations) / len(density.durations) + tail))
    return Estimate(presets * per_preset, seconds)


@dataclass
class PresetRef:
    key: str                     # stable id inside one plugin: p003, f0012, default
    kind: str                    # program | vstpreset | default
    name: str
    index: int | None = None
    path: str | None = None
    category: str = ""
    extra: dict = field(default_factory=dict)

    def to_job(self) -> dict:
        d = {"key": self.key, "kind": self.kind}
        if self.index is not None:
            d["index"] = self.index
        if self.path:
            d["path"] = self.path
        return d

    def to_dict(self) -> dict:
        return asdict(self)
