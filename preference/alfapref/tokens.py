"""Words from sample and preset paths (design.md s5)."""
import re
from pathlib import PurePath

FILE_WEIGHT = 1.0
FOLDER_WEIGHT = 0.5

FILLER = {
    "sample", "samples", "loop", "loops", "oneshot", "one", "shot", "shots", "wav", "wave", "aif",
    "aiff", "flac", "mp3", "ogg", "final", "copy", "new", "old", "take", "version", "ver", "edit",
    "export", "bounce", "render", "print", "untitled", "audio", "file", "files", "the", "and", "of",
    "a", "an", "my", "kit", "pack", "vol", "part", "mix", "stem", "stems", "wet", "dry", "real",
    "preset", "presets", "patch", "patches", "sound", "sounds", "user", "bpm",
}
KEY_LABEL = "key"            # "Key_Am" is a key label, but "Keys" is the instrument family
NEGATIONS = {"not", "no", "non", "without", "-"}

_KEY = re.compile(r"^[a-g](#|b|sharp|flat)?(m|min|maj|minor|major)?$")
_KEY_OCTAVE = re.compile(r"^[a-g](#|b|sharp|flat)?-?\d+$")   # note with octave: f#1, c-1, eb3
_VERSION = re.compile(r"^v\d+$")
_SPLIT_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_SEP = re.compile(r"[^\w#]+|_")


def normalise(word):
    w = word.lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    return w


def words(text):
    """Split one path component into cleaned words."""
    out = []
    for part in _SEP.split(text):
        for sub in _SPLIT_CAMEL.split(part):
            for w in _SPLIT_DIGIT.split(sub):
                if not w:
                    continue
                lw = w.lower()
                is_key = _KEY.match(lw) and w != "FM"            # "Fm" is a key, "FM" a method
                if lw.isdigit() or is_key or _KEY_OCTAVE.match(lw) or _VERSION.match(lw) or lw.endswith("bpm"):
                    continue
                if len(lw) < 2 and lw != "#":
                    continue
                n = normalise(w)
                if n in FILLER or lw in FILLER or lw == KEY_LABEL or n == "#":
                    continue
                out.append(n)
    return out


def path_words(rel_path):
    """{word: weight} for a path relative to the library root. File-name words weigh 1.0, folder
    words 0.5; a word in both keeps the larger weight."""
    p = PurePath(rel_path)
    tags = {}
    for folder in p.parent.parts:
        for w in words(folder):
            tags[w] = max(tags.get(w, 0.0), FOLDER_WEIGHT)
    for w in words(p.stem):
        tags[w] = max(tags.get(w, 0.0), FILE_WEIGHT)
    return tags


def parse_query(text):
    """'warm airy not dark' -> [('warm', 1), ('airy', 1), ('dark', -1)]."""
    out, sign = [], 1.0
    for raw in re.split(r"[\s,]+", text.strip()):
        if not raw:
            continue
        if raw.lower() in NEGATIONS:
            sign = -1.0
            continue
        if raw.startswith("-") and len(raw) > 1:
            sign, raw = -1.0, raw[1:]
        for w in words(raw):
            out.append((w, sign))
        sign = 1.0
    return out
