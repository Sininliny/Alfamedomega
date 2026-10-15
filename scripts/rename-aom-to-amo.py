#!/usr/bin/env python3
"""One-off rename of the old project abbreviation to AMO across a git checkout.

Run from the repository root, on a branch that still uses the old spelling
(for example an open PR started before the rename landed), before merging main into it:

    python scripts/rename-aom-to-amo.py          # rewrite tracked files and paths
    python scripts/rename-aom-to-amo.py --check  # list what would change, exit 1 if anything

It renames tracked paths (git mv) and rewrites file contents. Only whole tokens
are replaced (``aom`` not touching other lowercase letters, ``AOM`` not touching
other capitals, ``Aom`` at a word start), so unrelated words are left alone.
Loaders keep accepting the old ids; see docs/audio-tensor-space/mission.md.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

OLD = "a" + "om"  # spelled split so this script survives its own rewrite
PATTERNS = [
    (re.compile(rf"(?<![a-z]){OLD}(?![a-z])"), "amo"),
    (re.compile(rf"(?<![A-Z]){OLD.upper()}(?![A-Z])"), "AMO"),
    (re.compile(rf"(?<![A-Za-z]){OLD.capitalize()}(?![a-z])"), "Amo"),
]
SELF = Path(__file__).resolve()
# Lines carrying this marker keep the old spelling (loaders that still accept it);
# a marker line saying "next paragraph" protects everything up to the next blank line,
# and "rename-compat: whole file" protects the file.
COMPAT = "rename-compat"


def rewrite(text: str) -> str:
    if COMPAT + ": whole file" in text:
        return text
    lines = text.splitlines(keepends=True)
    protected = False
    for i, line in enumerate(lines):
        if not line.strip():
            protected = False
        if COMPAT in line:
            protected = protected or "next paragraph" in line
            continue
        if protected:
            continue
        for pattern, new in PATTERNS:
            line = pattern.sub(new, line)
        lines[i] = line
    return "".join(lines)


def tracked(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [root / p for p in out.decode().split("\0") if p]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only")
    args = parser.parse_args()
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True,
                               capture_output=True, text=True).stdout.strip())
    changed = []
    for path in tracked(root):
        if path.resolve() == SELF or not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new = rewrite(text)
        if new != text:
            changed.append(path.relative_to(root).as_posix())
            if not args.check:
                path.write_bytes(new.encode("utf-8"))
    moves = []
    for path in tracked(root):
        rel = path.relative_to(root).as_posix()
        if path.resolve() == SELF:
            continue
        new_rel = rewrite(rel)
        if new_rel != rel:
            moves.append((rel, new_rel))
    for rel, new_rel in moves:
        if not args.check:
            (root / new_rel).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "mv", rel, new_rel], cwd=root, check=True)
    for rel in changed:
        print("edit", rel)
    for rel, new_rel in moves:
        print("move", rel, "->", new_rel)
    return 1 if args.check and (changed or moves) else 0


if __name__ == "__main__":
    sys.exit(main())
