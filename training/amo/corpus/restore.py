"""Verify split archives and stream a corpus into the requested local project.

Standalone usage: py restore.py --source PATH_TO_DOWNLOADED_PARTS
Requires Python 3.12+ and zstandard==0.25.0; does not require Torch.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile


class Volumes(io.RawIOBase):
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def restore(source, project):
    import zstandard

    manifest = json.loads(
        (source / "download-manifest.json").read_text(encoding="utf-8")
    )
    paths = []
    for part in manifest["parts"]:
        name = part["name"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Invalid archive volume filename")
        path = source / name
        if path.stat().st_size != part["bytes"]:
            raise ValueError(f"Incomplete volume: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != part["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
        print(f"Verified {name}", flush=True)
        paths.append(path)
    if not paths:
        raise ValueError("Archive has no volumes")
    parent = project / "training" / "data" / "round-expanded-20261014"
    target = parent / "corpus-v2"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < manifest["unpacked_bytes"] + 1024**3:
        raise OSError("Insufficient free space for corpus plus 1 GiB reserve")
    count = 0
    with tempfile.TemporaryDirectory(prefix=".corpus-restore-", dir=parent) as stage:
        with Volumes(paths) as volumes, io.BufferedReader(volumes) as buffered:
            with zstandard.ZstdDecompressor().stream_reader(buffered) as decoded:
                with tarfile.open(fileobj=decoded, mode="r|") as archive:
                    for member in archive:
                        name = PurePosixPath(member.name)
                        if (
                            name.is_absolute()
                            or ".." in name.parts
                            or not name.parts
                            or name.parts[0] != "corpus-v2"
                            or not (member.isfile() or member.isdir())
                        ):
                            raise ValueError(
                                f"Unexpected archive member: {member.name}"
                            )
                        archive.extract(member, stage, filter="data")
                        count += int(member.isfile())
                # Consume the frame trailer so the decoder checks its checksum.
                while decoded.read(8 * 1024 * 1024):
                    pass
        if count != manifest["files"]:
            raise ValueError(f"Expected {manifest['files']} files; extracted {count}")
        (Path(stage) / "corpus-v2").rename(target)
    print(f"Restored {count} files to {target}", flush=True)
    return target


if __name__ == "__main__":
    if sys.version_info < (3, 12):
        raise SystemExit("Use Python 3.12 or newer")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--project", type=Path, default=Path(r"F:\source\code\Alfamedomega")
    )
    arguments = parser.parse_args()
    restore(arguments.source.resolve(), arguments.project.resolve())
