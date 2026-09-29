"""Library scanning (design.md s3.2, s7): descriptors and path words, never audio or paths.

Only WAV is read by this reference (PCM 8/16/24/32-bit and 32/64-bit float, which covers most
sample libraries). The plugin decodes other formats with JUCE and calls the same descriptor.
"""
import hashlib
import os
import struct

import numpy as np

from . import descriptor as desc_mod
from . import tokens

AUDIO_EXT = {".wav", ".wave"}
OTHER_AUDIO_EXT = {".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a"}


def salted(salt, text):
    return hashlib.sha256((salt + "\0" + text).encode()).hexdigest()


def read_wav(path):
    """-> (float32 mono-or-multichannel array [N] or [N, C], sample rate)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:              # WAVE_FORMAT_EXTENSIBLE
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("missing fmt or data chunk")
    tag, ch, sr, bits = fmt
    if tag == 3 and bits in (32, 64):
        x = np.frombuffer(pcm[: len(pcm) // (bits // 8) * (bits // 8)], dtype="<f%d" % (bits // 8)).astype(np.float32)
    elif tag == 1 and bits == 8:
        x = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif tag == 1 and bits == 16:
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
    elif tag == 1 and bits == 24:
        b = np.frombuffer(pcm[: len(pcm) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / (1 << 23)
    elif tag == 1 and bits == 32:
        x = np.frombuffer(pcm[: len(pcm) // 4 * 4], dtype="<i4").astype(np.float32) / 2 ** 31
    else:
        raise ValueError("unsupported WAV format tag %d, %d bits" % (tag, bits))
    x = x[: len(x) // ch * ch]
    return (x.reshape(-1, ch) if ch > 1 else x), sr


def scan(root, salt, index=None, max_files=None):
    """Scan one library folder. `index` is the previous list of entries (from any roots); entries
    of this root whose file is unchanged are reused, and deleted files drop out.
    Returns (entries for all roots, report dict)."""
    root = os.path.abspath(root)
    root_id = salted(salt, root)[:16]
    prev = {e["key"]: e for e in (index or []) if e["root"] == root_id}
    keep = [e for e in (index or []) if e["root"] != root_id]
    report = {"root": root_id, "added": 0, "unchanged": 0, "skipped_format": 0, "failed": 0}
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext in OTHER_AUDIO_EXT:
                report["skipped_format"] += 1
                continue
            if ext not in AUDIO_EXT:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            key = salted(salt, root_id + "/" + rel)
            st = os.stat(full)
            old = prev.get(key)
            if old and old["size"] == st.st_size and old["mtime"] == int(st.st_mtime) \
                    and old["descriptor_id"] == desc_mod.DESCRIPTOR_ID:
                keep.append(old)
                report["unchanged"] += 1
                continue
            try:
                x, sr = read_wav(full)
                d = desc_mod.descriptor(x, sr)
            except (ValueError, OSError, struct.error):
                report["failed"] += 1
                continue
            keep.append({"key": key, "root": root_id, "size": st.st_size, "mtime": int(st.st_mtime),
                         "descriptor_id": desc_mod.DESCRIPTOR_ID, "desc": [round(float(v), 5) for v in d],
                         "words": tokens.path_words(rel), "weight": 1.0})
            report["added"] += 1
            n += 1
            if max_files and n >= max_files:
                return keep, report
    return keep, report


def from_records(records, salt, progress=None):
    """Build entries from an index written by the control center (docs/control-center/
    preference-batch.md): one dict per file with at least "path", "library" and "rel". Words are
    recomputed from "rel" with tokens.path_words so there is one definition of them. The paths are
    only read here; the entries keep salted keys, as scan() does.
    Returns (entries, report)."""
    out = []
    report = {"added": 0, "skipped_format": 0, "failed": 0}
    for i, r in enumerate(records):
        if progress:
            progress(i)
        path, rel = r.get("path"), r.get("rel") or os.path.basename(r.get("path", ""))
        ext = os.path.splitext(path or "")[1].lower()
        if ext not in AUDIO_EXT:
            report["skipped_format" if ext in OTHER_AUDIO_EXT else "failed"] += 1
            continue
        root_id = salted(salt, r.get("library", ""))[:16]
        try:
            st = os.stat(path)
            x, sr = read_wav(path)
            d = desc_mod.descriptor(x, sr)
        except (ValueError, OSError, struct.error):
            report["failed"] += 1
            continue
        out.append({"key": salted(salt, root_id + "/" + rel.replace(os.sep, "/")), "root": root_id,
                    "size": st.st_size, "mtime": int(st.st_mtime), "descriptor_id": desc_mod.DESCRIPTOR_ID,
                    "desc": [round(float(v), 5) for v in d], "words": tokens.path_words(rel), "weight": 1.0})
        report["added"] += 1
    return out, report


def forget(index, salt, root):
    root_id = salted(salt, os.path.abspath(root))[:16]
    return [e for e in index if e["root"] != root_id]
