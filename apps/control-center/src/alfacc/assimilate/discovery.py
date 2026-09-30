"""Find installed instrument plugins and their preset files, and say which ones can be hosted.

Which formats a third-party host can load (docs/control-center/plugin-assimilation.md, section 2):

- VST3: yes, on macOS, Windows and Linux. This is the main path.
- AU: yes on macOS (Apple's own API), no elsewhere.
- VST2: only with Steinberg's VST2 SDK, which is no longer licensed to new developers, so
  the render host is built without it unless a licensed SDK is supplied.
- AAX: no. Only Pro Tools and Avid's tools load AAX, and most AAX plugins are iLok/PACE
  wrapped. Nearly every AAX instrument also ships as VST3 or AU; we point the user to that twin.
"""

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

FORMATS = ("VST3", "AU", "VST2", "AAX")
SUFFIXES = {".vst3": "VST3", ".component": "AU", ".aaxplugin": "AAX", ".vst": "VST2",
            ".dll": "VST2", ".so": "VST2"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def plugin_folders(platform: str | None = None) -> dict:
    """Standard install folders per format for this OS."""
    platform = platform or sys.platform
    home = Path.home()
    if platform == "darwin":
        lib = [Path("/Library"), home / "Library"]
        return {
            "VST3": [p / "Audio/Plug-Ins/VST3" for p in lib],
            "AU": [p / "Audio/Plug-Ins/Components" for p in lib],
            "VST2": [p / "Audio/Plug-Ins/VST" for p in lib],
            "AAX": [Path("/Library/Application Support/Avid/Audio/Plug-Ins")],
        }
    if platform == "win32":
        common = Path(_env("CommonProgramFiles", r"C:\Program Files\Common Files"))
        pf = Path(_env("ProgramFiles", r"C:\Program Files"))
        local = Path(_env("LOCALAPPDATA", str(home / "AppData/Local")))
        return {
            "VST3": [common / "VST3", local / "Programs/Common/VST3"],
            "AU": [],
            "VST2": [pf / "VSTPlugins", pf / "Steinberg/VSTPlugins", common / "VST2",
                     common / "Steinberg/VST2"],
            "AAX": [common / "Avid/Audio/Plug-Ins"],
        }
    return {
        "VST3": [home / ".vst3", Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3")],
        "AU": [],
        "VST2": [home / ".vst", Path("/usr/lib/vst"), Path("/usr/local/lib/vst")],
        "AAX": [],
    }


def preset_folders(vendor: str, name: str, platform: str | None = None) -> list:
    """Where .vstpreset files for one plugin live (VST3 spec's preset locations)."""
    platform = platform or sys.platform
    home = Path.home()
    if platform == "darwin":
        roots = [home / "Library/Audio/Presets", Path("/Library/Audio/Presets")]
    elif platform == "win32":
        roots = [home / "Documents/VST3 Presets",
                 Path(_env("APPDATA", str(home / "AppData/Roaming"))) / "VST3 Presets",
                 Path(_env("PROGRAMDATA", r"C:\ProgramData")) / "VST3 Presets"]
    else:
        roots = [home / ".vst3/presets", Path("/usr/share/vst3/presets"),
                 Path("/usr/local/share/vst3/presets")]
    return [r / vendor / name for r in roots]


@dataclass
class FoundPlugin:
    path: str
    name: str
    format: str
    loadable: bool
    reason: str = ""     # why not, or what to do instead
    twin: str = ""       # a loadable VST3/AU of the same plugin, for AAX and VST2 files

    def to_dict(self) -> dict:
        return asdict(self)


def _is_plugin(p: Path) -> str | None:
    fmt = SUFFIXES.get(p.suffix.lower())
    if fmt is None:
        return None
    if fmt == "VST2" and p.suffix.lower() == ".so" and sys.platform != "linux":
        return None
    return fmt


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def discover(extra_folders=(), host_formats=("VST3",), platform: str | None = None,
             folders: dict | None = None) -> list:
    """Every plugin file in the standard folders (and extra ones), with whether it can be hosted.

    host_formats are the formats the render host was built with (from `alfa-render-host formats`,
    where JUCE calls VST2 "VST" and AU "AudioUnit").
    """
    folders = folders if folders is not None else plugin_folders(platform)
    host = {{"VST": "VST2", "AudioUnit": "AU"}.get(f, f) for f in host_formats}
    seen, found = set(), []
    search = [(fmt, Path(d)) for fmt, dirs in folders.items() for d in dirs]
    search += [(None, Path(d)) for d in extra_folders]
    for _fmt, root in search:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            # Bundles (.vst3, .component, .aaxplugin, .vst on macOS) are folders: stop at them.
            for d in list(dirnames):
                fmt = _is_plugin(here / d)
                if fmt:
                    dirnames.remove(d)
                    _add(found, seen, here / d, fmt, host)
            for f in filenames:
                fmt = _is_plugin(here / f)
                if fmt:
                    _add(found, seen, here / f, fmt, host)
    # Point AAX and unhostable VST2 files at a loadable twin of the same plugin.
    loadable = {_norm(p.name): p.path for p in found if p.loadable}
    for p in found:
        if not p.loadable and _norm(p.name) in loadable:
            p.twin = loadable[_norm(p.name)]
    return sorted(found, key=lambda p: (p.name.lower(), FORMATS.index(p.format)))


def _add(found, seen, path: Path, fmt: str, host: set) -> None:
    key = str(path.resolve()) if path.exists() else str(path)
    if key in seen:
        return
    seen.add(key)
    if fmt == "AAX":
        ok, why = False, "AAX plugins load only in Pro Tools; use the VST3 or AU version"
    elif fmt == "AU" and "AU" not in host:
        ok, why = False, "Audio Units load only on macOS"
    elif fmt not in host:
        ok, why = False, ("VST2 needs the render host built with a licensed VST2 SDK"
                          if fmt == "VST2" else f"this render host has no {fmt} support")
    else:
        ok, why = True, ""
    found.append(FoundPlugin(str(path), path.stem, fmt, ok, why))


def find_preset_files(vendor: str, name: str, platform: str | None = None, limit: int = 5000) -> list:
    """.vstpreset files for a plugin, sorted by relative path (folders are often categories)."""
    out = []
    for root in preset_folders(vendor, name, platform):
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.vstpreset")):
            out.append({"path": str(p), "name": p.stem, "category": p.parent.relative_to(root).as_posix()})
            if len(out) >= limit:
                return out
    return out
