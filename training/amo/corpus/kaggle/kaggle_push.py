#!/usr/bin/env python3
"""Minimal standard-library Kaggle client for the corpus kernels.

Use it where the official ``kaggle`` CLI cannot be installed (the cloud
container has no PyPI access). With the CLI, ``kaggle kernels push -p DIR``
on the folders from make_kernels.py does the same as ``push``.

Credentials come from the environment, never from arguments:
``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` (classic kaggle.json key) or
``KAGGLE_API_TOKEN`` (newer bearer token), else ~/.kaggle/kaggle.json.

    python training/amo/corpus/kaggle/kaggle_push.py push build/kaggle/generate
    python training/amo/corpus/kaggle/kaggle_push.py status amo-corpus-generate
    python training/amo/corpus/kaggle/kaggle_push.py train build/kaggle      # alternates train-a / train-b
    python training/amo/corpus/kaggle/kaggle_push.py output amo-corpus-train-a build/kaggle/out-a
"""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

API = os.environ.get("KAGGLE_API_ENDPOINT", "https://www.kaggle.com/api/v1")


def credentials():
    user = os.environ.get("KAGGLE_USERNAME")
    if os.environ.get("KAGGLE_API_TOKEN"):
        return user, {"Authorization": "Bearer " + os.environ["KAGGLE_API_TOKEN"]}
    key = os.environ.get("KAGGLE_KEY")
    if not (user and key):
        path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
        if path.is_file():
            stored = json.loads(path.read_text())
            user, key = stored["username"], stored["key"]
    if not (user and key):
        raise SystemExit("Set KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN) in the environment")
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    return user, {"Authorization": "Basic " + token}


def request(method, path, body=None, query=None):
    _, headers = credentials()
    # Kaggle answers 404 to Python's default "Python-urllib" User-Agent (seen 2026-11).
    headers = {**headers, "User-Agent": "amo-kaggle-push/1"}
    url = API + path + ("?" + urllib.parse.urlencode(query) if query else "")
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers = {**headers, "Content-Type": "application/json"}
    call = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(call, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:2000]
        raise SystemExit(f"Kaggle {method} {path} -> HTTP {error.code}: {detail}") from None
    return json.loads(payload) if payload else {}


def push(directory, kernel_sources=None):
    directory = Path(directory)
    meta = json.loads((directory / "kernel-metadata.json").read_text())
    if kernel_sources is not None:
        meta["kernel_sources"] = kernel_sources
    owner, slug = meta["id"].split("/")
    body = {
        "slug": meta["id"], "newTitle": meta["title"],
        "text": (directory / meta["code_file"]).read_text(encoding="utf-8"),
        "language": meta["language"], "kernelType": meta["kernel_type"],
        "isPrivate": meta["is_private"], "enableGpu": meta["enable_gpu"],
        "enableTpu": meta.get("enable_tpu", False), "enableInternet": meta["enable_internet"],
        "datasetDataSources": meta["dataset_sources"], "competitionDataSources": meta["competition_sources"],
        "kernelDataSources": meta["kernel_sources"], "modelDataSources": meta.get("model_sources", []),
        "categoryIds": [],
    }
    if meta.get("machine_shape"):
        body["machineShape"] = meta["machine_shape"]
    result = request("POST", "/kernels/push", body)
    if result.get("error"):
        raise SystemExit(f"push {meta['id']} failed: {result}")
    print(f"pushed {meta['id']} version {result.get('versionNumber')} sources {meta['kernel_sources']}"
          f" -> {result.get('url')}")
    return result


def status(slug):
    user, _ = credentials()
    owner, _, name = slug.rpartition("/")
    return request("GET", "/kernels/status", query={"userName": owner or user, "kernelSlug": name})


def output(slug, destination):
    user, headers = credentials()
    owner, _, name = slug.rpartition("/")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = request("GET", "/kernels/output", query={"userName": owner or user, "kernelSlug": name})
    (destination / "kernel.log").write_text(result.get("log") or "", encoding="utf-8")
    for item in result.get("files", []):
        target = destination / item["fileName"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(item["url"], timeout=600) as response, target.open("wb") as stream:
            while block := response.read(1 << 20):
                stream.write(block)
        print("downloaded", target)
    return result


def train(kernels, prefix="train"):
    """Push the train kernel that did not run last, mounting the other's output.

    ``prefix`` "mix-train" alternates the mix-decomposition kernels the same way.
    """
    kernels = Path(kernels)
    state_path = kernels / "push-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    key = "last_train" if prefix == "train" else f"last_{prefix.replace('-', '_')}"
    last = state.get(key)
    slot = "b" if last == "a" else "a"
    directory = kernels / f"{prefix}-{slot}"
    meta = json.loads((directory / "kernel-metadata.json").read_text())
    family = meta["id"].split("/")[1][:-2]  # e.g. amo-corpus-train / amo-mix-train
    sources = [s for s in meta["kernel_sources"] if family not in s]
    if last:
        sources.append(meta["id"][:-1] + last)
    push(directory, sources)
    state[key] = slot
    state_path.write_text(json.dumps(state, indent=2))


def evaluate(kernels):
    """Push the evaluate kernel with the output of the train kernel that ran last."""
    kernels = Path(kernels)
    state_path = kernels / "push-state.json"
    last = json.loads(state_path.read_text()).get("last_train") if state_path.exists() else None
    if not last:
        raise SystemExit("no train kernel pushed from this folder yet")
    directory = kernels / "evaluate"
    meta = json.loads((directory / "kernel-metadata.json").read_text())
    owner = meta["id"].split("/")[0]
    push(directory, [s for s in meta["kernel_sources"] if "amo-corpus-train" not in s]
         + [f"{owner}/amo-corpus-train-{last}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("push").add_argument("directory")
    commands.add_parser("train").add_argument("kernels", help="folder written by make_kernels.py")
    commands.add_parser("mix-train").add_argument("kernels", help="folder written by make_kernels.py --mix")
    commands.add_parser("parts-train").add_argument("kernels", help="folder written by make_kernels.py --parts")
    commands.add_parser("evaluate").add_argument("kernels", help="folder written by make_kernels.py")
    commands.add_parser("status").add_argument("slug")
    out = commands.add_parser("output")
    out.add_argument("slug")
    out.add_argument("destination")
    args = parser.parse_args()
    if args.command == "push":
        push(args.directory)
    elif args.command == "train":
        train(args.kernels)
    elif args.command == "mix-train":
        train(args.kernels, "mix-train")
    elif args.command == "parts-train":
        train(args.kernels, "parts-train")
    elif args.command == "evaluate":
        evaluate(args.kernels)
    elif args.command == "status":
        print(json.dumps(status(args.slug), indent=2))
    else:
        output(args.slug, args.destination)


if __name__ == "__main__":
    sys.exit(main())
