#!/usr/bin/env python3
"""Drop a working `_audio_eot.{so,dylib,dll}` next to the
`livekit.plugins.turn_detector` package for local dev.

Resolution order:

  Path A — local-source build (preferred when available):
    1. Resolve sources directory:
       - `LK_AUDIO_EOT_SOURCES_DIR` env var, or
       - default sibling path `../open-access-audio-turn-detector`
    2. If resolved, invoke `uv build --wheel` against the plugin (which
       triggers the hatch hook → invokes the private build with weights
       picked up from the checkout's LFS-tracked file) and extracts the
       produced `.so` back into the editable source tree.

  Path B — PyPI wheel extraction (fallback when no local sources):
    1. Query the PyPI JSON API for the latest `livekit-plugins-turn-detector`
       wheel matching the host `(platform_tag, cpython_version)`.
    2. Download the wheel and extract `_audio_eot.{so,dylib,dll}` into the
       package directory.
    3. Until v0 is published, this path errors with a clear message.

Skip the work entirely if the `.so` is already present.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "livekit-plugins" / "livekit-plugins-turn-detector"
PACKAGE_DIR = PLUGIN_DIR / "livekit" / "plugins" / "turn_detector"


def _lib_filename() -> str:
    if sys.platform == "darwin":
        return "_audio_eot.dylib"
    if sys.platform == "win32":
        return "_audio_eot.dll"
    return "_audio_eot.so"


def _already_present() -> bool:
    return (PACKAGE_DIR / _lib_filename()).exists()


def _resolve_local_sources() -> Path | None:
    env_dir = os.environ.get("LK_AUDIO_EOT_SOURCES_DIR")
    default_dir = REPO_ROOT.parent / "open-access-audio-turn-detector"
    if env_dir:
        return Path(env_dir).resolve()
    if default_dir.exists() and any(default_dir.glob("*.c")):
        return default_dir
    return None


def _path_a_local_build(sources: Path) -> int:
    """Build the native lib via the hatch build hook, then copy the
    produced `.so` from the wheel into the editable source tree."""
    print(f"[fetch-audio-eot] Path A: building from local sources at {sources}")
    env = os.environ.copy()
    env["LK_AUDIO_EOT_SOURCES_DIR"] = str(sources)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["uv", "build", "--wheel", "--out-dir", tmp]
        proc = subprocess.run(cmd, cwd=PLUGIN_DIR, env=env)
        if proc.returncode != 0:
            return proc.returncode

        wheels = list(Path(tmp).glob("*.whl"))
        if not wheels:
            print("[fetch-audio-eot] uv build produced no wheel", file=sys.stderr)
            return 1
        wheel = wheels[0]
        canonical = PACKAGE_DIR / _lib_filename()
        with zipfile.ZipFile(wheel) as zf:
            for name in zf.namelist():
                if Path(name).name == _lib_filename():
                    canonical.write_bytes(zf.read(name))
                    print(f"[fetch-audio-eot] extracted {name} → {canonical}")
                    return 0

    print(
        "[fetch-audio-eot] wheel built but did not contain the native library; "
        "verify the hatch build hook resolved the local sources",
        file=sys.stderr,
    )
    return 1


def _path_b_pypi_wheel() -> int:
    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/livekit-plugins-turn-detector/json", timeout=10
        ) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"[fetch-audio-eot] Path B: failed to query PyPI: {e}", file=sys.stderr)
        return _missing_local_sources_error()

    latest = data.get("info", {}).get("version")
    releases = data.get("releases", {}).get(latest, [])
    if not releases:
        return _missing_local_sources_error()

    plat_tag = _host_platform_tag()
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    matching = [
        r
        for r in releases
        if r.get("packagetype") == "bdist_wheel"
        and plat_tag in r["filename"]
        and py_tag in r["filename"]
    ]
    if not matching:
        print(
            f"[fetch-audio-eot] Path B: no wheel for {py_tag}/{plat_tag} in version {latest}",
            file=sys.stderr,
        )
        return _missing_local_sources_error()

    url = matching[0]["url"]
    print(f"[fetch-audio-eot] Path B: downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        wheel_bytes = resp.read()

    canonical = PACKAGE_DIR / _lib_filename()
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
        for name in zf.namelist():
            base = name.rsplit("/", 1)[-1]
            if base == _lib_filename():
                canonical.write_bytes(zf.read(name))
                print(f"[fetch-audio-eot] extracted {name} → {canonical}")
                return 0

    print(
        f"[fetch-audio-eot] Path B: wheel did not contain {_lib_filename()}",
        file=sys.stderr,
    )
    return 1


def _host_platform_tag() -> str:
    plat = sysconfig.get_platform()
    return plat.replace("-", "_").replace(".", "_")


def _missing_local_sources_error() -> int:
    print(
        "[fetch-audio-eot] no published wheel available yet and no local sources found.\n"
        "  Set LK_AUDIO_EOT_SOURCES_DIR to a checkout of `open-access-audio-turn-detector`\n"
        "  (with `git lfs pull` run inside it), then re-run `make fetch-audio-eot`.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    if _already_present():
        print(f"[fetch-audio-eot] {_lib_filename()} already present; skipping")
        return 0

    sources = _resolve_local_sources()
    if sources is not None:
        return _path_a_local_build(sources)

    print("[fetch-audio-eot] no local sources resolved; trying PyPI wheel")
    return _path_b_pypi_wheel()


if __name__ == "__main__":
    raise SystemExit(main())
