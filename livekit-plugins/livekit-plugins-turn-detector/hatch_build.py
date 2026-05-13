"""Custom Hatch build hook that compiles the audio EOT native library.

This hook is a thin orchestration layer. It knows how to *find* the
proprietary build inputs but not how the build itself works, and it
deliberately stays unaware of the model's weight format or quantization
scheme. The private repo `open-access-audio-turn-detector` owns:

  - C/C++ sources for the model
  - `CMakeLists.txt` (kernel dispatch defines, source layout, link flags)
  - A `build.py` entry point that emits `_audio_eot.{so,dylib,dll}` with
    weights embedded
  - The LFS-tracked weights file itself (the manifest inside the repo
    declares its relative path; `build.py` resolves it automatically)

The contract between this public hook and the private repo is just:

    python <fetched-source>/build.py \
        --output-dir <dst> \
        [--archs arm64,x86_64]

Resolution order for sources:
  1. `LK_AUDIO_EOT_SOURCES_DIR` env var → local checkout
  2. Default sibling path `../../../open-access-audio-turn-detector`
  3. `git clone --depth 1 --branch <tag>` (with LFS resolution) at the
     version pinned in `pyproject.toml`

The git-clone path requires `git` and `git-lfs` on the build host; the
weights file inside the checkout is LFS-tracked. A clone without LFS
yields a small pointer file, which `build.py` detects and rejects.

If neither local sources nor a pinned source version are available, the
hook no-ops and the wheel ships pure-Python. The runtime loader in
`audio.py` then raises a clear `RuntimeError` if the local backend is
actually invoked.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

_HERE = Path(__file__).resolve().parent

_SOURCES_REPO = os.environ.get(
    "LK_AUDIO_EOT_SOURCES_REPO", "livekit/open-access-audio-turn-detector"
)


def _load_pins() -> dict[str, str]:
    with open(_HERE / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    cfg = pyproject.get("tool", {}).get("livekit-audio-eot", {})
    return {
        "sources_version": cfg.get("sources-version") or "",
    }


class CompileNativeHook(BuildHookInterface):  # type: ignore[type-arg]
    PLUGIN_NAME = "compile-audio-eot"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return

        pins = _load_pins()
        fetched = _HERE / "_fetched"
        source_dir = fetched / "source"

        if not _fetch_sources(self.app, source_dir, pins):
            self.app.display_info(
                "[audio-eot] no native sources available; shipping pure-Python wheel"
            )
            return

        build_script = source_dir / "build.py"
        if not build_script.exists():
            raise RuntimeError(
                f"private sources at {source_dir} are missing the `build.py` entry "
                f"point. The private repo must ship a `build.py` that accepts "
                f"`--output-dir <path>` and emits `_audio_eot.{{so,dylib,dll}}`."
            )

        out_dir = _HERE / "livekit" / "plugins" / "turn_detector"
        out_dir.mkdir(parents=True, exist_ok=True)

        if sys.platform == "darwin":
            ext = ".dylib"
        elif sys.platform == "win32":
            ext = ".dll"
        else:
            ext = ".so"

        out_path = out_dir / f"_audio_eot{ext}"

        cmd = [sys.executable, str(build_script), "--output-dir", str(out_dir)]

        # Local-only override: experiment with a different weights file
        # without bumping the source pin. CI never sets this.
        override = os.environ.get("LK_AUDIO_EOT_WEIGHTS_FILE")
        if override:
            override_path = Path(override).resolve()
            if not override_path.exists():
                raise RuntimeError(
                    f"LK_AUDIO_EOT_WEIGHTS_FILE points at {override_path} which does not exist"
                )
            cmd.extend(["--weights", str(override_path)])

        if sys.platform.startswith("darwin"):
            import re

            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmd.extend(["--archs", ",".join(archs)])

        self.app.display_info(f"[audio-eot] invoking private build: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        if not out_path.exists():
            raise RuntimeError(
                f"private build script did not produce expected output at {out_path}"
            )

        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        build_data.setdefault("force_include", {})[str(out_path)] = (
            f"livekit/plugins/turn_detector/_audio_eot{ext}"
        )

        self.app.display_info(
            f"[audio-eot] built native library for {platform.platform()} -> {out_path}"
        )


def _fetch_sources(app: Any, source_dir: Path, pins: dict[str, str]) -> bool:
    if source_dir.exists() and (source_dir / "build.py").exists():
        return True

    env_dir = os.environ.get("LK_AUDIO_EOT_SOURCES_DIR")
    default_dir = (_HERE / ".." / ".." / "..").resolve() / "open-access-audio-turn-detector"

    local_dir: Path | None = None
    if env_dir:
        local_dir = Path(env_dir).resolve()
        if not local_dir.exists():
            raise RuntimeError(
                f"LK_AUDIO_EOT_SOURCES_DIR points at {local_dir} which does not exist"
            )
    elif default_dir.exists() and (default_dir / "build.py").exists():
        local_dir = default_dir

    if local_dir is not None:
        app.display_info(f"[audio-eot] using local sources from {local_dir}")
        if source_dir.exists():
            shutil.rmtree(source_dir)
        shutil.copytree(
            local_dir,
            source_dir,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "build", "_fetched"),
        )
        return True

    version = pins["sources_version"]
    if not version:
        return False

    token = os.environ.get("GH_TOKEN_AUDIO_EOT_SOURCES") or os.environ.get("GH_TOKEN")
    clone_url = (
        f"https://x-access-token:{token}@github.com/{_SOURCES_REPO}.git"
        if token
        else f"https://github.com/{_SOURCES_REPO}.git"
    )

    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.parent.mkdir(parents=True, exist_ok=True)

    # Use git clone so LFS-tracked weights resolve to real bytes. The
    # tarball endpoint at archive/refs/tags/*.tar.gz returns LFS pointer
    # files instead, which `build.py` would reject.
    app.display_info(f"[audio-eot] cloning sources from {_SOURCES_REPO} at tag {version}")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            version,
            clone_url,
            str(source_dir),
        ],
        check=True,
    )
    return True
