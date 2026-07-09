# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin ctypes binding to the RNNoise C library bundled inside pyrnnoise's
site-packages directory.

This deliberately never imports the `pyrnnoise` Python package -- only its
package directory is located (via `importlib.util.find_spec`) so we can load
the native `librnnoise` shared library it ships. Importing `pyrnnoise` itself
pulls in `audiolab` (and transitively `matplotlib`) and forces an upper pin on
`av`, none of which this plugin needs.
"""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

import numpy as np

_RNNOISE_FRAME_SAMPLES = 480
_NATIVE_LIB_EXTENSIONS = (".dylib", ".so", ".dll")

_lib: ctypes.CDLL | None = None


def _find_bundled_lib() -> Path:
    """Locate the RNNoise native library inside pyrnnoise's package directory."""
    spec = importlib.util.find_spec("pyrnnoise")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Could not locate the pyrnnoise package, which bundles the native "
            "RNNoise library required by this plugin. Install it with "
            "`pip install pyrnnoise`."
        )
    pkg_dir = Path(next(iter(spec.submodule_search_locations)))

    candidates = sorted(
        path
        for ext in _NATIVE_LIB_EXTENSIONS
        for path in pkg_dir.glob(f"*{ext}")
        if "rnnoise" in path.name.lower()
    )
    if not candidates:
        raise RuntimeError(
            f"No bundled RNNoise native library found in {pkg_dir}. Install/reinstall "
            "pyrnnoise with `pip install pyrnnoise` to provide it."
        )
    return candidates[0]


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    lib = ctypes.CDLL(str(_find_bundled_lib()))

    lib.rnnoise_get_frame_size.argtypes = []
    lib.rnnoise_get_frame_size.restype = ctypes.c_int

    lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    lib.rnnoise_create.restype = ctypes.c_void_p

    lib.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rnnoise_process_frame.restype = ctypes.c_float

    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    lib.rnnoise_destroy.restype = None

    frame_size = lib.rnnoise_get_frame_size()
    if frame_size != _RNNOISE_FRAME_SAMPLES:
        raise RuntimeError(
            f"Unexpected RNNoise frame size {frame_size}, expected {_RNNOISE_FRAME_SAMPLES}"
        )

    _lib = lib
    return lib


class _RNNoiseDenoiser:
    """Owns one native RNNoise DenoiseState and processes 480-sample frames."""

    def __init__(self) -> None:
        self._lib = _load_lib()
        state = self._lib.rnnoise_create(None)  # NULL model -> baked-in default RNNModel
        if not state:
            raise RuntimeError("rnnoise_create(None) returned NULL")
        self._state: int | None = state

    def process_frame(self, frame_480_int16: np.ndarray) -> np.ndarray:
        """Denoise exactly one 480-sample mono int16 frame, in place semantics aside."""
        if self._state is None:
            raise RuntimeError("process_frame() called after close()")
        assert frame_480_int16.dtype == np.int16
        assert len(frame_480_int16) == _RNNOISE_FRAME_SAMPLES

        # RNNoise's C ABI operates on float32 samples in the same numeric range as
        # int16 (i.e. not normalized to [-1, 1]).
        buf = frame_480_int16.astype(np.float32)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.rnnoise_process_frame(self._state, ptr, ptr)
        return buf.astype(np.int16)

    def close(self) -> None:
        if self._state is not None:
            self._lib.rnnoise_destroy(self._state)
            self._state = None

    def __del__(self) -> None:
        self.close()
