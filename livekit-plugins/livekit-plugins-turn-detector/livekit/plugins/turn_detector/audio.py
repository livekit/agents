from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import sys
import time
import wave
import weakref
from pathlib import Path

from livekit import rtc
from livekit.agents import Plugin
from livekit.agents.inference.turn_detection import (
    AudioTurnDetector as _CloudAudioTurnDetector,
    BaseAudioTurnDetectionStream,
)
from livekit.agents.inference.turn_detection.detector import (
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)

from .log import logger
from .version import __version__

# Rolling buffer cap on the Python side. The native lib expects up to
# `_CLIENT_BUFFER_SECONDS` of 16 kHz s16le PCM per predict; it will pad
# or truncate internally to its fixed receptive window.
_CLIENT_BUFFER_SECONDS = 1.2
_BYTES_PER_SECOND = DEFAULT_SAMPLE_RATE * 2  # 16 kHz * int16
_CLIENT_BUFFER_BYTES = int(_CLIENT_BUFFER_SECONDS * _BYTES_PER_SECOND)

_LIB_BASENAME = "_audio_eot"


def _native_lib_path() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent
    if sys.platform == "darwin":
        ext = ".dylib"
    elif sys.platform == "win32":
        ext = ".dll"
    else:
        ext = ".so"
    candidate = pkg_dir / f"{_LIB_BASENAME}{ext}"
    return candidate if candidate.exists() else None


_lib: ctypes.CDLL | None = None
_lib_load_error: BaseException | None = None


def _load_lib() -> None:
    """Load the native shared library and bind `audio_eot_predict`.

    Errors are captured rather than raised: anyone who actually tries to
    use the local backend gets a clear error at use-time; merely
    importing the package on a host without the .so does not crash.
    """
    global _lib, _lib_load_error

    lib_path = _native_lib_path()
    if lib_path is None:
        _lib_load_error = RuntimeError(
            f"audio EOT native library not found next to livekit.plugins.turn_detector "
            f"(expected `{_LIB_BASENAME}.{{so,dylib,dll}}`). "
            f"Run `make fetch-audio-eot` or set LK_AUDIO_EOT_SOURCES_DIR + "
            f"LK_AUDIO_EOT_WEIGHTS_FILE and rebuild the plugin."
        )
        return

    try:
        lib = ctypes.CDLL(str(lib_path))
        lib.audio_eot_predict.argtypes = (ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t)
        lib.audio_eot_predict.restype = ctypes.c_float
    except OSError as e:
        _lib_load_error = RuntimeError(
            f"failed to load audio EOT native library at {lib_path}: {e}"
        )
        return

    _lib = lib

    # Force any lazy table init (FFT twiddles, mel filterbank) in this
    # process so children forked from the worker forkserver inherit them
    # via COW. Cheap zeros-buffer warmup — result discarded.
    try:
        _predict(b"\x00\x00" * 16)
    except Exception as e:
        logger.warning("audio EOT lib warmup predict failed: %s", e)


def _predict(pcm_bytes: bytes) -> float:
    if _lib is None:
        raise _lib_load_error or RuntimeError("audio EOT native library not loaded")
    n = len(pcm_bytes) // 2
    buf = (ctypes.c_int16 * n).from_buffer_copy(pcm_bytes[: n * 2])
    return float(_lib.audio_eot_predict(buf, ctypes.c_size_t(n)))


_load_lib()

LANGUAGES = {
    "en": 0.3,
    "fr": 0.3,
    "de": 0.3,
    "hi": 0.3,
    "ja": 0.3,
    "ko": 0.3,
    "zh": 0.3,
    "es": 0.3,
}


class AudioTurnDetector(_CloudAudioTurnDetector):
    """Local audio EOT model exposed by livekit-plugins-turn-detector.

    Runs entirely in-process via ctypes. The native library is loaded at
    module import and its weights/feature tables are resident pages
    inherited via COW by all forked job processes (when the worker
    start method is `forkserver`).

    Users wanting the cloud-backed model can import
    `livekit.agents.inference.AudioTurnDetector` directly instead.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._opts = TurnDetectorOptions(
            sample_rate=sample_rate,
            base_url="",
            api_key="",
            api_secret="",
            conn_options=conn_options,
        )
        self._session = None
        self._streams: weakref.WeakSet[BaseAudioTurnDetectionStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return "eot-audio-mini"

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> BaseAudioTurnDetectionStream:
        if _lib is None:
            raise _lib_load_error or RuntimeError("audio EOT native library not loaded")
        stream = _LocalAudioTurnDetectionStream(detector=self, opts=self._opts)
        self._streams.add(stream)
        return stream

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        lang_key = language.language if language is not None else "en"
        return LANGUAGES.get(lang_key)


class _LocalAudioTurnDetectionStream(BaseAudioTurnDetectionStream):
    """In-process audio EOT stream.

    Drains the audio channel into a rolling 1.2 s s16le @ 16 kHz buffer,
    fires a single `audio_eot_predict` call on `warmup()` via
    `asyncio.to_thread`, and emits the held probability when
    `set_active(True)` is called before the inference completes.
    """

    def __init__(
        self,
        *,
        detector: AudioTurnDetector,
        opts: TurnDetectorOptions,
    ) -> None:
        self._buf: bytearray = bytearray()
        self._held_probability: float | None = None
        super().__init__(detector=detector, opts=opts)

    # region: transport hooks

    def _on_warmup_start(self, request_id: str) -> None:
        snapshot = bytes(self._buf)
        fut = self._active_request_fut
        task = asyncio.create_task(self._run_predict(request_id, snapshot, fut))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_predict(
        self,
        request_id: str,
        pcm_snapshot: bytes,
        fut: asyncio.Future[float] | None,
    ) -> None:
        prob = 0.0
        try:
            prob = await asyncio.to_thread(_predict, pcm_snapshot)
        except Exception:
            logger.exception("local audio EOT prediction failed")

        if request_id != self._active_request_id:
            return

        if fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_result(prob)

        if self.is_active:
            self._emit_prediction(prob)
        else:
            self._held_probability = prob

    def _on_activate(self) -> None:
        if self._held_probability is not None:
            prob = self._held_probability
            self._held_probability = None
            self._emit_prediction(prob)

    async def _on_audio_chunk(self, frame: rtc.AudioFrame) -> None:
        data = bytes(frame.data)
        if not data:
            return
        self._buf.extend(data)
        overflow = len(self._buf) - _CLIENT_BUFFER_BYTES
        if overflow > 0:
            del self._buf[:overflow]

    async def _on_flush_sentinel(
        self, sentinel: BaseAudioTurnDetectionStream._FlushSentinel
    ) -> None:
        keep_bytes = sentinel.keep_tail_ms * _BYTES_PER_SECOND // 1000
        if keep_bytes < len(self._buf):
            del self._buf[: len(self._buf) - keep_bytes]
        self._held_probability = None

    async def _run_transport(self) -> None:
        await self._drain_audio_channel()

    # endregion

    async def aclose(self) -> None:
        await super().aclose()
        self._held_probability = None


class _AudioEotPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        # Weights are embedded in the native library; nothing to download.
        pass


Plugin.register_plugin(_AudioEotPlugin())
