from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable
from typing import Any

from ..log import logger

# Imported for its side effects, in the forkserver image too: module-level steps must stay
# fork-safe (imports, dlopen, cached SSL context; no clients, threads or event loop).

# Falsy: never preload the local models; any other value: preload both; unset: the VAD always,
# the end-of-turn weights only where every job shares them.
ENV_PRELOAD_LOCAL_INFERENCE = "LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE"

_FALSY = ("0", "false", "no", "off")


def _step(name: str, fnc: Callable[[], Any]) -> None:
    # A step returns False when it skipped itself.
    started = time.perf_counter()
    try:
        if fnc() is False:
            return
    except Exception as error:
        raise RuntimeError(f"preload {name!r} failed: {error}") from error
    logger.debug("preloaded %s", name, extra={"elapsed": round(time.perf_counter() - started, 3)})


def _shared() -> bool | None:
    # None: preloading off. True: a process without a multiprocessing parent (forkserver,
    # thread executor) shares the models across jobs, or the env var asks for them.
    value = os.environ.get(ENV_PRELOAD_LOCAL_INFERENCE)
    if value is not None and value.strip().lower() in _FALSY:
        return None
    return value is not None or multiprocessing.parent_process() is None


def _local_inference_models() -> bool:
    if (eot := _shared()) is None:
        return False
    # The EOT weights cost ~244 MB, so they load only where every job shares them. Once these
    # singletons exist, constructing the models in a job is free (~25 ms GIL-held otherwise).
    from .. import _core

    if _core.local_models_init(vad=True, eot=eot):
        return True

    import livekit.local_inference as li

    li.init_vad()
    if eot:
        li.init_eot()
    return True


def _rtc_native_library() -> None:
    # Only the dlopen (~150-350 ms): FfiClient.instance starts threads, so it stays per process.
    from livekit.rtc._ffi_client import get_ffi_lib

    get_ffi_lib()


def _rtc_runtime() -> None:
    # Starts threads: call only in the launched process, never the forkserver image.
    from livekit.rtc._ffi_client import FfiClient

    _ = FfiClient.instance


def _shared_ssl_context() -> None:
    from ..http.client import _ssl_context

    _ssl_context()


_step("the livekit-rtc native library", _rtc_native_library)
_step("the shared SSL context", _shared_ssl_context)
