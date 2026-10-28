"""Side-effect module: the framework's own one-time warm-up, run once per process image.

Each step is lazy one-time work that would otherwise run inside the first job, on the event
loop, as a 100-500 ms stall at session start. Everything imported here is fork-safe: imports, a
``dlopen`` and a cached SSL context; no clients, threads or event loop.

The forkserver imports this module before plugins and application preloads; spawned
children import it before their application preloads. Forked children inherit these
resources copy-on-write and their import is a no-op.

Active model initialization runs separately in each launched process, before readiness.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable
from typing import Any

from ..log import logger

# a falsy value never preloads the local inference models, any other value always preloads both,
# and unset preloads the VAD always and the end-of-turn weights only where every job shares them
ENV_PRELOAD_LOCAL_INFERENCE = "LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE"

_FALSY = ("0", "false", "no", "off")


def _step(name: str, fnc: Callable[[], Any]) -> None:
    # a step returns False when it skipped itself
    started = time.perf_counter()
    try:
        if fnc() is False:
            return
    except Exception as error:
        raise RuntimeError(f"preload {name!r} failed: {error}") from error
    logger.debug("preloaded %s", name, extra={"elapsed": round(time.perf_counter() - started, 3)})


def _shared() -> bool | None:
    """Whether the large models load here: ``None`` when preloading is off; else whether this
    process shares them across jobs (the forkserver, a thread executor's worker, which have no
    multiprocessing parent) or ``LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE`` asks for them anyway; a
    spawned job process loads only what it uses."""
    value = os.environ.get(ENV_PRELOAD_LOCAL_INFERENCE)
    if value is not None and value.strip().lower() in _FALSY:
        return None
    return value is not None or multiprocessing.parent_process() is None


def _local_inference_models() -> bool:
    if (eot := _shared()) is None:
        return False
    # the EOT weights cost ~244 MB: loaded where every job shares them
    # the VAD and the turn detector's local end-of-turn model: constructing them later in a
    # job is free once these singletons exist (~25 ms of GIL-held CPU otherwise). The core's
    # own when the build links them natively
    from .. import _core

    if _core.local_models_init(vad=True, eot=eot):
        return True

    import livekit.local_inference as li

    li.init_vad()
    if eot:
        li.init_eot()
    return True


def _rtc_native_library() -> None:
    # the dlopen (~150-350 ms). The runtime itself (FfiClient.instance) starts threads, so it
    # stays per process; with the library already mapped it takes a few milliseconds
    from livekit.rtc._ffi_client import get_ffi_lib

    get_ffi_lib()


def _rtc_runtime() -> None:
    # Active native state belongs to the launched process, never the forkserver image.
    from livekit.rtc._ffi_client import FfiClient

    _ = FfiClient.instance


def _openai_resources() -> None:
    # the openai SDK, which livekit.agents.inference is built on, imports its whole resources
    # tree on the first client attribute access (~300-550 ms)
    import openai.resources  # noqa: F401


def _shared_ssl_context() -> None:
    from ..utils.http_context import _create_ssl_context

    _create_ssl_context()


_step("the livekit-rtc native library", _rtc_native_library)
_step("the openai SDK resources", _openai_resources)
_step("the shared SSL context", _shared_ssl_context)
