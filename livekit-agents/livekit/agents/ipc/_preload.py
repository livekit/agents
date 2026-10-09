"""Side-effect module: the framework's own one-time warm-up, run once per process image.

Each step is lazy one-time work that would otherwise run inside the first job, on the event
loop, as a 100-500 ms stall at session start. Everything here is fork-safe: imports, a
``dlopen``, model weights, a cached SSL context; no threads, no event loop.

Where it runs decides how often it costs:

- with the ``forkserver`` start method (Linux) the worker lists this module in
  ``set_forkserver_preload``, ahead of ``_preload_freeze``, so it runs once in the forkserver
  and every job process inherits the result copy-on-write;
- with ``spawn`` (macOS, Windows) each job process imports it while it warms up, before any
  job is assigned.

The job process always imports it: under a forkserver the module is already in
``sys.modules`` and the import is a no-op. Only the local inference models depend on where
the module runs and on ``LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE`` (see
``_local_inference_models``).

Failures are logged at debug level only: the first real use reports a proper error.
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
    except Exception:
        logger.debug("could not preload %s", name, exc_info=True)
        return
    logger.debug("preloaded %s", name, extra={"elapsed": round(time.perf_counter() - started, 3)})


def _av() -> None:
    import av  # noqa: F401


def _local_inference_models() -> bool:
    value = os.environ.get(ENV_PRELOAD_LOCAL_INFERENCE)
    if value is not None and value.strip().lower() in _FALSY:
        return False

    # the VAD and the turn detector's local end-of-turn model: constructing them later in a
    # job is free once these singletons exist (~25 ms of GIL-held CPU otherwise)
    import livekit.local_inference as li

    li.init_vad()

    # the EOT weights cost ~244 MB: a process without a multiprocessing parent (the forkserver,
    # a thread executor's worker) shares them across jobs, a spawned job process would not
    if value is not None or multiprocessing.parent_process() is None:
        li.init_eot()
    return True


def _rtc_native_library() -> None:
    # the dlopen (~150-350 ms). The runtime itself (FfiClient.instance) starts threads, so it
    # stays per process; with the library already mapped it takes a few milliseconds
    from livekit.rtc._ffi_client import get_ffi_lib

    get_ffi_lib()


def _openai_resources() -> None:
    # the openai SDK, which livekit.agents.inference is built on, imports its whole resources
    # tree on the first client attribute access (~300-550 ms)
    import openai.resources  # noqa: F401


def _httpx_client() -> None:
    # the first AsyncClient in a process image pays ~40 ms (the SSL context from the CA bundle
    # among other lazy setup); later ones take a few milliseconds. The inference LLM, STT and
    # TTS each build one
    import httpx

    httpx.AsyncClient()


_step("av", _av)
_step("the local inference models", _local_inference_models)
_step("the livekit-rtc native library", _rtc_native_library)
_step("the openai SDK resources", _openai_resources)
_step("the httpx client", _httpx_client)
