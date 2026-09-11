# Copyright 2026 Komaa DigiTech
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

"""Zero-wiring startup: the listener starts with your worker.

Your file looks like any other agent example - `AgentServer()`, one decorated
entrypoint, and nothing starts except through ``cli.run_app(server)``:

    from livekit.plugins import standin

    server = AgentServer()

    @server.rtc_session(agent_name="msteams-agent")
    async def entrypoint(ctx: JobContext):
        session = AgentSession(llm=...)
        call = await standin.TeamsCall().start(session, ctx=ctx)
        await session.start(agent=MyAgent(call), room=ctx.room)

    if __name__ == "__main__":
        cli.run_app(server)

There is no bootstrap call. Importing the plugin registers it with the worker
(the same import-time registration every LiveKit plugin performs), and the call
listener is ARMED only when ``STANDIN_SECRET`` is set and STARTED only when the
worker actually runs, on its ``worker_started`` event - which fires once, in the
main process, with the loop already running. A worker without ``STANDIN_SECRET``
behaves exactly as if this plugin were not installed.

Why a listener at all: StandIn dials the worker per call, and no job exists
until that dial has been answered and an agent dispatched - so the listener
cannot live inside the entrypoint. It lives inside the worker's own lifecycle
instead, where the developer never sees it.

Configuration is environment-only, matching how every other plugin reads its
keys: ``STANDIN_SECRET`` (arms the plugin), ``STANDIN_PORT`` (default 9442, the
port every StandIn plugin listens on), ``STANDIN_HOST`` (default 0.0.0.0; set
127.0.0.1 when only a local tunnel should reach the listener), ``STANDIN_WS_PATH``
(default /msteams/calling), plus the standard ``LIVEKIT_*`` project variables the
worker already has.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import TYPE_CHECKING, Any

from .bridge import CallBridge
from .log import logger

if TYPE_CHECKING:
    from livekit.agents import AgentServer

_installed = False
_startup_tasks: set[asyncio.Task[None]] = set()


def _resolve_agent_name(server: AgentServer) -> str:
    """The name this worker registered with.

    Read off the worker rather than configured twice: the developer states it
    once, where the framework already requires it, in
    ``@server.rtc_session(agent_name=...)``. Two copies of a string that MUST
    match is how you get a room that is created, a job that never arrives, and
    a caller who hears silence.

    Resolved at worker_started, never earlier: it is the rtc_session decorator
    that sets it, and it applies the framework's own precedence (including
    LIVEKIT_AGENT_NAME_OVERRIDE), so the plugin cannot disagree with the worker
    about who it is.
    """
    name = getattr(server, "_agent_name", "") or ""
    if not name:
        # Private-attribute read, so fall back rather than break on an SDK that
        # renames it.
        name = os.environ.get("LIVEKIT_AGENT_NAME", "")
    return name


def _on_worker_started(server: AgentServer) -> None:
    """Runs once in the main worker process. Never raises: a misconfigured
    plugin must log why it is not answering calls, not kill the worker."""
    secret = os.environ.get("STANDIN_SECRET", "")
    if not secret:
        logger.debug("standin: STANDIN_SECRET is not set; not answering Teams calls")
        return

    try:
        bridge = CallBridge(
            secret=secret,
            host=os.environ.get("STANDIN_HOST", "0.0.0.0"),
            port=int(os.environ.get("STANDIN_PORT", "9442")),
            ws_path=os.environ.get("STANDIN_WS_PATH", "/msteams/calling"),
        )
    except Exception:
        logger.exception("standin: misconfigured; this worker takes no Teams calls")
        return

    bridge.agent_name = _resolve_agent_name(server)
    if not bridge.agent_name:
        # A worker with no registered name relies on AUTOMATIC dispatch: the
        # room creation itself assigns the job, so the listener still works.
        # Explicit dispatch is the recommended setup; say so, then proceed.
        logger.info(
            "standin: no agent name registered; relying on automatic dispatch "
            "(set @server.rtc_session(agent_name=...) for explicit dispatch)"
        )

    async def _start() -> None:
        try:
            await bridge.start()
        except Exception:
            logger.exception("standin: failed to start; this worker takes no Teams calls")

    # The loop holds only weak references to tasks; keeping ours in a module
    # set stops the startup task from being garbage-collected mid-flight.
    start_task = asyncio.ensure_future(_start())
    _startup_tasks.add(start_task)
    start_task.add_done_callback(_startup_tasks.discard)

    # Stop ACCEPTING new calls when the worker starts draining: live calls
    # continue, but a draining worker must not answer dials it will never
    # dispatch (the default drain window is an hour).
    original_drain = server.drain

    async def _drain(*args: Any, **kwargs: Any) -> Any:
        bridge.draining = True
        return await original_drain(*args, **kwargs)

    server.drain = _drain  # type: ignore[method-assign]

    # Close the listener when the worker shuts down. Wrapping aclose is the
    # only hook: the worker emits no stopped event. The startup task is
    # cancelled first so a shutdown racing a slow start cannot bind AFTER
    # cleanup and leak the port.
    original_aclose = server.aclose

    async def _aclose(*args: Any, **kwargs: Any) -> Any:
        start_task.cancel()
        with contextlib.suppress(BaseException):
            await start_task
        with contextlib.suppress(Exception):
            await bridge.aclose()
        return await original_aclose(*args, **kwargs)

    server.aclose = _aclose  # type: ignore[method-assign]


def _install() -> None:
    """Auto-subscribe every AgentServer to the startup hook, at plugin import.

    ``server.on("worker_started", ...)`` is the framework's own extension point;
    the only thing automated here is the subscription, so the developer's file
    stays exactly the shape of every other example. Idempotent, and inert on
    servers that never run (a subprocess imports the module but never fires
    worker_started, so nothing binds there - the reason this must NOT be done
    in setup_fnc, which runs once per job subprocess)."""
    global _installed
    if _installed:
        return
    _installed = True

    from livekit.agents import AgentServer

    def _hook(server: AgentServer) -> None:
        # Idempotent per instance, so init + run hooking (or a double import
        # path) can never subscribe twice.
        if getattr(server, "_standin_hooked", False):
            return
        server._standin_hooked = True  # type: ignore[attr-defined]
        try:
            server.on("worker_started", lambda: _on_worker_started(server))
        except Exception:
            logger.exception("standin: could not attach to the worker lifecycle")

    original_init = AgentServer.__init__

    def patched_init(self: AgentServer, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _hook(self)

    AgentServer.__init__ = patched_init  # type: ignore[method-assign]

    # Also hook at run(): a server CONSTRUCTED before this plugin was imported
    # missed the __init__ patch, and run() is the last moment before
    # worker_started fires. Between the two, import order cannot matter.
    original_run = AgentServer.run

    async def patched_run(self: AgentServer, *args: Any, **kwargs: Any) -> Any:
        _hook(self)
        return await original_run(self, *args, **kwargs)

    AgentServer.run = patched_run  # type: ignore[method-assign]


__all__ = ["_install"]
