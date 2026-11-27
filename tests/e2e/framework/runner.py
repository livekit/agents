"""Bootstrap an importable application; inference and tools live in the application module."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import signal
import socket
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..ready import write_marker

if TYPE_CHECKING:
    from starlette.types import ASGIApp


async def run(module: str, marker: Path, registered: bool) -> None:
    if os.environ.get("E2E_APP_MAIN"):
        # A script such as an example's main.py is `__main__` while its server runs, as under
        # `livekit.agents start main.py`: a spawned job process re-imports it as `__mp_main__`
        # before the worker's environment reaches it. Run as `__mp_main__`, its own
        # `cli.run_app` stays out.
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None
        application = types.ModuleType("__mp_main__")
        application.__file__ = spec.origin
        sys.modules["__main__"] = sys.modules["__mp_main__"] = application
        code = compile(Path(spec.origin).read_text(), spec.origin, "exec")
        exec(code, application.__dict__)
    else:
        application = importlib.import_module(module)
    factory = getattr(application, "create_app", None)
    if factory is not None:
        if registered:
            raise ValueError("an ASGI application cannot register as a voice worker")
        await _serve_asgi(factory, marker)
        return

    from livekit.agents import AgentServer

    from ..resources import finish_cleanup, serving

    server = application.server
    if not isinstance(server, AgentServer):
        raise TypeError(f"{module}.server must be an AgentServer")
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)

    if not registered:
        async with serving(server) as url:
            write_marker(marker, {"url": url})
            await stopped.wait()
        return

    server._update_options(listeners={"health": {"port": 0}}, num_idle_processes=1)
    started = asyncio.Event()
    server._on_started.append(started.set)
    running = asyncio.create_task(server._run(devmode=True))
    try:
        async with asyncio.timeout(30):
            while not started.is_set() or server.id in ("", "unregistered"):
                if running.done():
                    running.result()
                await asyncio.sleep(0.01)
        # Voice-only workers have no text listener. The client reaches them through dispatch.
        url = server._config.delegation_url("").rstrip("/") if server._a2a_endpoints else ""
        write_marker(marker, {"url": url})
        await stopped.wait()
    finally:

        async def close() -> None:
            await server.aclose()
            await running

        await finish_cleanup(close())


async def _serve_asgi(factory: Callable[[str], ASGIApp], marker: Path) -> None:
    """Run an ordinary ASGI factory without importing the LiveKit SDK into its process."""
    import uvicorn

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(
            uvicorn.Config(factory(url), access_log=False, timeout_graceful_shutdown=5)
        )
        running = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(30):
                while not server.started:
                    if running.done():
                        running.result()
                        raise RuntimeError("the ASGI server exited before startup")
                    await asyncio.sleep(0.01)
            write_marker(marker, {"url": url})
            await running
        finally:
            server.should_exit = True
            await running


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "voice"))
