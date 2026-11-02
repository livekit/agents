"""Non-billable connection warming; caller sessions remain caller-owned."""

import asyncio
import math
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

import aiohttp

HEARTBEAT_SECONDS = 20.0


class ConnectionWarmer:
    def __init__(self, session: Callable[[], aiohttp.ClientSession], base_url: str) -> None:
        self._session = session
        url = urlsplit(base_url)
        self._url = urlunsplit((url.scheme, url.netloc, "/__ingress/health", "", ""))
        self._fallback_url = urlunsplit((url.scheme, url.netloc, "/api/internal/health", "", ""))
        self._task: asyncio.Task[None] | None = None
        self._keepalive = False
        self._checked = asyncio.Event()
        self.ready = False
        self.failures = 0
        self.checks = 0

    def start(self, *, keepalive: bool = True) -> None:
        """Start a check, binding recurring work to the current job when present."""
        if self._task is not None and not self._task.done():
            self._keepalive = self._keepalive or keepalive
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Synchronous process setup can precede the session event loop.
            return
        from livekit.agents.job import get_job_context

        job = get_job_context(required=False)
        if job is not None:
            job.add_shutdown_callback(self.aclose)
        self._keepalive = keepalive
        self._checked = asyncio.Event()
        self._task = loop.create_task(self._run(), name="giggy-connection")

    async def connect(self) -> bool:
        self.start()
        await self._checked.wait()
        return self.ready

    async def _run(self) -> None:
        """Check the borrowed client without reopening it after its owner closes it."""
        session = None
        while True:
            interval = HEARTBEAT_SECONDS
            try:
                if session is None:
                    session = self._session()
                if session.closed:
                    self.ready = False
                    return
                if (
                    session.auth is not None
                    or session.cookie_jar
                    or any(
                        name.lower()
                        in {
                            "authorization",
                            "proxy-authorization",
                            "xi-api-key",
                            "x-api-key",
                            "cookie",
                            "x-giggy-input-token",
                        }
                        for name in session.headers
                    )
                ):
                    raise ValueError("Connection warming requires a credential-free caller session")
                connector = session.connector
                retention = getattr(connector, "_keepalive_timeout", 15)
                if (
                    isinstance(retention, (int, float))
                    and math.isfinite(retention)
                    and retention > 0
                ):
                    interval = min(interval, retention / 2)
                async with session.get(
                    self._url, timeout=aiohttp.ClientTimeout(total=5), allow_redirects=False
                ) as response:
                    await response.read()
                    self.ready = response.status == 200
                    missing = response.status == 404
                if missing and self._url != self._fallback_url:
                    async with session.get(
                        self._fallback_url,
                        timeout=aiohttp.ClientTimeout(total=5),
                        allow_redirects=False,
                    ) as response:
                        await response.read()
                        self.ready = response.status == 200
                    if self.ready:
                        self._url = self._fallback_url
                if not self.ready:
                    self.failures += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.ready = False
                self.failures += 1
            finally:
                self.checks += 1
                self._checked.set()
            if not self._keepalive:
                return
            await asyncio.sleep(interval)

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.ready = False
        self._checked.set()
