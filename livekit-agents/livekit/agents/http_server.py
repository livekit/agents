"""
Provides a lightweight HTTP server for agent utilities:
- Health check endpoints
- Metrics reporting
- Webhook receivers
- Debug interfaces

Integrated with the agent's async event loop for unified lifecycle management.
"""

from __future__ import annotations

import asyncio

from aiohttp import web


class HttpServer:
    """Managed HTTP server for agent-side web services.
    
    Features:
    - Async/await compatible
    - Shares agent's event loop
    - Graceful shutdown handling
    
    Typical uses:
    - Adding /health endpoint for container health checks
    - Exposing Prometheus metrics
    - Receiving webhook notifications
    
    Usage:
        server = HttpServer("0.0.0.0", 8080)
        server.app.add_routes([web.get('/health', handle_health)])
        asyncio.create_task(server.run())
    """
    
    def __init__(
        self,
        host: str,          # Bind host (e.g. "0.0.0.0" for external access)
        port: int,          # Bind port number
        loop: asyncio.AbstractEventLoop | None = None  # Optional event loop
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._host = host
        self._port = port
        self._app = web.Application(loop=self._loop)
        self._close_future = asyncio.Future[None](loop=self._loop)

    @property
    def app(self) -> web.Application:
        """Access the aiohttp Application to add routes/middleware.
        
        Example:
            @server.app.get("/metrics")
            async def metrics_handler(request):
                return web.Response(text=get_metrics())
        """
        return self._app

    async def run(self) -> None:
        """Start the HTTP server and run until shutdown.
        
        Typically called via asyncio.create_task():
            async with Agent() as agent:
                server = HttpServer(...)
                agent.create_task(server.run())
        """
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        try:
            await self._close_future  # Run until shutdown signal
        finally:
            await self._runner.cleanup()  # Graceful cleanup

    async def aclose(self) -> None:
        """Initiate graceful server shutdown.
        
        Part of standard agent cleanup process:
            async def shutdown():
                await server.aclose()
        """
        if not self._close_future.done():
            self._close_future.set_result(None)
