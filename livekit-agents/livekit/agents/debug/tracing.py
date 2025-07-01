from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal

from aiohttp import web

from .. import job

if TYPE_CHECKING:
    from ..worker import Worker


class TracingGraph:
    def __init__(
        self,
        title: str,
        y_label: str,
        x_label: str,
        y_range: tuple[float, float] | None,
        x_type: Literal["time", "value"],
        max_data_points: int,
    ) -> None:
        self._title = title
        self._y_label = y_label
        self._x_label = x_label
        self._y_range = y_range
        self._max_data_points = max_data_points
        self._x_type = x_type
        self._data: list[tuple[float | int, float]] = []

    def plot(self, x: float | int, y: float) -> None:
        self._data.append((x, y))
        if len(self._data) > self._max_data_points:
            self._data.pop(0)


class TracingHandle:
    def __init__(self) -> None:
        self._kv: dict[str, str | dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._graphs: list[TracingGraph] = []

    def store_kv(self, key: str, value: str | dict[str, Any]) -> None:
        self._kv[key] = value

    def log_event(self, name: str, data: dict[str, Any] | None) -> None:
        self._events.append({"name": name, "data": data, "timestamp": time.time()})

    def add_graph(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str,
        y_range: tuple[float, float] | None = None,
        x_type: Literal["time", "value"] = "value",
        max_data_points: int = 512,
    ) -> TracingGraph:
        graph = TracingGraph(title, y_label, x_label, y_range, x_type, max_data_points)
        self._graphs.append(graph)
        return graph

    def _export(self) -> dict[str, Any]:
        return {
            "kv": self._kv,
            "events": self._events,
            "graph": [
                {
                    "title": chart._title,
                    "x_label": chart._x_label,
                    "y_label": chart._y_label,
                    "y_range": chart._y_range,
                    "x_type": chart._x_type,
                    "data": chart._data,
                }
                for chart in self._graphs
            ],
        }


class Tracing:
    _instance = None

    def __init__(self) -> None:
        self._handles: dict[str, TracingHandle] = {}

    @classmethod
    def with_handle(cls, handle: str) -> TracingHandle:
        if cls._instance is None:
            cls._instance = cls()

        if handle not in cls._instance._handles:
            cls._instance._handles[handle] = TracingHandle()

        return cls._instance._handles[handle]

    @staticmethod
    def _get_current_handle() -> TracingHandle:
        try:
            job_id = job.get_job_context().job.id
            return Tracing._get_job_handle(job_id)
        except RuntimeError:
            pass

        return Tracing.with_handle("global")

    @staticmethod
    def _get_job_handle(job_id: str) -> TracingHandle:
        return Tracing.with_handle(f"job_{job_id}")

    @staticmethod
    def store_kv(key: str, value: str | dict[str, Any]) -> None:
        Tracing._get_current_handle().store_kv(key, value)

    @staticmethod
    def log_event(name: str, data: dict[str, Any] | None = None) -> None:
        Tracing._get_current_handle().log_event(name, data)

    @staticmethod
    def add_graph(
        *,
        title: str,
        x_label: str,
        y_label: str,
        y_range: tuple[float, float] | None = None,
        x_type: Literal["time", "value"] = "value",
        max_data_points: int = 512,
    ) -> TracingGraph:
        return Tracing._get_current_handle().add_graph(
            title=title,
            x_label=x_label,
            y_label=y_label,
            y_range=y_range,
            x_type=x_type,
            max_data_points=max_data_points,
        )


def _create_tracing_app(w: Worker) -> web.Application:
    async def tracing_index(request: web.Request) -> web.Response:
        import importlib.resources

        import aiofiles

        with importlib.resources.path("livekit.agents.debug", "index.html") as path:
            async with aiofiles.open(path) as f:
                content = await f.read()

        return web.Response(text=content, content_type="text/html")

    async def runners(request: web.Request) -> web.Response:
        data = {
            "runners": [
                {
                    "id": runner.id,
                    "status": runner.status.name,
                    "job_id": runner.running_job.job.id if runner.running_job else None,
                    "room": runner.running_job.job.room.name if runner.running_job else None,
                }
                for runner in w._proc_pool.processes
                if runner.started and runner.running_job
            ]
        }

        return web.json_response(data)

    async def runner(request: web.Request) -> web.Response:
        runner_id = request.query.get("id")
        if not runner_id:
            return web.Response(status=400)

        # TODO: avoid
        runner = next((r for r in w._proc_pool.processes if r.id == runner_id), None)
        if not runner:
            return web.Response(status=404)

        info = await asyncio.wait_for(runner.tracing_info(), timeout=5.0)  # proc could be stuck
        return web.json_response({"tracing": info})

    async def worker(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": w.id,
                "tracing": Tracing.with_handle("global")._export(),
            }
        )

    app = web.Application()
    app.add_routes([web.get("", tracing_index)])
    app.add_routes([web.get("/", tracing_index)])
    app.add_routes([web.get("/runners/", runners)])
    app.add_routes([web.get("/runner/", runner)])
    app.add_routes([web.get("/worker/", worker)])
    return app
