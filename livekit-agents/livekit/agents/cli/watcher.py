from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import socket
import urllib.parse
from importlib.metadata import Distribution, PackageNotFoundError
from typing import Any, Callable, Set

import watchfiles

from .. import utils
from ..ipc import channel
from ..log import DEV_LEVEL, logger
from ..plugin import Plugin
from ..worker import Worker
from . import proto


def _find_watchable_paths(main_file: pathlib.Path) -> list[pathlib.Path]:
    packages: list[Distribution] = []

    # also watch agents plugins in editable mode
    def _try_add(name: str) -> bool:
        nonlocal packages
        try:
            dist = Distribution.from_name(name)
            packages.append(dist)
            return True
        except PackageNotFoundError:
            return False

    if not _try_add("livekit.agents"):
        _try_add("livekit-agents")

    for plugin in Plugin.registered_plugins:
        if not _try_add(plugin.package):
            _try_add(plugin.package.replace(".", "-"))

    paths: list[pathlib.Path] = [main_file.absolute()]
    for pkg in packages:
        # https://packaging.python.org/en/latest/specifications/direct-url/
        durl = pkg.read_text("direct_url.json")
        if not durl:
            continue

        durl_json: dict[str, Any] = json.loads(durl)
        dir_info = durl_json.get("dir_info", {})
        if dir_info.get("editable", False):
            path: str | None = durl_json.get("url")
            if path and path.startswith("file://"):
                parsed_url = urllib.parse.urlparse(path)
                file_path = pathlib.Path(urllib.parse.unquote(parsed_url.path))
                paths.append(file_path)

    return paths


class WatchServer:
    def __init__(
        self,
        worker_runner: Callable[[proto.CliArgs], Any],
        main_file: pathlib.Path,
        cli_args: proto.CliArgs,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._mp_pch, cli_args.mp_cch = socket.socketpair()
        self._cli_args = cli_args
        self._worker_runner = worker_runner
        self._main_file = main_file
        self._loop = loop

        self._recv_jobs_fut = asyncio.Future[None]()
        self._reloading_jobs = False

    async def run(self) -> None:
        watch_paths = _find_watchable_paths(self._main_file)
        for pth in watch_paths:
            logger.log(DEV_LEVEL, f"Watching {pth}")

        self._pch = await utils.aio.duplex_unix._AsyncDuplex.open(self._mp_pch)

        read_ipc_task = self._loop.create_task(self._read_ipc_task())
        await watchfiles.arun_process(
            *watch_paths,
            target=self._worker_runner,
            args=(self._cli_args,),
            watch_filter=watchfiles.filters.PythonFilter(),
            callback=self._on_reload,
        )

        await utils.aio.gracefully_cancel(read_ipc_task)

    async def _on_reload(self, _: Set[watchfiles.main.FileChange]) -> None:
        if self._reloading_jobs:
            return

        await channel.asend_message(self._pch, proto.ActiveJobsRequest())
        self._working_reloading = True

        self._recv_jobs_fut = asyncio.Future()
        with contextlib.suppress(asyncio.TimeoutError):
            # wait max 1.5s to get the active jobs
            await asyncio.wait_for(self._recv_jobs_fut, timeout=1.5)

    @utils.log_exceptions(logger=logger)
    async def _read_ipc_task(self) -> None:
        active_jobs = []
        while True:
            msg = await channel.arecv_message(self._pch, proto.IPC_MESSAGES)
            if isinstance(msg, proto.ActiveJobsResponse) and self._working_reloading:
                active_jobs = msg.jobs
                with contextlib.suppress(asyncio.InvalidStateError):
                    self._recv_jobs_fut.set_result(None)
            if isinstance(msg, proto.ReloadJobsRequest):
                await channel.asend_message(
                    self._pch, proto.ReloadJobsResponse(jobs=active_jobs)
                )
            if isinstance(msg, proto.Reloaded):
                self._working_reloading = False


class WatchClient:
    def __init__(
        self,
        worker: Worker,
        mp_cch: socket.socket,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._worker = worker
        self._mp_cch = mp_cch

    def start(self) -> None:
        self._main_task = self._loop.create_task(self._run())

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        self._cch = await utils.aio.duplex_unix._AsyncDuplex.open(self._mp_cch)

        await channel.asend_message(self._cch, proto.ReloadJobsRequest())
        while True:
            try:
                msg = await channel.arecv_message(self._cch, proto.IPC_MESSAGES)
            except utils.aio.duplex_unix.DuplexClosed:
                break

            if isinstance(msg, proto.ActiveJobsRequest):
                jobs = self._worker.active_jobs

                await channel.asend_message(
                    self._cch, proto.ActiveJobsResponse(jobs=jobs)
                )
            elif isinstance(msg, proto.ReloadJobsResponse):
                # TODO(theomonnom): wait for the worker to be fully initialized/connected
                await self._worker._reload_jobs(msg.jobs)
                await channel.asend_message(self._cch, proto.Reloaded())

    async def aclose(self) -> None:
        if not self._main_task:
            return

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

        await self._cch.aclose()
