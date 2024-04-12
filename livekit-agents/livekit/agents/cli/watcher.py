from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing
import pathlib
import threading
from importlib.metadata import Distribution
from typing import Any, Callable, Set

import watchfiles

from .. import apipe, ipc_enc
from ..log import logger
from ..plugin import Plugin
from ..worker import Worker
from . import protocol


class WatchServer:
    def __init__(
        self,
        worker_runner: Callable[[protocol.CliArgs], Any],
        main_file: pathlib.Path,
        args: protocol.CliArgs,
        watch_plugins: bool = True,
    ) -> None:
        self._pch, args.cch = multiprocessing.Pipe(duplex=True)
        self._worker_runner = worker_runner
        self._main_file = main_file
        self._args = args
        self._watch_plugins = watch_plugins
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)

        self._jobs_recv = threading.Event()
        self._lock = threading.Lock()
        self._worker_valid = True

    def run(self) -> None:
        packages = []

        if self._watch_plugins:
            # also watch plugins that are installed in editable mode
            # this is particulary useful when developing plugins
            packages.append(Distribution.from_name("livekit.agents"))
            for p in Plugin.registered_plugins:
                packages.append(Distribution.from_name(p.package))

        paths: list[str | pathlib.Path] = [self._main_file.absolute()]
        for p in packages:
            # https://packaging.python.org/en/latest/specifications/direct-url/
            durl = p.read_text("direct_url.json")
            if not durl:
                continue

            durl = json.loads(durl)
            dir_info = durl.get("dir_info", {})
            if dir_info.get("editable", False):
                path = durl.get("url")
                if path.startswith("file://"):
                    paths.append(path[7:])

        for p in paths:
            logger.info(f"Watching {p}")

        self._read_thread.start()
        watchfiles.run_process(
            *paths,
            target=self._worker_runner,
            args=(self._args,),
            callback=self._on_reload,
        )

    def _read_loop(self) -> None:
        try:
            active_jobs = []
            while True:
                msg = ipc_enc.read_msg(self._pch, protocol.IPC_MESSAGES)
                if isinstance(msg, protocol.ActiveJobsResponse):
                    with self._lock:
                        if self._worker_valid:
                            active_jobs = msg.jobs

                    self._jobs_recv.set()
                elif isinstance(msg, protocol.ReloadJobsRequest):
                    ipc_enc.write_msg(
                        self._pch, protocol.ReloadJobsResponse(jobs=active_jobs)
                    )
                    self._jobs = []
                elif isinstance(msg, protocol.Reloaded):
                    with self._lock:
                        self._worker_valid = True

        except Exception:
            logger.exception("watcher failed")

    def _on_reload(self, _: Set[watchfiles.main.FileChange]):
        try:
            # get the current active jobs before reloading
            ipc_enc.write_msg(self._pch, protocol.ActiveJobsRequest())
            self._jobs_recv.wait(timeout=1)
        except Exception:
            logger.exception("failed to request active jobs")
        finally:
            with self._lock:
                self._worker_valid = False
            self._jobs_recv.clear()


class WatchClient:
    def __init__(
        self,
        worker: Worker,
        cch: ipc_enc.ProcessPipe,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._worker = worker
        self._apipe = apipe.AsyncPipe(cch, self._loop, protocol.IPC_MESSAGES)

    def start(self) -> None:
        self._main_task = self._loop.create_task(self._run())

    async def _run(self) -> None:
        try:
            await self.send(protocol.ReloadJobsRequest())

            while True:
                msg = await self._apipe.read()

                if isinstance(msg, protocol.ActiveJobsRequest):
                    jobs = self._worker.active_jobs
                    await self.send(protocol.ActiveJobsResponse(jobs=jobs))
                elif isinstance(msg, protocol.ReloadJobsResponse):
                    self._worker._reload_jobs(msg.jobs)
                    await self.send(protocol.Reloaded())

        except Exception:
            logger.exception("watcher failed")

    async def send(self, msg: ipc_enc.Message) -> None:
        await self._apipe.write(msg)

    async def aclose(self) -> None:
        if not self._main_task:
            return

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task
