# Copyright 2023 LiveKit, Inc.
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

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import (
    Callable,
    Coroutine,
)
from urllib.parse import urlparse

import aiohttp
import psutil
from attr import define
from livekit import api
from livekit.protocol import agent, models

from . import aio, consts, http_server, ipc
from .job_request import AcceptData, AvailRes, JobRequest
from .log import logger
from .version import __version__

JobRequestFnc = Callable[[JobRequest], Coroutine]
LoadFnc = Callable[[], float]


def cpu_load_fnc() -> float:
    [m1, m5, m15] = [x / psutil.cpu_count() for x in psutil.getloadavg()]
    return m1


@define(kw_only=True)
class WorkerPermissions:
    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
    can_update_metadata: bool = True
    hidden: bool = False


# NOTE: this object must be pickle-able
@define(kw_only=True)
class WorkerOptions:
    request_fnc: JobRequestFnc
    load_fnc: LoadFnc = cpu_load_fnc
    load_threshold: float = 0.8
    namespace: str = "default"
    permissions: WorkerPermissions = WorkerPermissions()
    worker_type: agent.JobType = agent.JobType.JT_ROOM
    max_retry: int = consts.MAX_RECONNECT_ATTEMPTS
    ws_url: str = "ws://localhost:7880"
    api_key: str | None = None
    api_secret: str | None = None
    host: str = "localhost"
    port: int = 8081


@define(kw_only=True)
class ActiveJob:
    job: agent.Job
    accept_data: AcceptData


class Worker:
    def __init__(
        self,
        opts: WorkerOptions,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        opts.ws_url = opts.ws_url or opts.ws_url or os.environ.get("LIVEKIT_URL") or ""
        opts.api_key = opts.api_key or os.environ.get("LIVEKIT_API_KEY") or ""
        opts.api_secret = opts.api_secret or os.environ.get("LIVEKIT_API_SECRET") or ""

        self._opts = opts
        self._loop = loop or asyncio.get_event_loop()
        self._id = "unregistered"
        self._session = None
        self._closed = False
        self._tasks = set()
        self._pending_assignments: dict[str, asyncio.Future[agent.JobAssignment]] = {}
        self._processes = dict[str, tuple[ipc.JobProcess, ActiveJob]]()
        self._close_future = asyncio.Future(loop=self._loop)

        self._chan = aio.Chan[agent.WorkerMessage](32, loop=self._loop)
        # We use the same event loop as the worker (so the health checks are more accurate)
        self._http_server = http_server.HttpServer(
            opts.host, opts.port, loop=self._loop
        )

    async def run(self):
        logger.info("starting worker", extra={"version": __version__})

        if not self._opts.ws_url:
            raise ValueError("ws_url is required, or set LIVEKIT_URL env var")

        if not self._opts.api_key:
            raise ValueError("api_key is required, or set LIVEKIT_API_KEY env var")

        if not self._opts.api_secret:
            raise ValueError(
                "api_secret is required, or set LIVEKIT_API_SECRET env var"
            )

        self._session = aiohttp.ClientSession()

        async def _worker_ws():
            assert self._session is not None

            retry_count = 0
            while not self._closed:
                try:
                    join_jwt = (
                        api.AccessToken(self._opts.api_key, self._opts.api_secret)
                        .with_grants(api.VideoGrants(agent=True))
                        .to_jwt()
                    )

                    headers = {"Authorization": f"Bearer {join_jwt}"}

                    parse = urlparse(self._opts.ws_url)
                    scheme = parse.scheme
                    if scheme.startswith("http"):
                        scheme = scheme.replace("http", "ws")
                    agent_url = (
                        f"{scheme}://{parse.netloc}/{parse.path.rstrip('/')}/agent"
                    )

                    ws = await self._session.ws_connect(agent_url, headers=headers)
                    retry_count = 0

                    await self._run_ws(ws)
                except Exception as e:
                    if self._closed:
                        break

                    if retry_count >= self._opts.max_retry:
                        raise Exception(
                            f"failed to connect to livekit-server after {retry_count} attempts: {e}"
                        )

                    retry_delay = min(retry_count * 2, 10)
                    retry_count += 1

                    logger.warning(
                        f"failed to connect to livekit-server, retrying in {retry_delay}s: {e}",
                    )
                    await asyncio.sleep(retry_delay)

        async def _http_server():
            await self._http_server.run()

        await asyncio.gather(_worker_ws(), _http_server())
        self._close_future.set_result(None)

    @property
    def id(self) -> str:
        return self._id

    @property
    def active_jobs(self) -> list[ActiveJob]:
        return [active_job for (_, active_job) in self._processes.values()]

    async def aclose(self) -> None:
        if self._closed:
            return

        logger.info("shutting down worker", extra={"id": self.id})

        # shutdown processes before closing the connection to the lkserver
        close_co = []
        for proc, _ in self._processes.values():
            close_co.append(proc.aclose())

        await asyncio.gather(*close_co, return_exceptions=True)

        await self._http_server.aclose()
        assert self._session is not None
        await self._session.close()

        self._closed = True
        self._chan.close()
        await self._close_future

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse):
        closing_ws = False

        # register the worker
        req = agent.WorkerMessage()
        req.register.type = self._opts.worker_type
        req.register.allowed_permissions.CopyFrom(
            models.ParticipantPermission(
                can_publish=self._opts.permissions.can_publish,
                can_subscribe=self._opts.permissions.can_subscribe,
                can_publish_data=self._opts.permissions.can_publish_data,
                can_update_metadata=self._opts.permissions.can_update_metadata,
                hidden=self._opts.permissions.hidden,
                agent=True,
            )
        )
        req.register.namespace = self._opts.namespace
        req.register.version = __version__
        await self._chan.send(req)

        async def load_monitor_task():
            interval = aio.interval(consts.LOAD_INTERVAL)
            registered = True
            while True:
                await interval.tick()
                load = self._opts.load_fnc()
                is_full = load >= self._opts.load_threshold
                should_register = not is_full

                update = agent.UpdateWorkerStatus(
                    load=load,
                    status=(
                        agent.WorkerStatus.WS_FULL
                        if is_full
                        else agent.WorkerStatus.WS_AVAILABLE
                    ),
                )

                if should_register != registered:
                    registered = should_register

                    extra = {"load": load, "threshold": self._opts.load_threshold}
                    if is_full:
                        logger.info(
                            "worker is at full capacity, marking as unavailable",
                            extra=extra,
                        )
                    else:
                        logger.info(
                            "worker is below capacity, marking as available",
                            extra=extra,
                        )

                msg = agent.WorkerMessage(update_worker=update)
                try:
                    self._chan.send_nowait(msg)
                except aio.ChanClosed:
                    return

        async def send_task():
            nonlocal closing_ws
            while True:
                try:
                    msg = await self._chan.recv()
                    await ws.send_bytes(msg.SerializeToString())
                except aio.ChanClosed:
                    closing_ws = True
                    return

        async def recv_task():
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return

                    raise Exception("worker connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.BINARY:
                    logger.warning("unexpected message type: %s", msg.type)
                    continue

                data = msg.data
                msg = agent.ServerMessage()
                msg.ParseFromString(data)
                which = msg.WhichOneof("message")
                if which == "register":
                    self._handle_register(
                        msg.register
                    )  # we assume this is the first message we receive
                elif which == "availability":
                    self._handle_availability(msg.availability)
                elif which == "assignment":
                    self._handle_assignment(msg.assignment)

        await asyncio.gather(send_task(), recv_task(), load_monitor_task())

    def _reload_jobs(self, jobs: list[ActiveJob]):
        for aj in jobs:
            logger.info("reloading job", extra={"job": aj.job})
            # reloading jobs doesn't work on third-party workers
            # so it is ok to use the ws_url from the local worker
            # (also create a token with the worker api key)
            url = self._opts.ws_url

            jwt = (
                api.AccessToken(self._opts.api_key, self._opts.api_secret)
                .with_grants(
                    api.VideoGrants(agent=True, room=aj.job.room.name, room_join=True)
                )
                .with_name(aj.accept_data.name)
                .with_metadata(aj.accept_data.metadata)
                .with_identity(aj.accept_data.identity)
                .to_jwt()
            )

            self._start_process(aj.job, url, jwt, aj.accept_data)

    def _start_process(
        self, job: agent.Job, url: str, token: str, accept_data: AcceptData
    ):
        proc = ipc.JobProcess(job, url, token, accept_data)
        self._processes[job.id] = (proc, ActiveJob(job=job, accept_data=accept_data))

        async def _run_proc():
            try:
                await proc.run()
            except Exception:
                logger.exception(
                    f"error running job process {proc.job.id}",
                    extra=proc.logging_extra(),
                )

            self._processes.pop(proc.job.id)

        task = self._loop.create_task(_run_proc())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _handle_register(self, reg: agent.RegisterWorkerResponse):
        self._id = reg.worker_id
        logger.info(
            "registered worker",
            extra={"id": reg.worker_id, "server_info": reg.server_info},
        )

    def _handle_availability(self, msg: agent.AvailabilityRequest):
        answer_tx, answer_rx = aio.channel(1)  # wait for the user res
        req = JobRequest(msg.job, answer_tx)

        async def _wait_response():
            async def _user_cb():
                try:
                    await self._opts.request_fnc(req)
                except Exception:
                    logger.exception(
                        f"user request handler for job {req.id} failed",
                        extra={"req": req},
                    )

                if not req.answered:
                    logger.warning(
                        f"no answer for job {req.id}, automatically rejecting the job",
                        extra={"req": req},
                    )
                    await _send_ignore_err(
                        self._chan,
                        agent.WorkerMessage(
                            availability=agent.AvailabilityResponse(available=False)
                        ),
                    )

            user_task = self._loop.create_task(_user_cb())

            av: AvailRes = await answer_rx.recv()  # wait for user answer
            msg = agent.WorkerMessage()
            msg.availability.job_id = req.id
            msg.availability.available = av.avail

            if not av.avail:
                await _send_ignore_err(self._chan, msg)
                return

            assert av.data is not None
            assert av.assignment_tx is not None
            msg.availability.participant_identity = av.data.identity
            msg.availability.participant_name = av.data.name
            msg.availability.participant_metadata = av.data.metadata

            wait_assignment = asyncio.Future[agent.JobAssignment]()
            self._pending_assignments[req.id] = wait_assignment

            await _send_ignore_err(self._chan, msg)

            # wait for server assignment
            try:
                await asyncio.wait_for(wait_assignment, consts.ASSIGNMENT_TIMEOUT)
                await av.assignment_tx.send(None)
            except asyncio.TimeoutError as e:
                logger.warning(
                    f"assignment for job {req.id} timed out",
                    extra={"req": req},
                )
                await av.assignment_tx.send(e)
                return
            finally:
                await user_task

            asgn = wait_assignment.result()
            url = asgn.url

            if not url:
                url = self._opts.ws_url

            self._start_process(asgn.job, url, asgn.token, av.data)

        task = self._loop.create_task(_wait_response())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _handle_assignment(self, assignment: agent.JobAssignment):
        job = assignment.job
        if job.id in self._pending_assignments:
            fut = self._pending_assignments.pop(job.id)
            fut.set_result(assignment)
        else:
            logger.warning(
                f"received assignment for unknown job {job.id}",
                extra={"job": job},
            )


async def _send_ignore_err(
    ch: aio.ChanSender[agent.WorkerMessage], msg: agent.WorkerMessage
):
    # Used when we don't care about the result of sending
    # e.g. when closing the worker, we close the channel.
    with contextlib.suppress(aio.ChanClosed):
        await ch.send(msg)
