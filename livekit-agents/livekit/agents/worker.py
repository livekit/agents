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
from attr import define

import psutil
import asyncio
import contextlib
import aiohttp
import os
from typing import (
    Callable,
    Coroutine,
)
from .log import logger
from urllib.parse import urlparse
from livekit.protocol import agent, models
from livekit import api
from .job_request import AvailRes, JobRequest, AgentEntry
from .version import __version__

from . import aio
from . import consts
from . import ipc
from . import http_server

JobRequestFnc = Callable[[JobRequest], Coroutine]
LoadFnc = Callable[[ipc.JobProcess | None], float]


def cpu_load_fnc(_: ipc.JobProcess | None = None) -> float:
    return psutil.cpu_percent()


@define(kw_only=True, frozen=True)
class WorkerOptions:
    request_fnc: JobRequestFnc
    load_fnc: LoadFnc
    namespace: str
    permissions: models.ParticipantPermission
    worker_type: agent.JobType.ValueType
    max_retry: int
    ws_url: str
    agent_url: str
    api_key: str
    api_secret: str
    host: str
    port: int


class Worker:
    def __init__(
        self,
        request_fnc: JobRequestFnc,
        *,
        load_fnc: LoadFnc = cpu_load_fnc,
        namespace: str = "default",
        permissions: models.ParticipantPermission = models.ParticipantPermission(),
        worker_type: agent.JobType.ValueType = agent.JobType.JT_ROOM,
        max_retry: int = consts.MAX_RECONNECT_ATTEMPTS,
        ws_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        host: str = "localhost",
        port: int = 80,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        ws_url = ws_url or os.environ.get("LIVEKIT_URL", "")
        api_key = api_key or os.environ.get("LIVEKIT_API_KEY", "")
        api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET", "")

        if not ws_url:
            raise ValueError("ws_url is required, or set LIVEKIT_URL env var")

        if not api_key:
            raise ValueError("api_key is required, or set LIVEKIT_API_KEY env var")

        if not api_secret:
            raise ValueError(
                "api_secret is required, or set LIVEKIT_API_SECRET env var"
            )

        parse = urlparse(ws_url)
        scheme = parse.scheme
        if scheme.startswith("http"):
            scheme = scheme.replace("http", "ws")
        agent_url = f"{scheme}://{parse.netloc}/{parse.path.rstrip("/")}/agent"

        self._opts = WorkerOptions(
            request_fnc=request_fnc,
            load_fnc=load_fnc,
            namespace=namespace,
            permissions=permissions,
            worker_type=worker_type,
            max_retry=max_retry,
            ws_url=ws_url,
            agent_url=agent_url,
            api_key=api_key,
            api_secret=api_secret,
            host=host,
            port=port,
        )

        self._loop = loop or asyncio.get_event_loop()
        self._id = "unregistered"
        self._session = aiohttp.ClientSession()
        self._closed = False
        self._tasks = set()
        self._pending_assignments: dict[str, asyncio.Future[agent.JobAssignment]] = {}
        self._processes = dict[str, ipc.JobProcess]()
        self._close_future = asyncio.Future()

        # We use the same event loop as the worker (so the health checks are more accurate)
        self._http_server = http_server.HttpServer(host, port, loop=self._loop)

    async def run(self):
        async def _worker_ws():
            retry_count = 0
            while not self._closed:
                try:
                    join_jwt = (
                        api.AccessToken(self._opts.api_key, self._opts.api_secret)
                        .with_grants(api.VideoGrants(agent=True))
                        .to_jwt()
                    )

                    headers = {"Authorization": f"Bearer {join_jwt}"}
                    ws = await self._session.ws_connect(
                        self._opts.agent_url, headers=headers
                    )
                    retry_count = 0

                    await self._run_ws(ws)
                except Exception as e:
                    if retry_count >= self._opts.max_retry:
                        logger.exception(
                            f"failed to connect to livekit-server after {retry_count} attempts"
                        )
                        break

                    retry_delay = min(retry_count * 2, 10)
                    retry_count += 1

                    logger.warning(
                        f"failed to connect to livekit-server, retrying in {retry_delay}s",
                        exc_info=e,
                    )
                    await asyncio.sleep(retry_delay)

        async def _http_server():
            await self._http_server.run()

        await asyncio.gather(_worker_ws(), _http_server())
        self._close_future.set_result(None)

    @property
    def id(self) -> str:
        return self._id

    async def aclose(self) -> None:
        self._closed = True
        self._chan.close()
        await self._http_server.aclose()
        await self._close_future

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse):
        self._chan = aio.Chan[agent.WorkerMessage](32, loop=self._loop)
        closing_ws = False

        # register the worker
        req = agent.WorkerMessage()
        req.register.type = self._opts.worker_type
        req.register.allowed_permissions.CopyFrom(self._opts.permissions)
        req.register.namespace = self._opts.namespace
        req.register.version = __version__
        await self._chan.send(req)

        async def load_monitor_task():
            interval = aio.interval(consts.LOAD_INTERVAL)
            while True:
                await interval.tick()
                load = self._opts.load_fnc(None)
                msg = agent.WorkerMessage(
                    update_worker=agent.UpdateWorkerStatus(load=load)
                )
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

    def _start_process(self, job: agent.Job, url: str, token: str, entry: AgentEntry):
        proc = ipc.JobProcess(job, url, token, entry)
        self._processes[job.id] = proc

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
            f"registered worker {reg.worker_id}",
            extra={"id": reg.worker_id, "server_info": reg.server_info},
        )

    def _handle_availability(self, msg: agent.AvailabilityRequest):
        answer_tx, answer_rx = aio.channel(1)  # wait for the user res
        req = JobRequest(msg.job, answer_tx)

        async def _wait_response():
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
                await send_ignore_err(
                    self._chan,
                    agent.WorkerMessage(
                        availability=agent.AvailabilityResponse(available=False)
                    ),
                )

            av: AvailRes = await answer_rx.recv()
            msg = agent.WorkerMessage()
            msg.availability.available = av.avail

            if not av.avail:
                await send_ignore_err(self._chan, msg)
                return

            assert av.data is not None
            msg.availability.job_id = req.id
            msg.availability.participant_identity = av.data.identity
            msg.availability.participant_name = av.data.name
            msg.availability.participant_metadata = av.data.metadata

            wait_assignment = asyncio.Future[agent.JobAssignment]()
            self._pending_assignments[req.id] = wait_assignment

            await send_ignore_err(self._chan, msg)

            try:
                await asyncio.wait_for(wait_assignment, consts.ASSIGNMENT_TIMEOUT)
                await av.data.assignment_tx.send(None)
            except asyncio.TimeoutError as e:
                await av.data.assignment_tx.send(e)
                logger.warning(
                    f"assignment for job {req.id} timed out",
                    extra={"req": req},
                )
                return

            asgn = wait_assignment.result()
            url = asgn.url
            if not url:
                url = self._opts.ws_url

            self._start_process(asgn.job, url, asgn.token, av.data.entry)

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


async def send_ignore_err(
    ch: aio.ChanSender[agent.WorkerMessage], msg: agent.WorkerMessage
):
    # Used when we don't care about the result of sending
    # e.g. when closing the worker, we close the channel.
    with contextlib.suppress(aio.ChanClosed):
        await ch.send(msg)
