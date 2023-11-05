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

import logging
import uuid
from dataclasses import dataclass
import asyncio
from typing import AsyncGenerator, Coroutine, Dict, Optional, Callable, Awaitable
from livekit import api, rtc
from .processor.processor import Processor
from ._proto import livekit_agent_pb2 as proto_agent
from ._proto import livekit_models_pb2 as proto_models
from urllib.parse import urlparse
import websockets
import os

MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_INTERVAL = 10
ASSIGNMENT_TIMEOUT = 15


class AssignmentTimeoutError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class JobCancelledError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Worker:
    def __init__(
        self,
        available_cb: Callable[["JobRequest"], Coroutine],
        worker_type: proto_agent.JobType.ValueType,
        ws_url: str = os.environ.get("LIVEKIT_WS_URL", "http://localhost:7880"),
        api_key: str = os.environ.get("LIVEKIT_API_KEY", ""),
        api_secret: str = os.environ.get("LIVEKIT_API_SECRET", "")
    ) -> None:
        parse_res = urlparse(ws_url)
        scheme = parse_res.scheme
        if scheme.startswith("http"):
            scheme = scheme.replace("http", "ws")

        path = parse_res.path.rstrip("/")
        url = f"{scheme}://{parse_res.netloc}/{path}"

        self._lock = asyncio.Lock()
        self._available_cb = available_cb
        self._agent_url = url + "/agent"
        self._rtc_url = url + "/rtc"
        self._wid = "AG-" + str(uuid.uuid4())[:12]
        self._worker_type = worker_type
        self._api_key = api_key
        self._api_secret = api_secret
        self._running = False
        self._running_jobs: list["JobContext"] = []
        self._pending_jobs: Dict[str, asyncio.Future[proto_agent.JobAssignment]] = {}

    async def _connect(self) -> proto_agent.RegisterWorkerResponse:
        join_jwt = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_grants(api.VideoGrants(agent=True))
            .to_jwt()
        )

        req = proto_agent.WorkerMessage()
        req.register.worker_id = self._wid
        req.register.type = self._worker_type

        headers = {"Authorization": f"Bearer {join_jwt}"}
        self._ws = await websockets.connect(self._agent_url, extra_headers=headers)
        await self._send(req)

        res = await self._recv()
        return res.register

    async def _send_availability(
        self, job_id: str, available: bool
    ) -> proto_agent.JobAssignment:
        """
        Send availability to the server, and wait for assignment
        """
        req = proto_agent.WorkerMessage()
        req.availability.available = available
        req.availability.job_id = job_id

        f = asyncio.Future()
        self._pending_jobs[job_id] = f
        await self._send(req)

        try:
            return await asyncio.wait_for(f, ASSIGNMENT_TIMEOUT)
        except asyncio.TimeoutError:
            raise AssignmentTimeoutError(f"assignment timeout for job {job_id}")

    async def _send_job_status(
        self,
        job_id: str,
        status: proto_agent.JobStatus.ValueType,
        error: str,
        user_data: str = "",
    ) -> None:
        req = proto_agent.WorkerMessage()
        req.job_update.job_id = job_id
        req.job_update.status = status
        req.job_update.error = error
        req.job_update.user_data = user_data
        await self._ws.send(req.SerializeToString())

    async def _recv(self) -> proto_agent.ServerMessage:
        message = await self._ws.recv()
        msg = proto_agent.ServerMessage()
        msg.ParseFromString(bytes(message))
        return msg

    async def _send(self, msg: proto_agent.WorkerMessage) -> None:
        try:
            await self._ws.send(msg.SerializeToString())
        except websockets.exceptions.ConnectionClosed:
            # TODO: Implement JobStatus resuming after reconnection
            pass

    async def _handle_new_job(self, job: "JobRequest") -> None:
        """
        Execute the available callback, and automatically deny the job if the callback
        does not send an answer or raises an exception
        """

        try:
            await self._available_cb(job)
        except Exception as e:
            logging.error(f"available callback failed: {e}")
            return

        if not job._answered:
            logging.warn(
                f"user did not answer availability for job {job.id}, rejecting"
            )
            await job.reject()

    async def _message_received(self, msg: proto_agent.ServerMessage) -> None:
        which = msg.WhichOneof("message")
        if which == "availability":
            # server is asking the worker if we are available for a job
            availability = msg.availability
            job = JobRequest(self, availability.job)
            asyncio.ensure_future(self._handle_new_job(job))
        elif which == "assignment":
            # server is assigning a job to the worker
            assignment = msg.assignment
            job_id = assignment.job.id
            f = self._pending_jobs.get(job_id)
            if f is None:
                logging.error(f"received assignment for unknown job {job_id}")
                return

            f.set_result(assignment)
            del self._pending_jobs[job_id]

    async def _reconnect(self) -> bool:
        for i in range(MAX_RECONNECT_ATTEMPTS):
            try:
                reg = await self._connect()
                logging.info(f"worker successfully re-registered: {reg}")
                return True
            except Exception as e:
                logging.error(f"failed to reconnect, attempt {i}: {e}")
                await asyncio.sleep(RECONNECT_INTERVAL)

        return False

    @property
    def id(self) -> str:
        return self._wid

    @property
    def running(self) -> bool:
        return self._running

    async def _run(self) -> None:
        reg = await self._connect()  # initial connection
        logging.info(f"worker successfully registered: {reg}")

        while True:
            try:
                while True:
                    await self._message_received(await self._recv())
            except websockets.exceptions.ConnectionClosed as e:
                if self._running:
                    logging.error(f"connection closed, trying to reconnect: {e}")
                    if not await self._reconnect():
                        break
            except Exception as e:
                logging.error(f"error while running worker: {e}")
                break

    async def run(self):
        if self._running:
            raise Exception("worker is already running")

        self._running = True

        try:
            await self._run()
        except asyncio.CancelledError:
            logging.info(f"cancel received for worker {self._wid}, closing...")
            await asyncio.shield(self.close())

        await self.close()

    async def close(self) -> None:
        async with self._lock:
            if not self._running:
                return

            logging.info(f"closing worker {self._wid}")

            # close the websocket and all running jobs
            await self._ws.close()
            await asyncio.gather(*[job.close() for job in self._running_jobs])

            self._running = False


class JobContext:
    def __init__(
        self,
        id: str,
        worker: Worker,
        room: rtc.Room,
        participant: Optional[rtc.RemoteParticipant],
    ) -> None:
        self._id = id
        self._worker = worker
        self._room = room
        self._participant = participant
        self._close_event = asyncio.Event()
        self._processors: list[Processor] = []
        self._closed = False
        self._lock = asyncio.Lock()
        self._worker._running_jobs.append(self)

    @property
    def id(self) -> str:
        return self._id

    @property
    def room(self) -> rtc.Room:
        return self._room

    @property
    def participant(self) -> Optional[rtc.RemoteParticipant]:
        return self._participant

    def add_processor(self, processor: Processor) -> None:
        self._processors.append(processor)

    async def wait_close(self) -> None:
        """
        Wait until close is requested
        """
        await self._close_event.wait()

    async def close(self) -> None:
        """
        Close the job and cleanup resources (linked processors & tasks)
        """
        async with self._lock:
            if self._closed:
                return

            self._closed = True
            self._close_event.set()

            # close all processors
            for p in self._processors:
                await p.close()

            self._worker._running_jobs.remove(self)

    async def update_status(
        self,
        status: proto_agent.JobStatus.ValueType,
        error: str = "",
        user_data: str = "",
    ) -> None:
        await self._worker._send_job_status(self._id, status, error, user_data)


class JobRequest:
    def __init__(
        self,
        worker: Worker,
        job_info: proto_agent.Job,
    ) -> None:
        self._worker = worker
        self._info = job_info
        self._room = rtc.Room()
        self._answered = False
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def room(self) -> proto_models.Room:
        return self._info.room

    @property
    def participant(self) -> Optional[proto_models.ParticipantInfo]:
        if self._info.participant.sid:
            return self._info.participant
        return None

    def _assert_not_answered(self) -> None:
        if self._answered:
            raise Exception("job already answered")

    async def reject(self) -> None:
        """
        Tell the server that we cannot handle the job
        """
        async with self._lock:
            self._assert_not_answered()
            self._answered = True
            await self._worker._send_availability(self.id, False)

    async def accept(
        self,
        agent: Callable[[JobContext], Coroutine],
        name: str = "",
        identity: str = "",
        metadata: str = "",
    ):
        """
        Tell the server that we can handle the job, if the server then assigns the job to us,
        we will connect to the room and call the agent callback
        """
        async with self._lock:
            self._assert_not_answered()
            self._answered = True

            identity = identity or "agent-" + self.id
            grants = api.VideoGrants(
                room=self.room.name,
                room_join=True,
                agent=True,
            )

            jwt = (
                api.AccessToken(self._worker._api_key, self._worker._api_secret)
                .with_identity(identity)
                .with_grants(grants)
                .with_metadata(metadata)
                .with_name(name)
                .to_jwt()
            )

            # raise AssignmentTimeoutError if assignment times out
            _ = await self._worker._send_availability(self.id, True)

            try:
                options = rtc.RoomOptions(auto_subscribe=True)
                await self._room.connect(self._worker._rtc_url, jwt, options)
            except rtc.ConnectError as e:
                logging.error(
                    f"failed to connect to the room, cancelling job {self.id}: {e}"
                )
                await self._worker._send_job_status(
                    self.id, proto_agent.JobStatus.JS_FAILED, str(e)
                )
                raise e

            participant = None
            if self.participant is not None:
                # cancel the job if the participant cannot be found
                # this can happen if the participant has left the room before the agent gets the job
                participant = self._room.participants.get(self.participant.sid)
                if participant is None:
                    logging.warn(
                        f"participant '{self.participant.sid}' not found, cancelling job {self.id}"
                    )
                    await self._worker._send_job_status(
                        self.id,
                        proto_agent.JobStatus.JS_FAILED,
                        "participant not found",
                    )
                    await self._room.disconnect()
                    raise JobCancelledError(
                        f"participant '{self.participant.sid}' not found"
                    )

            # start the agent
            job_ctx = JobContext(self.id, self._worker, self._room, participant)
            asyncio.ensure_future(agent(job_ctx))
