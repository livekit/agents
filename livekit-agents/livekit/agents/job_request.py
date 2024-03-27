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
from enum import Enum
from typing import Optional

from livekit import api, protocol, rtc
from livekit.protocol import agent

from . import aio
from .exceptions import JobRequestAnsweredError
from .job_context import JobContext
from .log import logger

AutoSubscribe = Enum(
    "AutoSubscribe", ["SUBSCRIBE_ALL", "SUBSCRIBE_NONE", "VIDEO_ONLY", "AUDIO_ONLY"]
)

AutoDisconnect = Enum(
    "AutoDisconnect",
    ["ROOM_EMPTY", "PUBLISHER_LEFT", "NONE"],
)


class JobRequest:
    def __init__(self, job: agent.Job, answer_tx: aio.ChanSender[bool]) -> None:
        self._job = job
        self._lock = asyncio.Lock()
        self._answer_tx = answer_tx
        self._answered = False

    @property
    def id(self) -> str:
        return self._job.id

    @property
    def job(self) -> agent.Job:
        return self._job

    @property
    def room(self) -> protocol.models.Room:
        return self._job.room

    @property
    def publisher(self) -> protocol.models.ParticipantInfo | None:
        return self._job.participant

    async def reject(self) -> None:
        async with self._lock:
            if self._answered:
                raise JobRequestAnsweredError

            await self._answer_tx.send(False)
            self._answer_tx.close()

        logger.info("rejected job %s", self.id)

    async def accept(
        self,
        agent: AgentEntry,
        auto_subscribe: AutoSubscribeCallback | bool = AutoSubscribe.SUBSCRIBE_NONE,
        auto_disconnect: AutoDisconnectCallback = AutoDisconnect.DEFAULT,
        disconnect_grace_period: float = 30,
        grants: api.VideoGrants = api.VideoGrants(),
        name: Optional[str] = None,
        identity: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> None:
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

            proc = self._ipc_server.new_process(
                self._info,
                self._worker._rtc_url,
                self._worker._api_key,
                self._worker._api_secret,
            )

            self._answered = True

            identity = identity or "agent-" + self.id
            grants.room_join = True
            grants.agent = True
            grants.room = self.room.name
            grants.can_update_own_metadata = True

            jwt = (
                api.AccessToken(self._worker._api_key, self._worker._api_secret)
                .with_identity(identity)
                .with_grants(grants)
                .with_metadata(metadata or "")
                .with_name(name or "")
                .to_jwt()
            )

            # raise AssignmentTimeoutError if assignment times out
            if not self._simulated:
                _ = await self._worker._send_availability(self.id, True)

            try:
                options = rtc.RoomOptions(
                    auto_subscribe=auto_subscribe == AutoSubscribe.SUBSCRIBE_ALL
                    or auto_subscribe is True
                )
                await self._room.connect(self._worker._rtc_url, jwt, options)
            except rtc.ConnectError as e:
                logging.exception(
                    "failed to connect to the room, cancelling job %s",
                    self.id,
                    extra={
                        "job_id": self.id,
                        "room": self.room.name,
                        "agent_identity": identity,
                    },
                )
                await self._worker._send_job_status(
                    self.id, proto_agent.JobStatus.JS_FAILED, str(e)
                )
                raise

            participant: Optional[rtc.Participant] = None
            if self._info.participant:
                participant = self._room.participants.get(self._info.participant.sid)

            job_ctx = JobContext(
                self.id,
                self._worker,
                self._room,
                participant=participant,
            )
            job_ctx.create_task(agent(job_ctx))

            async def disconnect_if_needed_after_grace_period():
                await asyncio.sleep(disconnect_grace_period)
                if auto_disconnect(job_ctx):
                    await job_ctx.disconnect()

            def disconnect_if_needed(*_):
                if auto_disconnect(job_ctx):
                    # If the auto_disconnect callback returns True, start the grace period.
                    # If there is already a grace period task running, keep that one running
                    # instead of starting a new one because we don't want to reset the timer.
                    if self._grace_period_disconnect_task is None:
                        self._grace_period_disconnect_task = asyncio.ensure_future(
                            disconnect_if_needed_after_grace_period(),
                            loop=self._worker._loop,
                        )
                else:
                    # If, during the grace period, the auto_disconnect callback returns False,
                    # cancel the grace period task
                    if self._grace_period_disconnect_task:
                        self._grace_period_disconnect_task.cancel()
                        self._grace_period_disconnect_task = None

            self._room.on("participant_disconnected", disconnect_if_needed)

            async def job_validity():
                # let time to the server to forward participant info
                await asyncio.sleep(15)
                disconnect_if_needed()

            job_ctx.create_task(job_validity())

            if auto_subscribe != AutoSubscribe.SUBSCRIBE_ALL and not isinstance(
                auto_subscribe, bool
            ):

                @self._room.on("track_published")
                def on_track_published(
                    publication: rtc.RemoteTrackPublication,
                    participant: rtc.RemoteParticipant,
                ):
                    if not auto_subscribe(publication, participant):
                        return

                    publication.set_subscribed(True)

                for participant in self._room.participants.values():
                    for publication in participant.tracks.values():
                        if not auto_subscribe(publication, participant):
                            continue

                        publication.set_subscribed(True)

        logging.info("accepted job %s", self.id, extra=job_ctx.logging_extra)
