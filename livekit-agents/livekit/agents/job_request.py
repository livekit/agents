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

import asyncio
import logging
from typing import Callable, Coroutine, Optional, TYPE_CHECKING, Union
from .job_context import JobContext
from livekit import rtc, protocol, api
from livekit.protocol import agent as proto_agent

# TODO: refactor worker so we can avoid this circular import
if TYPE_CHECKING:
    from .worker import Worker

AutoSubscribeCallback = Callable[[rtc.TrackPublication, rtc.RemoteParticipant], bool]
AutoDisconnectCallback = Callable[[JobContext], bool]
AgentEntry = Callable[[JobContext], Coroutine]


class AutoSubscribe:
    """Helper callbacks for common subscribe scenarios"""

    @staticmethod
    def SUBSCRIBE_ALL(
        publication: rtc.TrackPublication, participant: rtc.RemoteParticipant
    ) -> bool:
        """Subscribe to all tracks automatically. This will also set the LiveKit room option
        auto_subscribe to true as an optimization."""
        return True

    @staticmethod
    def SUBSCRIBE_NONE(
        publication: rtc.TrackPublication, participant: rtc.RemoteParticipant
    ) -> bool:
        """Don't subscribe to any tracks automatically"""
        return False

    @staticmethod
    def VIDEO_ONLY(
        publication: rtc.TrackPublication, participant: rtc.RemoteParticipant
    ) -> bool:
        """Subscribe to video tracks automatically"""
        return publication.kind == rtc.TrackKind.KIND_VIDEO

    @staticmethod
    def AUDIO_ONLY(
        publication: rtc.TrackPublication, participant: rtc.RemoteParticipant
    ) -> bool:
        """Subscribe to audio tracks automatically"""
        return publication.kind == rtc.TrackKind.KIND_AUDIO


class AutoDisconnect:
    """Helper callbacks for common auto disconnect scenarios"""

    @staticmethod
    def ROOM_EMPTY(ctx: JobContext) -> bool:
        if len(ctx.room.participants) == 0:
            return True

        if len(ctx.room.participants) > 1:
            return False

        for p in ctx.room.participants.values():
            if p.identity == ctx.agent.identity:
                return True

        return False

    @staticmethod
    def PUBLISHER_LEFT(ctx: JobContext) -> bool:
        if ctx.participant is None:
            logging.error(
                "Incorrect usage of PUBLISHER_LEFT, JobContext is tied to a Participant",
                extra=ctx.logging_extra,
            )
            return False

        return ctx.room.participants.get(ctx.participant.sid) is None

    @staticmethod
    def DEFAULT(ctx: JobContext):
        """
        Default shutdown options. If the agent is tied to a participant, it will shut down when that participant leaves.
        If the agent is not tied to a participant, it will shut down when the agent is the only remaining participant.
        """
        if ctx.participant is not None:
            return AutoDisconnect.PUBLISHER_LEFT(ctx)

        return AutoDisconnect.ROOM_EMPTY(ctx)


class JobRequest:
    """Represents a new job from the server, this worker can either accept or reject it."""

    def __init__(
        self,
        worker: "Worker",
        job_info: proto_agent.Job,
        simulated: bool = False,
    ) -> None:
        self._worker = worker
        self._info = job_info
        self._room = rtc.Room()
        self._answered = False
        self._simulated = simulated
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def room(self) -> protocol.models.Room:
        return self._info.room

    @property
    def publisher(self) -> Optional[protocol.models.ParticipantInfo]:
        return self._info.participant

    async def reject(self) -> None:
        """Tell the server we cannot handle this job"""
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

            self._answered = True
            if not self._simulated:
                await self._worker._send_availability(self.id, False)

        logging.info("rejected job %s", self.id)

    async def accept(
        self,
        agent: AgentEntry,
        auto_subscribe: Union[
            AutoSubscribeCallback, bool
        ] = AutoSubscribe.SUBSCRIBE_NONE,
        auto_disconnect: AutoDisconnectCallback = AutoDisconnect.DEFAULT,
        grants: api.VideoGrants = api.VideoGrants(),
        name: Optional[str] = None,
        identity: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> None:
        """Signal to the LiveKit Server that we can handle the job, and connect to the room.

        Args:
            agent:
                Your agent entrypoint.

            auto_subscribe:
                Callback that is called when a track is published in the room. Return True to subscribe to the track.

            auto_disconnect:
                Callback that is called when the agent should disconnect from the room. Return True to disconnect.
                The callback is called when:
                - Initially once the agent has connected to the room
                - A participant leaves the room

            grants:
                Additional grants to give to the agent participant in its token.
                Defaults to None.

            name:
                Name of the agent participant. Defaults to "".

            identity:
                Identity of the agent participant. Defaults to "".

            metadata:
                Metadata of the agent participant. Defaults to "".
        """
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

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

            def disconnect_if_needed(*_):
                if auto_disconnect(job_ctx):
                    asyncio.ensure_future(
                        job_ctx.disconnect(),
                        loop=self._worker._loop,
                    )

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
