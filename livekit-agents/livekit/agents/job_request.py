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
from dataclasses import dataclass
from typing import Callable, Coroutine, Optional, TYPE_CHECKING
from .job_context import JobContext
from livekit import rtc, protocol, api
from livekit.protocol import agent as proto_agent

# TODO: refactor worker so we can avoid this circular import
if TYPE_CHECKING:
    from .worker import Worker


@dataclass
class SubscribeCallbacks:
    """Helper callbacks for common subscribe scenarios"""

    @staticmethod
    def SUBSCRIBE_ALL(
        publication: rtc.TrackPublication, participant: rtc.RemoteParticipant
    ) -> bool:
        """
        Subscribe to all tracks automatically. This will also set the LiveKit room option
        auto_subscribe to true as an optimization.
        """
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
        """
        Subscribe to video tracks automatically
        """
        return publication.kind == rtc.TrackKind.KIND_VIDEO

    @staticmethod
    def AUDIO_ONLY(
        publication: rtc.TrackPublication, participant: rtc.RemoteParticipant
    ) -> bool:
        """
        Subscribe to audio tracks automatically
        """
        return publication.kind == rtc.TrackKind.KIND_AUDIO


@dataclass
class AutoDisconnectCallbacks:
    """Helper callbacks for common auto disconnect scenarios"""

    @staticmethod
    def ROOM_EMPTY(ctx: JobContext) -> bool:
        # Hot path, if there are no participants, we don't need to check
        if len(ctx.room.participants) == 0:
            return True

        # Hot path, if there are more than 1 participants, we don't need to check
        if len(ctx.room.participants) > 1:
            return False

        # If only participant is the agent
        for p in ctx.room.participants.values():
            if p.identity == ctx.agent_identity:
                return True

        return False

    @staticmethod
    def PUBLISHER_LEFT(ctx: JobContext) -> bool:
        if ctx.participant is None:
            logging.error(
                "Incorrect usage of PUBLISHER_LEFT, JobContext is tied to a Participant"
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
            return AutoDisconnectCallbacks.PUBLISHER_LEFT(ctx)

        return AutoDisconnectCallbacks.ROOM_EMPTY(ctx)


class JobRequest:
    """
    Represents a new job from the server, this worker can either accept or reject it.
    """

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

    async def reject(self) -> None:
        """
        Tell the server that we cannot handle the job
        """
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

            self._answered = True
            if not self._simulated:
                await self._worker._send_availability(self.id, False)

        logging.info("rejected job %s", self.id)

    async def accept(
        self,
        agent: Callable[[JobContext], Coroutine],
        subscribe_cb: Callable[
            [rtc.TrackPublication, rtc.RemoteParticipant], bool
        ] = SubscribeCallbacks.SUBSCRIBE_NONE,
        auto_disconnect_cb: Callable[
            [JobContext], bool
        ] = AutoDisconnectCallbacks.DEFAULT,
        grants: api.VideoGrants = None,
        name: str = "",
        identity: str = "",
        metadata: str = "",
    ) -> None:
        """
        Signal to the LiveKit Server that we can handle the job, and connect to the room.

        Args:
            agent (Callable[[JobContext], Coroutine]):
                Your agent entrypoint.

            subscribe_cb (Callable[[rtc.TrackPublication, rtc.RemoteParticipant], bool]):
                Callback that is called when a track is published in the room. Return True to subscribe to the track.

            auto_disconnect_cb (Callable[[JobContext], bool]):
                Callback that is called when the agent should disconnect from the room. Return True to disconnect.
                The callback is called when:
                - Initially once the agent has connected to the room
                - A participant leaves the room

            grants (api.VideoGrants, optional):
                Additional grants to give to the agent participant in its token.
                Defaults to None.

            name (str, optional):
                Name of the agent participant. Defaults to "".

            identity (str, optional):
                Identity of the agent participant. Defaults to "".

            metadata (str, optional):
                Metadata of the agent participant. Defaults to "".
        """
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

            self._answered = True

            identity = identity or "agent-" + self.id
            grants = grants or api.VideoGrants()
            grants.room_join = True
            grants.agent = True
            grants.room = self.room.name
            grants.can_update_own_metadata = True

            jwt = (
                api.AccessToken(self._worker._api_key, self._worker._api_secret)
                .with_identity(identity)
                .with_grants(grants)
                .with_metadata(metadata)
                .with_name(name)
                .to_jwt()
            )

            # raise AssignmentTimeoutError if assignment times out
            if not self._simulated:
                _ = await self._worker._send_availability(self.id, True)

            try:
                options = rtc.RoomOptions(
                    auto_subscribe=subscribe_cb == SubscribeCallbacks.SUBSCRIBE_ALL
                )
                await self._room.connect(self._worker._rtc_url, jwt, options)
            except rtc.ConnectError as e:
                logging.error(
                    "failed to connect to the room, cancelling job %s: %s", self.id, e
                )
                await self._worker._send_job_status(
                    self.id, proto_agent.JobStatus.JS_FAILED, str(e)
                )
                raise e

            participant: Optional[rtc.Participant] = None
            if self._info.participant:
                participant = self._room.participants.get(self._info.participant.sid)

            job_ctx = JobContext(
                self.id,
                self._worker,
                self._room,
                participant=participant,
                agent_identity=identity,
            )

            def done_callback(t: asyncio.Task):
                try:
                    if t.cancelled():
                        logging.info(
                            "Task was cancelled. Worker: %s Job: %s",
                            self._worker.id,
                            self.id,
                        )
                    else:
                        logging.info(
                            "Task completed successfully. Worker: %s Job: %s",
                            self._worker.id,
                            self.id,
                        )
                except asyncio.CancelledError:
                    logging.info(
                        "Task was cancelled. Worker: %s Job: %s",
                        self._worker.id,
                        self.id,
                    )
                except Exception as e:
                    logging.error(
                        "Task raised an uncaught exception. Worker: %s Job: %s Exception: %s",
                        self._worker.id,
                        self.id,
                        e,
                    )

            task = self._worker._loop.create_task(agent(job_ctx))
            task.add_done_callback(done_callback)

            def disconnect_if_needed(*_):
                if auto_disconnect_cb(job_ctx):
                    asyncio.ensure_future(
                        job_ctx.disconnect(),
                        loop=self._worker._loop,
                    )

            @self._room.on("track_published")
            def on_track_published(
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                if not subscribe_cb(publication, participant):
                    return

                publication.set_subscribed(True)

            self._room.on("participant_disconnected", disconnect_if_needed)

            for participant in self._room.participants.values():
                for publication in participant.tracks.values():
                    if not subscribe_cb(publication, participant):
                        continue

                    publication.set_subscribed(True)

            # Call disconnect_if_needed() once to check if the conditions
            # for auto disconnect are already met
            disconnect_if_needed()

        logging.info("accepted job %s", self.id)
