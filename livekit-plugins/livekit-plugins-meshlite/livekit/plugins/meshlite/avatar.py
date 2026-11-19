# Copyright 2026 LiveKit, Inc.
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

"""MeshLite avatar for LiveKit Agents.

The MeshLite avatar is rendered by the visitor's own browser, with the MeshLite
web SDK. There is no avatar participant and no video track: this module only
routes the agent's voice to the visitor's page, over the avatar data-stream
protocol that the other avatar plugins use between the agent and their worker:

* agent -> page: one ``lk.audio_stream`` byte stream per speech segment
  (16-bit PCM, ``sample_rate`` and ``num_channels`` stream attributes);
* agent -> page: RPC ``lk.clear_buffer`` on interruption;
* page -> agent: RPC ``lk.playback_started`` and ``lk.playback_finished``.
"""

from __future__ import annotations

import asyncio
import time

from livekit import rtc
from livekit.agents import AgentSession, utils
from livekit.agents.metrics.base import AvatarMetrics, Metadata
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput

from .log import logger

DEFAULT_SAMPLE_RATE = 24000
_MIN_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 48000


class AvatarSession(BaseAvatarSession):
    """Route an :class:`~livekit.agents.AgentSession` voice to a MeshLite avatar.

    Call :meth:`start` after the room is connected and **before**
    ``AgentSession.start()``: the agent then sends its voice to the visitor's
    page instead of publishing an audio track, and the page plays it in sync
    with the avatar's face.

    Because the avatar lives in the visitor's page, :attr:`avatar_identity` is
    the visitor's identity, :meth:`wait_for_join` returns once the visitor is in
    the room, and :meth:`aclose` never removes a participant from the room.

    Args:
        visitor_identity: identity of the browser participant whose page shows
            the avatar. When omitted, the first standard (non-agent, non-SIP)
            remote participant is used, which is the participant
            :class:`~livekit.agents.AgentSession` links to by default.
        sample_rate: PCM sample rate sent to the page. 24 kHz matches the
            MeshLite engine and keeps the stream small.
        clear_buffer_timeout: seconds the agent waits for the page to confirm an
            interruption before it considers the speech stopped.
    """

    def __init__(
        self,
        *,
        visitor_identity: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        clear_buffer_timeout: float | None = 2.0,
    ) -> None:
        super().__init__()
        if visitor_identity is not None and (
            not isinstance(visitor_identity, str) or not visitor_identity.strip()
        ):
            raise ValueError("visitor_identity must be a non-empty string")
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or not _MIN_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE
        ):
            raise ValueError(
                f"sample_rate must be an integer between {_MIN_SAMPLE_RATE} and {_MAX_SAMPLE_RATE}"
            )
        if clear_buffer_timeout is not None and not clear_buffer_timeout >= 0:
            raise ValueError("clear_buffer_timeout must be >= 0 or None")

        self._visitor_identity = visitor_identity
        self._sample_rate = sample_rate
        self._clear_buffer_timeout = clear_buffer_timeout
        self._audio_output: DataStreamAudioOutput | None = None
        self._starting = False
        self._closed = False

    @property
    def provider(self) -> str:
        return "meshlite"

    @property
    def visitor_identity(self) -> str | None:
        """Identity of the participant whose page renders the avatar (known after start)."""
        return self._visitor_identity

    @property
    def avatar_identity(self) -> str:
        """The participant whose page renders the avatar: the visitor."""
        if self._visitor_identity is None:
            raise RuntimeError("the visitor identity is known once start() has returned")
        return self._visitor_identity

    @property
    def audio_output(self) -> DataStreamAudioOutput | None:
        return self._audio_output

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        if self._closed:
            raise RuntimeError("this MeshLite AvatarSession is closed")
        if self._starting or self._audio_output is not None:
            raise RuntimeError("MeshLite AvatarSession.start() can only be called once")
        self._starting = True
        try:
            identity = self._visitor_identity
            if identity is None:
                if not room.isconnected():
                    raise RuntimeError(
                        "connect to the room before starting the MeshLite avatar, "
                        "or pass visitor_identity"
                    )
                logger.debug("waiting for the visitor participant")
                participant = await utils.wait_for_participant(
                    room, kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
                )
                identity = participant.identity
            if self._closed:
                raise RuntimeError("this MeshLite AvatarSession was closed while starting")
            self._visitor_identity = identity

            await super().start(agent_session, room)

            if agent_session.output.audio is not None:
                logger.warning(
                    "replacing the session audio output with the MeshLite avatar output; "
                    "call AvatarSession.start() before AgentSession.start() so the agent "
                    "does not publish its own audio track"
                )
            output = DataStreamAudioOutput(
                room,
                destination_identity=identity,
                sample_rate=self._sample_rate,
                clear_buffer_timeout=self._clear_buffer_timeout,
                # the page reports when the avatar really starts speaking
                wait_playback_start=True,
            )
            agent_session.output.audio = output
            self._audio_output = output
            logger.debug(
                "MeshLite avatar voice routed to the visitor page",
                extra={"visitor_identity": identity, "sample_rate": self._sample_rate},
            )
        finally:
            self._starting = False

    async def _wait_avatar_join(self) -> None:
        # The avatar is drawn in the visitor's page: it has joined once the
        # visitor is in the room. The visitor publishes no avatar video track.
        assert self._room is not None
        started_time = time.time()
        await utils.wait_for_participant(room=self._room, identity=self.avatar_identity)
        joined_time = time.time()
        self._emit_metrics(
            AvatarMetrics(
                timestamp=joined_time,
                session_started_time=started_time,
                avatar_joined_time=joined_time,
                metadata=Metadata(model_provider=self.provider),
            )
        )

    async def aclose(self) -> None:
        """Stop routing the voice to the avatar.

        Unlike the base class this never removes a participant from the room:
        the only participant involved is the visitor.
        """
        if self._closed:
            return
        self._closed = True

        session, output = self._agent_session, self._audio_output
        if session is not None and output is not None and session.output.audio is output:
            session.output.audio = None
        if output is not None:
            # The output starts a task that waits for the visitor, one loop turn
            # after it is created; stop it if the visitor never joined.
            await asyncio.sleep(0)
            start_task = getattr(output, "_start_atask", None)
            if start_task is not None and not start_task.done():
                await utils.aio.cancel_and_wait(start_task)

        if self._agent_session is not None:
            self._agent_session.off("conversation_item_added", self._on_conversation_item_added)
            self._agent_session = None
        if self._room is not None:
            self._room.off("connection_state_changed", self._on_connection_state_changed)
            self._room = None
        if self._wait_avatar_join_task is not None:
            await utils.aio.cancel_and_wait(self._wait_avatar_join_task)
            self._wait_avatar_join_task = None
