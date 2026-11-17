"""A real room caller. The providers are deterministic; audio still crosses the SFU."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from livekit import api, rtc

from ..resources import finish_cleanup
from ..sfu import KEY, SECRET
from .app import RunningApp
from .media import PHRASES, Heard, hearing, next_transcription, read_transcriptions, speak


@dataclass(frozen=True)
class ReceivedSpeech:
    text: str
    audio_seconds: float
    ended_at: float


class VoiceClient:
    def __init__(
        self,
        room: rtc.Room,
        source: rtc.AudioSource,
        agent: rtc.RemoteParticipant,
        heard: Heard,
        phrases: Mapping[int, str],
    ) -> None:
        self.room, self.source, self.agent = room, source, agent
        self.heard, self.phrases = heard, phrases
        self._transcriptions = read_transcriptions(room)
        self._audio_position = 0

    async def say(self, text: str) -> None:
        await speak(self.source, text, phrases=self.phrases)

    async def speech_started(self) -> float:
        """The next unconsumed utterance's first actual audio frame at the receiver."""
        async with asyncio.timeout(15):
            while len(self.heard.tone_at) == self._audio_position:
                await asyncio.sleep(0.01)
        return self.heard.tone_at[self._audio_position]

    async def hear(self) -> ReceivedSpeech:
        _, reader = await next_transcription(self._transcriptions, self.agent.identity)
        text = await reader.read_all()
        # A transcript alone cannot prove the receiver got audio.
        async with asyncio.timeout(15):
            while len(self.heard.tone_at) == self._audio_position:
                await asyncio.sleep(0.01)
        position = len(self.heard.tone_at)
        seconds = (position - self._audio_position) / 100
        self._audio_position = position
        return ReceivedSpeech(text, seconds, time.monotonic())


@asynccontextmanager
async def connected_voice(
    app: RunningApp, *, phrases: Mapping[int, str] = PHRASES
) -> AsyncIterator[VoiceClient]:
    if not app.voice:
        raise ValueError("start the application with voice=True")
    name = "e2e-" + uuid.uuid4().hex
    token = (
        api.AccessToken(KEY, SECRET)
        .with_identity("caller")
        .with_grants(api.VideoGrants(room_join=True, room=name))
        .with_room_config(
            api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=app.agent_name)])
        )
        .to_jwt()
    )
    room = rtc.Room()
    stack = AsyncExitStack()
    try:
        await room.connect(app.env["LIVEKIT_URL"], token)
        # Own the source before the next await, including failed publication or registration.
        source = rtc.AudioSource(48000, 1)
        stack.push_async_callback(source.aclose)
        track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        async with asyncio.timeout(20):
            while not (
                agents := [
                    p
                    for p in room.remote_participants.values()
                    if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
                    and "lk.publish_on_behalf" not in p.attributes
                    and p.state == rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE
                ]
            ):
                await asyncio.sleep(0.01)
        agent = agents[0]
        heard = await stack.enter_async_context(hearing(agent))
        caller = VoiceClient(room, source, agent, heard, phrases)
        async with asyncio.timeout(20):
            while agent.attributes.get("lk.agent.state") != "listening":
                await asyncio.sleep(0.01)
        yield caller
    finally:

        async def close() -> None:
            try:
                await stack.aclose()
            finally:
                await room.disconnect()

        await finish_cleanup(close())
