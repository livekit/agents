"""Explicit bindings use the same real room path as convenience session startup."""

import asyncio
import uuid

import pytest

from livekit import rtc
from livekit.agents import Agent, RoomIO, VoiceSession
from livekit.agents.voice.room_io import RoomOptions
from tests.fakes import FakeLLM

from .sfu import connected


async def test_explicit_binding_delivers_room_text_and_releases_its_key(sfu: str) -> None:
    name = "binding-" + uuid.uuid4().hex
    async with connected(sfu, name, "person") as person, connected(sfu, name, "agent") as agent:
        delivered: asyncio.Queue[str] = asyncio.Queue()
        readers: list[asyncio.Task[None]] = []

        def transcription(reader: rtc.TextStreamReader, _: str) -> None:
            async def read() -> None:
                delivered.put_nowait(await reader.read_all())

            readers.append(asyncio.create_task(read()))

        person.register_text_stream_handler("lk.transcription", transcription)
        io = RoomIO(agent)
        binding = io.bind("voice")
        try:
            async with VoiceSession(
                llm=FakeLLM({"hello": "binding reply"}),
                vad=None,
                turn_handling={"turn_detection": None},
            ) as session:
                await session.start(
                    Agent(instructions="Reply."),
                    io=binding,
                    room_options=RoomOptions(audio_input=False, audio_output=False),
                )
                async with VoiceSession(
                    llm=FakeLLM(), vad=None, turn_handling={"turn_detection": None}
                ) as rejected:
                    with pytest.raises(RuntimeError, match="already been activated"):
                        await rejected.start(Agent(instructions="Another consumer."), io=binding)
                await person.local_participant.send_text("hello", topic="lk.chat")
                async with asyncio.timeout(10):
                    while await delivered.get() != "binding reply":
                        pass
            replacement = io.bind("voice")
            await replacement.aclose()
            await io.aclose()
            assert agent.isconnected()
        finally:
            await binding.aclose()
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
