"""Past voice-session bugs reproduced through the public VoiceSession API and Python plugins."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

from livekit.agents import Agent, VoiceSession
from livekit.agents.utils import media
from livekit.agents.voice.events import UserInputTranscribedEvent
from tests.fakes import FakeAudioInput, FakeSTT, FakeVAD, until

from .conftest import ConsolePeer

RECORDING_AGENT_MODULE = """
import asyncio

from livekit.agents import Agent, AgentServer, VoiceSession, cli
from tests.fakes import FakeTTS

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx):
    session = VoiceSession(tts=FakeTTS(streaming=False))
    await session.start(Agent(instructions="Greet."))
    await session.say("hello there")
    await asyncio.sleep(0.5)  # the call lasts half a second
    await session.aclose()
    ctx.shutdown()

if __name__ == "__main__":
    cli.run_app(server)
"""


async def test_a_session_without_an_llm_still_transcribes_the_user() -> None:
    """A session with only VAD and STT transcribes what it hears and records the user turn; an
    STT transcript names no item, as only a realtime server's does.
    https://github.com/livekit/agents/issues/2436"""
    heard = "my name is Theo"
    session = VoiceSession(
        vad=FakeVAD(),
        stt=FakeSTT([heard]),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    )
    microphone = FakeAudioInput()
    session.input.audio = microphone
    transcripts: list[UserInputTranscribedEvent] = []
    session.on("user_input_transcribed", transcripts.append)

    def users() -> list[str | None]:
        return [i.text_content for i in session.history.items if i.type == "message"]

    async with session:
        await session.start(Agent(instructions="Listen.", name="listener"))
        microphone.speak(1.0)
        await until(lambda: heard in users(), f"no user turn recorded: {users()}")

    assert [(t.transcript, t.is_final, t.item_id) for t in transcripts] == [(heard, True, None)]


async def test_a_recorded_console_call_writes_its_audio_without_a_room(
    console: ConsolePeer,
    agent_module: Callable[[str], Path],
    agent_env: dict[str, str],
    tmp_path: Path,
) -> None:
    """``console --record`` records the call's audio although the console has no room, and keeps
    it with the report under ``.livekit/sessions/``. https://github.com/livekit/agents/pull/4575"""
    agent = agent_module(RECORDING_AGENT_MODULE)

    console_call = await asyncio.create_subprocess_exec(
        sys.executable,
        agent,
        "console",
        "--record",
        "--connect-addr",
        console.addr,
        env=agent_env,
        cwd=tmp_path,
    )
    assert await console_call.wait() == 0

    [kept] = list((tmp_path / ".livekit" / "sessions").iterdir())
    assert (kept / "session_report.json").is_file()
    recording = kept / "audio.ogg"
    decoder = media.AudioStreamDecoder(format="audio/ogg")
    decoder.push(recording.read_bytes())
    decoder.end_input()
    duration = sum([frame.duration async for frame in decoder])
    await decoder.aclose()
    assert duration >= 0.4, f"the half-second call recorded {duration:.2f} s of audio"
