"""Simulated calls on an isolated local SFU and a real job process: the simulator is heard as a
caller is."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Callable
from pathlib import Path

import pytest

from livekit import rtc
from livekit.agents import Agent, ModelSettings, RunContext, VoiceSession, _core
from livekit.agents.job import JobContext
from livekit.agents.voice.events import AgentStateChangedEvent
from livekit.agents.voice.room_io import RoomOptions
from livekit.protocol import agent, models
from tests.fakes import FakeTTS, FakeVAD, until

from .processes import job_runner, running_job
from .sfu import connected, room_token, tone, write_marker


async def _greeting_job(ctx: JobContext) -> None:
    marker = Path(ctx.job.metadata)
    await ctx.connect()
    if ctx.proc.user_arguments == "warmup_off":
        session = VoiceSession(
            vad=FakeVAD(),
            tts=FakeTTS(),
            turn_handling={"turn_detection": None},
            aec_warmup_duration=None,
        )
    else:
        session = VoiceSession(vad=FakeVAD(), tts=FakeTTS(), turn_handling={"turn_detection": None})

    @session.tts_node
    async def voice(
        ctx: RunContext, text: AsyncIterable[str], settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        async for _ in text:
            pass
        async for frame in tone(1000, 400):
            yield frame

    await session.start(
        Agent(instructions="Greet."),
        room=ctx.room,
        room_options=RoomOptions(text_input=False, transcription_output=False),
    )
    started: asyncio.Future[float] = asyncio.get_running_loop().create_future()
    silenced: asyncio.Future[float] = asyncio.get_running_loop().create_future()

    @session.on("agent_state_changed")
    def speaking_edges(event: AgentStateChangedEvent) -> None:
        if event.new_state == "speaking" and not started.done():
            started.set_result(time.monotonic())
        if event.old_state == "speaking" and not silenced.done():
            silenced.set_result(time.monotonic())

    speech = session.say("a long greeting")
    stopped = await silenced
    await speech.wait_for_playout()
    write_marker(marker, [speech.interrupted, stopped - started.result()])
    await session.aclose()
    ctx.shutdown("greeted")


@pytest.mark.parametrize(
    "warmup",
    [
        "warmup_off",
        pytest.param(
            "simulation_default",
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/pull/7674 reproduces: a simulated "
                "call keeps the 3 s AEC warmup, so the simulator cannot interrupt",
            ),
        ),
    ],
)
async def test_a_simulator_interrupts_the_greeting_without_an_echo_warmup(
    sfu: str,
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    tmp_path: Path,
    warmup: str,
) -> None:
    """In a simulated job the simulator's speech interrupts the agent from its first words: a
    simulator's synthesized audio never echoes the agent, so no AEC warmup holds it off.
    https://github.com/livekit/agents/pull/7674

    The simulator speaks once the agent says it is speaking: speech already under way when the
    greeting is scheduled interrupts it before it ever plays."""
    name = "regression-" + uuid.uuid4().hex
    async with connected(sfu, name, "simulator") as simulator:
        source = rtc.AudioSource(48000, 1)
        await simulator.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("microphone", source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        marker = tmp_path / "greeting"
        process = own_executor(
            job_runner(_greeting_job, mp_ctx=mp.get_context("spawn"), user_arguments=warmup)
        )
        await process.start()
        await process.initialize()
        dispatch = json.dumps({"simulationRunId": "SR_regression", "jobId": "SJ_regression"})
        info = running_job(
            agent.Job(
                id="AJ_" + uuid.uuid4().hex,
                room=models.Room(name=name),
                metadata=str(marker),
                attributes={"lk.simulator.dispatch": dispatch},
            ),
            fake_job=False,
        )
        info.url, info.token = sfu, room_token(name, "agent")
        await process.launch_job(info)

        def greeting() -> bool:
            job = simulator.remote_participants.get("agent")
            return job is not None and job.attributes.get("lk.agent.state") == "speaking"

        await until(greeting, "the agent never started its greeting", timeout=30)
        # six seconds of speech over the greeting, then quiet until the job reports
        quiet = rtc.AudioFrame.create(48000, 1, 960)
        async for frame in tone(400, 300):
            await source.capture_frame(frame)
        async with asyncio.timeout(30):
            while not marker.exists():
                await source.capture_frame(quiet)
        interrupted, after = json.loads(marker.read_text())

    assert interrupted, "the simulator's speech never interrupted the 8 s greeting"
    assert after < 2.5, f"the agent kept speaking over the simulator for {after:.1f} s"
