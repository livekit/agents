"""A call's room I/O against LiveKit Cloud: a worker registered under a fresh agent name, rooms and
dispatches made through the server API, and people played by ``livekit.rtc`` clients that publish
a microphone from a WAV file, type on the chat topic and watch what the agent publishes. The DST
runs the same room I/O over a fake room; the real SFU, real WebRTC tracks, data streams and
participant attributes are what it cannot reach. Hang-ups and rejoins are
``scripts/e2e.py``'s, which compares them with the reference.

The agent is this file run as a worker (``python tests/test_it_room_cloud.py start``), with
scripted providers: an STT that hears any loud stretch of audio as "heard speech", an LLM that
answers "You said: <text>", and a TTS that speaks a tone. A dispatch's metadata picks the
participant to link and how the call ends.

Needs LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
import uuid
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from livekit import api, rtc

pytestmark = pytest.mark.plugin("livekit")

_core = pytest.importorskip("livekit.agents._core", reason="needs livekit-agents-next")

from livekit.agents import (  # noqa: E402
    DEFAULT_API_CONNECT_OPTIONS,
    Agent,
    AgentServer,
    JobContext,
    VoiceSession,
    cli,
    llm,
    room_io,
    stt,
    tts,
)
from livekit.agents.utils import shortuuid  # noqa: E402

logger = logging.getLogger("it.agent")

CREDENTIALS = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
WAV = Path(__file__).parent / "change-sophie.wav"
SAMPLE_RATE = 24000
LOUD_RMS = 500.0

if not all(os.environ.get(k) for k in CREDENTIALS):
    pytest.skip(f"needs {', '.join(CREDENTIALS)}", allow_module_level=True)


# ============================================================================ the person


@dataclass
class Person:
    """A participant as a person's client sees the room."""

    room: rtc.Room
    identity: str
    transcripts: list[dict[str, Any]] = field(default_factory=list)
    agent_states: list[str] = field(default_factory=list)
    agent_audio_peaks: list[float] = field(default_factory=list)
    agent_track: rtc.RemoteTrackPublication | None = None
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    @property
    def agent(self) -> rtc.RemoteParticipant | None:
        return next(
            (
                p
                for p in self.room.remote_participants.values()
                if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
            ),
            None,
        )

    def said(self, by: str, text: str) -> list[dict[str, Any]]:
        return [t for t in self.transcripts if t["identity"] == by and text in t["text"]]

    def heard_agent_audio(self) -> bool:
        return any(p > LOUD_RMS for p in self.agent_audio_peaks)


async def join(room_name: str, identity: str) -> Person:
    token = (
        api.AccessToken()
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    person = Person(room=room, identity=identity)

    def on_text(reader: rtc.TextStreamReader, sender: str) -> None:
        async def read() -> None:
            text = await reader.read_all()
            person.transcripts.append(
                {"identity": sender, "text": text, "attributes": dict(reader.info.attributes)}
            )

        _spawn(person, read())

    def on_attributes(changed: dict[str, str], participant: rtc.Participant) -> None:
        state = changed.get("lk.agent.state")
        if state and (not person.agent_states or person.agent_states[-1] != state):
            person.agent_states.append(state)

    def on_track(
        track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            person.agent_track = pub
            _spawn(person, _listen(person, track))

    room.register_text_stream_handler("lk.transcription", on_text)
    room.on("participant_attributes_changed", on_attributes)
    room.on("track_subscribed", on_track)
    await room.connect(os.environ["LIVEKIT_URL"], token)
    return person


def _spawn(person: Person, coro: Any) -> None:
    task = asyncio.ensure_future(coro)
    person.tasks.add(task)
    task.add_done_callback(person.tasks.discard)


async def _listen(person: Person, track: rtc.Track) -> None:
    async for ev in rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1):
        samples = np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32)
        person.agent_audio_peaks.append(float(np.sqrt(np.mean(samples**2))) if samples.size else 0)


def _wav_frames(seconds_of_silence: float = 1.5) -> list[rtc.AudioFrame]:
    """The WAV in 10 ms frames, then silence so a turn can end."""
    with wave.open(str(WAV)) as w:
        rate, data = w.getframerate(), w.readframes(w.getnframes())
    data += b"\x00\x00" * int(rate * seconds_of_silence)
    step = rate // 100
    pcm = np.frombuffer(data, dtype=np.int16)
    return [
        rtc.AudioFrame(pcm[i : i + step].tobytes(), rate, 1, step)
        for i in range(0, len(pcm) - step + 1, step)
    ]


async def publish_mic(person: Person, *, preconnect: bool = False) -> rtc.AudioSource:
    source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE, preconnect_buffer=preconnect
    )
    await person.room.local_participant.publish_track(track, options)
    return source


async def speak(source: rtc.AudioSource) -> None:
    for frame in _wav_frames():
        await source.capture_frame(frame)


async def wait_for(predicate: Any, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            tail = _Worker.last.tail() if _Worker.last else ""
            raise AssertionError(f"timed out after {timeout}s waiting for {what}\n{tail}")
        await asyncio.sleep(0.05)


def listening(person: Person) -> bool:
    agent = person.agent
    return agent is not None and agent.attributes.get("lk.agent.state") == "listening"


async def leave(*people: Person) -> None:
    for p in people:
        for t in list(p.tasks):
            t.cancel()
        await p.room.disconnect()


# ============================================================================ the fixtures


@pytest.fixture(scope="module")
def agent_worker(tmp_path_factory: pytest.TempPathFactory):
    """One worker for the module, registered under a fresh agent name."""
    worker = _Worker(tmp_path_factory.mktemp("agent"))
    yield worker
    worker.stop()


class _Worker:
    last: _Worker | None = None

    def __init__(self, logs: Path) -> None:
        _Worker.last = self
        self.name = f"it-room-{uuid.uuid4().hex[:8]}"
        self.log_path = logs / "worker.log"
        self.log = self.log_path.open("w")
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "start", "--log-level", "DEBUG"],
            env={**os.environ, "LIVEKIT_AGENT_NAME": self.name, "PYTHONUNBUFFERED": "1"},
            stdout=self.log,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parents[1]),
        )
        deadline = time.monotonic() + 60
        while "registered worker" not in self.log_path.read_text():
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.stop()
                pytest.fail(f"the worker did not register:\n{self.tail()}")
            time.sleep(0.1)

    def tail(self, n: int = 60) -> str:
        noise = ("preloaded", "Using selector", "initializ")
        lines = [
            x for x in self.log_path.read_text().splitlines() if not any(w in x for w in noise)
        ]
        return "\n".join(lines[-n:])

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


@pytest.fixture
async def lkapi() -> AsyncIterator[api.LiveKitAPI]:
    async with api.LiveKitAPI() as client:
        yield client


@pytest.fixture
async def room_name(lkapi: api.LiveKitAPI) -> AsyncIterator[str]:
    name = f"it-room-{uuid.uuid4().hex[:10]}"
    await lkapi.room.create_room(api.CreateRoomRequest(name=name, empty_timeout=60))
    yield name
    try:
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=name))
    except api.TwirpError:
        pass


async def dispatch(lkapi: api.LiveKitAPI, worker: _Worker, room: str, **options: Any) -> None:
    await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name=worker.name, room=room, metadata=json.dumps(options)
        )
    )


async def agent_joined(person: Person, worker: _Worker, timeout: float = 30) -> None:
    try:
        await wait_for(
            lambda: person.agent is not None and person.agent_track is not None,
            timeout,
            "the agent to join and publish its audio",
        )
    except AssertionError:
        raise AssertionError(f"no agent in the room:\n{worker.tail()}") from None


# ============================================================================ the tests


async def test_spoken_turn_round_trip(agent_worker, lkapi, room_name) -> None:
    """Guards the call's whole audio path on a real SFU: the agent joining and publishing its
    microphone, subscribing the person's, transcribing them, replying with audible speech, the
    transcription streams keyed on the right tracks, and lk.agent.state following the turn."""
    person = await join(room_name, "user")
    mic = await publish_mic(person)
    await dispatch(lkapi, agent_worker, room_name)
    await agent_joined(person, agent_worker)
    await wait_for(lambda: listening(person), 15, "the agent to listen")

    await speak(mic)
    agent = person.agent
    assert agent is not None
    await wait_for(lambda: person.said(agent.identity, "You said: heard speech"), 20, "a reply")
    await wait_for(person.heard_agent_audio, 10, "the agent's voice")

    # the person's words come back as their segment's streams, the last one final
    user_lines = person.said("user", "heard speech")
    mic_sid = next(iter(person.room.local_participant.track_publications))
    assert all(x["attributes"]["lk.transcribed_track_id"] == mic_sid for x in user_lines)
    assert user_lines[-1]["attributes"]["lk.transcription_final"] == "true", user_lines
    reply = person.said(agent.identity, "You said: heard speech")[0]
    assert person.agent_track is not None
    assert reply["attributes"]["lk.transcribed_track_id"] == person.agent_track.sid, reply
    assert person.agent_track.source == rtc.TrackSource.SOURCE_MICROPHONE

    await wait_for(lambda: listening(person), 15, "the agent to listen again")
    states = person.agent_states
    assert states.index("thinking") < states.index("speaking") < len(states) - 1, states
    await leave(person)


async def test_text_input_on_the_chat_topic(agent_worker, lkapi, room_name) -> None:
    """Guards a message typed on lk.chat not reaching the agent as a user turn (or its spoken reply
    not reaching the room) on a call with no microphone at all."""
    person = await join(room_name, "user")
    await dispatch(lkapi, agent_worker, room_name)
    await agent_joined(person, agent_worker)
    agent = person.agent
    assert agent is not None
    await wait_for(lambda: listening(person), 15, "the agent to listen")

    await person.room.local_participant.send_text("the blue door", topic="lk.chat")
    await wait_for(lambda: person.said(agent.identity, "You said: the blue door"), 20, "a reply")
    await wait_for(person.heard_agent_audio, 10, "the agent's voice")
    await leave(person)


async def test_agent_links_the_participant_it_was_told(agent_worker, lkapi, room_name) -> None:
    """Guards RoomOptions.participant_identity: with two people in the room the agent must listen
    to the one it was given and ignore the other, whoever joined first."""
    other = await join(room_name, "bystander")
    caller = await join(room_name, "caller")
    other_mic, caller_mic = await publish_mic(other), await publish_mic(caller)
    await dispatch(lkapi, agent_worker, room_name, participant="caller")
    await agent_joined(caller, agent_worker)
    await wait_for(lambda: listening(caller), 15, "the agent to listen")
    agent = caller.agent
    assert agent is not None

    await speak(other_mic)
    await asyncio.sleep(3)
    assert not caller.said(agent.identity, "You said"), "the agent answered the bystander"

    await speak(caller_mic)
    await wait_for(lambda: caller.said(agent.identity, "You said: heard speech"), 20, "a reply")
    await leave(other, caller)


async def test_pre_connect_audio_is_heard(agent_worker, lkapi, room_name) -> None:
    """Guards the pre-connect buffer: what the person said before the agent joined (sent on
    lk.agent.pre-connect-audio-buffer for a track published with preconnect_buffer) must be
    transcribed and answered, not dropped."""
    person = await join(room_name, "user")
    mic = await publish_mic(person, preconnect=True)
    mic_sid = next(iter(person.room.local_participant.track_publications))
    await dispatch(lkapi, agent_worker, room_name)
    # a client sends its buffer once the agent is active: data sent to a participant that has
    # only joined can be lost
    await wait_for(
        lambda: (
            person.agent is not None
            and person.agent.state == rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE
        ),
        30,
        "the agent to become active",
    )
    agent = person.agent
    assert agent is not None

    # what a client buffered while the agent was on its way, then live silence
    writer = await person.room.local_participant.stream_bytes(
        "preconnect",
        topic="lk.agent.pre-connect-audio-buffer",
        mime_type="audio/pcm",
        attributes={"trackId": mic_sid, "sampleRate": "48000", "channels": "1"},
        destination_identities=[agent.identity],
    )
    await writer.write(b"".join(bytes(f.data) for f in _wav_frames(seconds_of_silence=0)))
    await writer.aclose()
    silence = rtc.AudioFrame(b"\x00\x00" * 480, 48000, 1, 480)
    for _ in range(300):
        await mic.capture_frame(silence)
        if person.said(agent.identity, "You said: heard speech"):
            break
    await wait_for(lambda: person.said(agent.identity, "You said: heard speech"), 15, "a reply")
    await leave(person)


def test_sigterm_drains_the_running_call(tmp_path: Path) -> None:
    """Guards a rolling deploy cutting calls: on SIGTERM the worker must stop taking jobs, let the
    running call end on its own (its shutdown callbacks run), and only then exit 0."""
    worker = _Worker(tmp_path)
    try:
        asyncio.run(_drain_scenario(worker, tmp_path))
    finally:
        worker.stop()


async def _drain_scenario(worker: _Worker, tmp_path: Path) -> None:
    async with api.LiveKitAPI() as lkapi:
        name = f"it-drain-{uuid.uuid4().hex[:10]}"
        await lkapi.room.create_room(api.CreateRoomRequest(name=name, empty_timeout=60))
        person = await join(name, "user")
        try:
            marker = tmp_path / "ended"
            await dispatch(lkapi, worker, name, end_after=6, marker=str(marker))
            await agent_joined(person, worker)
            worker.proc.send_signal(signal.SIGTERM)
            await asyncio.sleep(2)
            assert worker.proc.poll() is None, "the worker exited with a call running"
            assert person.agent is not None, "the call was cut on SIGTERM"
            exitcode = await asyncio.get_running_loop().run_in_executor(None, worker.proc.wait, 30)
            assert exitcode == 0, worker.tail()
            assert marker.read_text() == "ended", "the call's shutdown callback did not run"
        finally:
            await leave(person)
            try:
                await lkapi.room.delete_room(api.DeleteRoomRequest(room=name))
            except api.TwirpError:
                pass


# ============================================================================ the agent


class HearsLoudAudio(stt.STT):
    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(self, **_: object) -> stt.RecognizeStream:
        return _Hearing(stt=self, conn_options=DEFAULT_API_CONNECT_OPTIONS)


class _Hearing(stt.RecognizeStream):
    async def _run(self) -> None:
        loud = quiet = 0.0
        started = False
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame):
                continue
            pcm = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
            if pcm.size and math.sqrt(float(np.mean(pcm**2))) > LOUD_RMS:
                loud, quiet = loud + frame.duration, 0.0
                if not started and loud >= 0.2:
                    started = True
                    self.send(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            elif loud:
                quiet += frame.duration
                if quiet >= 1.0:
                    if started:
                        said = stt.SpeechData(language="en", text="heard speech", confidence=1)
                        self.send(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[said]
                            )
                        )
                        self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
                    loud = quiet = 0.0
                    started = False


class Echo(llm.LLM):
    def chat(
        self, *, chat_ctx: llm.ChatContext, tools: list[llm.Tool] | None = None, **_: object
    ) -> llm.LLMStream:
        return _Echoing(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )


class _Echoing(llm.LLMStream):
    async def _run(self) -> None:
        last = self._chat_ctx.items[-1]
        text = (last.text_content or "") if last.type == "message" else ""
        delta = llm.ChoiceDelta(role="assistant", content=f"You said: {text}")
        self.send(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))


class Tone(tts.TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

    def synthesize(self, text: str, **_: Any) -> tts.ChunkedStream:
        return _Toning(tts=self, input_text=text, conn_options=DEFAULT_API_CONNECT_OPTIONS)


class _Toning(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid("req_"),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        seconds = min(2.0, 0.05 * len(self.input_text))
        t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
        output_emitter.push((np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).tobytes())
        output_emitter.flush()


# a busy test machine must not mark the worker full
server = AgentServer(load_fnc=lambda: 0.0, num_idle_processes=2, port=0)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    options = json.loads(ctx.job.metadata or "{}")
    if marker := options.get("marker"):

        async def ended(reason: str) -> None:
            Path(marker).write_text("ended")

        ctx.add_shutdown_callback(ended)

    room_options = room_io.RoomOptions()
    if participant := options.get("participant"):
        room_options.participant_identity = participant
    session = VoiceSession(stt=HearsLoudAudio(), llm=Echo(), tts=Tone(), vad=None)
    # what the session did, for a failing test's worker log
    for event in ("conversation_item_added", "agent_state_changed", "error", "close"):
        session.on(event, lambda ev, event=event: logger.info(f"it session {event}: {ev!r}"))
    await session.start(
        Agent(instructions="Echo the person."), room=ctx.room, room_options=room_options
    )
    await ctx.connect()
    if end_after := options.get("end_after"):
        await asyncio.sleep(end_after)
        ctx.shutdown("it: the call ended")


if __name__ == "__main__":
    cli.run_app(server)
