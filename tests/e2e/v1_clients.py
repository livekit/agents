"""The 2.0 agent the v1 client compatibility journeys talk to, and the client moves they repeat.

The agent is a real ``AgentServer`` dispatching a real job that runs a ``VoiceSession``; only its
models are scripted, by sound: the user "says" a phrase by playing that phrase's tone, and the
agent "speaks" by playing a 1 kHz tone, so what a client hears is real audio. The client helpers
are plain ``livekit.rtc`` calls doing what the cited v1 client does on the wire.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager
from typing import Literal, NamedTuple, TypedDict, get_args

import numpy as np
import pytest

from livekit import api, rtc
from livekit.agents import Agent, AgentServer, LanguageCode, VoiceSession, function_tool, tts
from livekit.agents.job import JobContext, get_job_context
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.tts import TTS, ChunkedStream, SynthesizeStream, TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    AvatarSession,
    DataStreamAudioOutput,
    DataStreamAudioReceiver,
)
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.room_io import RoomOptions
from tests.fakes import FakeLLM, FakeVideoGenerator, Reply, last_input, tool_call

from .sfu import KEY, SECRET

AGENT_NAME = "compat-agent"
PHRASES = {500: "hello there agent", 700: "what time is it", 900: "early words before connect"}
AGENT_TONE = 1000
AVATAR_IDENTITY = "avatar-worker"
V1_AGENT_IDENTITY = "v1-agent"


def loudness_and_pitch(pcm: bytes, sample_rate: int) -> tuple[float, int]:
    """RMS and the dominant frequency (to 100 Hz) of PCM16 mono."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if len(samples) < 32:
        return 0.0, 0
    peak = np.fft.rfftfreq(len(samples), 1 / sample_rate)[np.argmax(abs(np.fft.rfft(samples)))]
    return float(np.sqrt(np.mean(samples**2))), round(float(peak) / 100) * 100


def tone(frequency: int, seconds: float, sample_rate: int) -> bytes:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (6000 * np.sin(2 * np.pi * frequency * t)).astype(np.int16).tobytes()


class Script(TypedDict, total=False):
    """The dispatch metadata scripting the agent: its replies by what it hears, and options."""

    replies: dict[str, str]
    tool_calls: dict[str, str]  # what it hears -> the tool it calls
    marker: str  # a file the job writes its metadata to
    shutdown_on: str
    app_rpc: bool
    avatar: bool
    delete_room: bool
    second_session: bool


# --- the agent --------------------------------------------------------------------------------


class ToneSTT(STT):
    """Hears a phrase's tone: interims grow a word per 150 ms, the final follows 300 ms of quiet."""

    def __init__(self) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))

    async def _recognize_impl(self, buffer: object, **_: object) -> SpeechEvent:
        raise NotImplementedError

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS, **_: object
    ) -> RecognizeStream:
        return _ToneRecognition(stt=self, conn_options=conn_options)


class _ToneRecognition(RecognizeStream):
    async def _run(self) -> None:
        phrase, heard, quiet, shown, pending = None, 0.0, 0.0, 0, b""
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame):
                continue
            rate, step = frame.sample_rate, frame.sample_rate // 50 * 2  # 20 ms, first channel
            pending += np.frombuffer(frame.data, np.int16)[:: frame.num_channels].tobytes()
            while len(pending) >= step:
                window, pending = pending[:step], pending[step:]
                rms, pitch = loudness_and_pitch(window, rate)
                if rms > 300 and pitch in PHRASES:
                    if phrase is None:
                        phrase, heard, shown = PHRASES[pitch], 0.0, 0
                        self._send(SpeechEventType.START_OF_SPEECH)
                    heard, quiet = heard + 0.02, 0.0
                    words = phrase.split()[: 1 + int(heard / 0.15)]
                    if len(words) > shown:
                        shown = len(words)
                        self._send(SpeechEventType.INTERIM_TRANSCRIPT, " ".join(words))
                elif phrase is not None:
                    quiet += 0.02
                    if quiet >= 0.3:
                        self._send(SpeechEventType.FINAL_TRANSCRIPT, phrase)
                        self._send(SpeechEventType.END_OF_SPEECH)
                        phrase = None

    def _send(self, kind: SpeechEventType, text: str = "") -> None:
        said = [SpeechData(text=text, language=LanguageCode("en"))] if text else []
        self.send(SpeechEvent(type=kind, alternatives=said))


class ToneTTS(TTS):
    """Speaks a 1 kHz tone, 60 ms per character."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True), sample_rate=24000, num_channels=1
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return _ToneChunks(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return _ToneStream(tts=self, conn_options=conn_options)


def _speak(emitter: tts.AudioEmitter, text: str) -> None:
    audio = tone(AGENT_TONE, min(6.0, 0.06 * max(len(text), 5)), 24000)
    for offset in range(0, len(audio), 960):
        emitter.push(audio[offset : offset + 960])


class _ToneChunks(ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="tone", sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        _speak(output_emitter, self._input_text)
        output_emitter.flush()


class _ToneStream(SynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="tone", sample_rate=24000, num_channels=1, mime_type="audio/pcm", stream=True
        )
        text = ""
        async for data in self.input:
            if isinstance(data, str):
                text += data
            elif text.strip():
                output_emitter.start_segment(segment_id=f"tone-{id(text)}")
                _speak(output_emitter, text)
                output_emitter.flush()
                text = ""


class _Avatar(AvatarSession):
    @property
    def avatar_identity(self) -> str:
        return AVATAR_IDENTITY

    async def start(self, agent_session: VoiceSession, room: rtc.Room) -> None:
        await super().start(agent_session, room)
        output = DataStreamAudioOutput(room=room, destination_identity=AVATAR_IDENTITY)
        self._attach_audio_output(agent_session, output)


@function_tool
async def read_device_temperature() -> str:
    """Read the CPU temperature of the caller's device."""
    room = get_job_context().room
    caller = next(
        p
        for p in room.remote_participants.values()
        if p.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    )
    return await room.local_participant.perform_rpc(
        destination_identity=caller.identity, method="get_cpu_temp", payload=""
    )


async def compat_agent(ctx: JobContext) -> None:
    """The agent every journey dispatches, started the documented default way; the dispatch
    metadata scripts its replies and options."""
    script: Script = json.loads(ctx.job.metadata or "{}")
    if marker := script.get("marker"):
        with open(marker, "w") as out:
            out.write(ctx.job.metadata)
    replies: dict[str, Reply] = {
        **script.get("replies", {}),
        **{said: tool_call(tool) for said, tool in script.get("tool_calls", {}).items()},
    }
    session = VoiceSession(
        stt=ToneSTT(),
        # what a caller says without a scripted answer gets an empty reply
        llm=FakeLLM(lambda chat_ctx: replies.get(last_input(chat_ctx) or "", "")),
        tts=ToneTTS(),
        vad=None,
        turn_handling={"turn_detection": "stt"},
        aec_warmup_duration=0,
    )
    if goodbye := script.get("shutdown_on"):

        @session.on("conversation_item_added")
        def leave(event: ConversationItemAddedEvent) -> None:
            if event.item.type == "message" and event.item.text_content == goodbye:
                session.shutdown(drain=False)

    if script.get("app_rpc"):
        # An application method, registered before the call starts.
        await ctx.connect()

        async def echo(data: rtc.RpcInvocationData) -> str:
            return f"echo:{data.payload}"

        ctx.room.local_participant.register_rpc_method("app.echo", echo)

    if script.get("avatar"):
        await _Avatar().start(session, ctx.room)
    await session.start(
        Agent(instructions="Answer the caller.", tools=[read_device_temperature]),
        room=ctx.room,
        room_options=RoomOptions(delete_room_on_close=script.get("delete_room", False)),
    )

    if script.get("second_session"):
        second = VoiceSession(llm=FakeLLM(), vad=None, turn_handling={"turn_detection": "manual"})
        await second.start(
            Agent(instructions="A second session of the job."),
            room=ctx.room,
            room_options=RoomOptions(
                audio_input=False, audio_output=False, text_input=False, transcription_output=False
            ),
        )


@asynccontextmanager
async def agent_server(url: str, agent_name: str = AGENT_NAME) -> AsyncIterator[AgentServer]:
    """A registered worker serving ``compat_agent`` on the isolated SFU."""
    server = AgentServer(
        ws_url=url,
        api_key=KEY,
        api_secret=SECRET,
        agent_name=agent_name,
        num_idle_processes=1,
        listeners={"health": {"port": 0}},
    )
    server.rtc_session()(compat_agent)
    ready = asyncio.Event()
    server._on_started.append(ready.set)
    run = asyncio.create_task(server._run(devmode=True))
    try:
        await asyncio.wait_for(ready.wait(), 20)
        async with asyncio.timeout(10):
            while server.id in ("", "unregistered"):
                await asyncio.sleep(0.01)
        yield server
    finally:
        await asyncio.wait_for(server.aclose(), 15)
        await asyncio.wait_for(run, 15)


@pytest.fixture
async def compat_worker(sfu: str) -> AsyncIterator[AgentServer]:
    """The named worker the journeys' tokens dispatch."""
    async with agent_server(sfu) as running:
        yield running


# --- what every client does -------------------------------------------------------------------


def starter_token(
    room: str,
    identity: str = "user",
    *,
    agent: str | None = AGENT_NAME,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """A token as the agent starters' connection-details endpoint mints it: join, publish,
    subscribe and data, with ``roomConfig.agents`` dispatching ``agent``
    (agent-starter-react ``app/api/connection-details/route.ts``; client-sdk-js ``TokenSource``)."""
    token = (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
    )
    if agent is not None:
        dispatch = api.RoomAgentDispatch(agent_name=agent, metadata=json.dumps(metadata or {}))
        token = token.with_room_config(api.RoomConfiguration(agents=[dispatch]))
    return token.to_jwt()


@asynccontextmanager
async def joined(url: str, token: str) -> AsyncIterator[rtc.Room]:
    room = rtc.Room()
    try:
        await asyncio.wait_for(room.connect(url, token), 15)
        yield room
    finally:
        await asyncio.wait_for(room.disconnect(), 5)


def is_agent(participant: rtc.Participant) -> bool:
    """components-js ``useAgent.ts:521-575``: kind AGENT, not an avatar worker."""
    return (
        participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        and "lk.publish_on_behalf" not in participant.attributes
    )


async def wait_for_agent(room: rtc.Room, timeout: float = 20) -> rtc.RemoteParticipant:
    """The agent once ACTIVE (client-sdk-js ``waitUntilActiveAgentPresent``)."""
    active = rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE
    async with asyncio.timeout(timeout):
        while not (
            agents := [
                p for p in room.remote_participants.values() if is_agent(p) and p.state == active
            ]
        ):
            await asyncio.sleep(0.01)
    return agents[0]


async def wait_for_agent_state(room: rtc.Room, state: str, timeout: float = 20) -> None:
    """``lk.agent.state`` as components-js ``useAgent`` reads it."""
    agent = await wait_for_agent(room, timeout)
    async with asyncio.timeout(timeout):
        while agent.attributes.get("lk.agent.state") != state:
            await asyncio.sleep(0.01)


Transcription = tuple[str, dict[str, str], rtc.TextStreamReader]  # sender, header, stream


def read_transcriptions(room: rtc.Room) -> asyncio.Queue[Transcription]:
    """Every ``lk.transcription`` stream as it opens, as components-js ``useTranscriptions``
    receives them: the sender, the header's attributes, and the stream (whose
    ``info.attributes`` take the trailer's once read to its end)."""
    opened: asyncio.Queue[Transcription] = asyncio.Queue()
    room.register_text_stream_handler(
        "lk.transcription",
        lambda reader, sender: opened.put_nowait(
            (sender, dict(reader.info.attributes or {}), reader)
        ),
    )
    return opened


async def next_transcription(
    transcriptions: asyncio.Queue[Transcription], sender: str, timeout: float = 15
) -> tuple[dict[str, str], rtc.TextStreamReader]:
    """The next stream ``sender`` opens."""
    async with asyncio.timeout(timeout):
        while (transcription := await transcriptions.get())[0] != sender:
            pass
    return transcription[1], transcription[2]


async def publish_microphone(
    room: rtc.Room, *, preconnect: bool = False, sample_rate: int = 48000
) -> tuple[rtc.AudioSource, rtc.LocalTrackPublication]:
    source = rtc.AudioSource(sample_rate, 1)
    options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE, preconnect_buffer=preconnect
    )
    track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
    return source, await room.local_participant.publish_track(track, options)


async def speak(source: rtc.AudioSource, phrase: str, seconds: float = 0.8) -> None:
    """Say ``phrase`` into the microphone in real time, then stay quiet for 0.6 s."""
    pitch = next(f for f, text in PHRASES.items() if text == phrase)
    await play(source, tone(pitch, seconds, 48000) + bytes(48000 * 2 * 6 // 10))


async def play(source: rtc.AudioSource, pcm: bytes) -> None:
    for offset in range(0, len(pcm), 960):
        chunk = pcm[offset : offset + 960].ljust(960, b"\0")
        await source.capture_frame(
            rtc.AudioFrame(data=chunk, sample_rate=48000, num_channels=1, samples_per_channel=480)
        )


# --- pre-connect buffers, as each SDK sends them ------------------------------------------------

V1Client = Literal["js", "js-safari", "react-native", "swift", "android", "flutter"]
V1_CLIENTS: tuple[V1Client, ...] = get_args(V1Client)


# How each SDK sends ``lk.chat``: one text stream, text/plain, no attributes. client-sdk-js
# (components-js chat.ts:274) sends it inline and deflated (data streams v2) once every peer
# advertises support; the mobile SDKs send a plain v1 text stream (client-sdk-swift
# TextMessageSender.swift:31, client-sdk-android via components-android Chat.kt:82,
# client-sdk-flutter text_message_sender.dart:31).
CHAT_DEFLATED: dict[V1Client, bool] = {
    "js": True,
    "js-safari": True,
    "react-native": True,
    "swift": False,
    "android": False,
    "flutter": False,
}


class PreconnectWire(NamedTuple):
    sample_rate: int
    mime_type: str
    stream_name: str
    attributes: dict[str, str]
    total_size: bool
    chunk_bytes: int | None  # None: the whole buffer in one write
    recorder: str | None  # the MediaRecorder container, None for headerless PCM16


PRECONNECT_WIRE: dict[V1Client, PreconnectWire] = {
    # client-sdk-js LocalParticipant.ts:1105-1124: MediaRecorder WebM/Opus (Chrome, Firefox),
    # streamed chunk by chunk once the agent is active.
    "js": PreconnectWire(
        48000, "audio/webm;codecs=opus", "preconnect-buffer", {}, False, 1000, "webm"
    ),
    # The same sender in iOS Safari, whose MediaRecorder writes fragmented MP4/AAC.
    "js-safari": PreconnectWire(48000, "video/mp4", "preconnect-buffer", {}, False, 1000, "mp4"),
    # client-sdk-react-native runs the client-sdk-js sender over its MediaRecorder shim
    # (src/audio/MediaRecorder.ts:34): raw PCM16 from the native audio sink as audio/pcm, at the
    # track's 48 kHz mono, in 100 ms chunks.
    "react-native": PreconnectWire(48000, "audio/pcm", "preconnect-buffer", {}, False, 9600, None),
    # client-sdk-swift PreConnectAudioBuffer.swift:30,190-210: PCM16 at 24 kHz, no mime type
    # (StreamOptions.swift:163: application/octet-stream), no total size, buffer by buffer.
    "swift": PreconnectWire(24000, "application/octet-stream", "", {}, False, 4800, None),
    # client-sdk-android PreconnectAudioBuffer.kt:130-160: PCM16 at 48 kHz, octet-stream
    # (StreamOptions.kt:59), its total size, one write.
    "android": PreconnectWire(
        48000, "application/octet-stream", "preconnect-audio-buffer", {}, True, None, None
    ),
    # client-sdk-flutter pre_connect_audio_buffer.dart:322-340: PCM16 tagged commonFormat=int16,
    # octet-stream, its total size, one write.
    "flutter": PreconnectWire(
        48000, "application/octet-stream", "", {"commonFormat": "int16"}, True, None, None
    ),
}


async def send_preconnect_buffer(
    client: V1Client,
    room: rtc.Room,
    agent: str,
    track: str,
    recording: Callable[[int], bytes],
    attributes: dict[str, str] | None = None,
) -> None:
    """Stream what ``recording(sample_rate)`` returns to the agent the way ``client``'s SDK
    sends its pre-connect buffer."""
    wire = PRECONNECT_WIRE[client]
    data = recording(wire.sample_rate)
    if wire.recorder is not None:
        data = media_recorder(data, wire.recorder)
    writer = await room.local_participant.stream_bytes(
        wire.stream_name,
        total_size=len(data) if wire.total_size else None,
        mime_type=wire.mime_type,
        attributes={
            "trackId": track,
            "sampleRate": str(wire.sample_rate),
            "channels": "1",
            **wire.attributes,
            **(attributes or {}),
        },
        destination_identities=[agent],
        topic="lk.agent.pre-connect-audio-buffer",
    )
    step = wire.chunk_bytes or len(data)
    for offset in range(0, len(data), step):
        await writer.write(data[offset : offset + step])
    await writer.aclose()


def media_recorder(pcm: bytes, container: str) -> bytes:
    """What a browser's MediaRecorder writes for 48 kHz ``pcm``: ``webm`` (Opus) or fragmented
    ``mp4`` (AAC)."""
    import av

    out = io.BytesIO()
    fragmented = {"movflags": "frag_keyframe+empty_moov+default_base_moof"}
    with av.open(out, "w", format=container, options=fragmented if container == "mp4" else {}) as f:
        stream = f.add_stream("libopus" if container == "webm" else "aac", rate=48000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            np.frombuffer(pcm, dtype=np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = 48000
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            f.mux(packet)
    return out.getvalue()


# --- avatars: what the person sees and hears, and the v1 peers of a 2.0 avatar -------------------


def avatar_token(room: str, agent_identity: str) -> str:
    """An avatar worker's token as the avatar plugins mint it: kind AGENT, ``lk.publish_on_behalf``
    naming the agent it speaks for (v1 ``voice/avatar/_types.py``; tavus, bey, anam ``avatar.py``)."""
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(AVATAR_IDENTITY)
        .with_kind("agent")
        .with_attributes({"lk.publish_on_behalf": agent_identity})
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


def v1_agent_token(room: str) -> str:
    """A v1 agent's job token: kind AGENT, never an avatar."""
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(V1_AGENT_IDENTITY)
        .with_kind("agent")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


async def wait_for_avatar(room: rtc.Room, agent_identity: str) -> rtc.RemoteParticipant:
    """The worker speaking for the agent (components-js ``useAgent.ts:521-575``: kind AGENT with
    ``lk.publish_on_behalf`` naming it)."""
    async with asyncio.timeout(20):
        while not (
            avatars := [
                p
                for p in room.remote_participants.values()
                if p.attributes.get("lk.publish_on_behalf") == agent_identity
            ]
        ):
            await asyncio.sleep(0.01)
    return avatars[0]


async def track_of(
    participant: rtc.RemoteParticipant, source: rtc.TrackSource.ValueType
) -> rtc.Track:
    """``participant``'s subscribed track of ``source``, once it publishes one (clients pick the
    avatar's audio and video by source: components-js ``useAgent.ts:179-202``)."""
    async with asyncio.timeout(20):
        while not (
            tracks := [
                p.track
                for p in participant.track_publications.values()
                if p.source == source and p.track is not None
            ]
        ):
            await asyncio.sleep(0.01)
    return tracks[0]


class Heard:
    """What the person hears on a participant's microphone: when each 10 ms frame carried the
    agent's tone."""

    def __init__(self) -> None:
        self.tone_at: list[float] = []

    @property
    def seconds(self) -> float:
        return len(self.tone_at) / 100


class Seen:
    """What the person sees on a participant's camera: how many frames were lit (the face
    mouthing speech) and dark (idle)."""

    def __init__(self) -> None:
        self.lit = 0
        self.dark = 0


@asynccontextmanager
async def hearing(participant: rtc.RemoteParticipant) -> AsyncIterator[Heard]:
    heard = Heard()

    async def listen() -> None:
        track = await track_of(participant, rtc.TrackSource.SOURCE_MICROPHONE)
        stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1, frame_size_ms=10)
        try:
            async for event in stream:
                rms, pitch = loudness_and_pitch(bytes(event.frame.data), 48000)
                if rms > 300 and pitch == AGENT_TONE:
                    heard.tone_at.append(time.monotonic())
        finally:
            await stream.aclose()

    task = asyncio.create_task(listen())
    try:
        yield heard
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def watching(participant: rtc.RemoteParticipant) -> AsyncIterator[Seen]:
    seen = Seen()

    async def watch() -> None:
        track = await track_of(participant, rtc.TrackSource.SOURCE_CAMERA)
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
        try:
            async for event in stream:
                brightness = np.frombuffer(bytes(event.frame.data), dtype=np.uint8)[0::4].mean()
                if brightness > 192:
                    seen.lit += 1
                elif brightness < 64:
                    seen.dark += 1
        finally:
            await stream.aclose()

    task = asyncio.create_task(watch())
    try:
        yield seen
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def avatar_worker(
    url: str, room: str, agent_identity: str
) -> AsyncIterator[FakeVideoGenerator]:
    """A 2.0 avatar worker: the real ``DataStreamAudioReceiver`` and ``AvatarRunner``, publishing
    the agent's speech and ``FakeVideoGenerator``'s face (yielded) as the avatar. Its code is an
    avatar provider's v1 worker unchanged, so it also stands for a provider upgraded to 2.0.
    Yields once it is in the room, its runner starting: the receiver waits there for its agent,
    which may join after it."""
    async with joined(url, avatar_token(room, agent_identity)) as worker:
        face = FakeVideoGenerator()
        runner = AvatarRunner(
            worker,
            audio_recv=DataStreamAudioReceiver(worker, frame_size_ms=40),
            video_gen=face,
            options=AvatarOptions(FakeVideoGenerator.SIZE, FakeVideoGenerator.SIZE, 25, 24000, 1),
        )
        starting = asyncio.create_task(runner.start())
        try:
            yield face
        finally:
            starting.cancel()
            await asyncio.gather(starting, return_exceptions=True)
            await runner.aclose()


class V1AvatarWorker:
    """An avatar provider's worker built on Agents v1, as it acts on the wire (v1
    ``voice/avatar/_datastream_io.py:362-620`` ``DataStreamAudioReceiver`` and ``_runner.py``
    ``AvatarRunner``). It plays each ``lk.audio_stream`` segment from the agent on its microphone
    in real time. It reports on the plain methods: ``lk.playback_started`` at the first frame,
    giving up on UNSUPPORTED_METHOD, then ``lk.playback_finished``. On ``lk.clear_buffer`` it
    stops and reports the segment interrupted. ``reports`` keeps each finish payload and the
    agent's answer."""

    def __init__(self, room: rtc.Room, agent_identity: str) -> None:
        self.room = room
        self.agent_identity = agent_identity
        self.reports: list[tuple[dict[str, object], str]] = []
        self.clears = 0
        self._speaker = rtc.AudioSource(24000, 1, queue_size_ms=100)
        self._camera = rtc.VideoSource(FakeVideoGenerator.SIZE, FakeVideoGenerator.SIZE)
        self._segments: asyncio.Queue[rtc.ByteStreamReader] = asyncio.Queue()
        self._played: float | None = None  # the playing segment's position
        self._tasks: set[asyncio.Task[object]] = set()
        room.register_byte_stream_handler("lk.audio_stream", self._received)
        room.local_participant.register_rpc_method("lk.clear_buffer", self._clear)

    async def start(self) -> None:
        """``_runner.py:104-119``: its audio, then its video."""
        await self.room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("avatar_audio", self._speaker),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        self._camera.capture_frame(FakeVideoGenerator.face(lit=False))
        await self.room.local_participant.publish_track(
            rtc.LocalVideoTrack.create_video_track("avatar_video", self._camera),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )
        self._spawn(self._play())

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _spawn(self, work: Coroutine[object, object, object]) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _received(self, reader: rtc.ByteStreamReader, sender: str) -> None:
        if sender == self.agent_identity:  # streams from anyone else are ignored (:437-444)
            self._segments.put_nowait(reader)

    async def _play(self) -> None:
        while True:
            reader = await self._segments.get()
            rate = int((reader.info.attributes or {})["sample_rate"])
            step = rate // 50 * 2  # 20 ms of PCM16 mono
            pending = b""
            self._played = 0.0
            async for chunk in reader:
                pending += chunk
                while self._played is not None and len(pending) >= step:
                    frame = rtc.AudioFrame(pending[:step], rate, 1, step // 2)
                    pending = pending[step:]
                    await self._speaker.capture_frame(frame)
                    if self._played == 0.0:
                        self._spawn(self._report("lk.playback_started", ""))
                    if self._played is not None:
                        self._played += 0.02
            if self._played is None:
                continue  # cleared: the clear reported it, and the rest was discarded
            await self._speaker.wait_for_playout()
            await self._finished(self._played, interrupted=False)
            self._played = None

    async def _clear(self, data: rtc.RpcInvocationData) -> str:
        if data.caller_identity != self.agent_identity:
            return "reject"
        self.clears += 1
        if self._played is not None:
            self._speaker.clear_queue()
            self._spawn(self._finished(self._played, interrupted=True))
            self._played = None
        return "ok"

    async def _finished(self, position: float, *, interrupted: bool) -> None:
        payload: dict[str, object] = {
            "playback_position": position,
            "interrupted": interrupted,
            "synchronized_transcript": None,
        }
        answer = await self._report("lk.playback_finished", json.dumps(payload))
        self.reports.append((payload, answer))

    async def _report(self, method: str, payload: str) -> str:
        try:
            return await self.room.local_participant.perform_rpc(
                destination_identity=self.agent_identity, method=method, payload=payload
            )
        except rtc.RpcError as error:
            unsupported = error.code == rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD
            if method == "lk.playback_started" and unsupported:
                return "unsupported"  # the agent does not wait for playback starts
            raise


@asynccontextmanager
async def v1_avatar_worker(
    url: str, room: str, agent_identity: str
) -> AsyncIterator[V1AvatarWorker]:
    async with joined(url, avatar_token(room, agent_identity)) as joined_room:
        worker = V1AvatarWorker(joined_room, agent_identity)
        await worker.start()
        try:
            yield worker
        finally:
            await worker.aclose()


class V1AvatarAgent:
    """An agent built on Agents v1 streaming its speech to an avatar worker, as it acts on the
    wire (v1 ``voice/avatar/_datastream_io.py:36-360`` ``DataStreamAudioOutput``). It takes the
    worker's reports on the plain ``lk.playback_finished`` from the moment it is in the room
    (``finished``), and leaves ``lk.playback_started`` unregistered, as without
    ``wait_playback_start``. Each segment waits only for the worker to be in the room, not for
    its video (no ``wait_remote_track``), then is one ``lk.audio_stream`` named
    ``AUDIO_<shortuuid>`` with its ``sample_rate`` and ``num_channels``; ``lk.clear_buffer``
    interrupts."""

    def __init__(self, room: rtc.Room) -> None:
        self.room = room
        self.finished: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        room.local_participant.register_rpc_method("lk.playback_finished", self._on_finished)

    def _on_finished(self, data: rtc.RpcInvocationData) -> str:
        if data.caller_identity != AVATAR_IDENTITY:
            return "reject"
        self.finished.put_nowait(json.loads(data.payload))
        return "ok"

    async def say(self, seconds: float) -> None:
        """One segment of the agent's 1 kHz tone, sent once the worker is in the room
        (``_start_task``: ``wait_for_participant``) and written as its TTS produces it, faster
        than real time, then closed (``flush()``)."""
        async with asyncio.timeout(20):
            while AVATAR_IDENTITY not in self.room.remote_participants:
                await asyncio.sleep(0.005)
        writer = await self.room.local_participant.stream_bytes(
            name=shortuuid("AUDIO_"),
            topic="lk.audio_stream",
            destination_identities=[AVATAR_IDENTITY],
            attributes={"sample_rate": "24000", "num_channels": "1"},
        )
        audio = tone(AGENT_TONE, seconds, 24000)
        for offset in range(0, len(audio), 960):
            await writer.write(audio[offset : offset + 960])
        await writer.aclose()

    async def interrupt(self) -> None:
        await self.room.local_participant.perform_rpc(
            destination_identity=AVATAR_IDENTITY, method="lk.clear_buffer", payload=""
        )
