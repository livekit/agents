"""End-to-end calls on LiveKit Cloud: one agent file served by livekit-agents-next, scripted
callers driving it.

This file runs as a worker (``e2e.py agent``) on the current interpreter, registered on the
project. Each scenario makes a room, dispatches the agent with the scenario in the metadata, and
drives a ``livekit.rtc`` caller through it: speech from WAVs (synthesized once and cached), chat on
``lk.chat``, host commands over RPC, DTMF, leaving, rejoining, and a small avatar worker that
publishes video from the audio it receives.

Recorded per run: every ``session.on(...)`` event with its payload, ``session.history``, the
item metrics (latency per stage), what the caller saw (``lk.transcription`` streams, the agent
state attribute, the agent's audio, the avatar's audio and video) and the worker's error lines.
Each run is judged on its own: the scenario did what it is for (its checks), what the caller saw
is what the history says, no error events, no timeouts. There is no comparison with any other
implementation (owner decision 22). A failing scenario is run again; both attempts are kept and
reported, and a scenario fails when either attempt fails.

    python livekit-agents-next/scripts/e2e.py                    # everything
    python livekit-agents-next/scripts/e2e.py turns barge_in     # some scenarios
    python livekit-agents-next/scripts/e2e.py --jobs 6
    python livekit-agents-next/scripts/e2e.py --list
    python livekit-agents-next/scripts/e2e.py --env-file ~/livekit/.env --keep runs/
    python livekit-agents-next/scripts/e2e.py --report runs/     # judge kept runs again

Needs LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET (a project with LiveKit Inference),
from the environment or ``--env-file``, and livekit-agents-next with
livekit-plugins-noise-cancellation installed in the current interpreter. Exits 0 when every
scenario passed, 1 otherwise. Replaced by the tests/e2e pytest suite (K-18).
"""

import argparse
import asyncio
import contextlib
import dataclasses
import enum
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STACK = "next"
SAMPLE_RATE = 24000
LOUD = 400.0  # int16 RMS of an audible frame
STT, LLM, TTS, REALTIME = (
    "deepgram/nova-3",
    "openai/gpt-4.1-mini",
    "cartesia/sonic-3",
    "openai/gpt-realtime",
)
AVATAR = "e2e-avatar"

LINES = {
    "hello": "Hi there, can you hear me?",
    "weather": "What's the weather like in Paris today?",
    "delete": "Please delete my account.",
    "yes": "Yes, go ahead.",
    "book": "Can you book a table for eight pm?",
    "billing": "I have a question about my bill, please transfer me to billing.",
    "interrupt": "Sorry to interrupt, what time do you close today?",
    "hmm": "Hmm, umm.",
    "count": "Please count slowly from one to twenty.",
    "receipt": "Please send my refund receipt to a different email address.",
    "email": "It's theo at example dot com.",
}
LONG = (
    "Our store opens at nine in the morning and closes at six in the evening. "
    "On weekends we open a little later, at ten, and we close at five. "
    "We are closed on public holidays, except for the week before the new year, "
    "when we stay open until eight every evening to help with the holiday rush."
)
HOST_SAY = "This is a message from the host."
BILLING_GREETING = "Billing desk here, how can I help?"
RECEIPT_SET = "The refund receipt email is now"
# texts the scenarios fix: compared whole, where a model's words are compared by shape
FIXED = [LONG, HOST_SAY, BILLING_GREETING, "Please confirm", RECEIPT_SET]

INSTRUCTIONS = (
    "You are a test agent on a phone line. Answer in one short sentence of at most ten words "
    "(when asked to count, count out loud, one number per sentence), "
    "and never ask a follow-up question. If asked about the weather, call get_weather. If asked "
    "to delete the account, call delete_account. If asked to book a table, call book_table. If "
    "the person asks for billing, call transfer_to_billing."
)
EMAIL_INSTRUCTIONS = (
    "You collect the person's new receipt email address on a phone line. Ask for it in one short "
    "sentence. As soon as they say one, call confirm_email with it written as an address "
    "(theo at example dot com is theo@example.com); never ask them to confirm it."
)
EVENTS = (
    "user_state_changed",
    "agent_state_changed",
    "user_input_transcribed",
    "user_transcription_timeout",
    "conversation_item_added",
    "agent_false_interruption",
    "overlapping_speech",
    "function_tools_executed",
    "metrics_collected",
    "speech_created",
    "tool_execution_updated",
    "error",
    "close",
)


# ====================================================================================== agent
# Runs in each stack's venv: everything here is the developer's side of one agent file.


def jsonable(v: Any, depth: int = 0) -> Any:
    """A payload as JSON, whatever the stack built it from (pydantic, dataclass, plain)."""
    if depth > 8:
        return repr(v)[:120]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, enum.Enum):
        return jsonable(v.value, depth + 1)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(v)} bytes>"
    if isinstance(v, dict):
        return {str(k): jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [jsonable(x, depth + 1) for x in v]
    name = type(v).__name__
    if name == "SpeechHandle":
        return {"__type": name, "id": v.id, "allow_interruptions": v.allow_interruptions}
    if name in ("Agent", "AgentTask") or "Agent" in [c.__name__ for c in type(v).__mro__]:
        return {"__type": "Agent", "id": getattr(v, "id", None)}
    if hasattr(v, "model_dump"):
        with contextlib.suppress(Exception):
            return {"__type": name, **jsonable(v.model_dump(), depth + 1)}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        fields = {f.name: getattr(v, f.name, None) for f in dataclasses.fields(v)}
        return {"__type": name, **jsonable(fields, depth + 1)}
    if hasattr(v, "__dict__") and not callable(v):
        public = {k: x for k, x in vars(v).items() if not k.startswith("_")}
        return {"__type": name, **jsonable(public, depth + 1)}
    return repr(v)[:200]


class Recorder:
    def __init__(self, path: str) -> None:
        self._f = open(path, "a", buffering=1)  # noqa: SIM115

    def write(self, kind: str, **fields: Any) -> None:
        line = {"t": time.time(), "kind": kind, **fields}
        self._f.write(json.dumps(line, default=repr) + "\n")


try:  # either stack's livekit.agents; the orchestrator's own python may have none
    from livekit.agents import Agent, RunContext, function_tool
except ImportError:
    function_tool = None

if function_tool is not None:  # module level: the reference pickles an agent's on_enter
    try:
        from livekit.agents.llm.async_toolset import AsyncToolset  # the reference
    except ImportError:
        from livekit.agents.llm.tool_context import AsyncToolset  # type: ignore[no-redef]

    billing = Agent(instructions=INSTRUCTIONS + " You are the billing desk.")

    @billing.on_enter
    async def greet(session: Any) -> None:
        session.say(BILLING_GREETING)

    @function_tool
    async def get_weather(ctx: RunContext, city: str) -> str:
        """Get today's weather in a city.

        Args:
            city: the city's name
        """
        return f"Sunny and 21 degrees in {city}."

    @function_tool(needs_approval=True)
    async def delete_account(ctx: RunContext) -> str:
        """Delete the caller's account."""
        return "The account is deleted."

    @function_tool
    async def book_table(ctx: RunContext, time: str) -> str:
        """Book a table at the restaurant.

        Args:
            time: when, as the person said it
        """
        await ctx.update("Checking which tables are free.")
        await asyncio.sleep(3.0)
        return f"A table is booked at {time}."

    try:  # 2.0's live task; the reference has none, and runs no scenario using it
        from livekit.agents import Completed, VoiceTask
    except ImportError:
        VoiceTask = None

    if VoiceTask is not None:
        email_collector = Agent(instructions=EMAIL_INSTRUCTIONS, output_type=str)

        @email_collector.tool
        async def confirm_email(ctx: RunContext, email_address: str) -> Completed[str]:
            """Confirm the receipt email address.

            Args:
                email_address: the address, as an email address
            """
            return Completed(email_address)

        @function_tool
        async def change_receipt_email(ctx: RunContext) -> str:
            """Collect and confirm a new email address for the refund receipt."""
            email = await VoiceTask(agent=email_collector).run()
            return f"{RECEIPT_SET} {email}."

    @function_tool
    async def transfer_to_billing(ctx: RunContext) -> str:
        """Transfer the caller to the billing desk."""
        ctx.session.update_agent(billing)  # 2.0 hands off explicitly; returning an Agent is 1.x
        return "Transferred to billing."


async def entry(ctx: Any) -> None:
    import inspect

    from livekit import rtc
    from livekit.agents import VoiceSession, inference, room_io

    meta = json.loads(ctx.job.metadata or "{}")
    cfg = SCENARIOS[meta["scenario"]].agent
    rec = Recorder(meta["out"])
    rec.write("start", scenario=meta["scenario"])

    kwargs: dict[str, Any] = {"tools": [AsyncToolset(id="background", tools=[book_table])]}
    if cfg.get("realtime"):
        kwargs["llm"] = inference.RealtimeModel(REALTIME)
    else:
        kwargs.update(stt=STT, llm=LLM, tts=TTS)
    if "user_away_timeout" in cfg:
        kwargs["user_away_timeout"] = cfg["user_away_timeout"]
    session: Any = VoiceSession(**kwargs)
    tools = [get_weather, delete_account, transfer_to_billing]
    instructions = INSTRUCTIONS
    if cfg.get("voice_task"):
        tools.append(change_receipt_email)
        instructions += " If asked to change the receipt email, call change_receipt_email."
    agent = Agent(instructions=instructions, tools=tools)

    for name in EVENTS:

        def on_event(ev: Any, name: str = name) -> None:
            rec.write("event", name=name, type=type(ev).__name__, data=jsonable(ev))
            if name == "close":
                snapshot("close")

        # 2.0 removed metrics_collected from the session
        with contextlib.suppress(ValueError):
            session.on(name, on_event)

    def snapshot(why: str) -> None:
        with contextlib.suppress(Exception):
            rec.write(
                "snapshot",
                why=why,
                history=jsonable(list(session.history.items)),
                agent_state=getattr(session, "agent_state", None),
                user_state=getattr(session, "user_state", None),
            )

    ctx.room.on(
        "sip_dtmf_received",
        lambda ev: rec.write("room", name="dtmf", digit=ev.digit, code=ev.code),
    )

    options: dict[str, Any] = {}
    for key in ("close_on_disconnect", "delete_room_on_close"):
        if key in cfg:
            options[key] = cfg[key]
    if cfg.get("noise_cancellation"):
        from livekit.plugins import noise_cancellation

        options["audio_input"] = room_io.AudioInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    start: dict[str, Any] = {"agent": agent, "room": ctx.room}
    if options:
        start["room_options"] = room_io.RoomOptions(**options)
    params = inspect.signature(session.start).parameters
    rec.write("capabilities", start_params=sorted(params))
    if "record" in params:
        start["record"] = bool(cfg.get("record"))
    if cfg.get("avatar"):
        from livekit.agents.voice.avatar import DataStreamAudioOutput

        session.output.replace_audio_tail(
            DataStreamAudioOutput(
                ctx.room, destination_identity=AVATAR, wait_remote_track=rtc.TrackKind.KIND_VIDEO
            )
        )
    # the session connects the room itself: before it, the pre-connect audio would be missed
    await session.start(**start)
    # after the session's own close, which start() registered first
    rec.write("started")
    if not ctx.room.isconnected():
        rec.write("note", what="session.start() left the room unconnected")
        await ctx.connect()

    @ctx.room.local_participant.register_rpc_method("e2e.cmd")
    async def _cmd(data: rtc.RpcInvocationData) -> str:
        cmd = json.loads(data.payload)
        rec.write("cmd", cmd=cmd)
        if cmd["op"] == "say":
            session.say(cmd["text"])
        elif cmd["op"] == "reply":
            session.generate_reply(instructions=cmd["text"])
        elif cmd["op"] == "end":
            snapshot("end")
        return "ok"


def run_agent(name: str) -> None:
    from livekit.agents import AgentServer, cli

    # the scripted calls are the only load: a busy machine must not turn them away
    server = AgentServer(agent_name=name, _load_fnc=lambda: 0.0)
    server._update_options(listeners={"health": {"port": 0}})
    server.rtc_session()(entry)
    sys.argv = [sys.argv[0], "start"]
    cli.run_app(server)


# ===================================================================================== caller
# Runs in the driving process: the person, the avatar, and the scenario scripts.


async def speech_frames(cache: Path) -> dict[str, list[Any]]:
    """Each caller line as 10 ms frames, synthesized once (LiveKit Inference) and kept as WAVs."""
    import aiohttp
    import numpy as np

    from livekit import rtc
    from livekit.agents import inference

    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[Any]] = {}
    async with aiohttp.ClientSession() as http:
        for key, text in LINES.items():
            path = cache / f"{hashlib.sha1((TTS + text).encode()).hexdigest()[:12]}.wav"
            if not path.exists():
                tts = inference.TTS(TTS, http_session=http, sample_rate=SAMPLE_RATE)
                frames = [a.frame async for a in tts.synthesize(text)]
                pcm = b"".join(bytes(f.data) for f in frames)
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(pcm)
            with wave.open(str(path)) as w:
                assert w.getframerate() == SAMPLE_RATE
                pcm16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if key == "hmm":  # soft, twice: voice activity for over a second, no words
                pcm16 = (np.tile(pcm16, 2) * 0.5).astype(np.int16)
            step = SAMPLE_RATE // 100
            out[key] = [
                rtc.AudioFrame(pcm16[i : i + step].tobytes(), SAMPLE_RATE, 1, step)
                for i in range(0, len(pcm16) - step + 1, step)
            ]
    return out


def rms(data: Any) -> float:
    import numpy as np

    a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(math.sqrt(float(np.mean(a**2)))) if a.size else 0.0


class Call:
    """One scenario on one stack: its room, its caller, what the caller saw."""

    def __init__(
        self, stack: str, scenario: str, agent_name: str, lkapi: Any, speech: Any, out: Path
    ) -> None:
        self.stack, self.scenario, self.agent_name = stack, scenario, agent_name
        self.lkapi, self.speech, self.out = lkapi, speech, out
        self.room_name = f"e2e-{scenario}-{stack}-{uuid.uuid4().hex[:6]}"
        self.room: Any = None
        self.tasks: set[asyncio.Task[Any]] = set()
        self.log: list[dict[str, Any]] = []  # what the caller saw, in order
        self.agent_state = "initializing"
        self.loud_at = 0.0
        self.queue: deque[Any] = deque()
        self.drained = asyncio.Event()
        self.heard: list[float] = []  # caller stopped speaking -> first loud agent frame
        self.envelopes: dict[str, list[tuple[float, float]]] = {"audio": [], "video": []}
        self.avatar: Any = None
        self.buffered: bytearray | None = None  # the pre-connect microphone, until the agent joins

    def note(self, kind: str, **fields: Any) -> None:
        self.log.append({"t": time.time(), "kind": kind, **fields})

    def spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    # ------------------------------------------------------------------------------ the room

    def token(self, identity: str, attributes: dict[str, str] | None = None) -> str:
        from livekit import api

        t = api.AccessToken().with_identity(identity)
        t = t.with_grants(api.VideoGrants(room_join=True, room=self.room_name))
        if attributes:
            t = t.with_attributes(attributes)
        return t.to_jwt()

    async def join(self, *, preconnect: bool = False) -> None:
        from livekit import rtc

        room = rtc.Room()
        self.room = room

        def on_text(reader: Any, sender: str) -> None:
            async def read() -> None:
                attrs = dict(reader.info.attributes)
                text = await reader.read_all()
                self.note("transcript", sender=sender, text=text, attrs=attrs)

            self.spawn(read())

        def on_attrs(changed: dict[str, str], participant: Any) -> None:
            if "lk.agent.state" in changed:
                self.agent_state = changed["lk.agent.state"]
                self.note("agent_state", state=self.agent_state)

        def on_track(track: Any, pub: Any, participant: Any) -> None:
            kind = "audio" if track.kind == rtc.TrackKind.KIND_AUDIO else "video"
            self.note("track", media=kind, who=_who(participant.identity))
            self.spawn(self._watch(track, kind))

        room.register_text_stream_handler("lk.transcription", on_text)
        room.on("participant_attributes_changed", on_attrs)
        room.on("track_subscribed", on_track)
        room.on("participant_connected", lambda p: self.note("joined", who=_who(p.identity)))
        room.on("participant_disconnected", lambda p: self.note("left", who=_who(p.identity)))
        await room.connect(os.environ["LIVEKIT_URL"], self.token("caller"))
        source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=100)
        track = rtc.LocalAudioTrack.create_audio_track("mic", source)
        pub = await room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_MICROPHONE, preconnect_buffer=preconnect
            ),
        )
        self.note("mic", sid=pub.sid)
        self.spawn(self._mic(source))

    async def _mic(self, source: Any) -> None:
        """A microphone: the queued speech, silence the rest of the time, in real time."""
        from livekit import rtc

        silence = rtc.AudioFrame(b"\x00\x00" * 240, SAMPLE_RATE, 1, 240)
        while True:
            frame = self.queue.popleft() if self.queue else silence
            if self.buffered is not None:
                self.buffered += bytes(frame.data)
            await source.capture_frame(frame)
            if frame is not silence and not self.queue:
                self.drained.set()

    async def send_buffered(self) -> None:
        """What a client SDK does with a pre-connect microphone (the Python SDK only flags the
        track): the audio from before the agent joined, as one byte stream to it."""
        data, self.buffered = bytes(self.buffered or b""), None
        mic = next(x["sid"] for x in reversed(self.log) if x["kind"] == "mic")
        writer = await self.room.local_participant.stream_bytes(
            "pre-connect-audio",
            mime_type="audio/pcm",
            topic="lk.agent.pre-connect-audio-buffer",
            destination_identities=[self.agent.identity],
            attributes={"trackId": mic, "sampleRate": str(SAMPLE_RATE), "channels": "1"},
        )
        await writer.write(data)
        await writer.aclose()
        self.note("sent_buffered", seconds=round(len(data) / 2 / SAMPLE_RATE, 2))

    async def _watch(self, track: Any, kind: str) -> None:
        import numpy as np

        from livekit import rtc

        if kind == "audio":
            async for ev in rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1):
                level = rms(ev.frame.data)
                now = time.time()
                self.envelopes["audio"].append((now, level))
                if level > LOUD:
                    self.loud_at = now
        else:
            async for ev in rtc.VideoStream(track):
                frame = ev.frame.convert(rtc.VideoBufferType.RGBA)
                px = np.frombuffer(frame.data, dtype=np.uint8)[:: 4 * 97]
                self.envelopes["video"].append((time.time(), float(px.mean())))

    async def start(self, *, preconnect: bool = False, avatar: bool = False) -> None:
        from livekit import api

        await self.lkapi.room.create_room(api.CreateRoomRequest(name=self.room_name))
        if preconnect:
            self.buffered = bytearray()
        await self.join(preconnect=preconnect)
        if preconnect:
            await self.say("hello", settle=False)
        await self.lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=self.agent_name,
                room=self.room_name,
                metadata=json.dumps({"scenario": self.scenario, "out": str(self.out)}),
            )
        )
        await self.until(lambda: self.agent is not None, 30, "the agent to join")
        if preconnect:
            await self.send_buffered()
        if avatar:
            await self.start_avatar()
        if not preconnect:
            await self.until(lambda: self.agent_state == "listening", 30, "the agent to listen")

    @property
    def agent(self) -> Any:
        from livekit import rtc

        kind = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        return next((p for p in self.room.remote_participants.values() if p.kind == kind), None)

    # ------------------------------------------------------------------------ the person acts

    async def until(self, predicate: Callable[[], bool], timeout: float, what: str) -> bool:
        deadline = time.time() + timeout
        while not predicate():
            if time.time() > deadline:
                self.note("timeout", what=what)
                return False
            await asyncio.sleep(0.05)
        return True

    async def say(self, line: str, *, settle: bool = True) -> None:
        self.note("say", line=line)
        self.drained.clear()
        self.queue.extend(self.speech[line])
        await self.drained.wait()
        stopped = time.time()
        if settle:
            if await self.until(lambda: self.loud_at > stopped, 15, f"a reply to {line}"):
                self.heard.append(self.loud_at - stopped)
            await self.settle()

    async def settle(self, timeout: float = 30) -> None:
        """The agent listens and has been quiet for 1.5 s."""
        await self.until(
            lambda: self.agent_state == "listening" and time.time() - self.loud_at > 1.5,
            timeout,
            "the agent to settle",
        )

    async def speaking_for(self, seconds: float) -> None:
        await self.until(lambda: time.time() - self.loud_at < 0.1, 20, "the agent to speak")
        await asyncio.sleep(seconds)

    async def cmd(self, op: str, text: str = "") -> None:
        agent = self.agent
        if agent is None:
            self.note("timeout", what=f"no agent for {op}")
            return
        try:
            await self.room.local_participant.perform_rpc(
                destination_identity=agent.identity,
                method="e2e.cmd",
                payload=json.dumps({"op": op, "text": text}),
                response_timeout=10,
            )
        except Exception as e:
            self.note("rpc_error", op=op, error=str(e))

    async def chat(self, text: str) -> None:
        self.note("chat", text=text)
        await self.room.local_participant.send_text(text, topic="lk.chat")

    async def dtmf(self, digit: str) -> None:
        self.note("dtmf", digit=digit)
        await self.room.local_participant.publish_dtmf(code=int(digit), digit=digit)

    async def leave(self) -> None:
        self.note("leave")
        for t in list(self.tasks):
            t.cancel()
        await self.room.disconnect()
        self.queue.clear()

    async def room_gone(self, timeout: float) -> bool:
        from livekit import api

        deadline = time.time() + timeout
        while time.time() < deadline:
            rooms = await self.lkapi.room.list_rooms(api.ListRoomsRequest(names=[self.room_name]))
            if not rooms.rooms:
                return True
            await asyncio.sleep(0.5)
        return False

    # ------------------------------------------------------------------------------ the avatar

    async def start_avatar(self) -> None:
        """An avatar worker in the room: the agent's audio back out, with a video frame per 40 ms
        whose brightness is that audio's loudness, through AvatarRunner."""
        from livekit import rtc
        from livekit.agents.voice.avatar import (
            AudioSegmentEnd,
            AvatarOptions,
            AvatarRunner,
            DataStreamAudioReceiver,
            VideoGenerator,
        )

        room = rtc.Room()
        attrs = {"lk.publish_on_behalf": self.agent.identity}
        await room.connect(os.environ["LIVEKIT_URL"], self.token(AVATAR, attrs))
        fps, w, h = 25, 64, 48
        chunk = SAMPLE_RATE // fps

        class Envelope(VideoGenerator):
            def __init__(self) -> None:
                self.q: asyncio.Queue[Any] = asyncio.Queue()
                self.pending = b""

            async def push_audio(self, frame: Any) -> None:
                await self.q.put(frame)

            def clear_buffer(self) -> None:
                while not self.q.empty():
                    self.q.get_nowait()
                self.pending = b""

            def __aiter__(self) -> Any:
                return self._frames()

            def _video(self, level: float) -> Any:
                shade = max(0, min(255, int(level / 8)))
                return rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, bytes([shade]) * (w * h * 4))

            async def _frames(self) -> Any:
                while True:
                    try:
                        item = await asyncio.wait_for(self.q.get(), 1 / fps)
                    except TimeoutError:
                        yield self._video(0)
                        continue
                    if isinstance(item, AudioSegmentEnd):
                        self.pending = b""
                        yield item
                        continue
                    self.pending += bytes(item.data)
                    while len(self.pending) >= chunk * 2:
                        data, self.pending = self.pending[: chunk * 2], self.pending[chunk * 2 :]
                        yield self._video(rms(data))
                        yield rtc.AudioFrame(data, SAMPLE_RATE, 1, chunk)

        self.avatar = AvatarRunner(
            room,
            audio_recv=DataStreamAudioReceiver(room),
            video_gen=Envelope(),
            options=AvatarOptions(
                video_width=w,
                video_height=h,
                video_fps=fps,
                audio_sample_rate=SAMPLE_RATE,
                audio_channels=1,
            ),
        )
        await self.avatar.start()
        self._avatar_room = room

    def av_lag(self) -> float | None:
        """Video behind audio, in seconds, from the loudness the caller heard and saw."""
        import numpy as np

        audio, video = self.envelopes["audio"], self.envelopes["video"]
        if len(audio) < 50 or len(video) < 25:
            return None
        t0 = min(audio[0][0], video[0][0])
        t1 = max(audio[-1][0], video[-1][0])
        grid = np.arange(t0, t1, 0.01)
        a = np.interp(grid, [t for t, _ in audio], [v > LOUD for _, v in audio])
        v = np.interp(grid, [t for t, _ in video], [x > LOUD / 8 for _, x in video])
        if a.sum() < 10 or v.sum() < 10:
            return None
        lags = range(-50, 51)
        scores = [
            float(np.dot(a[max(0, -k) : len(a) - max(0, k)], v[max(0, k) : len(v) - max(0, -k)]))
            for k in lags
        ]
        return lags[int(np.argmax(scores))] / 100

    async def close(self) -> None:
        from livekit import api

        with contextlib.suppress(Exception):
            if self.room is not None and self.room.isconnected():
                await self.cmd("end")
        for t in list(self.tasks):
            t.cancel()
        with contextlib.suppress(Exception):
            if self.avatar is not None:
                await self.avatar.aclose()
                await self._avatar_room.disconnect()
        with contextlib.suppress(Exception):
            if self.room is not None:
                await self.room.disconnect()
        with contextlib.suppress(Exception):
            await self.lkapi.room.delete_room(api.DeleteRoomRequest(room=self.room_name))


def _who(identity: str) -> str:
    if identity == AVATAR:
        return "avatar"
    if identity == "caller":
        return "caller"
    return "agent"


# ================================================================================== scenarios


@dataclasses.dataclass
class Scenario:
    drive: Callable[[Call], Awaitable[None]]
    agent: dict[str, Any] = dataclasses.field(default_factory=dict)
    # what the scenario is for: name -> predicate over the run's view
    expect: dict[str, Callable[[dict[str, Any]], bool]] = dataclasses.field(default_factory=dict)


async def sc_turns(c: Call) -> None:
    await c.start()
    await c.say("hello")
    await c.say("weather")


async def sc_barge_in(c: Call) -> None:
    await c.start()
    await c.cmd("say", LONG)
    await c.speaking_for(2.0)
    await c.say("interrupt")


async def sc_false_interruption(c: Call) -> None:
    await c.start()
    await c.cmd("say", LONG)
    await c.speaking_for(2.0)
    await c.say("hmm", settle=False)
    await asyncio.sleep(4.0)  # past the false interruption timeout, then the reply resumes
    await c.settle(40)


async def sc_tools(c: Call) -> None:
    await c.start()
    await c.say("weather")
    await c.say("delete")
    await c.say("yes")
    await c.say("book")
    await asyncio.sleep(4.0)  # the background tool's result comes back on its own
    await c.settle()


async def sc_voice_task(c: Call) -> None:
    await c.start()
    await c.say("receipt")
    await c.say("email")


async def sc_handoff(c: Call) -> None:
    await c.start()
    await c.say("billing")
    await c.say("hello")


async def sc_host(c: Call) -> None:
    await c.start()
    await c.cmd("say", HOST_SAY)
    await c.speaking_for(0.2)
    await c.settle()
    await c.cmd("reply", "Tell the person the meeting starts at noon.")
    await c.speaking_for(0.2)
    await c.settle()


async def sc_chat(c: Call) -> None:
    await c.start()
    await c.chat("What's the weather in Tokyo?")
    await c.speaking_for(0.2)
    await c.settle()


async def sc_preconnect(c: Call) -> None:
    await c.start(preconnect=True)
    await c.until(lambda: c.loud_at > 0, 30, "a reply to the pre-connect speech")
    await c.settle()


async def sc_leave(c: Call) -> None:
    await c.start()
    await c.say("hello")
    await c.leave()
    c.note("room_gone", gone=await c.room_gone(20))


async def sc_reconnect(c: Call) -> None:
    await c.start()
    await c.say("hello")
    await c.leave()
    await asyncio.sleep(2.0)
    await c.join()
    await c.until(lambda: c.agent is not None, 10, "the agent after rejoining")
    await asyncio.sleep(1.0)
    await c.say("weather")


async def sc_dtmf(c: Call) -> None:
    await c.start()
    for digit in "1234":
        await c.dtmf(digit)
        await asyncio.sleep(2.0)
    await asyncio.sleep(7.0)  # past user_away_timeout since the last key
    await c.dtmf("5")
    await asyncio.sleep(1.0)


async def sc_noise_cancellation(c: Call) -> None:
    await c.start()
    await c.say("weather")


async def sc_recording(c: Call) -> None:
    await c.start()
    await c.say("hello")


async def sc_realtime(c: Call) -> None:
    await c.start()
    await c.say("hello")
    await c.cmd("reply", "Tell the person the meeting starts at noon.")
    await c.speaking_for(0.2)
    await c.settle()
    await c.say("count", settle=False)
    await c.speaking_for(2.5)
    await c.say("interrupt")


async def sc_avatar(c: Call) -> None:
    await c.start(avatar=True)
    await c.say("hello")
    await c.cmd("say", LONG)
    await c.speaking_for(2.5)
    await c.say("interrupt")
    c.note("av_lag", seconds=c.av_lag())


def _tool_ran(name: str) -> Callable[[dict[str, Any]], bool]:
    return lambda v: name in v["tools_executed"]


def _said(word: str) -> Callable[[dict[str, Any]], bool]:
    return lambda v: any(word in r for r in v["words"]["replies"])


def _cut(v: dict[str, Any]) -> bool:
    return any(i.startswith("assistant (interrupted)") for i in v["history"])


SCENARIOS: dict[str, Scenario] = {
    "turns": Scenario(sc_turns, expect={"get_weather ran": _tool_ran("get_weather")}),
    "barge_in": Scenario(sc_barge_in, expect={"a reply was cut": _cut}),
    "false_interruption": Scenario(
        sc_false_interruption,
        expect={"a false interruption": lambda v: bool(v["false_interruptions"])},
    ),
    "tools": Scenario(
        sc_tools,
        expect={
            "get_weather ran": _tool_ran("get_weather"),
            "delete_account ran after approval": _tool_ran("delete_account"),
            "book_table ran": _tool_ran("book_table"),
            "book_table sent an update": lambda v: bool(v["tool_updates"]),
        },
    ),
    "handoff": Scenario(
        sc_handoff,
        expect={
            "handed off": lambda v: sum(i == "agent_handoff" for i in v["history"]) >= 2,
            "billing greeted": lambda v: any(_norm(BILLING_GREETING) in i for i in v["history"]),
        },
    ),
    "voice_task": Scenario(
        sc_voice_task,
        agent={"voice_task": True},
        expect={
            "the task returned the address to the tool": lambda v: any(
                i.startswith("function_call_output change_receipt_email")
                and "theo example com" in i
                for i in v["history"]
            ),
            "the task's agent took the floor and gave it back": lambda v: (
                sum(i == "agent_handoff" for i in v["history"]) >= 2
            ),
            "the frontend replied to its tool": _tool_ran("change_receipt_email"),
        },
    ),
    "host": Scenario(
        sc_host,
        expect={
            "say() spoke": lambda v: any(_norm(HOST_SAY) in i for i in v["history"]),
            "the reply followed the host's instructions": _said("noon"),
        },
    ),
    "chat": Scenario(
        sc_chat,
        expect={
            "the typed text is a user turn": lambda v: any(
                i.startswith("user") and "tokyo" in i for i in v["history"]
            ),
            "get_weather ran": _tool_ran("get_weather"),
        },
    ),
    "preconnect": Scenario(
        sc_preconnect,
        expect={
            "the pre-connect speech is a user turn": lambda v: any(
                i.startswith("user") and "hear me" in i for i in v["history"]
            )
        },
    ),
    "leave": Scenario(
        sc_leave,
        agent={"close_on_disconnect": True, "delete_room_on_close": True},
        expect={
            "closed on leave": lambda v: v["close"] == "participant_disconnected",
            "room deleted": lambda v: v["caller"]["room_gone"] is True,
        },
    ),
    "reconnect": Scenario(
        sc_reconnect,
        agent={"close_on_disconnect": False},
        expect={"answered after rejoining": _tool_ran("get_weather")},
    ),
    "dtmf": Scenario(
        sc_dtmf,
        agent={"user_away_timeout": 5.0},
        expect={
            "the agent saw the keys": lambda v: v["dtmf"] == list("12345"),
            "away, then back": lambda v: v["user_states"][-2:] == ["away", "listening"],
        },
    ),
    "noise_cancellation": Scenario(
        sc_noise_cancellation,
        agent={"noise_cancellation": True},
        expect={"get_weather ran": _tool_ran("get_weather")},
    ),
    "recording": Scenario(sc_recording, agent={"record": True}),
    "realtime": Scenario(
        sc_realtime,
        agent={"realtime": True},
        # the model follows generate_reply(instructions=) as it likes: not checked here
        expect={"a reply was cut": _cut},
    ),
    "avatar": Scenario(
        sc_avatar,
        agent={"avatar": True},
        expect={
            "a reply was cut": _cut,
            "video within 300 ms of audio": lambda v: (
                v["caller"]["av_lag"] is not None and abs(v["caller"]["av_lag"]) <= 0.3
            ),
        },
    ),
}


# ================================================================================ judge


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def _shape(text: str) -> str:
    """A model's words are not compared, the texts a scenario fixes are."""
    norm = _norm(text)
    for fixed in map(_norm, FIXED):
        if norm != fixed and fixed.startswith(norm) and len(norm) > 10:
            return "<cut>"  # where a reply was cut is timing
        if norm.startswith(fixed[:24]):
            return norm
    return "<words>" if norm else "<empty>"


def _collapse(seq: list[Any]) -> list[Any]:
    out: list[Any] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _text(item: dict[str, Any]) -> str:
    content = item.get("content") or []
    if isinstance(content, str):
        return content
    return " ".join(c for c in content if isinstance(c, str))


def history_view(items: list[dict[str, Any]]) -> list[str]:
    out = []
    for it in items:
        kind = it.get("type")
        if kind == "message":
            role, text = it.get("role"), _text(it)
            if it.get("interrupted"):  # where it was cut is timing; that it was cut is not
                out.append(f"{role} (interrupted): " + ("<cut>" if text else "<empty>"))
            else:
                out.append(f"{role}: " + (_norm(text) if role == "user" else _shape(text)))
        elif kind == "function_call":
            out.append(f"function_call {it.get('name')}")
        elif kind == "function_call_output":
            err = " error" if it.get("is_error") else ""
            out.append(
                f"function_call_output {it.get('name')}{err}: {_shape(str(it.get('output')))}"
            )
        elif kind == "agent_handoff":
            out.append("agent_handoff")
        else:
            out.append(str(kind))
    return out


def _keys(v: Any, prefix: str = "") -> set[str]:
    """A payload's shape: its field paths, ids and times included, values left out."""
    if isinstance(v, dict):
        out = set()
        for k, x in v.items():
            out.add(prefix + k)
            if k not in ("metrics", "data"):
                out |= _keys(x, prefix + k + ".")
        return out
    if isinstance(v, list):
        return {k for x in v for k in _keys(x, prefix + "[].")}
    return set()


def view(stack_dir: dict[str, Any]) -> dict[str, Any]:
    """What one run shows, normalised: ids, times and a model's words left out."""
    lines, caller = stack_dir["agent"], stack_dir["caller"]
    events = [x for x in lines if x["kind"] == "event"]

    def of(name: str) -> list[dict[str, Any]]:
        return [x["data"] for x in events if x["name"] == name]

    snaps = [x for x in lines if x["kind"] == "snapshot"]
    history = snaps[-1]["history"] if snaps else []
    schemas: dict[str, list[str]] = {}
    for x in events:
        schemas.setdefault(x["name"], [])
        schemas[x["name"]] = sorted(set(schemas[x["name"]]) | _keys(x["data"]))
    item_metrics: dict[str, set[str]] = {}
    item_times = set()
    for d in of("conversation_item_added"):
        item = d.get("item") or {}
        if item.get("type") == "message":
            metrics = item.get("metrics") or {}
            item_metrics.setdefault(item.get("role"), set()).update(metrics)
            start = metrics.get("started_speaking_at")
            if isinstance(start, (int, float)) and isinstance(item.get("created_at"), float):
                when = "speech start" if abs(item["created_at"] - start) < 0.05 else "later"
                item_times.add(f"{item.get('role')} created at {when}")
    seen = caller["log"]
    transcripts = [x for x in seen if x["kind"] == "transcript"]
    mics = {x["sid"] for x in seen if x["kind"] == "mic"}
    # the agent's words stream in (a stream per segment, not marked final), the person's are
    # interim streams then a final one; both are published by the agent
    agent_said = [x["text"] for x in transcripts if not _of_mic(x, mics)]
    user_said = [
        x["text"]
        for x in transcripts
        if _of_mic(x, mics) and x["attrs"].get("lk.transcription_final") == "true"
    ]
    caps = next((x for x in lines if x["kind"] == "capabilities"), {})
    v = {
        "started": any(x["kind"] == "started" for x in lines),
        "history": history_view(history),
        "agent_states": _collapse([d.get("new_state") for d in of("agent_state_changed")]),
        "user_states": _collapse([d.get("new_state") for d in of("user_state_changed")]),
        "speeches": [d.get("source") for d in of("speech_created")],
        "tools_executed": [
            c.get("name")
            for d in of("function_tools_executed")
            for c in d.get("function_calls", [])
        ],
        "tool_updates": [(d.get("update") or {}).get("type") for d in of("tool_execution_updated")],
        "false_interruptions": [d.get("resumed") for d in of("agent_false_interruption")],
        "errors": [
            f"{(d.get('error') or {}).get('__type')}:{(d.get('error') or {}).get('recoverable')}"
            for d in of("error")
        ],
        "close": (of("close") or [{}])[-1].get("reason"),
        "events_seen": sorted({x["name"] for x in events}),
        "event_types": sorted({f"{x['name']}:{x['type']}" for x in events}),
        "event_schemas": schemas,
        "item_metrics": {k: sorted(s) for k, s in item_metrics.items() if k},
        "item_times": sorted(item_times),
        "start_params": caps.get("start_params"),
        "dtmf": [x["digit"] for x in lines if x["kind"] == "room" and x["name"] == "dtmf"],
        # what a final transcript event carries: which fields are set, not their values
        "final_transcripts": sorted(
            {
                ",".join(k for k in ("item_id", "language", "speaker_id") if d.get(k))
                or "no item_id, language or speaker_id"
                for d in of("user_input_transcribed")
                if d.get("is_final")
            }
        ),
        "caller": {
            "agent_states": _collapse([x["state"] for x in seen if x["kind"] == "agent_state"]),
            "agent_said": [_shape(x) for x in agent_said],
            "user_said": [_norm(x) for x in user_said],
            "transcript_attrs": sorted({k for x in transcripts for k in x["attrs"]}),
            "tracks": sorted({f"{x['who']}:{x['media']}" for x in seen if x["kind"] == "track"}),
            "heard_agent": bool(caller["heard"]),
            "room_gone": next((x["gone"] for x in seen if x["kind"] == "room_gone"), None),
            "av_lag": next((x["seconds"] for x in seen if x["kind"] == "av_lag"), None),
            "timeouts": [x["what"] for x in seen if x["kind"] == "timeout"],
            "rpc_errors": [x["op"] for x in seen if x["kind"] == "rpc_error"],
        },
        # for the checks on one stack, not compared: the words themselves
        "words": {
            "agent_said": [_norm(x) for x in agent_said],
            "replies": [
                _norm(_text(it))
                for it in history
                if it.get("type") == "message" and it.get("role") == "assistant"
            ],
            "cut": [
                _norm(_text(it))
                for it in history
                if it.get("type") == "message" and it.get("interrupted")
            ],
        },
    }
    return v


def _of_mic(x: dict[str, Any], mics: set[str]) -> bool:
    """The agent publishes the person's words too, naming the person's microphone track."""
    return x["attrs"].get("lk.transcribed_track_id") in mics


def _get(v: dict[str, Any], path: str) -> Any:
    for part in path.split("."):
        v = v.get(part) if isinstance(v, dict) else None  # type: ignore[assignment]
    return v


def consistency(v: dict[str, Any]) -> list[str]:
    """What must hold on one stack alone."""
    problems = []
    words = v["words"]
    said = " | ".join(words["agent_said"])
    for text in words["replies"]:
        if text and text[:40] not in said:
            problems.append(f"the caller never saw the transcript of '{text[:40]}'")
    for text in words["cut"]:
        if text and not any(_norm(f).startswith(text) for f in FIXED) and text[:40] not in said:
            problems.append(f"a cut reply that is no prefix of what was said: '{text[:40]}'")
    for text in words["agent_said"]:
        kept = [r for r in words["replies"] if r]
        if text and not any(text.startswith(r[:30]) or r.startswith(text[:30]) for r in kept):
            problems.append(f"the caller saw '{text[:40]}', which the history does not have")
    if v["errors"]:
        problems.append(f"error events: {v['errors']}")
    if v["caller"]["timeouts"]:
        problems.append(f"timed out waiting for: {v['caller']['timeouts']}")
    if v["caller"]["rpc_errors"]:
        problems.append(f"host commands failed: {v['caller']['rpc_errors']}")
    if not v["started"]:
        problems.append("session.start() never returned")
    return problems


# ======================================================================= latency per stage

STAGES = {
    "end_of_turn": ("user", "end_of_turn_delay"),
    "transcription_delay": ("user", "transcription_delay"),
    "llm_ttft": ("assistant", "llm_node_ttft"),
    "tts_ttfb": ("assistant", "tts_node_ttfb"),
    "first_audio_out": ("assistant", "e2e_latency"),
}


def latencies(runs: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {s: [] for s in STAGES}
    out["caller_heard"] = []
    for run in runs.values():
        for x in run["agent"]:
            if x["kind"] != "event" or x["name"] != "conversation_item_added":
                continue
            item = x["data"].get("item") or {}
            metrics = item.get("metrics") or {}
            for stage, (role, key) in STAGES.items():
                if item.get("role") == role and isinstance(metrics.get(key), (int, float)):
                    out[stage].append(float(metrics[key]))
        out["caller_heard"].extend(run["caller"]["heard"])
    return out


# ============================================================================== the runner


def load_env(path: str | None) -> None:
    if not path:
        return
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("'\"")  # the last one wins, as sourced


class Worker:
    def __init__(self, stack: str, name: str, log: Path) -> None:
        self.stack, self.name, self.log = stack, name, log
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "agent", "--name", name],
            stdout=open(log, "w"),  # noqa: SIM115
            stderr=subprocess.STDOUT,
            cwd=str(REPO),
            env=env,
        )

    async def registered(self, timeout: float = 90) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if "registered worker" in self.log.read_text(errors="replace"):
                return
            if self.proc.poll() is not None:
                break
            await asyncio.sleep(0.5)
        tail = "\n".join(self.log.read_text(errors="replace").splitlines()[-30:])
        raise SystemExit(f"the {self.stack} worker did not register:\n{tail}")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


async def run_one(
    name: str, stack: str, agent: str, lkapi: Any, speech: Any, runs: Path
) -> dict[str, Any]:
    out = runs / f"{name}.{stack}.jsonl"
    out.unlink(missing_ok=True)
    call = Call(stack, name, agent, lkapi, speech, out)
    began = time.time()
    try:
        await asyncio.wait_for(SCENARIOS[name].drive(call), 150)
    except Exception as e:
        call.note("timeout", what=f"the script: {type(e).__name__}: {e}")
    finally:
        await call.close()
    # the agent's close and last snapshot are written as the room goes
    deadline = time.time() + 15
    while time.time() < deadline:
        text = out.read_text() if out.exists() else ""
        if '"name": "close"' in text:
            break
        await asyncio.sleep(0.5)
    lines = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    return {
        "agent": lines,
        "caller": {"log": call.log, "heard": call.heard},
        "seconds": round(time.time() - began, 1),
        "room": call.room_name,
    }


async def drive(args: argparse.Namespace) -> int:
    from livekit import api

    runs = Path(args.keep or tempfile.mkdtemp(prefix="lk-e2e-"))
    runs.mkdir(parents=True, exist_ok=True)
    speech = await speech_frames(Path(tempfile.gettempdir()) / "lk-e2e-speech")
    worker = Worker(STACK, f"e2e-{STACK}-{uuid.uuid4().hex[:6]}", runs / f"{STACK}.log")
    lkapi = api.LiveKitAPI()
    sem = asyncio.Semaphore(args.jobs)
    results: dict[str, dict[str, Any]] = {}

    async def scenario(name: str, attempt: int) -> None:
        async with sem:
            print(f"  {name} (attempt {attempt}) ...", flush=True)
            run = await run_one(name, STACK, worker.name, lkapi, speech, runs)
            results[name] = {STACK: run}

    try:
        await worker.registered()
        print(f"worker registered; runs in {runs}", flush=True)
        await asyncio.gather(*(scenario(n, 1) for n in args.scenarios))
        again = [] if args.no_confirm else [n for n in args.scenarios if _problems(n, results[n])]
        if again:
            print(f"running again {again}", flush=True)
            for n in again:
                results[n + "#1"] = results[n]
            await asyncio.gather(*(scenario(n, 2) for n in again))
    finally:
        await lkapi.aclose()
        worker.stop()

    (runs / "results.json").write_text(json.dumps(results, indent=1, default=repr))
    return report(args.scenarios, results, runs)


def _problems(name: str, run: dict[str, Any]) -> list[str]:
    """What the run's checks and its consistency say is wrong with it."""
    v = view(run[STACK])
    failed = [f"did not: {k}" for k, f in SCENARIOS[name].expect.items() if not _safe(f, v)]
    return failed + consistency(v)


def _safe(f: Callable[[dict[str, Any]], bool], v: dict[str, Any]) -> bool:
    try:
        return bool(f(v))
    except Exception:
        return False


def report(names: list[str], results: dict[str, Any], runs: Path) -> int:
    """Judge saved runs and print the report: 0 when every scenario passed every attempt."""
    if not names:
        print("no scenarios to judge", file=sys.stderr)
        return 1
    failed = []
    print("\n" + "=" * 100)
    for n in names:
        attempts = [results[k] for k in (n + "#1", n) if k in results]
        problems = [_problems(n, run) for run in attempts]
        bad = any(problems)
        print(f"{n}: {'FAILED' if bad else 'passed'}")
        for i, found in enumerate(problems, 1):
            for p in found:
                print(f"    (attempt {i}) {p}")
        lag = view(results[n][STACK])["caller"]["av_lag"]
        if lag is not None:
            print(f"    video behind audio (s): {lag}")
        if bad:
            failed.append(n)

    print("\nlatency per stage (s), every reply of every scenario")
    lat = latencies({n: results[n][STACK] for n in names})
    print(f"{'stage':<20}{'p50':>10}{'p90':>10}{'n':>6}")
    for stage, values in lat.items():
        if values:
            print(f"{stage:<20}{_p50(values):>10.3f}{_p90(values):>10.3f}{len(values):>6}")
        else:
            print(f"{stage:<20}{'-':>10}{'-':>10}{0:>6}")

    errs = log_errors(runs / f"{STACK}.log")
    print(f"\nworker: {len(errs)} error lines ({runs / (STACK + '.log')})")
    for line in _dedupe(errs)[:25]:
        print(f"    {line}")
    print(f"\n{len(names) - len(failed)}/{len(names)} scenarios passed; runs kept in {runs}")
    return 1 if failed else 0


def log_errors(log: Path) -> list[str]:
    """A worker's warning and error lines, each with its traceback's last line."""
    out = []
    for line in log.read_text(errors="replace").splitlines() if log.exists() else []:
        try:
            rec = json.loads(line)
        except ValueError:
            if re.search(r"Error|Warning", line):
                out.append(line.strip()[:240])
            continue
        if rec.get("level") in ("WARNING", "ERROR", "CRITICAL"):
            exc = (rec.get("exc_info") or "").strip().splitlines()
            tail = f" | {exc[-1]}" if exc else ""
            out.append(f"{rec['level']} {rec.get('message')}{tail}"[:300])
    return out


def _p50(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]


def _dedupe(lines: list[str]) -> list[str]:
    seen, out = set(), []
    for line in lines:
        key = re.sub(r"[0-9a-f]{6,}|\d+(\.\d+)?", "#", line)[:120]
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        p = argparse.ArgumentParser()
        p.add_argument("role")
        p.add_argument("--name", required=True)
        run_agent(p.parse_args().name)
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("scenarios", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--jobs", type=int, default=4, help="scenarios at once (two rooms each)")
    p.add_argument("--env-file", help="KEY=value lines to load (LiveKit credentials)")
    p.add_argument("--keep", help="write the runs here instead of a temp dir")
    p.add_argument("--no-confirm", action="store_true", help="do not rerun a failing scenario")
    p.add_argument("--report", metavar="RUNS", help="judge the runs kept in RUNS again")
    args = p.parse_args()
    if args.list:
        print("\n".join(SCENARIOS))
        return
    if args.report:
        results = json.loads((Path(args.report) / "results.json").read_text())
        names = args.scenarios or [n for n in SCENARIOS if n in results]
        sys.exit(report(names, results, Path(args.report)))
    args.scenarios = args.scenarios or list(SCENARIOS)
    if unknown := [n for n in args.scenarios if n not in SCENARIOS]:
        p.error(f"unknown scenarios: {unknown}")
    load_env(args.env_file)
    missing = [
        k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.environ.get(k)
    ]
    if missing:
        sys.exit(f"needs {', '.join(missing)} (or --env-file)")
    sys.exit(asyncio.run(drive(args)))


if __name__ == "__main__":
    main()
