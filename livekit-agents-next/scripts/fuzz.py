"""Property-based runs of the call path through the real Python ``VoiceSession``: each seed draws a
call (pipeline or realtime providers, their delays from the DST's calibration, room options,
toolsets, a job around it or not) and a sequence of what a person and a developer do, runs it on
the reference tests' virtual-time loop over ``_core.FakeRoom``, and checks what must hold whatever
was drawn. The DST (``rust/crates/lk-agents-fuzz``) searches the core below the Python glue; this
searches the glue: wiring, job context, exceptions, and latency the glue adds.

What every seed checks:

- nothing raises and nothing is logged with a traceback (a drawn toolset that fails its setup
  logs one, as the reference does, and is the only one allowed), and no error event is emitted:
  nothing is made to fail, so one is a failure the session made up (a plugin stream ending in
  the close's cancellation reported as a provider error);
- every speech ``say()`` and ``generate_reply()`` returned is done once the call settles, and the
  session closes once, within a bound, leaving no task running;
- a reply asked of an idle agent is heard within its providers' drawn delays and a budget
  (``BUDGET_S``): the LLM's first chunk and the voice's first audio, or the realtime server's;
- a realtime session is asked for a reply only once the updates sent before it applied, and the
  server's turn detection is kept unless a detector was named;
- in a job: the entrypoint, the session end callback and the shutdown callbacks see the job's
  context, the session started on the job's room is the job's primary session and connected the
  room once, and the job tells its parent it is shutting down and exiting.

    python livekit-agents-next/scripts/fuzz.py              # seeds 0..63
    python livekit-agents-next/scripts/fuzz.py --seeds 500 --first 1000
    python livekit-agents-next/scripts/fuzz.py 17           # one seed, with its call printed

It needs the venv ``parity.py`` builds (``livekit-agents-next`` and pytest). Exit status 1 names
each failing seed and what failed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import gc
import json
import logging
import random
import socket
import struct
import sys
import time
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any
from unittest import mock

from livekit.agents import RunContext

REPO = Path(__file__).resolve().parents[2]
CALIBRATION = REPO / "rust/crates/lk-agents-fuzz/corpus/calibration.json"
SAMPLE_RATE = 24000
# What a reply may take beyond its providers' drawn delays before its first audio.
BUDGET_S = 0.4
# A realtime server streams a reply this much faster than it plays.
RT_FRAME_S = 0.1
RT_FRAME_GAP_S = 0.05
# Log lines a drawn setup is expected to produce (the reference logs them too).
EXPECTED_LOGS = ("failed to set up a toolset",)


class Failed(AssertionError):
    pass


def check(ok: object, what: str) -> None:
    if not ok:
        raise Failed(what)


class Draw:
    """The seed's randomness, and the calibrated delays drawn from it (seconds)."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        cal = json.loads(CALIBRATION.read_text())
        self.llm_first = cal["llm"]["openai/gpt-4.1-mini"]["first_chunk"]
        self.tts_first = cal["tts"]["cartesia/sonic-3"]["first_audio"]
        rt = cal["realtime"]
        self.rt_created = rt["created_after_create"]
        self.rt_item = rt["item_after_created"]
        self.rt_audio = rt["first_audio_after_item"]

    def ms(self, samples: list[float]) -> float:
        return float(self.rng.choice(samples)) / 1000.0

    def chance(self, p: float) -> bool:
        return self.rng.random() < p


class Call:
    """What one seed saw: the session's events on the virtual clock, and what the fakes did."""

    def __init__(self, draw: Draw, verbose: bool) -> None:
        self.draw = draw
        self.verbose = verbose
        self.closes = 0
        self.handles: list[tuple[str, Any]] = []
        # the providers' drawn delays (seconds): the LLM's and the realtime server's by the user
        # message they answered, the voice's by when each synthesis opened
        self.llm_delays: dict[str, float] = {}
        self.tts_delays: list[tuple[float, float]] = []
        self.rt_delays: dict[str, float] = {}
        self.violations: list[str] = []
        self.room: Any = None
        self.session: Any = None

    def log(self, *what: object) -> None:
        if self.verbose:
            print(f"  {asyncio.get_running_loop().time():8.3f}", *what)

    async def settle(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / 0.02))):
            await asyncio.sleep(0.02)
            if self.room is not None:
                self.room.advance(0.02)
            for _ in range(3):
                await asyncio.sleep(0)


# ============================================================================ providers


def providers(call: Call) -> Any:
    from livekit import rtc
    from livekit.agents import (
        VAD,
        APIConnectOptions,
        VADCapabilities,
        VADEvent,
        VADEventType,
        VADStream,
        llm,
        stt,
        tts,
    )
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

    draw = call.draw

    class Ns:
        pass

    ns = Ns()

    def last_text(chat_ctx: Any) -> str:
        last = chat_ctx.items[-1] if chat_ctx.items else None
        return (last.text_content or "") if last is not None and last.type == "message" else ""

    def reply_to(chat_ctx: Any, tools: list[Any]) -> tuple[str | None, str]:
        """(a tool to call, else None; the words): the tool a message asks for, a tool's output
        read back, else an echo."""
        names = {getattr(getattr(t, "info", None), "name", None) for t in tools}
        last = chat_ctx.items[-1] if chat_ctx.items else None
        if last is not None and last.type == "function_call_output":
            return None, f"The tool said {last.output}. Anything else?"
        text = (last.text_content or "") if last is not None and last.type == "message" else ""
        for word, tool in (("weather", "get_weather"), ("billing", "to_billing")):
            if word in text and tool in names:
                return tool, ""
        return None, f"You said {text or 'nothing'}. Tell me more about it, please."

    class FuzzLLM(llm.LLM):
        @property
        def model(self) -> str:
            return "fuzz-llm"

        def chat(
            self, *, chat_ctx: llm.ChatContext, tools: list[Any] | None = None, **_: object
        ) -> llm.LLMStream:
            first = draw.ms(draw.llm_first)
            call.llm_delays[last_text(chat_ctx)] = first
            return _LLMStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools or [],
                conn_options=DEFAULT_API_CONNECT_OPTIONS,
                first=first,
            )

    class _LLMStream(llm.LLMStream):
        def __init__(self, *args: Any, first: float, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._first = first

        async def _run(self) -> None:
            tool, words = reply_to(self._chat_ctx, self._tools)
            await asyncio.sleep(self._first)
            if tool is not None:
                arguments = json.dumps({"city": "Paris"} if tool == "get_weather" else {})
                c = llm.FunctionToolCall(name=tool, arguments=arguments, call_id=f"call_{id(self)}")
                self.send(
                    llm.ChatChunk(id="c", delta=llm.ChoiceDelta(role="assistant", tool_calls=[c]))
                )
                return
            for i, word in enumerate(words.split(" ")):
                if i:
                    await asyncio.sleep(0.01)
                delta = llm.ChoiceDelta(role="assistant", content=(" " if i else "") + word)
                self.send(llm.ChatChunk(id="c", delta=delta))

    class StreamingSTT(stt.STT):
        """Hears "<utterance>" in each burst of half a second or more."""

        # a plugin whose reader task the close cancels raises that cancellation from its stream
        cancels_at_close = False

        def __init__(self, name: str = "fuzz-stt") -> None:
            super().__init__(
                capabilities=stt.STTCapabilities(streaming=True, interim_results=False)
            )
            self._name = name

        @property
        def model(self) -> str:
            return self._name

        async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
            raise NotImplementedError

        def stream(self, **_: object) -> stt.RecognizeStream:
            return _Recognize(stt=self, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    class _Recognize(stt.RecognizeStream):
        async def __anext__(self) -> stt.SpeechEvent:
            try:
                return await super().__anext__()
            except StopAsyncIteration:
                if getattr(self._stt, "cancels_at_close", False):
                    raise asyncio.CancelledError() from None
                raise

        async def _run(self) -> None:
            heard = 0.0
            async for frame in self.input:
                if not isinstance(frame, rtc.AudioFrame):
                    continue
                if heard == 0.0:
                    self._emit(stt.SpeechEventType.START_OF_SPEECH)
                heard += frame.duration
                if heard >= 0.5:
                    self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, "tell me a story")
                    self._emit(stt.SpeechEventType.END_OF_SPEECH)
                    heard = -1e9  # one utterance per burst: the rest of it is silence

        def _emit(self, kind: Any, text: str | None = None) -> None:
            alts = [stt.SpeechData(language="en", text=text, confidence=1.0)] if text else []
            self.send(stt.SpeechEvent(type=kind, alternatives=alts))

    class BatchSTT(stt.STT):
        """Recognize-only: it hears "tell me a story" in any utterance the VAD ends."""

        def __init__(self) -> None:
            super().__init__(
                capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
            )

        @property
        def model(self) -> str:
            return "fuzz-batch-stt"

        async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
            said = stt.SpeechData(language="en", text="tell me a story", confidence=1.0)
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[said])

    class FuzzVAD(VAD):
        """Speech from the first frame of a burst until half a second of it."""

        def __init__(self) -> None:
            super().__init__(capabilities=VADCapabilities(update_interval=0.02))

        @property
        def model(self) -> str:
            return "fuzz-vad"

        def stream(self) -> VADStream:
            return _Detect(self)

    class _Detect(VADStream):
        async def _main_task(self) -> None:
            speech: list[rtc.AudioFrame] = []
            async for frame in self.input:
                if not isinstance(frame, rtc.AudioFrame) or speech is None:
                    continue
                if not speech:
                    self._emit(VADEventType.START_OF_SPEECH, [], True)
                speech.append(frame)
                if sum(f.duration for f in speech) >= 0.5:
                    self._emit(VADEventType.END_OF_SPEECH, list(speech), False)
                    speech = None  # type: ignore[assignment]

        def _emit(self, kind: Any, frames: list[Any], speaking: bool) -> None:
            self.send(
                VADEvent(
                    type=kind,
                    samples_index=0,
                    timestamp=0.0,
                    speech_duration=0.0,
                    silence_duration=0.0,
                    frames=frames,
                    probability=1.0 if speaking else 0.0,
                    speaking=speaking,
                )
            )

    def pcm(text: str) -> bytes:
        # 50 ms of silence per character, as a voice reads
        return b"\x00\x00" * (SAMPLE_RATE // 20 * max(1, len(text)))

    class ChunkedTTS(tts.TTS):
        """A non-streaming voice: each sentence's audio after the voice's drawn first-audio delay."""

        def __init__(self) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            )

        @property
        def model(self) -> str:
            return "fuzz-chunked-tts"

        def synthesize(
            self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
        ) -> tts.ChunkedStream:
            first = draw.ms(draw.tts_first)
            call.tts_delays.append((asyncio.get_running_loop().time(), first))
            return _Chunked(tts=self, input_text=text, conn_options=conn_options, first=first)

    class _Chunked(tts.ChunkedStream):
        def __init__(self, *args: Any, first: float, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._first = first

        async def _run(self, output_emitter: tts.AudioEmitter) -> None:
            output_emitter.initialize(
                request_id="req", sample_rate=SAMPLE_RATE, num_channels=1, mime_type="audio/pcm"
            )
            await asyncio.sleep(self._first)
            output_emitter.push(pcm(self.input_text))
            output_emitter.flush()

    class StreamingTTS(tts.TTS):
        """A streaming voice: each segment's audio once flushed, after the drawn first-audio delay."""

        def __init__(self) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=True),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            )

        @property
        def model(self) -> str:
            return "fuzz-streaming-tts"

        def synthesize(
            self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
        ) -> tts.ChunkedStream:
            first = draw.ms(draw.tts_first)
            call.tts_delays.append((asyncio.get_running_loop().time(), first))
            return _Chunked(tts=self, input_text=text, conn_options=conn_options, first=first)

        def stream(
            self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
        ) -> tts.SynthesizeStream:
            first = draw.ms(draw.tts_first)
            call.tts_delays.append((asyncio.get_running_loop().time(), first))
            return _Streamed(tts=self, conn_options=conn_options, first=first)

    class _Streamed(tts.SynthesizeStream):
        def __init__(self, *args: Any, first: float, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._first = first

        async def _run(self, output_emitter: tts.AudioEmitter) -> None:
            output_emitter.initialize(
                request_id="req",
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
                stream=True,
            )
            text, segment, first = "", 0, True
            async for data in self.input:
                if isinstance(data, self.Flush):
                    if not text:
                        continue
                    output_emitter.start_segment(segment_id=f"seg_{segment}")
                    if first:
                        await asyncio.sleep(self._first)
                        first = False
                    output_emitter.push(pcm(text))
                    output_emitter.end_segment()
                    text, segment = "", segment + 1
                else:
                    text += data
            output_emitter.end_input()

    class FuzzRealtime(llm.RealtimeModel):
        """A realtime server: a reply's audio streamed faster than it plays, after the corpus's
        delays; a reply to a message asking for the weather calls the tool. A reply asked while an
        update sent before it has not applied is a violation (the server would not have read it)."""

        def __init__(self, server_turns: bool) -> None:
            super().__init__(
                capabilities=llm.RealtimeCapabilities(
                    message_truncation=False,
                    turn_detection=server_turns,
                    auto_turn_reply_generation=server_turns,
                    can_disable_turn_detection=server_turns,
                    user_transcription=False,
                    auto_tool_reply_generation=False,
                    audio_output=True,
                    manual_function_calls=True,
                )
            )

        def session(self, *, turn_detection_disabled: bool = False) -> llm.RealtimeSession:
            # the server's own turn detection stays on unless the developer named a detector
            if self.capabilities.turn_detection and turn_detection_disabled:
                call.violations.append("the realtime server's turn detection was turned off")
            return _RealtimeSession(self)

    class _RealtimeSession(llm.RealtimeSession):
        def __init__(self, model: FuzzRealtime) -> None:
            super().__init__(model)
            self._chat_ctx = llm.ChatContext.empty()
            self._tools: list[Any] = []
            self._updating = 0
            self._n = 0

        @property
        def chat_ctx(self) -> llm.ChatContext:
            return self._chat_ctx

        @property
        def tools(self) -> llm.ToolContext:
            return llm.ToolContext.empty()

        async def _apply(self) -> None:
            # a server takes a moment to apply an update
            self._updating += 1
            try:
                await asyncio.sleep(0.03)
            finally:
                self._updating -= 1

        async def update_instructions(self, instructions: str) -> None:
            await self._apply()

        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            self._updating += 1
            try:
                await asyncio.sleep(0.03)
                self._chat_ctx = chat_ctx.copy()
            finally:
                self._updating -= 1

        async def update_tools(self, tools: list[llm.Tool]) -> None:
            self._tools = list(tools)
            await self._apply()

        def update_options(self, **_: object) -> None:
            pass

        def push_audio(self, frame: rtc.AudioFrame) -> None:
            pass

        def push_video(self, frame: rtc.VideoFrame) -> None:
            pass

        def generate_reply(self, **_: object) -> asyncio.Future[llm.GenerationCreatedEvent]:
            if self._updating:
                call.violations.append(
                    "a realtime reply was asked before an earlier update applied"
                )
            loop = asyncio.get_running_loop()
            self._n += 1
            n = self._n
            tool, words = reply_to(self._chat_ctx, self._tools)
            created = draw.ms(draw.rt_created)
            first = draw.ms(draw.rt_item) + draw.ms(draw.rt_audio)
            call.rt_delays[last_text(self._chat_ctx)] = created + first
            frames: asyncio.Queue[Any] = asyncio.Queue()
            ended = asyncio.Event()

            async def server() -> None:
                # generated on the server's clock, read or not
                await asyncio.sleep(first)
                if tool is None:
                    for _ in range(max(1, round(len(words) * 0.05 / RT_FRAME_S))):
                        frames.put_nowait(
                            rtc.AudioFrame(
                                b"\x00\x00" * int(SAMPLE_RATE * RT_FRAME_S),
                                SAMPLE_RATE,
                                1,
                                int(SAMPLE_RATE * RT_FRAME_S),
                            )
                        )
                        await asyncio.sleep(RT_FRAME_GAP_S)
                frames.put_nowait(None)
                ended.set()

            async def audio() -> AsyncIterable[rtc.AudioFrame]:
                while (frame := await frames.get()) is not None:
                    yield frame

            async def text() -> AsyncIterable[str]:
                yield words

            async def messages() -> AsyncIterable[llm.MessageGeneration]:
                if tool is None:
                    modalities: asyncio.Future[list[str]] = loop.create_future()
                    modalities.set_result(["audio", "text"])
                    yield llm.MessageGeneration(
                        message_id=f"msg_{n}",
                        text_stream=text(),
                        audio_stream=audio(),
                        modalities=modalities,
                    )
                await ended.wait()

            async def functions() -> AsyncIterable[Any]:
                await ended.wait()
                if tool is not None:
                    arguments = json.dumps({"city": "Paris"} if tool == "get_weather" else {})
                    yield llm.FunctionCall(name=tool, arguments=arguments, call_id=f"rtcall_{n}")

            fut: asyncio.Future[llm.GenerationCreatedEvent] = loop.create_future()

            def create() -> None:
                task = asyncio.ensure_future(server())
                call.background.append(task)
                fut.set_result(
                    llm.GenerationCreatedEvent(
                        message_stream=messages(),
                        function_stream=functions(),
                        user_initiated=True,
                        response_id=f"resp_{n}",
                    )
                )

            loop.call_later(created, create)
            return fut

        def commit_audio(self) -> None:
            pass

        def clear_audio(self) -> None:
            pass

        def interrupt(self) -> None:
            pass

        def truncate(self, **_: object) -> None:
            pass

        async def aclose(self) -> None:
            pass

    class FailingToolset(llm.Toolset):
        """A toolset whose back office cannot be reached."""

        async def setup(self) -> FailingToolset:
            raise ConnectionError("the back office is unreachable")

    ns.FuzzLLM = FuzzLLM
    ns.StreamingSTT = StreamingSTT
    ns.BatchSTT = BatchSTT
    ns.FuzzVAD = FuzzVAD
    ns.ChunkedTTS = ChunkedTTS
    ns.StreamingTTS = StreamingTTS
    ns.FuzzRealtime = FuzzRealtime
    ns.FailingToolset = FailingToolset
    ns.fallback_stt = lambda: stt.FallbackAdapter([StreamingSTT("primary"), StreamingSTT("backup")])
    return ns


# ============================================================================ the call


def agents(call: Call) -> tuple[Any, Any]:
    from livekit.agents import Agent, function_tool

    class Billing(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You handle billing.")

        async def on_enter(self) -> None:
            call.handles.append(("billing on_enter", self.session.say("Billing here.")))

    class FrontDesk(Agent):
        def __init__(self, tools: list[Any]) -> None:
            super().__init__(instructions="You are the front desk.", tools=tools)

        async def on_enter(self) -> None:
            call.handles.append(("on_enter", self.session.say("Hello, front desk.")))

        @function_tool
        async def get_weather(self, ctx: RunContext, city: str) -> str:
            """The weather in a city."""
            return f"sunny in {city}"

        @function_tool
        async def to_billing(self, ctx: RunContext) -> None:
            """Transfer the caller to billing."""
            ctx.session.update_agent(Billing())

    return FrontDesk, Billing


def build_session(call: Call, p: Any) -> tuple[Any, list[Any], str]:
    """The drawn session, the front desk's tools, and a description of the draw."""
    from livekit.agents import VoiceSession

    d = call.draw
    realtime = d.chance(0.3)
    kwargs: dict[str, Any] = {"aec_warmup_duration": None, "user_away_timeout": None}
    desc: list[str] = []
    if realtime:
        server_turns = d.chance(0.5)
        kwargs.update(llm=p.FuzzRealtime(server_turns))
        default_vad = d.chance(0.5)
        if not default_vad:
            kwargs["vad"] = None
        desc.append(f"realtime server_turns={server_turns} default_vad={default_vad}")
    else:
        stt_kind = d.rng.choice(["streaming", "batch", "fallback"])
        vad = p.FuzzVAD() if stt_kind == "batch" or d.chance(0.4) else None
        stt = {
            "streaming": p.StreamingSTT,
            "batch": p.BatchSTT,
            "fallback": p.fallback_stt,
        }[stt_kind]()
        if stt_kind == "streaming" and d.chance(0.5):
            stt.cancels_at_close = True
            stt_kind += "(cancelled at close)"
        voice = d.rng.choice(["chunked", "streaming"])
        tts = p.ChunkedTTS() if voice == "chunked" else p.StreamingTTS()
        kwargs.update(stt=stt, llm=p.FuzzLLM(), tts=tts, vad=vad)
        detection = "vad" if vad is not None else "stt"
        kwargs["turn_handling"] = {"turn_detection": detection}
        desc.append(f"pipeline stt={stt_kind} vad={vad is not None} tts={voice}")
    tools: list[Any] = []
    if d.chance(0.25):
        tools.append(p.FailingToolset(id="back_office"))
        desc.append("failing-toolset")
    return VoiceSession(**kwargs), tools, " ".join(desc)


def room_options(call: Call) -> Any:
    from livekit.agents import room_io

    d = call.draw
    if d.chance(0.5):
        return None
    return room_io.RoomOptions(
        close_on_disconnect=d.chance(0.5),
        text_input=d.chance(0.5),
        participant_identity="user",
        delete_room_on_close=False,
    )


async def measured_reply(call: Call, realtime: bool) -> None:
    """A reply asked of an idle agent is heard within its providers' delays and the budget."""
    session = call.session
    quiet = 0
    for _ in range(600):
        idle = session.agent_state == "listening" and session.user_state == "listening"
        quiet = quiet + 1 if idle and all(h.done() for _, h in call.handles) else 0
        if quiet >= 20:  # two seconds with nothing going on
            break
        await call.settle(0.1)
    else:
        return  # never idle: the other checks say why
    loop = asyncio.get_running_loop()
    asked = f"say something {len(call.handles)}"
    start = loop.time()
    handle = session.generate_reply(user_input=asked)
    call.handles.append(("measured", handle))
    heard_at = None
    for _ in range(200):
        await call.settle(0.02)
        if session.agent_state == "speaking":
            heard_at = loop.time()
            break
    if realtime:
        owed = call.rt_delays.get(asked, 0.0)
    else:
        voice = next((d for at, d in call.tts_delays if at >= start), 0.0)
        owed = call.llm_delays.get(asked, 0.0) + voice
    check(heard_at is not None, f"a reply asked at {start:.2f}s was not heard within 4 s")
    # the server answers what it was sent before the reply: the user message is in its context
    check(
        not realtime or asked in call.rt_delays,
        "the realtime server was asked for a reply before the user message sent ahead of it applied",
    )
    took = heard_at - start  # type: ignore[operator]
    call.log(f"measured reply: {took:.3f}s, providers owed {owed:.3f}s")
    check(
        took <= owed + BUDGET_S,
        f"a reply took {took:.2f}s to be heard, its providers {owed:.2f}s (budget {BUDGET_S}s)",
    )


async def room_call(call: Call) -> None:
    try:
        await room_call_body(call)
    finally:
        if call.session is not None and call.closes == 0:
            task = asyncio.ensure_future(call.session.aclose())
            for _ in range(500):
                if task.done():
                    break
                await call.settle(0.02)
            if task.done():
                task.exception()


async def room_call_body(call: Call) -> None:
    from livekit.agents import _core

    p = providers(call)
    FrontDesk, Billing = agents(call)
    session, tools, desc = build_session(call, p)
    call.session = session
    realtime = desc.startswith("realtime")
    options = room_options(call)
    call.log("call:", desc, "room_options" if options is not None else "")
    session.on("close", lambda ev: setattr(call, "closes", call.closes + 1))
    # nothing is made to fail: an error event is a provider failure the session made up
    session.on("error", lambda ev: call.violations.append(f"an error event: {ev.error}"))
    call.room = _core.FakeRoom(sample_rate=48000)
    call.room.join("user")
    kwargs = {"room_options": options} if options is not None else {}
    await session.start(FrontDesk(tools), room=call.room, **kwargs)
    await call.settle(0.5)
    d = call.draw
    for _ in range(d.rng.randint(3, 10)):
        op = d.rng.choice(
            ["speak", "say", "say", "reply", "reply", "interrupt", "handoff", "wait", "measure"]
        )
        call.log("op", op)
        if op == "speak":
            call.room.speak(frames=d.rng.randint(25, 60))
        elif op == "say":
            call.handles.append(
                ("say", session.say("One moment, please.", allow_interruptions=d.chance(0.8)))
            )
        elif op == "reply":
            text = d.rng.choice(["what's the weather?", "hello there", "billing, please"])
            call.handles.append(("reply", session.generate_reply(user_input=text)))
        elif op == "interrupt":
            # refused (a failed future) while the speech playing does not allow interruptions
            session.interrupt().add_done_callback(lambda f: f.cancelled() or f.exception())
        elif op == "handoff":
            session.update_agent(Billing())
        elif op == "measure":
            await measured_reply(call, realtime)
        await call.settle(d.rng.uniform(0.1, 3.0))
    await call.settle(10)
    for label, handle in call.handles:
        check(handle.done(), f"the {label} speech never finished")
    task = asyncio.ensure_future(session.aclose())
    for _ in range(500):
        if task.done():
            break
        await call.settle(0.02)
    check(task.done(), "aclose() did not return within 10 s")
    await task
    check(call.closes == 1, f"the session closed {call.closes} times")
    await call.settle(1)
    me = asyncio.current_task()
    left = [t for t in asyncio.all_tasks() if t is not me and t not in call.background]
    check(
        not left,
        f"tasks still running a second after aclose(): {sorted(t.get_name() for t in left)}",
    )


async def job_call(call: Call) -> None:
    """A console job, as a job process runs it, whose entrypoint starts a session on its room."""
    from livekit.agents import _core, get_job_context
    from livekit.agents.ipc.job_proc_lazy_main import _JobProc
    from livekit.agents.job import (
        JobAcceptArguments,
        JobContext,
        JobExecutorType,
        RunningJobInfo,
    )
    from livekit.protocol import agent as agent_proto, models

    p = providers(call)
    FrontDesk, _ = agents(call)
    seen: dict[str, Any] = {}

    def in_job(where: str) -> None:
        ctx = get_job_context(required=False)
        seen[where] = ctx is not None and ctx is seen.get("ctx")

    async def entrypoint(ctx: JobContext) -> None:
        seen["ctx"] = ctx
        in_job("entrypoint")
        # the mock room is one per process: count this job's connects
        connects = ctx.room.connect.await_count

        async def on_shutdown(reason: str) -> None:
            in_job("shutdown callback")

        ctx.add_shutdown_callback(on_shutdown)
        session, tools, desc = build_session(call, p)
        call.log("job:", desc)
        call.session = session
        session.on("close", lambda ev: setattr(call, "closes", call.closes + 1))
        # nothing is made to fail: an error event is a provider failure the session made up
        session.on("error", lambda ev: call.violations.append(f"an error event: {ev.error}"))
        await session.start(FrontDesk(tools))
        seen["primary"] = ctx._primary_agent_session is session
        seen["connects"] = ctx.room.connect.await_count - connects
        await asyncio.sleep(call.draw.rng.uniform(0.05, 0.3))
        ctx.shutdown("done")

    async def session_end(ctx: JobContext) -> None:
        in_job("session end")

    runner = _JobProc(
        entrypoint,
        session_end,
        session_end_timeout=5.0,
        executor_type=JobExecutorType.THREAD,
    )
    # the parent's side of the socket, written and read by hand; the handshake's request is
    # already there, so the client reads it at once
    parent, child = socket.socketpair()

    def frame(msg_id: int, fields: dict[str, Any]) -> bytes:
        data = _core.ipc_encode(msg_id, fields)
        return struct.pack(">I", len(data)) + data

    init = {"asyncio_debug": False, "http_proxy": ""}
    init.update(ping_interval=0.0, ping_timeout=0.0, high_ping_threshold=0.0)
    parent.sendall(frame(0, init))
    runner._client = _core.IpcClient(child.detach(), runner.initialize, job=True)
    reader, writer = await asyncio.open_connection(sock=parent)

    async def received() -> str:
        (size,) = struct.unpack(">I", await reader.readexactly(4))
        msg_id, _ = _core.ipc_decode(await reader.readexactly(size))
        return {1: "InitializeResponse", 6: "Exiting", 11: "ShuttingDown"}[msg_id]

    sent = [await received()]
    # the core's IPC client starts the job from its own task, whose context the steps it later
    # runs (the entrypoint, the callbacks) do not share
    serving = contextvars.Context().run(asyncio.ensure_future, runner._client.serve(runner))
    request = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent", metadata=""),
        job=agent_proto.Job(id="AJ_fuzz", room=models.Room(name="console", sid="RM_fuzz")),
        url="",
        token="",
        worker_id="fuzz",
        fake_job=True,
    )
    writer.write(frame(4, request._to_wire()))
    sent += [await asyncio.wait_for(received(), 60) for _ in range(2)]
    await asyncio.wait_for(serving, 60)
    assert runner._job_task is not None
    await asyncio.wait_for(runner._job_task, 60)
    await runner._client.aclose()
    writer.close()
    for where in ("entrypoint", "shutdown callback", "session end"):
        check(seen.get(where), f"the {where} ran without the job's context")
    check(seen.get("primary"), "the session on the job's room is not the job's primary session")
    check(seen.get("connects") == 1, f"the job's room was connected {seen.get('connects')} times")
    check(call.closes == 1, f"the job's session closed {call.closes} times")
    told = ["InitializeResponse", "ShuttingDown", "Exiting"]
    check(sent == told, f"the job told its parent {sent}")


# ============================================================================ the runner


class _Logs(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.bad: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if any(message.startswith(e) for e in EXPECTED_LOGS):
            return
        if record.exc_info or record.levelno >= logging.ERROR:
            self.bad.append(f"{record.levelname} {record.name}: {message}")


async def one_call(seed: int, verbose: bool, *, job: bool = False) -> list[str]:
    draw = Draw(seed)
    draw.chance(0.25)  # the job draw, made by run_seed
    call = Call(draw, verbose)
    call.background = []  # type: ignore[attr-defined]
    try:
        if job:
            await job_call(call)
        else:
            await room_call(call)
    except Failed as e:
        call.violations.append(str(e))
    except Exception as e:  # noqa: BLE001
        import traceback

        if verbose:
            traceback.print_exc()
        call.violations.append(f"raised {type(e).__name__}: {e}")
    for task in call.background:  # type: ignore[attr-defined]
        task.cancel()
    return call.violations


def run_seed(seed: int, verbose: bool) -> list[str]:
    sys.path.insert(0, str(REPO))
    from tests.virtual_time import _now, _perf, _VirtualTimeLoop

    logs = _Logs()
    logging.getLogger().addHandler(logs)
    if Draw(seed).chance(0.25):
        # a job's lifecycle runs on the core's threads: on the wall clock, as a job process runs
        try:
            found = asyncio.run(asyncio.wait_for(one_call(seed, verbose, job=True), 60))
        except Exception as e:  # noqa: BLE001
            found = [f"the seed did not finish: {type(e).__name__}: {e}"]
        finally:
            gc.collect()  # a task the seed left pending is reported as its own
            logging.getLogger().removeHandler(logs)
        return found + [f"logged {line}" for line in logs.bad]
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(time, "time", _now))
        stack.enter_context(mock.patch.object(time, "perf_counter", _perf))
        with contextlib.suppress(Exception):
            from tests.virtual_time import _patched_model_clocks

            stack.enter_context(_patched_model_clocks())
        loop = _VirtualTimeLoop()
        asyncio.set_event_loop(loop)
        try:
            found = loop.run_until_complete(asyncio.wait_for(one_call(seed, verbose), 300))
        except Exception as e:  # noqa: BLE001
            found = [f"the seed did not finish: {type(e).__name__}: {e}"]
        finally:
            # what a failed seed left running ends with it, not in a later seed
            left = asyncio.all_tasks(loop)
            for task in left:
                task.cancel()
            with contextlib.suppress(Exception):
                loop.run_until_complete(asyncio.gather(*left, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            gc.collect()  # a task the seed left pending is reported as its own
            logging.getLogger().removeHandler(logs)
    return found + [f"logged {line}" for line in logs.bad]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("seed", nargs="*", type=int, help="seeds to run, verbosely")
    parser.add_argument("--seeds", type=int, default=64, help="how many seeds")
    parser.add_argument("--first", type=int, default=0, help="the first seed")
    args = parser.parse_args()
    verbose = bool(args.seed)
    if verbose:
        logging.basicConfig(level=logging.WARNING)
    seeds = args.seed or list(range(args.first, args.first + args.seeds))
    failed = 0
    for seed in seeds:
        if verbose:
            print(f"seed {seed}")
        found = run_seed(seed, verbose)
        if found:
            failed += 1
            print(f"seed {seed}: " + "; ".join(dict.fromkeys(found)))
    print(f"{len(seeds) - failed} of {len(seeds)} seeds held")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    sys.exit(main())
