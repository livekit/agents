"""Scripted providers for the native integration checks."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from aiohttp import web

from livekit import rtc
from livekit.agents import (
    Agent,
    APIConnectOptions,
    APIStatusError,
    RunContext,
    VoiceSession,
    llm,
    stt,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

SAMPLE_RATE = 24000


TOOL_CALLS = [
    ("weather", "get_weather", lambda text: {"city": "Paris"}),
    ("billing", "to_billing", lambda text: {}),
    ("order", "look_up_order", lambda text: {}),
    ("keypad", "send_dtmf_events", lambda text: {"events": ["1", "#"]}),
    ("bye", "end_call", lambda text: {}),
    ("book", "book", lambda text: {"day": "Monday"}),
    ("table", "reserve_table", lambda text: {"guests": 2}),
    ("receipt", "change_receipt_email", lambda text: {}),
    ("@", "confirm_email", lambda text: {"email_address": text.split()[-1]}),
]


class ScriptedLLM(llm.LLM):
    """Calls the tool a message asks for when the agent has it, reads a tool's output back,
    else echoes."""

    def chat(
        self, *, chat_ctx: llm.ChatContext, tools: list[llm.Tool] | None = None, **_: object
    ) -> llm.LLMStream:
        return _Stream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )


class _Stream(llm.LLMStream):
    async def _run(self) -> None:
        last = self._chat_ctx.items[-1]
        names = {t.info.name for t in self._tools if isinstance(t, llm.FunctionTool)}
        text = (last.text_content or "") if last.type == "message" else ""
        delta = llm.ChoiceDelta(role="assistant", content=f"You said: {text}")
        if last.type == "function_call_output":
            delta = llm.ChoiceDelta(role="assistant", content=f"The tool said: {last.output}")
        for words, name, arguments in TOOL_CALLS:
            if words in text and name in names:
                call = llm.FunctionToolCall(
                    name=name, arguments=json.dumps(arguments(text)), call_id=shortuuid("call_")
                )
                delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
                break
        self.send(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))


class ScriptedSTT(stt.STT):
    """A streaming STT that hears "my name is Theo" in half a second of anything."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(self, **_: object) -> stt.RecognizeStream:
        return _Recognize(stt=self, conn_options=DEFAULT_API_CONNECT_OPTIONS)


class _Recognize(stt.RecognizeStream):
    async def _run(self) -> None:
        heard = 0.0
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame) or heard < 0:
                continue
            heard += frame.duration
            if heard >= 0.5:
                heard = -1.0
                said = stt.SpeechData(language="en", text="my name is Theo", confidence=1.0)
                for kind, alternatives in (
                    (stt.SpeechEventType.START_OF_SPEECH, []),
                    (stt.SpeechEventType.FINAL_TRANSCRIPT, [said]),
                    (stt.SpeechEventType.END_OF_SPEECH, []),
                ):
                    self.send(stt.SpeechEvent(type=kind, alternatives=alternatives))


class BatchSTT(stt.STT):
    """A recognize-only STT: it hears "batch words" in any utterance of a tenth of a second."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.utterances: list[float] = []

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        self.utterances.append(sum(f.duration for f in frames))  # type: ignore[attr-defined]
        said = stt.SpeechData(language="en", text="batch words", confidence=1.0)
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[said])


class SilentTTS(tts.TTS):
    """20 ms of silence per character, as one chunked synthesis."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return _Chunked(tts=self, input_text=chunk.text, conn_options=conn_options)


class _Chunked(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid("req_"),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(b"\x00\x00" * (SAMPLE_RATE // 50 * max(1, len(self.input_text))))
        output_emitter.flush()


async def until(done: Callable[[], bool], what: str, timeout: float = 10) -> None:
    try:
        async with asyncio.timeout(timeout):
            while not done():
                await asyncio.sleep(0.01)
    except TimeoutError:
        raise AssertionError(what) from None


def billing() -> Agent:
    agent = Agent(instructions="You handle billing.", name="Billing")

    @agent.on_enter
    async def greet(session: VoiceSession) -> None:
        session.say("Billing here, how can I help?")

    return agent


def front_desk() -> Agent:
    agent = Agent(instructions="You are the front desk.", name="FrontDesk")

    @agent.on_enter
    async def greet(session: VoiceSession) -> None:
        session.say("Hello, front desk.")

    @agent.tool
    async def get_weather(ctx: RunContext, city: str) -> str:
        """The weather in a city."""
        return f"sunny in {city}"

    @agent.tool
    async def to_billing(ctx: RunContext) -> None:
        """Transfer the caller to billing."""
        assert ctx.live_session is not None
        ctx.live_session.update_agent(billing())

    return agent


def print_item(ev: object) -> None:
    item = ev.item  # type: ignore[attr-defined]
    if item.type == "message":
        print(f"conversation_item {item.role}: {item.text_content!r}")
    else:
        print(f"conversation_item {item.type}: {item.old_agent_id} -> {item.new_agent_id}")


class ScriptedRealtime(llm.RealtimeModel):
    """A realtime model that answers each reply request with one spoken message."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=True,
            )
        )

    def session(self, *, turn_detection_disabled: bool = False) -> llm.RealtimeSession:
        return _RealtimeSession(self)


class _RealtimeSession(llm.RealtimeSession):
    def __init__(self, model: ScriptedRealtime) -> None:
        super().__init__(model)
        self._chat_ctx = llm.ChatContext.empty()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        pass

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    def update_options(self, **_: object) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        pass

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(self, **_: object) -> asyncio.Future[llm.GenerationCreatedEvent]:
        async def _one(value: object) -> object:
            yield value

        modalities: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
        modalities.set_result(["audio", "text"])
        silence = rtc.AudioFrame(b"\x00\x00" * 4800, SAMPLE_RATE, 1, 4800)
        message = llm.MessageGeneration(
            message_id=shortuuid("msg_"),
            text_stream=_one("Hello from the realtime model."),
            audio_stream=_one(silence),
            modalities=modalities,
        )
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.get_running_loop().create_future()
        fut.set_result(
            llm.GenerationCreatedEvent(
                message_stream=_one(message),
                function_stream=_empty(),
                user_initiated=True,
                response_id=shortuuid("resp_"),
            )
        )
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


async def _empty() -> object:
    return
    yield


class _DyingSession(_RealtimeSession):
    def __init__(self, model: ScriptedRealtime) -> None:
        super().__init__(model)
        asyncio.get_running_loop().call_soon(
            self.emit,
            "error",
            llm.RealtimeModelError(
                timestamp=0.0, label="dying", error=RuntimeError("gone"), recoverable=False
            ),
        )


class DyingRealtime(ScriptedRealtime):
    def session(self, *, turn_detection_disabled: bool = False) -> llm.RealtimeSession:
        return _DyingSession(self)


def _square(level: float, ms: int = 100) -> rtc.AudioFrame:
    """A square-wave frame whose RMS is ``level``."""
    n = SAMPLE_RATE * ms // 1000
    amp = int(level * 32767)
    data = b"".join(
        (amp if i % 2 == 0 else -amp).to_bytes(2, "little", signed=True) for i in range(n)
    )
    return rtc.AudioFrame(data, SAMPLE_RATE, 1, n)


class ScriptedDuplex(llm.DuplexModel):
    """A duplex model that answers each ask with one utterance and its transcript."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=False, auto_tool_reply_generation=False
            )
        )
        self.sessions: list[_DuplexScript] = []

    def session(self) -> llm.DuplexSession:
        self.sessions.append(_DuplexScript(self))
        return self.sessions[-1]

    async def aclose(self) -> None:
        pass


class _DuplexScript(llm.DuplexSession):
    def __init__(self, model: ScriptedDuplex) -> None:
        super().__init__(model)
        self.audio: asyncio.Queue[llm.DuplexAudioFrame] = asyncio.Queue()
        self.configured_with: list[object] = []

    @property
    def audio_stream(self) -> AsyncIterator[llm.DuplexAudioFrame]:
        async def _frames() -> AsyncIterator[llm.DuplexAudioFrame]:
            while True:
                yield await self.audio.get()

        return _frames()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def _update_instructions(self, instructions: str) -> None:
        self.configured_with.append(instructions)

    async def _append_items(self, items: list[llm.ChatItem]) -> None:
        pass

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    def _update_options(self, **_: object) -> None:
        pass

    def _generate_reply(self, **_: object) -> None:
        for level, count in ((0.001, 20), (0.3, 3)):
            for _ in range(count):
                self.audio.put_nowait(llm.DuplexAudioFrame(_square(level)))
        self.emit(
            "transcript_delta",
            llm.DuplexOutputTranscriptDelta(
                text="Hello from the duplex model.", start_ms=2000, end_ms=2300
            ),
        )
        for _ in range(8):
            self.audio.put_nowait(llm.DuplexAudioFrame(_square(0.001)))


async def _openai_server(ws: web.WebSocketResponse, sessions: list[dict[str, Any]]) -> None:
    """The OpenAI Realtime protocol's happy path: items are acked, a reply is one spoken message."""
    tail: str | None = None
    async for msg in ws:
        ev = json.loads(msg.data)
        kind = ev["type"]
        if kind == "session.update":
            sessions.append(ev["session"])
        elif kind == "conversation.item.create":
            item = {**ev["item"], "id": ev["item"].get("id") or shortuuid("item_")}
            await ws.send_json(
                {"type": "conversation.item.added", "previous_item_id": tail, "item": item}
            )
            tail = item["id"]
        elif kind == "conversation.item.delete":
            await ws.send_json({"type": "conversation.item.deleted", "item_id": ev["item_id"]})
        elif kind == "response.create":
            resp, item = shortuuid("resp_"), shortuuid("item_")
            meta = ev["response"]["metadata"]
            pcm = base64.b64encode(b"\x00\x00" * 4800).decode()
            for out in (
                {"type": "response.created", "response": {"id": resp, "metadata": meta}},
                {"type": "response.output_item.added", "item": {"id": item, "type": "message"}},
                {
                    "type": "conversation.item.added",
                    "previous_item_id": tail,
                    "item": {"id": item, "type": "message", "role": "assistant", "content": []},
                },
                {"type": "response.content_part.added", "item_id": item, "part": {"type": "audio"}},
                {
                    "type": "response.output_audio_transcript.delta",
                    "item_id": item,
                    "delta": "Hello from the core's realtime client.",
                },
                {"type": "response.output_audio.delta", "item_id": item, "delta": pcm},
                {"type": "response.output_item.done", "item": {"id": item, "type": "message"}},
                {"type": "response.done", "response": {"id": resp, "status": "completed"}},
            ):
                await ws.send_json(out)
            tail = item


QUICK_RETRIES = APIConnectOptions(max_retry=2, retry_interval=0.0)


class FlakyLLM(ScriptedLLM):
    """Answers once ``fails`` attempts have failed with a 503."""

    def __init__(self, fails: int) -> None:
        super().__init__()
        self.fails = fails
        self.calls = 0

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = QUICK_RETRIES,
        **_: Any,
    ) -> llm.LLMStream:
        return _FlakyStream(self, chat_ctx=chat_ctx, tools=[], conn_options=conn_options)


class _FlakyStream(_Stream):
    async def _run(self) -> None:
        flaky = self._llm
        assert isinstance(flaky, FlakyLLM)
        flaky.calls += 1
        if flaky.calls <= flaky.fails:
            raise APIStatusError("unavailable", status_code=503)
        await super()._run()


class FlakyTTS(SilentTTS):
    """Fails each sentence's first ``fails`` attempts with a 503; streams when ``streaming``."""

    def __init__(self, fails: int, *, streaming: bool = False) -> None:
        super().__init__()
        self._capabilities = tts.TTSCapabilities(streaming=streaming)
        self.fails = fails
        self.attempts: dict[str, int] = {}

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return _FlakyChunked(tts=self, input_text=chunk.text, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> tts.SynthesizeStream:
        return _FlakySynthesis(tts=self, conn_options=conn_options)

    def attempt(self, text: str) -> None:
        self.attempts[text] = self.attempts.get(text, 0) + 1
        if self.attempts[text] <= self.fails:
            raise APIStatusError("unavailable", status_code=503)


class _FlakyChunked(_Chunked):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        assert isinstance(self._tts, FlakyTTS)
        self._tts.attempt(self.input_text)
        await super()._run(output_emitter)


class _FlakySynthesis(tts.SynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid("req_"),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        text = "".join([t.text async for t in self.input if isinstance(t, tts.SpeechChunk)])
        assert isinstance(self._tts, FlakyTTS)
        self._tts.attempt(text)
        output_emitter.start_segment(segment_id=shortuuid("seg_"))
        output_emitter.push(b"\x00\x00" * (SAMPLE_RATE // 50 * max(1, len(text))))
        output_emitter.end_segment()


class FlakySTT(BatchSTT):
    def __init__(self, fails: int) -> None:
        super().__init__()
        self.fails = fails

    async def _recognize_impl(self, buffer: object, **kw: object) -> stt.SpeechEvent:
        if len(self.utterances) < self.fails:
            self.utterances.append(0.0)
            raise APIStatusError("unavailable", status_code=503)
        return await super()._recognize_impl(buffer, **kw)


def error_events(provider: Any) -> list[bool]:
    seen: list[bool] = []
    provider.on("error", lambda ev: seen.append(ev.recoverable))
    return seen


def metrics_events(provider: Any) -> list[Any]:
    seen: list[Any] = []
    provider.on("metrics_collected", seen.append)
    return seen
