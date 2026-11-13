"""Test doubles: Python providers and audio I/O written against the public base classes, exactly
as a plugin would be, so every test also drives them through the native bridge.

    FakeLLM(["Hi!", tool_call("lookup", city="Paris"), "It is sunny."])   # replies in order
    FakeLLM({"hello": "Hi!", "sunny": "It is sunny."})                  # keyed by the last input
    FakeLLM(lambda chat_ctx: ...)                                        # computed per request
    FakeSTT(["hello", "how are you"])     # one final transcript per utterance it hears
    FakeVAD()                             # loud audio is speech, 0.2 s of quiet ends it
    FakeTTS()                             # 0.1 s of a 440 Hz tone per synthesis
    FakeAudioInput().speak(0.5)           # half a second of speech, then a pause
    FakeVideoGenerator()                  # an avatar face: lit while it mouths audio, dark idle

A reply is a str, a tool call, a raw ``ChatChunk``, an exception (raised) or a ``Hang`` (blocks
until released or cancelled), or a list of those streamed in order. Each attempt a provider makes
is one request: a retried request takes the next reply.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from typing import Literal, TypeAlias, TypeVar

import aiohttp
import numpy as np
from aiohttp import web

from livekit import api, rtc
from livekit.agents import (
    VAD,
    TokenUsage,
    VADCapabilities,
    VADEvent,
    VADEventType,
    VADStream,
    llm,
    stt,
    tokenize,
    tts,
)
from livekit.agents._generated.language import (
    LanguageCode,  # SpeechData's field type, not re-exported
)
from livekit.agents.tokenize import SentenceStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import aio, shortuuid
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarSession,
    DataStreamAudioOutput,
    VideoGenerator,
)
from livekit.agents.voice.io import (
    AudioInput,
    AudioOutput,
    AudioOutputCapabilities,
    PlaybackFinishedEvent,
    PlaybackSegment,
    TranscriptionOutput,
)
from livekit.protocol import agent, models

SAMPLE_RATE = 24000
LOUD = 0.01  # RMS above which audio is speech
PAUSE = 0.2  # seconds of quiet that end an utterance

_call_ids = itertools.count(1)


class Hang:
    """A reply part that blocks the stream until ``release()``d or cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.ended = asyncio.Event()
        self._released = asyncio.Event()

    def release(self) -> None:
        self._released.set()

    async def wait(self) -> None:
        self.started.set()
        try:
            await self._released.wait()
        finally:
            self.ended.set()


Part: TypeAlias = "str | llm.FunctionToolCall | llm.ChatChunk | BaseException | Hang"
Reply: TypeAlias = "Part | list[Part]"


def tool_call(
    name: str, /, *, call_id: str | None = None, **arguments: object
) -> llm.FunctionToolCall:
    return llm.FunctionToolCall(
        name=name, arguments=json.dumps(arguments), call_id=call_id or f"call_{next(_call_ids)}"
    )


def audio_frame(seconds: float = 0.1, *, level: float = 0.3) -> rtc.AudioFrame:
    """``seconds`` of a square wave whose RMS is ``level``: 0 is silence."""
    n = round(SAMPLE_RATE * seconds)
    samples = np.where(np.arange(n) % 2 == 0, 1, -1) * int(level * 32767)
    return rtc.AudioFrame(samples.astype(np.int16).tobytes(), SAMPLE_RATE, 1, n)


def _tone(seconds: float, frequency: float | None) -> bytes:
    n = round(SAMPLE_RATE * seconds)
    if frequency is None:
        return bytes(2 * n)
    samples = 8000 * np.sin(2 * np.pi * frequency * np.arange(n) / SAMPLE_RATE)
    return samples.astype(np.int16).tobytes()


async def until(done: Callable[[], object], what: str, timeout: float = 10) -> None:
    """Waits until ``done()`` holds, failing with ``what`` after ``timeout`` seconds."""
    try:
        async with asyncio.timeout(timeout):
            while not done():
                await asyncio.sleep(0.01)
    except TimeoutError:
        raise AssertionError(what) from None


# -- LLM


def last_input(chat_ctx: llm.ChatContext) -> str | None:
    """What a model answers: the last user or system text, or the last tool output."""
    items = [i for i in chat_ctx.items if i.type not in ("agent_handoff", "agent_config_update")]
    if not items:
        return None
    item = items[-1]
    if item.type == "message" and item.role in ("user", "system"):
        return item.text_content
    if item.type == "function_call_output":
        return item.output
    return None


class FakeLLM(llm.LLM):
    """Answers requests in order from a list, by their last input from a dict, or with what a
    function of the request's chat context returns.

    Keeps each request's chat context in ``requests`` and the tools it offered in ``tools``;
    ``extend()`` queues more replies on a list-answering model.
    """

    def __init__(
        self,
        replies: Iterable[Reply] | Mapping[str, Reply] | Callable[[llm.ChatContext], Reply] = (),
        *,
        prewarm_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.replies: deque[Reply] | Mapping[str, Reply] | Callable[[llm.ChatContext], Reply] = (
            replies if isinstance(replies, Mapping) or callable(replies) else deque(replies)
        )
        self.requests: list[llm.ChatContext] = []
        self.tools: list[list[llm.Tool]] = []
        self.prewarmed = asyncio.Event()
        self._prewarm_error = prewarm_error

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: object,
    ) -> llm.LLMStream:
        return _LLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)

    def extend(self, replies: Iterable[Reply]) -> None:
        """Queue more replies after those left; only for a model answering from a list."""
        if not isinstance(self.replies, deque):
            raise TypeError("only a FakeLLM answering from a list takes more replies")
        self.replies.extend(replies)

    async def _prewarm_impl(self) -> None:
        self.prewarmed.set()
        if self._prewarm_error is not None:
            raise self._prewarm_error

    def _next(self, chat_ctx: llm.ChatContext, tools: list[llm.Tool]) -> Reply:
        self.requests.append(chat_ctx.copy())
        self.tools.append(list(tools))
        if callable(self.replies):
            return self.replies(chat_ctx)
        text = last_input(chat_ctx)
        if isinstance(self.replies, Mapping):
            if text is None or text not in self.replies:
                raise RuntimeError(f"FakeLLM has no reply for {text!r}")
            return self.replies[text]
        if not self.replies:
            raise RuntimeError(f"FakeLLM has no reply left for {text!r}")
        return self.replies.popleft()


class _LLMStream(llm.LLMStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, FakeLLM)
        reply = self._llm._next(self._chat_ctx, self._tools)
        for part in reply if isinstance(reply, list) else [reply]:
            if isinstance(part, BaseException):
                raise part
            if isinstance(part, Hang):
                await part.wait()
            elif isinstance(part, llm.ChatChunk):
                self.send(part)
            elif isinstance(part, llm.FunctionToolCall):
                delta = llm.ChoiceDelta(role="assistant", tool_calls=[part])
                self.send(llm.ChatChunk(id="reply", delta=delta))
            else:
                delta = llm.ChoiceDelta(role="assistant", content=part)
                self.send(llm.ChatChunk(id="reply", delta=delta))


# -- STT and VAD


class _Ear:
    """Splits audio into utterances: loud audio is speech, ``PAUSE`` seconds of quiet end it."""

    def __init__(self) -> None:
        self.speech: list[rtc.AudioFrame] | None = None
        self.quiet = 0.0
        self.heard = 0.0

    def hear(self, frame: rtc.AudioFrame) -> str | None:
        """``"start"`` or ``"end"`` when this frame starts or ends an utterance."""
        self.heard += frame.duration
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float64)
        rms = float(np.sqrt(np.mean(samples**2))) / 32767 if samples.size else 0.0
        if self.speech is None:
            if rms < LOUD:
                return None
            self.speech, self.quiet = [frame], 0.0
            return "start"
        self.speech.append(frame)
        self.quiet = 0.0 if rms >= LOUD else self.quiet + frame.duration
        if self.quiet < PAUSE:
            return None
        return "end"


class FakeSTT(stt.STT):
    """Hears ``phrases`` in order: one final transcript per utterance, or per ``recognize``.

    An exception in ``phrases`` is raised by the request that reaches it.
    """

    def __init__(
        self, phrases: Iterable[str | BaseException] = (), *, streaming: bool = True
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=streaming, interim_results=False)
        )
        self.phrases = deque(phrases)

    def _raise_failure(self) -> None:
        if self.phrases and isinstance(failure := self.phrases[0], BaseException):
            self.phrases.popleft()
            raise failure

    def _final(self) -> stt.SpeechEvent:
        self._raise_failure()
        if not self.phrases:
            raise RuntimeError("FakeSTT has no phrase left")
        text = self.phrases.popleft()
        assert isinstance(text, str)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LanguageCode("en"), text=text, confidence=1.0)],
        )

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        return self._final()

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS, **_: object
    ) -> stt.RecognizeStream:
        return _RecognizeStream(stt=self, conn_options=conn_options)


class _RecognizeStream(stt.RecognizeStream):
    async def _run(self) -> None:
        provider = self._stt
        assert isinstance(provider, FakeSTT)
        provider._raise_failure()
        ear = _Ear()
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame) or ear.hear(frame) != "end":
                continue
            ear.speech = None
            if not provider.phrases:
                continue
            final = provider._final()
            self.send(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            self.send(final)
            self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
            provider._raise_failure()


class FakeVAD(VAD):
    """Hears loud audio as speech; ``PAUSE`` seconds of quiet end it. ``error`` fails every
    stream as it starts."""

    def __init__(self, *, error: Exception | None = None) -> None:
        super().__init__(capabilities=VADCapabilities(update_interval=0.1))
        self.error = error

    def stream(self) -> VADStream:
        return _VADStream(self)


class _VADStream(VADStream):
    async def _run(self) -> None:
        assert isinstance(self._vad, FakeVAD)
        if self._vad.error is not None:
            raise self._vad.error
        ear = _Ear()
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame):
                continue
            heard = ear.hear(frame)
            speech = ear.speech or []
            speaking = ear.speech is not None and heard != "end"
            spoken = sum(f.duration for f in speech)
            kinds = {
                "start": [VADEventType.START_OF_SPEECH, VADEventType.INFERENCE_DONE],
                "end": [VADEventType.INFERENCE_DONE, VADEventType.END_OF_SPEECH],
            }.get(heard or "", [VADEventType.INFERENCE_DONE])
            for kind in kinds:
                self.send(
                    VADEvent(
                        type=kind,
                        samples_index=0,
                        timestamp=ear.heard,
                        speech_duration=spoken,
                        silence_duration=ear.quiet,
                        raw_accumulated_speech=spoken if speaking else 0.0,
                        raw_accumulated_silence=ear.quiet,
                        frames=speech if kind == VADEventType.END_OF_SPEECH else [frame],
                        probability=1.0 if speaking else 0.0,
                        speaking=speaking,
                    )
                )
            if heard == "end":
                ear.speech = None


# -- TTS


class FakeTTS(tts.TTS):
    """Speaks ``seconds`` of a ``tone`` Hz tone (silence when None) per synthesis or streamed
    segment, as soon as its first text arrives. Keeps each synthesized text in ``texts``.

    ``error`` fails every synthesis, ``failures`` one attempt each, in order; ``usage`` is
    reported for each synthesis.
    """

    def __init__(
        self,
        *,
        streaming: bool = True,
        seconds: float = 0.1,
        tone: float | None = 440,
        error: Exception | None = None,
        failures: Iterable[Exception] = (),
        usage: TokenUsage | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=streaming),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self.seconds = seconds
        self.tone = tone
        self.error = error
        self.failures = deque(failures)
        self.usage = usage
        self.texts: list[str] = []

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        self.texts.append(chunk.text)
        return _ChunkedStream(tts=self, input_text=chunk.text, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> tts.SynthesizeStream:
        return _SynthesizeStream(tts=self, conn_options=conn_options)

    def _speak(self, emitter: tts.AudioEmitter, text: str) -> None:
        if self.error is not None:
            raise self.error
        if self.failures:
            raise self.failures.popleft()
        emitter.push(_tone(self.seconds, self.tone))
        if self.usage is not None:
            emitter.report_usage(self.usage, characters_count=len(text))


class _ChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        assert isinstance(self._tts, FakeTTS)
        output_emitter.initialize(
            request_id=shortuuid("req_"),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        self._tts._speak(output_emitter, self.input_text)
        output_emitter.flush()


class _SynthesizeStream(tts.SynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        provider = self._tts
        assert isinstance(provider, FakeTTS)
        output_emitter.initialize(
            request_id=shortuuid("req_"),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        text: str | None = None
        async for data in self.input:
            if isinstance(data, tts.SpeechChunk):
                if text is None:
                    text = ""
                    output_emitter.start_segment(segment_id=shortuuid("segment_"))
                    provider._speak(output_emitter, data.text)
                text += data.text
            elif text is not None:
                provider.texts.append(text)
                text = None
                output_emitter.end_segment()
        if text is not None:
            provider.texts.append(text)
            output_emitter.end_segment()


class FakeSentenceTokenizer(tokenize.SentenceTokenizer):
    """Splits sentences on ``|``, nothing a default tokenizer would do; counts closed streams."""

    def __init__(self) -> None:
        self.closed = 0

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [part for part in text.split("|") if part]

    def stream(self, *, language: str | None = None) -> SentenceStream:
        return _SentenceStream(self)


class _SentenceStream(tokenize.BufferedSentenceStream):
    def __init__(self, owner: FakeSentenceTokenizer) -> None:
        super().__init__(tokenizer=owner.tokenize, min_token_len=1, min_ctx_len=1)
        self._owner = owner

    async def aclose(self) -> None:
        self._owner.closed += 1
        await super().aclose()


# -- audio input and output


class FakeAudioInput(AudioInput):
    """Hands the session the audio a test pushes, no faster than real time, as a microphone
    does: a burst never queues more audio than a live call would."""

    def __init__(self) -> None:
        super().__init__(label="fake")
        self._frames = aio.Chan[rtc.AudioFrame]()
        self._next_at = 0.0

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self._frames.__anext__()
        now = time.monotonic()
        if self._next_at > now:
            await asyncio.sleep(self._next_at - now)  # the frame is still being captured
        self._next_at = max(self._next_at, now) + frame.duration
        return frame

    def push(self, frame: rtc.AudioFrame) -> None:
        self._frames.send_nowait(frame)

    def speak(self, seconds: float = 0.5) -> None:
        """``seconds`` of speech, then a pause that ends the utterance."""
        for _ in range(round(seconds / 0.1)):
            self.push(audio_frame(0.1))
        for _ in range(round(1.5 * PAUSE / 0.1)):
            self.push(audio_frame(0.1, level=0))


class FakeAudioOutput(AudioOutput):
    """Plays what it is given in real time and reports each segment played or interrupted;
    counts the frames it was given in ``frames``. With ``fail_at_frame``, that frame (counted
    from 0) raises once its segment opened."""

    def __init__(self, *, fail_at_frame: int | None = None) -> None:
        super().__init__(
            label="fake",
            next_in_chain=None,
            sample_rate=None,
            capabilities=AudioOutputCapabilities(pause=False),
        )
        self.captured = asyncio.Event()
        self.frames = 0
        self._fail_at_frame = fail_at_frame
        self._pushed = 0.0
        self._started_at: float | None = None
        self._segment: PlaybackSegment | None = None
        self._finish: asyncio.TimerHandle | None = None

    def _played(self) -> float:
        return 0.0 if self._started_at is None else time.time() - self._started_at

    def _finished(self, *, interrupted: bool, position: float) -> None:
        if self._finish is not None:
            self._finish.cancel()
            self._finish = None
        self._pushed, self._started_at = 0.0, None
        assert self._segment is not None, "a segment finished before it started"
        self.on_playback_finished(
            segment=self._segment,
            playback_position=position,
            interrupted=interrupted,
            synchronized_transcript=None,
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.frames += 1
        if self.frames - 1 == self._fail_at_frame:
            raise RuntimeError("the audio output failed")
        self._segment = self.playback_segment
        assert self._segment is not None, "audio captured outside a playback segment"
        if self._started_at is None:
            self._started_at = time.time()
            self.on_playback_started(segment=self._segment, created_at=self._started_at)
        self._pushed += frame.duration
        self.captured.set()

    def flush(self) -> None:
        super().flush()
        if not self._pushed:
            return
        if self._finish is not None:
            self._finish.cancel()
        pushed = self._pushed
        self._finish = asyncio.get_running_loop().call_later(
            max(pushed - self._played(), 0.0),
            lambda: self._finished(interrupted=False, position=pushed),
        )

    def clear_buffer(self) -> None:
        if self._pushed:
            self._finished(interrupted=True, position=min(self._played(), self._pushed))


class FakeAudioStage(AudioOutput):
    """A user's own stage in front of ``next_in_chain``: passes every frame, flush and clear on,
    counts the frames in ``frames`` and keeps the playback reports it hears in ``finished``."""

    def __init__(self, next_in_chain: AudioOutput | DataStreamAudioOutput) -> None:
        super().__init__(
            label="stage",
            capabilities=AudioOutputCapabilities(pause=True),
            next_in_chain=next_in_chain,
        )
        self.frames = 0
        self.finished: list[PlaybackFinishedEvent] = []
        self.on("playback_finished", self.finished.append)
        self._below = next_in_chain

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.frames += 1
        await self._below.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self._below.flush()

    def clear_buffer(self) -> None:
        self._below.clear_buffer()


class FakeTranscriptionOutput(TranscriptionOutput):
    """Keeps each flushed transcript in ``texts``."""

    def __init__(self) -> None:
        super().__init__(label="fake", next_in_chain=None)
        self.texts: list[str] = []
        self._text = ""

    async def capture_text(self, text: str) -> None:
        self._text += text

    def flush(self) -> None:
        self.texts.append(self._text)
        self._text = ""


# -- realtime, duplex and avatar


class FakeRealtime(llm.RealtimeModel):
    """Answers each reply request with one spoken message; with ``dies``, each session it opens
    fails at once with an unrecoverable error."""

    def __init__(
        self, reply: str = "Hello from the realtime model.", *, dies: bool = False
    ) -> None:
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
        self.reply = reply
        self.dies = dies

    def session(self, *, turn_detection_disabled: bool = False) -> FakeRealtimeSession:
        return FakeRealtimeSession(self)


_T = TypeVar("_T")


async def _one(value: _T) -> AsyncIterator[_T]:
    yield value


async def _none() -> AsyncIterator[llm.FunctionCall]:
    return
    yield


class FakeRealtimeSession(llm.RealtimeSession):
    def __init__(self, model: FakeRealtime) -> None:
        super().__init__(model)
        self._reply = model.reply
        self._chat_ctx = llm.ChatContext.empty()
        if model.dies:
            died = llm.RealtimeModelError(
                timestamp=time.time(), label="fake", error=RuntimeError("gone"), recoverable=False
            )
            asyncio.get_running_loop().call_soon(self.emit, "error", died)

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
        loop = asyncio.get_running_loop()
        modalities: asyncio.Future[list[Literal["text", "audio"]]] = loop.create_future()
        modalities.set_result(["audio", "text"])
        message = llm.MessageGeneration(
            message_id=shortuuid("msg_"),
            text_stream=_one(self._reply),
            audio_stream=_one(audio_frame(0.2, level=0)),
            modalities=modalities,
        )
        created: asyncio.Future[llm.GenerationCreatedEvent] = loop.create_future()
        created.set_result(
            llm.GenerationCreatedEvent(
                message_stream=_one(message),
                function_stream=_none(),
                user_initiated=True,
                response_id=shortuuid("resp_"),
            )
        )
        return created

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


class FakeDuplex(llm.DuplexModel):
    """Answers each ask with two seconds of quiet, a short utterance and its transcript; keeps
    its sessions."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=False, auto_tool_reply_generation=False
            )
        )
        self.sessions: list[FakeDuplexSession] = []

    def session(self) -> llm.DuplexSession:
        self.sessions.append(FakeDuplexSession(self))
        return self.sessions[-1]

    async def aclose(self) -> None:
        pass


class FakeDuplexSession(llm.DuplexSession):
    """Keeps the instructions it was configured with."""

    def __init__(self, model: FakeDuplex) -> None:
        super().__init__(model)
        self.audio: asyncio.Queue[llm.DuplexAudioFrame] = asyncio.Queue()
        self.configured_with: list[object] = []

    @property
    def audio_stream(self) -> AsyncIterator[llm.DuplexAudioFrame]:
        async def frames() -> AsyncIterator[llm.DuplexAudioFrame]:
            while True:
                yield await self.audio.get()

        return frames()

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

    def _generate_reply(self, **options: object) -> None:
        for level, count in ((0.001, 20), (0.3, 3)):
            for _ in range(count):
                self.audio.put_nowait(llm.DuplexAudioFrame(audio_frame(0.1, level=level)))
        self.emit(
            "transcript_delta",
            llm.DuplexOutputTranscriptDelta(
                text="Hello from the duplex model.", start_ms=2000, end_ms=2300
            ),
        )
        for _ in range(8):
            self.audio.put_nowait(llm.DuplexAudioFrame(audio_frame(0.1, level=0.001)))


class FakeAvatar(AvatarSession):
    """An avatar that never joins; counts its closes."""

    closes = 0

    @property
    def avatar_identity(self) -> str:
        return "avatar"

    @property
    def provider(self) -> str:
        return "test-provider"

    async def aclose(self) -> None:
        self.closes += 1
        await super().aclose()


class FakeVideoGenerator(VideoGenerator):
    """An avatar's face for ``AvatarRunner``: one 64x64 frame per audio frame it mouths, white,
    followed by that audio; black frames every 40 ms while idle. ``clear_buffer`` drops the
    audio not yet mouthed and counts in ``clears``."""

    SIZE = 64
    IDLE = 0.04

    def __init__(self) -> None:
        self.clears = 0
        self._audio: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        self._audio.put_nowait(frame)

    def clear_buffer(self) -> None:
        self.clears += 1
        while not self._audio.empty():
            self._audio.get_nowait()

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._frames()

    async def _frames(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        while True:
            try:
                item = await asyncio.wait_for(self._audio.get(), self.IDLE)
            except TimeoutError:
                yield self.face(lit=False)
                continue
            if isinstance(item, rtc.AudioFrame):
                yield self.face(lit=True)
            yield item

    @classmethod
    def face(cls, *, lit: bool) -> rtc.VideoFrame:
        pixels = bytes([255 if lit else 0, 255 if lit else 0, 255 if lit else 0, 255])
        return rtc.VideoFrame(cls.SIZE, cls.SIZE, rtc.VideoBufferType.RGBA, pixels * cls.SIZE**2)


# -- LiveKit server


class FakeLiveKit:
    """A LiveKit server's worker endpoint (``/agent``), as a worker sees it: it answers the
    registration, records what the worker reports, and dispatches jobs the way the server does.

        async with FakeLiveKit(api_key, api_secret) as livekit:   # ``livekit.url`` is LIVEKIT_URL
            answer = await livekit.dispatch(job)   # offered once the worker is available
            answer = await livekit.offer(job)      # offered now, whatever the worker reported
            await livekit.terminate(job.id)
            await livekit.disconnect()             # the worker reconnects and registers again

    ``hold_registration=True`` (or ``hold_registration()`` later, before a reconnection) leaves
    registrations unanswered until ``answer_registration()``.
    An assigned job's room URL has no server behind it: its job never connects to the room.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        region: str = "",
        hold_registration: bool = False,
        timeout: float = 30,
    ) -> None:
        self.api_key, self.api_secret, self.region = api_key, api_secret, region
        self.timeout = timeout
        self.url = ""
        # one entry per worker connection: its query string, `Authorization` and registration
        self.connections: list[tuple[dict[str, str], str, agent.RegisterWorkerRequest]] = []
        self.statuses: list[agent.UpdateWorkerStatus] = []
        self.answers: dict[str, agent.AvailabilityResponse] = {}
        self.job_statuses: dict[str, list[agent.JobStatus]] = {}
        # the jobs a worker reported holding after it registered again
        self.migrations: list[agent.MigrateJobRequest] = []
        self._registration = asyncio.Event()
        if not hold_registration:
            self._registration.set()
        self._offered: dict[str, agent.Job] = {}
        self._handlers: set[asyncio.Task[object]] = set()
        self._ws: web.WebSocketResponse | None = None
        self._transport: asyncio.Transport | None = None
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> FakeLiveKit:
        app = web.Application()
        app.router.add_get("/agent", self._serve)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"ws://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"  # type: ignore[union-attr]
        return self

    async def __aexit__(self, *_: object) -> None:
        for handler in self._handlers:
            handler.cancel()
        await asyncio.gather(*self._handlers, return_exceptions=True)
        if self._runner is not None:
            await self._runner.cleanup()

    @property
    def registered(self) -> bool:
        return self._ws is not None and self._registration.is_set()

    @property
    def available(self) -> bool:
        """The worker's last report: it takes jobs."""
        return bool(self.statuses) and self.statuses[-1].status == agent.WorkerStatus.WS_AVAILABLE

    def hold_registration(self) -> None:
        self._registration.clear()

    def answer_registration(self) -> None:
        self._registration.set()

    async def until_registered(self) -> None:
        await until(lambda: self.registered, "the worker registered", self.timeout)

    async def dispatch(self, job: agent.Job) -> agent.AvailabilityResponse:
        """Offers ``job`` once the worker reports itself available, and assigns it if accepted."""
        await until(lambda: self.available, "the worker reported itself available", self.timeout)
        return await self.offer(job)

    async def offer(self, job: agent.Job) -> agent.AvailabilityResponse:
        """Offers ``job`` now, and assigns it if accepted."""
        self._offered[job.id] = job
        await self._send(agent.ServerMessage(availability=agent.AvailabilityRequest(job=job)))
        await until(lambda: job.id in self.answers, f"an answer to {job.id}", self.timeout)
        return self.answers[job.id]

    async def terminate(self, job_id: str) -> None:
        await self._send(agent.ServerMessage(termination=agent.JobTermination(job_id=job_id)))

    async def disconnect(self) -> None:
        """Drops the worker's connection without a close handshake."""
        assert self._transport is not None, "no worker connected"
        self._ws, transport, self._transport = None, self._transport, None
        transport.abort()

    async def _serve(self, request: web.Request) -> web.WebSocketResponse:
        handler = asyncio.current_task()
        assert handler is not None
        self._handlers.add(handler)
        try:
            return await self._register(request)
        finally:
            self._handlers.discard(handler)

    async def _register(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        register = agent.WorkerMessage.FromString(await ws.receive_bytes()).register
        self.connections.append(
            (dict(request.query), request.headers.get("Authorization", ""), register)
        )
        await self._registration.wait()
        self._ws, self._transport = ws, request.transport
        server_info = models.ServerInfo(protocol=16, region=self.region)
        response = agent.RegisterWorkerResponse(worker_id="AW_fake", server_info=server_info)
        await self._send(agent.ServerMessage(register=response))
        async for raw in ws:
            if raw.type == aiohttp.WSMsgType.BINARY:
                await self._receive(agent.WorkerMessage.FromString(raw.data))
        return ws

    async def _receive(self, msg: agent.WorkerMessage) -> None:
        kind = msg.WhichOneof("message")
        if kind == "update_worker":
            self.statuses.append(msg.update_worker)
        elif kind == "update_job":
            self.job_statuses.setdefault(msg.update_job.job_id, []).append(msg.update_job.status)
        elif kind == "migrate_job":
            self.migrations.append(msg.migrate_job)
        elif kind == "ping":
            pong = agent.WorkerPong(
                last_timestamp=msg.ping.timestamp, timestamp=int(time.time() * 1000)
            )
            await self._send(agent.ServerMessage(pong=pong))
        elif kind == "availability":
            answer = msg.availability
            self.answers[answer.job_id] = answer
            if answer.available:
                await self._assign(self._offered[answer.job_id], answer.participant_identity)

    async def _assign(self, job: agent.Job, identity: str) -> None:
        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(identity)
            .with_grants(api.VideoGrants(room_join=True, room=job.room.name, agent=True))
            .to_jwt()
        )
        assignment = agent.JobAssignment(job=job, url="ws://127.0.0.1:9", token=token)
        await self._send(agent.ServerMessage(assignment=assignment))

    async def _send(self, msg: agent.ServerMessage) -> None:
        assert self._ws is not None, "no worker connected"
        await self._ws.send_bytes(msg.SerializeToString())
