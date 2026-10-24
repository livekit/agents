# SPDX-License-Identifier: Apache-2.0
"""Prism speech-to-speech through the Rime SDK.

LiveKit history, generations and playback are adapted here. The SDK owns all
wire protocol and admission rules. Adapted from sglang-omni's reference plugin
at bc8a500249536f369d20bc094baa9169e5986758.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import OrderedDict
from collections.abc import Coroutine, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from rimelabs_sdk import Rime, RimeInputError, RimeResourceLimitError, realtime as r

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.generation import remove_instructions

from ._realtime_history import HistoryUpdateError, PrismHistory


class PrismUnsupported(llm.RealtimeError):
    """An operation Prism cannot express. Also emitted as a fatal model error."""


class PrismError(llm.RealtimeError):
    """A Prism refusal with its original fault and retry classification."""

    def __init__(self, fault: r.RealtimeFault):
        super().__init__(
            f"{fault.code}: {fault.message}",
            code=fault.code,
            retryable=fault.retryable,
            request_id=fault.request_id,
            response_id=fault.response_id,
            scope=fault.scope,
        )
        self.fault = fault


class RealtimeModel(
    llm.RealtimeModel,
    rtc.EventEmitter[
        Literal["session_updated", "generation_created", "speech_stopped", "speech_hint"]
    ],
):
    """Prism for one AgentSession. Requires the Rime SDK realtime extra and Python 3.11.

    Omit endpoint to use public Prism. An injected Rime client stays caller-owned.
    Otherwise Rime reads RIME_API_KEY. Settings updates wait for acknowledgment.
    A closed session may be replaced with a fresh conversation, such as an
    explicit fallback restart. Concurrent sessions are unsupported.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        client: Rime | None = None,
        voice: str | None = None,
        turn_detection: r.TurnDetection | None = None,
        truncation: r.Truncation | None = None,
        timeouts: r.RealtimeTimeouts | None = None,
        interrupt_on_speech: bool = True,
        request_id: str | None = None,
        response_metadata: Mapping[str, str] | None = None,
        tool_fillers: Mapping[str, r.PrismFiller] | None = None,
        speech_hint_gain: float = 0.25,
        speech_hint_timeout: float = 0.6,
    ):
        if not math.isfinite(speech_hint_gain) or not 0 <= speech_hint_gain <= 1:
            raise ValueError("speech_hint_gain must be between 0 and 1")
        if not math.isfinite(speech_hint_timeout) or speech_hint_timeout <= 0:
            raise ValueError("speech_hint_timeout must be finite and positive")
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
                mutable_chat_context=True,
                mutable_instructions=True,
                mutable_tools=True,
            )
        )
        rtc.EventEmitter.__init__(self)
        self._endpoint = endpoint
        self._voice = voice
        self._turn_detection = turn_detection
        self._truncation = truncation
        self._timeouts = timeouts or r.RealtimeTimeouts()
        self._interrupt_on_speech = interrupt_on_speech
        self._request_id = request_id
        self._response_metadata = dict(response_metadata) if response_metadata is not None else None
        self._tool_fillers = dict(tool_fillers or {})
        self._speech_hint_gain = speech_hint_gain
        self._speech_hint_timeout = speech_hint_timeout
        self._client = client if client is not None else Rime()
        self._owns_client = client is None
        self._session: RealtimeSession | None = None

    @property
    def session_info(self) -> r.SessionInfo | None:
        """Latest acknowledged session information, including Prism's request ID."""
        if self._session is not None and self._session._sdk is not None:
            return self._session._sdk.info
        return None

    async def update_session(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        interrupt_on_speech: NotGivenOr[bool] = NOT_GIVEN,
        response_metadata: NotGivenOr[Mapping[str, str]] = NOT_GIVEN,
        tool_fillers: NotGivenOr[Mapping[str, r.PrismFiller]] = NOT_GIVEN,
        truncation: NotGivenOr[r.Truncation] = NOT_GIVEN,
    ) -> None:
        """Update provider settings on the live session; return after acknowledgment."""
        if self._session is None:
            raise llm.RealtimeError("Prism session is not open")
        await self._session.update_session(
            voice=voice,
            interrupt_on_speech=interrupt_on_speech,
            response_metadata=response_metadata,
            tool_fillers=tool_fillers,
            truncation=truncation,
        )

    async def get_history(self) -> tuple[r.HistoryItem, ...]:
        """Read completed server history. Running replies are not included."""
        if self._session is None:
            raise llm.RealtimeError("Prism session is not open")
        return await self._session.get_history()

    @property
    def model(self) -> str:
        return "prism"

    @property
    def provider(self) -> str:
        return "rime"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        session = RealtimeSession(self)
        if turn_detection_disabled:
            session._failure = PrismUnsupported("Prism requires server turn detection")
        elif self._session is not None and (
            self._session._shutdown_task is None or not self._session._shutdown_task.done()
        ):
            session._failure = PrismUnsupported(
                "Close the previous Prism session before opening a fresh conversation"
            )
        else:
            self._session = session
        return session

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.aclose()
        if self._owns_client:
            await self._client.close()


@dataclass(frozen=True)
class _TypedTurn:
    text: str
    item_id: str
    revision: int


@dataclass
class _Message:
    output: r.OutputRef
    text: utils.aio.Chan[str] = field(default_factory=lambda: utils.aio.Chan(maxsize=256))
    audio: utils.aio.Chan[rtc.AudioFrame] = field(
        default_factory=lambda: utils.aio.Chan(maxsize=256)
    )
    transcript: str = ""
    played_transcript: str | None = None
    discarded: bool = False
    playback_ms: int | None = None


@dataclass
class _Generation:
    ref: r.ResponseRef
    event: llm.GenerationCreatedEvent
    messages: utils.aio.Chan[llm.MessageGeneration]
    functions: utils.aio.Chan[llm.FunctionCall]
    outputs: dict[str, _Message] = field(default_factory=dict)
    ended: bool = False
    skipped_by_user: bool = False
    abandoned: bool = False
    interrupt_on_speech: bool = True
    playback_done: bool = False
    response_playback_ms: int | None = None
    response_stopped: bool = False
    caller_interrupted: bool = False
    cancel_requested: bool = False


class RealtimeSession(llm.RealtimeSession):
    def __init__(self, model: RealtimeModel):
        super().__init__(model)
        self._model = model
        self._sdk: r.RealtimeSession | None = None
        self._connection: AbstractAsyncContextManager[r.RealtimeSession] | None = None
        self._initialized: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._initialized.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        self._started = False
        self._closed = False
        self._failure: Exception | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._fatal_emitted = False
        self._instructions: str | None = None
        self._tool_list: list[llm.Tool] = []
        self._chat = llm.ChatContext.empty()
        self._typed: _TypedTurn | None = None
        self._typed_revision = 0
        self._typed_retry = False
        self._generations: OrderedDict[str, _Generation] = OrderedDict()
        self._generation_ready = asyncio.Condition()
        self._calls: OrderedDict[str, tuple[r.ToolCallRef, llm.FunctionCall]] = OrderedDict()
        self._turn_started: OrderedDict[str, float] = OrderedDict()
        self._latest: r.ResponseRef | None = None
        self._active: r.ResponseRef | None = None
        self._parent: r.ResponseRef | None = None
        self._report: r.ToolCallRef | None = None
        self._caller_speaking = False
        self._audio: asyncio.Queue[r.AudioChunk] = asyncio.Queue(100)
        self._audio_format: tuple[int, int] | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._settings_lock = asyncio.Lock()
        self._tool_fillers = dict(model._tool_fillers)
        self._interruption_policies: OrderedDict[int, bool] = OrderedDict()
        self._transcript_deltas: dict[str, str] = {}
        self._history: PrismHistory | None = None
        self._history_lock = asyncio.Lock()

    def _sdk_tools(
        self, tools: list[llm.Tool], fillers: Mapping[str, r.PrismFiller]
    ) -> list[r.ToolDefinition]:
        definitions = [_tool(tool, fillers) for tool in tools]
        unknown = fillers.keys() - {tool.name for tool in definitions}
        if unknown:
            raise RimeInputError(f"Tool fillers name unregistered tools: {sorted(unknown)}")
        return definitions

    def _record_settings(self) -> None:
        info = self._require().info
        if info.settings_revision is not None:
            # Retain policy changes, not every settings revision. Voice, labels,
            # and truncation updates must not evict an older reply's policy.
            previous = next(reversed(self._interruption_policies.values()), None)
            if previous is None or previous != info.interrupt_on_speech:
                self._interruption_policies[info.settings_revision] = info.interrupt_on_speech
                while len(self._interruption_policies) > 128:
                    self._interruption_policies.popitem(last=False)
        self._model.emit("session_updated", info)

    def _response_policy(self, revision: int | None) -> bool:
        if revision is not None:
            for first, policy in reversed(self._interruption_policies.items()):
                if first <= revision:
                    # A newer revision may already be acknowledged by the SDK
                    # before the update coroutine records it here.
                    info = self._require().info
                    if info.settings_revision is not None and info.settings_revision <= revision:
                        return info.interrupt_on_speech
                    return policy
        return self._require().info.interrupt_on_speech

    def _playback_interrupts(self) -> bool:
        for generation in reversed(self._generations.values()):
            if not generation.playback_done:
                return generation.interrupt_on_speech
        return self._require().info.interrupt_on_speech

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext(self._tool_list)

    def can_reuse(self, chat_ctx: llm.ChatContext) -> bool:
        # LiveKit filters tools absent from the receiving agent. Keep Prism's
        # original call references, including a tool awaiting an inline task.
        def visible(ctx: llm.ChatContext) -> llm.ChatContext:
            projected = ctx.copy(
                exclude_handoff=True,
                exclude_config_update=True,
                exclude_function_call=True,
            )
            remove_instructions(projected)
            return projected

        return visible(self._chat).is_equivalent(visible(chat_ctx))

    def _require(self) -> r.RealtimeSession:
        if self._failure:
            raise self._failure
        if self._sdk is None or self._closed:
            raise self._unsupported("Prism session is not open")
        return self._sdk

    async def initialize(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        if self._failure:
            self._fatal(self._failure)
            raise self._failure
        if self._started:
            if is_given(chat_ctx):
                if self.can_reuse(chat_ctx):
                    # Handoff filters tools absent from the next agent. Keep their records.
                    preserved = [
                        i
                        for i in self._chat.items
                        if isinstance(i, (llm.FunctionCall, llm.FunctionCallOutput))
                    ]
                    chat_ctx = chat_ctx.copy()
                    ids = {i.id for i in chat_ctx.items}
                    chat_ctx.insert([i for i in preserved if i.id not in ids])
                await self.update_chat_ctx(chat_ctx)
            await self._apply_settings(instructions=instructions, tools=tools)
            return
        self._started = True
        self._instructions = instructions if is_given(instructions) else self._instructions
        self._tool_list = list(tools) if is_given(tools) else self._tool_list
        try:
            seed = chat_ctx if is_given(chat_ctx) else llm.ChatContext.empty()
            PrismHistory.validate(seed)
            self._connection = self._model._client.realtime.connect(
                endpoint=self._model._endpoint,
                voice=self._model._voice,
                # The SDK treats omitted instructions as the deployment default
                # and rejects blank strings. Preserve LiveKit's value locally.
                instructions=self._instructions
                if self._instructions and self._instructions.strip()
                else None,
                tools=self._sdk_tools(self._tool_list, self._tool_fillers),
                turn_detection=self._model._turn_detection,
                truncation=self._model._truncation,
                timeouts=self._model._timeouts,
                interrupt_on_speech=self._model._interrupt_on_speech,
                request_id=self._model._request_id,
                response_metadata=self._model._response_metadata,
            )
            self._sdk = await self._connection.__aenter__()
            self._record_settings()
            self._spawn(self._read_events())
            self._history = PrismHistory(self._sdk, self._chat, self._calls)
            await self._history.sync(seed)
            self._initialized.set_result(None)
            self._spawn(self._send_audio())
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception as error:
            self._fatal(error)
            await self.aclose()
            raise

    async def update_instructions(self, instructions: str) -> None:
        if not self._started:
            self._instructions = instructions
        else:
            await self._apply_settings(instructions=instructions)

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        if not self._started:
            self._tool_list = list(tools)
        else:
            await self._apply_settings(tools=tools)

    async def update_session(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        interrupt_on_speech: NotGivenOr[bool] = NOT_GIVEN,
        response_metadata: NotGivenOr[Mapping[str, str]] = NOT_GIVEN,
        tool_fillers: NotGivenOr[Mapping[str, r.PrismFiller]] = NOT_GIVEN,
        truncation: NotGivenOr[r.Truncation] = NOT_GIVEN,
    ) -> None:
        await self._apply_settings(
            voice=voice,
            interrupt_on_speech=interrupt_on_speech,
            response_metadata=response_metadata,
            tool_fillers=tool_fillers,
            truncation=truncation,
        )

    async def _apply_settings(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        interrupt_on_speech: NotGivenOr[bool] = NOT_GIVEN,
        response_metadata: NotGivenOr[Mapping[str, str]] = NOT_GIVEN,
        tool_fillers: NotGivenOr[Mapping[str, r.PrismFiller]] = NOT_GIVEN,
        truncation: NotGivenOr[r.Truncation] = NOT_GIVEN,
    ) -> None:
        # A cancelled caller still waits for acknowledgment so local settings cannot
        # contradict an update the server accepted. No update is replayed.
        async def apply() -> None:
            async with self._settings_lock:
                sdk = self._require()
                fillers = dict(tool_fillers) if is_given(tool_fillers) else self._tool_fillers
                new_tools = list(tools) if is_given(tools) else self._tool_list
                if is_given(tools) and not is_given(tool_fillers):
                    names = {_tool(tool).name for tool in new_tools}
                    fillers = {name: filler for name, filler in fillers.items() if name in names}
                definitions = self._sdk_tools(new_tools, fillers)
                try:
                    if any(
                        is_given(v)
                        for v in (
                            instructions,
                            tools,
                            voice,
                            interrupt_on_speech,
                            tool_fillers,
                            truncation,
                        )
                    ):
                        await sdk.update(
                            truncation=cast(r.Truncation, truncation)
                            if is_given(truncation)
                            else None,
                            instructions=instructions if is_given(instructions) else None,
                            tools=definitions
                            if is_given(tools) or is_given(tool_fillers)
                            else None,
                            voice=voice if is_given(voice) else None,
                            interrupt_on_speech=interrupt_on_speech
                            if is_given(interrupt_on_speech)
                            else None,
                        )
                        if is_given(instructions):
                            self._instructions = instructions
                        self._tool_list = new_tools
                        self._tool_fillers = fillers
                        self._model._tool_fillers = dict(fillers)
                        if is_given(voice):
                            self._model._voice = voice
                        if is_given(interrupt_on_speech):
                            self._model._interrupt_on_speech = interrupt_on_speech
                        if is_given(truncation):
                            self._model._truncation = cast(r.Truncation, truncation)
                        self._record_settings()
                    if is_given(response_metadata):
                        await sdk.set_response_metadata(dict(response_metadata))
                        self._model._response_metadata = dict(response_metadata)
                        self._record_settings()
                except r.RimeRealtimeError as error:
                    raise PrismError(error.fault) from error

        task = asyncio.create_task(apply())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if self._closed:
                    raise
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if is_given(tool_choice) and tool_choice not in (None, "auto"):
            raise self._unsupported("Prism cannot force or disable tools for a reply")

    async def set_input_text(
        self, text: str, *, message_id: NotGivenOr[str] = NOT_GIVEN
    ) -> llm.ChatMessage:
        self._require()
        if self._typed is not None and not self._typed_retry:
            raise self._unsupported("A typed turn is already staged")
        if not text.strip() or len(text) > 4000:
            raise self._unsupported("Typed input must be nonblank and at most 4000 characters")
        if self._typed is not None:
            self._forget_unsent_turn(self._typed)
        message = self._chat.add_message(role="user", content=text, id=message_id)
        self._typed_revision += 1
        self._typed = _TypedTurn(text, message.id, self._typed_revision)
        self._typed_retry = False
        return message.model_copy()

    def _forget_unsent_turn(self, turn: _TypedTurn) -> None:
        self._chat.items[:] = [item for item in self._chat.items if item.id != turn.item_id]

    def _restore_unsent_turn(self, turn: _TypedTurn | None) -> None:
        if turn is None or self._closed or self._failure is not None:
            return
        if self._typed_revision == turn.revision and self._typed is None:
            # No automatic retry. Only a later explicit generate_reply can use it.
            self._typed = turn
            self._typed_retry = True
        elif self._typed_revision != turn.revision:
            # A newer explicit user turn supersedes this known-unsent question.
            self._forget_unsent_turn(turn)

    async def get_history(self) -> tuple[r.HistoryItem, ...]:
        """Read Prism's completed history without changing LiveKit's conversation record."""
        async with self._history_lock:
            try:
                return await self._require().get_history()
            except r.RimeRealtimeError as error:
                raise PrismError(error.fault) from error

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async def apply() -> None:
            async with self._history_lock:
                self._require()
                assert self._history is not None
                try:
                    submitted = await self._history.sync(chat_ctx)
                    for ref, output in submitted:
                        if output.reply_required:
                            if self._latest and ref.response_id == self._latest.response_id:
                                self._parent = self._latest
                            else:
                                self._report = ref
                except HistoryUpdateError as error:
                    if error.partial:
                        self._fatal(error)
                    raise
                except r.RimeRealtimeError as error:
                    raise PrismError(error.fault) from error
                except Exception as error:
                    if not isinstance(error, RimeInputError):
                        self._fatal(error)
                    raise

        task = asyncio.create_task(apply())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if self._closed:
                    raise
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._require()
        try:
            source_format = (frame.sample_rate, frame.num_channels)
            if source_format != self._audio_format:
                self._audio_format = source_format
                self._audio_resampler = None

            # Leave SDK-supported input unchanged. Convert only formats that its
            # PCMFormat cannot represent, such as a 44.1 kHz custom audio input.
            if frame.num_channels > 2:
                samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
                mono = (samples.sum(axis=1, dtype=np.int32) // frame.num_channels).astype(np.int16)
                frame = rtc.AudioFrame(
                    data=mono.tobytes(),
                    sample_rate=frame.sample_rate,
                    num_channels=1,
                    samples_per_channel=frame.samples_per_channel,
                )
            if frame.sample_rate not in (8000, 16000, 24000, 48000):
                if self._audio_resampler is None:
                    self._audio_resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=16000,
                        num_channels=frame.num_channels,
                    )
                frames = self._audio_resampler.push(frame)
            else:
                frames = [frame]
            for converted in frames:
                self._audio.put_nowait(
                    r.AudioChunk(
                        data=bytes(converted.data),
                        format=r.PCMFormat(
                            sample_rate=cast(
                                Literal[8000, 16000, 24000, 48000], converted.sample_rate
                            ),
                            channels=cast(Literal[1, 2], converted.num_channels),
                        ),
                    )
                )
        except Exception as error:
            self._fatal(error)

    async def _send_audio(self) -> None:
        sdk = self._require()
        while True:
            await sdk.send_audio(await self._audio.get())

    def push_video(self, frame: rtc.VideoFrame) -> None:
        raise self._unsupported("Prism does not accept video")

    def commit_audio(self) -> None:
        raise self._unsupported("Prism commits speech through server turn detection")

    def clear_audio(self) -> None:
        self._audio_resampler = None
        self._audio_format = None
        while not self._audio.empty():
            self._audio.get_nowait()
        # The SDK holds its audio lock across sends and clears, so an in-flight
        # chunk finishes before this clear reaches the server.
        self._spawn(self._require().clear_audio())

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        self.update_options(tool_choice=tool_choice)
        if is_given(tools):
            raise self._unsupported("Prism does not support per-reply tools")
        typed, self._typed = self._typed, None
        self._typed_retry = False
        instruction = instructions if is_given(instructions) and instructions else None
        if typed is not None and instruction is not None:
            raise self._unsupported("Prism cannot combine typed input with a reply instruction")
        parent, self._parent = self._parent, None
        report, self._report = self._report, None
        return self._spawn(self._reply(typed, instruction, parent, report), request=True)

    async def _reply(
        self,
        typed: _TypedTurn | None,
        instruction: str | None,
        parent: r.ResponseRef | None,
        report: r.ToolCallRef | None,
    ) -> llm.GenerationCreatedEvent:
        await asyncio.shield(self._initialized)
        # Wait for an edit already in progress, then release the lock before
        # admission. Readiness can depend on tool results submitted through history.
        async with self._history_lock:
            self._require()
        return await self._reply_unlocked(typed, instruction, parent, report)

    async def _reply_unlocked(
        self,
        typed: _TypedTurn | None,
        instruction: str | None,
        parent: r.ResponseRef | None,
        report: r.ToolCallRef | None,
    ) -> llm.GenerationCreatedEvent:
        await asyncio.shield(self._initialized)
        sdk = self._require()
        ref: r.ResponseRef | None = None
        try:
            if typed is not None:
                ref = await sdk.send_text(typed.text)
            elif instruction is not None:
                ref = await sdk.request_reply(instruction=instruction)
            elif parent is not None:
                try:
                    ref = await sdk.continue_reply(parent)
                except (RimeInputError, r.RimeRealtimeError) as error:
                    if isinstance(error, r.RimeRealtimeError) and (
                        error.fault.scope != "event"
                        or error.fault.code != "tool_continuation_unavailable"
                    ):
                        raise
                    # A new user turn can supersede the round between result acknowledgment
                    # and this call, or the server's tool wait can expire. Reporting a
                    # recorded result is a separate operation and never reruns the tool.
                    call = next(
                        (
                            c
                            for c, _ in reversed(self._calls.values())
                            if c.response_id == parent.response_id
                        ),
                        None,
                    )
                    if call is None:
                        raise
                    ref = await sdk.request_reply(tool_call=call)
            else:
                ref = await sdk.request_reply(tool_call=report)
            async with self._generation_ready:
                await self._generation_ready.wait_for(
                    lambda: ref.response_id in self._generations or self._failure is not None
                )
            self._require()
            generation = self._generations[ref.response_id].event
            if typed is not None and self._history is not None:
                self._history.bind_typed(typed.item_id, generation.input_item_ids)
            return generation
        except asyncio.CancelledError:
            if ref is not None and not self._closed:
                self._spawn(self._abandon_reply(ref))
            raise
        except r.RealtimeAdmissionTimeout as error:
            if ref is None:
                self._restore_unsent_turn(typed)
            raise llm.RealtimeError(str(error), code="admission_timeout") from error
        except r.RimeRealtimeError as error:
            fault = error.fault
            if (
                ref is None
                and fault.scope == "event"
                and fault.item_id is None
                and fault.response_id is None
                and fault.code not in ("typed_turn_superseded", "response_superseded")
            ):
                self._restore_unsent_turn(typed)
            raise PrismError(fault) from error

    async def _abandon_reply(self, response: r.ResponseRef) -> None:
        async with self._generation_ready:
            await self._generation_ready.wait_for(
                lambda: response.response_id in self._generations or self._failure is not None
            )
        if self._closed or self._failure is not None:
            return
        generation = self._generations[response.response_id]
        if not generation.abandoned:
            generation.abandoned = True
            self.playback_finished(
                response_id=response.response_id,
                message_id=None,
                audio_end_ms=0,
                interrupted=True,
                interrupted_by_user=False,
            )

    def interrupt(self) -> None:
        if (
            not self._closed
            and self._failure is None
            and self._active is not None
            and not (self._caller_speaking and self._playback_interrupts())
        ):
            self._spawn(self._cancel(self._active))

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        generation = next((g for g in self._generations.values() if message_id in g.outputs), None)
        if generation is not None:
            self.playback_finished(
                response_id=generation.ref.response_id,
                message_id=message_id,
                audio_end_ms=audio_end_ms,
                interrupted=True,
                interrupted_by_user=self._caller_speaking,
            )

    def playback_finished(
        self,
        *,
        response_id: str | None,
        message_id: str | None,
        audio_end_ms: int,
        interrupted: bool,
        interrupted_by_user: bool,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
        modalities: list[Literal["text", "audio"]] | None = None,
    ) -> None:
        if self._closed or self._failure is not None or response_id is None:
            return
        generation = self._generations.get(response_id)
        if generation is None or generation.playback_done:
            return
        if message_id is not None:
            message = generation.outputs.get(message_id)
            if message is None or message.playback_ms is not None:
                return
            outputs = [message]
            message.playback_ms = audio_end_ms
        else:
            if generation.response_playback_ms is not None:
                return
            generation.response_playback_ms = audio_end_ms
            generation.response_stopped = interrupted
            outputs = [m for m in generation.outputs.values() if m.playback_ms is None]
            for message in outputs:
                message.playback_ms = 0
                if interrupted:
                    # A response-wide stop has no remaining player, including
                    # outputs announced before this callback reached the adapter.
                    message.discarded = True
                    message.played_transcript = ""
                    message.text.close()
                    message.audio.close()
        if interrupted and interrupted_by_user:
            generation.caller_interrupted = True
            generation.response_stopped = True
            generation.skipped_by_user = True
            for message in outputs:
                self._interrupt_output(
                    generation, message, audio_end_ms, audio_transcript=audio_transcript
                )
            if message_id is not None:
                # Caller speech stops the response's remaining outputs too. They
                # may already be announced but have no player after interruption.
                after_interrupted = False
                for message in generation.outputs.values():
                    if message.output.item_id == message_id:
                        after_interrupted = True
                    elif after_interrupted and message.playback_ms is None:
                        message.playback_ms = 0
                        self._interrupt_output(generation, message, 0)
        elif interrupted:
            generation.response_stopped = True
            if message_id is not None:
                after_interrupted = False
                for message in generation.outputs.values():
                    if message.output.item_id == message_id:
                        after_interrupted = True
                    elif after_interrupted and message.playback_ms is None:
                        message.playback_ms = 0
                        message.discarded = True
                        message.played_transcript = ""
                        message.text.close()
                        message.audio.close()
            if not generation.cancel_requested:
                generation.cancel_requested = True
                self._spawn(self._cancel(generation.ref))
        self._maybe_finish_playback(generation)

    def _interrupt_output(
        self,
        generation: _Generation,
        message: _Message,
        audio_end_ms: int,
        *,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        message.discarded = True
        message.text.close()
        message.audio.close()
        if is_given(audio_transcript) or audio_end_ms == 0:
            message.played_transcript = audio_transcript if is_given(audio_transcript) else ""
            if generation.ended:
                self._chat._upsert_item(
                    llm.ChatMessage(
                        id=message.output.item_id,
                        role="assistant",
                        content=[message.played_transcript],
                    )
                )
        self._spawn(
            self._report_playback(
                r.PlaybackInterrupted(output=message.output, audio_end_ms=audio_end_ms)
            )
        )

    def _maybe_finish_playback(self, generation: _Generation) -> None:
        if generation.playback_done or not generation.ended:
            return
        if not generation.outputs and generation.response_playback_ms is None:
            return
        if any(message.playback_ms is None for message in generation.outputs.values()):
            return
        generation.playback_done = True
        if generation.caller_interrupted:
            # Caller interruption uses per-output truncations, not a response drain.
            return
        if generation.response_playback_ms is not None:
            played_ms = generation.response_playback_ms
        else:
            played_ms = sum(message.playback_ms or 0 for message in generation.outputs.values())
        self._spawn(self._finish_playback(generation.ref, played_ms))

    async def _finish_playback(self, response: r.ResponseRef, played_ms: int) -> None:
        await self._report_playback(r.PlaybackFinished(response=response, played_ms=played_ms))

    async def _cancel(self, response: r.ResponseRef) -> None:
        if self._closed or self._failure is not None:
            return
        try:
            await self._require().cancel(response)
        except r.RimeRealtimeError as error:
            if error.fault.scope != "event" or error.fault.code != "response_cancel_not_active":
                raise
            # The response ended before cancellation reached the server. Playback
            # completion still needs its receipt once response.done arrives.
            self._emit_error(error, recoverable=True)

    async def _report_playback(self, report: r.PlaybackReport) -> None:
        if self._closed or self._failure is not None:
            return
        try:
            await self._require().report_playback(report)
        except r.RimeRealtimeError as error:
            if (
                not isinstance(report, r.PlaybackInterrupted)
                or error.fault.scope != "event"
                or error.fault.code != "truncate_not_current"
            ):
                raise
            # A newer response owns the playback window; this report changes nothing.
            self._emit_error(error, recoverable=True)

    async def _read_events(self) -> None:
        async for event in self._require().events:
            self._on_event(event)
            async with self._generation_ready:
                self._generation_ready.notify_all()
        if not self._closed:
            raise ConnectionError("Prism session ended")

    def _on_event(self, event: r.SessionEvent) -> None:
        payload = event.payload
        if self._history is not None:
            self._history.observe(payload)
        if isinstance(payload, r.SpeechHint):
            if self._playback_interrupts():
                hint = llm.InputSpeechHintEvent(
                    audio_start_ms=payload.audio_start_ms,
                    gain=self._model._speech_hint_gain,
                    timeout=self._model._speech_hint_timeout,
                )
                self.emit("input_speech_hint", hint)
                self._model.emit("speech_hint", hint)
        elif isinstance(payload, r.SpeechStarted):
            interrupt_response = self._playback_interrupts()
            self._caller_speaking = True
            self._latest = None
            self._parent = None
            self._turn_started[payload.item_id] = time.time()
            while len(self._turn_started) > 128:
                self._turn_started.popitem(last=False)
            self.emit(
                "input_speech_started",
                llm.InputSpeechStartedEvent(interrupt_response=interrupt_response),
            )
        elif isinstance(payload, r.SpeechStopped):
            self._caller_speaking = False
            stopped = llm.InputSpeechStoppedEvent(
                user_transcription_enabled=True, end_reason=payload.end_reason
            )
            self.emit("input_speech_stopped", stopped)
            self._model.emit("speech_stopped", stopped)
        elif isinstance(payload, r.InputCommitted):
            if not self._chat.get_by_id(payload.item_id):
                self._chat.items.append(
                    llm.ChatMessage(
                        id=payload.item_id,
                        role="user",
                        content=[],
                        created_at=self._turn_started.setdefault(payload.item_id, time.time()),
                    )
                )
        elif isinstance(payload, r.TranscriptDelta):
            text = self._transcript_deltas.get(payload.item_id, "") + payload.delta
            if len(text) > 65536 or (
                payload.item_id not in self._transcript_deltas
                and len(self._transcript_deltas) >= 128
            ):
                raise RuntimeError("Prism input transcript exceeds the adapter limit")
            self._transcript_deltas[payload.item_id] = text
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=payload.item_id,
                    transcript=text,
                    is_final=False,
                    turn_started_at=self._turn_started.get(payload.item_id),
                ),
            )
        elif isinstance(payload, r.TranscriptFinal):
            self._transcript_deltas.pop(payload.item_id, None)
            started = self._turn_started.pop(payload.item_id, None)
            self._chat._upsert_item(
                llm.ChatMessage(
                    id=payload.item_id,
                    role="user",
                    content=[payload.text],
                    created_at=started or time.time(),
                )
            )
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=payload.item_id,
                    transcript=payload.text,
                    is_final=True,
                    turn_started_at=started,
                ),
            )
        elif isinstance(payload, r.TranscriptFailed):
            self._transcript_deltas.pop(payload.item_id, None)
            self._turn_started.pop(payload.item_id, None)
        elif isinstance(payload, r.ResponseStarted):
            if len(self._generations) >= 128:
                oldest = next(iter(self._generations.values()))
                if not oldest.playback_done:
                    # The SDK also expires old response references at this limit.
                    # Retaining the record cannot make its eventual receipt valid.
                    raise RimeResourceLimitError(
                        "Prism generation limit reached with unresolved playback"
                    )
                self._generations.popitem(last=False)
            messages = utils.aio.Chan[llm.MessageGeneration](maxsize=128)
            functions = utils.aio.Chan[llm.FunctionCall](maxsize=128)
            created = llm.GenerationCreatedEvent(
                message_stream=messages,
                function_stream=functions,
                user_initiated=event.request_id is not None,
                response_id=payload.response.response_id,
                metadata=dict(payload.metadata),
                input_item_ids=payload.input_item_ids,
                settings_revision=payload.settings_revision,
                provider_request_id=self._require().info.request_id,
            )
            self._generations[payload.response.response_id] = _Generation(
                payload.response,
                created,
                messages,
                functions,
                interrupt_on_speech=self._response_policy(payload.settings_revision),
            )
            self._latest = self._active = payload.response
            self._parent = self._report = None
            self._model.emit("generation_created", created)
            if event.request_id is None:
                self.emit("generation_created", created)
        elif isinstance(payload, r.ResponseAbandoned):
            self._spawn(self._abandon_reply(payload.response))
        elif isinstance(payload, r.MessageStarted):
            generation = self._generations[payload.output.response.response_id]
            if generation.abandoned:
                return
            message = _Message(payload.output)
            generation.outputs[payload.output.item_id] = message
            if generation.response_stopped:
                message.playback_ms = 0
                message.discarded = True
                message.played_transcript = ""
                message.text.close()
                message.audio.close()
                if generation.skipped_by_user:
                    self._interrupt_output(generation, message, 0)
                return
            modalities: asyncio.Future[list[Literal["text", "audio"]]] = (
                asyncio.get_running_loop().create_future()
            )
            modalities.set_result(["text", "audio"])
            generation.messages.send_nowait(
                llm.MessageGeneration(
                    message_id=payload.output.item_id,
                    text_stream=message.text,
                    audio_stream=message.audio,
                    modalities=modalities,
                )
            )
        elif isinstance(payload, (r.TextDelta, r.TextDone, r.AudioDelta, r.AudioDone)):
            generation = self._generations[payload.output.response.response_id]
            if generation.abandoned or generation.skipped_by_user:
                return
            message = generation.outputs[payload.output.item_id]
            if message.discarded:
                return
            if isinstance(payload, r.TextDelta):
                message.transcript += payload.delta
                if len(message.transcript) > 65536:
                    raise RuntimeError("Prism response text exceeds the adapter limit")
                message.text.send_nowait(payload.delta)
            elif isinstance(payload, r.TextDone):
                message.transcript = payload.text
                message.text.close()
            elif isinstance(payload, r.AudioDelta):
                data = payload.audio.data
                if data:
                    message.audio.send_nowait(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=24000,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    )
            else:
                message.audio.close()
        elif isinstance(payload, r.ToolCall):
            if self._generations[payload.call.response_id].abandoned:
                return
            call = llm.FunctionCall(
                id=payload.item_id,
                call_id=payload.call.call_id,
                name=payload.name,
                arguments=json.dumps(payload.arguments),
            )
            self._calls[payload.call.call_id] = (payload.call, call)
            while len(self._calls) > 1024:
                self._calls.popitem(last=False)
            self._chat.items.append(call)
            self._generations[payload.call.response_id].functions.send_nowait(call)
        elif isinstance(payload, r.ResponseEnded):
            self._end(self._generations[payload.response.response_id])
            if self._active == payload.response:
                self._active = None
        elif isinstance(payload, r.FaultEvent) and payload.error.request_id is None:
            self._emit_error(
                PrismError(payload.error), recoverable=payload.error.scope != "session"
            )

    def _end(self, generation: _Generation) -> None:
        if generation.ended:
            return
        generation.ended = True
        for item_id, message in generation.outputs.items():
            transcript = (
                message.transcript
                if message.played_transcript is None
                else message.played_transcript
            )
            if transcript:
                self._chat._upsert_item(
                    llm.ChatMessage(id=item_id, role="assistant", content=[transcript])
                )
            message.text.close()
            message.audio.close()
        generation.messages.close()
        generation.functions.close()
        if self._failure is None and not self._closed:
            self._maybe_finish_playback(generation)

    def _unsupported(self, message: str) -> PrismUnsupported:
        error = PrismUnsupported(message)
        self._fatal(error)
        return error

    def _emit_error(self, error: Exception, *, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._model.label,
                error=error,
                recoverable=recoverable,
                retryable=error.retryable
                if isinstance(error, (llm.RealtimeError,))
                else (error.fault.retryable if isinstance(error, r.RimeRealtimeError) else None),
            ),
        )

    def _fatal(self, error: Exception) -> None:
        if isinstance(error, r.RimeRealtimeError):
            error = PrismError(error.fault)
        if self._failure is None:
            self._failure = error
        self._active = None
        if self._fatal_emitted:
            return
        self._fatal_emitted = True
        self._emit_error(error, recoverable=False)
        if not self._initialized.done():
            self._initialized.set_exception(error)
        if self._sdk is not None and not self._closed:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        for generation in self._generations.values():
            self._end(generation)

    def _spawn(self, coro: Coroutine[Any, Any, Any], *, request: bool = False) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(lambda finished: self._done(finished, request=request))
        return task

    def _done(self, task: asyncio.Task[Any], *, request: bool) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            # The caller receives reply refusals and admission timeouts. An error
            # in an unobserved background operation must also stop the session.
            if isinstance(error, Exception) and not (
                request and isinstance(error, llm.RealtimeError)
            ):
                self._fatal(error)

    async def aclose(self) -> None:
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        self._closed = True
        if not self._initialized.done():
            self._initialized.cancel()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if self._connection is not None:
            await self._connection.__aexit__(None, None, None)
        await asyncio.gather(*tasks, return_exceptions=True)
        for generation in self._generations.values():
            self._end(generation)


def _tool(tool: llm.Tool, fillers: Mapping[str, r.PrismFiller] | None = None) -> r.ToolDefinition:
    if isinstance(tool, llm.FunctionTool):
        function = llm.utils.build_legacy_openai_schema(tool)["function"]
    elif isinstance(tool, llm.RawFunctionTool):
        function = tool.info.raw_schema
    else:
        raise PrismUnsupported("Prism accepts function tools only")
    return r.ToolDefinition(
        name=function["name"],
        description=function.get("description", ""),
        parameters=function.get("parameters") or {"type": "object", "properties": {}},
        prism_filler=(fillers or {}).get(function["name"]),
    )
