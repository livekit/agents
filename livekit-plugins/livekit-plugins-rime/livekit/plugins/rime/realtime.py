# SPDX-License-Identifier: Apache-2.0
"""Prism speech-to-speech through the Rime SDK.

LiveKit history, generations and playback are adapted here. The SDK owns all
wire protocol and admission rules. Adapted from sglang-omni's reference plugin
at bc8a500249536f369d20bc094baa9169e5986758.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from collections.abc import Coroutine
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from rimelabs_sdk import Rime, RimeInputError, realtime as r

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given


class PrismUnsupported(llm.RealtimeError):
    """An operation Prism cannot express. Also emitted as a fatal model error."""


class RealtimeModel(llm.RealtimeModel):
    """Prism for one AgentSession. Requires the Rime SDK realtime extra and Python 3.11.

    Supply the full WebSocket endpoint. An injected Rime client stays caller-owned.
    Otherwise Rime reads RIME_API_KEY. Session configuration is immutable.
    A closed session may be replaced with a fresh conversation, such as an
    explicit fallback restart. Concurrent sessions and agent handoff are unsupported.
    """

    def __init__(self, *, endpoint: str, client: Rime | None = None, voice: str | None = None):
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            )
        )
        self._endpoint = endpoint
        self._voice = voice
        self._client = client if client is not None else Rime()
        self._owns_client = client is None
        self._session: RealtimeSession | None = None

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
        self._typed: str | None = None
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

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext(self._tool_list)

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
            raise self._unsupported(
                "Prism configuration cannot be replaced; handoff is unsupported"
            )
        self._started = True
        self._instructions = instructions if is_given(instructions) else self._instructions
        self._tool_list = list(tools) if is_given(tools) else self._tool_list
        try:
            seed = _history(chat_ctx) if is_given(chat_ctx) else []
            self._connection = self._model._client.realtime.connect(
                endpoint=self._model._endpoint,
                voice=self._model._voice,
                # The SDK treats omitted instructions as the deployment default
                # and rejects blank strings. Preserve LiveKit's value locally.
                instructions=self._instructions
                if self._instructions and self._instructions.strip()
                else None,
                tools=[_tool(t) for t in self._tool_list],
            )
            self._sdk = await self._connection.__aenter__()
            if not self._sdk.info.interrupt_on_speech:
                raise PrismUnsupported("Prism must allow caller speech to interrupt")
            self._spawn(self._read_events())
            for message in seed:
                await self._sdk.add_message(
                    cast(Literal["user", "assistant"], message.role), message.text_content or ""
                )
                self._chat.items.append(message.model_copy())
            self._initialized.set_result(None)
            self._spawn(self._send_audio())
        except Exception as error:
            self._fatal(error)
            await self.aclose()
            raise

    async def update_instructions(self, instructions: str) -> None:
        if not self._started:
            self._instructions = instructions
        elif instructions != self._instructions:
            raise self._unsupported("Prism instructions cannot change during a session")

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        if not self._started:
            self._tool_list = list(tools)
        elif [_tool(t) for t in tools] != [_tool(t) for t in self._tool_list]:
            raise self._unsupported("Prism tools cannot change during a session")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if is_given(tool_choice) and tool_choice not in (None, "auto"):
            raise self._unsupported("Prism cannot force or disable tools for a reply")

    async def set_input_text(
        self, text: str, *, message_id: NotGivenOr[str] = NOT_GIVEN
    ) -> llm.ChatMessage:
        self._require()
        if self._typed is not None:
            raise self._unsupported("A typed turn is already staged")
        if not text.strip() or len(text) > 4000:
            raise self._unsupported("Typed input must be nonblank and at most 4000 characters")
        self._typed = text
        message = self._chat.add_message(role="user", content=text, id=message_id)
        return message.model_copy()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        sdk = self._require()
        known = {item.id: item for item in self._chat.items}
        supplied = {item.id: item for item in chat_ctx.items}
        # Assistant playback may replace the local transcript. The wire update is
        # reported separately with playback_finished; it is not a history edit.
        for key, old in known.items():
            if key not in supplied:
                raise self._unsupported("Prism history items cannot be removed")
            new = supplied[key]
            if old.type == "message" and old.role == "user" and new != old:
                raise self._unsupported("Prism user history cannot be edited")
        additions: list[
            tuple[llm.ChatMessage | llm.FunctionCallOutput, r.ToolCallRef | None, str]
        ] = []
        # Check the complete update before sending any part of it. A later
        # unsupported item must not leave earlier additions on the server.
        for item in chat_ctx.items:
            if item.id in known:
                continue
            if isinstance(item, llm.FunctionCallOutput):
                entry = self._calls.get(item.call_id)
                if entry is None or entry[1].extra.get("__livekit_agents_tool_non_blocking"):
                    raise self._unsupported(
                        "Unknown tool call or progress tool; Prism needs a final result"
                    )
                ref, _ = entry
                text = item.output
                if item.is_error:
                    text = json.dumps(
                        {
                            "status": "failed",
                            "reason": "tool_error",
                            "retryable": False,
                            "message": text,
                            "result": None,
                        }
                    )
                additions.append((item.model_copy(), ref, text))
            elif (
                isinstance(item, llm.ChatMessage)
                and item.role in ("user", "assistant")
                and all(isinstance(c, str) for c in item.content)
            ):
                text = item.text_content or ""
                # The SDK validates each add_message call separately. Validate
                # its text limit here so invalid batches send no history.
                if not text.strip() or len(text) > 4000:
                    raise RimeInputError("Text must be nonblank and at most 4000 characters")
                additions.append((item.model_copy(), None, text))
            else:
                raise self._unsupported(
                    "Prism accepts only text history and results of its own tool calls"
                )

        for addition, tool_ref, text in additions:
            if isinstance(addition, llm.FunctionCallOutput):
                assert tool_ref is not None
                try:
                    await sdk.submit_tool_result(tool_ref, text)
                except Exception as error:
                    self._fatal(error)
                    raise
                self._chat.items.append(addition)
                if addition.reply_required:
                    if self._latest and tool_ref.response_id == self._latest.response_id:
                        self._parent = self._latest
                    else:
                        self._report = tool_ref
            else:
                await sdk.add_message(cast(Literal["user", "assistant"], addition.role), text)
                self._chat.items.append(addition)

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
        instruction = instructions if is_given(instructions) and instructions else None
        if typed is not None and instruction is not None:
            raise self._unsupported("Prism cannot combine typed input with a reply instruction")
        parent, self._parent = self._parent, None
        report, self._report = self._report, None
        return self._spawn(self._reply(typed, instruction, parent, report))

    async def _reply(
        self,
        typed: str | None,
        instruction: str | None,
        parent: r.ResponseRef | None,
        report: r.ToolCallRef | None,
    ) -> llm.GenerationCreatedEvent:
        await asyncio.shield(self._initialized)
        sdk = self._require()
        ref: r.ResponseRef | None = None
        try:
            if typed is not None:
                ref = await sdk.send_text(typed)
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
            return self._generations[ref.response_id].event
        except asyncio.CancelledError:
            if ref is not None and not self._closed:
                self._spawn(self._abandon_reply(ref))
            raise
        except r.RealtimeAdmissionTimeout as error:
            raise llm.RealtimeError(str(error), code="admission_timeout") from error
        except r.RimeRealtimeError as error:
            raise llm.RealtimeError(str(error), code=error.fault.code) from error

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
            await self._finish_playback(response, 0, True)

    def interrupt(self) -> None:
        if (
            not self._closed
            and self._failure is None
            and self._active is not None
            and not self._caller_speaking
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
        if generation is None:
            return
        if message_id is not None and message_id not in generation.outputs:
            return
        if interrupted and interrupted_by_user:
            outputs = (
                [generation.outputs[message_id]]
                if message_id is not None
                else list(generation.outputs.values())
            )
            if message_id is None:
                generation.skipped_by_user = True
            for message in outputs:
                message.discarded = True
                message.text.close()
                message.audio.close()
                if is_given(audio_transcript) or generation.skipped_by_user or audio_end_ms == 0:
                    message.played_transcript = (
                        audio_transcript if is_given(audio_transcript) else ""
                    )
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
        else:
            self._spawn(self._finish_playback(generation.ref, audio_end_ms, interrupted))

    async def _finish_playback(
        self, response: r.ResponseRef, played_ms: int, interrupted: bool
    ) -> None:
        if interrupted:
            await self._cancel(response)
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
        if isinstance(payload, r.SpeechStarted):
            self._caller_speaking = True
            self._latest = None
            self._parent = None
            self._turn_started[payload.item_id] = time.time()
            while len(self._turn_started) > 128:
                self._turn_started.popitem(last=False)
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        elif isinstance(payload, r.SpeechStopped):
            self._caller_speaking = False
            self.emit(
                "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True)
            )
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
        elif isinstance(payload, r.TranscriptFinal):
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
            self._turn_started.pop(payload.item_id, None)
        elif isinstance(payload, r.ResponseStarted):
            messages = utils.aio.Chan[llm.MessageGeneration](maxsize=128)
            functions = utils.aio.Chan[llm.FunctionCall](maxsize=128)
            created = llm.GenerationCreatedEvent(
                message_stream=messages,
                function_stream=functions,
                user_initiated=event.request_id is not None,
                response_id=payload.response.response_id,
            )
            self._generations[payload.response.response_id] = _Generation(
                payload.response, created, messages, functions
            )
            while len(self._generations) > 128:
                self._generations.popitem(last=False)
            self._latest = self._active = payload.response
            self._parent = self._report = None
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
            if generation.skipped_by_user:
                message.discarded = True
                message.played_transcript = ""
                message.text.close()
                message.audio.close()
                self._spawn(
                    self._report_playback(
                        r.PlaybackInterrupted(output=payload.output, audio_end_ms=0)
                    )
                )
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
            self._emit_error(r.RimeRealtimeError(payload.error), recoverable=True)

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

    def _unsupported(self, message: str) -> PrismUnsupported:
        error = PrismUnsupported(message)
        self._fatal(error)
        return error

    def _emit_error(self, error: Exception, *, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(), label=self._model.label, error=error, recoverable=recoverable
            ),
        )

    def _fatal(self, error: Exception) -> None:
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

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            if isinstance(error, Exception) and not isinstance(error, llm.RealtimeError):
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


def _tool(tool: llm.Tool) -> r.ToolDefinition:
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
    )


def _history(ctx: llm.ChatContext) -> list[llm.ChatMessage]:
    messages = []
    for item in ctx.items:
        if isinstance(item, llm.AgentConfigUpdate):
            # Framework bookkeeping is not conversation history. initialize()
            # supplies the current instructions and tools for the new session.
            continue
        if (
            isinstance(item, llm.ChatMessage)
            and item.role in ("user", "assistant")
            and all(isinstance(c, str) for c in item.content)
            and item.text_content
        ):
            messages.append(item)
        else:
            raise PrismUnsupported(
                "Initial Prism history must contain user and assistant text only"
            )
    return messages
