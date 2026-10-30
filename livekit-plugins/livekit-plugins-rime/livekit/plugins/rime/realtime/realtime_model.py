# SPDX-License-Identifier: Apache-2.0
"""Prism speech-to-speech through the Rime SDK.

LiveKit history, generations and playback are adapted here. The SDK owns all
wire protocol and admission rules. Adapted from sglang-omni's reference plugin
at bc8a500249536f369d20bc094baa9169e5986758.
"""

from __future__ import annotations

import asyncio
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

from ._history import HistoryUpdateError, PrismHistory


class PrismUnsupported(llm.RealtimeError):
    """An operation Prism cannot express. Also emitted as a fatal model error."""


class PrismError(llm.RealtimeError):
    """A Prism refusal with its original fault and retry classification."""

    def __init__(self, fault: r.RealtimeFault):
        super().__init__(f"{fault.code}: {fault.message}", code=fault.code)
        self.fault = fault
        self.retryable = fault.retryable
        self.request_id = fault.request_id
        self.response_id = fault.response_id
        self.scope = fault.scope


class PrismModelError(llm.RealtimeModelError):
    """A LiveKit model error with Prism's separate retry classification."""

    retryable: bool | None = None
    """Whether the unchanged request may succeed again; does not initiate a retry."""


class RealtimeModel(llm.RealtimeModel):
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
        request_id: str | None = None,
        response_metadata: Mapping[str, str] | None = None,
        tool_fillers: Mapping[str, r.PrismFiller] | None = None,
    ):
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
        self._endpoint = endpoint
        self._voice = voice
        self._turn_detection = turn_detection
        self._truncation = truncation
        self._timeouts = timeouts or r.RealtimeTimeouts()
        self._request_id = request_id
        self._response_metadata = dict(response_metadata) if response_metadata is not None else None
        self._tool_fillers = dict(tool_fillers or {})
        self._client = client if client is not None else Rime()
        self._owns_client = client is None
        self._session: RealtimeSession | None = None

    async def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        response_metadata: NotGivenOr[Mapping[str, str]] = NOT_GIVEN,
        tool_fillers: NotGivenOr[Mapping[str, r.PrismFiller]] = NOT_GIVEN,
        truncation: NotGivenOr[r.Truncation] = NOT_GIVEN,
    ) -> None:
        """Update provider settings on the live session; return after acknowledgment."""
        if self._session is None:
            raise llm.RealtimeError("Prism session is not open")
        await self._session.update_session(
            voice=voice,
            response_metadata=response_metadata,
            tool_fillers=tool_fillers,
            truncation=truncation,
        )

    async def _get_history(self) -> tuple[r.HistoryItem, ...]:
        """Read completed server history. Running replies are not included."""
        if self._session is None:
            raise llm.RealtimeError("Prism session is not open")
        return await self._session._get_history()

    async def _delete_instruction_item(self, item_id: str) -> None:
        """Remove a saved nudge instruction after its reply ends."""
        if self._session is None:
            raise llm.RealtimeError("Prism session is not open")
        await self._session._delete_instruction_item(item_id)

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
    retained_text: str | None = None
    discarded: bool = False


@dataclass
class _Generation:
    ref: r.ResponseRef
    event: llm.GenerationCreatedEvent
    messages: utils.aio.Chan[llm.MessageGeneration]
    functions: utils.aio.Chan[llm.FunctionCall]
    instruction_item_id: str | None = None
    outputs: dict[str, _Message] = field(default_factory=dict)
    ended: bool = False
    abandoned: bool = False
    tools_disabled: bool = False


class RealtimeSession(llm.RealtimeSession[Literal["rime_server_event_received"]]):
    """A Prism session that emits SDK SessionEvent objects before processing them."""

    def __init__(self, model: RealtimeModel):
        super().__init__(model)
        self._model = model
        self._sdk: r.RealtimeSession | None = None
        self._connection: AbstractAsyncContextManager[r.RealtimeSession] | None = None
        self._initialized: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._initialized.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        self._closed = False
        self._failure: Exception | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._fatal_emitted = False
        self._tool_list: list[llm.Tool] = []
        self._generations: OrderedDict[str, _Generation] = OrderedDict()
        self._generation_ready = asyncio.Condition()
        self._turn_started: OrderedDict[str, float] = OrderedDict()
        self._latest: r.ResponseRef | None = None
        self._active: r.ResponseRef | None = None
        self._parent: r.ResponseRef | None = None
        self._report: r.ToolCallRef | None = None
        self._audio: asyncio.Queue[r.AudioChunk] = asyncio.Queue(100)
        self._audio_format: tuple[int, int] | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._settings_lock = asyncio.Lock()
        self._tool_fillers = dict(model._tool_fillers)
        self._tool_choice: Literal["auto", "none"] = "auto"
        self._suppress_tools = False
        self._suppressed_revisions: set[int] = set()
        self._tool_policy_floor = 0
        self._transcript_deltas: dict[str, str] = {}
        self._history: PrismHistory | None = None
        self._instruction_lock = asyncio.Lock()
        self._instruction_items: OrderedDict[str, tuple[str, bool]] = OrderedDict()
        self._spawn(self._connect())

    def _sdk_tools(
        self, tools: list[llm.Tool], fillers: Mapping[str, r.PrismFiller]
    ) -> list[r.ToolDefinition]:
        definitions = [_tool(tool, fillers) for tool in tools]
        unknown = fillers.keys() - {tool.name for tool in definitions}
        if unknown:
            raise RimeInputError(f"Tool fillers name unregistered tools: {sorted(unknown)}")
        return definitions

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._history.chat_ctx if self._history is not None else llm.ChatContext.empty()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext(self._tool_list)

    def _require(self) -> r.RealtimeSession:
        if self._failure:
            raise self._failure
        if self._sdk is None or self._closed:
            raise llm.RealtimeError("Prism session is not open")
        return self._sdk

    async def _connect(self) -> None:
        try:
            if self._failure:
                raise self._failure
            self._connection = self._model._client.realtime.connect(
                endpoint=self._model._endpoint,
                voice=self._model._voice,
                turn_detection=self._model._turn_detection,
                truncation=self._model._truncation,
                timeouts=self._model._timeouts,
                interrupt_on_speech=True,
                request_id=self._model._request_id,
                response_metadata=self._model._response_metadata,
            )
            self._sdk = await self._connection.__aenter__()
            if self._failure is not None:
                # A startup failure can occur while the handshake is pending,
                # before _fatal has an SDK connection to close.
                self._shutdown_task = asyncio.create_task(self._shutdown())
                return
            self._history = PrismHistory(self._sdk)
            self._spawn(self._read_events())
            self._initialized.set_result(None)
            self._spawn(self._send_audio())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fatal(error)

    async def update_instructions(self, instructions: str) -> None:
        await self._apply_settings(instructions=instructions)

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        await self._apply_settings(tools=tools)

    async def update_session(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        response_metadata: NotGivenOr[Mapping[str, str]] = NOT_GIVEN,
        tool_fillers: NotGivenOr[Mapping[str, r.PrismFiller]] = NOT_GIVEN,
        truncation: NotGivenOr[r.Truncation] = NOT_GIVEN,
    ) -> None:
        await self._apply_settings(
            voice=voice,
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
        response_metadata: NotGivenOr[Mapping[str, str]] = NOT_GIVEN,
        tool_fillers: NotGivenOr[Mapping[str, r.PrismFiller]] = NOT_GIVEN,
        truncation: NotGivenOr[r.Truncation] = NOT_GIVEN,
    ) -> None:
        # A cancelled caller still waits for acknowledgment so local settings cannot
        # contradict an update the server accepted. No update is replayed.
        await asyncio.shield(self._initialized)
        self._require()

        async def apply() -> None:
            async with self._settings_lock:
                sdk = self._require()
                fillers = dict(tool_fillers) if is_given(tool_fillers) else self._tool_fillers
                new_tools = list(tools) if is_given(tools) else self._tool_list
                if is_given(tools) and not is_given(tool_fillers):
                    names = {_tool(tool).name for tool in new_tools}
                    fillers = {name: filler for name, filler in fillers.items() if name in names}
                definitions = (
                    self._sdk_tools(new_tools, fillers)
                    if is_given(tools) or is_given(tool_fillers)
                    else None
                )
                try:
                    if any(
                        is_given(v)
                        for v in (
                            instructions,
                            tools,
                            voice,
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
                        )
                        self._tool_list = new_tools
                        self._tool_fillers = fillers
                        self._model._tool_fillers = dict(fillers)
                        if is_given(voice):
                            self._model._voice = voice
                        if is_given(truncation):
                            self._model._truncation = cast(r.Truncation, truncation)
                    if is_given(response_metadata):
                        await sdk.set_response_metadata(dict(response_metadata))
                        self._model._response_metadata = dict(response_metadata)
                except r.RimeRealtimeError as error:
                    raise PrismError(error.fault) from error

        await self._settle_update(apply())

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if not is_given(tool_choice):
            return
        if tool_choice not in (None, "auto", "none"):
            raise PrismUnsupported("Prism supports only tool_choice='auto' or 'none'")
        self._tool_choice = "none" if tool_choice == "none" else "auto"

    async def _get_history(self) -> tuple[r.HistoryItem, ...]:
        """Read Prism's completed history without changing LiveKit's conversation record."""
        await asyncio.shield(self._initialized)
        self._require()
        assert self._history is not None
        try:
            return await self._history.get_history()
        except r.RimeRealtimeError as error:
            raise PrismError(error.fault) from error

    async def _delete_instruction_item(self, item_id: str) -> None:
        """Delete an instruction observed in this session without changing LiveKit history."""
        await asyncio.shield(self._initialized)
        sdk = self._require()
        assert self._history is not None
        async with self._instruction_lock:
            state = self._instruction_items.get(item_id)
            if state is None:
                raise RimeInputError("Unknown saved Prism instruction item")
            if not state[1]:
                raise RimeInputError("Cannot delete a saved instruction before its reply ends")
            await self._history.wait_for_updates()
            try:
                await sdk.delete_item(item_id)
            except r.RimeRealtimeError as error:
                raise PrismError(error.fault) from error
            if self._instruction_items.get(item_id) == state:
                del self._instruction_items[item_id]

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        await asyncio.shield(self._initialized)

        async def apply() -> None:
            self._require()
            assert self._history is not None
            try:
                submitted = await self._history.sync(chat_ctx)
                for ref, output in submitted:
                    if output.reply_required:
                        if (
                            self._latest
                            and ref.response_id == self._latest.response_id
                            and not self._history.has_pending_calls(self._latest)
                        ):
                            self._parent = self._latest
                        else:
                            # SDK continuation waits for every result. A partial
                            # update can still request a report for an accepted result.
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

        await self._settle_update(apply())

    async def _settle_update(self, operation: Coroutine[Any, Any, None]) -> None:
        """Finish acknowledged state changes before propagating caller cancellation.

        Session shutdown still cancels the operation. An operation failure takes
        precedence over caller cancellation so uncertain state is not hidden.
        """
        task = asyncio.create_task(operation)
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
        if self._failure is not None or self._closed:
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
            chunk = await self._audio.get()
            await sdk.send_audio(chunk)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        raise self._unsupported("Prism does not accept video")

    def commit_audio(self) -> None:
        raise self._unsupported("Prism commits speech through server turn detection")

    def clear_audio(self) -> None:
        if self._failure is not None or self._closed:
            self._require()
        self._audio_resampler = None
        self._audio_format = None
        while not self._audio.empty():
            self._audio.get_nowait()
        # Before connection, audio exists only in the local queue. The sender
        # starts after the SDK handshake, so no server clear is needed yet.
        if self._sdk is None:
            return
        # The SDK holds its audio lock across sends and clears, so an in-flight
        # chunk finishes before this clear reaches the server.
        self._spawn(self._require().clear_audio())

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
        metadata: Mapping[str, str] | None = None,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if is_given(tools):
            raise self._unsupported("Prism does not support per-reply tools")
        self.update_options(tool_choice=tool_choice)
        parent, self._parent = self._parent, None
        report, self._report = self._report, None
        return self._spawn(
            self._reply(
                instructions if is_given(instructions) and instructions else None,
                parent,
                report,
                self._tool_choice == "none",
                dict(metadata) if metadata is not None else None,
            ),
            request=True,
        )

    async def _reply(
        self,
        instruction: str | None,
        parent: r.ResponseRef | None,
        report: r.ToolCallRef | None,
        disable_tools: bool,
        metadata: dict[str, str] | None,
    ) -> llm.GenerationCreatedEvent:
        await asyncio.shield(self._initialized)
        # Do not hold the history lock during admission: a pending tool result
        # may need that lock before Prism can admit another reply.
        self._require()
        assert self._history is not None
        await self._history.wait_for_updates()
        try:
            if not disable_tools:
                return await self._request_reply(instruction, parent, report, metadata)
            async with self._settings_lock:
                sdk = self._require()
                restore = self._sdk_tools(self._tool_list, self._tool_fillers)
                self._suppress_tools = True
                try:
                    info = await sdk.update(tools=[])
                    if info.settings_revision is not None:
                        self._suppressed_revisions.add(info.settings_revision)
                        if len(self._suppressed_revisions) > 128:
                            oldest = min(self._suppressed_revisions)
                            self._suppressed_revisions.remove(oldest)
                            self._tool_policy_floor = oldest + 1
                    # Shield admission through response creation. Restoring the tools
                    # before an accepted late response starts changes its settings.
                    task = self._spawn(
                        self._request_reply(instruction, parent, report, metadata), request=True
                    )
                    try:
                        return await asyncio.shield(task)
                    except asyncio.CancelledError:
                        try:
                            created = await asyncio.shield(task)
                            if created.response_id is not None:
                                await self._abandon_reply(
                                    self._generations[created.response_id].ref
                                )
                        finally:
                            raise
                finally:
                    if not self._closed and self._failure is None:
                        try:
                            await sdk.update(tools=restore)
                        except BaseException:
                            self._fatal(RuntimeError("Prism tool restoration did not complete"))
                            raise
                    self._suppress_tools = False
        except r.RimeRealtimeError as error:
            raise PrismError(error.fault) from error
        except r.RealtimeAdmissionTimeout as error:
            raise llm.RealtimeError(str(error), code="admission_timeout") from error

    async def _request_reply(
        self,
        instruction: str | None,
        parent: r.ResponseRef | None,
        report: r.ToolCallRef | None,
        metadata: dict[str, str] | None,
    ) -> llm.GenerationCreatedEvent:
        sdk = self._require()
        assert self._history is not None
        ref: r.ResponseRef | None = None
        try:
            if instruction is not None:
                ref = await sdk.request_reply(instruction=instruction, metadata=metadata)
            elif parent is not None:
                try:
                    ref = await sdk.continue_reply(parent, metadata=metadata)
                except (RimeInputError, r.RimeRealtimeError) as error:
                    if isinstance(error, r.RimeRealtimeError) and (
                        error.fault.scope != "event"
                        or error.fault.code != "tool_continuation_unavailable"
                    ):
                        raise
                    assert self._history is not None
                    call = self._history.last_call(parent)
                    if call is None:
                        raise
                    ref = await sdk.request_reply(tool_call=call, metadata=metadata)
            else:
                ref = await sdk.request_reply(tool_call=report, metadata=metadata)
            async with self._generation_ready:
                await self._generation_ready.wait_for(
                    lambda: ref.response_id in self._generations or self._failure is not None
                )
            self._require()
            return self._generations[ref.response_id].event
        except r.RimeRealtimeError as error:
            raise PrismError(error.fault) from error
        except r.RealtimeAdmissionTimeout as error:
            raise llm.RealtimeError(str(error), code="admission_timeout") from error
        except asyncio.CancelledError:
            if ref is not None and not self._closed:
                self._spawn(self._abandon_reply(ref))
            raise

    async def _abandon_reply(self, response: r.ResponseRef) -> None:
        generation = self._generations.get(response.response_id)
        if generation is not None:
            generation.abandoned = True
            self._end(generation)
        await self._cancel(response)

    def interrupt(self) -> None:
        if not self._closed and self._failure is None and self._active is not None:
            generation = self._generations[self._active.response_id]
            generation.abandoned = True
            self._spawn(self._cancel(self._active))
            self._end(generation)

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if self._closed or self._failure is not None:
            return
        generation = next((g for g in self._generations.values() if message_id in g.outputs), None)
        if generation is None:
            return
        message = generation.outputs[message_id]
        message.discarded = True
        message.text.close()
        message.audio.close()
        if is_given(audio_transcript) or audio_end_ms == 0:
            message.played_transcript = audio_transcript if is_given(audio_transcript) else ""
            assert self._history is not None
            self._history.record_assistant(message_id, message.played_transcript)
        self._spawn(
            self._report_playback(
                r.PlaybackInterrupted(
                    output=message.output,
                    audio_end_ms=audio_end_ms,
                )
            )
        )

    async def _cancel(self, response: r.ResponseRef) -> None:
        if self._closed or self._failure is not None:
            return
        try:
            await self._require().cancel(response)
        except r.RimeRealtimeError as error:
            if error.fault.scope != "event" or error.fault.code != "response_cancel_not_active":
                raise
            # The response ended before cancellation reached the server. Playback
            # the server retains its playback estimate.
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
        self.emit("rime_server_event_received", event)
        payload = event.payload
        assert self._history is not None
        if isinstance(payload, r.InputCommitted):
            self._turn_started.setdefault(payload.item_id, time.time())
        self._history.observe(
            payload,
            turn_started_at=self._turn_started.get(payload.item_id)
            if isinstance(payload, (r.InputCommitted, r.TranscriptFinal))
            else None,
        )
        if isinstance(payload, r.SpeechStarted):
            self._latest = None
            self._parent = None
            self._turn_started[payload.item_id] = time.time()
            while len(self._turn_started) > 128:
                self._turn_started.popitem(last=False)
            self.emit(
                "input_speech_started",
                llm.InputSpeechStartedEvent(),
            )
        elif isinstance(payload, r.SpeechStopped):
            stopped = llm.InputSpeechStoppedEvent(user_transcription_enabled=True)
            self.emit("input_speech_stopped", stopped)
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
            if (
                payload.settings_revision is not None
                and payload.settings_revision < self._tool_policy_floor
            ):
                raise RimeResourceLimitError("Prism response tool policy is no longer retained")
            if len(self._generations) >= 128:
                oldest = next(iter(self._generations.values()))
                if not oldest.ended:
                    raise RimeResourceLimitError(
                        "Prism generation limit reached with an active reply"
                    )
                self._generations.popitem(last=False)
            messages = utils.aio.Chan[llm.MessageGeneration](maxsize=128)
            functions = utils.aio.Chan[llm.FunctionCall](maxsize=128)
            created = llm.GenerationCreatedEvent(
                message_stream=messages,
                function_stream=functions,
                user_initiated=event.request_id is not None,
                response_id=payload.response.response_id,
            )
            self._generations[payload.response.response_id] = _Generation(
                payload.response,
                created,
                messages,
                functions,
                instruction_item_id=payload.instruction_item_id,
                # A speculative speech reply can arrive after a settings update
                # while still using the older revision's registered tools.
                tools_disabled=(
                    payload.settings_revision in self._suppressed_revisions
                    if payload.settings_revision is not None
                    else self._suppress_tools
                ),
            )
            if payload.instruction_item_id is not None:
                self._instruction_items.pop(payload.instruction_item_id, None)
                if len(self._instruction_items) >= 128:
                    self._instruction_items.popitem(last=False)
                self._instruction_items[payload.instruction_item_id] = (
                    payload.response.response_id,
                    False,
                )
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
            if generation.abandoned:
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
            generation = self._generations[payload.call.response_id]
            if generation.tools_disabled:
                raise PrismUnsupported("Prism emitted a tool call during a tool-disabled reply")
            if generation.abandoned:
                return
            call = self._history.record_tool_call(payload)
            self._generations[payload.call.response_id].functions.send_nowait(call)
        elif isinstance(payload, r.ResponseEnded):
            instruction_item_id = self._generations[
                payload.response.response_id
            ].instruction_item_id
            if instruction_item_id is not None:
                state = self._instruction_items.get(instruction_item_id)
                if state is not None and state[0] == payload.response.response_id:
                    self._instruction_items[instruction_item_id] = (state[0], True)
            self._end(self._generations[payload.response.response_id])
            if self._active == payload.response:
                self._active = None
        elif isinstance(payload, r.ItemTruncated):
            if payload.kept_text is not None and payload.content_index == 0:
                for generation in self._generations.values():
                    if not generation.ended and (
                        truncated_message := generation.outputs.get(payload.item_id)
                    ):
                        truncated_message.retained_text = payload.kept_text
                        break
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
            if message.retained_text is not None:
                transcript = message.retained_text
            if transcript or message.retained_text is not None:
                assert self._history is not None
                self._history.record_assistant(item_id, transcript)
            message.retained_text = None
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
            PrismModelError(
                timestamp=time.time(),
                label=self._model.label,
                error=error,
                recoverable=recoverable,
                retryable=error.retryable
                if isinstance(error, PrismError)
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
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._connection is not None and self._sdk is not None:
            await self._connection.__aexit__(None, None, None)
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
