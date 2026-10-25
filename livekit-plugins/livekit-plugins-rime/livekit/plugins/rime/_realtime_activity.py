# SPDX-License-Identifier: Apache-2.0
"""Rime-scoped realtime activity for LiveKit 1.7 and 1.8.

Only the provider lifecycle methods are specialized. Scheduling, tools, rooms,
AgentTask and handoff resources continue to use the installed LiveKit core.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import time
from collections.abc import AsyncIterable, Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, cast

from opentelemetry import trace

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm.realtime import MessageGeneration
from livekit.agents.log import logger
from livekit.agents.telemetry import trace_types, tracer
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice import agent_activity as core_activity, generation as core_generation
from livekit.agents.voice._utils import _set_participant_attributes
from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.agents.voice.events import FunctionToolsExecutedEvent
from livekit.agents.voice.generation import (
    ToolExecutionOutput,
    _AudioOutput,
    _ForwardOutput,
    forward_generation,
    perform_tool_executions,
    perform_tts_inference,
    remove_instructions,
    update_instructions,
)
from livekit.agents.voice.io import AudioOutput as CoreAudioOutput
from livekit.agents.voice.speech_handle import SpeechHandle

from ._realtime_fallback import RealtimeModelFallbackAdapter, _FallbackRealtimeSession
from ._realtime_io import AudioOutput
from ._realtime_types import InputSpeechHintEvent, can_reuse, playback_finished, set_input_text

gen_ai_telemetry: ModuleType | None
try:
    from livekit.agents.telemetry import gen_ai as _gen_ai
except ImportError:
    gen_ai_telemetry = None
else:
    gen_ai_telemetry = _gen_ai


_aligned_transcript_or_text = core_activity._aligned_transcript_or_text
_interrupted_tool_output = core_generation._interrupted_tool_output
_INTERRUPT_HAS_SOURCE = (
    "source" in inspect.signature(core_activity.AgentActivity.interrupt).parameters
)


def _reply_tool_choice(requests: Any) -> Any:
    distinct = []
    for request in requests:
        if request not in distinct:
            distinct.append(request)
    return distinct[0] if len(distinct) == 1 else None


async def _release_running_tools(session: Any, calls: Any, handle: Any) -> None:
    release = getattr(core_generation, "_release_running_tools", None)
    if release is not None:
        await release(session, calls, handle)


def _record_interruption(handle: SpeechHandle) -> None:
    record = getattr(core_activity, "_record_interruption", None)
    if record is not None:
        record(handle)


def _record_queue_wait(handle: SpeechHandle) -> None:
    record = getattr(core_activity, "_record_queue_wait", None)
    if record is not None:
        record(handle)


@contextlib.contextmanager
def _agent_turn(handle: SpeechHandle, **kwargs: Any) -> Iterator[Any]:
    native = getattr(core_activity, "_agent_turn", None)
    if native is not None:
        with native(handle, **kwargs) as span:
            yield span
    else:
        with tracer.start_as_current_span("agent_turn", context=kwargs.get("root_context")) as span:
            handle._agent_turn_context = trace.set_span_in_context(span)
            yield span


class RimeAgentActivity(core_activity.AgentActivity):
    """Own the Prism lifecycle without changing shared activities or globals.

    LiveKit 1.7 and 1.8 keep per-message forwarding results inside their realtime
    tasks and expose no hooks for typed-turn admission or playback receipts.
    The task overrides retain those results and the original branch's ordering;
    scheduling, tools and resource management still use the installed activity.
    Keep these overrides covered by the published-version and parity tests.
    """

    def __init__(self, agent: Agent, session: Any) -> None:
        super().__init__(agent, session)
        self._rt_chat_ctx_lock = asyncio.Lock()
        self._speech_hint_timer: asyncio.TimerHandle | None = None
        self._speech_hint_output: AudioOutput | None = None
        self._speech_hint_previous_gain = 1.0
        self._speech_hint_gain_revision: object | None = None
        self._rime_interrupt_sources: dict[SpeechHandle, str] = {}
        self._realtime_user_input_answered_at = 0.0
        if not hasattr(self, "_rt_overlapping_speech_enabled"):
            self._rt_overlapping_speech_enabled = False
        model = self.llm
        if isinstance(model, RealtimeModelFallbackAdapter):
            model._agent_session = session

    def _interrupted_by_user(self, handle: SpeechHandle) -> bool:
        source = getattr(handle, "_interrupt_source", None)
        return (source or self._rime_interrupt_sources.get(handle)) == "audio_activity"

    def interrupt(
        self,
        *,
        force: bool = False,
        source: Literal["audio_activity", "user_turn", "programmatic"] = "programmatic",
    ) -> asyncio.Future[None]:
        handles = [*self._background_speeches, *[item[2] for item in self._speech_q]]
        if self._current_speech is not None:
            handles.append(self._current_speech)
        for handle in handles:
            if not handle.interrupted and (force or handle.allow_interruptions):
                if handle not in self._rime_interrupt_sources:
                    handle.add_done_callback(self._forget_interrupt_source)
                self._rime_interrupt_sources[handle] = source
        if _INTERRUPT_HAS_SOURCE:
            return super().interrupt(force=force, source=source)
        return super().interrupt(force=force)

    def _forget_interrupt_source(self, handle: SpeechHandle) -> None:
        self._rime_interrupt_sources.pop(handle, None)

    def _interrupt_by_audio_activity(self, *args: Any, **kwargs: Any) -> Any:
        handles = [*self._background_speeches, *[item[2] for item in self._speech_q]]
        if self._current_speech is not None:
            handles.append(self._current_speech)
        eligible = [handle for handle in handles if not handle.interrupted]
        result = super()._interrupt_by_audio_activity(*args, **kwargs)
        if not _INTERRUPT_HAS_SOURCE:
            for handle in eligible:
                if handle.interrupted:
                    if handle not in self._rime_interrupt_sources:
                        handle.add_done_callback(self._forget_interrupt_source)
                    self._rime_interrupt_sources[handle] = "audio_activity"
        return result

    def _gain_output(self) -> AudioOutput | None:
        output = self._session.output.audio
        if output is None:
            return None
        current: CoreAudioOutput | None = output
        while current is not None:
            if isinstance(current, AudioOutput):
                return current
            current = current.next_in_chain
        wrapped = AudioOutput(output)
        self._session.output.audio = wrapped
        return wrapped

    async def _start_session(self, **kwargs: Any) -> None:
        self._gain_output()
        await super()._start_session(**kwargs)
        if self._rt_session is not None:
            self._rt_session.on("input_speech_hint", self._on_input_speech_hint)
            if isinstance(self._rt_session, _FallbackRealtimeSession):
                self._rt_session._agent_session = self._session

    async def _close_session(self) -> None:
        self._restore_speech_hint()
        if self._rt_session is not None:
            self._rt_session.off("input_speech_hint", self._on_input_speech_hint)
        await super()._close_session()
        self._rime_interrupt_sources.clear()

    async def _detach_reusable_resources(self, new_activity: Any) -> Any:
        session = self._rt_session
        reusable = session is None or can_reuse(session, new_activity.agent.chat_ctx)
        resources = await super()._detach_reusable_resources(new_activity)
        if resources.rt_session is not None:
            self._restore_speech_hint()
            resources.rt_session.off("input_speech_hint", self._on_input_speech_hint)
            if not reusable:
                self._rt_session = resources.rt_session
                resources.rt_session = None
        return resources

    def _on_input_speech_stopped(self, event: llm.InputSpeechStoppedEvent) -> None:
        self._restore_speech_hint()
        super()._on_input_speech_stopped(event)

    def _on_input_audio_transcription_completed(
        self, event: llm.InputTranscriptionCompleted
    ) -> None:
        self._restore_speech_hint()
        super()._on_input_audio_transcription_completed(event)

    def _on_end_of_agent_speech(self, *, ended_at: float) -> None:
        native = getattr(super(), "_on_end_of_agent_speech", None)
        if native is not None:
            native(ended_at=ended_at)
        elif self._audio_recognition is not None:
            legacy_recognition: Any = self._audio_recognition
            legacy_recognition._on_end_of_agent_speech(ignore_user_transcript_until=ended_at)

    @property
    def interruption_by_audio_activity_enabled(self) -> bool:
        return self._interruption_by_audio_activity_enabled

    @interruption_by_audio_activity_enabled.setter
    def interruption_by_audio_activity_enabled(self, enabled: bool) -> None:
        self._interruption_by_audio_activity_enabled = enabled

    @property
    def _rime_last_user_speaking_time(self) -> float | None:
        native = getattr(self, "_last_user_speaking_time", None)
        if native is not None:
            return cast(float, native)
        return cast(float | None, getattr(self._audio_recognition, "last_speaking_time", None))

    @property
    def _no_pending_speech(self) -> bool:
        return self._current_speech is None and not self._speech_q

    async def update_chat_ctx(
        self, chat_ctx: llm.ChatContext, *, exclude_invalid_function_calls: bool = True
    ) -> None:
        chat_ctx = chat_ctx.copy(tools=self.tools if exclude_invalid_function_calls else NOT_GIVEN)
        if self._rt_session is None:
            update_instructions(
                chat_ctx, instructions=self._agent.instructions, add_if_missing=True
            )
            self._agent._chat_ctx = chat_ctx
            return

        remove_instructions(chat_ctx)
        previous_ids = {item.id for item in self._agent._chat_ctx.items}

        async def apply() -> None:
            async with self._rt_chat_ctx_lock:
                assert self._rt_session is not None
                await self._rt_session.update_chat_ctx(chat_ctx)
                # Provider acknowledgments can wait while speech adds conversation
                # items. Commit the requested edit without losing those new items.
                target_ids = {item.id for item in chat_ctx.items}
                arrived = [
                    item
                    for item in self._agent._chat_ctx.items
                    if item.id not in previous_ids and item.id not in target_ids
                ]
                self._agent._chat_ctx = llm.ChatContext([*chat_ctx.items, *arrived])

        task = asyncio.create_task(apply())
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    def _restore_speech_hint(self) -> None:
        if self._speech_hint_timer is not None:
            self._speech_hint_timer.cancel()
            self._speech_hint_timer = None
        if self._speech_hint_output is not None:
            if self._speech_hint_output._gain_revision is self._speech_hint_gain_revision:
                self._speech_hint_output.set_gain(self._speech_hint_previous_gain)
            self._speech_hint_output = None
            self._speech_hint_gain_revision = None

    def _on_input_speech_hint(self, ev: InputSpeechHintEvent) -> None:
        output = self._gain_output()
        if output is None or not self._session.output.audio_enabled:
            return
        if (
            self._speech_hint_output is not output
            or output._gain_revision is not self._speech_hint_gain_revision
        ):
            self._restore_speech_hint()
            previous = output.gain
            if not output.set_gain(previous * ev.gain):
                return
            self._speech_hint_output = output
            self._speech_hint_previous_gain = previous
            self._speech_hint_gain_revision = output._gain_revision
        elif self._speech_hint_timer is not None:
            self._speech_hint_timer.cancel()
        self._speech_hint_timer = asyncio.get_running_loop().call_later(
            ev.timeout, self._restore_speech_hint
        )

    def _on_input_speech_started(self, ev: llm.InputSpeechStartedEvent) -> None:
        self._restore_speech_hint()
        if self.vad is None or self.using_default_vad:
            self._session._update_user_state("speaking")
            if self._audio_recognition:
                self._audio_recognition._on_start_of_speech(
                    started_at=time.time(),
                    user_speaking_span=self._session._user_speaking_span,
                )

        if self._rt_overlapping_speech_enabled or not getattr(ev, "interrupt_response", True):
            # the caller talking is not an interruption here; the model ends its own turn
            return

        try:
            # the server's own speech detection: a barge-in, like the VAD path
            self.interrupt(source="audio_activity")
        except RuntimeError:
            # only out of sync when the server cancelled its own response, with client-side turn
            # taking an uninterruptible speech is expected
            if self._rt_turn_detection_enabled:
                logger.exception(
                    "RealtimeAPI input_speech_started, but current speech is not interruptable, this should never happen!"  # noqa: E501
                )

    @utils.log_exceptions(logger=logger)
    async def _realtime_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        model_settings: ModelSettings,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        user_input: str | None = None,
        instructions: str | None = None,
        tool_reply: bool = False,
        text: str | AsyncIterable[str] | None = None,
    ) -> None:
        assert self._rt_session is not None, "rt_session is not available"
        # realtime_reply_task is called only when there's text input, native audio input is handled by _realtime_generation_task
        authorization_tasks: list[asyncio.Future[Any]] = [
            asyncio.ensure_future(speech_handle._wait_for_authorization()),
            asyncio.ensure_future(self._authorization_allowed.wait()),
        ]
        if speech_handle.allow_interruptions and not self._rt_overlapping_speech_enabled:
            authorization_tasks.append(asyncio.ensure_future(self._user_silence_event.wait()))
        await speech_handle.wait_if_not_interrupted(authorization_tasks)
        # the queue wait is recorded by _realtime_generation_task, which owns the agent_turn span
        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*authorization_tasks)
            return

        if text is not None:
            try:
                generation_ev = await self._rt_session.say(text)
            except llm.RealtimeError as e:
                logger.error("failed to say text: %s", str(e))
                speech_handle._mark_done(error=e)
                return

            await self._realtime_generation_task(
                speech_handle=speech_handle,
                generation_ev=generation_ev,
                model_settings=model_settings,
            )
            return

        if user_input is not None:
            # Keep a local item even when a legacy provider's history acknowledgment fails.
            msg = llm.ChatMessage(role="user", content=[user_input])
            try:
                async with self._rt_chat_ctx_lock:
                    msg = await set_input_text(self._rt_session, user_input, message_id=msg.id)
            except llm.RealtimeError as e:
                # the push is best-effort (the items were sent; only the ack timed out),
                # so still generate the reply rather than dropping the whole turn
                logger.warning(
                    "failed to update the chat context before generating the reply",
                    extra={"error": str(e)},
                )
            except Exception as e:
                logger.exception("failed to update the chat context before generating the reply")
                speech_handle._mark_done(error=e)
                return
            self._agent._chat_ctx._upsert_item(msg)
            self._session._conversation_item_added(msg)

        # inside on_enter, hide flagged tools even when no tools= was passed (fall back to self.tools)
        turn_tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN
        tool_ctx = llm.ToolContext(tools if tools is not None else self.tools)
        on_enter_ignored = self._on_enter_ignored_tools(tool_ctx)
        if tools is not None or on_enter_ignored:
            tool_ctx._exclude(on_enter_ignored)
            turn_tools = tool_ctx.flatten()

        ori_tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
        ori_tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN
        reply_error: llm.RealtimeError | None = None
        try:
            if not (
                per_response_tool_choice
                := self._rt_session.realtime_model.capabilities.per_response_tool_choice
            ):
                # update the tool and tool choice at the session level if they are specified
                if (
                    is_given(model_settings.tool_choice)
                    and model_settings.tool_choice != self._tool_choice
                ):
                    ori_tool_choice = self._tool_choice
                    self._rt_session.update_options(tool_choice=model_settings.tool_choice)

                if is_given(turn_tools):
                    ori_tools = self._rt_session.tools.flatten()
                    await self._rt_session.update_tools(turn_tools)

            if speech_handle.input_details.modality == "text":
                self._realtime_user_input_answered_at = time.time()
                self._session._unanswered_user_metrics = None
            generate_reply_fut = self._rt_session.generate_reply(
                instructions=instructions or NOT_GIVEN,
                tool_choice=(model_settings.tool_choice if per_response_tool_choice else NOT_GIVEN),
                tools=(turn_tools if per_response_tool_choice else NOT_GIVEN),
            )
            await speech_handle.wait_if_not_interrupted([generate_reply_fut])
            if speech_handle.interrupted:
                # cancel the pending generation; the plugin emits response.cancel
                if not generate_reply_fut.done():
                    generate_reply_fut.cancel()
                else:
                    # the response already landed, cancelling the future no longer reaches it
                    if (
                        not generate_reply_fut.cancelled()
                        and generate_reply_fut.exception() is None
                    ):
                        accepted = generate_reply_fut.result()
                        playback_session = accepted.playback_session or self._rt_session
                        playback_session.interrupt()
                        playback_finished(
                            playback_session,
                            response_id=accepted.response_id,
                            message_id=None,
                            audio_end_ms=0,
                            interrupted=True,
                            interrupted_by_user=self._interrupted_by_user(speech_handle),
                        )
                return

            try:
                generation_ev = await generate_reply_fut
            except llm.RealtimeError as e:
                logger.error(
                    "failed to generate a reply%s: %s",
                    " after tool execution" if tool_reply else "",
                    str(e),
                )
                reply_error = e
                self._session._update_agent_state("listening")
                return

            # _realtime_generation_task will clear the authorization
            await self._realtime_generation_task(
                speech_handle=speech_handle,
                generation_ev=generation_ev,
                model_settings=model_settings,
                instructions=instructions,
            )
        except llm.RealtimeError as error:
            logger.error("failed to configure realtime reply: %s", error)
            reply_error = error
            self._session._update_agent_state("listening")
        finally:
            # reset tool_choice and tools
            restore_error: Exception | None = None
            if is_given(ori_tool_choice):
                try:
                    self._rt_session.update_options(tool_choice=ori_tool_choice)
                except Exception as error:
                    logger.exception("failed to reset tool_choice")
                    restore_error = error

            if is_given(ori_tools):
                try:
                    await self._rt_session.update_tools(ori_tools)
                except Exception as error:
                    logger.exception("failed to reset tools")
                    restore_error = restore_error or error

            if restore_error is not None:
                # The provider may have partially applied a temporary setting.
                # Do not let later replies use that uncertain configuration.
                error_event = llm.RealtimeModelError(
                    timestamp=time.time(),
                    label=self._rt_session.realtime_model.label,
                    error=restore_error,
                    recoverable=False,
                )
                if isinstance(self._rt_session, _FallbackRealtimeSession):
                    self._rt_session._on_child_error(error_event)
                else:
                    self._rt_session.emit("error", error_event)
            if reply_error is not None:
                # Keep the speech scheduled until its temporary settings have
                # been restored or the uncertain provider has been rejected.
                speech_handle._mark_done(error=reply_error)

    @utils.log_exceptions(logger=logger)
    async def _realtime_generation_task(
        self,
        *,
        speech_handle: SpeechHandle,
        generation_ev: llm.GenerationCreatedEvent,
        model_settings: ModelSettings,
        instructions: str | None = None,
    ) -> None:
        with _agent_turn(
            speech_handle,
            root_context=self._session._root_span_context,
            agent_label=self._agent.label,
        ):
            inference_span = tracer.start_span("realtime_inference")
            try:
                await self._realtime_generation_task_impl(
                    speech_handle=speech_handle,
                    generation_ev=generation_ev,
                    model_settings=model_settings,
                    instructions=instructions,
                    inference_span=inference_span,
                )
            except llm.RealtimeError as error:
                logger.error("realtime generation failed: %s", error)
                speech_handle._mark_done(error=error)
                self._session._update_agent_state("listening")
            finally:
                inference_span.end()

    async def _realtime_generation_task_impl(
        self,
        *,
        speech_handle: SpeechHandle,
        generation_ev: llm.GenerationCreatedEvent,
        model_settings: ModelSettings,
        instructions: str | None = None,
        inference_span: trace.Span,
    ) -> None:
        current_span = trace.get_current_span(context=speech_handle._agent_turn_context)
        current_span.set_attribute(trace_types.ATTR_SPEECH_ID, speech_handle.id)

        room_io = self._session._room_io
        if room_io and room_io.room.isconnected():
            _set_participant_attributes(current_span, room_io.room.local_participant)

        assert self._rt_session is not None, "rt_session is not available"
        assert isinstance(self.llm, llm.RealtimeModel), "llm is not a realtime model"

        playback_session = getattr(generation_ev, "playback_session", None) or self._rt_session
        audio_output = self._session.output.audio if self._session.output.audio_enabled else None
        text_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )

        if gen_ai_telemetry is not None:
            gen_ai_telemetry.set_request_attributes(
                inference_span,
                operation=trace_types.GenAIOperationName.GENERATE_CONTENT,
                provider=self.llm.provider,
                model=self.llm.model,
                stream=True,
                # a realtime model can be configured text-only
                output_type=(
                    trace_types.GenAIOutputType.SPEECH
                    if audio_output is not None
                    else trace_types.GenAIOutputType.TEXT
                ),
            )
        # the provider metrics land here rather than on `agent_turn`; they can arrive after
        # the turn ends, in which case record_realtime_metrics opens its own child span
        if self._realtime_spans is not None and generation_ev.response_id:
            self._realtime_spans[generation_ev.response_id] = inference_span
        tool_ctx = llm.ToolContext(self.tools)

        tasks: list[asyncio.Task[Any]] = []
        tees: list[utils.aio.itertools.Tee[Any]] = []

        msg_tee = utils.aio.itertools.tee(generation_ev.message_stream, 2)
        msg_stream, msg_stream_to_drain = msg_tee
        tees.append(msg_tee)

        fnc_tee = utils.aio.itertools.tee(generation_ev.function_stream, 2)
        fnc_stream, fnc_stream_for_tracing = fnc_tee
        tees.append(fnc_tee)

        function_calls: list[llm.FunctionCall] = []
        generation_ended = False

        async def _read_fnc_stream() -> None:
            async for fnc in fnc_stream_for_tracing:
                function_calls.append(fnc)

        async def _drain_msg_stream() -> None:
            async for _ in msg_stream_to_drain:
                pass

        async def _watch_generation_end() -> None:
            # the provider closes both streams when it ends the response
            nonlocal generation_ended
            await asyncio.gather(_drain_msg_stream(), _read_fnc_stream())
            generation_ended = True

        tasks.append(
            asyncio.create_task(
                _watch_generation_end(),
                name="AgentActivity.realtime_generation.watch_end",
            )
        )

        authorization_tasks: list[asyncio.Future[Any]] = [
            asyncio.ensure_future(speech_handle._wait_for_authorization()),
            asyncio.ensure_future(self._authorization_allowed.wait()),
        ]
        if speech_handle.allow_interruptions and not self._rt_overlapping_speech_enabled:
            authorization_tasks.append(asyncio.ensure_future(self._user_silence_event.wait()))
        await speech_handle.wait_if_not_interrupted(authorization_tasks)
        speech_handle._clear_authorization()
        _record_queue_wait(speech_handle)

        if speech_handle.interrupted:
            # nothing was played, but the response may still be generating server-side
            if not generation_ended:
                playback_session.interrupt()
            await utils.aio.cancel_and_wait(*authorization_tasks, *tasks)
            for tee in tees:
                await tee.aclose()
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            _record_interruption(speech_handle)
            playback_finished(
                playback_session,
                response_id=generation_ev.response_id,
                message_id=None,
                audio_end_ms=0,
                interrupted=True,
                interrupted_by_user=self._interrupted_by_user(speech_handle),
            )
            return

        started_speaking_at: float | None = None
        stopped_speaking_at: float | None = None
        started_forwarding_at: float | None = None
        e2e_latency: float | None = None
        first_played_message_id: str | None = None

        def _on_first_frame(
            fut: asyncio.Future[float] | asyncio.Future[None], audio_out: _AudioOutput | None = None
        ) -> None:
            """
            Callback to update the agent state when the first frame is captured:
            1. _AudioOutput.first_frame_fut (float)
            2. _TextOutput.first_text_fut (None)
            """
            nonlocal started_speaking_at, started_forwarding_at, e2e_latency
            # only the first message's first frame should trigger state transitions
            if started_speaking_at is not None:
                return
            try:
                started_speaking_at = fut.result() or time.time()
                started_forwarding_at = (
                    audio_out.started_forwarding_at
                    if audio_out and audio_out.started_forwarding_at is not None
                    else started_speaking_at
                )
            except BaseException:
                return

            if speech_handle.input_details.modality == "audio":
                last_speaking_time = self._rime_last_user_speaking_time
                if (
                    last_speaking_time is not None
                    and last_speaking_time > self._realtime_user_input_answered_at
                    and last_speaking_time <= started_speaking_at
                ):
                    e2e_latency = started_speaking_at - last_speaking_time
                    current_span.set_attribute(trace_types.ATTR_E2E_LATENCY, e2e_latency)
                # A late VAD EOS must not make the same speech eligible for another reply.
                self._realtime_user_input_answered_at = max(
                    self._realtime_user_input_answered_at, started_speaking_at
                )
                self._session._unanswered_user_metrics = None

            early_metrics: llm.MetricsReport = {
                "playback_latency": started_speaking_at - started_forwarding_at
            }
            if e2e_latency is not None:
                early_metrics["e2e_latency"] = e2e_latency
            self._session._early_assistant_metrics = early_metrics

            self._session._update_agent_state(
                "speaking",
                start_time=started_speaking_at,
                otel_context=speech_handle._agent_turn_context,
            )
            if self._audio_recognition:
                self._audio_recognition._on_start_of_agent_speech(started_at=started_speaking_at)
            if self.interruption_enabled:
                self._disable_vad_interruption_soon()

        read_transcript_from_tts = False

        # multiple message items may be produced for a single realtime response
        # (e.g. GPT-Realtime-2.0). We process each one serially: push frames,
        # flush, wait_for_playout
        @dataclass
        class _MsgOutput:
            msg: MessageGeneration
            out: _ForwardOutput

        message_outputs: list[_MsgOutput] = []

        async def _process_one_message(msg: MessageGeneration) -> _MsgOutput:
            """Resolve a message's audio/text sources, then forward and wait for playout."""
            nonlocal read_transcript_from_tts
            assert isinstance(self.llm, llm.RealtimeModel)

            msg_modalities = await msg.modalities
            tts_text_input: AsyncIterable[str] | None = None
            if "audio" not in msg_modalities and self.tts:
                if self.llm.capabilities.audio_output:
                    logger.warning(
                        "text response received from realtime API, falling back to use a TTS model."  # noqa: E501
                    )
                tee = utils.aio.itertools.tee(msg.text_stream, 2)
                tts_text_input, tr_text_input = tee
                tees.append(tee)
            else:
                tr_text_input = msg.text_stream.__aiter__()

            audio_source: AsyncIterable[rtc.AudioFrame] | None = None
            if audio_output is not None:
                if tts_text_input is not None:
                    tts_task, tts_gen_data = perform_tts_inference(
                        node=self._agent.tts_node,
                        input=tts_text_input,
                        model_settings=model_settings,
                        text_transforms=self._session.options.tts_text_transforms,
                        model=self.tts.model if self.tts else None,
                        provider=self.tts.provider if self.tts else None,
                    )

                    if (
                        self.use_tts_aligned_transcript
                        and (tts := self.tts)
                        and (tts.capabilities.aligned_transcript or not tts.capabilities.streaming)
                        and (timed_texts := await tts_gen_data.timed_texts_fut)
                    ):
                        tr_text_input = _aligned_transcript_or_text(timed_texts, tr_text_input)
                        read_transcript_from_tts = True

                    tasks.append(tts_task)
                    audio_source = tts_gen_data.audio_ch
                elif "audio" in msg_modalities:
                    realtime_audio = self._agent.realtime_audio_output_node(
                        msg.audio_stream, model_settings
                    )
                    audio_source = (
                        await realtime_audio
                        if asyncio.iscoroutine(realtime_audio)
                        else realtime_audio
                    )
                elif self.llm.capabilities.audio_output:
                    logger.error(
                        "Text message received from Realtime API with audio modality. "
                        "This usually happens when text chat context is synced to the API. "
                        "Try to add a TTS model as fallback or use text modality with TTS instead."  # noqa: E501
                    )
                else:
                    logger.warning(
                        "audio output is enabled but neither tts nor realtime audio is available",
                    )

            async def _discard_audio() -> None:
                async for _ in msg.audio_stream:
                    pass

            def _on_message_first_frame(
                fut: asyncio.Future[float] | asyncio.Future[None],
                audio_out: _AudioOutput | None = None,
            ) -> None:
                nonlocal first_played_message_id
                _on_first_frame(fut, audio_out)
                if first_played_message_id is None and started_speaking_at is not None:
                    first_played_message_id = msg.message_id

            # Providers can have bounded output queues even when playback is disabled.
            # Consume their audio alongside text so an unused stream cannot block generation.
            discard_audio_task = (
                asyncio.create_task(_discard_audio())
                if audio_output is None and "audio" in msg_modalities
                else None
            )

            async def _forward_message() -> _ForwardOutput:
                tr_node = self._agent.transcription_node(tr_text_input, model_settings)
                text_source = await tr_node if asyncio.iscoroutine(tr_node) else tr_node
                return await forward_generation(
                    speech_handle=speech_handle,
                    audio_output=audio_output,
                    text_output=text_output,
                    audio_source=audio_source,
                    text_source=text_source,
                    on_first_frame=_on_message_first_frame,
                    reconcile_playout_pause=lambda: self._reconcile_playout_pause(speech_handle),
                )

            if discard_audio_task is None:
                return _MsgOutput(msg=msg, out=await _forward_message())

            forwarding = asyncio.create_task(_forward_message())
            message_tasks = asyncio.gather(forwarding, discard_audio_task)
            try:
                await speech_handle.wait_if_not_interrupted([message_tasks])
                if not speech_handle.interrupted:
                    message_tasks.result()
                return _MsgOutput(msg=msg, out=await forwarding)
            finally:
                await utils.aio.cancel_and_wait(forwarding, discard_audio_task)
                await asyncio.gather(message_tasks, return_exceptions=True)

        @utils.log_exceptions(logger=logger)
        async def _process_messages() -> None:
            async for msg in msg_stream:
                if speech_handle.interrupted:
                    # remaining messages are left out of message_outputs so
                    # update_chat_ctx below removes them server-side.
                    break
                entry = await _process_one_message(msg)
                message_outputs.append(entry)
                if entry.out.played == "partial":
                    break

        process_msg_task = asyncio.create_task(
            _process_messages(), name="AgentActivity.realtime_generation.process_messages"
        )
        tasks.append(process_msg_task)

        started_calls: list[llm.FunctionCall] = []

        # messages in RunResult are ordered by the `created_at` field
        def _tool_execution_started_cb(fnc_call: llm.FunctionCall) -> None:
            # function call is created during the realtime generation, before the assistant
            # message it belongs to is placed at `started_speaking_at`
            # reset the `created_at` to the start time of the tool execution
            fnc_call.created_at = time.time()
            started_calls.append(fnc_call)
            speech_handle._item_added([fnc_call])
            self._agent._chat_ctx._upsert_item(fnc_call)
            self._session._tool_items_added([fnc_call])

        def _tool_execution_completed_cb(out: ToolExecutionOutput) -> None:
            speech_handle._item_added([out.fnc_call_out])

        exe_task, tool_output = perform_tool_executions(
            session=self._session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=model_settings.tool_choice,
            function_stream=fnc_stream,
            tool_execution_started_cb=_tool_execution_started_cb,
            tool_execution_completed_cb=_tool_execution_completed_cb,
        )

        # Propagate a processing failure even if the provider's other streams remain open.
        processing = asyncio.gather(*tasks)
        try:
            await speech_handle.wait_if_not_interrupted([processing])
            if not speech_handle.interrupted:
                processing.result()

            # the cancel is session-wide, so send it only while this response is still generating
            if speech_handle.interrupted and not generation_ended:
                playback_session.interrupt()

            current_span.set_attribute(
                trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted
            )
            _record_interruption(speech_handle)
            current_span.set_attribute(
                trace_types.ATTR_RESPONSE_FUNCTION_CALLS,
                json.dumps(
                    [fnc.model_dump(exclude={"type", "created_at"}) for fnc in function_calls]
                ),
            )

            # _process_messages handles its own playout waits and interrupt cleanup
            await process_msg_task
        except BaseException:
            if not generation_ended:
                with contextlib.suppress(Exception):
                    playback_session.interrupt()
            await utils.aio.cancel_and_wait(*tasks, exe_task, *authorization_tasks)
            for tee in tees:
                await tee.aclose()
            raise
        finally:
            if not processing.done():
                processing.cancel()
            await asyncio.gather(processing, return_exceptions=True)

        self._session._update_agent_state("thinking" if self._background_speeches else "listening")
        self._on_end_of_agent_speech(ended_at=time.time())
        if self.interruption_enabled:
            self.interruption_by_audio_activity_enabled = True

        stopped_speaking_at = time.time()

        def _create_assistant_message(
            message_id: str, forwarded_text: str, interrupted: bool
        ) -> llm.ChatMessage:
            assistant_metrics: llm.MetricsReport = {}

            if generation_ev.response_id:
                assistant_metrics["provider_request_ids"] = [generation_ev.response_id]
            if e2e_latency is not None and message_id == first_played_message_id:
                assistant_metrics["e2e_latency"] = e2e_latency

            if stopped_speaking_at and started_speaking_at:
                assistant_metrics["started_speaking_at"] = started_speaking_at
                assistant_metrics["stopped_speaking_at"] = stopped_speaking_at

                if started_forwarding_at is not None:
                    assistant_metrics["playback_latency"] = (
                        started_speaking_at - started_forwarding_at
                    )

            msg = llm.ChatMessage(
                role="assistant",
                content=[forwarded_text],
                id=message_id,
                interrupted=interrupted,
            )
            if started_speaking_at is not None:
                msg.created_at = started_speaking_at
            msg.metrics = assistant_metrics
            return msg

        if (
            not speech_handle.interrupted
            and read_transcript_from_tts
            and any(
                e.out.played != "skipped" and e.out.text_out is not None and not e.out.text_out.text
                for e in message_outputs
            )
        ):
            logger.warning(
                "`use_tts_aligned_transcript` is enabled but no agent transcript was returned from tts"  # noqa: E501
            )

        # Even a response with no messages can require a playback receipt before
        # the provider admits the next turn.
        if not message_outputs:
            playback_finished(
                playback_session,
                response_id=generation_ev.response_id,
                message_id=None,
                audio_end_ms=0,
                interrupted=speech_handle.interrupted,
                interrupted_by_user=self._interrupted_by_user(speech_handle),
            )

        # create assistant message per generated message
        trace_text_parts: list[str] = []
        any_skipped = False
        for entry in message_outputs:
            if entry.out.played == "skipped":
                any_skipped = True
                playback_finished(
                    playback_session,
                    response_id=generation_ev.response_id,
                    message_id=entry.msg.message_id,
                    audio_end_ms=0,
                    interrupted=True,
                    interrupted_by_user=self._interrupted_by_user(speech_handle),
                )
                continue

            msg_interrupted = entry.out.played == "partial"
            forwarded_text = entry.out.forwarded_text

            playback_finished(
                playback_session,
                response_id=generation_ev.response_id,
                message_id=entry.msg.message_id,
                modalities=await entry.msg.modalities,
                audio_end_ms=int(entry.out.playback_position * 1000),
                audio_transcript=forwarded_text,
                interrupted=msg_interrupted,
                interrupted_by_user=self._interrupted_by_user(speech_handle),
            )

            if not forwarded_text:
                continue

            trace_text_parts.append(forwarded_text)
            chat_msg = _create_assistant_message(
                message_id=entry.msg.message_id,
                forwarded_text=forwarded_text,
                interrupted=msg_interrupted,
            )
            self._agent._chat_ctx._upsert_item(chat_msg)
            speech_handle._item_added([chat_msg])
            self._session._conversation_item_added(chat_msg)

        if trace_text_parts:
            current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, "\n".join(trace_text_parts))

        # sync local chat ctx to the realtime server to remove any items the
        # model added but the user never heard (interrupted before we pulled
        # them, or message_outputs entries left in "skipped")
        if speech_handle.interrupted and any_skipped and self.llm.capabilities.mutable_chat_context:
            try:
                async with self._rt_chat_ctx_lock:
                    await self._rt_session.update_chat_ctx(self._agent._chat_ctx)
            except llm.RealtimeError as e:
                logger.warning(
                    "failed to sync chat context to remove never-played messages",
                    extra={"error": str(e)},
                )

        for entry in message_outputs:
            if entry.out.audio_out is not None and not entry.out.audio_out.first_frame_fut.done():
                entry.out.audio_out.first_frame_fut.cancel()

        for tee in tees:
            await tee.aclose()
        speech_handle._mark_generation_done()

        if speech_handle.interrupted:
            timeout_cancel: asyncio.CancelledError | None = None
            try:
                await utils.aio.cancel_and_wait(exe_task)
            except asyncio.CancelledError as e:
                # the interruption timeout stopped waiting for the tools: release them as an
                # async tool's first update does, and commit that update below
                timeout_cancel = e
                await _release_running_tools(self._session, started_calls, speech_handle)
                await asyncio.wait([exe_task])

            # commit results of tools that finished despite the interruption, as the pipeline
            # task does. the calls are already recorded, so each one answers or the model waits
            interrupted_calls: list[llm.FunctionCall] = []
            interrupted_fnc_outputs: list[llm.FunctionCallOutput] = []
            for sanitized_out in tool_output.output:
                interrupted_calls.append(sanitized_out.fnc_call)
                interrupted_fnc_outputs.append(_interrupted_tool_output(sanitized_out))

            if interrupted_fnc_outputs:
                self._session.emit(
                    "function_tools_executed",
                    FunctionToolsExecutedEvent(
                        function_calls=interrupted_calls,
                        function_call_outputs=interrupted_fnc_outputs,
                    ),
                )
                self._agent._chat_ctx.insert(interrupted_fnc_outputs)
                self._session._tool_items_added(interrupted_fnc_outputs)

                # unlike the pipeline, a realtime model holds the call open server-side
                try:
                    async with self._rt_chat_ctx_lock:
                        chat_ctx = self._rt_session.chat_ctx.copy()
                        chat_ctx.items.extend(interrupted_fnc_outputs)
                        await self._rt_session.update_chat_ctx(chat_ctx)
                except llm.RealtimeError as e:
                    logger.warning(
                        "failed to sync the tool results of an interrupted generation",
                        extra={"error": str(e)},
                    )
            if timeout_cancel is not None:
                raise timeout_cancel
            return

        # wait for the tool execution to complete
        tool_output.first_tool_started_fut.add_done_callback(
            lambda _: self._session._update_agent_state("thinking")
        )

        self._background_speeches.add(speech_handle)
        try:
            await exe_task
        finally:
            self._background_speeches.discard(speech_handle)

        # important: no agent output should be used after this point

        tool_reply_expected = False
        if len(tool_output.output) > 0:
            max_steps_reached = speech_handle.num_steps >= self._session.options.max_tool_steps + 1
            speech_handle._num_steps += 1

            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            fnc_executed_ev = FunctionToolsExecutedEvent(
                function_calls=[], function_call_outputs=[]
            )
            new_agent_task: Agent | None = None
            ignore_task_switch = False

            for sanitized_out in tool_output.output:
                fnc_executed_ev.function_calls.append(sanitized_out.fnc_call)
                fnc_executed_ev.function_call_outputs.append(sanitized_out.fnc_call_out)

                new_fnc_outputs.append(sanitized_out.fnc_call_out)

                # record the call with its output, as the pipeline task does. a call rejected
                # before execution never reached the started callback
                self._agent._chat_ctx._upsert_item(sanitized_out.fnc_call)
                self._agent._chat_ctx._upsert_item(sanitized_out.fnc_call_out)
                self._session._tool_items_added(
                    [sanitized_out.fnc_call, sanitized_out.fnc_call_out]
                )

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error(
                        "expected to receive only one Agent from the tool executions",
                    )
                    ignore_task_switch = True

                if sanitized_out.agent_task is not None:
                    new_agent_task = sanitized_out.agent_task

            if new_agent_task and not ignore_task_switch:
                fnc_executed_ev._handoff_required = True

            self._session.emit("function_tools_executed", fnc_executed_ev)

            draining = self.scheduling_paused
            if fnc_executed_ev._handoff_required and new_agent_task and not ignore_task_switch:
                self._session.update_agent(new_agent_task)
                draining = True

            if len(new_fnc_outputs) > 0:
                # wait all speeches played before updating the tool output and generating the response
                # most realtime models don't support generating multiple responses at the same time
                while self._current_speech or self._speech_q:
                    if (
                        self._current_speech
                        and not self._current_speech.done()
                        and self._current_speech is not speech_handle
                    ):
                        await self._current_speech
                    else:
                        await asyncio.sleep(0)

                # if the realtime model auto-generates the tool reply, install a
                # placeholder so the active RunResult waits for that reply
                auto_reply_fut: asyncio.Future[None] | None = None
                if (
                    self._rt_session.capabilities.auto_tool_reply_generation
                    and fnc_executed_ev.has_tool_reply
                    and self._pending_auto_tool_reply_fut is None
                    and (run_state := self._session._global_run_state) is not None
                    and not run_state.done()
                ):
                    auto_reply_fut = asyncio.get_event_loop().create_future()
                    self._pending_auto_tool_reply_fut = auto_reply_fut
                    llm_label = self.llm._label

                    async def _wait_for_auto_tool_reply() -> None:
                        try:
                            await asyncio.wait_for(asyncio.shield(auto_reply_fut), 5.0)
                        except asyncio.TimeoutError:
                            logger.warning(
                                "timed out waiting for realtime auto tool reply from %s",
                                llm_label,
                            )
                        finally:
                            if self._pending_auto_tool_reply_fut is auto_reply_fut:
                                self._pending_auto_tool_reply_fut = None

                    task = asyncio.create_task(_wait_for_auto_tool_reply())
                    run_state._watch_handle(task)

                try:
                    async with self._rt_chat_ctx_lock:
                        chat_ctx = self._rt_session.chat_ctx.copy()
                        chat_ctx.items.extend(new_fnc_outputs)
                        await self._rt_session.update_chat_ctx(chat_ctx)
                except llm.RealtimeError as e:
                    logger.warning(
                        "failed to update chat context before generating the function calls results",  # noqa: E501
                        extra={"error": str(e)},
                    )
                    if auto_reply_fut is not None and not auto_reply_fut.done():
                        if self._pending_auto_tool_reply_fut is auto_reply_fut:
                            self._pending_auto_tool_reply_fut = None
                        auto_reply_fut.set_result(None)

            tool_reply_expected = fnc_executed_ev.has_tool_reply
            if tool_reply_expected and not self._rt_session.capabilities.auto_tool_reply_generation:
                self._rt_session.interrupt()

                if max_steps_reached:
                    logger.warning(
                        "maximum number of function calls steps reached, "
                        "generating final response with tool_choice='none'",
                        extra={"speech_id": speech_handle.id},
                    )

                self._create_speech_task(
                    self._realtime_reply_task(
                        speech_handle=speech_handle,
                        model_settings=ModelSettings(
                            # Avoid setting tool_choice to "required" or a specific function when
                            # passing tool response back to the LLM
                            tool_choice="none"
                            if max_steps_reached or draining or model_settings.tool_choice == "none"
                            else _reply_tool_choice(
                                getattr(out, "reply_tool_choice", None)
                                for out in tool_output.output
                                if out.fnc_call_out.reply_required
                            )
                            or "auto",
                        ),
                        tool_reply=True,
                    ),
                    speech_handle=speech_handle,
                    name="AgentActivity.realtime_reply",
                )
                self._schedule_speech(
                    speech_handle, SpeechHandle.SPEECH_PRIORITY_NORMAL, force=True
                )

        # no reply follows, so nothing else clears the "thinking" the tool asserted
        if not tool_reply_expected and self._no_pending_speech:
            self._session._update_agent_state(
                "thinking" if self._background_speeches else "listening"
            )
