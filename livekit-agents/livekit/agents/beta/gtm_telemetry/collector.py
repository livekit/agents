"""Post-call GTM/CRM telemetry collector for :class:`~livekit.agents.voice.AgentSession`.

Subscribes to session events live (``conversation_item_added``,
``tool_execution_updated``, ``function_tools_executed``, ``metrics_collected``,
``close``) and aggregates them into a :class:`~.models.PostCallReport` at the
end of the call. Works with or without a ``JobContext``, on both the pipeline
(STT->LLM->TTS) and realtime paths.

Design notes / limitations:

- **Transcript** is built from a live ``conversation_item_added`` subscription
  (fires on both pipeline and realtime paths). An equivalent alternative is to
  read ``session.history`` at close; we subscribe live anyway for tool timing,
  so one mechanism covers everything.
- **Tool latency is self-timed** from ``tool_execution_updated`` event
  ``created_at`` deltas keyed by ``call_id`` — no duration field exists on the
  events. Calls that never reach the executor (unknown tools, malformed
  arguments, rejected duplicates) only appear in the ``function_tools_executed``
  batch and are recorded untimed (``duration_ms=None``).
- **Tool error visibility:** the report carries the diagnostic
  ``ToolCallEnded.message`` (the actual exception text) rather than the
  LLM-facing redacted ``"An internal error occurred"`` string — webhook
  receivers are the operator, not the LLM; redaction protects the model prompt,
  not the owner's telemetry. Treat webhook payloads as sensitive.
- **Realtime metric gaps:** realtime sessions route usage into LLM usage (not
  STT/TTS), so ``user/agent_speech_duration_seconds`` are ``None`` on
  realtime-only sessions (by design, not 0.0). ``RealtimeModelMetrics.ttft``
  of ``-1`` (no audio token) is skipped, so ``avg_llm_ttft_ms`` may be ``None``.
  ``STTMetrics.audio_duration`` is *pushed-audio* duration, not strictly speech.
- **Flush lifecycle:** the sync ``close`` handler spawns the flush task; await
  it before process exit from the ``@server.rtc_session(on_session_end=...)``
  worker hook (300s default budget) via :meth:`PostCallTelemetryCollector.aflush`.
  ``JobContext.add_shutdown_callback`` is a degraded fallback only: those
  callbacks run last under the ~10s ``shutdown_process_timeout``, which does
  not cover the dispatcher's ~31.5s retry worst case.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

from ...log import logger
from ...metrics import LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics
from ...metrics.usage import STTModelUsage, TTSModelUsage
from ...voice.events import (
    CloseEvent,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    ToolExecutionUpdatedEvent,
)
from .models import CallMetrics, PostCallReport, ToolInvocationRecord, TranscriptTurn
from .webhook import WebhookDispatcher

if TYPE_CHECKING:
    from ...llm.chat_context import FunctionCall, FunctionCallOutput
    from ...voice import AgentSession


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Parse a ``FunctionCall.arguments`` JSON string into a dict."""
    try:
        parsed = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {"_raw": raw}
    if not isinstance(parsed, dict):
        return {"_raw": raw}
    return parsed


class PostCallTelemetryCollector:
    """Collects transcript turns, tool invocations, and call metrics from an
    :class:`AgentSession` and packages them into a :class:`PostCallReport`.

    Usage::

        collector = PostCallTelemetryCollector(
            session,
            room_name=ctx.room.name,
            dispatcher=WebhookDispatcher("https://example.com/webhook", webhook_secret="..."),
        )
        collector.attach()
        # ... session runs ...
        # in the on_session_end= worker hook:
        await collector.aflush()
        report = collector.generate_report()
        await collector.aclose()

    When a ``dispatcher`` is provided, the report is automatically flushed to
    the webhook when the session closes.
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        room_name: str | None = None,
        room_id: str | None = None,
        job_id: str | None = None,
        participant_identity: str | None = None,
        metadata: dict[str, Any] | None = None,
        dispatcher: WebhookDispatcher | None = None,
    ) -> None:
        self._session = session
        self._room_name = room_name
        self._room_id = room_id
        self._job_id = job_id
        self._participant_identity = participant_identity
        self._metadata = metadata or {}
        self._dispatcher = dispatcher

        self._turns: list[TranscriptTurn] = []
        self._pending_tools: dict[str, ToolInvocationRecord] = {}
        self._completed_tools: dict[str, ToolInvocationRecord] = {}
        self._llm_ttfts: list[float] = []
        self._stt_audio_s: float | None = None
        self._tts_audio_s: float | None = None
        self._close_event: CloseEvent | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._started_at: float | None = None

    def attach(self) -> None:
        """Register the session event handlers and start the call timer."""
        if self._started_at is not None:
            return

        self._started_at = time.time()
        self._session.on("conversation_item_added", self._on_conversation_item_added)
        self._session.on("tool_execution_updated", self._on_tool_execution_updated)
        self._session.on("function_tools_executed", self._on_function_tools_executed)
        # metrics_collected is deprecated, but it is still the only event carrying
        # per-request LLM ttft and STT/TTS audio_duration; registering logs a single
        # deprecation warning. If upstream removes it, the fallback is
        # ChatMessage.metrics (pipeline) + session.usage (durations).
        self._session.on("metrics_collected", self._on_metrics_collected)
        self._session.on("close", self._on_close)

    async def aclose(self) -> None:
        """Unregister all handlers and await/cancel any in-flight flush task."""
        if self._started_at is not None:
            self._session.off("conversation_item_added", self._on_conversation_item_added)
            self._session.off("tool_execution_updated", self._on_tool_execution_updated)
            self._session.off("function_tools_executed", self._on_function_tools_executed)
            self._session.off("metrics_collected", self._on_metrics_collected)
            self._session.off("close", self._on_close)

        if self._flush_task is not None:
            if not self._flush_task.done():
                logger.warning(
                    "aclose() called with flush in-flight; awaiting up to 5s grace period."
                )
                await self.aflush(timeout=5.0)
            else:
                with contextlib.suppress(asyncio.CancelledError):
                    await self._flush_task

        self._started_at = None
        self._flush_task = None

    def generate_report(self) -> PostCallReport:
        """Build the :class:`PostCallReport` from the collected state."""
        if self._started_at is None:
            raise RuntimeError("collector not attached — call attach() first")

        end = self._close_event.created_at if self._close_event else time.time()
        all_tools = [*self._completed_tools.values(), *self._pending_tools.values()]
        metrics = CallMetrics(
            total_duration_seconds=end - self._started_at,
            user_speech_duration_seconds=self._speech_duration_s(self._stt_audio_s, STTModelUsage),
            agent_speech_duration_seconds=self._speech_duration_s(self._tts_audio_s, TTSModelUsage),
            total_tool_calls=len(all_tools),
            failed_tool_calls=sum(1 for t in all_tools if t.status in ("error", "cancelled")),
            avg_llm_ttft_ms=(
                sum(self._llm_ttfts) / len(self._llm_ttfts) * 1000 if self._llm_ttfts else None
            ),
        )
        return PostCallReport(
            room_name=self._room_name,
            room_id=self._room_id,
            job_id=self._job_id,
            participant_identity=self._participant_identity,
            close_reason=self._close_event.reason.value if self._close_event else None,
            metadata=self._metadata,
            turns=list(self._turns),
            tool_invocations=all_tools,
            metrics=metrics,
        )

    async def aflush(self, *, timeout: float = 45.0) -> None:
        """Await the auto-flush task spawned by the ``close`` handler.

        ``asyncio.wait_for`` CANCELS the flush task on timeout, so the default
        45s budget deliberately exceeds the dispatcher's worst case (~31.5s
        with default retry settings) — lowering it trades away retries. No-op
        when no flush was started (no dispatcher, or the session hasn't closed).
        """
        if self._flush_task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._flush_task), timeout=timeout)
        except asyncio.TimeoutError:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
            logger.warning("timed out waiting for the post-call report flush")
        except asyncio.CancelledError:
            if self._flush_task.cancelled():
                # the flush task itself was cancelled, not this caller
                logger.warning("post-call report flush was cancelled before completing")
                return
            logger.warning("post-call report flush was cancelled before completing")
            raise

    # -- sync event handlers (session.on() rejects coroutines) --

    def _on_conversation_item_added(self, ev: ConversationItemAddedEvent) -> None:
        if ev.item.type != "message":
            return
        if ev.item.role not in ("user", "assistant"):
            return
        text = ev.item.text_content
        if not text:
            return
        self._turns.append(
            TranscriptTurn(
                speaker="user" if ev.item.role == "user" else "agent",
                text=text,
                interrupted=ev.item.interrupted,
                timestamp=ev.item.created_at,
            )
        )

    def _on_tool_execution_updated(self, ev: ToolExecutionUpdatedEvent) -> None:
        update = ev.update
        if update.type == "tool_call_started":
            fc = update.function_call
            self._pending_tools[fc.call_id] = ToolInvocationRecord(
                tool_name=fc.name,
                call_id=fc.call_id,
                arguments=_parse_arguments(fc.arguments),
                started_at=ev.created_at,
            )
        elif update.type == "tool_call_ended":
            rec = self._pending_tools.pop(update.call_id, None)
            if rec is None:
                return
            # always update timing/status; merge result/error text only when the
            # terminal event carries it — for DEFERRED tools the batch event can
            # backfill the pending record before the terminal event fires, and a
            # terminal message=None must not erase that value.
            rec.completed_at = ev.created_at
            rec.duration_ms = (rec.completed_at - rec.started_at) * 1000
            rec.status = update.status
            if update.message is not None:
                if update.status == "done":
                    rec.result = update.message
                else:  # error / cancelled
                    rec.error = update.message
            self._completed_tools[rec.call_id] = rec

    def _on_function_tools_executed(self, ev: FunctionToolsExecutedEvent) -> None:
        for fc, out in ev.zipped():
            rec = self._find_record(fc.call_id)
            if rec is not None:
                # backfill ONLY fields still None: ToolCallEnded.message holds the
                # diagnostic exception text while FunctionCallOutput.output is
                # redacted to "An internal error occurred" for generic exceptions —
                # never overwrite the diagnostic text with the redacted one.
                if out is not None and not out.is_error and rec.result is None:
                    rec.result = out.output
                if out is not None and out.is_error:
                    if rec.error is None:
                        rec.error = out.output
                    if rec.status == "running":
                        rec.status = "error"
            else:
                # never reached the executor (unknown tool, malformed arguments,
                # rejected duplicate) — no lifecycle events, record untimed
                rec = self._untimed_record(fc, out)
                self._completed_tools[rec.call_id] = rec

    def _on_metrics_collected(self, ev: MetricsCollectedEvent) -> None:
        m = ev.metrics
        if isinstance(m, LLMMetrics):
            self._llm_ttfts.append(m.ttft)
        elif isinstance(m, RealtimeModelMetrics):
            if m.ttft != -1:  # -1 means no audio token was sent
                self._llm_ttfts.append(m.ttft)
        elif isinstance(m, STTMetrics):
            self._stt_audio_s = (self._stt_audio_s or 0.0) + m.audio_duration
        elif isinstance(m, TTSMetrics):
            self._tts_audio_s = (self._tts_audio_s or 0.0) + m.audio_duration

    def _on_close(self, ev: CloseEvent) -> None:
        self._close_event = ev
        if self._dispatcher is not None and self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_impl())

    # -- internals --

    def _find_record(self, call_id: str) -> ToolInvocationRecord | None:
        return self._pending_tools.get(call_id) or self._completed_tools.get(call_id)

    def _untimed_record(
        self, fc: FunctionCall, out: FunctionCallOutput | None
    ) -> ToolInvocationRecord:
        result: str | None = None
        error: str | None = None
        status: str = "done"
        if out is not None:
            if out.is_error:
                error = out.output
                status = "error"
            else:
                result = out.output
        return ToolInvocationRecord(
            tool_name=fc.name,
            call_id=fc.call_id,
            arguments=_parse_arguments(fc.arguments),
            result=result,
            error=error,
            status=status,
            started_at=fc.created_at,
            completed_at=None,
            duration_ms=None,
        )

    def _speech_duration_s(
        self, cached: float | None, usage_type: type[STTModelUsage] | type[TTSModelUsage]
    ) -> float | None:
        if cached is not None:
            return cached
        usages = [u for u in self._session.usage.model_usage if isinstance(u, usage_type)]
        if usages:
            return sum(u.audio_duration for u in usages)
        return None

    async def _flush_impl(self) -> None:
        if self._dispatcher is None:
            return
        try:
            report = self.generate_report()
            await self._dispatcher.dispatch(report)
        except Exception:
            logger.exception("failed to flush the post-call report")
