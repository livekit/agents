from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import aiohttp

from livekit.agents import APIConnectionError, llm
from livekit.agents.inference._utils import (
    HEADER_INFERENCE_PROVIDER,
    create_access_token,
    get_default_inference_url,
    get_inference_headers,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from openai.types.beta.realtime.session import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    TurnDetection,
)
from openai.types.realtime import (
    AudioTranscription,
    NoiseReductionType,
    RealtimeAudioInputTurnDetection,
    RealtimeReasoning,
)
from openai.types.realtime.realtime_audio_config_input import NoiseReduction
from openai.types.realtime.realtime_session_create_response import Tracing
from openai.types.realtime.realtime_truncation import RealtimeTruncation

from .realtime_model import (
    DEFAULT_VOICE,
    RealtimeModel,
    RealtimeSession,
    process_base_url,
)

InferenceClass = Literal["priority", "standard", "low"]
_T = TypeVar("_T")

# At 24 kHz mono PCM16 this retains about 44 seconds, well beyond a normal spoken turn.
DEFAULT_MAX_UNCOMMITTED_AUDIO_BYTES = 2 * 1024 * 1024
_MAX_REPLAY_TRAILING_SILENCE_BYTES = 48_000  # one second of 24 kHz mono PCM16
_FAILOVER_PROTOCOL_HEADER = "X-LiveKit-Realtime-Failover-Protocol"
_FAILOVER_PROTOCOL_VERSION = 1
_MAX_REPLAY_TIMEOUT_MS = 30_000
_REPLAY_DEADLINE_SAFETY_FRACTION = 0.1
_MAX_REPLAY_DEADLINE_SAFETY_SECONDS = 1.0
_INTERRUPT_TIMEOUT_FRACTION = 0.2
_MAX_INTERRUPT_TIMEOUT_SECONDS = 2.0
_REPLAY_AUDIO_EVENT_ID_PREFIX = "livekit_replay_audio_"


@dataclass
class _InferenceOptions:
    provider: str | None
    api_key: str
    api_secret: str
    inference_class: InferenceClass | None
    max_uncommitted_audio_bytes: int


class InferenceRealtimeModel(RealtimeModel):
    """OpenAI-compatible realtime model authenticated through LiveKit Inference.

    ``max_uncommitted_audio_bytes`` bounds failover memory. Exceeding it disables
    active-turn replay until the provider commits or clears the audio buffer.
    """

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: InferenceClass | None = None,
        max_uncommitted_audio_bytes: int = DEFAULT_MAX_UNCOMMITTED_AUDIO_BYTES,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[
            AudioTranscription | InputAudioTranscription | None
        ] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            NoiseReductionType | NoiseReduction | InputAudioNoiseReduction | None
        ] = NOT_GIVEN,
        turn_detection: NotGivenOr[
            RealtimeAudioInputTurnDetection | TurnDetection | None
        ] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        truncation: NotGivenOr[RealtimeTruncation | None] = NOT_GIVEN,
        reasoning: NotGivenOr[RealtimeReasoning | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if "/" not in model:
            raise ValueError("model must be provider-prefixed, for example 'openai/gpt-realtime'")
        if max_uncommitted_audio_bytes <= 0:
            raise ValueError("max_uncommitted_audio_bytes must be greater than zero")

        resolved_api_key = api_key or os.getenv(
            "LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", "")
        )
        if not resolved_api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
            )

        resolved_api_secret = api_secret or os.getenv(
            "LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", "")
        )
        if not resolved_api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
            )

        super().__init__(
            model=model,
            voice=voice,
            modalities=modalities,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            speed=speed,
            tracing=tracing,
            truncation=truncation,
            reasoning=reasoning,
            api_key="livekit-inference",
            base_url=base_url or get_default_inference_url(),
            http_session=http_session,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
            temperature=temperature,
        )
        self._inference_opts = _InferenceOptions(
            provider=provider,
            api_key=resolved_api_key,
            api_secret=resolved_api_secret,
            inference_class=inference_class,
            max_uncommitted_audio_bytes=max_uncommitted_audio_bytes,
        )
        self._provider_label = "LiveKit Inference Realtime"

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self, *, turn_detection_disabled: bool = False) -> InferenceRealtimeSession:
        sess = InferenceRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        return sess


class InferenceRealtimeSession(RealtimeSession):
    """Realtime session with explicit replay coordination for gateway failover."""

    def __init__(
        self,
        realtime_model: InferenceRealtimeModel,
        *,
        turn_detection_disabled: bool = False,
    ) -> None:
        self._inference_model = realtime_model
        self._uncommitted_audio: list[bytes] = []
        self._uncommitted_audio_bytes = 0
        self._audio_buffer_overflowed = False
        self._replay_trailing_silence_bytes = 0
        self._active_user_turn_uncommitted = False
        self._user_turn_commit_count = 0
        self._gateway_failover_in_progress = False
        self._live_forwarding_allowed = asyncio.Event()
        self._live_forwarding_allowed.set()
        self._gateway_failover_lock = asyncio.Lock()
        self._deferred_chat_ctx: llm.ChatContext | None = None
        self._deferred_chat_ctx_tasks: set[asyncio.Task[None]] = set()
        self._pending_interrupt_futures: set[asyncio.Future[None]] = set()
        self._failover_support_closing = False
        super().__init__(realtime_model, turn_detection_disabled=turn_detection_disabled)

    def _create_ws_url_and_headers(self) -> tuple[str, dict[str, str]]:
        opts = self._inference_model._inference_opts
        headers = get_inference_headers(inference_class=opts.inference_class)
        headers["Authorization"] = f"Bearer {create_access_token(opts.api_key, opts.api_secret)}"
        headers[_FAILOVER_PROTOCOL_HEADER] = str(_FAILOVER_PROTOCOL_VERSION)
        if opts.provider:
            headers[HEADER_INFERENCE_PROVIDER] = opts.provider
        return process_base_url(self._opts.base_url, self._opts.model), headers

    async def _wait_before_live_send(self) -> None:
        await self._live_forwarding_allowed.wait()

    async def _send_ws_event(
        self,
        ws_conn: aiohttp.ClientWebSocketResponse,
        event: Any,
    ) -> None:
        await super()._send_ws_event(ws_conn, event)

        event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
        event_id = (
            event.get("event_id") if isinstance(event, dict) else getattr(event, "event_id", None)
        )
        audio = event.get("audio") if isinstance(event, dict) else getattr(event, "audio", None)
        if (
            event_type == "input_audio_buffer.append"
            and isinstance(audio, str)
            and not (
                isinstance(event_id, str) and event_id.startswith(_REPLAY_AUDIO_EVENT_ID_PREFIX)
            )
        ):
            self._record_sent_input_audio(base64.b64decode(audio))

    def _record_sent_input_audio(self, data: bytes) -> None:
        """Retain audio only after it reached the provider-facing WebSocket.

        Events still queued in ``_msg_ch`` resume normally after failover. If
        they were included here too, replay would send them once directly and
        the normal sender would send them a second time after the gate reopened.
        """
        if not self._audio_buffer_overflowed:
            if not any(data):
                remaining_silence = (
                    _MAX_REPLAY_TRAILING_SILENCE_BYTES - self._replay_trailing_silence_bytes
                )
                if remaining_silence <= 0:
                    return
                data = data[:remaining_silence]
                self._replay_trailing_silence_bytes += len(data)
            else:
                self._replay_trailing_silence_bytes = 0

            next_size = self._uncommitted_audio_bytes + len(data)
            if next_size <= self._inference_model._inference_opts.max_uncommitted_audio_bytes:
                self._uncommitted_audio.append(data)
                self._uncommitted_audio_bytes = next_size
            else:
                # A truncated turn is unsafe to replay, but losing failover
                # fidelity must not terminate an otherwise healthy live call.
                self._uncommitted_audio.clear()
                self._uncommitted_audio_bytes = 0
                self._audio_buffer_overflowed = True
                self._emit_error(
                    llm.RealtimeError(
                        "active-turn failover replay disabled until the next audio commit: "
                        "max_uncommitted_audio_bytes exceeded"
                    ),
                    recoverable=True,
                )

    def _clear_uncommitted_audio(self) -> None:
        self._uncommitted_audio.clear()
        self._uncommitted_audio_bytes = 0
        self._audio_buffer_overflowed = False
        self._replay_trailing_silence_bytes = 0
        self._active_user_turn_uncommitted = False

    def start_user_activity(self) -> None:
        super().start_user_activity()
        self._mark_user_activity()

    def _mark_user_activity(self) -> None:
        self._active_user_turn_uncommitted = True

    def commit_audio(self) -> None:
        super().commit_audio()
        self._user_turn_commit_count += 1
        self._clear_uncommitted_audio()

    def clear_audio(self) -> None:
        super().clear_audio()
        self._clear_uncommitted_audio()

    def interrupt(self) -> None:
        if self._gateway_failover_in_progress:
            self._close_current_generation("gateway failover")
            return
        super().interrupt()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if self._gateway_failover_in_progress:
            return
        super().truncate(
            message_id=message_id,
            modalities=modalities,
            audio_end_ms=audio_end_ms,
            audio_transcript=audio_transcript,
        )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        if self._gateway_failover_in_progress:
            # Keep the latest full snapshot. The failover handler includes it
            # in the replay if it arrives before replay starts, or schedules a
            # normal sync afterward if it arrives while replay is in flight.
            # Waiting for an acknowledgement while sends are paused would
            # consume the gateway replay timeout.
            self._deferred_chat_ctx = chat_ctx.copy()
            return
        await super().update_chat_ctx(chat_ctx)

    async def _run_deferred_chat_ctx_sync(self, chat_ctx: llm.ChatContext) -> None:
        try:
            await self.update_chat_ctx(chat_ctx)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._emit_error(
                llm.RealtimeError(f"failed to synchronize chat context after failover: {e}"),
                recoverable=True,
            )

    def _sync_deferred_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        if self._failover_support_closing:
            return
        task = asyncio.create_task(self._run_deferred_chat_ctx_sync(chat_ctx))
        self._deferred_chat_ctx_tasks.add(task)
        task.add_done_callback(self._deferred_chat_ctx_tasks.discard)

    def _track_pending_interrupt(self, future: asyncio.Future[None]) -> None:
        self._pending_interrupt_futures.add(future)

        def _done(completed: asyncio.Future[None]) -> None:
            self._pending_interrupt_futures.discard(completed)
            if not completed.cancelled():
                completed.exception()

        future.add_done_callback(_done)

    async def _interrupt_agent_for_failover(self, timeout: float) -> None:
        if self._agent_session is None or self._failover_support_closing:
            return

        try:
            future = asyncio.ensure_future(self._agent_session.interrupt(force=True))
        except Exception:
            return
        self._track_pending_interrupt(future)
        done, _ = await asyncio.wait((future,), timeout=timeout)
        if future in done:
            with contextlib.suppress(Exception):
                future.result()
            return

        self._emit_error(
            llm.RealtimeError(
                f"timed out interrupting the agent after {timeout:.3f}s; "
                "continuing gateway failover replay"
            ),
            recoverable=True,
        )

    @staticmethod
    async def _await_before_deadline(awaitable: Awaitable[_T], deadline: float) -> _T:
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        return await asyncio.wait_for(awaitable, timeout=remaining)

    async def aclose(self) -> None:
        # Set before taking snapshots; no new deferred work may register after
        # this point without an intervening await.
        self._failover_support_closing = True
        deferred_tasks = tuple(self._deferred_chat_ctx_tasks)
        pending_interrupts = tuple(self._pending_interrupt_futures)
        for task in deferred_tasks:
            task.cancel()
        for future in pending_interrupts:
            future.cancel()
        await super().aclose()
        if deferred_tasks:
            await asyncio.gather(*deferred_tasks, return_exceptions=True)
        if pending_interrupts:
            await asyncio.gather(*pending_interrupts, return_exceptions=True)

    async def _handle_extra_server_event(
        self, event: dict[str, Any], ws_conn: aiohttp.ClientWebSocketResponse
    ) -> bool:
        event_type = event.get("type")
        if event_type == "input_audio_buffer.speech_started":
            self._mark_user_activity()
            return False
        if event_type == "input_audio_buffer.committed":
            self._clear_uncommitted_audio()
            return False
        if event_type != "livekit.session.failover":
            return False

        version = event.get("protocol_version")
        if type(version) is not int or version != _FAILOVER_PROTOCOL_VERSION:
            raise APIConnectionError(
                f"unsupported realtime failover protocol version: {version!r}",
                retryable=False,
            )

        replay_timeout_ms = event.get("replay_timeout_ms")
        if (
            type(replay_timeout_ms) is not int
            or replay_timeout_ms <= 0
            or replay_timeout_ms > _MAX_REPLAY_TIMEOUT_MS
        ):
            raise APIConnectionError(
                f"invalid realtime failover replay_timeout_ms: {replay_timeout_ms!r}",
                retryable=False,
            )

        context_lost = event.get("context_lost")
        if type(context_lost) is not bool:
            raise APIConnectionError(
                f"invalid realtime failover context_lost: {context_lost!r}",
                retryable=False,
            )

        await self._handle_gateway_failover(
            ws_conn,
            replay_timeout_ms=replay_timeout_ms,
            context_lost=context_lost,
        )
        return True

    async def _handle_gateway_failover(
        self,
        ws_conn: aiohttp.ClientWebSocketResponse,
        *,
        replay_timeout_ms: int,
        context_lost: bool = True,
    ) -> None:
        async with self._gateway_failover_lock:
            replay_timeout = replay_timeout_ms / 1000
            safety_margin = min(
                _MAX_REPLAY_DEADLINE_SAFETY_SECONDS,
                replay_timeout * _REPLAY_DEADLINE_SAFETY_FRACTION,
            )
            operation_timeout = replay_timeout - safety_margin
            deadline = asyncio.get_running_loop().time() + operation_timeout

            if not context_lost:
                send_lock_acquired = False
                try:
                    await self._await_before_deadline(self._ws_send_lock.acquire(), deadline)
                    send_lock_acquired = True
                    await self._await_before_deadline(
                        self._send_ws_event(ws_conn, {"type": "livekit.session.replay_completed"}),
                        deadline,
                    )
                except asyncio.TimeoutError as e:
                    raise APIConnectionError(
                        "timed out acknowledging context-preserving realtime failover"
                    ) from e
                except Exception as e:
                    if isinstance(e, APIConnectionError):
                        raise
                    raise APIConnectionError(
                        "failed to acknowledge context-preserving realtime failover"
                    ) from e
                finally:
                    if send_lock_acquired:
                        self._ws_send_lock.release()
                return

            self._gateway_failover_in_progress = True
            self._live_forwarding_allowed.clear()
            user_turn_commit_count = self._user_turn_commit_count
            interrupt_timeout = min(
                _MAX_INTERRUPT_TIMEOUT_SECONDS,
                operation_timeout * _INTERRUPT_TIMEOUT_FRACTION,
            )
            should_regenerate = self.has_active_generation or (
                self._agent_session is not None
                and self._agent_session.agent_state in ("speaking", "thinking")
            )
            old_remote_chat_ctx: llm.remote_chat_context.RemoteChatContext | None = None
            replay_chat_ctx: llm.ChatContext | None = None
            send_lock_acquired = False

            try:
                try:
                    await self._await_before_deadline(self._ws_send_lock.acquire(), deadline)
                    send_lock_acquired = True

                    # Clearing _live_forwarding_allowed before taking this lock
                    # prevents new sends. Any append already inside send_str()
                    # finishes first and records itself, so this snapshot cannot
                    # omit bytes the failed provider actually observed.
                    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                    await self._interrupt_agent_for_failover(min(interrupt_timeout, remaining))

                    chat_ctx = self._deferred_chat_ctx
                    self._deferred_chat_ctx = None
                    if chat_ctx is None:
                        chat_ctx = (
                            self._agent_session.current_agent.chat_ctx.copy()
                            if self._agent_session is not None
                            else self.chat_ctx.copy()
                        )
                    replay_chat_ctx = chat_ctx
                    replay_audio = tuple(self._uncommitted_audio)

                    for fut in self._response_created_futures.values():
                        if not fut.done():
                            fut.set_exception(
                                llm.RealtimeError(
                                    "pending response discarded due to gateway failover"
                                )
                            )
                    self._response_created_futures.clear()
                    self._discarded_event_ids.clear()
                    self._close_current_generation("gateway failover")

                    events, old_remote_chat_ctx = self._prepare_connection_replay(
                        include_session_state=False,
                        chat_ctx=chat_ctx,
                    )

                    for replay_event in events:
                        await self._await_before_deadline(
                            self._send_ws_event(ws_conn, replay_event), deadline
                        )
                    for chunk in replay_audio:
                        await self._await_before_deadline(
                            self._send_ws_event(
                                ws_conn,
                                {
                                    "type": "input_audio_buffer.append",
                                    "event_id": _REPLAY_AUDIO_EVENT_ID_PREFIX + uuid.uuid4().hex,
                                    "audio": base64.b64encode(chunk).decode("utf-8"),
                                },
                            ),
                            deadline,
                        )
                    await self._await_before_deadline(
                        self._send_ws_event(ws_conn, {"type": "livekit.session.replay_completed"}),
                        deadline,
                    )
                except asyncio.TimeoutError as e:
                    if old_remote_chat_ctx is not None:
                        self._remote_chat_ctx = old_remote_chat_ctx
                    if self._deferred_chat_ctx is None and replay_chat_ctx is not None:
                        self._deferred_chat_ctx = replay_chat_ctx
                    raise APIConnectionError(
                        "timed out replaying realtime session after gateway failover"
                    ) from e
                except Exception as e:
                    if old_remote_chat_ctx is not None:
                        self._remote_chat_ctx = old_remote_chat_ctx
                    if self._deferred_chat_ctx is None and replay_chat_ctx is not None:
                        self._deferred_chat_ctx = replay_chat_ctx
                    if isinstance(e, APIConnectionError):
                        raise
                    raise APIConnectionError(
                        "failed to replay realtime session after gateway failover"
                    ) from e
                finally:
                    if send_lock_acquired:
                        self._ws_send_lock.release()
            finally:
                self._gateway_failover_in_progress = False
                self._live_forwarding_allowed.set()
                deferred_after_replay = self._deferred_chat_ctx
                self._deferred_chat_ctx = None
                if deferred_after_replay is not None:
                    self._sync_deferred_chat_ctx(deferred_after_replay)

            if (
                should_regenerate
                and not self._active_user_turn_uncommitted
                and self._user_turn_commit_count == user_turn_commit_count
                and self._agent_session is not None
            ):
                self._agent_session.generate_reply()
