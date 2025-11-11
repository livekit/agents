from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from asyncio import Queue
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Awaitable, Callable, Optional, Sequence

from livekit import rtc
from livekit.agents import stt, vad
from livekit.agents.metrics.base import VADMetrics


_logger = logging.getLogger(__name__)


class AgentSpeechState(Enum):
    IDLE = auto()
    SPEAKING = auto()


@dataclass
class VADExtensionConfig:
    max_interruption_silence: float = 0.6
    """Maximum silence duration (seconds) before an utterance is considered complete."""

    pre_speech_duration: float = 0.4
    """Amount of recent audio (seconds) to prepend when speech begins."""

    decision_timeout: float = 0.7
    """Decision window duration in seconds for interruption detection."""

    ignored_words: list[str] = field(
        default_factory=lambda: ["uh", "um", "umm", "hmm", "haan"]
    )
    """Words that should be ignored during interruption detection."""

class VADExtension(vad.VAD):
    """Wraps a base VAD implementation with buffering and pause-aware utterance detection."""

    def __init__(
        self,
        base_vad: vad.VAD,
        *,
        config: VADExtensionConfig | None = None,
    ) -> None:
        super().__init__(capabilities=base_vad.capabilities)
        self._base_vad = base_vad
        self._config = config or VADExtensionConfig()

        self._base_vad.on("metrics_collected", self._forward_metrics)
        self._agent_state = AgentSpeechState.IDLE
        self._asr_factory: Optional[Callable[[], stt.RecognizeStream]] = None
        self._on_interruption: Optional[Callable[[str], Awaitable[None] | None]] = None

    def set_agent_state(self, state: AgentSpeechState) -> None:
        self._agent_state = state

    def set_asr_stream_factory(
        self, factory: Optional[Callable[[], stt.RecognizeStream]]
    ) -> None:
        self._asr_factory = factory

    def set_interruption_handler(
        self, handler: Optional[Callable[[str], Awaitable[None] | None]]
    ) -> None:
        self._on_interruption = handler

    def stream(self) -> vad.VADStream:
        base_stream = self._base_vad.stream()
        return _VADExtensionStream(
            extension=self,
            base_stream=base_stream,
            config=self._config,
        )

    def _forward_metrics(self, metrics: VADMetrics) -> None:
        # Forward metrics from the wrapped VAD so consumers can subscribe normally.
        self.emit("metrics_collected", metrics)


class _VADExtensionStream(vad.VADStream):
    def __init__(
        self,
        *,
        extension: VADExtension,
        base_stream: vad.VADStream,
        config: VADExtensionConfig,
    ) -> None:
        self._extension = extension
        self._base_stream = base_stream
        self._config = config
        self._ignored_words = {
            word.strip().lower() for word in config.ignored_words if word.strip()
        }

        self._intercept_active = False
        self._intercept_queue: Queue[rtc.AudioFrame | None] | None = None
        self._asr_stream: stt.RecognizeStream | None = None
        self._asr_task: asyncio.Task[None] | None = None
        self._intercept_transcripts: list[str] = []
        self._partial_history: list[str] = []
        self._decision_task: asyncio.Task[None] | None = None

        super().__init__(extension)



    async def _main_task(self) -> None:
        try:
            await asyncio.gather(
                self._forward_input_loop(),
                self._consume_base_events(),
            )
        finally:
            await self._finalize_intercept(force=True)
            await self._base_stream.aclose()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._intercept_active and self._intercept_queue is not None:
            try:
                self._intercept_queue.put_nowait(frame)
            except asyncio.QueueFull:
                _logger.warning("intercept queue is full; dropping audio frame")
        super().push_frame(frame)

    def flush(self) -> None:
        super().flush()

    def end_input(self) -> None:
        super().end_input()

    async def aclose(self) -> None:
        await self._finalize_intercept(force=True)
        await self._base_stream.aclose()
        await super().aclose()

    async def _forward_input_loop(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, vad.VADStream._FlushSentinel):
                self._base_stream.flush()
                continue

            frame: rtc.AudioFrame = item
            self._base_stream.push_frame(frame)

        self._base_stream.end_input()

    async def _consume_base_events(self) -> None:
        async for event in self._base_stream:
            should_forward = await self._handle_vad_event(event)
            if should_forward:
                self._event_ch.send_nowait(event)

    async def _handle_vad_event(self, event: vad.VADEvent) -> bool:
        if event.type == vad.VADEventType.START_OF_SPEECH:
            if self._extension._agent_state == AgentSpeechState.SPEAKING:
                if await self._start_intercept(event):
                    return False
            return True

        if event.type == vad.VADEventType.END_OF_SPEECH:
            if self._intercept_active:
                await self._finalize_intercept()
                return False
            return True

        return not self._intercept_active

    async def _start_intercept(self, event: vad.VADEvent) -> bool:
        if self._intercept_active:
            return True

        if self._extension._asr_factory is None:
            _logger.warning("ASR factory not configured; cannot intercept user speech.")
            return False

        try:
            self._asr_stream = self._extension._asr_factory()
        except Exception:
            _logger.exception("Failed to create ASR stream for interruption handling.")
            self._asr_stream = None
            return False

        self._intercept_active = True
        self._intercept_queue = Queue(maxsize=512)
        self._intercept_transcripts = []
        self._partial_history = []
        self._pending_valid_transcript = None
        self._interruption_detected = False
        self._capturing = False
        self._asr_task = asyncio.create_task(self._run_intercept_asr(), name="vad_extension_asr")
        self._decision_task = asyncio.create_task(
            self._decision_timer(), name="vad_extension_decision_timer"
        )

        if event.frames:
            for frame in event.frames:
                try:
                    self._intercept_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    _logger.warning("intercept queue is full; dropping initial audio frame")

        return True

    async def _finalize_intercept(self, force: bool = False) -> None:
        if not self._intercept_active and not force:
            return

        current_task = asyncio.current_task()
        decision_task = self._decision_task
        self._decision_task = None
        if decision_task:
            if decision_task is not current_task:
                decision_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await decision_task
            else:
                decision_task.cancel()

        queue = self._intercept_queue
        if queue is not None:
            with contextlib.suppress(Exception):
                await queue.put(None)

        if self._asr_task and self._asr_task is not current_task:
            with contextlib.suppress(Exception):
                await self._asr_task

        self._intercept_active = False
        self._intercept_queue = None
        self._asr_task = None
        self._asr_stream = None
        self._partial_history = []
        self._intercept_transcripts = []
        self._pending_valid_transcript = None
        self._interruption_detected = False

    async def _run_intercept_asr(self) -> None:
        if self._intercept_queue is None or self._asr_stream is None:
            return

        stream = self._asr_stream
        queue = self._intercept_queue
        transcripts: list[str] = []

        async def _feeder() -> None:
            while True:
                frame = await queue.get()
                if frame is None:
                    break
                stream.push_frame(frame)
            stream.end_input()

        async def _listener() -> None:
            async for ev in stream:
                if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT and ev.alternatives:
                    await self._handle_partial_transcript(ev.alternatives[0].text)
                elif ev.type == stt.SpeechEventType.PREFLIGHT_TRANSCRIPT and ev.alternatives:
                    await self._handle_partial_transcript(ev.alternatives[0].text)
                elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                    text = ev.alternatives[0].text
                    transcripts.append(text)
                    await self._handle_partial_transcript(text)
                elif ev.type == stt.SpeechEventType.END_OF_SPEECH:
                    break

        try:
            await asyncio.gather(_feeder(), _listener())
        except Exception:
            _logger.exception("Error while running interruption ASR stream.")
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

            self._intercept_transcripts = transcripts
            if transcripts:
                transcript_text = " ".join(transcripts).strip()
                _logger.debug(
                    "Interruption transcript captured",
                    extra={"raw_transcript": transcript_text},
                )

    async def _decision_timer(self) -> None:
        try:
            await asyncio.sleep(self._extension._config.decision_timeout)
        except asyncio.CancelledError:
            return

        if self._interruption_detected:
            return

        fillers = " ".join(self._partial_history)
        if fillers:
            _logger.info(
                "Ignoring filler words: '%s'. Agent continues.",
                fillers,
            )
        await self._finalize_intercept()

    async def _handle_partial_transcript(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        self._partial_history.append(cleaned)
        normalized_tokens = [
            "".join(ch for ch in token.lower() if ch.isalnum())
            for token in cleaned.split()
        ]
        real_word = any(token and token not in self._ignored_words for token in normalized_tokens)

        if real_word:
            self._interruption_detected = True
            self._pending_valid_transcript = cleaned
            await self._on_real_interruption()

    async def _on_real_interruption(self) -> None:
        transcript = self._pending_valid_transcript or ""
        _logger.info("Valid interruption detected: '%s'. Pausing agent.", transcript)
        await self._notify_interruption(transcript)
        self._extension.set_agent_state(AgentSpeechState.IDLE)
        await self._finalize_intercept()

    async def _notify_interruption(self, transcript: str) -> None:
        if not self._extension._on_interruption:
            return

        try:
            maybe_coro = self._extension._on_interruption(transcript)
            if inspect.isawaitable(maybe_coro):
                await maybe_coro
        except Exception:
            _logger.exception("Interruption handler raised an exception.")


