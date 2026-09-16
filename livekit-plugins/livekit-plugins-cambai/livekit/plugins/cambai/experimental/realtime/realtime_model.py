# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camb.ai realtime speech-to-speech translation as a LiveKit ``RealtimeModel``."""

from __future__ import annotations

import asyncio
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from camb.realtime import (
    AudioDeltaEvent,
    ClosedEvent,
    ErrorEvent,
    RealtimeError,
    RealtimeSession as CambSession,
    ServerEventType,
    TextDeltaEvent,
    TextDoneEvent,
    TranscriptCompletedEvent,
    connect as camb_connect,
)

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ...log import logger
from ...models import DEFAULT_REALTIME_MODE, NUM_CHANNELS, REALTIME_SAMPLE_RATE, RealtimeMode


@dataclass
class _RealtimeOptions:
    source_language: str
    target_language: str
    mode: RealtimeMode
    voice_id: int | None
    api_key: str
    base_url: str | None


@dataclass
class _Generation:
    """One translated utterance in flight."""

    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    started_at: float
    text_done: bool = False
    audio_done: bool = False


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        source_language: str,
        target_language: str,
        mode: RealtimeMode = DEFAULT_REALTIME_MODE,
        voice_id: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Translate speech to speech with Camb.ai.

        Args:
            source_language: BCP-47 tag of the speaker's language, e.g. ``"en-US"``.
            target_language: BCP-47 tag to translate into, e.g. ``"es-ES"``.
            mode: ``"fast"`` (default) or ``"slow"``; see ``RealtimeMode``.
            voice_id: Synthesize the translation with one of your cloned voices. When
                omitted the server picks a built-in voice for ``target_language``.
            api_key: Camb.ai API key. Falls back to the ``CAMB_API_KEY`` env var.
            base_url: Override the realtime endpoint, e.g. to reach a non-production
                deployment. Defaults to whatever the installed SDK points at.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                # The endpoint segments utterances itself, so the session must not run its
                # own barge-in detection: a translator's speaker never stops talking.
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            )
        )

        camb_api_key = api_key or os.environ.get("CAMB_API_KEY")
        if not camb_api_key:
            raise ValueError(
                "Camb.ai API key is required, either as `api_key` or by setting the "
                "CAMB_API_KEY environment variable"
            )

        self._opts = _RealtimeOptions(
            source_language=source_language,
            target_language=target_language,
            mode=mode,
            voice_id=voice_id,
            api_key=camb_api_key,
            base_url=base_url,
        )
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return f"camb-realtime-{self._opts.mode}"

    @property
    def provider(self) -> str:
        return "Camb.ai"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        for sess in list(self._sessions):
            await sess.aclose()


class RealtimeSession(llm.RealtimeSession[Literal["cambai_server_event_received"]]):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = realtime_model._opts

        self._msg_ch = utils.aio.Chan[bytes]()
        self._current: _Generation | None = None
        self._input_resampler: rtc.AudioResampler | None = None
        # The server sends 400ms blobs; the room pipeline expects small, even frames.
        self._bstream = utils.audio.AudioByteStream(
            REALTIME_SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=REALTIME_SAMPLE_RATE // 10
        )
        self._chat_ctx = llm.ChatContext.empty()
        self._pending_reply: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._turn_started_at: float | None = None
        self._item_id = 0

        self._main_atask = asyncio.create_task(self._main_task(), name="cambai-realtime")

    async def _main_task(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("the camb.ai realtime session ended with an error")
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label=self._realtime_model.label,
                    error=e,
                    recoverable=False,
                ),
            )
            raise

    async def _run(self) -> None:
        try:
            overrides: dict[str, Any] = (
                {"base_url": self._opts.base_url} if self._opts.base_url else {}
            )
            session = await camb_connect(
                self._opts.api_key,
                **overrides,
                source_language=self._opts.source_language,
                target_language=self._opts.target_language,
                mode=self._opts.mode,
                voice_id=self._opts.voice_id,
            )
        except RealtimeError as e:
            raise APIConnectionError("failed to connect to the Camb.ai realtime endpoint") from e

        self._subscribe(session)

        tasks = [asyncio.create_task(self._send_task(session), name="cambai-realtime-send")]
        try:
            await asyncio.gather(session.run_until_closed(), *tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await session.close()
            self._finish_generation()
            self._fail_pending_reply()

    def _subscribe(self, session: CambSession) -> None:
        def on_transcript(event: TranscriptCompletedEvent) -> None:
            self._emit_input_transcript(event.transcript)

        def on_text_delta(event: TextDeltaEvent) -> None:
            gen = self._ensure_generation()
            if event.delta:
                gen.text_ch.send_nowait(event.delta)

        def on_text_done(_: TextDoneEvent) -> None:
            gen = self._ensure_generation()
            gen.text_done = True
            self._finish_if_complete(gen)

        def on_audio_delta(event: AudioDeltaEvent) -> None:
            if event.data:
                self._push_audio(event.data)

        def on_audio_done(_: object) -> None:
            current = self._current
            if current is not None:
                current.audio_done = True
                self._finish_if_complete(current)

        def on_error(event: ErrorEvent) -> None:
            logger.error("camb.ai realtime error: %s", event.message)
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label=self._realtime_model.label,
                    error=APIStatusError(event.message, status_code=500, body=None),
                    recoverable=True,
                ),
            )

        def on_closed(event: ClosedEvent) -> None:
            if not session.is_ready:
                logger.error(
                    "camb.ai realtime closed before the session became ready: %s %s",
                    event.code,
                    event.reason,
                )

        session.on(ServerEventType.TRANSCRIPT_COMPLETED, on_transcript)
        session.on(ServerEventType.TEXT_DELTA, on_text_delta)
        session.on(ServerEventType.TEXT_DONE, on_text_done)
        session.on(ServerEventType.AUDIO_DELTA, on_audio_delta)
        session.on(ServerEventType.AUDIO_DONE, on_audio_done)
        session.on(ServerEventType.ERROR, on_error)
        session.on(ServerEventType.CLOSED, on_closed)

    async def _send_task(self, session: CambSession) -> None:
        # Audio sent before the session exists is discarded by the server.
        try:
            await session.wait_until_ready(timeout=None)
        except RealtimeError as e:
            raise APIConnectionError(
                "the Camb.ai realtime session closed before it became ready"
            ) from e

        async for pcm in self._msg_ch:
            await session.send_audio(pcm)

    def _finish_if_complete(self, gen: _Generation) -> None:
        """A response ends when the server has reported both its text and its audio done."""
        if gen.text_done and gen.audio_done:
            self._finish_generation()

    def _ensure_generation(self) -> _Generation:
        if self._current is not None:
            return self._current

        self._item_id += 1
        message_id = f"camb-translation-{self._item_id}"
        gen = _Generation(
            message_id=message_id,
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            modalities=asyncio.Future[list[Literal["text", "audio"]]](),
            started_at=self._turn_started_at or time.time(),
        )
        gen.modalities.set_result(["audio", "text"])
        self._current = gen

        gen.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=message_id,
                text_stream=gen.text_ch,
                audio_stream=gen.audio_ch,
                modalities=gen.modalities,
            )
        )

        pending = self._pending_reply
        if pending is not None and pending.done():
            pending = None

        ev = llm.GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=gen.function_ch,
            user_initiated=pending is not None,
            response_id=message_id,
        )
        if pending is not None:
            pending.set_result(ev)
            self._pending_reply = None
        self.emit("generation_created", ev)
        return gen

    def _push_audio(self, data: bytes) -> None:
        if not data:
            return
        gen = self._ensure_generation()
        for frame in self._bstream.push(data):
            gen.audio_ch.send_nowait(frame)

    def _fail_pending_reply(self) -> None:
        pending, self._pending_reply = self._pending_reply, None
        if pending is not None and not pending.done():
            pending.set_exception(
                llm.RealtimeError("the camb.ai realtime session closed before a translation")
            )

    def _finish_generation(self) -> None:
        gen, self._current = self._current, None
        if gen is None:
            return
        for frame in self._bstream.flush():
            gen.audio_ch.send_nowait(frame)
        for ch in (gen.text_ch, gen.audio_ch, gen.message_ch, gen.function_ch):
            if not ch.closed:
                ch.close()
        self._turn_started_at = None

    def _emit_input_transcript(self, transcript: str) -> None:
        self._item_id += 1
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=f"camb-source-{self._item_id}",
                transcript=transcript,
                is_final=True,
                turn_started_at=self._turn_started_at,
            ),
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._turn_started_at is None:
            self._turn_started_at = time.time()
        for f in self._resample(frame):
            self._msg_ch.send_nowait(f.data.tobytes())

    def _resample(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler and frame.sample_rate != self._input_resampler._input_rate:
            self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != REALTIME_SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=REALTIME_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Resolve with the next translation the server produces.

        Translation is driven by the incoming speech, so this does not prompt the server;
        it hands back the generation that the next utterance creates. ``instructions`` is
        not supported and is ignored.
        """
        if is_given(instructions):
            logger.warning("camb.ai realtime translation ignores per-reply instructions")

        if self._pending_reply is not None and not self._pending_reply.done():
            return self._pending_reply

        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._pending_reply = fut
        return fut

    def interrupt(self) -> None:
        pass

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        # AgentSession sets instructions on startup; a translation has none to steer.
        if instructions:
            logger.warning("camb.ai realtime translation ignores instructions")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        pass

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        if tools:
            raise llm.RealtimeError("Camb.ai realtime translation does not support tools")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    async def aclose(self) -> None:
        self._msg_ch.close()
        self._fail_pending_reply()
        await utils.aio.cancel_and_wait(self._main_atask)


__all__ = ["RealtimeModel", "RealtimeSession"]
