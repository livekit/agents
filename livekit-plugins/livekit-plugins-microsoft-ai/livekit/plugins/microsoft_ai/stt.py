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
#
# Adapted from the OpenAI STT plugin: Microsoft AI transcription only, with
# client-owned VAD boundaries, ordered audio drain and explicit finalization.

from __future__ import annotations

import asyncio
import base64
import math
import weakref
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import aiohttp
from pydantic import BaseModel, ConfigDict, ValidationError

from livekit import rtc
from livekit.agents import (
    VAD,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    LanguageCode,
    VADEvent,
    VADEventType,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

from ._http import Configuration, HTTPClient, positive_timeout, status_error
from .log import logger

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
_TRANSCRIPTION = "conversation.item.input_audio_transcription."
_MAX_TEXT_LENGTH = 65536


class _WSMessage(Protocol):
    @property
    def type(self) -> aiohttp.WSMsgType: ...

    @property
    def data(self) -> object: ...


class _WebSocket(Protocol):
    async def send_json(self, data: dict[str, object]) -> None: ...

    async def receive(self) -> _WSMessage: ...

    async def close(self) -> bool: ...


def _session_update(model: str, language: str | None) -> dict[str, object]:
    transcription = {"model": model}
    if language is not None:
        transcription["language"] = language
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": transcription,
                    "turn_detection": None,
                    "noise_reduction": None,
                }
            },
        },
    }


class _ProviderError(BaseModel):
    model_config = ConfigDict(strict=True)
    status_code: object = None
    code: object = None


class _Event(BaseModel):
    model_config = ConfigDict(strict=True)
    type: str
    error: _ProviderError | None = None
    event_id: object = None
    item_id: object = None
    intermediate: object = None
    delta: object = None
    transcript: object = None


def _string(value: object, key: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise APIError(f"Microsoft AI STT event requires a string {key}", retryable=False)
    if len(value) > _MAX_TEXT_LENGTH:
        raise APIError("Microsoft AI STT event exceeds the text limit", retryable=False)
    return value


def _provider_error(error: _ProviderError | None) -> APIError:
    if error is not None:
        status = error.status_code
        if isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 600:
            return status_error("STT", status)
        code = error.code
        if isinstance(code, str):
            status = {
                "invalid_api_key": 401,
                "rate_limit_exceeded": 429,
                "content_filter": 403,
                "safety_violation": 403,
            }.get(code)
            if status is not None:
                return status_error("STT", status)
    # Do not put provider messages/bodies in errors: they can echo audio, text, URLs or keys.
    return APIError("Microsoft AI STT rejected the transcription request", retryable=False)


@dataclass
class _Item:
    id: str
    finalized: str = ""
    hypothesis: str = ""
    last_interim: str = ""


class STT(stt.STT):
    """Native streaming Microsoft AI transcription with explicit client commits.

    See the package README for deployment-specific contract and validation limits.

    Args:
        vad: A VAD emitting ordered INFERENCE_DONE events even during silence,
            with input-relative timestamps, and START_OF_SPEECH
            frames containing the detected onset/prefix through that timestamp
            (e.g. LiveKit's Silero VAD). Idle audio is withheld; these public frames
            recover the onset even when the start notification is delayed.
            Required explicitly: pass None only when driving flush/end_input yourself.
            VoiceSession's separate VAD does not commit a native STT stream.
        url: Full WebSocket URL, or MICROSOFT_AI_STT_URL. No paths/query parameters
            are added automatically.
        model: Deployment model ID, or MICROSOFT_AI_STT_MODEL. No model is assumed.
        api_key: Credential, or MICROSOFT_AI_STT_API_KEY, sent using auth_header.
        auth_header: Authorization (the default, with Bearer prefix) or api-key
            (raw credential), or MICROSOFT_AI_STT_AUTH_HEADER. No auth fallback occurs.
        headers: Explicit authentication headers instead of api_key/auth_header
            environment lookup. Cannot be combined with either constructor argument.
        language: Optional transcription language hint, or MICROSOFT_AI_STT_LANGUAGE.
        http_session: Optional caller-owned aiohttp session.
        env_file: Explicit dotenv file, or MICROSOFT_AI_ENV_FILE. Constructor
            arguments override environment variables, which override this file.
        max_buffered_audio: Client-side queued-audio and VAD start-prefix limits
            in seconds. Queue overflow or an oversized VAD prefix fails explicitly
            instead of silently dropping speech.
    """

    def __init__(
        self,
        *,
        vad: VAD | None,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        auth_header: Literal["Authorization", "api-key"] | None = None,
        headers: Mapping[str, str] | None = None,
        language: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        env_file: str | Path | None = None,
        max_buffered_audio: float = 5.0,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, offline_recognize=False
            )
        )
        config = Configuration(env_file)
        positive_timeout(max_buffered_audio, "max_buffered_audio")
        if language is None:
            language = config.get("MICROSOFT_AI_STT_LANGUAGE") or None
        if language is not None and not language.strip():
            raise ValueError("language must be nonempty when supplied")
        self._model = config.required(model, "MICROSOFT_AI_STT_MODEL")
        self._client = HTTPClient(
            config=config,
            service="STT",
            url=url,
            api_key=api_key,
            headers=headers,
            http_session=http_session,
            auth_header=auth_header,
        )
        self._vad = vad
        self._language = language
        self._max_buffered_audio = max_buffered_audio
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()
        self._closed = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Microsoft AI"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Microsoft AI STT supports stream(), not batch recognize()")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a transcription stream; flush commits a turn, end_input also drains finals."""
        if self._closed:
            raise RuntimeError("Microsoft AI STT is closed")
        selected_language = language if utils.is_given(language) else self._language
        if selected_language is not None and not selected_language.strip():
            raise ValueError("language must be nonempty when supplied")
        stream = SpeechStream(
            stt=self,
            client=self._client,
            max_buffered_audio=self._max_buffered_audio,
            vad=self._vad,
            language=selected_language,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        self._closed = True
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))
        await self._client.aclose()


class SpeechStream(stt.RecognizeStream):
    """A socket with ordered audio/commit writes and concurrent transcript reception."""

    def __init__(
        self,
        *,
        stt: STT,
        client: HTTPClient,
        max_buffered_audio: float,
        vad: VAD | None,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._stt: STT = stt
        self._client = client
        self._max_buffered_audio = max_buffered_audio
        self._vad = vad
        self._language_hint = language
        self._language = LanguageCode(language or "")
        self._input_duration = 0.0
        self._processed_samples = 0
        self._segment_samples = 0
        self._input_consumed = False
        self._input_error: APIError | None = None
        self._closed = False
        self._speaking = False
        self._item: _Item | None = None
        self._commit_future: asyncio.Future[None] | None = None
        self._committed_item_id: str | None = None
        self._finished_items: deque[str] = deque(maxlen=128)
        self._event_ids: deque[str] = deque(maxlen=256)
        self._ws: _WebSocket | None = None

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._check_input_not_ended()
        self._check_not_closed()
        if frame.num_channels != NUM_CHANNELS:
            raise ValueError("Microsoft AI STT requires mono audio")
        if frame.samples_per_channel <= 0 or frame.sample_rate <= 0:
            raise ValueError("Microsoft AI STT requires nonempty audio frames")
        buffered = self._input_duration - self._processed_samples / SAMPLE_RATE
        if buffered + frame.duration > self._max_buffered_audio or self.input.qsize() >= 1024:
            self._input_error = APIConnectionError(
                "Microsoft AI STT audio buffer is full; pace input or raise max_buffered_audio",
                retryable=False,
            )
            self.input.close()
            raise self._input_error
        super().push_frame(frame)
        self._input_duration += frame.duration

    def flush(self) -> None:
        if self.input.qsize() >= 1024:
            self._input_error = APIConnectionError(
                "Microsoft AI STT input queue is full", retryable=False
            )
            self.input.close()
            raise self._input_error
        super().flush()

    async def aclose(self) -> None:
        self._closed = True
        await super().aclose()

    async def __anext__(self) -> stt.SpeechEvent:
        if self._closed:
            raise StopAsyncIteration
        event = await super().__anext__()
        if self._closed:
            raise StopAsyncIteration
        return event

    async def _run(self) -> None:
        try:
            if self._input_error is not None:
                raise self._input_error
            await self._attempt()
        except APIError as error:
            if self._input_consumed:
                error.retryable = False
            raise

    async def _attempt(self) -> None:
        tasks: list[asyncio.Task[None]] = []
        try:
            await self._handshake()
            assert self._ws is not None
            sender = asyncio.create_task(self._send(self._ws))
            receiver = asyncio.create_task(self._receive(self._ws))
            tasks = [sender, receiver]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if sender in done:
                await sender
            else:
                await receiver
                raise APIConnectionError("Microsoft AI STT receiver stopped unexpectedly")
        except aiohttp.WSServerHandshakeError as error:
            raise status_error("STT", error.status) from None
        except TimeoutError:
            raise APITimeoutError("Microsoft AI STT operation timed out") from None
        except (aiohttp.ClientError, ConnectionError, OSError):
            raise APIConnectionError("Microsoft AI STT transport failed") from None
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            if self._commit_future is not None:
                self._commit_future.cancel()
                self._commit_future = None
            if self._ws is not None:
                try:
                    await asyncio.wait_for(self._ws.close(), self._conn_options.total_timeout)
                except (TimeoutError, aiohttp.ClientError, ConnectionError):
                    logger.warning("Microsoft AI STT socket cleanup failed")
                self._ws = None

    async def _handshake(self) -> None:
        ws = await utils.http_context.ws_connect(
            self._client.session(),
            self._client.url,
            headers=self._client.headers,
            heartbeat=10.0,
            max_msg_size=1024 * 1024,
            conn_options=self._conn_options,
        )
        self._ws = ws
        created = await asyncio.wait_for(self._receive_event(ws), self._conn_options.read_timeout)
        if created.type != "session.created":
            raise APIError("Microsoft AI STT expected session.created", retryable=False)
        await self._write(ws, _session_update(self._stt.model, self._language_hint))
        updated = await asyncio.wait_for(self._receive_event(ws), self._conn_options.read_timeout)
        if updated.type != "session.updated":
            raise APIError("Microsoft AI STT expected session.updated", retryable=False)

    async def _receive_event(self, ws: _WebSocket) -> _Event:
        message = await ws.receive()
        if message.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.ERROR,
        ):
            raise APIConnectionError("Microsoft AI STT connection closed unexpectedly")
        if message.type != aiohttp.WSMsgType.TEXT or not isinstance(message.data, (str, bytes)):
            raise APIError("Microsoft AI STT expected a JSON text event", retryable=False)
        try:
            event = _Event.model_validate_json(message.data)
        except ValidationError as exc:
            raise APIError("Microsoft AI STT sent malformed JSON", retryable=False) from exc
        if event.type in ("error", _TRANSCRIPTION + "failed"):
            raise _provider_error(event.error)
        return event

    async def _write(self, ws: _WebSocket, data: dict[str, object]) -> None:
        await asyncio.wait_for(ws.send_json(data), self._conn_options.write_timeout)

    async def _append(
        self, ws: _WebSocket, frame: rtc.AudioFrame, *, account_input: bool = True
    ) -> None:
        await self._write(
            ws,
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame.data).decode("ascii"),
            },
        )
        if account_input:
            self._processed_samples += frame.samples_per_channel
        self._segment_samples += frame.samples_per_channel

    async def _commit(self, ws: _WebSocket, buffer: utils.audio.AudioByteStream) -> None:
        for frame in buffer.flush():
            await self._append(ws, frame)
        if not self._segment_samples:
            return
        self._commit_future = asyncio.get_running_loop().create_future()
        self._committed_item_id = None
        try:
            await self._write(ws, {"type": "input_audio_buffer.commit"})
            await asyncio.wait_for(self._commit_future, self._conn_options.read_timeout)
            self._segment_samples = 0
        finally:
            self._commit_future = None

    async def _send(self, ws: _WebSocket) -> None:
        buffer = utils.audio.AudioByteStream(
            sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, samples_per_channel=800
        )
        if self._vad is None:
            async for data in self.input:
                if self._input_error is not None:
                    raise self._input_error
                if isinstance(data, self.Flush):
                    await self._commit(ws, buffer)
                else:
                    self._input_consumed = True
                    for frame in buffer.push(data.data.cast("B")):
                        await self._append(ws, frame)
        else:
            while True:
                await self._send_vad_segment(ws, buffer, self._vad)
                if self.input.closed and self.input.empty():
                    break
        if self._input_error is not None:
            raise self._input_error

    async def _send_vad_segment(
        self,
        ws: _WebSocket,
        buffer: utils.audio.AudioByteStream,
        detector: VAD,
    ) -> None:
        vad_stream = detector.stream()
        pending = bytearray()
        consumed_samples = 0
        fed_samples = 0
        feed_ended = False
        speaking = False
        committed_samples = 0

        async def feed() -> None:
            nonlocal fed_samples, feed_ended
            async for data in self.input:
                if self._input_error is not None:
                    raise self._input_error
                if isinstance(data, self.Flush):
                    break
                self._input_consumed = True
                pending.extend(data.data.cast("B"))
                fed_samples += data.samples_per_channel
                vad_stream.push_frame(data)
            if self._input_error is not None:
                raise self._input_error
            feed_ended = True
            vad_stream.end_input()

        async def drain(position: int) -> None:
            nonlocal consumed_samples
            if position < consumed_samples or position > fed_samples:
                raise APIError(
                    "Microsoft AI STT requires ordered, input-relative VAD timestamps",
                    retryable=False,
                )
            size = (position - consumed_samples) * 2
            data = bytes(pending[:size]) if speaking else b""
            del pending[:size]
            consumed_samples = position
            if speaking:
                for frame in buffer.push(data):
                    await self._append(ws, frame)
            else:
                self._processed_samples += size // 2

        async def send_prefix(event: VADEvent, position: int) -> None:
            if not event.frames or any(
                frame.sample_rate != SAMPLE_RATE
                or frame.num_channels != NUM_CHANNELS
                or frame.samples_per_channel <= 0
                for frame in event.frames
            ):
                raise APIError(
                    "Microsoft AI STT requires nonempty mono 16kHz VAD start frames "
                    "ending at the event timestamp to preserve the speech prefix",
                    retryable=False,
                )
            samples = sum(frame.samples_per_channel for frame in event.frames)
            if samples > int(self._max_buffered_audio * SAMPLE_RATE):
                raise APIError(
                    "Microsoft AI STT VAD start prefix exceeds max_buffered_audio",
                    retryable=False,
                )
            if samples > position or position <= committed_samples:
                raise APIError(
                    "Microsoft AI STT received invalid VAD start timing", retryable=False
                )
            prefix = rtc.combine_audio_frames(event.frames).data.tobytes()
            # The VAD may retain audio from the preceding committed turn. Never replay it.
            overlap = max(0, committed_samples - (position - samples))
            prefix = prefix[overlap * 2 :]
            for frame in buffer.push(prefix):
                await self._append(ws, frame, account_input=False)
            # Prefix samples were already accounted as idle input. Keep them separate
            # from subsequent, not-yet-accounted audio in the framing buffer.
            for frame in buffer.flush():
                await self._append(ws, frame, account_input=False)

        async def consume() -> None:
            nonlocal speaking, committed_samples
            async for event in vad_stream:
                if not math.isfinite(event.timestamp) or event.timestamp < 0:
                    raise APIError(
                        "Microsoft AI STT received an invalid VAD timestamp", retryable=False
                    )
                position = round(event.timestamp * SAMPLE_RATE)
                await drain(position)
                if event.type == VADEventType.START_OF_SPEECH:
                    if speaking:
                        raise APIError(
                            "Microsoft AI STT received overlapping VAD speech starts",
                            retryable=False,
                        )
                    await send_prefix(event, position)
                    speaking = True
                    self._start_speaking()
                elif event.type == VADEventType.END_OF_SPEECH:
                    if not speaking:
                        raise APIError(
                            "Microsoft AI STT received a VAD end without a speech start",
                            retryable=False,
                        )
                    await self._commit(ws, buffer)
                    committed_samples = position
                    speaking = False
            if not feed_ended:
                raise APIError("Microsoft AI STT VAD stopped before end of input", retryable=False)
            # A VAD's last inference window can be incomplete. Send the actual remaining
            # samples, not invented silence, before the explicit final commit.
            await drain(fed_samples)
            if speaking:
                await self._commit(ws, buffer)

        producer = asyncio.create_task(feed())
        consumer = asyncio.create_task(consume())
        try:
            done, _ = await asyncio.wait([producer, consumer], return_when=asyncio.FIRST_COMPLETED)
            if producer in done:
                await producer
                # Finite input: the VAD's last results are a response the stream is waiting on.
                await asyncio.wait_for(consumer, self._conn_options.read_timeout)
            else:
                # Live input may remain open indefinitely; surface the consumer's error now.
                await consumer
        finally:
            await utils.aio.cancel_and_wait(producer, consumer)
            await vad_stream.aclose()

    def _start_speaking(self) -> None:
        if not self._speaking:
            self._speaking = True
            self.send(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _speech_event(self, type: stt.SpeechEventType, item: _Item, text: str) -> None:
        self._start_speaking()
        self.send(
            stt.SpeechEvent(
                type=type,
                request_id=item.id,
                alternatives=[stt.SpeechData(language=self._language, text=text)],
            )
        )

    async def _receive(self, ws: _WebSocket) -> None:
        while True:
            event = await self._receive_event(ws)
            if "event_id" in event.model_fields_set:
                event_id = _string(event.event_id, "event_id", nonempty=True)
                if event_id in self._event_ids:
                    continue
                self._event_ids.append(event_id)
            event_type = event.type
            if not (
                event_type.startswith(_TRANSCRIPTION)
                or event_type == "input_audio_buffer.committed"
            ):
                continue
            item_id = _string(event.item_id, "item_id", nonempty=True)
            if item_id in self._finished_items:
                continue
            if self._item is None:
                self._item = _Item(id=item_id)
            item = self._item
            if item.id != item_id:
                raise APIError("Microsoft AI STT changed item before completion", retryable=False)
            if event_type == "input_audio_buffer.committed":
                if self._commit_future is None:
                    raise APIError(
                        "Microsoft AI STT acknowledged an unsolicited commit", retryable=False
                    )
                self._committed_item_id = item_id
            elif event_type == _TRANSCRIPTION + "intermediate":
                item.hypothesis = _string(event.intermediate, "intermediate")
                self._interim(item)
            elif event_type == _TRANSCRIPTION + "delta":
                item.finalized += _string(event.delta, "delta")
                item.hypothesis = ""
                self._interim(item)
            elif event_type == _TRANSCRIPTION + "completed":
                self._complete(item, _string(event.transcript, "transcript"))
            else:
                raise APIError(
                    "Microsoft AI STT sent an unsupported transcription event", retryable=False
                )

    def _interim(self, item: _Item) -> None:
        text = item.finalized + item.hypothesis
        if len(text) > _MAX_TEXT_LENGTH:
            raise APIError("Microsoft AI STT transcript exceeds the text limit", retryable=False)
        if text != item.last_interim:
            item.last_interim = text
            self._speech_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, item, text)

    def _complete(self, item: _Item, transcript: str) -> None:
        if (
            self._commit_future is None
            or self._commit_future.done()
            or self._committed_item_id != item.id
        ):
            raise APIError(
                "Microsoft AI STT completed without an acknowledged commit", retryable=False
            )
        # A completion can legitimately retract the revisable interim hypothesis.
        if transcript:
            self._speech_event(stt.SpeechEventType.FINAL_TRANSCRIPT, item, transcript)
        self.send(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=item.id,
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=self._segment_samples / SAMPLE_RATE
                ),
            )
        )
        if self._speaking:
            self._speaking = False
            self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=item.id))
        self._finished_items.append(item.id)
        self._item = None
        self._commit_future.set_result(None)
