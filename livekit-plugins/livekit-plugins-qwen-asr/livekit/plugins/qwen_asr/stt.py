# Copyright 2026 LiveKit, Inc.
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

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
    vad,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

# vLLM realtime expects PCM16 mono at the Whisper feature rate.
REALTIME_SAMPLE_RATE = 16000
_CHUNK_MS = 100
_CHUNK_BYTES = REALTIME_SAMPLE_RATE * _CHUNK_MS // 1000 * 2
# Same horizon as Silero's default max_buffered_speech. A short ring drops the
# start of the utterance when VAD classification lags behind the input loop.
_HELD_BYTES = 60 * REALTIME_SAMPLE_RATE * 2
# Audio newer than this may still contain the silence that ends the turn.
# It stays buffered until END, so the next utterance is not committed early.
_UNCONFIRMED_SAMPLES = int(0.5 * REALTIME_SAMPLE_RATE)


class STT(stt.STT[Any]):
    """Qwen3-ASR on a self-hosted vLLM server.

    Batch recognition posts the utterance to ``/v1/audio/transcriptions``.
    ``language`` and ``prompt`` are omitted from the request when unset, so the
    model keeps its own language detection and an empty context.

    Realtime recognition speaks vLLM's ``/v1/realtime`` socket: a flat
    ``session.update``, 16 kHz PCM, and ``transcription.delta`` /
    ``transcription.done``. The server has no voice activity detection, so a
    Silero VAD closes each turn. Pass ``vad=None`` to close the turn yourself
    with ``flush()``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "Qwen/Qwen3-ASR-1.7B",
        language: str | None = None,
        prompt: str | None = None,
        api_key: str | None = None,
        use_realtime: bool = False,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            base_url: OpenAI-compatible root, including ``/v1``.
                Example: ``http://127.0.0.1:8000/v1``.
            model: Name the vLLM server is serving.
            language: BCP-47 or ISO-639-1 code, such as ``"tr"``. ``None`` leaves
                detection to the model.
            prompt: Optional context. vLLM places this in Qwen3-ASR's system turn
                on the batch endpoint. The same text is sent on
                ``session.update`` for realtime; vLLM 0.30 does not apply it there.
            api_key: Sent as ``Authorization: Bearer`` when set. A server started
                without ``--api-key`` does not need one.
            use_realtime: Stream over ``/v1/realtime``. The server must be started
                with architecture ``Qwen3ASRRealtimeGeneration``, otherwise the
                socket is not mounted.
            vad: End-of-speech detector for realtime. The bundled Silero model is
                used when this is omitted. ``None`` keeps the socket open until
                ``flush()``.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_realtime,
                interim_results=use_realtime,
                keyterms=True,
            )
        )
        if use_realtime and not is_given(vad):
            from livekit.agents.inference import VAD as SileroVAD

            vad = SileroVAD(model="silero")
        self._vad = vad if is_given(vad) else None
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language or None
        self._prompt = prompt or None
        self._api_key = api_key or None
        self._session_keyterms: list[str] = []
        self._http: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Qwen3-ASR"

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Replace the model, language, or context used by later requests."""
        if is_given(model):
            self._model = model
        if is_given(language):
            self._language = language or None
        if is_given(prompt):
            self._prompt = prompt or None

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._session_keyterms = list(keyterms)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if not self.capabilities.streaming:
            return super().stream(language=language, conn_options=conn_options)  # type: ignore[return-value]
        if is_given(language):
            self._language = language or None
        return SpeechStream(stt=self, conn_options=conn_options)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        if is_given(language):
            self._language = language or None

        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()
        form = aiohttp.FormData()
        form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", self._model)
        if self._language:
            form.add_field("language", self._language)
        context = self._context_prompt()
        if context:
            form.add_field("prompt", context)

        timeout = aiohttp.ClientTimeout(total=conn_options.timeout)
        try:
            async with self._ensure_http().post(
                f"{self._base_url}/audio/transcriptions",
                data=form,
                headers=self._headers(),
                timeout=timeout,
            ) as resp:
                body = await _read_body(resp)
                if resp.status >= 400:
                    # The server body can quote the prompt or the audio. The STT
                    # retry logger prints the exception, so the message stays generic.
                    raise APIStatusError(
                        f"Qwen3-ASR transcription failed ({resp.status})",
                        status_code=resp.status,
                    )
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except TimeoutError as exc:
            raise APITimeoutError() from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError("failed to reach Qwen3-ASR") from exc

        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise APIStatusError(
                "Qwen3-ASR response did not include a transcript",
                status_code=500,
            )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(self._language or ""),
                    text=text,
                )
            ],
        )

    async def aclose(self) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None

    def _context_prompt(self) -> str | None:
        """User prompt plus LiveKit keyterms, which Qwen only accepts as context text."""
        parts: list[str] = []
        if self._prompt:
            parts.append(self._prompt)
        if self._session_keyterms:
            parts.append("Vocabulary: " + ", ".join(self._session_keyterms))
        text = "\n".join(parts).strip()
        return text or None

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "LiveKit Agents"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http

    def _realtime_url(self) -> str:
        url = self._base_url
        if url.startswith("https://"):
            url = "wss://" + url.removeprefix("https://")
        elif url.startswith("http://"):
            url = "ws://" + url.removeprefix("http://")
        return f"{url}/realtime"

    def _session_update(self) -> dict[str, object]:
        """vLLM reads ``model`` from the top level. language and prompt are included for newer servers."""
        event: dict[str, object] = {"type": "session.update", "model": self._model}
        if self._language:
            event["language"] = self._language
        context = self._context_prompt()
        if context:
            event["prompt"] = context
        return event


class SpeechStream(stt.SpeechStream):
    """One vLLM realtime socket.

    Audio is 16 kHz PCM16. A non-final ``input_audio_buffer.commit`` starts a
    generation; ``final: true`` ends it and the server answers with
    ``transcription.done``. Silence is not forwarded: a VAD opens the
    generation on speech and closes it when speech stops.
    """

    def __init__(self, *, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=REALTIME_SAMPLE_RATE)
        self._stt: STT = stt

    async def _run(self) -> None:
        ws = await self._connect()
        try:
            await self._expect_session(ws)
            await ws.send_json(self._stt._session_update())
            await self._transcribe(ws)
        finally:
            await ws.close()

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        try:
            return await self._stt._ensure_http().ws_connect(
                self._stt._realtime_url(),
                headers=self._stt._headers(),
                timeout=aiohttp.ClientWSTimeout(ws_close=self._conn_options.timeout),
            )
        except aiohttp.WSServerHandshakeError as exc:
            raise APIStatusError(
                f"Qwen3-ASR realtime handshake failed: {exc.status}",
                status_code=exc.status,
                body=exc.message,
            ) from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError("failed to reach Qwen3-ASR realtime") from exc

    async def _expect_session(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            msg = await ws.receive(timeout=self._conn_options.timeout)
        except TimeoutError as exc:
            raise APITimeoutError() from exc
        if msg.type != aiohttp.WSMsgType.TEXT:
            raise APIConnectionError("Qwen3-ASR realtime closed before session.created")
        event = json.loads(msg.data)
        if event.get("type") == "error":
            raise APIError("Qwen3-ASR realtime request failed", retryable=False)
        if event.get("type") != "session.created":
            raise APIError(
                f"expected session.created, got {event.get('type')}",
                body=event,
                retryable=False,
            )

    async def _transcribe(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # Set while no generation is running. Cleared when a turn starts and
        # set again when transcription.done arrives, so the next turn cannot
        # start while vLLM is still clearing the previous audio queue.
        turn_idle = asyncio.Event()
        turn_idle.set()
        state = _RealtimeTurn()

        async def send_audio() -> None:
            vad_stream = self._stt._vad.stream() if self._stt._vad is not None else None
            vad_events: asyncio.Queue[vad.VADEvent | None] = asyncio.Queue()
            held = bytearray()
            held_origin = 0
            tail = bytearray()
            pushed = 0
            index_base = 0
            last_raw = -1
            pending_flush: list[int] = []
            confirmed = 0
            speaking = vad_stream is None

            async def read_vad() -> None:
                assert vad_stream is not None
                async for event in vad_stream:
                    if event.type in (
                        vad.VADEventType.START_OF_SPEECH,
                        vad.VADEventType.END_OF_SPEECH,
                        vad.VADEventType.INFERENCE_DONE,
                    ):
                        vad_events.put_nowait(event)
                vad_events.put_nowait(None)

            vad_task = asyncio.create_task(read_vad()) if vad_stream is not None else None

            async def open_turn() -> None:
                await turn_idle.wait()
                await ws.send_json({"type": "input_audio_buffer.commit", "final": False})
                turn_idle.clear()
                state.active = True
                state.samples = 0
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

            async def append(pcm: bytes) -> None:
                tail.extend(pcm)
                while len(tail) >= _CHUNK_BYTES:
                    chunk = bytes(tail[:_CHUNK_BYTES])
                    del tail[:_CHUNK_BYTES]
                    await _send_audio(ws, chunk)
                    state.samples += _CHUNK_BYTES // 2

            async def close_turn() -> None:
                if tail:
                    await _send_audio(ws, bytes(tail))
                    state.samples += len(tail) // 2
                    tail.clear()
                await ws.send_json({"type": "input_audio_buffer.commit", "final": True})
                state.active = False
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

            def absolute(raw: int) -> int:
                nonlocal index_base, last_raw
                # Flush restarts Silero's counter. The new epoch begins at the
                # pushed-sample count recorded with the flush, which can sit
                # past the last event index.
                if pending_flush and last_raw >= 0:
                    epoch_len = pending_flush[0] - index_base
                    if raw < last_raw or raw > epoch_len:
                        index_base = pending_flush.pop(0)
                        last_raw = -1
                last_raw = raw
                return index_base + raw

            def take_before(sample: int) -> bytes:
                nonlocal held_origin
                count = min(len(held) // 2, max(0, sample - held_origin))
                chunk = bytes(held[: count * 2])
                del held[: count * 2]
                held_origin += count
                return chunk

            async def release_confirmed() -> None:
                # The newest half-second can still hold the silence that ends
                # the turn. END assigns that tail; it is not sent early.
                chunk = take_before(confirmed - _UNCONFIRMED_SAMPLES)
                if chunk:
                    await append(chunk)

            async def apply_vad(event: vad.VADEvent) -> None:
                nonlocal speaking, confirmed
                boundary = absolute(int(event.samples_index))
                confirmed = max(confirmed, boundary)
                if event.type == vad.VADEventType.START_OF_SPEECH and not speaking:
                    speaking = True
                    await open_turn()
                    # Frames are this utterance's onset. Audio past the index stays
                    # buffered so a later utterance is not committed with this one.
                    onset = b"".join(bytes(frame.data) for frame in event.frames)
                    take_before(boundary)
                    if onset:
                        await append(onset)
                elif event.type == vad.VADEventType.END_OF_SPEECH and speaking:
                    speaking = False
                    rest = take_before(boundary)
                    if rest:
                        await append(rest)
                    if state.active:
                        await close_turn()
                    return
                if speaking:
                    await release_confirmed()

            try:
                async for data in self._input_ch:
                    if vad_stream is not None:
                        while not vad_events.empty():
                            queued = vad_events.get_nowait()
                            if queued is not None:
                                await apply_vad(queued)
                    if isinstance(data, rtc.AudioFrame):
                        pcm = bytes(data.data)
                        if vad_stream is None:
                            if not state.active:
                                await open_turn()
                            await append(pcm)
                        else:
                            held.extend(pcm)
                            pushed += data.samples_per_channel
                            # Only unanswered silence is capped. Speech already
                            # inside a turn stays until the VAD index releases it.
                            if not speaking:
                                overflow = len(held) - _HELD_BYTES
                                if overflow > 0:
                                    del held[:overflow]
                                    held_origin += overflow // 2
                            vad_stream.push_frame(data)
                    elif vad_stream is None and state.active:
                        await close_turn()
                    elif vad_stream is not None:
                        pending_flush.append(pushed)
                        vad_stream.flush()
                if vad_stream is not None:
                    vad_stream.end_input()
                    while True:
                        queued = await vad_events.get()
                        if queued is None:
                            break
                        await apply_vad(queued)
                elif state.active:
                    await close_turn()
            finally:
                if vad_stream is not None:
                    await vad_stream.aclose()
                if vad_task is not None:
                    await asyncio.gather(vad_task, return_exceptions=True)

        async def receive() -> None:
            # vLLM 0.30 streams the raw "language Turkish<asr_text>..." preamble.
            # Hold it back until the transcript after the tag is known.
            cleaner = _AsrText()
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    turn_idle.set()
                    return
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(msg.data)
                kind = event.get("type")
                if kind == "transcription.delta":
                    delta = event.get("delta") or ""
                    if not isinstance(delta, str) or not delta:
                        continue
                    visible = cleaner.push(delta)
                    if not visible:
                        continue
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[self._speech(cleaner.emitted)],
                        )
                    )
                elif kind == "transcription.done":
                    text = event.get("text")
                    transcript = _visible_transcript(text if isinstance(text, str) else cleaner.raw)
                    cleaner.reset()
                    if transcript:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[self._speech(transcript)],
                            )
                        )
                    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.RECOGNITION_USAGE,
                            recognition_usage=stt.RecognitionUsage(
                                audio_duration=state.samples / REALTIME_SAMPLE_RATE,
                                input_tokens=int(usage.get("prompt_tokens") or 0),
                                output_tokens=int(usage.get("completion_tokens") or 0),
                            ),
                        )
                    )
                    state.samples = 0
                    turn_idle.set()
                elif kind == "error":
                    turn_idle.set()
                    raise APIError("Qwen3-ASR realtime request failed", retryable=False)

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(receive())
        try:
            done, pending = await asyncio.wait(
                {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    for other in pending:
                        other.cancel()
                    raise exc
            if send_task in done and not recv_task.done():
                try:
                    await asyncio.wait_for(turn_idle.wait(), self._conn_options.timeout)
                except TimeoutError:
                    logger.warning("Qwen3-ASR realtime turn did not finish before the timeout")
                await ws.close()
                await recv_task
        finally:
            for task in (send_task, recv_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(send_task, recv_task, return_exceptions=True)

    def _speech(self, text: str) -> stt.SpeechData:
        return stt.SpeechData(language=LanguageCode(self._stt._language or ""), text=text)


class _RealtimeTurn:
    def __init__(self) -> None:
        self.active = False
        self.samples = 0


async def _send_audio(ws: aiohttp.ClientWebSocketResponse, pcm: bytes) -> None:
    await ws.send_json(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }
    )


_ASR_TAG = "<asr_text>"
_LANGUAGE_PREFIX = "language "


class _AsrText:
    """Drop Qwen's ``language …<asr_text>`` preamble from a realtime stream."""

    def __init__(self) -> None:
        self.raw = ""
        self.emitted = ""

    def reset(self) -> None:
        self.raw = ""
        self.emitted = ""

    def push(self, delta: str) -> str:
        self.raw += delta
        visible = _visible_transcript(self.raw, partial=True)
        if len(visible) < len(self.emitted):
            self.emitted = ""
        chunk = visible[len(self.emitted) :]
        self.emitted = visible
        return chunk


def _visible_transcript(text: str, *, partial: bool = False) -> str:
    if _ASR_TAG in text:
        return text.rsplit(_ASR_TAG, 1)[1]
    stripped = text.lstrip()
    holding_preamble = (
        stripped == ""
        or _LANGUAGE_PREFIX.startswith(stripped)
        or (stripped.startswith(_LANGUAGE_PREFIX) and len(stripped) < 50 and "\n" not in stripped)
    )
    if partial and holding_preamble:
        return ""
    return text


async def _read_body(resp: aiohttp.ClientResponse) -> object:
    try:
        return await resp.json(content_type=None)
    except (aiohttp.ContentTypeError, ValueError):
        return await resp.text()
