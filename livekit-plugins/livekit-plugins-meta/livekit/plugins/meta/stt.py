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

"""Meta Muse Voice Transcribe speech-to-text.

Realtime transcription over `wss://api.meta.ai/v1/asr/realtime`, plus batch
recognition over `POST /v1/asr/transcribe`.

See https://dev.meta.ai/docs/speech-to-text.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
import weakref
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.stt import (
    RecognitionUsage,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, http_context, is_given

from ._utils import PeriodicCollector
from .log import logger
from .models import (
    SAMPLE_RATES,
    MuseEncoding,
    MuseMode,
    MuseModels,
    MusePartialMode,
    to_language_bias,
)

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL: MuseModels = "muse-voice-transcribe-1.0"

# Muse holds a realtime session for at most 60 minutes. Rotate well before that so the
# swap never lands inside a turn; a rotation mid-utterance would drop its final.
_SESSION_ROTATE_AFTER = 55 * 60
# How long rotation will wait for a turn to finish before reconnecting anyway. The
# server drops the session at 60 minutes regardless, so this cannot be open-ended.
_ROTATE_GRACE = 4 * 60

# The server closes a stream that stops sending audio without an endStream, so gaps in
# the input are filled with PCM silence rather than left empty.
_SILENCE_AFTER = 1.0
_CHUNK_MS = 80  # Muse consumes audio in 80 ms chunks.


@dataclass
class _STTOptions:
    model: MuseModels | str
    mode: MuseMode
    partial_mode: MusePartialMode
    encoding: MuseEncoding
    languages: list[str]
    keywords: list[str]
    emit_audio_progress: bool
    zdr_override: NotGivenOr[bool]
    # Keyterms the framework manages (session config + auto-detection). Kept apart from
    # `keywords` so a session update never clobbers what the caller asked for.
    session_keyterms: list[str] = field(default_factory=list)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATES[self.encoding]

    def all_keywords(self) -> list[str]:
        merged = list(self.keywords)
        merged.extend(k for k in self.session_keyterms if k not in merged)
        return merged

    def handshake(self, api_key: str) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "authorization": {"accessToken": f"Bearer {api_key}"},
            "audioEncoding": self.encoding,
            "model": self.model,
            "mode": self.mode,
            "partialMode": self.partial_mode,
            "emitAudioProgress": self.emit_audio_progress,
        }
        if keywords := self.all_keywords():
            msg["keywords"] = keywords
        if bias := to_language_bias(self.languages):
            msg["languageBias"] = bias
        if is_given(self.zdr_override):
            msg["zdrOverride"] = self.zdr_override
        return msg


class STT(stt.STT):
    """Speech-to-text with Meta's Muse Voice Transcribe.

    Streams audio to the realtime ASR WebSocket and maps its events onto
    ``SpeechEvent``s; ``recognize()`` uses the batch endpoint. Keywords and
    language bias are handshake-only on the Meta side, so changing them reconnects
    any running stream.
    """

    def __init__(
        self,
        *,
        model: MuseModels | str = DEFAULT_MODEL,
        mode: MuseMode = "ENDPOINTING",
        partial_mode: MusePartialMode = "CUMULATIVE",
        encoding: MuseEncoding = "PCM_24KHZ",
        language: NotGivenOr[str | list[str]] = NOT_GIVEN,
        keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        emit_audio_progress: bool = False,
        zdr_override: NotGivenOr[bool] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Meta Muse Voice Transcribe STT.

        Args:
            model: Muse model id.
            mode: ``ENDPOINTING`` lets the model detect turn boundaries and is what
                ``turn_detection="stt"`` needs. ``DIARIZATION`` adds speaker labels.
                ``PUSH_TO_TALK`` leaves the turn boundary to the caller.
            partial_mode: ``CUMULATIVE`` partials each carry the whole hypothesis and
                replace the previous one. ``DELTA`` is for the batch endpoint's SSE mode
                and is rejected here.
            encoding: ``PCM_24KHZ`` or ``PCM_16KHZ``; sets the stream's sample rate.
            language: BCP-47 code(s) biasing recognition. Muse takes language *names*,
                so codes outside its supported set are dropped with a warning.
            keywords: Product names, people, acronyms and other terms to bias toward.
            emit_audio_progress: Ask the server for periodic ``audioProgress`` events.
            zdr_override: Zero-data-retention override, forwarded as sent.
            api_key: Meta Model API key. Falls back to the ``META_API_KEY``
                environment variable.
            base_url: Override the API root, e.g. for a proxy.
            http_session: Reuse an existing aiohttp session.
        """
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=mode == "DIARIZATION",
                # Muse timestamps turns (audioProcessedMs), not words, so there is no
                # aligned transcript to offer. Consumers that need one - the adaptive
                # interruption detector, for instance - fall back accordingly.
                aligned_transcript=False,
                offline_recognize=True,
                keyterms=True,
                chat_context=False,
            )
        )

        if partial_mode == "DELTA":
            # DELTA is documented for /v1/asr/transcribe with Accept: text/event-stream.
            # On the realtime socket a partial would then have to be concatenated rather
            # than replaced, which is not what the interim contract here does.
            raise ValueError("partialMode DELTA is not supported on the realtime stream")

        meta_api_key = api_key if is_given(api_key) else os.environ.get("META_API_KEY")
        if not meta_api_key:
            raise ValueError(
                "Meta API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `META_API_KEY` environment variable"
            )

        if isinstance(language, str):
            languages = [language]
        elif is_given(language):
            languages = list(language)
        else:
            languages = []

        self._api_key: str = meta_api_key
        self._base_url = base_url if is_given(base_url) else DEFAULT_BASE_URL
        self._session = http_session
        self._opts = _STTOptions(
            model=model,
            mode=mode,
            partial_mode=partial_mode,
            encoding=encoding,
            languages=languages,
            keywords=list(keywords) if is_given(keywords) else [],
            emit_audio_progress=emit_audio_progress,
            zdr_override=zdr_override,
        )
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        """The Muse model id in use, e.g. ``muse-voice-transcribe-1.0``."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """Provider name reported in STT metrics."""
        return "Meta"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[MuseModels | str] = NOT_GIVEN,
        mode: NotGivenOr[MuseMode] = NOT_GIVEN,
        language: NotGivenOr[str | list[str]] = NOT_GIVEN,
        keywords: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        """Update the options applied to new and running streams.

        Muse takes all of these in the connection handshake, so every live stream
        reconnects to pick them up.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(mode):
            self._opts.mode = mode
            # Speaker labels only arrive in DIARIZATION mode, so the advertised
            # capability has to move with it; adapters read it to decide whether to
            # trust SpeechData.speaker_id.
            self._capabilities.diarization = mode == "DIARIZATION"
        if is_given(language):
            self._opts.languages = [language] if isinstance(language, str) else list(language)
        if is_given(keywords):
            self._opts.keywords = list(keywords)

        self._push_options()

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if self._opts.session_keyterms == keyterms:
            return
        self._opts.session_keyterms = list(keyterms)
        self._push_options()

    def _push_options(self) -> None:
        # Each stream holds its own copy of the options - it may carry a per-stream
        # language override - so a change here has to be handed down explicitly.
        # Reconnecting a stream that still held the old copy would re-handshake with
        # exactly the settings the update was meant to replace.
        for stream in list(self._streams):
            stream._apply_options(self._opts)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a realtime transcription stream.

        Args:
            language: BCP-47 code biasing this stream alone. It survives a later
                ``update_options(language=...)``; streams without one follow it.
            conn_options: Connection timeout and retry policy.
        """
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.languages = [language]

        stream = SpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            api_key=self._api_key,
            base_url=self._base_url,
            http_session=self._ensure_session(),
            language_override=is_given(language),
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close the recognizer and every stream it handed out.

        The base implementation is a no-op, which would leave each stream's task and
        WebSocket running past the recognizer. The HTTP session is not touched: it is
        either the caller's or the job-scoped one from ``http_context``.
        """
        streams = list(self._streams)
        self._streams.clear()
        # A stream that fails to close must not strand the ones behind it.
        await asyncio.gather(*(stream.aclose() for stream in streams), return_exceptions=True)
        await super().aclose()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.languages = [language]

        request: dict[str, Any] = {
            "model": opts.model,
            # The batch endpoint takes a RIFF/WAVE container, not raw PCM.
            "audioEncoding": "WAV",
            "mode": opts.mode,
        }
        if keywords := opts.all_keywords():
            request["keywords"] = keywords
        if bias := to_language_bias(opts.languages):
            request["languageBias"] = bias

        form = aiohttp.FormData()
        form.add_field("request", json.dumps(request), content_type="application/json")
        form.add_field(
            "audio",
            rtc.combine_audio_frames(buffer).to_wav_bytes(),
            filename="audio.wav",
            content_type="audio/wav",
        )

        try:
            async with self._ensure_session().post(
                f"{self._base_url}/asr/transcribe",
                data=form,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=conn_options.timeout),
            ) as res:
                if res.status != 200:
                    raise APIStatusError(
                        message=await res.text(),
                        status_code=res.status,
                        request_id=res.headers.get("x-request-id"),
                    )
                data = await res.json()
        except TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=data.get("sessionId") or "",
            alternatives=[
                SpeechData(
                    language=_first_language(opts.languages),
                    text=_transcript_from_response(data),
                )
            ],
        )


class SpeechStream(stt.RecognizeStream):
    """One realtime Muse ASR session."""

    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        base_url: str,
        http_session: aiohttp.ClientSession,
        language_override: bool = False,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._base_url = base_url
        self._session = http_session
        self._language_override = language_override
        self._reconnect_event = asyncio.Event()
        self._reconnect_requested = False
        self._turn_active = False

        self._request_id = ""
        self._connected_at = time.time()
        self._speaker_id: str | None = None
        # speechEnd arrives *before* speechComplete, but the framework runs end-of-turn
        # detection the moment it sees END_OF_SPEECH - against whatever transcript has
        # landed so far. Emitting in wire order would judge the turn without its final
        # text, so the boundary is held here and released after the final.
        self._pending_eos: SpeechEvent | None = None

        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

    def _apply_options(self, opts: _STTOptions) -> None:
        """Take the recognizer's current options and reconnect to apply them.

        A language passed to ``stream(language=...)`` speaks for this stream alone and
        survives a recognizer-wide update; without that override the stream follows the
        recognizer, so an ``update_options(language=...)`` does reach it.
        """
        if self._language_override:
            self._opts = dataclasses.replace(opts, languages=self._opts.languages)
        else:
            self._opts = dataclasses.replace(opts)
        self._request_reconnect()

    def _request_reconnect(self) -> None:
        """Reconnect at the next turn boundary.

        Handshake-only settings can only change by reopening the socket, and doing that
        mid-utterance loses the turn: its final never arrives. Mid-turn requests are
        held until END_OF_SPEECH releases them.
        """
        self._reconnect_requested = True
        if not self._turn_active:
            self._reconnect_event.set()

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=RecognitionUsage(audio_duration=duration),
            )
        )

    async def _run(self) -> None:
        closing_ws = False
        last_sent = 0.0

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws, last_sent

            samples_per_chunk = self._opts.sample_rate * _CHUNK_MS // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )

            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())

                    for frame in frames:
                        self._audio_duration_collector.push(frame.duration)
                        await ws.send_bytes(frame.data.tobytes())
                        last_sent = time.monotonic()

                # Input ended inside the collector's period; without this the final
                # partial interval never reaches STT metrics.
                self._audio_duration_collector.flush()
                closing_ws = True
                await ws.send_str(json.dumps({"type": "endStream"}))
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError("Meta STT connection closed unexpectedly") from e

        @utils.log_exceptions(logger=logger)
        async def silence_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # A stream that goes quiet without an endStream is closed server-side, so a
            # gap in the input track is padded rather than left silent on the wire.
            silence = bytes(self._opts.sample_rate * _CHUNK_MS // 1000 * 2)
            while True:
                await asyncio.sleep(_SILENCE_AFTER / 2)
                if closing_ws:
                    return
                if time.monotonic() - last_sent >= _SILENCE_AFTER:
                    await ws.send_bytes(silence)

        @utils.log_exceptions(logger=logger)
        async def rotate_task() -> None:
            await asyncio.sleep(_SESSION_ROTATE_AFTER)
            logger.debug("rotating Meta STT session before the 60 minute limit")
            self._request_reconnect()
            # _request_reconnect defers to the end of the turn. The server closes the
            # session at 60 minutes either way, so a turn that never completes cannot
            # hold the rotation forever.
            deadline = time.monotonic() + _ROTATE_GRACE
            while not self._reconnect_event.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(1)
            self._reconnect_event.set()

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(
                        message="Meta STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    parsed = json.loads(msg.data)
                except Exception:
                    logger.exception("failed to parse Meta STT message")
                    continue

                self._process_stream_event(parsed)

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    asyncio.create_task(silence_task(ws)),
                    asyncio.create_task(rotate_task()),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.cancel_and_wait(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Open the socket, send the handshake, and wait for the session id.

        Muse authenticates in the first frame rather than a header and expects it within
        ten seconds of the connection opening; audio may only follow the server's
        `{"sessionId": ...}` reply.
        """
        url = self._base_url.replace("https://", "wss://").replace("http://", "ws://")
        session_id = utils.shortuuid("lk_")
        ws_url = f"{url}/asr/realtime?{urlencode({'sessionId': session_id})}"

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(ws_url),
                self._conn_options.timeout,
            )
        except TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e

        try:
            await ws.send_str(json.dumps(self._opts.handshake(self._api_key)))
            msg = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
        except TimeoutError as e:
            await ws.close()
            raise APITimeoutError("timed out waiting for the Meta STT handshake") from e
        except Exception:
            await ws.close()
            raise

        if msg.type != aiohttp.WSMsgType.TEXT:
            await ws.close()
            raise APIStatusError(
                message="Meta STT rejected the handshake",
                status_code=ws.close_code or -1,
                body=f"{msg.type=} {msg.data=}",
            )

        ack = json.loads(msg.data)
        if ack.get("type") == "error":
            await ws.close()
            raise APIStatusError(
                message=ack.get("message", "Meta STT handshake failed"),
                status_code=-1,
                request_id=ack.get("sessionId"),
            )

        self._request_id = ack.get("sessionId") or session_id
        # audioProcessedMs is measured from this connection, so the wall-clock anchor
        # has to move with it - the stream outlives any one socket.
        self._connected_at = time.time()
        self._pending_eos = None
        self._turn_active = False
        self._reconnect_requested = False
        return ws

    def _emit(self, ev: SpeechEvent) -> None:
        self._event_ch.send_nowait(ev)

    def _speech_data(self, text: str) -> SpeechData:
        return SpeechData(
            language=_first_language(self._opts.languages),
            text=text,
            speaker_id=self._speaker_id,
        )

    def _flush_pending_eos(self) -> None:
        if self._pending_eos is not None:
            self._emit(self._pending_eos)
            self._pending_eos = None

        self._turn_active = False
        if self._reconnect_requested:
            self._reconnect_event.set()

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        event_type = data.get("type")

        if event_type == "speechStart":
            # A turn that never produced its speechComplete would otherwise strand the
            # held boundary; release it before opening the next one.
            self._flush_pending_eos()
            self._turn_active = True
            self._speaker_id = None
            audio_ms = data.get("audioProcessedMs")
            self._emit(
                SpeechEvent(
                    type=SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                    speech_start_time=(
                        self._connected_at + audio_ms / 1000 if audio_ms is not None else None
                    ),
                )
            )

        elif event_type == "transcript":
            text = data.get("transcript") or ""
            if not text:
                return
            self._emit(
                SpeechEvent(
                    # A final partial is stable but not the turn's answer - the model may
                    # still post-process it into speechComplete - which is what preflight
                    # means here.
                    type=(
                        SpeechEventType.PREFLIGHT_TRANSCRIPT
                        if data.get("final")
                        else SpeechEventType.INTERIM_TRANSCRIPT
                    ),
                    request_id=self._request_id,
                    alternatives=[self._speech_data(text)],
                )
            )

        elif event_type == "speechEnd":
            self._pending_eos = SpeechEvent(
                type=SpeechEventType.END_OF_SPEECH,
                request_id=self._request_id,
            )

        elif event_type == "speechComplete":
            # The turn's text comes from here, not the last partial: the model may
            # post-process a turn between the boundary and its completion.
            self._emit(
                SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[self._speech_data(data.get("transcript") or "")],
                )
            )
            if self._pending_eos is None:
                self._pending_eos = SpeechEvent(
                    type=SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                )
            self._flush_pending_eos()

        elif event_type == "speaker":
            self._speaker_id = data.get("label")

        elif event_type == "error":
            raise APIStatusError(
                message=data.get("message", "Meta STT error"),
                status_code=-1,
                request_id=data.get("sessionId") or self._request_id,
            )

        elif event_type not in ("audioProgress", None):
            logger.debug("unhandled Meta STT event", extra={"event": event_type})


def _first_language(languages: list[str]) -> LanguageCode:
    # Muse does not report a language per transcript, so the bias is the best label
    # available; "multi" when nothing was pinned.
    return LanguageCode(languages[0]) if languages else LanguageCode("multi")


def _transcript_from_response(data: dict[str, Any]) -> str:
    if isinstance(text := data.get("transcript"), str):
        return text
    # Turn-segmented shapes (DIARIZATION, ENDPOINTING) come back as a list; join them
    # rather than return nothing when the flat field is absent.
    for key in ("turns", "segments", "results"):
        if isinstance(items := data.get(key), list):
            return " ".join(
                str(t) for item in items if (t := (item or {}).get("transcript"))
            ).strip()
    return ""
