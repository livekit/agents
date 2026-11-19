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

"""AirCaps Live speech-to-text.

Each stream is one AirCaps session: the plugin asks the AirCaps API where to
connect, opens a WebSocket there, sends 16 kHz mono PCM16 and turns the
session's transcript events into LiveKit speech events.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import weakref
from dataclasses import dataclass
from typing import Any

import aiohttp

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
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import EndOfTurnMode, STTModels

SAMPLE_RATE = 16000
"""The model takes 16 kHz mono PCM16; other input rates are resampled to it."""

_CHUNK_SAMPLES = SAMPLE_RATE // 50  # 20 ms of audio per WebSocket frame
_DEFAULT_BASE_URL = "https://api.aircaps.com"

# The server refuses audio sent more than 2 s ahead of real time, counted from just after
# `ready` (a 1.75 s burst at `ready` passes, 1.9 s does not). Audio is paced to stay within
# _MAX_BURST_S, and a (re)connection sends at most _MAX_BACKLOG_S of the audio that queued
# while it connected: older audio could only be sent late, adding its length to the latency
# of every later transcript.
_MAX_BURST_S = 1.5
_MAX_BACKLOG_S = 1.25

# The server closes a session that receives nothing for 45 s.
_KEEPALIVE_INTERVAL_S = 20.0

# The model may punctuate the end of the previous turn once it hears more ("evening"
# becomes "evening. Actually ..."): that punctuation belongs to the previous turn.
_LEADING_PUNCTUATION = " .,!?;:"

# Session errors that a new attempt does not fix.
_PERMANENT_ERRORS = frozenset(
    {
        "unauthorized",
        "unsupported_model",
        "unsupported_option",
        "invalid_start",
        "invalid_message",
        "invalid_audio_frame",
        "frame_too_large",
    }
)

_END_OF_TURN_OVERRIDES = (
    "end_of_turn_threshold",
    "min_turn_silence_ms",
    "max_turn_silence_ms",
    "vad_threshold",
)


@dataclass
class STTOptions:
    model: STTModels | str
    end_of_turn: bool
    end_of_turn_mode: EndOfTurnMode
    end_of_turn_threshold: NotGivenOr[float]
    min_turn_silence_ms: NotGivenOr[int]
    max_turn_silence_ms: NotGivenOr[int]
    vad_threshold: NotGivenOr[float]

    def end_of_turn_fields(self) -> dict[str, Any]:
        """The end-of-turn fields of a ``start`` or ``configure`` message."""
        if not self.end_of_turn:
            return {"end_of_turn": False}
        fields: dict[str, Any] = {"end_of_turn": True, "end_of_turn_mode": self.end_of_turn_mode}
        for name in _END_OF_TURN_OVERRIDES:
            value = getattr(self, name)
            if is_given(value):
                fields[name] = value
        return fields


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "aircaps-live-1",
        end_of_turn: bool = False,
        end_of_turn_mode: EndOfTurnMode = "balanced",
        end_of_turn_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_turn_silence_ms: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = _DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of AirCaps Live STT.

        Turns end in one of two ways. By default the session's turn handling decides:
        LiveKit flushes the stream when its VAD or turn detector says the user stopped
        speaking, and the final transcript comes back at once. With ``end_of_turn=True``
        the model detects the end of each turn itself; use it with
        ``turn_detection="stt"`` on the ``AgentSession``.

        Args:
            model: The AirCaps model. Defaults to ``"aircaps-live-1"``.
            end_of_turn: Let the model detect the end of each turn. Defaults to False.
            end_of_turn_mode: With ``end_of_turn``, how the model trades speed for
                certainty: ``"min_latency"``, ``"balanced"`` (default) or ``"max_accuracy"``.
                The settings below override single values of the mode.
            end_of_turn_threshold: How sure the model must be that the turn ended,
                0.2 to 0.9.
            min_turn_silence_ms: The shortest pause that can end a turn, 0 to 1000 ms.
            max_turn_silence_ms: A pause this long always ends the turn, 500 to 5000 ms,
                or 0 for no limit.
            vad_threshold: How sure the model's voice detector must be that a sound is
                speech, 0.3 to 0.9.
            api_key: AirCaps API key. Defaults to the ``AIRCAPS_API_KEY`` environment
                variable.
            base_url: The AirCaps API address. Defaults to ``"https://api.aircaps.com"``.
            http_session: Optional aiohttp session to use for requests.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=False,
                manual_flush=True,
            )
        )
        aircaps_api_key = api_key if is_given(api_key) else os.environ.get("AIRCAPS_API_KEY")
        if not aircaps_api_key:
            raise ValueError(
                "AirCaps API key is required, either as argument or set "
                "AIRCAPS_API_KEY environment variable"
            )
        self._api_key = aircaps_api_key
        self._base_url = base_url.rstrip("/")
        self._opts = STTOptions(
            model=model,
            end_of_turn=end_of_turn,
            end_of_turn_mode=end_of_turn_mode,
            end_of_turn_threshold=end_of_turn_threshold,
            min_turn_silence_ms=min_turn_silence_ms,
            max_turn_silence_ms=max_turn_silence_ms,
            vad_threshold=vad_threshold,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "AirCaps"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("AirCaps Live STT supports streaming recognition only")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a streaming session. ``language`` is not used: the model transcribes English."""
        stream = SpeechStream(
            stt=self,
            opts=dataclasses.replace(self._opts),
            conn_options=conn_options,
            api_key=self._api_key,
            base_url=self._base_url,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        end_of_turn: NotGivenOr[bool] = NOT_GIVEN,
        end_of_turn_mode: NotGivenOr[EndOfTurnMode] = NOT_GIVEN,
        end_of_turn_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_turn_silence_ms: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Change the end-of-turn settings, for new streams and for open ones mid-session.

        Args:
            end_of_turn: Let the model detect the end of each turn.
            end_of_turn_mode: ``"min_latency"``, ``"balanced"`` or ``"max_accuracy"``.
            end_of_turn_threshold: How sure the model must be that the turn ended, 0.2 to 0.9.
            min_turn_silence_ms: The shortest pause that can end a turn, 0 to 1000 ms.
            max_turn_silence_ms: A pause this long always ends the turn, 500 to 5000 ms,
                or 0 for no limit.
            vad_threshold: How sure the model's voice detector must be that a sound is
                speech, 0.3 to 0.9.
        """
        changes: dict[str, Any] = {
            name: value
            for name, value in (
                ("end_of_turn", end_of_turn),
                ("end_of_turn_mode", end_of_turn_mode),
                ("end_of_turn_threshold", end_of_turn_threshold),
                ("min_turn_silence_ms", min_turn_silence_ms),
                ("max_turn_silence_ms", max_turn_silence_ms),
                ("vad_threshold", vad_threshold),
            )
            if is_given(value)
        }
        if not changes:
            return
        self._opts = dataclasses.replace(self._opts, **changes)
        for stream in self._streams:
            stream._update_options(changes)


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        base_url: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._opts = opts
        self._api_key = api_key
        self._base_url = base_url
        self._session = http_session
        # `configure` messages for the open session
        self._control_ch = utils.aio.Chan[str]()
        self._usage_s = 0.0  # audio sent and not yet reported
        # per session: the transcript so far, and how much of it the last final covered
        self._text = ""
        self._final_length = 0
        self._last_interim = ""
        self._in_turn = False

    def _update_options(self, changes: dict[str, Any]) -> None:
        self._opts = dataclasses.replace(self._opts, **changes)
        # a session that is not open yet starts with the new options
        self._control_ch.send_nowait(
            json.dumps({"type": "configure", **self._opts.end_of_turn_fields()})
        )

    async def _run(self) -> None:
        # control messages queued before this session opened are covered by its start message
        while not self._control_ch.empty():
            self._control_ch.recv_nowait()
        ws, monitor = await self._connect()
        self._text, self._final_length, self._last_interim = "", 0, ""
        closing = False  # `finish` sent: the session ends with `finished`

        async def send_task() -> None:
            nonlocal closing
            loop = asyncio.get_running_loop()
            started = loop.time()
            sent_s = 0.0
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=_CHUNK_SAMPLES
            )
            backlog = self._take_backlog()
            try:
                while True:
                    if backlog:
                        data = backlog.pop(0)
                    else:
                        try:
                            data = await self._input_ch.recv()
                        except utils.aio.ChanClosed:
                            break
                    if isinstance(data, self._FlushSentinel):
                        frames = audio_bstream.flush()
                    else:
                        frames = audio_bstream.write(data.data.tobytes())
                    for frame in frames:
                        ahead = sent_s + frame.duration - (loop.time() - started)
                        if ahead > _MAX_BURST_S:
                            await asyncio.sleep(ahead - _MAX_BURST_S)
                        await ws.send_bytes(frame.data.tobytes())
                        sent_s += frame.duration
                        self._usage_s += frame.duration
                    if isinstance(data, self._FlushSentinel):
                        await ws.send_str(json.dumps({"type": "finalize"}))

                closing = True
                await ws.send_str(json.dumps({"type": "finish"}))
            except (aiohttp.ClientError, ConnectionError) as e:
                raise APIConnectionError("AirCaps connection closed unexpectedly") from e

        # a failed send of these ends nothing: the receive side reports the closed connection
        async def control_task() -> None:
            async for message in self._control_ch:
                if not closing:
                    with contextlib.suppress(aiohttp.ClientError, ConnectionError):
                        await ws.send_str(message)

        async def keepalive_task() -> None:
            while not closing:
                await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
                if not closing:
                    with contextlib.suppress(aiohttp.ClientError, ConnectionError):
                        await ws.send_str(json.dumps({"type": "ping"}))

        async def recv_task() -> None:
            while True:
                msg = await monitor.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    raise APIStatusError(
                        "AirCaps connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                        retryable=True,
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    logger.warning("AirCaps sent a message that is not JSON")
                    continue
                kind = data.get("type")
                if kind == "transcript":
                    self._mark_provider_progress()
                    self._on_transcript(data)
                elif kind == "finished":
                    return
                elif kind == "error":
                    raise _session_error(data, ws.close_code)
                elif kind == "session_limit":
                    # the server ends the session next; the stream then opens a new one
                    logger.warning("AirCaps session limit: %s", data.get("message"))

        tasks = [
            asyncio.create_task(send_task(), name="AirCaps.send"),
            asyncio.create_task(recv_task(), name="AirCaps.recv"),
            asyncio.create_task(control_task(), name="AirCaps.control"),
            asyncio.create_task(keepalive_task(), name="AirCaps.keepalive"),
        ]
        try:
            # the session is over when the receive side sees `finished`, or either side fails
            done, _ = await asyncio.wait(tasks[:2], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if tasks[1] not in done:
                await tasks[1]
            elif not closing:
                raise APIConnectionError("AirCaps ended the session")
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await ws.close()
            # a turn lives in one session: one cut short ends with its text so far
            self._end_turn(self._turn_text())

    async def _connect(
        self,
    ) -> tuple[aiohttp.ClientWebSocketResponse, utils.aio.WebSocketMonitor]:
        """Look up the session address, open the WebSocket and wait until the session is ready.

        The address is looked up before every session: it is the server with the lowest
        latency that can take a session now, and can differ from one session to the next.
        """
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with self._session.get(
                f"{self._base_url}/v1/realtime/endpoint",
                params={"model": self._opts.model},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                text = await resp.text()
                try:
                    body = json.loads(text)
                except ValueError:
                    body = text
                if resp.status != 200:
                    raise APIStatusError(
                        f"AirCaps endpoint lookup failed: {_error_message(body)}",
                        status_code=resp.status,
                        body=body,
                    )
                url = body["websocket_url"]
            ws = await asyncio.wait_for(
                self._session.ws_connect(url, headers=headers, autoping=False),
                self._conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                "AirCaps refused the connection", status_code=e.status, body=e.message
            ) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("AirCaps connection timed out") from e
        except (aiohttp.ClientError, KeyError, TypeError) as e:
            raise APIConnectionError("failed to connect to AirCaps") from e

        start = {
            "type": "start",
            "model": self._opts.model,
            "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        }
        if self._opts.end_of_turn:
            start.update(self._opts.end_of_turn_fields())
        monitor = utils.aio.WebSocketMonitor(
            ws, liveness_timeout=self._conn_options.liveness_timeout
        )

        async def ready() -> None:
            # status messages can come before `ready`; audio must not
            while True:
                msg = await monitor.receive()
                if msg.type != aiohttp.WSMsgType.TEXT:
                    raise APIStatusError(
                        "AirCaps closed the session before it was ready",
                        status_code=ws.close_code or -1,
                        retryable=True,
                    )
                data = json.loads(msg.data)
                if data.get("type") == "ready":
                    return
                if data.get("type") == "error":
                    raise _session_error(data, ws.close_code)

        try:
            await ws.send_str(json.dumps(start))
            await asyncio.wait_for(ready(), self._conn_options.timeout)
        except BaseException as e:
            await ws.close()
            if isinstance(e, asyncio.TimeoutError):
                raise APITimeoutError("AirCaps session was not ready in time") from e
            if isinstance(e, (aiohttp.ClientError, ConnectionError, ValueError)):
                raise APIConnectionError("AirCaps session failed to start") from e
            raise
        return ws, monitor

    def _take_backlog(self) -> list[Any]:
        """Take the input queued so far, keeping at most _MAX_BACKLOG_S of its newest audio."""
        queued: list[Any] = []
        while not self._input_ch.empty():
            queued.append(self._input_ch.recv_nowait())
        audio_s = sum(item.duration for item in queued if not isinstance(item, self._FlushSentinel))
        dropped_s = 0.0
        kept: list[Any] = []
        for item in queued:
            if (
                not isinstance(item, self._FlushSentinel)
                and audio_s - dropped_s > _MAX_BACKLOG_S + 1e-6
            ):
                dropped_s += item.duration
                continue
            kept.append(item)
        if dropped_s:
            logger.warning(
                "dropped %.1f s of audio that queued while connecting to AirCaps", dropped_s
            )
        return kept

    def _turn_text(self) -> str:
        """The current turn's words: the transcript after the last final."""
        turn = self._text[self._final_length :].lstrip(_LEADING_PUNCTUATION).strip()
        # punctuation alone is not speech: the model can add "?" to the end of the previous
        # turn after its final
        return turn if any(c.isalnum() for c in turn) else ""

    def _on_transcript(self, data: dict[str, Any]) -> None:
        # each event carries the session's whole transcript; the characters before the last
        # final never change, so the current turn is the rest
        self._text = str(data.get("text") or "")
        turn = self._turn_text()
        if turn and not self._in_turn:
            self._in_turn = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

        if not data.get("is_final"):
            if turn and turn != self._last_interim:
                self._last_interim = turn
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[_speech_data(turn)],
                    )
                )
            return

        # the answer to a flush, or the model ended the turn: all of the text is final
        self._final_length = len(self._text)
        self._end_turn(turn, final=True)

    def _end_turn(self, turn: str, *, final: bool = False) -> None:
        """Send the turn's final transcript and close the turn.

        ``final`` is False for a turn cut short by its session ending: it is closed only
        when it has words. A final with no words still answers a flush.
        """
        self._last_interim = ""
        if self._usage_s:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(audio_duration=self._usage_s),
                )
            )
            self._usage_s = 0.0
        if final or turn:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[_speech_data(turn)],
                )
            )
        if self._in_turn:
            self._in_turn = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))


def _speech_data(text: str) -> stt.SpeechData:
    return stt.SpeechData(language=LanguageCode("en"), text=text)


def _error_message(body: Any) -> str:
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return f"{error.get('code', 'error')}: {error.get('message', '')}"
    return str(body)


def _session_error(data: dict[str, Any], close_code: int | None) -> APIStatusError:
    code = str(data.get("code") or "error")
    return APIStatusError(
        f"AirCaps error {code}: {data.get('message', '')}",
        status_code=close_code or -1,
        body=data,
        retryable=code not in _PERMANENT_ERRORS,
    )
