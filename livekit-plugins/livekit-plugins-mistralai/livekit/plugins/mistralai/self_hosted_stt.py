"""Voxtral Realtime served by your own vLLM, over its `/v1/realtime` WebSocket.

vLLM's protocol (vllm/entrypoints/speech_to_text/realtime) differs from Mistral's
hosted realtime API, so this does not go through the `mistralai` SDK:

- a non-final `input_audio_buffer.commit` starts a run, which transcribes the appended
  audio as it arrives and streams `transcription.delta` events;
- a final commit marks the end of the audio. On its own, with no run going, it is never
  answered. The run reads up to the mark and ends with one `transcription.done`;
- after a `done` the server drops whatever audio it still holds, so audio appended
  between the final commit and the `done` would be lost. The stream holds it locally
  and sends it once the next run has started.

Each user turn is one run: the framework's flush at the end of the turn sends the final
commit, the `done` becomes the FINAL_TRANSCRIPT, and the next run starts at once so
interim transcripts keep flowing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import weakref
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
    vad,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

DEFAULT_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
DEFAULT_BASE_URL = "http://localhost:8000"

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
# vLLM's Voxtral Realtime consumes audio in 80 ms steps
CHUNK_DURATION_MS = 80
# how long the end of the input waits for the last turn's transcript
FINAL_DONE_TIMEOUT = 5.0
# Voxtral Realtime spends one token of context per 80 ms of audio, silence included
SECONDS_PER_TOKEN = 0.08
# a run past this share of the context ends at the next pause, and past the second regardless
ROLLOVER_AT_PAUSE = 0.6
ROLLOVER_FORCED = 0.85
# audio with no new text that counts as a pause
ROLLOVER_PAUSE_SECONDS = 1.0


@dataclass
class _SelfHostedSTTOptions:
    model: str
    base_url: str
    api_key: str | None
    language: LanguageCode | None
    max_model_len: int


class SelfHostedSTT(stt.STT):
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        max_model_len: int = 2048,
        vad: vad.VAD | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Streaming STT for Voxtral Realtime running on your own vLLM server.

        Start the server with e.g. ``vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602``.
        No Mistral API key is needed.

        Turns end on the framework's flush: the AgentSession's VAD or turn detector decides
        when the user stopped speaking. Pass ``vad`` only when the session has neither and
        relies on the STT for end of speech.

        Args:
            base_url: The vLLM server, e.g. ``http://gpu-box:8000``. ``http(s)://`` is
                turned into ``ws(s)://`` and ``/v1/realtime`` is appended unless present.
            model: The model name the server was started with.
            api_key: Sent as a bearer token, for a server started with ``--api-key``.
            language: Reported on transcripts. Voxtral Realtime detects the language itself
                and vLLM's realtime endpoint takes no language hint.
            max_model_len: The server's ``--max-model-len``. A run holds one token per 80 ms
                of audio, silence between turns included (2048 is about 164 s), and degrades
                near the limit, so a long run is ended at a pause past 60% of it, and
                regardless at 85%. The text comes back as a final transcript and the next
                run picks up the audio.
            vad: Optional VAD that ends turns from inside the STT and emits
                START_OF_SPEECH / END_OF_SPEECH events.
            http_session: Optional aiohttp session to open the WebSocket with.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
                manual_flush=True,
            )
        )
        self._opts = _SelfHostedSTTOptions(
            model=model,
            base_url=base_url,
            api_key=api_key if is_given(api_key) else None,
            language=LanguageCode(language) if is_given(language) else None,
            max_model_len=max_model_len,
        )
        self._vad = vad
        self._session = http_session
        self._streams: weakref.WeakSet[SelfHostedSpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "vLLM"

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
        raise NotImplementedError(
            "SelfHostedSTT is streaming only; use stream(), or mistralai.STT for batch"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SelfHostedSpeechStream:
        stream = SelfHostedSpeechStream(
            stt=self,
            conn_options=conn_options,
            language=language,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream


def _realtime_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url.removeprefix("https://")
    elif url.startswith("http://"):
        url = "ws://" + url.removeprefix("http://")
    if not url.endswith("/v1/realtime"):
        url = url.removesuffix("/v1") + "/v1/realtime"
    return url


class SelfHostedSpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: SelfHostedSTT,
        conn_options: APIConnectOptions,
        language: NotGivenOr[str],
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._opts = stt._opts
        self._vad = stt._vad
        self._session = http_session
        self._language = LanguageCode(language) if is_given(language) else self._opts.language
        self._request_id = ""
        self._speaking = False

    async def _run(self) -> None:
        ws = await self._connect_ws()
        try:
            await _RealtimeSession(self, ws).run()
        finally:
            await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        headers = {}
        if self._opts.api_key:
            headers["Authorization"] = f"Bearer {self._opts.api_key}"
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(_realtime_url(self._opts.base_url), headers=headers),
                self._conn_options.timeout,
            )
            # the server announces the session before it reads anything
            msg = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("timed out connecting to vLLM realtime") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError("failed to connect to vLLM realtime") from e

        event = json.loads(msg.data) if msg.type == aiohttp.WSMsgType.TEXT else {}
        if event.get("type") != "session.created":
            await ws.close()
            raise APIConnectionError(f"expected session.created from vLLM, got {msg.data!r}")
        self._request_id = event.get("id", "")
        await ws.send_str(json.dumps({"type": "session.update", "model": self._opts.model}))
        return ws

    def _speech_data(self, text: str) -> stt.SpeechData:
        return stt.SpeechData(text=text, language=self._language or LanguageCode(""))

    def _emit_interim(self, text: str) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[self._speech_data(text)],
            )
        )

    def _emit_final(self, text: str, usage: dict | None, audio_duration: float) -> None:
        if text:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[self._speech_data(text)],
                )
            )
        usage = usage or {}
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=audio_duration,
                    input_tokens=usage.get("prompt_tokens") or 0,
                    output_tokens=usage.get("completion_tokens") or 0,
                ),
            )
        )

    def _emit_speaking(self, speaking: bool) -> None:
        if speaking == self._speaking:
            return
        self._speaking = speaking
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH
                if speaking
                else stt.SpeechEventType.END_OF_SPEECH
            )
        )


class _RealtimeSession:
    """One WebSocket's run bookkeeping. Sends happen from the send, VAD and receive
    tasks, so every frame goes out under `_lock` and the run state changes with it."""

    def __init__(self, stream: SelfHostedSpeechStream, ws: aiohttp.ClientWebSocketResponse):
        self._stream = stream
        self._ws = ws
        self._lock = asyncio.Lock()
        # final commit sent, its `done` not in yet: new audio is held in `_held`
        self._awaiting_done = False
        self._held: list[bytes] = []
        self._uncommitted_audio = False
        self._run_audio_duration = 0.0
        self._audio_at_last_text = 0.0
        self._text = ""
        run_seconds = stream._opts.max_model_len * SECONDS_PER_TOKEN
        self._rollover_at_pause = run_seconds * ROLLOVER_AT_PAUSE
        self._rollover_forced = run_seconds * ROLLOVER_FORCED
        self._input_ended = False
        self._turn_done = asyncio.Event()
        self._turn_done.set()

    async def run(self) -> None:
        async with self._lock:
            await self._start_run()

        vad_stream = self._stream._vad.stream() if self._stream._vad else None
        tasks = [
            asyncio.create_task(self._send_task(vad_stream)),
            asyncio.create_task(self._recv_task()),
        ]
        if vad_stream is not None:
            tasks.append(asyncio.create_task(self._vad_task(vad_stream)))
        try:
            # the receive task never ends on its own; the send task ends the session
            # once the last turn's transcript is in
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            if vad_stream is not None:
                await vad_stream.aclose()

    async def _send(self, event: dict) -> None:
        try:
            await self._ws.send_str(json.dumps(event))
        except (aiohttp.ClientError, ConnectionError) as e:
            raise APIConnectionError("vLLM realtime connection lost") from e

    async def _start_run(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})
        self._run_audio_duration = 0.0
        self._audio_at_last_text = 0.0
        self._text = ""

    async def _append(self, pcm: bytes) -> None:
        if self._awaiting_done:
            self._held.append(pcm)
            return
        await self._send(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
        )
        self._uncommitted_audio = True
        self._run_audio_duration += len(pcm) / (SAMPLE_RATE * NUM_CHANNELS * 2)
        quiet = self._run_audio_duration - self._audio_at_last_text
        if self._run_audio_duration >= self._rollover_forced or (
            self._run_audio_duration >= self._rollover_at_pause and quiet >= ROLLOVER_PAUSE_SECONDS
        ):
            logger.debug("vLLM realtime run near max_model_len; ending it and starting another")
            await self._final_commit()

    async def _final_commit(self) -> None:
        """End the current run, unless it has no audio or its end is already out. Called
        with `_lock` held; the run's `done` starts the next one."""
        if self._awaiting_done or not self._uncommitted_audio:
            return
        await self._send({"type": "input_audio_buffer.commit", "final": True})
        self._awaiting_done = True
        self._uncommitted_audio = False
        self._turn_done.clear()

    async def _end_turn(self) -> None:
        async with self._lock:
            await self._final_commit()

    async def _send_task(self, vad_stream: vad.VADStream | None) -> None:
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=SAMPLE_RATE * CHUNK_DURATION_MS // 1000,
        )
        async for data in self._stream._input_ch:
            if isinstance(data, rtc.AudioFrame):
                if vad_stream is not None:
                    vad_stream.push_frame(data)
                frames = audio_bstream.write(data.data.tobytes())
            else:
                frames = audio_bstream.flush()
            async with self._lock:
                for frame in frames:
                    await self._append(frame.data.tobytes())
            if isinstance(data, self._stream._FlushSentinel):
                await self._end_turn()

        if vad_stream is not None:
            vad_stream.end_input()
        async with self._lock:
            for frame in audio_bstream.flush():
                await self._append(frame.data.tobytes())
            self._input_ended = True
        await self._end_turn()
        try:
            await asyncio.wait_for(self._turn_done.wait(), FINAL_DONE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("vLLM realtime: no transcription.done for the last turn")

    async def _vad_task(self, vad_stream: vad.VADStream) -> None:
        async for ev in vad_stream:
            if ev.type == vad.VADEventType.START_OF_SPEECH:
                self._stream._emit_speaking(True)
            elif ev.type == vad.VADEventType.END_OF_SPEECH:
                await self._end_turn()
                self._stream._emit_speaking(False)

    async def _recv_task(self) -> None:
        while True:
            msg = await self._ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                raise APIStatusError(
                    message="vLLM realtime connection closed unexpectedly",
                    status_code=self._ws.close_code or -1,
                )
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                event = json.loads(msg.data)
                await self._handle_event(event)
            except APIError:
                raise
            except Exception:
                logger.exception("failed to process vLLM realtime event")

    async def _handle_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "transcription.delta":
            delta = event.get("delta") or ""
            if not delta:
                return
            self._text += delta
            self._audio_at_last_text = self._run_audio_duration
            if text := self._text.strip():
                self._stream._emit_interim(text)
        elif kind == "transcription.done":
            await self._on_done(event)
        elif kind == "error":
            message = event.get("error") or "vLLM realtime error"
            code = event.get("code")
            if code in ("model_not_found", "model_not_validated", "invalid_event"):
                raise APIStatusError(message=message, status_code=400, body=event, retryable=False)
            if code == "processing_error":
                # the run died without a `done` and nothing would answer the turn: reconnect
                raise APIStatusError(message=message, status_code=500, body=event)
            # a bad audio chunk is dropped by the server and the run goes on
            logger.warning("vLLM realtime error: %s (%s)", message, code)

    async def _on_done(self, event: dict) -> None:
        text = (event.get("text") or self._text).strip()
        async with self._lock:
            self._stream._emit_final(text, event.get("usage"), self._run_audio_duration)
            # a `done` nobody asked for: the run hit max_model_len while the user talked
            if not self._awaiting_done:
                logger.warning("vLLM realtime run ended before the turn did; starting another")
            self._awaiting_done = False
            held, self._held = self._held, []
            if self._input_ended and not held:
                self._turn_done.set()
                return
            await self._start_run()
            for pcm in held:
                await self._append(pcm)
            if not self._input_ended:
                self._turn_done.set()
                return
            # the input ended while this turn was finishing: transcribe what came after it
            await self._final_commit()
            if not self._awaiting_done:
                self._turn_done.set()
