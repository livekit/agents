from __future__ import annotations

import asyncio
import base64
import wave
from io import BytesIO

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIError, LanguageCode, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._client import DEFAULT_OPTIONS, SAMPLE_RATE, Client, duplex, error, receive

PREFIX = "conversation.item.input_audio_transcription."


class STT(stt.STT):
    """Kitsch Labs file and streaming transcription using a workspace API key.

    Streaming accepts mono audio, resamples to 24 kHz, and enables replaceable
    partial transcripts. Server VAD endpoints speech; flush also commits manually.
    Connections recover according to the caller's retry options.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        base_url: str = "https://api.kitschlabs.com",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, manual_flush=True
            )
        )
        self._client = Client(api_key, base_url, http_session)
        self._language = language
        self._prompt = prompt

    @property
    def model(self) -> str:
        return "kitsch-stt-v1"

    @property
    def provider(self) -> str:
        return "Kitsch Labs"

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Transcribe mono audio through the Kitsch Labs file transcription API."""
        return await super().recognize(buffer, language=language, conn_options=conn_options)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        if frame.num_channels != 1 or not frame.samples_per_channel:
            raise ValueError("STT requires nonempty mono audio.")
        if len(frame.data) * 2 > 100 * 1024 * 1024 - 44:
            raise ValueError("STT audio exceeds the 100 MiB file limit.")
        output = BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(frame.sample_rate)
            wav.writeframes(frame.data.cast("B"))
        form = aiohttp.FormData()
        form.add_field("file", output.getvalue(), filename="audio.wav", content_type="audio/wav")
        form.add_field("model", self.model)
        selected_language = language if is_given(language) else self._language
        if selected_language:
            form.add_field("language", selected_language)
        if self._prompt:
            form.add_field("prompt", self._prompt)
        async with self._client.post("/v1/speech-to-text", conn_options, data=form) as response:
            try:
                data = await response.json()
            except (ValueError, aiohttp.ContentTypeError):
                raise error("Invalid Kitsch Labs transcription response.") from None
            if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                raise error("Invalid Kitsch Labs transcription response.")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=response.headers.get("x-request-id", ""),
                alternatives=[
                    stt.SpeechData(
                        language=LanguageCode(selected_language or ""),
                        text=data["text"],
                    )
                ],
            )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_OPTIONS,
    ) -> RecognizeStream:
        """Create a realtime stream with concurrent audio input and transcription."""
        selected_language = language if is_given(language) else self._language
        return RecognizeStream(self, selected_language, conn_options)


class RecognizeStream(stt.RecognizeStream):
    _stt: STT

    def __init__(self, stt_instance: STT, language: str | None, options: APIConnectOptions) -> None:
        self._language = language
        super().__init__(stt=stt_instance, conn_options=options, sample_rate=SAMPLE_RATE)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if frame.num_channels != 1:
            raise ValueError("STT streaming requires mono audio.")
        super().push_frame(frame)

    async def _run(self) -> None:
        try:
            await self._run_session()
        except APIError as exc:
            # A new connection cannot recover audio already consumed at end_input.
            if self._input_ch.closed and self._input_ch.qsize() == 0:
                exc.retryable = False
            raise

    async def _run_session(self) -> None:
        async with self._stt._client.websocket(
            "/v1/speech-to-text/stream", self._conn_options
        ) as ws:
            created = await receive(ws, self._conn_options.timeout)
            if created.get("type") != "session.created":
                raise error("Kitsch Labs STT did not create a session.")
            request_id = created.get("session", {}).get("id", "")
            transcription = {"model": self._stt.model, "include_partial": True}
            if self._language:
                transcription["language"] = self._language
            if self._stt._prompt:
                transcription["prompt"] = self._stt._prompt
            await ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                                "transcription": transcription,
                                "turn_detection": {"type": "server_vad"},
                            }
                        },
                    },
                }
            )
            updated = await receive(ws, self._conn_options.timeout)
            if updated.get("type") != "session.updated" or (
                updated.get("session", {})
                .get("audio", {})
                .get("input", {})
                .get("transcription", {})
                .get("include_partial")
                is not True
            ):
                raise error("Kitsch Labs STT did not enable partial transcripts.")

            final_received = asyncio.Event()
            final_received.set()
            pending_commits = 0
            pending_items: set[str] = set()
            completed_items: set[str] = set()
            finished = False
            speaking = False

            async def send() -> None:
                nonlocal pending_commits, finished
                audio_sent = False
                async for part in self._input_ch:
                    if isinstance(part, rtc.AudioFrame):
                        data = part.data.cast("B")
                        for offset in range(0, len(data), 4800):
                            await ws.send_json(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(data[offset : offset + 4800]).decode(
                                        "ascii"
                                    ),
                                }
                            )
                        audio_sent = True
                    elif audio_sent:
                        final_received.clear()
                        pending_commits += 1
                        await ws.send_json({"type": "input_audio_buffer.commit"})
                        audio_sent = False
                await asyncio.wait_for(final_received.wait(), self._conn_options.timeout)
                finished = True
                await ws.close()

            def emit_text(event_type: stt.SpeechEventType, text: str) -> None:
                nonlocal speaking
                if not speaking:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.START_OF_SPEECH,
                            request_id=request_id,
                        )
                    )
                    speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=event_type,
                        request_id=request_id,
                        alternatives=[
                            stt.SpeechData(language=LanguageCode(self._language or ""), text=text)
                        ],
                    )
                )

            async def read() -> None:
                nonlocal speaking, pending_commits
                while True:
                    try:
                        data = await receive(ws, 930)
                    except APIError:
                        if finished:
                            return
                        raise
                    event_type = data.get("type")
                    if event_type == "error":
                        code = data.get("error", {}).get("code")
                        if pending_commits and code == "buffer_empty":
                            # An empty input buffer can still have transcriptions in flight.
                            pending_commits -= 1
                            if not pending_commits and not pending_items:
                                final_received.set()
                            continue
                        if code in {
                            "session_time_limit",
                            "upstream_error",
                            "provider_disconnected",
                        }:
                            raise APIConnectionError("Kitsch Labs transcription session ended.")
                        raise error("Kitsch Labs transcription failed.")
                    item_id = data.get("item_id")
                    if not isinstance(item_id, str) or not item_id or item_id in completed_items:
                        continue
                    if event_type == "input_audio_buffer.committed":
                        pending_items.add(item_id)
                        final_received.clear()
                        if pending_commits:
                            pending_commits -= 1
                        continue
                    if event_type == PREFIX + "partial":
                        stable = data.get("stable_text")
                        provisional = data.get("provisional_text")
                        if not isinstance(stable, str) or not isinstance(provisional, str):
                            raise error("Invalid Kitsch Labs partial transcript.")
                        self._mark_provider_progress()
                        emit_text(stt.SpeechEventType.INTERIM_TRANSCRIPT, stable + provisional)
                    elif event_type == PREFIX + "completed":
                        text = data.get("transcript")
                        if not isinstance(text, str):
                            raise error("Invalid Kitsch Labs final transcript.")
                        emit_text(stt.SpeechEventType.FINAL_TRANSCRIPT, text)
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.END_OF_SPEECH,
                                request_id=request_id,
                            )
                        )
                        speaking = False
                        completed_items.add(item_id)
                        pending_items.discard(item_id)
                        if not pending_commits and not pending_items:
                            final_received.set()
                        seconds = data.get("usage", {}).get("seconds")
                        if isinstance(seconds, (int, float)) and seconds >= 0:
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                                    request_id=request_id,
                                    recognition_usage=stt.RecognitionUsage(audio_duration=seconds),
                                )
                            )

            await duplex(send(), read())
