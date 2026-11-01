from __future__ import annotations

import asyncio
import base64
import wave
from dataclasses import replace
from io import BytesIO

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, LanguageCode, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._client import DEFAULT_OPTIONS, SAMPLE_RATE, Client, duplex, error, receive

PREFIX = "conversation.item.input_audio_transcription."


class STT(stt.STT):
    """Kitsch file and streaming transcription using a workspace API key.

    Streaming accepts mono audio, resamples to 24 kHz, and enables replaceable
    partial transcripts. Server VAD endpoints speech; flush also commits manually.
    Connections follow the server's session limit without reconnect or replay.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        base_url: str = "https://api.kitschlabs.com",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, manual_flush=True
            )
        )
        self._client = Client(api_key, base_url)
        self._language = language
        self._prompt = prompt

    @property
    def model(self) -> str:
        return "kitsch-stt-v1"

    @property
    def provider(self) -> str:
        return "Kitsch"

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Transcribe mono audio through the Kitsch file transcription API."""
        return await super().recognize(
            buffer, language=language, conn_options=replace(conn_options, max_retry=0)
        )

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
                raise error("Invalid Kitsch transcription response.") from None
            if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                raise error("Invalid Kitsch transcription response.")
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
        """Create a realtime stream; flush waits for final text before more audio."""
        selected_language = language if is_given(language) else self._language
        return RecognizeStream(self, selected_language, replace(conn_options, max_retry=0))

    async def aclose(self) -> None:
        await self._client.close()


class RecognizeStream(stt.RecognizeStream):
    _stt: STT

    def __init__(self, stt_instance: STT, language: str | None, options: APIConnectOptions) -> None:
        self._language = language
        super().__init__(stt=stt_instance, conn_options=options, sample_rate=SAMPLE_RATE)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if frame.num_channels != 1:
            raise ValueError("STT streaming requires mono audio.")
        # Bound the SDK's synchronous input queue instead of accumulating microphone data.
        if self._input_ch.qsize() >= 500:
            raise ValueError("STT input is full; consume events before sending more audio.")
        super().push_frame(frame)

    async def _run(self) -> None:
        async with self._stt._client.websocket(
            "/v1/speech-to-text/stream", self._conn_options
        ) as ws:
            created = await receive(ws, self._conn_options.timeout)
            if created.get("type") != "session.created":
                raise error("Kitsch STT did not create a session.")
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
                raise error("Kitsch STT did not enable partial transcripts.")

            final_received = asyncio.Event()
            pending_commit = False
            commit_confirmed = False
            pending_items: set[str] = set()
            finished = False
            speaking = False
            last_final = None

            async def send() -> None:
                nonlocal pending_commit, commit_confirmed, finished
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
                        pending_commit = True
                        commit_confirmed = False
                        await ws.send_json({"type": "input_audio_buffer.commit"})
                        await asyncio.wait_for(final_received.wait(), self._conn_options.timeout)
                        pending_commit = False
                        audio_sent = False
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
                nonlocal last_final, speaking, commit_confirmed
                while True:
                    try:
                        data = await receive(ws, 930)
                    except APIError:
                        if finished:
                            return
                        raise
                    event_type = data.get("type")
                    if event_type == "error":
                        if pending_commit and data.get("error", {}).get("code") == "buffer_empty":
                            # An empty input buffer can still have transcriptions in flight.
                            commit_confirmed = True
                            if not pending_items:
                                final_received.set()
                            continue
                        raise error("Kitsch transcription failed.")
                    item_id = data.get("item_id")
                    if not isinstance(item_id, str) or not item_id or item_id == last_final:
                        continue
                    if event_type == "input_audio_buffer.committed":
                        pending_items.add(item_id)
                        if pending_commit:
                            commit_confirmed = True
                        continue
                    if event_type == PREFIX + "partial":
                        stable = data.get("stable_text")
                        provisional = data.get("provisional_text")
                        if not isinstance(stable, str) or not isinstance(provisional, str):
                            raise error("Invalid Kitsch partial transcript.")
                        emit_text(stt.SpeechEventType.INTERIM_TRANSCRIPT, stable + provisional)
                    elif event_type == PREFIX + "completed":
                        text = data.get("transcript")
                        if not isinstance(text, str):
                            raise error("Invalid Kitsch final transcript.")
                        emit_text(stt.SpeechEventType.FINAL_TRANSCRIPT, text)
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.END_OF_SPEECH,
                                request_id=request_id,
                            )
                        )
                        speaking = False
                        last_final = item_id
                        pending_items.discard(item_id)
                        if pending_commit and commit_confirmed and not pending_items:
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
