from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Literal

import websockets

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, utils

from ...log import logger
from .api_proto import ClientEvents, LiveAPIModels

EventTypes = Literal["input_speech_started", "input_speech_done"]

DEFAULT_LANGUAGE = "English"

SYSTEM_INSTRUCTIONS = f"""
You are an **Audio Transcriber**. Your task is to convert audio content into accurate and precise text.
- Transcribe verbatim; exclude non-speech sounds.
- Provide only transcription; no extra text or explanations.
- If audio is unclear, respond with: `...`
- Ensure error-free transcription, preserving meaning and context.
- Use proper punctuation and formatting.
- Do not add explanations, comments, or extra information.
- Do not include timestamps, speaker labels, or annotations unless specified.
- Audio Language: {DEFAULT_LANGUAGE}
"""


@dataclass
class TranscriptionContent:
    response_id: str
    text: str


class TranscriberSession(utils.EventEmitter[EventTypes]):
    """
    Handles live audio transcription using the realtime API.
    """

    def __init__(self, *, client: genai.Client, model: LiveAPIModels | str):
        super().__init__()
        self._client = client
        self._model = model
        self._needed_sr = 16000
        self._closed = False

        system_instructions = types.Content(parts=[types.Part(text=SYSTEM_INSTRUCTIONS)])
        self._config = types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction=system_instructions,
            generation_config=types.GenerationConfig(temperature=0.0),
        )
        self._main_atask = asyncio.create_task(
            self._main_task(), name="gemini-realtime-transcriber"
        )
        self._send_ch = utils.aio.Chan[ClientEvents]()
        self._resampler: rtc.AudioResampler | None = None
        self._active_response_id = None

    def _push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        if frame.sample_rate != self._needed_sr:
            if not self._resampler:
                self._resampler = rtc.AudioResampler(
                    frame.sample_rate,
                    self._needed_sr,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )

        if self._resampler:
            for f in self._resampler.push(frame):
                self._queue_msg(
                    types.LiveClientRealtimeInput(
                        media_chunks=[types.Blob(data=f.data.tobytes(), mime_type="audio/pcm")]
                    )
                )
        else:
            self._queue_msg(
                types.LiveClientRealtimeInput(
                    media_chunks=[types.Blob(data=frame.data.tobytes(), mime_type="audio/pcm")]
                )
            )

    def _queue_msg(self, msg: ClientEvents) -> None:
        if not self._closed:
            self._send_ch.send_nowait(msg)

    async def aclose(self) -> None:
        if self._send_ch.closed:
            return
        self._closed = True
        self._send_ch.close()
        await self._main_atask

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        @utils.log_exceptions(logger=logger)
        async def _send_task():
            try:
                async for msg in self._send_ch:
                    if self._closed:
                        break
                    await self._session.send(input=msg)
            except websockets.exceptions.ConnectionClosedError as e:
                logger.exception(f"Transcriber session closed in _send_task: {e}")
                self._closed = True
            except Exception as e:
                logger.exception(f"Uncaught error in transcriber _send_task: {e}")
                self._closed = True

        @utils.log_exceptions(logger=logger)
        async def _recv_task():
            try:
                while not self._closed:
                    async for response in self._session.receive():
                        if self._closed:
                            break
                        if self._active_response_id is None:
                            self._active_response_id = utils.shortuuid()
                            content = TranscriptionContent(
                                response_id=self._active_response_id,
                                text="",
                            )
                            self.emit("input_speech_started", content)

                        server_content = response.server_content
                        if server_content:
                            model_turn = server_content.model_turn
                            if model_turn:
                                for part in model_turn.parts:
                                    if part.text:
                                        content.text += part.text

                            if server_content.turn_complete:
                                content.text = clean_transcription(content.text)
                                self.emit("input_speech_done", content)
                                self._active_response_id = None

            except websockets.exceptions.ConnectionClosedError as e:
                logger.exception(f"Transcriber session closed in _recv_task: {e}")
                self._closed = True
            except Exception as e:
                logger.exception(f"Uncaught error in transcriber _recv_task: {e}")
                self._closed = True

        async with self._client.aio.live.connect(model=self._model, config=self._config) as session:
            self._session = session
            tasks = [
                asyncio.create_task(_send_task(), name="gemini-realtime-transcriber-send"),
                asyncio.create_task(_recv_task(), name="gemini-realtime-transcriber-recv"),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
                await self._session.close()


class ModelTranscriber(utils.EventEmitter[EventTypes]):
    """
    Transcribes agent audio using model generation.
    """

    def __init__(self, *, client: genai.Client, model: LiveAPIModels | str):
        super().__init__()
        self._client = client
        self._model = model
        self._needed_sr = 16000
        self._system_instructions = types.Content(parts=[types.Part(text=SYSTEM_INSTRUCTIONS)])
        self._config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=self._system_instructions,
            # TODO: add response_schem
        )
        self._resampler: rtc.AudioResampler | None = None
        self._buffer: rtc.AudioFrame | None = None
        self._audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._main_atask = asyncio.create_task(self._main_task(), name="gemini-model-transcriber")

    async def aclose(self) -> None:
        if self._audio_ch.closed:
            return
        self._audio_ch.close()
        await self._main_atask

    def _push_audio(self, frames: list[rtc.AudioFrame]) -> None:
        if not frames:
            return

        buffer = utils.merge_frames(frames)

        if buffer.sample_rate != self._needed_sr:
            if self._resampler is None:
                self._resampler = rtc.AudioResampler(
                    input_rate=buffer.sample_rate,
                    output_rate=self._needed_sr,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )

            buffer = utils.merge_frames(self._resampler.push(buffer))

        self._audio_ch.send_nowait(buffer)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        request_id = utils.shortuuid()
        try:
            async for buffer in self._audio_ch:
                # TODO: stream content for better latency
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(text=SYSTEM_INSTRUCTIONS),
                                types.Part.from_bytes(
                                    data=buffer.to_wav_bytes(),
                                    mime_type="audio/wav",
                                ),
                            ],
                            role="user",
                        )
                    ],
                    config=self._config,
                )
                content = TranscriptionContent(
                    response_id=request_id, text=clean_transcription(response.text)
                )
                self.emit("input_speech_done", content)

        except (ClientError, ServerError, APIError) as e:
            raise APIStatusError(
                f"model transcriber error: {e}",
                status_code=e.code,
                body=e.message,
                request_id=request_id,
            ) from e
        except Exception as e:
            raise APIConnectionError("Error generating transcription") from e


def clean_transcription(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
