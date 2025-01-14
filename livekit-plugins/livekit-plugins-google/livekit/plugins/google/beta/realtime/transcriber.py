from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Literal

import websockets
from livekit import rtc
from livekit.agents import utils

from google import genai
from google.genai import types

from ...log import logger
from .api_proto import ClientEvents, LiveAPIModels

EventTypes = Literal[
    "input_speech_started",
    "input_speech_done",
]

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
    def __init__(
        self,
        *,
        client: genai.Client,
        model: LiveAPIModels | str,
    ):
        """
        Initializes a TranscriberSession instance for interacting with Google's Realtime API.
        """
        super().__init__()
        self._client = client
        self._model = model
        self._closed = False

        self._config = types.LiveConnectConfigDict(
            response_modalities="TEXT",
            system_instruction=SYSTEM_INSTRUCTIONS,
            generation_config=types.GenerationConfigDict(
                temperature=0.0,
            ),
        )
        self._main_atask = asyncio.create_task(
            self._main_task(), name="gemini-realtime-transcriber"
        )
        self._send_ch = utils.aio.Chan[ClientEvents]()
        self._active_response_id = None

    def _push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        data = base64.b64encode(frame.data).decode("utf-8")
        self._queue_msg({"mime_type": "audio/pcm", "data": data})

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
                                self.emit("input_speech_done", content)
                                self._active_response_id = None

            except websockets.exceptions.ConnectionClosedError as e:
                logger.exception(f"Transcriber session closed in _recv_task: {e}")
                self._closed = True
            except Exception as e:
                logger.exception(f"Uncaught error in transcriber _recv_task: {e}")
                self._closed = True

        async with self._client.aio.live.connect(
            model=self._model, config=self._config
        ) as session:
            self._session = session
            tasks = [
                asyncio.create_task(
                    _send_task(), name="gemini-realtime-transcriber-send"
                ),
                asyncio.create_task(
                    _recv_task(), name="gemini-realtime-transcriber-recv"
                ),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
                await self._session.close()
