from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import cast

import httpx
import openai
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, stt
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_SYSTEM_PROMPT = "Perform ASR."


@dataclass
class _STTOptions:
    language: str
    system_prompt: str


class STT(stt.STT):
    """Speech-to-Text using LiquidAI LFM2.5-Audio model."""

    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        system_prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of LiquidAI STT.

        Args:
            base_url: The base URL of the LFM2.5-Audio server (default: http://127.0.0.1:8080/v1)
            api_key: API key for authentication (default: "dummy")
            language: Language code for transcription (default: "en")
            system_prompt: System prompt for ASR (default: "Perform ASR.")
        """
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False, interim_results=False, aligned_transcript=False
            )
        )

        self._opts = _STTOptions(
            language=language if is_given(language) else "en",
            system_prompt=system_prompt if is_given(system_prompt) else DEFAULT_SYSTEM_PROMPT,
        )

        self._client = openai.AsyncClient(
            max_retries=0,
            api_key=api_key if is_given(api_key) else DEFAULT_API_KEY,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=60.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
                ),
            ),
        )

    @property
    def model(self) -> str:
        return "LFM2.5-Audio"

    @property
    def provider(self) -> str:
        return "LiquidAI"

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        system_prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(system_prompt):
            self._opts.system_prompt = system_prompt

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            if is_given(language):
                self._opts.language = language

            # Convert audio buffer to WAV bytes and base64 encode
            wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
            encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")

            # Create messages for the API
            messages = [
                {"role": "system", "content": self._opts.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded_audio, "format": "wav"},
                        }
                    ],
                },
            ]

            # Call the streaming chat completion API
            response = await self._client.chat.completions.create(
                model="LFM2.5-Audio",
                messages=messages,  # type: ignore
                stream=True,
                max_tokens=512,
                extra_body={"reset_context": True},
                timeout=conn_options.timeout,
            )
            # When stream=True, the response is always an AsyncStream
            stream = cast(AsyncStream[ChatCompletionChunk], response)

            # Collect text from the stream
            text_chunks: list[str] = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_chunks.append(chunk.choices[0].delta.content)

            text = "".join(text_chunks)
            logger.debug(f"STT transcription: {text}")

            return self._transcription_to_speech_event(text=text)

        except openai.APITimeoutError as e:
            raise APIConnectionError() from e
        except openai.APIStatusError as e:
            raise APIConnectionError() from e
        except Exception as e:
            logger.error(f"STT error: {e}")
            raise APIConnectionError() from e

    def _transcription_to_speech_event(self, text: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=self._opts.language)],
        )

    async def aclose(self) -> None:
        await self._client.close()
