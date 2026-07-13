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

from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN

from ._recognize_streams.auto_finalize_recognize_stream import AutoFinalizeRecognizeStream
from ._recognize_streams.cartesia_recognize_stream import CartesiaRecognizeStream
from ._recognize_streams.legacy_recognize_stream import LegacyRecognizeStream
from .constants import AUDIO_ENCODING
from .log import logger

if TYPE_CHECKING:
    from typing import Literal

    from typing_extensions import Never

    from livekit.agents import APIConnectOptions
    from livekit.agents.types import NotGivenOr

    from .models import STTEncoding, STTLanguages, STTModels

    _ResolvedFinalTranscriptMode = Literal["auto", "legacy"]


def _is_whisper_model(model: STTModels | str) -> bool:
    return str(model).startswith("ink-whisper")


def _base_url_to_ws_base_url(base_url: str) -> str:
    # If base_url already has a protocol, replace it, otherwise add wss://
    if base_url.startswith(("http://", "https://")):
        return base_url.replace("http", "ws", 1)
    else:
        return f"wss://{base_url}"


class STT(stt.STT):
    """Cartesia speech to text.

    Model ``ink-2`` supports:
        - Streaming
        - Turn detection
        - Interim results

    Model ``ink-whisper`` supports:
        - Streaming
        - Word aligned transcripts

    See also:
        https://docs.cartesia.ai/build-with-cartesia/stt-models/latest

    Examples:

        ```# Turn detecting
        from livekit.agents import AgentSession
        from livekit.plugins import cartesia

        session = AgentSession(
            stt=cartesia.STT(),
            llm=LLM(),  # choose your favorite LLM
            tts=cartesia.TTS(),
            turn_handling={
                "turn_detection": "stt",
            },
        )
        ```
    """

    def __init__(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        sample_rate: int = 16000,
        api_key: str | None = None,
        audio_chunk_duration_ms: int = 160,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "https://api.cartesia.ai",
        language: STTLanguages | str | None = None,
        encoding: STTEncoding = AUDIO_ENCODING,
    ) -> None:
        """
        Create a new instance of Cartesia STT.

        Model ``ink-2`` supports:
            - Streaming
            - Turn detection
            - Interim results

        Model ``ink-whisper`` supports:
            - Streaming
            - Word aligned transcripts

        See also:
            https://docs.cartesia.ai/build-with-cartesia/stt-models/latest

        Args:
            model: The Cartesia STT model to use.
                Defaults to ``ink-2`` if language is ``en``.
                Defaults to ``ink-whisper`` for other languages.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16 kHz.
            api_key: The Cartesia API key. If not provided, it will be read from
                the ``CARTESIA_API_KEY`` environment variable.
            audio_chunk_duration_ms: Duration in milliseconds of each audio chunk
                sent to the Cartesia STT websocket. Defaults to 160 ms.
            http_session: Optional aiohttp ClientSession to use for requests.
            base_url: The base URL for the Cartesia API.
                Defaults to ``https://api.cartesia.ai``.
            language: The language code for recognition.
                This plugin only supports ``en`` for ``ink-2``.
            encoding: The audio encoding format. Must be ``pcm_s16le``.

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Examples:

            ```# Turn detecting
            from livekit.agents import AgentSession
            from livekit.plugins import cartesia

            session = AgentSession(
                stt=cartesia.STT(),
                llm=LLM(),  # choose your favorite LLM
                tts=cartesia.TTS(),
                turn_handling={
                    "turn_detection": "stt",
                },
            )
            ```
        """
        resolved_api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Cartesia API key is required, either as argument or set"
                " CARTESIA_API_KEY environment variable"
            )

        language_code = None if language is None else LanguageCode(language)

        # TODO: default all languages to ink-2 once they are supported
        if utils.is_given(model):
            resolved_model = model
        elif language_code is None or language_code.language == "en":
            resolved_model = "ink-2"
        else:
            resolved_model = "ink-whisper"

        is_whisper = _is_whisper_model(resolved_model)

        resolved_final_transcript_mode: _ResolvedFinalTranscriptMode
        if is_whisper:
            resolved_final_transcript_mode = "legacy"
        else:
            resolved_final_transcript_mode = "auto"

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=resolved_final_transcript_mode != "legacy",
                aligned_transcript="word" if is_whisper else False,
                offline_recognize=False,
                diarization=False,
            )
        )

        self._api_key = resolved_api_key
        self._audio_chunk_duration_ms = audio_chunk_duration_ms
        self._final_transcript_mode: _ResolvedFinalTranscriptMode = resolved_final_transcript_mode
        self._encoding: STTEncoding = encoding
        self._language = language_code
        self._model = resolved_model
        self._sample_rate = sample_rate
        self._session = http_session
        self._ws_base_url = _base_url_to_ws_base_url(base_url=base_url)

        self._streams = weakref.WeakSet[CartesiaRecognizeStream]()

        self._warn_on_unexpected_args(language=self._language)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Cartesia"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Cartesia STT does not support batch recognition, use stream() instead"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> CartesiaRecognizeStream:
        if utils.is_given(language):
            resolved_language = LanguageCode(language)
        elif self._language is not None:
            resolved_language = LanguageCode(self._language)
        else:
            resolved_language = None

        self._warn_on_unexpected_args(language=resolved_language)

        if self._session is None:
            session = utils.http_context.http_session()
            self._session = session
        else:
            session = self._session

        stream: CartesiaRecognizeStream
        match self._final_transcript_mode:
            case "auto":
                stream = AutoFinalizeRecognizeStream(
                    stt=self,
                    conn_options=conn_options,
                    sample_rate=self._sample_rate,
                    encoding=self._encoding,
                    audio_chunk_duration_ms=self._audio_chunk_duration_ms,
                    model=self._model,
                    api_key=self._api_key,
                    ws_base_url=self._ws_base_url,
                    session=session,
                    language=resolved_language or LanguageCode("en"),
                )
            case "legacy":
                stream = LegacyRecognizeStream(
                    stt=self,
                    conn_options=conn_options,
                    sample_rate=self._sample_rate,
                    encoding=self._encoding,
                    audio_chunk_duration_ms=self._audio_chunk_duration_ms,
                    model=self._model,
                    api_key=self._api_key,
                    ws_base_url=self._ws_base_url,
                    session=session,
                    language=resolved_language,
                )
            case _:
                _exhaustive_check: Never = self._final_transcript_mode
                raise RuntimeError(
                    f"Cartesia STT has unexpected final_transcript_mode={_exhaustive_check}"
                )

        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close every stream created by :meth:`stream`.

        The HTTP session is left open: it is either supplied by the caller or
        owned by the shared HTTP context, so it is not ours to close.
        """
        streams = list(self._streams)
        self._streams.clear()
        # return_exceptions=True so one stream failing to close doesn't abandon the rest
        await asyncio.gather(*(stream.aclose() for stream in streams), return_exceptions=True)

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
    ) -> None:
        """Change Cartesia STT options.

        Also propagates changes to all :class:`SpeechStream` created by :meth:`stream`.

        Args:
            language: Used to change the language to match what the user is speaking.
                Ink 2 does not have multi-lingual support yet and only works with English.
            model: Deprecated. This is a no-op. Construct a new STT instance to change the model.
        """
        if utils.is_given(model) and model != self._model:
            logger.warning(
                "Cartesia STT update_options() ignores the model kwarg. Construct a new STT instance to change the model."
            )

        if utils.is_given(language):
            self._language = LanguageCode(language)
            self._warn_on_unexpected_args(language=self._language)

        for stream in self._streams:
            # do not update model since this is likely user error
            stream.update_options(language=language)

    def _warn_on_unexpected_args(self, language: LanguageCode | None) -> None:
        """Logs a warning when arguments are unexpected.

        This is not necessarily an error since the API may support languages in the future.
        """
        if self._final_transcript_mode == "auto" and language and language.language != "en":
            logger.warning(
                f'Cartesia STT model="{self._model}" currently only supports English. You provided {language}, which may produce unexpected results.'
            )


@dataclass
class STTOptions:
    """
    .. deprecated::
        1.5.12 Not used anymore. Kept for backward compatibility.
    """

    model: STTModels | str
    language: LanguageCode | None
    encoding: STTEncoding
    sample_rate: int
    api_key: str
    base_url: str

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{_base_url_to_ws_base_url(self.base_url)}{path}"


SpeechStream = CartesiaRecognizeStream
