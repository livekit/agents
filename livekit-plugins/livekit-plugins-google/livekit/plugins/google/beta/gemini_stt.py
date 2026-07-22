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
import contextlib
import json
from dataclasses import dataclass
from typing import Any

from google.genai import Client, types
from google.genai.errors import APIError, ClientError, ServerError
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SAMPLE_RATE = 16000


@dataclass
class _STTOptions:
    model: str
    language: LanguageCode | str | None
    language_hints: list[str] | None
    sample_rate: int
    vertexai: bool | None
    project: str | None
    location: str | None


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language: LanguageCode | str | None = "en-US",
        language_hints: list[str] | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        credentials: Any | None = None,
        credentials_path: str | None = None,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        http_options: Any | None = None,
    ) -> None:
        """Create a new instance of Gemini STT.

        Args:
            model: Gemini model identifier for STT. Defaults to "gemini-2.5-flash".
            language: Target language BCP-47 code or LanguageCode. Defaults to "en-US".
            language_hints: List of language code hints for recognition.
            sample_rate: Sample rate in Hz. Defaults to 16000.
            api_key: Optional Gemini API key. If not set, uses environment variables.
            vertexai: Whether to use Vertex AI backend.
            credentials: Service account credentials object or JSON string.
            credentials_path: Path to service account credentials JSON file.
            project: Google Cloud project ID (required for Vertex AI).
            location: Google Cloud region (e.g. "us-central1").
            http_options: Optional HTTP options for Client.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        lang_code: LanguageCode | None = None
        if language is not None:
            lang_code = LanguageCode(language) if isinstance(language, str) else language

        self._opts = _STTOptions(
            model=model,
            language=lang_code,
            language_hints=language_hints,
            sample_rate=sample_rate,
            vertexai=vertexai if is_given(vertexai) else None,
            project=project if is_given(project) else None,
            location=location if is_given(location) else None,
        )

        self._api_key = api_key if is_given(api_key) else None
        self._credentials = credentials
        self._credentials_path = credentials_path
        self._http_options = http_options

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "google"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        opts = self._opts
        if is_given(language):
            opts = _STTOptions(
                model=self._opts.model,
                language=LanguageCode(language) if isinstance(language, str) else language,
                language_hints=self._opts.language_hints,
                sample_rate=self._opts.sample_rate,
                vertexai=self._opts.vertexai,
                project=self._opts.project,
                location=self._opts.location,
            )

        return RecognizeStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            api_key=self._api_key,
            credentials=self._credentials,
            credentials_path=self._credentials_path,
            http_options=self._http_options,
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Gemini STT only supports streaming recognition")


class RecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        api_key: str | None,
        credentials: Any | None,
        credentials_path: str | None,
        http_options: Any | None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._credentials = credentials
        self._credentials_path = credentials_path
        self._http_options = http_options

    async def _run(self) -> None:
        creds = None
        project_id = self._opts.project

        if self._credentials:
            if isinstance(self._credentials, str):
                from google.oauth2 import service_account

                json_account_info = json.loads(self._credentials)
                project_id = project_id or json_account_info.get("project_id")
                creds = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                    json_account_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            else:
                creds = self._credentials
        elif self._credentials_path:
            from google.oauth2 import service_account

            with open(self._credentials_path) as f:
                json_account_info = json.load(f)
                project_id = project_id or json_account_info.get("project_id")
            creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
                self._credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        is_enterprise = self._opts.vertexai
        if is_enterprise is None:
            if self._api_key is not None:
                is_enterprise = False
            elif creds or project_id or self._opts.location:
                is_enterprise = True

        client_kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "http_options": self._http_options,
        }
        if is_enterprise:
            client_kwargs["enterprise"] = True
            if project_id:
                client_kwargs["project"] = project_id
            client_kwargs["location"] = self._opts.location or "global"
            if creds:
                client_kwargs["credentials"] = creds

        client = Client(**client_kwargs)

        lang_hints = self._opts.language_hints
        if not lang_hints and self._opts.language:
            lang_hints = [str(self._opts.language)]

        if getattr(client._api_client, "vertexai", False):
            if lang_hints:
                input_audio_transcription = types.AudioTranscriptionConfig(
                    language_hints={"language_codes": lang_hints},
                )
            else:
                input_audio_transcription = types.AudioTranscriptionConfig(
                    language_auto={},
                )
        else:
            if lang_hints:
                logger.warning(
                    "language_hints/language is only supported on Vertex AI (enterprise=True). "
                    "Ignoring language configuration for Gemini Developer API."
                )
            input_audio_transcription = types.AudioTranscriptionConfig()

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=input_audio_transcription,
        )

        try:
            async with client.aio.live.connect(model=self._opts.model, config=config) as session:
                send_task = asyncio.create_task(self._send_loop(session))
                receive_task = asyncio.create_task(self._receive_loop(session))

                done, pending = await asyncio.wait(
                    [send_task, receive_task], return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                for task in done:
                    task.result()

        except (ClientError, ServerError, APIError) as e:
            raise APIStatusError(
                message=str(e),
                status_code=getattr(e, "code", -1),
                body=None,
            ) from e
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise APIConnectionError(f"Gemini STT connection failed: {e}") from e

    async def _send_loop(self, session: Any) -> None:
        sample_rate = self._opts.sample_rate

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                pcm_data = data.data.tobytes()
                if pcm_data:
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm_data,
                            mime_type=f"audio/pcm;rate={sample_rate}",
                        )
                    )
            elif isinstance(data, self._FlushSentinel):
                pass

    async def _receive_loop(self, session: Any) -> None:
        lang = (
            self._opts.language
            if isinstance(self._opts.language, LanguageCode)
            else LanguageCode(str(self._opts.language or "en-US"))
        )

        try:
            turn = session.receive()
            async for message in turn:
                sc = getattr(message, "server_content", None)
                if not sc:
                    continue

                if getattr(sc, "input_transcription", None):
                    text = sc.input_transcription.text
                    if text:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        language=lang,
                                        text=text,
                                        confidence=1.0,
                                    )
                                ],
                            )
                        )

                if getattr(sc, "interim_input_transcription", None):
                    text = sc.interim_input_transcription.text
                    if text:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        language=lang,
                                        text=text,
                                        confidence=1.0,
                                    )
                                ],
                            )
                        )
        except APIError as e:
            if getattr(e, "code", None) == 1000 or "1000" in str(e):
                logger.debug(f"Gemini ASR session closed normally: {e}")
            else:
                logger.warning(f"Gemini ASR receive error: {e}")
                raise
        except Exception as e:
            logger.debug(f"Gemini ASR receive loop ended: {e}")
            raise
