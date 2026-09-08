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

# Must be a Live-capable model: the plain chat models (e.g. gemini-2.5-flash) are
# rejected by the Live endpoint with "not supported for bidiGenerateContent".
DEFAULT_MODEL = "gemini-3.5-transcribe-live"
DEFAULT_SAMPLE_RATE = 16000

# The finalized transcript for the last turn lands after audio_stream_end, so teardown
# waits briefly for it. Bounded so a silent session can't hang.
FINALIZE_TIMEOUT = 2.0


def _is_session_duration_close(e: BaseException) -> bool:
    """True when the socket closed because a session hit its duration cap.

    Live transcription sessions stream for up to 10 minutes, then the server sends a
    GoAway and closes with 1008. The retry layer above reconnects within ~0.1s and
    events keep flowing until the actual close, so this is routine rather than a
    failure and shouldn't be logged as one every ten minutes.
    """
    text = str(e)
    return "GoAway" in text or "session duration" in text


@dataclass
class _STTOptions:
    model: str
    language: LanguageCode | str | None
    language_codes: list[str] | None
    custom_vocabulary: list[str] | None
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
        language_codes: list[str] | None = None,
        custom_vocabulary: list[str] | None = None,
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
            model: Live-capable Gemini model identifier. Defaults to
                "gemini-3.5-transcribe-live". Plain chat models are not accepted by the
                Live endpoint.
            language: Target language BCP-47 code or LanguageCode. Defaults to "en-US".
            language_codes: BCP-47 codes for the languages in the audio. Omit, or pass an
                empty list, to let the model detect the language.
            custom_vocabulary: Up to 1000 terms that bias recognition -- names, acronyms
                and jargon the model would otherwise mishear.
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
                offline_recognize=False,
            )
        )

        lang_code = LanguageCode(language) if language is not None else None

        self._opts = _STTOptions(
            model=model,
            language=lang_code,
            language_codes=language_codes,
            custom_vocabulary=custom_vocabulary,
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
                language=LanguageCode(language),
                language_codes=self._opts.language_codes,
                custom_vocabulary=self._opts.custom_vocabulary,
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

        lang_codes = self._opts.language_codes
        if lang_codes is None and self._opts.language:
            lang_codes = [str(self._opts.language)]

        # An empty list means "detect the language", per
        # https://ai.google.dev/gemini-api/docs/live-api/live-transcribe
        # Both backends accept these -- verified against gemini-3.5-transcribe-live on the
        # Gemini Developer API, which takes them the same as Vertex.
        input_audio_transcription = types.AudioTranscriptionConfig(
            language_codes=lang_codes or [],
            custom_vocabulary=self._opts.custom_vocabulary or None,
        )

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=input_audio_transcription,
        )

        try:
            async with client.aio.live.connect(model=self._opts.model, config=config) as session:
                send_task = asyncio.create_task(self._send_loop(session))
                receive_task = asyncio.create_task(self._receive_loop(session))
                tasks = [send_task, receive_task]

                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                    # A closed input channel doesn't mean Gemini is done: the tail of the
                    # transcript is only emitted once `finished`/`turn_complete` arrives.
                    # Give the receive loop a bounded grace period instead of cancelling
                    # it the moment _send_loop returns, which would drop that final.
                    if send_task.done() and not receive_task.done():
                        await asyncio.wait([receive_task], timeout=FINALIZE_TIMEOUT)
                finally:
                    await utils.aio.cancel_and_wait(*tasks)

                for task in tasks:
                    if not task.cancelled():
                        task.result()

        # Provider errors can quote the audio payload or transcript, so the text is logged
        # as a redactable attribute and kept out of the raised message, whose body the
        # framework interpolates into its retry log (see REVIEW.md).
        except (ClientError, ServerError, APIError) as e:
            log = logger.debug if _is_session_duration_close(e) else logger.warning
            log(
                "Gemini STT request failed",
                extra={"error_type": type(e).__name__, "lk.pii.error": str(e)},
            )
            raise APIStatusError(
                message=f"Gemini STT request failed ({type(e).__name__})",
                status_code=getattr(e, "code", -1),
                body=None,
            ) from None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log = logger.debug if _is_session_duration_close(e) else logger.warning
            log(
                "Gemini STT connection failed",
                extra={"error_type": type(e).__name__, "lk.pii.error": str(e)},
            )
            raise APIConnectionError(f"Gemini STT connection failed ({type(e).__name__})") from None

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

        # Tell Gemini the audio stream ended so it finalizes the pending transcript
        # rather than waiting for more frames that will never arrive.
        with contextlib.suppress(Exception):
            await session.send_realtime_input(audio_stream_end=True)

    def _speech_event(
        self,
        event_type: stt.SpeechEventType,
        text: str,
        language: LanguageCode,
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=event_type,
            alternatives=[stt.SpeechData(language=language, text=text, confidence=1.0)],
        )

    async def _receive_loop(self, session: Any) -> None:
        default_lang = (
            self._opts.language
            if isinstance(self._opts.language, LanguageCode)
            else LanguageCode(str(self._opts.language or "en-US"))
        )
        # Both streams carry a complete hypothesis, not deltas -- neither is accumulated:
        #   interim_input_transcription - speculative, resent in full while the user speaks
        #   input_transcription - the finalized, authoritative transcript, emitted once the
        #     speech turn ends. One per turn, so a session yields several.
        # See https://ai.google.dev/gemini-api/docs/live-api/live-transcribe
        pending_interim = ""
        lang = default_lang

        try:
            turn = session.receive()
            async for message in turn:
                sc = getattr(message, "server_content", None)
                if not sc:
                    continue

                # interim first: when a message carries both, the final supersedes it
                if interim := getattr(sc, "interim_input_transcription", None):
                    if code := getattr(interim, "language_code", None):
                        lang = LanguageCode(code)
                    if text := (interim.text or "").strip():
                        pending_interim = text
                        self._event_ch.send_nowait(
                            self._speech_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, text, lang)
                        )

                if transcription := getattr(sc, "input_transcription", None):
                    if code := getattr(transcription, "language_code", None):
                        # the model detects the language when no hints are configured
                        lang = LanguageCode(code)
                    if text := (transcription.text or "").strip():
                        pending_interim = ""
                        self._event_ch.send_nowait(
                            self._speech_event(stt.SpeechEventType.FINAL_TRANSCRIPT, text, lang)
                        )
                        lang = default_lang

                # a turn that produced only interims would otherwise be dropped
                if getattr(sc, "generation_complete", False) or getattr(sc, "turn_complete", False):
                    if pending_interim:
                        self._event_ch.send_nowait(
                            self._speech_event(
                                stt.SpeechEventType.FINAL_TRANSCRIPT, pending_interim, lang
                            )
                        )
                        pending_interim = ""
                    lang = default_lang

        except APIError as e:
            # Provider errors can quote the audio payload or transcript, so the text stays
            # out of the message body (see REVIEW.md).
            if getattr(e, "code", None) == 1000 or "1000" in str(e):
                logger.debug("Gemini ASR session closed normally", extra={"lk.pii.error": str(e)})
            elif _is_session_duration_close(e):
                logger.debug(
                    "Gemini ASR session reached its duration limit, reconnecting",
                    extra={"lk.pii.error": str(e)},
                )
                raise
            else:
                logger.warning("Gemini ASR receive error", extra={"lk.pii.error": str(e)})
                raise
        except Exception as e:
            logger.debug("Gemini ASR receive loop ended", extra={"lk.pii.error": str(e)})
            raise
