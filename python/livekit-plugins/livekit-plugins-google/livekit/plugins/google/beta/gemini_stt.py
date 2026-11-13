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

import json
from dataclasses import dataclass
from typing import Any

from google.genai import types
from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, LanguageCode, http, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ..realtime.api_proto import ClientMessage, LiveEndpoint, ServerMessage, dump

# Must be a Live-capable model: the plain chat models (e.g. gemini-2.5-flash) are
# rejected by the Live endpoint with "not supported for bidiGenerateContent".
DEFAULT_MODEL = "gemini-3.5-transcribe-live"
DEFAULT_SAMPLE_RATE = 16000


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
            http_options: Optional ``types.HttpOptions`` (or its dict): base URL, API version
                and headers.
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
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Gemini STT only supports streaming recognition")


class RecognizeStream(stt.SocketStream[ServerMessage, ClientMessage]):
    """One Gemini Live transcription session: its setup asks for text and input transcription,
    the audio follows, and ``audioStreamEnd`` asks for the last turn's transcript."""

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
        self._endpoint = _endpoint(opts, api_key, credentials, credentials_path, http_options)
        self._default_language = LanguageCode(str(opts.language or "en-US"))
        self._language = self._default_language
        self._pending_interim = ""
        self._audio_ended = False

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        target, model = await self._endpoint.target(self._opts.model)
        language_codes = self._opts.language_codes
        if language_codes is None and self._opts.language:
            language_codes = [str(self._opts.language)]
        # An empty list means "detect the language", per
        # https://ai.google.dev/gemini-api/docs/live-api/live-transcribe
        # Both backends accept these -- verified against gemini-3.5-transcribe-live on the
        # Gemini Developer API, which takes them the same as Vertex.
        transcription = types.AudioTranscriptionConfig(
            language_codes=language_codes or [],
            custom_vocabulary=self._opts.custom_vocabulary or None,
        )
        self.send_command(
            {
                "setup": {
                    "model": model,
                    "generationConfig": {"responseModalities": [types.Modality.TEXT.value]},
                    "inputAudioTranscription": dump(transcription),
                }
            }
        )
        self._pending_interim = ""
        self._language = self._default_language
        return target

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[ClientMessage]:
        if isinstance(item, http.EndOfInput):
            # Gemini finalizes the pending transcript rather than waiting for more audio
            self._audio_ended = True
            return {"realtimeInput": {"audioStreamEnd": True}}
        if isinstance(item, stt.RecognizeStream.Flush) or not item.data:
            return None
        blob = types.Blob(data=bytes(item.data), mime_type=f"audio/pcm;rate={item.sample_rate}")
        return {"realtimeInput": {"audio": dump(blob)}}

    def on_message(self, message: ServerMessage) -> list[stt.Report] | http.Done | None:
        # Both transcriptions carry a complete hypothesis, not deltas -- neither is accumulated:
        #   interimInputTranscription - speculative, resent in full while the user speaks
        #   inputTranscription - the finalized, authoritative transcript, emitted once the
        #     speech turn ends. One per turn, so a session yields several.
        # See https://ai.google.dev/gemini-api/docs/live-api/live-transcribe
        sc = message.get("serverContent")
        if not sc:
            return None
        reports: list[stt.Report] = []

        # interim first: when a message carries both, the final supersedes it
        if interim := sc.get("interimInputTranscription"):
            if code := interim.get("languageCode"):
                self._language = LanguageCode(code)
            if text := interim.get("text", "").strip():
                self._pending_interim = text
                reports.append(self._transcript(text, final=False))

        if transcription := sc.get("inputTranscription"):
            if code := transcription.get("languageCode"):
                # the model detects the language when no hints are configured
                self._language = LanguageCode(code)
            if text := transcription.get("text", "").strip():
                self._pending_interim = ""
                reports.append(self._transcript(text, final=True))
                self._language = self._default_language

        # a turn that produced only interims would otherwise be dropped
        if sc.get("generationComplete") or sc.get("turnComplete"):
            if self._pending_interim:
                reports.append(self._transcript(self._pending_interim, final=True))
                self._pending_interim = ""
            self._language = self._default_language
            if self._audio_ended:
                # the last turn after the audio ended: the session is done
                for report in reports:
                    self.report(report)
                return http.DONE
        return reports

    def _transcript(self, text: str, *, final: bool) -> stt.Transcript:
        return stt.Transcript(text, self._language, final=final, confidence=1.0)


def _endpoint(
    opts: _STTOptions,
    api_key: str | None,
    credentials: Any | None,
    credentials_path: str | None,
    http_options: Any | None,
) -> LiveEndpoint:
    creds = None
    project_id = opts.project
    if credentials:
        if isinstance(credentials, str):
            from google.oauth2 import service_account

            json_account_info = json.loads(credentials)
            project_id = project_id or json_account_info.get("project_id")
            creds = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                json_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        else:
            creds = credentials
    elif credentials_path:
        from google.oauth2 import service_account

        with open(credentials_path) as f:
            project_id = project_id or json.load(f).get("project_id")
        creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    vertexai = opts.vertexai
    if vertexai is None:
        vertexai = api_key is None and bool(creds or project_id or opts.location)
    return LiveEndpoint(
        vertexai=vertexai,
        api_key=api_key,
        project=project_id if vertexai else None,
        location=(opts.location or "global") if vertexai else None,
        credentials=creds if vertexai else None,
        http_options=(
            types.HttpOptions.model_validate(http_options) if http_options is not None else None
        ),
    )
