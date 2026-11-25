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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import assert_never

from google.auth.credentials import Credentials
from google.genai import types
from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, LanguageCode, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .realtime.api_proto import ClientMessage, LiveEndpoint, ServerMessage, dump
from .utils import gemini_backend

# Must be a Live-capable model: the plain chat models (e.g. gemini-2.5-flash) are
# rejected by the Live endpoint with "not supported for bidiGenerateContent".
DEFAULT_MODEL = "gemini-3.5-transcribe-live"
DEFAULT_SAMPLE_RATE = 16000


@dataclass
class _STTOptions:
    model: str
    language: LanguageCode | None
    language_codes: list[str] | None
    custom_vocabulary: list[str] | None
    sample_rate: int


class GeminiSTT(stt.STT):
    """Speech-to-text over a Gemini Live transcription session."""

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
        credentials: NotGivenOr[Credentials] = NOT_GIVEN,
        credentials_info: NotGivenOr[Mapping[str, object]] = NOT_GIVEN,
        credentials_file: NotGivenOr[str] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        http_options: types.HttpOptions | None = None,
    ) -> None:
        """Create a new instance of Gemini STT.

        Args:
            model: Live-capable Gemini model identifier. Defaults to
                "gemini-3.5-transcribe-live". Plain chat models are not accepted by the
                Live endpoint.
            language: Target language BCP-47 code or LanguageCode. Defaults to "en-US".
            language_codes: BCP-47 codes for the languages in the audio. Omit, or pass an
                empty list, to let the model detect the language.
            custom_vocabulary: Up to 1000 terms that bias recognition: names, acronyms
                and jargon the model would otherwise mishear.
            sample_rate: Sample rate in Hz. Defaults to 16000.
            api_key: Gemini API key. Defaults to the `GOOGLE_API_KEY` environment variable.
            vertexai: Whether to use Vertex AI. Defaults to `GOOGLE_GENAI_USE_VERTEXAI`, else
                to Vertex AI when credentials, a project or a location are given.
            credentials: Credentials for Vertex AI.
            credentials_info: Service account info for Vertex AI.
            credentials_file: Path to a service account JSON file for Vertex AI.
            project: Google Cloud project for Vertex AI. Defaults to `GOOGLE_CLOUD_PROJECT`,
                then the credentials' project.
            location: Google Cloud region for Vertex AI. Defaults to `GOOGLE_CLOUD_LOCATION`,
                then "global".
            http_options: Base URL, API version and headers.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=False,
            )
        )

        self._opts = _STTOptions(
            model=model,
            language=LanguageCode(language) if language is not None else None,
            language_codes=language_codes,
            custom_vocabulary=custom_vocabulary,
            sample_rate=sample_rate,
        )
        backend = gemini_backend(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            credentials=credentials,
            credentials_info=credentials_info,
            credentials_file=credentials_file,
            default_location="global",
        )
        self._endpoint = LiveEndpoint(
            vertexai=backend.vertexai,
            api_key=backend.api_key,
            project=backend.project,
            location=backend.location,
            credentials=backend.credentials,
            http_options=http_options,
        )

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
        opts = (
            replace(self._opts, language=LanguageCode(language))
            if is_given(language)
            else self._opts
        )
        return RecognizeStream(
            stt=self, opts=opts, conn_options=conn_options, endpoint=self._endpoint
        )

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Gemini STT only supports streaming recognition")


class RecognizeStream(stt.WebSocketStream[ServerMessage, ClientMessage]):
    """One Gemini Live transcription session: its setup asks for text and input transcription,
    the audio follows, and ``audioStreamEnd`` asks for the last turn's transcript."""

    def __init__(
        self,
        *,
        stt: GeminiSTT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        endpoint: LiveEndpoint,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._endpoint = endpoint
        self._default_language = opts.language or LanguageCode("en-US")
        self._language = self._default_language
        self._pending_interim = ""
        self._setup: ClientMessage = {}

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
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
        self._setup = {
            "setup": {
                "model": model,
                "generationConfig": {"responseModalities": [types.Modality.TEXT.value]},
                "inputAudioTranscription": dump(transcription),
            }
        }
        self._pending_interim = ""
        self._language = self._default_language
        return target

    def setup(self) -> ClientMessage:
        return self._setup

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> ClientMessage | None:
        match item:
            case rtc.AudioFrame():
                if not item.data:
                    return None
                blob = types.Blob(
                    data=bytes(item.data), mime_type=f"audio/pcm;rate={item.sample_rate}"
                )
                return {"realtimeInput": {"audio": dump(blob)}}
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                # Gemini finalizes the pending transcript rather than waiting for more audio
                return {"realtimeInput": {"audioStreamEnd": True}}
            case _:
                assert_never(item)

    def on_message(self, message: ServerMessage) -> None:
        # Both transcriptions carry a complete hypothesis, not deltas -- neither is accumulated:
        #   interimInputTranscription - speculative, resent in full while the user speaks
        #   inputTranscription - the finalized, authoritative transcript, emitted once the
        #     speech turn ends. One per turn, so a session yields several.
        # See https://ai.google.dev/gemini-api/docs/live-api/live-transcribe
        sc = message.get("serverContent")
        if not sc:
            return

        # interim first: when a message carries both, the final supersedes it
        if interim := sc.get("interimInputTranscription"):
            if code := interim.get("languageCode"):
                self._language = LanguageCode(code)
            if text := interim.get("text", "").strip():
                self._pending_interim = text
                self.report(self._transcript(text, final=False))

        if transcription := sc.get("inputTranscription"):
            if code := transcription.get("languageCode"):
                # the model detects the language when no hints are configured
                self._language = LanguageCode(code)
            if text := transcription.get("text", "").strip():
                self._pending_interim = ""
                self.report(self._transcript(text, final=True))
                self._language = self._default_language

        # a turn that produced only interims would otherwise be dropped
        if sc.get("generationComplete") or sc.get("turnComplete"):
            if self._pending_interim:
                self.report(self._transcript(self._pending_interim, final=True))
                self._pending_interim = ""
            self._language = self._default_language
            # A turn boundary can belong to earlier audio. It does not acknowledge
            # audioStreamEnd or promise that no later transcript will arrive.

    def _transcript(self, text: str, *, final: bool) -> stt.Transcript:
        return stt.Transcript(text, self._language, final=final, confidence=1.0)
