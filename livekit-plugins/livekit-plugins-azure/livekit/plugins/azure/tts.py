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
import os
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlparse

import azure.cognitiveservices.speech as speechsdk  # type: ignore
from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger

SUPPORTED_OUTPUT_FORMATS = {
    8000: speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
    16000: speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
    22050: speechsdk.SpeechSynthesisOutputFormat.Raw22050Hz16BitMonoPcm,
    24000: speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
    44100: speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm,
    48000: speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
}

_CANCELLATION_STATUS_CODES = {
    speechsdk.CancellationErrorCode.BadRequest: 400,
    speechsdk.CancellationErrorCode.AuthenticationFailure: 401,
    speechsdk.CancellationErrorCode.Forbidden: 403,
    speechsdk.CancellationErrorCode.TooManyRequests: 429,
    speechsdk.CancellationErrorCode.ServiceError: 500,
    speechsdk.CancellationErrorCode.ServiceUnavailable: 503,
}

_SYNTHESIS_TIMEOUT = 30.0


@dataclass
class ProsodyConfig:
    rate: Literal["x-slow", "slow", "medium", "fast", "x-fast"] | float | None = None
    volume: Literal["silent", "x-soft", "soft", "medium", "loud", "x-loud"] | float | None = None
    pitch: Literal["x-low", "low", "medium", "high", "x-high"] | None = None

    def validate(self) -> None:
        if self.rate:
            if isinstance(self.rate, float) and not 0.5 <= self.rate <= 2:
                raise ValueError("Prosody rate must be between 0.5 and 2")
            if isinstance(self.rate, str) and self.rate not in [
                "x-slow",
                "slow",
                "medium",
                "fast",
                "x-fast",
            ]:
                raise ValueError(
                    "Prosody rate must be one of 'x-slow', 'slow', 'medium', 'fast', 'x-fast'"
                )
        if self.volume:
            if isinstance(self.volume, float) and not 0 <= self.volume <= 100:
                raise ValueError("Prosody volume must be between 0 and 100")
            if isinstance(self.volume, str) and self.volume not in [
                "silent",
                "x-soft",
                "soft",
                "medium",
                "loud",
                "x-loud",
            ]:
                raise ValueError(
                    "Prosody volume must be one of 'silent', 'x-soft', 'soft', 'medium', 'loud', 'x-loud'"  # noqa: E501
                )
        if self.pitch and self.pitch not in [
            "x-low",
            "low",
            "medium",
            "high",
            "x-high",
        ]:
            raise ValueError(
                "Prosody pitch must be one of 'x-low', 'low', 'medium', 'high', 'x-high'"
            )

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class StyleConfig:
    style: str
    degree: float | None = None

    def validate(self) -> None:
        if self.degree is not None and not 0.1 <= self.degree <= 2.0:
            raise ValueError("Style degree must be between 0.1 and 2.0")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class _TTSOptions:
    sample_rate: int
    voice: str
    language: LanguageCode | None
    speech_endpoint: str | None
    speech_host: str | None
    region: str | None
    deployment_id: str | None
    prosody: NotGivenOr[ProsodyConfig]
    style: NotGivenOr[StyleConfig]
    lexicon_uri: NotGivenOr[str]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "en-US-JennyNeural",
        language: str | None = None,
        sample_rate: int = 24000,
        prosody: NotGivenOr[ProsodyConfig] = NOT_GIVEN,
        style: NotGivenOr[StyleConfig] = NOT_GIVEN,
        lexicon_uri: NotGivenOr[str] = NOT_GIVEN,
        speech_key: str | None = None,
        speech_region: str | None = None,
        speech_endpoint: str | None = None,
        speech_host: str | None = None,
        deployment_id: str | None = None,
        speech_auth_token: str | None = None,
        http_session: Any | None = None,
    ) -> None:
        """
        Create a new instance of Azure TTS, backed by the Azure Speech SDK.

        One of ``speech_endpoint``, ``speech_host``, ``speech_key`` and ``speech_region``, or
        ``speech_auth_token`` and ``speech_region`` must be set, either as arguments or through
        the ``AZURE_SPEECH_ENDPOINT``, ``AZURE_SPEECH_HOST``, ``AZURE_SPEECH_KEY`` and
        ``AZURE_SPEECH_REGION`` environment variables. ``speech_auth_token`` must be passed as an
        argument since it's an ephemeral token.

        Args:
            voice: Voice name, e.g. ``en-US-JennyNeural``.
            language: SSML ``xml:lang``, defaults to ``en-US``.
            sample_rate: Output sample rate, one of 8000, 16000, 22050, 24000, 44100, 48000.
            prosody: Rate, volume and pitch applied to the synthesized text.
            style: Speaking style (``mstts:express-as``) applied to the synthesized text.
            lexicon_uri: Custom lexicon URI.
            speech_key: Speech resource key.
            speech_region: Speech resource region.
            speech_endpoint: Speech SDK endpoint, takes precedence over ``speech_region``.
            speech_host: Speech container host, e.g. ``ws://localhost:5000``.
            deployment_id: Custom voice deployment ID.
            speech_auth_token: Microsoft Entra authorization token.
            http_session: Unused, the Speech SDK manages its own connections.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        if sample_rate not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. Supported: {list(SUPPORTED_OUTPUT_FORMATS)}"  # noqa: E501
            )

        if http_session is not None:
            logger.warning("http_session is ignored, Azure TTS uses the Speech SDK's connections")

        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        speech_endpoint = speech_endpoint or os.environ.get("AZURE_SPEECH_ENDPOINT")
        speech_host = speech_host or os.environ.get("AZURE_SPEECH_HOST")

        if not (
            speech_endpoint
            or speech_host
            or (speech_key and speech_region)
            or (speech_auth_token and speech_region)
        ):
            raise ValueError(
                "Authentication requires one of: speech_endpoint (AZURE_SPEECH_ENDPOINT), "
                "speech_host (AZURE_SPEECH_HOST), "
                "speech_key & speech_region (AZURE_SPEECH_KEY & AZURE_SPEECH_REGION), "
                "or speech_auth_token & speech_region."
            )

        if speech_endpoint or speech_host:
            speech_region = None

        if is_given(prosody):
            prosody.validate()
        if is_given(style):
            style.validate()

        # let the SpeechConfig constructor validate the arguments
        self._speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region,
            endpoint=speech_endpoint,
            host=speech_host,
            auth_token=speech_auth_token,
        )
        self._speech_config.set_speech_synthesis_output_format(
            SUPPORTED_OUTPUT_FORMATS[sample_rate]
        )
        if deployment_id:
            self._speech_config.endpoint_id = deployment_id

        self._opts = _TTSOptions(
            sample_rate=sample_rate,
            voice=voice,
            language=LanguageCode(language) if language else None,
            speech_endpoint=speech_endpoint,
            speech_host=speech_host,
            region=speech_region,
            deployment_id=deployment_id,
            prosody=prosody,
            style=style,
            lexicon_uri=lexicon_uri,
        )
        self._pool = utils.ConnectionPool[speechsdk.SpeechSynthesizer](
            connect_cb=self._connect_synthesizer,
            close_cb=self._close_synthesizer,
        )

    @property
    def model(self) -> str:
        if location := (self._opts.speech_endpoint or self._opts.speech_host):
            location = location.strip()
            parsed = urlparse(location if "://" in location else f"//{location}")
            location = parsed.hostname or location
        elif self._opts.region:
            location = self._opts.region
        else:
            return "unknown"
        model = f"{location}:{self._opts.voice}"
        if self._opts.deployment_id:
            model += f":{self._opts.deployment_id}"
        return model

    @property
    def provider(self) -> str:
        return "Azure TTS"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        prosody: NotGivenOr[ProsodyConfig] = NOT_GIVEN,
        style: NotGivenOr[StyleConfig] = NOT_GIVEN,
        lexicon_uri: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(prosody):
            prosody.validate()
            self._opts.prosody = prosody
        if is_given(style):
            style.validate()
            self._opts.style = style
        if is_given(lexicon_uri):
            self._opts.lexicon_uri = lexicon_uri

    async def _connect_synthesizer(self, timeout: float) -> speechsdk.SpeechSynthesizer:
        def _connect() -> speechsdk.SpeechSynthesizer:
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config, audio_config=None
            )
            speechsdk.Connection.from_speech_synthesizer(synthesizer).open(True)
            return synthesizer

        return await asyncio.wait_for(asyncio.to_thread(_connect), timeout)

    async def _close_synthesizer(self, synthesizer: speechsdk.SpeechSynthesizer) -> None:
        await asyncio.to_thread(speechsdk.Connection.from_speech_synthesizer(synthesizer).close)

    def prewarm(self) -> None:
        self._pool.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _build_ssml(self) -> str:
        lang = self._opts.language or "en-US"
        ssml = (
            f'<speak version="1.0" '
            f'xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="http://www.w3.org/2001/mstts" '
            f'xml:lang="{lang}">'
        )
        ssml += f'<voice name="{self._opts.voice}">'

        if is_given(self._opts.lexicon_uri):
            ssml += f'<lexicon uri="{self._opts.lexicon_uri}"/>'

        if is_given(self._opts.style):
            degree = f' styledegree="{self._opts.style.degree}"' if self._opts.style.degree else ""
            ssml += f'<mstts:express-as style="{self._opts.style.style}"{degree}>'

        if is_given(self._opts.prosody):
            p = self._opts.prosody

            rate_attr = f' rate="{p.rate}"' if p.rate is not None else ""
            vol_attr = f' volume="{p.volume}"' if p.volume is not None else ""
            pitch_attr = f' pitch="{p.pitch}"' if p.pitch is not None else ""
            ssml += f"<prosody{rate_attr}{vol_attr}{pitch_attr}>{self.input_text}</prosody>"
        else:
            ssml += self.input_text

        if is_given(self._opts.style):
            ssml += "</mstts:express-as>"

        ssml += "</voice></speak>"
        return ssml

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()

        def _push(data: bytes) -> None:
            if not done.done():
                output_emitter.push(data)

        def _resolve(error: Exception | None) -> None:
            if done.done():
                return
            if error is None:
                done.set_result(None)
            else:
                done.set_exception(error)

        # SDK callbacks run on SDK threads, the loop may already be closed
        def _threadsafe(cb: Any, *args: Any) -> None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(cb, *args)

        def _on_synthesizing(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
            _threadsafe(_push, evt.result.audio_data)

        def _on_completed(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
            _threadsafe(_resolve, None)

        def _on_canceled(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
            _threadsafe(_resolve, _cancellation_error(evt.result.cancellation_details))

        try:
            synthesizer = await self._tts._pool.get(timeout=self._conn_options.timeout)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError(str(e)) from e

        synthesizer.synthesizing.connect(_on_synthesizing)
        synthesizer.synthesis_completed.connect(_on_completed)
        synthesizer.synthesis_canceled.connect(_on_canceled)
        succeeded = False
        try:
            synthesizer.speak_ssml_async(self._build_ssml())
            await asyncio.wait_for(done, _SYNTHESIS_TIMEOUT)
            succeeded = True
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(str(e)) from e
        finally:
            synthesizer.synthesizing.disconnect_all()
            synthesizer.synthesis_completed.disconnect_all()
            synthesizer.synthesis_canceled.disconnect_all()
            if succeeded:
                self._tts._pool.put(synthesizer)
            else:
                synthesizer.stop_speaking_async()
                self._tts._pool.remove(synthesizer)


def _cancellation_error(details: speechsdk.SpeechSynthesisCancellationDetails) -> APIError:
    message = details.error_details or str(details.reason)
    if details.error_code == speechsdk.CancellationErrorCode.ServiceTimeout:
        return APITimeoutError(message)
    if status_code := _CANCELLATION_STATUS_CODES.get(details.error_code):
        return APIStatusError(message, status_code=status_code)
    return APIConnectionError(message)
