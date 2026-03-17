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
from dataclasses import dataclass, field, replace
from typing import Literal

import aiohttp

import azure.cognitiveservices.speech as speechsdk
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    tokenize,
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

# REST API format strings (for non-streaming synthesize())
SUPPORTED_OUTPUT_FORMATS = {
    8000: "raw-8khz-16bit-mono-pcm",
    16000: "raw-16khz-16bit-mono-pcm",
    22050: "raw-22050hz-16bit-mono-pcm",
    24000: "raw-24khz-16bit-mono-pcm",
    44100: "raw-44100hz-16bit-mono-pcm",
    48000: "raw-48khz-16bit-mono-pcm",
}

# SDK output format enums (for streaming via WebSocket V2)
SUPPORTED_SDK_FORMATS = {
    8000: speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
    16000: speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
    22050: speechsdk.SpeechSynthesisOutputFormat.Raw22050Hz16BitMonoPcm,
    24000: speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
    44100: speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm,
    48000: speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
}


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
    subscription_key: str | None
    region: str | None
    voice: str
    language: LanguageCode | None
    speech_endpoint: str | None
    deployment_id: str | None
    prosody: NotGivenOr[ProsodyConfig]
    style: NotGivenOr[StyleConfig]
    lexicon_uri: NotGivenOr[str]
    auth_token: str | None = None
    sentence_tokenizer: tokenize.SentenceTokenizer = field(
        default_factory=lambda: tokenize.blingfire.SentenceTokenizer(retain_format=True)
    )

    def get_endpoint_url(self) -> str:
        base = (
            self.speech_endpoint
            or f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        )
        if self.deployment_id:
            return f"{base}?deploymentId={self.deployment_id}"
        return base

    def get_ws_endpoint_url(self) -> str:
        """Construct the WebSocket V2 endpoint URL for text streaming."""
        if self.speech_endpoint:
            return self.speech_endpoint.replace("https://", "wss://").replace(
                "/cognitiveservices/v1", "/cognitiveservices/websocket/v2"
            )
        if self.region:
            return f"wss://{self.region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
        raise ValueError("Cannot construct WebSocket endpoint: region or speech_endpoint required")


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
        deployment_id: str | None = None,
        speech_auth_token: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        use_streaming: bool = False,
        sentence_tokenizer: tokenize.SentenceTokenizer | None = None,
    ) -> None:
        # Text streaming (WSv2) does not support SSML features
        has_ssml_features = is_given(prosody) or is_given(style) or is_given(lexicon_uri)
        if use_streaming and has_ssml_features:
            logger.warning(
                "Azure TTS text streaming does not support SSML features "
                "(prosody, style, lexicon). These will be ignored in streaming mode. "
                "Set use_streaming=False to use SSML features via the REST API."
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=use_streaming),
            sample_rate=sample_rate,
            num_channels=1,
        )
        if sample_rate not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. Supported: {list(SUPPORTED_OUTPUT_FORMATS)}"  # noqa: E501
            )

        if not speech_key:
            speech_key = os.environ.get("AZURE_SPEECH_KEY")

        if not speech_region:
            speech_region = os.environ.get("AZURE_SPEECH_REGION")

        if not speech_endpoint:
            speech_endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT")

        has_endpoint = bool(speech_endpoint)
        has_key_and_region = bool(speech_key and speech_region)
        has_token_and_region = bool(speech_auth_token and speech_region)
        if not (has_endpoint or has_key_and_region or has_token_and_region):
            raise ValueError(
                "Authentication requires one of: speech_endpoint (AZURE_SPEECH_ENDPOINT), "
                "speech_key & speech_region (AZURE_SPEECH_KEY & AZURE_SPEECH_REGION), "
                "or speech_auth_token & speech_region."
            )

        if is_given(prosody):
            prosody.validate()
        if is_given(style):
            style.validate()

        self._session = http_session
        self._use_streaming = use_streaming
        self._opts = _TTSOptions(
            sample_rate=sample_rate,
            subscription_key=speech_key,
            region=speech_region,
            speech_endpoint=speech_endpoint,
            voice=voice,
            deployment_id=deployment_id,
            language=LanguageCode(language) if language else None,
            prosody=prosody,
            style=style,
            lexicon_uri=lexicon_uri,
            auth_token=speech_auth_token,
            sentence_tokenizer=sentence_tokenizer
            or tokenize.blingfire.SentenceTokenizer(retain_format=True),
        )

    @property
    def model(self) -> str:
        return "unknown"

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

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _create_speech_config(self, opts: _TTSOptions) -> speechsdk.SpeechConfig:
        """Create a SpeechConfig for the WebSocket V2 text streaming endpoint."""
        ws_endpoint = opts.get_ws_endpoint_url()

        if opts.subscription_key:
            config = speechsdk.SpeechConfig(
                endpoint=ws_endpoint, subscription=opts.subscription_key
            )
        elif opts.auth_token:
            config = speechsdk.SpeechConfig(endpoint=ws_endpoint)
            config.authorization_token = opts.auth_token
        else:
            raise ValueError("Streaming TTS requires subscription_key or auth_token with a region")

        config.speech_synthesis_voice_name = opts.voice
        config.set_speech_synthesis_output_format(SUPPORTED_SDK_FORMATS[opts.sample_rate])

        if opts.language:
            config.speech_synthesis_language = str(opts.language)

        return config

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming TTS via Azure REST API. Supports full SSML features."""

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
        headers = {
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": SUPPORTED_OUTPUT_FORMATS[self._opts.sample_rate],
            "User-Agent": "LiveKit Agents",
        }
        if self._opts.auth_token:
            headers["Authorization"] = f"Bearer {self._opts.auth_token}"

        elif self._opts.subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = self._opts.subscription_key

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        try:
            async with self._tts._ensure_session().post(
                url=self._opts.get_endpoint_url(),
                headers=headers,
                data=self._build_ssml(),
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError(str(e)) from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS via Azure Speech SDK WebSocket V2 with text streaming.

    Text tokens from the LLM are fed incrementally into the Azure synthesizer
    as they arrive, enabling audio output to begin before the full utterance
    is complete. This significantly reduces time-to-first-audio-byte compared
    to the REST API approach.

    Note: Text streaming does not support SSML features (prosody, style, lexicon).
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            """Read text tokens from input channel and split into sentence segments."""
            input_stream: tokenize.SentenceStream | None = None
            async for data in self._input_ch:
                if isinstance(data, str):
                    if input_stream is None:
                        input_stream = self._opts.sentence_tokenizer.stream()
                        self._segments_ch.send_nowait(input_stream)
                    input_stream.push_text(data)
                elif isinstance(data, self._FlushSentinel):
                    if input_stream is not None:
                        input_stream.end_input()
                    input_stream = None

            if input_stream is not None:
                input_stream.end_input()
            self._segments_ch.close()

        async def _run_segments() -> None:
            """Process each sentence segment sequentially via Azure SDK."""
            async for input_stream in self._segments_ch:
                await self._run_stream(input_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)

    async def _run_stream(
        self,
        input_stream: tokenize.SentenceStream,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        """Run a single synthesis segment using Azure SDK text streaming.

        Creates a SpeechSynthesisRequest with TextStream input, feeds tokenized
        sentences as they arrive, and collects audio chunks via the synthesizing
        event callback, bridging the SDK's callback thread to asyncio.
        """
        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        error_holder: list[Exception] = []

        speech_config = self._tts._create_speech_config(self._opts)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        # Bridge SDK callbacks (fired on background thread) to asyncio
        def _on_synthesizing(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
            if evt.result.audio_data:
                loop.call_soon_threadsafe(audio_queue.put_nowait, evt.result.audio_data)

        def _on_completed(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        def _on_canceled(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
            details = evt.result.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error:
                error_holder.append(
                    APIConnectionError(f"Azure TTS synthesis canceled: {details.error_details}")
                )
            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        synthesizer.synthesizing.connect(_on_synthesizing)
        synthesizer.synthesis_completed.connect(_on_completed)
        synthesizer.synthesis_canceled.connect(_on_canceled)

        try:
            # Pre-connect the WebSocket for lower TTFB
            connection = speechsdk.Connection.from_speech_synthesizer(synthesizer)
            await loop.run_in_executor(None, lambda: connection.open(True))

            # Create a text stream request — tokens are fed incrementally
            request = speechsdk.SpeechSynthesisRequest(
                input_type=speechsdk.SpeechSynthesisRequestInputType.TextStream
            )
            result_future = synthesizer.speak_async(request)

            output_emitter.start_segment(segment_id=utils.shortuuid())

            async def _feed_text() -> None:
                """Feed tokenized sentences into the SDK as they arrive."""
                async for token_event in input_stream:
                    self._mark_started()
                    request.input_stream.write(token_event.token)
                request.input_stream.close()

            async def _consume_audio() -> None:
                """Drain audio chunks and push to emitter as they arrive."""
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    output_emitter.push(chunk)

            # Run text feeding and audio consumption concurrently so audio
            # is pushed to the emitter as soon as the SDK produces it,
            # rather than waiting for all text to be fed first.
            feed_task = asyncio.create_task(_feed_text())
            consume_task = asyncio.create_task(_consume_audio())
            try:
                await asyncio.gather(feed_task, consume_task)
            finally:
                await utils.aio.cancel_and_wait(feed_task, consume_task)

            if error_holder:
                raise error_holder[0]

            # Wait for the SDK to fully finalize (run in executor to avoid blocking)
            await loop.run_in_executor(None, result_future.get)

            output_emitter.end_segment()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (APIConnectionError, APIStatusError):
            raise
        except Exception as e:
            raise APIConnectionError(str(e)) from e
        finally:
            del synthesizer
