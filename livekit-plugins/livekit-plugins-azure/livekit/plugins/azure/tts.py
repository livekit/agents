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
from dataclasses import dataclass
from typing import Callable, Literal

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    tts,
    utils,
)

import azure.cognitiveservices.speech as speechsdk  # type: ignore

from .log import logger

# only raw & pcm
SUPPORTED_SAMPLE_RATE = {
    8000: speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
    16000: speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
    22050: speechsdk.SpeechSynthesisOutputFormat.Raw22050Hz16BitMonoPcm,
    24000: speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
    44100: speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm,
    48000: speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
}


@dataclass
class ProsodyConfig:
    """
    Prosody configuration for Azure TTS.

    Args:
        rate: Speaking rate. Can be one of "x-slow", "slow", "medium", "fast", "x-fast", or a float. A float value of 1.0 represents normal speed.
        volume: Speaking volume. Can be one of "silent", "x-soft", "soft", "medium", "loud", "x-loud", or a float. A float value of 100 (x-loud) represents the highest volume and it's the default pitch.
        pitch: Speaking pitch. Can be one of "x-low", "low", "medium", "high", "x-high". The default pitch is "medium".
    """

    rate: Literal["x-slow", "slow", "medium", "fast", "x-fast"] | float | None = None
    volume: (
        Literal["silent", "x-soft", "soft", "medium", "loud", "x-loud"] | float | None
    ) = None
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
                    "Prosody volume must be one of 'silent', 'x-soft', 'soft', 'medium', 'loud', 'x-loud'"
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

    def __post_init__(self):
        self.validate()


@dataclass
class StyleConfig:
    """
    Style configuration for Azure TTS neural voices.

    Args:
        style: Speaking style for neural voices. Examples: "cheerful", "sad", "angry", etc.
        degree: Intensity of the style, from 0.1 to 2.0.
    """

    style: str
    degree: float | None = None

    def validate(self) -> None:
        if self.degree is not None and not 0.1 <= self.degree <= 2.0:
            raise ValueError("Style degree must be between 0.1 and 2.0")

    def __post_init__(self):
        self.validate()


@dataclass
class _TTSOptions:
    sample_rate: int
    speech_key: str | None = None
    speech_region: str | None = None
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-container-ntts?tabs=container#use-the-container
    speech_host: str | None = None
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    voice: str | None = None
    # for using custom voices (see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis?tabs=browserjs%2Cterminal&pivots=programming-language-python#use-a-custom-endpoint)
    endpoint_id: str | None = None
    # for using Microsoft Entra auth (see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-configure-azure-ad-auth?tabs=portal&pivots=programming-language-python)
    speech_auth_token: str | None = None
    # Useful to specify the language with multi-language voices
    language: str | None = None
    # See https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice#adjust-prosody
    prosody: ProsodyConfig | None = None
    speech_endpoint: str | None = None
    style: StyleConfig | None = None
    # See https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis?tabs=browserjs%2Cterminal&pivots=programming-language-python
    on_bookmark_reached_event: Callable | None = None,
    on_synthesis_canceled_event: Callable | None = None,
    on_synthesis_completed_event: Callable | None = None,
    on_synthesis_started_event: Callable | None = None
    on_synthesizing_event: Callable | None = None
    on_viseme_event: Callable | None = None
    on_word_boundary_event: Callable | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        voice: str | None = None,
        language: str | None = None,
        prosody: ProsodyConfig | None = None,
        speech_key: str | None = None,
        speech_region: str | None = None,
        speech_host: str | None = None,
        speech_auth_token: str | None = None,
        endpoint_id: str | None = None,
        style: StyleConfig | None = None,
        on_bookmark_reached_event: Callable | None = None,
        on_synthesis_canceled_event: Callable | None = None,
        on_synthesis_completed_event: Callable | None = None,
        on_synthesis_started_event: Callable | None = None,
        on_synthesizing_event: Callable | None = None,
        on_viseme_event: Callable | None = None,
        on_word_boundary_event: Callable | None = None,
    ) -> None:
        """
        Create a new instance of Azure TTS.

        Either ``speech_host`` or ``speech_key`` and ``speech_region`` or
        ``speech_auth_token`` and ``speech_region`` must be set using arguments.
         Alternatively,  set the ``AZURE_SPEECH_HOST``, ``AZURE_SPEECH_KEY``
        and ``AZURE_SPEECH_REGION`` environmental variables, respectively.
        ``speech_auth_token`` must be set using the arguments as it's an ephemeral token.
        """

        if sample_rate not in SUPPORTED_SAMPLE_RATE:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. Supported sample rates: {list(SUPPORTED_SAMPLE_RATE.keys())}"
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        speech_host = speech_host or os.environ.get("AZURE_SPEECH_HOST")
        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")

        if not (
            speech_host
            or (speech_key and speech_region)
            or (speech_auth_token and speech_region)
        ):
            raise ValueError(
                "AZURE_SPEECH_HOST or AZURE_SPEECH_KEY and AZURE_SPEECH_REGION or speech_auth_token and AZURE_SPEECH_REGION must be set"
            )

        if prosody:
            prosody.validate()

        if style:
            style.validate()

        self._opts = _TTSOptions(
            sample_rate=sample_rate,
            speech_key=speech_key,
            speech_region=speech_region,
            speech_host=speech_host,
            speech_auth_token=speech_auth_token,
            voice=voice,
            endpoint_id=endpoint_id,
            language=language,
            prosody=prosody,
            style=style,
            on_bookmark_reached_event=on_bookmark_reached_event,
            on_synthesis_canceled_event=on_synthesis_canceled_event,
            on_synthesis_completed_event=on_synthesis_completed_event,
            on_synthesis_started_event=on_synthesis_started_event,
            on_synthesizing_event=on_synthesizing_event,
            on_viseme_event=on_viseme_event,
            on_word_boundary_event=on_word_boundary_event,
        )

    def update_options(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        prosody: ProsodyConfig | None = None,
        style: StyleConfig | None = None,
    ) -> None:
        self._opts.voice = voice or self._opts.voice
        self._opts.language = language or self._opts.language
        self._opts.prosody = prosody or self._opts.prosody
        self._opts.style = style or self._opts.style

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self, input_text=text, conn_options=conn_options, opts=self._opts
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self):
        stream_callback = speechsdk.audio.PushAudioOutputStream(
            _PushAudioOutputStreamCallback(
                self._opts, asyncio.get_running_loop(), self._event_ch
            )
        )
        synthesizer = _create_speech_synthesizer(
            config=self._opts,
            stream=stream_callback,
        )

        def _synthesize() -> speechsdk.SpeechSynthesisResult:
            if self._opts.prosody or self._opts.style:
                ssml = (
                    '<speak version="1.0" '
                    'xmlns="http://www.w3.org/2001/10/synthesis" '
                    'xmlns:mstts="http://www.w3.org/2001/mstts" '
                    f'xml:lang="{self._opts.language or "en-US"}">'
                )
                ssml += f'<voice name="{self._opts.voice}">'

                # Add style if specified
                if self._opts.style:
                    style_degree = (
                        f' styledegree="{self._opts.style.degree}"'
                        if self._opts.style.degree
                        else ""
                    )
                    ssml += f'<mstts:express-as style="{self._opts.style.style}"{style_degree}>'

                # Add prosody if specified
                if self._opts.prosody:
                    ssml += "<prosody"
                    if self._opts.prosody.rate:
                        ssml += f' rate="{self._opts.prosody.rate}"'
                    if self._opts.prosody.volume:
                        ssml += f' volume="{self._opts.prosody.volume}"'
                    if self._opts.prosody.pitch:
                        ssml += f' pitch="{self._opts.prosody.pitch}"'
                    ssml += ">"
                    ssml += self._input_text
                    ssml += "</prosody>"
                else:
                    ssml += self._input_text

                # Close style tag if it was opened
                if self._opts.style:
                    ssml += "</mstts:express-as>"

                ssml += "</voice></speak>"
                return synthesizer.speak_ssml_async(ssml).get()  # type: ignore

            return synthesizer.speak_text_async(self.input_text).get()  # type: ignore

        result = None
        try:
            result = await asyncio.to_thread(_synthesize)
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                if (
                    result.cancellation_details.error_code
                    == speechsdk.CancellationErrorCode.ServiceTimeout
                ):
                    raise APITimeoutError()
                else:
                    cancel_details = result.cancellation_details
                    raise APIConnectionError(cancel_details.error_details)
        finally:

            def _cleanup() -> None:
                # cleanup resources inside an Executor
                # to avoid blocking the event loop
                nonlocal synthesizer, stream_callback, result
                del synthesizer
                del stream_callback

                if result is not None:
                    del result

            try:
                await asyncio.to_thread(_cleanup)
            except Exception:
                logger.exception("failed to cleanup Azure TTS resources")


class _PushAudioOutputStreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    def __init__(
        self,
        opts: _TTSOptions,
        loop: asyncio.AbstractEventLoop,
        event_ch: utils.aio.ChanSender[tts.SynthesizedAudio],
    ):
        super().__init__()
        self._event_ch = event_ch
        self._opts = opts
        self._loop = loop
        self._request_id = utils.shortuuid()

        self._bstream = utils.audio.AudioByteStream(
            sample_rate=opts.sample_rate, num_channels=1
        )

    def write(self, audio_buffer: memoryview) -> int:
        for frame in self._bstream.write(audio_buffer.tobytes()):
            audio = tts.SynthesizedAudio(
                request_id=self._request_id,
                frame=frame,
            )
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._event_ch.send_nowait, audio)

        return audio_buffer.nbytes

    def close(self) -> None:
        for frame in self._bstream.flush():
            audio = tts.SynthesizedAudio(
                request_id=self._request_id,
                frame=frame,
            )
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._event_ch.send_nowait, audio)


def _create_speech_synthesizer(
    *, config: _TTSOptions, stream: speechsdk.audio.AudioOutputStream
) -> speechsdk.SpeechSynthesizer:
    # let the SpeechConfig constructor to validate the arguments
    speech_config = speechsdk.SpeechConfig(
        subscription=config.speech_key,
        region=config.speech_region,
        endpoint=config.speech_endpoint,
        host=config.speech_host,
        auth_token=config.speech_auth_token,
        speech_recognition_language=config.language or "en-US",
    )

    speech_config.set_speech_synthesis_output_format(
        SUPPORTED_SAMPLE_RATE[config.sample_rate]
    )
    stream_config = speechsdk.audio.AudioOutputConfig(stream=stream)
    if config.voice is not None:
        speech_config.speech_synthesis_voice_name = config.voice
        if config.endpoint_id is not None:
            speech_config.endpoint_id = config.endpoint_id

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=stream_config
    )

    if config.on_bookmark_reached_event:
        synthesizer.bookmark_reached.connect(config.on_bookmark_reached_event)
    if config.on_synthesis_canceled_event:
        synthesizer.synthesis_canceled.connect(config.on_synthesis_canceled_event)
    if config.on_synthesis_completed_event:
        synthesizer.synthesis_completed.connect(config.on_synthesis_completed_event)
    if config.on_synthesis_started_event:
        synthesizer.synthesis_started.connect(config.on_synthesis_started_event)
    if config.on_synthesizing_event:
        synthesizer.synthesizing.connect(config.on_synthesizing_event)
    if config.on_viseme_event:
        synthesizer.viseme_received.connect(config.on_viseme_event)
    if config.on_word_boundary_event:
        synthesizer.synthesis_word_boundary.connect(config.on_word_boundary_event)

    return synthesizer
