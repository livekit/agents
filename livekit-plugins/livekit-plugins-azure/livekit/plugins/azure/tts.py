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
from dataclasses import dataclass
from typing import Literal

from livekit.agents import tts, utils

import azure.cognitiveservices.speech as speechsdk  # type: ignore

AZURE_SAMPLE_RATE: int = 16000
AZURE_BITS_PER_SAMPLE: int = 16
AZURE_NUM_CHANNELS: int = 1


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
class _TTSOptions:
    speech_key: str | None = None
    speech_region: str | None = None
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    voice: str | None = None
    # for using custom voices (see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis?tabs=browserjs%2Cterminal&pivots=programming-language-python#use-a-custom-endpoint)
    endpoint_id: str | None = None
    # Useful to specify the language with multi-language voices
    language: str | None = None
    # See https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice#adjust-prosody
    prosody: ProsodyConfig | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        prosody: ProsodyConfig | None = None,
        speech_key: str | None = None,
        speech_region: str | None = None,
        endpoint_id: str | None = None,
    ) -> None:
        """
        Create a new instance of Azure TTS.

        ``speech_key`` and ``speech_region`` must be set, either using arguments or by setting the
        ``AZURE_SPEECH_KEY`` and ``AZURE_SPEECH_REGION`` environmental variables, respectively.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=AZURE_SAMPLE_RATE,
            num_channels=AZURE_NUM_CHANNELS,
        )

        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        if not speech_key:
            raise ValueError("AZURE_SPEECH_KEY must be set")

        speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        if not speech_region:
            raise ValueError("AZURE_SPEECH_REGION must be set")

        if prosody:
            prosody.validate()

        self._opts = _TTSOptions(
            speech_key=speech_key,
            speech_region=speech_region,
            voice=voice,
            endpoint_id=endpoint_id,
            language=language,
            prosody=prosody,
        )

    def update_options(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        prosody: ProsodyConfig | None = None,
    ) -> None:
        self._opts.voice = voice or self._opts.voice
        self._opts.language = language or self._opts.language
        self._opts.prosody = prosody or self._opts.prosody

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(self, text, self._opts)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, tts: TTS, text: str, opts: _TTSOptions) -> None:
        super().__init__(tts)
        self._text, self._opts = text, opts

    @utils.log_exceptions()
    async def _main_task(self):
        stream_callback = speechsdk.audio.PushAudioOutputStream(
            _PushAudioOutputStreamCallback(asyncio.get_running_loop(), self._event_ch)
        )
        synthesizer = _create_speech_synthesizer(
            config=self._opts,
            stream=stream_callback,
        )

        def _synthesize() -> speechsdk.SpeechSynthesisResult:
            if self._opts.prosody:
                ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self._opts.language or "en-US"}">'
                prosody_ssml = f'<voice name="{self._opts.voice}"><prosody'
                if self._opts.prosody.rate:
                    prosody_ssml += f' rate="{self._opts.prosody.rate}"'
                if self._opts.prosody.volume:
                    prosody_ssml += f' volume="{self._opts.prosody.volume}"'
                if self._opts.prosody.pitch:
                    prosody_ssml += f' pitch="{self._opts.prosody.pitch}"'
                prosody_ssml += ">"
                ssml += prosody_ssml
                ssml += self._text
                ssml += "</prosody></voice></speak>"
                return synthesizer.speak_ssml_async(ssml).get()  # type: ignore

            return synthesizer.speak_text_async(self._text).get()  # type: ignore

        result = None
        try:
            result = await asyncio.to_thread(_synthesize)
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                if result.cancellation_details:
                    raise ValueError(
                        f"failed to synthesize audio: {result.reason}: {result.cancellation_details.reason} ({result.cancellation_details.error_details})"
                    )
                else:
                    raise ValueError(f"failed to synthesize audio: {result.reason}")
        finally:

            def _cleanup() -> None:
                # cleanup resources inside an Executor
                # to avoid blocking the event loop
                nonlocal synthesizer, stream_callback, result
                del synthesizer
                del stream_callback
                del result

            await asyncio.to_thread(_cleanup)


class _PushAudioOutputStreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        event_ch: utils.aio.ChanSender[tts.SynthesizedAudio],
    ):
        super().__init__()
        self._event_ch = event_ch
        self._loop = loop
        self._request_id = utils.shortuuid()
        self._segment_id = utils.shortuuid()

        self._bstream = utils.audio.AudioByteStream(
            sample_rate=AZURE_SAMPLE_RATE, num_channels=AZURE_NUM_CHANNELS
        )

    def write(self, audio_buffer: memoryview) -> int:
        for frame in self._bstream.write(audio_buffer.tobytes()):
            audio = tts.SynthesizedAudio(
                request_id=self._request_id,
                segment_id=self._segment_id,
                frame=frame,
            )
            self._loop.call_soon_threadsafe(self._event_ch.send_nowait, audio)

        return audio_buffer.nbytes

    def close(self) -> None:
        for frame in self._bstream.flush():
            audio = tts.SynthesizedAudio(
                request_id=self._request_id,
                segment_id=self._segment_id,
                frame=frame,
            )
            self._loop.call_soon_threadsafe(self._event_ch.send_nowait, audio)


def _create_speech_synthesizer(
    *, config: _TTSOptions, stream: speechsdk.audio.AudioOutputStream
) -> speechsdk.SpeechSynthesizer:
    speech_config = speechsdk.SpeechConfig(
        subscription=config.speech_key,
        region=config.speech_region,
        speech_recognition_language=config.language or "en-US",
    )
    stream_config = speechsdk.audio.AudioOutputConfig(stream=stream)
    if config.voice is not None:
        speech_config.speech_synthesis_voice_name = config.voice
        if config.endpoint_id is not None:
            speech_config.endpoint_id = config.endpoint_id

    return speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=stream_config
    )
