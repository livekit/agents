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

from livekit.agents import tts, utils

import azure.cognitiveservices.speech as speechsdk  # type: ignore

AZURE_SAMPLE_RATE: int = 16000
AZURE_BITS_PER_SAMPLE: int = 16
AZURE_NUM_CHANNELS: int = 1


@dataclass
class _TTSOptions:
    speech_key: str | None = None
    speech_region: str | None = None
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    voice: str | None = None
    # for using custom voices (see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis?tabs=browserjs%2Cterminal&pivots=programming-language-python#use-a-custom-endpoint)
    endpoint_id: str | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        speech_key: str | None = None,
        speech_region: str | None = None,
        voice: str | None = None,
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

        self._opts = _TTSOptions(
            speech_key=speech_key,
            speech_region=speech_region,
            voice=voice,
            endpoint_id=endpoint_id,
        )

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(text, self._opts)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, text: str, opts: _TTSOptions) -> None:
        super().__init__()
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
        subscription=config.speech_key, region=config.speech_region
    )
    stream_config = speechsdk.audio.AudioOutputConfig(stream=stream)
    if config.voice is not None:
        speech_config.speech_synthesis_voice_name = config.voice
        if config.endpoint_id is not None:
            speech_config.endpoint_id = config.endpoint_id

    return speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=stream_config
    )
