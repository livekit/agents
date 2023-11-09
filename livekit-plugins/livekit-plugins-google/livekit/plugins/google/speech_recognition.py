import asyncio
import json
from typing import AsyncIterator
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from livekit import rtc
from livekit.plugins import core


class SpeechRecognition:
    def __init__(self, *, google_credentials_filepath: str):
        self._current_id = 1
        self._processing_id = -1
        self._google_json = json.loads(
            open(google_credentials_filepath, encoding='utf8').read())
        decoding = cloud_speech.ExplicitDecodingConfig(encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                                                       sample_rate_hertz=48000,
                                                       audio_channel_count=1
                                                       )
        recognition_features = cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,)
        self._recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=decoding,
            features=recognition_features,
            language_codes=["en-US"],
            model="long")

        recognizer = f"projects/{self._google_json['project_id']}/locations/global/recognizers/_"
        self._streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self._recognition_config)
        self._config_request = cloud_speech.StreamingRecognizeRequest(recognizer=recognizer,
                                                                      streaming_config=self._streaming_config)
        self._result_queue = asyncio.Queue[cloud_speech.StreamingRecognizeResponse](
        )
        self._result_iterator = core.AsyncQueueIterator(
            self._result_queue)

    def push_frames(self, frames: AsyncIterator[rtc.AudioFrame]) -> AsyncIterator[core.STTPluginResult]:
        client = SpeechAsyncClient.from_service_account_info(self._google_json)

        resp_queue = asyncio.Queue[core.STTPluginResult](
        )
        resp_iterator = core.AsyncQueueIterator(resp_queue)

        async def req_iterator():
            yield self._config_request
            for f in frames:
                resampled = f.remix_and_resample(48000, 1)
                yield cloud_speech.StreamingRecognizeRequest(audio=resampled.data.tobytes())

        async def resp_stream():
            res_generator = await client.streaming_recognize(requests=req_iterator())
            async for r in self.iterate_results(res_generator):
                await resp_queue.put(r)

        asyncio.create_task(resp_stream())

        return resp_iterator

    async def iterate_results(self, generator):
        async for res in generator:
            await self._result_queue.put(res.results[0].alternatives[0].transcript)


class SpeechRecognitionPlugin(core.STTPlugin):
    def __init__(self, *, google_credentials_filepath: str):
        self._speech_recognition = SpeechRecognition(
            google_credentials_filepath=google_credentials_filepath)

        super().__init__(process=self._speech_recognition.push_frames)
