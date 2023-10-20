import asyncio
import json
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
import livekit


class SpeechRecognition:
    def __init__(self):
        self._current_id = 1
        self._processing_id = -1
        self._google_json = json.loads(open("google.json", encoding='utf8').read())
        decoding = cloud_speech.ExplicitDecodingConfig(encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                                                       sample_rate_hertz=48000,
                                                       audio_channel_count=1
                                                       )
        recognition_features = cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True,)
        self._recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=decoding,
            features=recognition_features,
            language_codes=["en-US"],
            model="long")

        recognizer = f"projects/{self._google_json['project_id']}/locations/global/recognizers/_"
        self._streaming_config = cloud_speech.StreamingRecognitionConfig(config=self._recognition_config)
        self._config_request = cloud_speech.StreamingRecognizeRequest(recognizer=recognizer,
                                                                      streaming_config=self._streaming_config)
        self._result_queue = asyncio.Queue[cloud_speech.StreamingRecognizeResponse]()

    async def push_frames(self, frames: [livekit.AudioFrame]):
        client = SpeechAsyncClient.from_service_account_info(self._google_json)

        async def req_iterator():
            yield self._config_request
            for req in self.frames:
                yield cloud_speech.StreamingRecognizeRequest(audio=req.frame)
        
        return await client.streaming_recognize(requests=req_iterator())
