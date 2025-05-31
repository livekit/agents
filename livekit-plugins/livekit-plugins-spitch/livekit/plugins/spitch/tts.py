import uuid
from dataclasses import dataclass

import httpx

import spitch
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from spitch import AsyncSpitch

SAMPLE_RATE = 24_000
NUM_CHANNELS = 1
MIME_TYPE = "audio/wav"


@dataclass
class _TTSOptions:
    language: str
    voice: str


class TTS(tts.TTS):
    def __init__(self, language: str, voice: str):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24_000, num_channels=1
        )

        self._opts = _TTSOptions(language=language, voice=voice)
        self._client = AsyncSpitch()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            client=self._client,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        client: AsyncSpitch,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = client
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        spitch_stream = self._client.speech.with_streaming_response.generate(
            text=self.input_text,
            language=self._opts.language,
            voice=self._opts.voice,
            timeout=httpx.Timeout(30, connect=self._conn_options.timeout),
        )

        request_id = str(uuid.uuid4().hex)[:12]
        try:
            async with spitch_stream as stream:
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type=MIME_TYPE,
                )

                async for data in stream.iter_bytes():
                    output_emitter.push(data)

            output_emitter.flush()

        except spitch.APITimeoutError:
            raise APITimeoutError() from None
        except spitch.APIStatusError as e:
            raise APIStatusError(
                e.message, status_code=e.status_code, request_id=request_id, body=e.body
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
