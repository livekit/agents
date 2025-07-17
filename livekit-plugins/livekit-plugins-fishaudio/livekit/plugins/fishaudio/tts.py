import os
import uuid
from dataclasses import dataclass

from fish_audio_sdk import TTSRequest, WebSocketSession

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
)

FISHAUDIO_API_KEY = os.getenv("FISHAUDIO_API_KEY")
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
MIME_TYPE = "audio/wav"


@dataclass
class _TTSOptions:
    language: str
    reference_id: str | None = None
    temperature: float = 0.7
    top_p: float = 0.7


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        language: str = "en",
        reference_id: str = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            language=language, reference_id=reference_id, temperature=temperature, top_p=top_p
        )
        if not FISHAUDIO_API_KEY:
            raise APIConnectionError("FISHAUDIO_API_KEY not set")
        self._ws = WebSocketSession(FISHAUDIO_API_KEY)

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
            ws=self._ws,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        ws: WebSocketSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._ws = ws
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = str(uuid.uuid4().hex)[:12]
        try:
            tts_request = TTSRequest(
                text=self.input_text,
                reference_id=self._opts.reference_id,
                format="wav",
                temperature=self._opts.temperature,
                top_p=self._opts.top_p,
            )
            # fish-audio-sdk is sync, so run in thread
            import asyncio

            loop = asyncio.get_running_loop()

            def run_tts():
                return list(self._ws.tts(tts_request, [], backend="s1"))

            audio_chunks = await loop.run_in_executor(None, run_tts)
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type=MIME_TYPE,
            )
            for chunk in audio_chunks:
                output_emitter.push(chunk)
            output_emitter.flush()
        except Exception as e:
            raise APIConnectionError() from e
