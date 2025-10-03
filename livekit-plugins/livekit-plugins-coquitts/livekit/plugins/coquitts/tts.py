from __future__ import annotations
from dataclasses import dataclass, replace

import httpx

from livekit.agents import (
    APIConnectOptions,
    tts,
    utils
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr
)

@dataclass
class _TTSOptions:
    api_endpoint: str
    sample_rate: int

class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_endpoint: str,
        sample_rate: int = 16000
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = _TTSOptions(
            api_endpoint=api_endpoint,
            sample_rate=sample_rate,
        )

    @property
    def api_endpoint(self) -> str:
        return self._opts.api_endpoint

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, text=text, conn_options=conn_options)

    def update_options(
        self,
        *,
        api_endpoint: NotGivenOr[str] = NOT_GIVEN
    ) -> None:
        if is_given(api_endpoint):
            self._opts.api_endpoint = api_endpoint


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: TTS, text: str, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._opts.api_endpoint,
                    headers={"Content-Type": "application/json"},
                    json={"input": self._input_text},
                )

                if response.status_code == 200 and "audio/mpeg" in response.headers.get("Content-Type", ""):
                    output_emitter.initialize(
                        request_id=utils.shortuuid(),
                        sample_rate=self._opts.sample_rate,
                        num_channels=1,
                        mime_type="audio/mp3",
                    )

                    async for chunk in response.aiter_bytes():
                        output_emitter.push(chunk)

        except Exception as e:
            print(e)
