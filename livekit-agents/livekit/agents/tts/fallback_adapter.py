from __future__ import annotations

import asyncio

from .. import utils
from .tts import (
    TTS,
    ChunkedStream,
    SynthesizeStream,
)


class FallbackAdapter(TTS):
    def __init__(
        self,
        *providers: TTS,
        connect_timeout: float,
        keepalive_timeout: float,
    ) -> None:
        assert len(providers) > 0
        # TODO(nbsp): figure out sample rate switching
        super().__init__(
            capabilities=providers[0].capabilities,
            sample_rate=providers[0].sample_rate,
            num_channels=providers[0].num_channels,
        )
        self._providers = providers
        self._connect_timeout = connect_timeout
        self._keepalive_timeout = keepalive_timeout

    def synthesize(self, text: str) -> ChunkedStream:
        return FallbackChunkedStream(
            providers=self._providers,
            connect_timeout=self._connect_timeout,
            keepalive_timeout=self._keepalive_timeout,
            text=text,
        )

    def stream(self) -> SynthesizeStream:
        return FallbackSynthesizeStream(
            providers=self._providers,
            connect_timeout=self._connect_timeout,
            keepalive_timeout=self._keepalive_timeout,
        )

class FallbackChunkedStream(ChunkedStream):
    def __init__(
        self,
        *,
        providers: list[TTS],
        connect_timeout: float,
        keepalive_timeout: float,
        text: str,
    ) -> None:
        super().__init__()
        self._text = text
        self._connect_timeout = connect_timeout
        self._keepalive_timeout = keepalive_timeout
        self._providers = providers

    async def _main_task(self) -> None:
        for provider in self._providers:
            stream = provider.synthesize(self._text)
            timeout = self._connect_timeout
            while True:
                try:
                    item = await asyncio.wait_for(
                        stream.__anext__(), timeout
                    )
                    timeout = self._keepalive_timeout
                    self._event_ch.send_nowait(item)
                except Exception as e:
                    if isinstance(e, asyncio.TimeoutError):
                        break
                    return
        raise Exception("all providers failed")

class FallbackSynthesizeStream(SynthesizeStream):
    def __init__(
        self,
        *,
        providers: list[TTS],
        connect_timeout: float,
        keepalive_timeout: float,
    ) -> None:
        super().__init__()
        self._connect_timeout = connect_timeout
        self._keepalive_timeout = keepalive_timeout
        self._providers = providers
        self._timeout: float
        self._current_provider = 0
        self._init_tts()

    def _init_tts(self) -> None:
        self._stream = self._providers[self._current_provider].stream()
        self._timeout = self._connect_timeout

    async def _main_task(self) -> None:
        async def _pass_input():
            async for input in self._input_ch:
                self._stream.push_text(input)

        async def _pass_output():
            while True:
                try:
                    # TODO(nbsp): this errors at end of transcription, TimeoutError swallows CancelledError
                    item = await asyncio.wait_for(
                        self._stream.__anext__(), self._timeout
                    )
                    self._timeout = self._keepalive_timeout
                    self._event_ch.send_nowait(item)
                except Exception as e:
                    if isinstance(e, asyncio.TimeoutError):
                        if self._current_provider + 1 < len(self._providers):
                            self._current_provider += 1
                            self._init_tts()
                        else:
                            raise Exception("all providers failed")
                    else:
                        return

        tasks = [
            asyncio.create_task(_pass_input()),
            asyncio.create_task(_pass_output()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
