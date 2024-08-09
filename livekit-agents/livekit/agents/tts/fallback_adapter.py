from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .. import utils
from ..log import logger
from .tts import (
    TTS,
    ChunkedStream,
    SynthesizeStream,
    TTSCapabilities,
)


@dataclass
class TTSProvider:
    tts: TTS
    cooldown: float
    _active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    def deactivate(self) -> None:
        async def _wait_and_reactivate():
            await asyncio.sleep(self.cooldown)
            logger.info(f"provider {self.tts.__class__.__module__} reactivated")
            self._active = True

        self._active = False
        if self.cooldown > 0:
            asyncio.create_task(_wait_and_reactivate())


class FallbackAdapter(TTS):
    def __init__(
        self,
        *providers: TTS,
        connect_timeout: float = 5.0,
        keepalive_timeout: float = 120.0,
        cooldown: float = 60.0,
    ) -> None:
        if len(providers) == 0:
            raise ValueError("FallbackAdapter requires at least one provider")
        if not all(
            x.sample_rate == providers[0].sample_rate for x in providers
        ) or not all(x.num_channels == providers[0].num_channels for x in providers):
            raise ValueError(
                "all providers in a FallbackAdapter must have the same sample rate and channel count"
            )
        super().__init__(
            capabilities=TTSCapabilities(
                streaming=all(x.capabilities.streaming for x in providers),
            ),
            sample_rate=providers[0].sample_rate,
            num_channels=providers[0].num_channels,
        )
        self._providers = [TTSProvider(tts=x, cooldown=cooldown) for x in providers]
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
        if self.capabilities.streaming:
            return FallbackSynthesizeStream(
                providers=self._providers,
                connect_timeout=self._connect_timeout,
                keepalive_timeout=self._keepalive_timeout,
            )
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"
        )


class FallbackChunkedStream(ChunkedStream):
    def __init__(
        self,
        *,
        providers: list[TTSProvider],
        connect_timeout: float,
        keepalive_timeout: float,
        text: str,
    ) -> None:
        super().__init__()
        self._text = text
        self._connect_timeout = connect_timeout
        self._keepalive_timeout = keepalive_timeout
        self._providers = providers
        self.timeout: float
        self._init_tts()

    @utils.log_exceptions(logger=logger)
    def _init_tts(self) -> None:
        index = next((i for i, x in enumerate(self._providers) if x.active), None)
        if index is not None:
            self._index = index
            self._stream = self._providers[self._index].tts.synthesize(self._text)
            self._timeout = self._connect_timeout
        else:
            raise Exception("all providers failed")

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        while True:
            try:
                item = await asyncio.wait_for(self._stream.__anext__(), self._timeout)
                self._timeout = self._keepalive_timeout
                self._event_ch.send_nowait(item)
            except (TimeoutError, asyncio.TimeoutError):
                logger.warn(
                    f"provider {self._stream.__class__.__module__} failed, attempting to switch"
                )
                self._providers[self._index].deactivate()
                self._init_tts()
            except StopAsyncIteration:
                return


class FallbackSynthesizeStream(SynthesizeStream):
    def __init__(
        self,
        *,
        providers: list[TTSProvider],
        connect_timeout: float,
        keepalive_timeout: float,
    ) -> None:
        super().__init__()
        self._connect_timeout = connect_timeout
        self._keepalive_timeout = keepalive_timeout
        self._providers = providers
        self._timeout: float
        self._init_tts()

    @utils.log_exceptions(logger=logger)
    def _init_tts(self) -> None:
        index = next((i for i, x in enumerate(self._providers) if x.active), None)
        if index is not None:
            self._index = index
            self._stream = self._providers[self._index].tts.stream()
            self._timeout = self._connect_timeout
        else:
            raise Exception("all providers failed")

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        async def _pass_input():
            async for input in self._input_ch:
                self._stream.push_text(input)

        async def _pass_output():
            while True:
                try:
                    # TODO(nbsp): elevenlabs _stream._run_ws()#_recv_task() simply exits when it's
                    #             done, and the iterator hangs indefinitely until it times out.
                    #             probably needs a change to SynthesizeStream to make it work
                    item = await asyncio.wait_for(
                        self._stream.__anext__(), self._timeout
                    )
                    self._timeout = self._keepalive_timeout
                    self._event_ch.send_nowait(item)
                except (TimeoutError, asyncio.TimeoutError):
                    logger.warn(
                        f"provider {self._stream.__class__.__module__} failed, attempting to switch"
                    )
                    self._providers[self._index].deactivate()
                    self._init_tts()
                except StopAsyncIteration:
                    return

        tasks = [
            asyncio.create_task(_pass_input()),
            asyncio.create_task(_pass_output()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
