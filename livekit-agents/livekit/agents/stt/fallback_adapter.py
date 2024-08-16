from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .. import utils
from ..log import logger
from .stt import (
    STT,
    SpeechEvent,
    SpeechStream,
    STTCapabilities,
)


@dataclass
class STTProvider:
    stt: STT
    cooldown: float
    _active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    def deactivate(self) -> None:
        async def _wait_and_reactivate():
            await asyncio.sleep(self.cooldown)
            logger.info(f"provider {self.stt.__class__.__module__} reactivated")
            self._active = True

        self._active = False
        if self.cooldown > 0:
            asyncio.create_task(_wait_and_reactivate())


class FallbackAdapter(STT):
    def __init__(
        self,
        *providers: STT,
        connect_timeout: float = 5.0,
        keepalive_timeout: float = 120.0,
        cooldown: float = 60.0,
    ) -> None:
        if len(providers) == 0:
            raise ValueError("FallbackAdapter requires at least one provider")
        super().__init__(
            capabilities=STTCapabilities(
                streaming=all(x.capabilities.streaming for x in providers),
                interim_results=all(x.capabilities.interim_results for x in providers),
            )
        )
        self._providers = [STTProvider(stt=x, cooldown=cooldown) for x in providers]
        self._connect_timeout = connect_timeout
        self._keepalive_timeout = keepalive_timeout

    async def recognize(
        self, buffer: utils.AudioBuffer, *, language: str | None = None
    ) -> SpeechEvent:
        def _init_stt():
            index = next((i for i, x in enumerate(self._providers) if x.active), None)
            if isinstance(index, int):
                return index
            else:
                raise Exception("all providers failed")

        i = _init_stt()
        while True:
            try:
                return await asyncio.wait_for(
                    self._providers[i].stt.recognize(buffer, language=language),
                    self._connect_timeout,
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.warn(
                    f"provider {self._providers[i].__class__.__module__} failed, attempting to switch"
                )
                self._providers[i].deactivate()
                i = _init_stt()

    def stream(self, *, language: str | None = None) -> SpeechStream:
        if self.capabilities.streaming:
            return FallbackSpeechStream(
                providers=self._providers,
                connect_timeout=self._connect_timeout,
                keepalive_timeout=self._keepalive_timeout,
                language=language,
            )
        raise NotImplementedError(
            "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"
        )


class FallbackSpeechStream(SpeechStream):
    def __init__(
        self,
        *,
        providers: list[STTProvider],
        connect_timeout: float,
        keepalive_timeout: float,
        language: str | None,
    ) -> None:
        super().__init__()
        self._connect_timeout = connect_timeout
        self._keepalive_timeout = keepalive_timeout
        self._providers = providers
        self._language = language
        self._timeout: float
        self._init_stt()

    @utils.log_exceptions(logger=logger)
    def _init_stt(self) -> None:
        index = next((i for i, x in enumerate(self._providers) if x.active), None)
        if index is not None:
            self._index = index
            self._stream = self._providers[self._index].stt.stream(
                language=self._language
            )
            self._timeout = self._connect_timeout
        else:
            raise Exception("all providers failed")

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        async def _pass_input():
            async for input in self._input_ch:
                self._stream.push_frame(input)

        async def _pass_output():
            while True:
                try:
                    async for item in self._stream:
                        self._timeout = self._keepalive_timeout
                        self._event_ch.send_nowait(item)
                except Exception:
                    logger.warn(
                        f"provider {self._stream.__class__.__module__} failed, attempting to switch"
                    )
                    self._providers[self._index].deactivate()
                    self._init_stt()

        tasks = [
            asyncio.create_task(_pass_input()),
            asyncio.create_task(_pass_output()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
