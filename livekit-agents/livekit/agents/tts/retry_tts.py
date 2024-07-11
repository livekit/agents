from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from .tts import TTS, ChunkedStream, SynthesisEvent, SynthesizedAudio, SynthesizeStream

RetryCallback = Callable[[int], float]


@dataclass
class RetryPolicy:
    max_retries: int
    delay: float | RetryCallback

    @staticmethod
    def exponential_backoff(
        *, initial_delay: float, max_delay: float, max_retries: int
    ) -> RetryPolicy:
        def cb(attempts: int) -> float:
            return min(max_delay, initial_delay * 2**attempts)

        return RetryPolicy(max_retries=max_retries, delay=cb)


class RetryTTS(TTS):
    def __init__(
        self,
        source_tts: TTS,
        first_packet_timeout: float,
        retry_policy: RetryPolicy = RetryPolicy.exponential_backoff(
            initial_delay=0.5, max_delay=5, max_retries=3
        ),
    ) -> None:
        super().__init__(
            streaming_supported=source_tts.streaming_supported,
            sample_rate=source_tts.sample_rate,
            num_channels=source_tts.num_channels,
        )
        self._first_packet_timeout = first_packet_timeout
        self._source_tts = source_tts
        self._retry_policy = retry_policy

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def synthesize(self, text: str) -> ChunkedStream:
        return RetryChunkedStream(
            source_tts=self._source_tts,
            first_packet_timeout=self._first_packet_timeout,
            text=text,
            retry_policy=RetryPolicy(max_retries=3, delay=0.5),
        )

    def stream(self) -> SynthesizeStream:
        return RetrySynthesizeStream(
            source_tts=self._source_tts,
            first_packet_timeout=self._first_packet_timeout,
            retry_policy=self._retry_policy,
        )


class RetryChunkedStream(ChunkedStream):
    def __init__(
        self,
        source_tts: TTS,
        text: str,
        first_packet_timeout: float,
        retry_policy: RetryPolicy,
    ) -> None:
        super().__init__()
        self._text = text
        self._first_packet_timeout = first_packet_timeout
        self._source_tts = source_tts
        self._source_cs: ChunkedStream | None
        self._retry_policy = retry_policy
        self._attempt = 0
        self._run()

    def _run(self) -> None:
        self._source_cs = self._source_tts.synthesize(self._text)

    async def aclose(self) -> None:
        assert self._source_cs is not None
        await self._source_cs.aclose()

    async def __anext__(self) -> SynthesizedAudio:
        sent_first_audio = False
        while True:
            try:
                assert self._source_cs is not None
                if not sent_first_audio:
                    item = await asyncio.wait_for(
                        self._source_cs.__anext__(), self._first_packet_timeout
                    )
                else:
                    item = await self._source_cs.__anext__()
                if not sent_first_audio:
                    sent_first_audio = True
                return item
            except Exception as e:
                # raise if we have already sent the first audio (or max retries)
                if sent_first_audio or self._attempt == self._retry_policy.max_retries:
                    raise e

                delay = self._retry_policy.delay
                if callable(delay):
                    delay = delay(self._retry_policy.max_retries)

                await asyncio.sleep(delay)
                self._attempt += 1
                self._run()


class RetrySynthesizeStream(SynthesizeStream):
    def __init__(
        self,
        first_packet_timeout: float,
        source_tts: TTS,
        retry_policy: RetryPolicy,
    ) -> None:
        super().__init__()
        self._first_packet_timeout = first_packet_timeout
        self._source_tts = source_tts
        self._source_ss: SynthesizeStream | None
        self._retry_policy = retry_policy
        self._attempt = 0
        self._total_text: list[str | None] = []
        self._run()

    def _run(self) -> None:
        self._source_ss = self._source_tts.stream()
        for token in self._total_text:
            self._source_ss.push_text(token)

    def push_text(self, token: str | None) -> None:
        assert self._source_ss is not None
        self._total_text.append(token)
        self._source_ss.push_text(token)

    async def aclose(self, *, wait: bool = True) -> None:
        assert self._source_ss is not None
        await self._source_ss.aclose(wait=wait)

    async def __anext__(self) -> SynthesisEvent:
        sent_first_event = False
        while True:
            try:
                assert self._source_ss is not None
                if not sent_first_event:
                    item = await asyncio.wait_for(
                        self._source_ss.__anext__(), self._first_packet_timeout
                    )
                else:
                    item = await self._source_ss.__anext__()
                if not sent_first_event:
                    sent_first_event = True
                return item
            except Exception as e:
                # raise if we have already sent the first audio (or max retries)
                if sent_first_event or self._attempt == self._retry_policy.max_retries:
                    raise e

                delay = self._retry_policy.delay
                if callable(delay):
                    delay = delay(self._retry_policy.max_retries)

                await asyncio.sleep(delay)
                self._attempt += 1
                self._run()
