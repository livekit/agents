import asyncio
from typing import Tuple

from .tts import (
    TTS,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
)

from .. import utils
from ..utils import aio
from .._exceptions import APIConnectionError, APIError, APITimeoutError, APIStatusError

from ..log import logger

from dataclasses import dataclass


@dataclass
class _TTSStatus:
    available: bool
    recovering_task: asyncio.Task | None


class FallbackAdapter(TTS):
    def __init__(
        self,
        *tts: TTS,
        timeout: float = 7.5,
        no_fallback_after_audio_duration: float = 3.0,
        sample_rate: int | None = None,
    ) -> None:
        streaming = all(t.capabilities.streaming for t in tts)

        if not sample_rate:
            sample_rate = max(t.sample_rate for t in tts)

        if len(set(t.num_channels for t in tts)) != 1:
            raise ValueError("all TTS must have the same number of channels")

        super().__init__(
            capabilities=TTSCapabilities(
                streaming=streaming,
            ),
            sample_rate=sample_rate,
            num_channels=tts[0].num_channels,
        )

        self._wrapped_tts = tts
        self._timeout = timeout
        self._no_fallback_after_audio_duration = no_fallback_after_audio_duration
        self._status = [_TTSStatus(True, None) for _ in tts]

    def synthesize(self, text: str) -> "FallbackChunkedStream":
        return FallbackChunkedStream(self, text)

    def stream(self) -> "FallbackSynthesizeStream":
        raise NotImplementedError("not implemented")

    async def aclose(self) -> None:
        pass


class FallbackChunkedStream(ChunkedStream):
    def __init__(self, tts: TTS, text: str) -> None:
        super().__init__(tts, text)


    async def _try_synthesize(self, tts: TTS, *, output_ch: aio.Chan[SynthesizedAudio] | None) -> float:
        assert isinstance(self._tts, FallbackAdapter)

        audio_duration = 0.0

        async def _synthesize(self) -> None:
            nonlocal audio_duration
            stream = tts.synthesize(self._input_text)
            try:
                while True:
                    try:
                        audio = await asyncio.wait_for(
                            stream.__anext__(),
                            self._tts._timeout if audio_duration == 0.0 else None,
                        )

                        audio_duration += audio.frame.duration
                        if output_ch is not None:
                            output_ch.send_nowait(audio)
                    except StopAsyncIteration:
                        return
            finally:
                await stream.aclose()

        synthesize_task = asyncio.create_task(_synthesize(self))

        try:
            await synthesize_task
        except asyncio.TimeoutError:
            logger.warning(
                f"tts.FallbackAdapter, {tts} timed out, switching to next TTS"
            ):
        except APIError:
            logger.warning(
                f"tts.FallbackAdapter, {tts} failed, switching to next TTS"
            )
        except Exception:
            logger.exception(
                f"tts.FallbackAdapter, {tts} unexpected error, switching to next TTS"
            )
        finally:
            await utils.aio.gracefully_cancel(synthesize_task)

        return audio_duration

        # if audio_duration >= self._tts._no_fallback_after_audio_duration:
        #     logger.warning(
        #         f"tts.FallbackAdapter, already synthesized {audio_duration}s of audio, ignoring fallback"
        #      )

    async def _main_task(self) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        for i, tts in enumerate(self._tts._wrapped_tts):
            tts_status =self._tts._status[i]
            if tts_status.available:
                audio_duration, ok = await self._try_synthesize(tts, output_ch=self._event_ch)
                
                if ok:
                    return # success



            else:
                async def _recover_task(self) -> None:
                    await self._try_synthesize(tts, output_ch=None)

                tts_status.recovering_task = asyncio.create_task(_recover_task(self))


            






class FallbackSynthesizeStream(SynthesizeStream):
    def __init__(self, tts: TTS) -> None:
        super().__init__(tts)
