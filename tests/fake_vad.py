from __future__ import annotations

import asyncio
import time

from fake_stt import FakeUserSpeech

from livekit.agents.vad import VAD, VADCapabilities, VADEvent, VADEventType, VADStream


class FakeVAD(VAD):
    def __init__(
        self,
        *,
        fake_user_speeches: list[FakeUserSpeech] | None = None,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
    ) -> None:
        super().__init__(capabilities=VADCapabilities(update_interval=0.1))

        if fake_user_speeches is not None:
            fake_user_speeches = sorted(fake_user_speeches, key=lambda x: x.start_time)
            for prev, next in zip(fake_user_speeches[:-1], fake_user_speeches[1:]):
                if prev.end_time > next.start_time:
                    raise ValueError("fake user speeches overlap")
        self._fake_user_speeches = fake_user_speeches
        self._min_speech_duration = min_speech_duration
        self._min_silence_duration = min_silence_duration

    def stream(self) -> VADStream:
        return FakeVADStream(self)


class FakeVADStream(VADStream):
    def __init__(self, vad: FakeVAD) -> None:
        super().__init__(vad)

    async def _main_task(self) -> None:
        assert isinstance(self._vad, FakeVAD)

        if not self._vad._fake_user_speeches:
            return

        await self._input_ch.recv()
        start_time = time.time()

        def current_time() -> float:
            return time.time() - start_time

        for fake_speech in self._vad._fake_user_speeches:
            next_start_of_speech_time = fake_speech.start_time + self._vad._min_speech_duration
            next_end_of_speech_time = fake_speech.end_time + self._vad._min_silence_duration

            if current_time() < next_start_of_speech_time:
                await asyncio.sleep(next_start_of_speech_time - current_time())

            self._send_vad_event(VADEventType.START_OF_SPEECH, fake_speech, current_time())

            inference_interval = 0.05  # 20fps
            while current_time() < next_end_of_speech_time - inference_interval * 2:
                await asyncio.sleep(inference_interval)
                self._send_vad_event(VADEventType.INFERENCE_DONE, fake_speech, current_time())

            await asyncio.sleep(next_end_of_speech_time - current_time())
            self._send_vad_event(VADEventType.END_OF_SPEECH, fake_speech, current_time())

        async for _ in self._input_ch:
            # wait for the input to be ended
            pass

    def _send_vad_event(
        self, type: VADEventType, fake_speech: FakeUserSpeech, curr_time: float
    ) -> None:
        self._event_ch.send_nowait(
            VADEvent(
                type=type,
                samples_index=0,
                timestamp=curr_time,
                speech_duration=min(curr_time, fake_speech.end_time) - fake_speech.start_time,
                silence_duration=max(0.0, curr_time - fake_speech.end_time),
            )
        )
