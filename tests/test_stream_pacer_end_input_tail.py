from __future__ import annotations

import asyncio

import pytest

from livekit.agents import APIConnectOptions, tts, utils
from livekit.agents.tts import SentenceStreamPacer, StreamAdapter

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SAMPLE_RATE = 24000
REPLY = (
    "I'm doing well, thanks! How are you? Today we'll practice minimal "
    "pairs, words that differ by just one sound, like sheep and ship."
)


class FakeChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        synthesized.append(self._input_text)
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        await asyncio.sleep(0.6)
        seconds = len(self._input_text) * 0.08
        sent = 0.0
        while sent < seconds:
            output_emitter.push(b"\x00\x00" * int(SAMPLE_RATE * 0.1))
            sent += 0.1
            await asyncio.sleep(0.01)
        output_emitter.flush()


class FakeTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

    def synthesize(self, text, *, conn_options: APIConnectOptions | None = None):
        return FakeChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or APIConnectOptions(),
        )


synthesized: list[str] = []
pacers: list[object] = []


@pytest.fixture(autouse=True)
def _record_pacers(monkeypatch: pytest.MonkeyPatch):
    synthesized.clear()
    pacers.clear()
    orig = SentenceStreamPacer.wrap

    def recording_wrap(self, sent_stream, audio_emitter):
        wrapper = orig(self, sent_stream, audio_emitter)
        pacers.append(wrapper)
        return wrapper

    monkeypatch.setattr(SentenceStreamPacer, "wrap", recording_wrap)
    yield


@pytest.mark.parametrize("k", [-1, 0, 1])
async def test_stream_pacer_keeps_tail_across_end_input_wakeup_race(k: int) -> None:
    """Every sentence reaches the TTS when the pacer wakes up k loop turns around end_input."""
    adapter = StreamAdapter(
        tts=FakeTTS(),
        text_pacing=SentenceStreamPacer(min_remaining_audio=3.0, max_text_length=200),
    )
    stream = adapter.stream()

    async def produce() -> None:
        for i in range(0, len(REPLY), 4):
            stream.push_text(REPLY[i : i + 4])
            await asyncio.sleep(0.003)
        while not (
            pacers
            and synthesized
            and pacers[0]._wakeup_timer is not None
            and not pacers[0]._sentences
        ):
            await asyncio.sleep(0.003)
        pacer = pacers[0]
        # k=0 is the race: the wakeup runs before the tokenizer flushes its tail
        if k >= 0:
            stream.end_input()
            for _ in range(k):
                await asyncio.sleep(0)
            pacer._wakeup_event.set()
        else:
            pacer._wakeup_event.set()
            for _ in range(-k):
                await asyncio.sleep(0)
            stream.end_input()

    async def consume() -> None:
        async for _ in stream:
            pass

    await asyncio.gather(produce(), consume())
    await stream.aclose()
    assert len(synthesized) == 2
