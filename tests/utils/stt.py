from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence

from livekit import rtc
from livekit.agents import stt


async def collect_stream_events(
    stream: stt.RecognizeStream,
    frames: Sequence[rtc.AudioFrame],
    *,
    drain_timeout: float = 10.0,
) -> tuple[list[stt.SpeechEvent], float]:
    """Pace captured audio, then drain until closure or the post-input timeout."""
    events: list[stt.SpeechEvent] = []
    loop = asyncio.get_running_loop()
    audio_started_at = time.time()
    deadline = loop.time()

    async def send() -> None:
        nonlocal deadline
        for frame in frames:
            # A captured frame becomes available at the end of its duration.
            deadline += frame.duration
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            stream.push_frame(frame)
        stream.end_input()

    async def receive() -> None:
        async for event in stream:
            events.append(event)

    reader = asyncio.create_task(receive())
    sender = asyncio.create_task(send())
    try:
        await asyncio.wait((reader, sender), return_when=asyncio.FIRST_COMPLETED)
        if reader.done():
            reader.result()
        await sender
        # Some providers keep their socket open after end_input(). Earlier finals
        # or EOS cannot prove completion: another final or turn may still arrive.
        done, _ = await asyncio.wait((reader,), timeout=drain_timeout)
        if reader in done:
            reader.result()
    finally:
        tasks = (sender, reader)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return events, audio_started_at


def assert_speech_end_time(
    events: Sequence[stt.SpeechEvent],
    *,
    audio_started_at: float,
    expected_end: float,
    tolerance: float = 0.5,
    untimed_reason: str | None = None,
) -> float | None:
    """Check the last EOS on the input audio timeline, with the core's precedence."""
    endpoint = None
    for event in reversed(events):
        assert event.type != stt.SpeechEventType.START_OF_SPEECH, (
            "Stream ended with an open speech turn"
        )
        if event.type == stt.SpeechEventType.END_OF_SPEECH:
            endpoint = event
            break
    assert endpoint is not None, "No END_OF_SPEECH received"

    if endpoint.speech_end_time is not None:
        end_time = endpoint.speech_end_time - audio_started_at
    elif endpoint.alternatives and endpoint.alternatives[0].end_time > 0:
        end_time = endpoint.alternatives[0].end_time
    else:
        assert untimed_reason, "The last END_OF_SPEECH has no speech-end timestamp"
        print(f"EOS timing unavailable: {untimed_reason}")
        return None

    assert math.isfinite(end_time) and abs(end_time - expected_end) <= tolerance, (
        f"EOS reports {end_time:.3f}s; expected {expected_end:.3f}s ± {tolerance:.3f}s"
    )
    return end_time
