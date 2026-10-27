"""Live protocol checks for the same providers as the hermetic STT contracts.

Run with --stt. Missing credentials are reported as skips. Use -k to select a
provider, and --show-capture=no when recording results to keep provider logs private.
"""

import asyncio
import os
import re
import wave
from pathlib import Path

import pytest
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import APIConnectOptions, stt

from .stt_harness.discovery import providers

pytestmark = pytest.mark.stt


async def _send_utterance(stream, name):
    with wave.open(str(Path(__file__).parent / "test_realtime" / f"{name}.wav")) as wav:
        assert wav.getsampwidth() == 2 and wav.getnchannels() == 1
        rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    pcm += bytes(rate * 2 * 2)  # Two seconds of trailing silence for server endpointing.
    chunk_bytes = rate // 50 * 2
    loop = asyncio.get_running_loop()
    deadline = loop.time()
    for offset in range(0, len(pcm), chunk_bytes):
        data = pcm[offset : offset + chunk_bytes]
        stream.push_frame(rtc.AudioFrame(data, rate, 1, len(data) // 2))
        deadline += len(data) / (rate * 2)
        await asyncio.sleep(max(0, deadline - loop.time()))


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("case", providers(), ids=lambda case: case.name)
async def test_live_stream_flush_idle_and_completion(case):
    if not case.live_constructor:
        pytest.skip("live driver is not configured for this variant")
    load_dotenv()
    missing = [name for name in case.required_env if not os.environ.get(name)]
    if missing:
        pytest.skip("Missing live test configuration: " + ", ".join(missing))

    async with case.create_live() as provider:
        stream = provider.stream(
            conn_options=APIConnectOptions(
                max_retry=0,
                timeout=15,
                liveness_timeout=6,
            )
        )
        finals = asyncio.Queue()
        received = []

        async def collect():
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text
                    received.append(text)
                    finals.put_nowait(text)

        collector = asyncio.create_task(collect())
        try:
            for index, (name, expected) in enumerate(
                [
                    ("hello_world", {"hello", "world"}),
                    ("weather_question", {"weather", "paris"}),
                ]
            ):
                await _send_utterance(stream, name)
                stream.flush()

                async def wait_for_words(expected=expected):
                    words = set()
                    while not expected <= words:
                        text = await finals.get()
                        words.update(re.findall(r"[a-z]+", text.lower()))

                try:
                    await asyncio.wait_for(wait_for_words(), 20)
                except asyncio.TimeoutError:
                    if collector.done():
                        collector.result()
                    raise AssertionError(
                        f"No final transcript for {name}; received: {received!r}"
                    ) from None
                assert not collector.done(), "flush ended the public stream"
                if index == 0:
                    # No outbound audio for longer than the liveness window.
                    await asyncio.sleep(7)
                    assert not collector.done(), "idle keepalive failed"

            stream.end_input()
            # Consume through completion; stopping at the first final hides drain hangs.
            await asyncio.wait_for(collector, 17)
        except Exception:
            if collector.done() and not collector.cancelled():
                collector.result()
            raise
        finally:
            await stream.aclose()
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)
