from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import DeadlineExceeded
from google.cloud import texttospeech

from livekit.agents import APITimeoutError, tokenize
from livekit.plugins.google.tts import TTS

pytestmark = pytest.mark.plugin("google")


@pytest.mark.asyncio
async def test_input_gen_close_does_not_mask_api_error() -> None:
    """gRPC drives input_gen from its own task, so closing it must not replace the real error."""
    tts = TTS(credentials_info={"type": "service_account"})

    parked = asyncio.Event()
    consumer: asyncio.Task[None] | None = None

    async def fake_streaming_synthesize(input_gen, timeout=None):  # noqa: ANN001
        nonlocal consumer

        async def consume() -> None:
            async for _ in input_gen:
                parked.set()

        consumer = asyncio.create_task(consume())
        await parked.wait()  # first request yielded; generator now awaits the next sentence
        await asyncio.sleep(0)
        raise DeadlineExceeded("deadline exceeded")

    client = MagicMock()
    client.streaming_synthesize = fake_streaming_synthesize
    tts._client = client

    stream = tts.stream()
    input_stream = tokenize.basic.SentenceTokenizer().stream()
    try:
        with pytest.raises(APITimeoutError):
            await stream._run_stream(
                input_stream,
                MagicMock(),
                texttospeech.StreamingSynthesizeConfig(),
            )
    finally:
        if consumer is not None:
            consumer.cancel()
        await stream.aclose()
