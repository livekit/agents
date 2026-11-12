from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import DeadlineExceeded
from google.cloud import texttospeech

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APITimeoutError, tts, utils
from livekit.plugins.google.tts import TTS

pytestmark = pytest.mark.plugin("google")


@pytest.mark.asyncio
async def test_input_gen_close_does_not_mask_api_error() -> None:
    """gRPC drives input_gen from its own task, so closing it must not replace the real error."""
    provider = TTS(credentials_info={"type": "service_account"})

    parked = asyncio.Event()
    consumer: asyncio.Task[None] | None = None

    async def fake_streaming_synthesize(input_gen, timeout=None, retry=None):  # noqa: ANN001
        nonlocal consumer

        async def consume() -> None:
            async for _ in input_gen:
                parked.set()

        consumer = asyncio.create_task(consume())
        await parked.wait()  # first request yielded; generator now awaits the next chunk
        await asyncio.sleep(0)
        raise DeadlineExceeded("deadline exceeded")

    client = MagicMock()
    client.streaming_synthesize = fake_streaming_synthesize
    provider._client = client

    stream = provider._stream(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    input_stream = utils.aio.Chan[str]()
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


@pytest.mark.asyncio
async def test_input_generator_failure_reaches_caller() -> None:
    provider = TTS(credentials_info={"type": "service_account"})

    class FailingInput:
        async def __aiter__(self):
            raise RuntimeError("input failed")
            yield

    async def fake_streaming_synthesize(input_gen, timeout=None, retry=None):
        async for _ in input_gen:
            pass

        async def empty():
            if False:
                yield texttospeech.StreamingSynthesizeResponse()

        return empty()

    client = MagicMock()
    client.streaming_synthesize = fake_streaming_synthesize
    provider._client = client
    stream = provider._stream(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    try:
        with pytest.raises(RuntimeError, match="input failed"):
            await stream._run_stream(
                FailingInput(), MagicMock(), texttospeech.StreamingSynthesizeConfig()
            )
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_each_chunk_is_one_request_and_flush_ends_the_segment() -> None:
    provider = TTS(credentials_info={"type": "service_account"})
    segments: list[list[str]] = []

    async def fake_streaming_synthesize(input_gen, timeout=None, retry=None):  # noqa: ANN001
        texts = [req.input.text async for req in input_gen if req.input.text]
        segments.append(texts)

        async def respond():
            yield texttospeech.StreamingSynthesizeResponse(audio_content=b"\x00\x00" * 240)

        return respond()

    client = MagicMock()
    client.streaming_synthesize = fake_streaming_synthesize
    provider._client = client

    stream = provider._stream(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    for text in ("Hello there.", "How are you?"):
        stream._push_chunk(tts.SpeechChunk(text=text, said=text))
    stream.flush()
    stream._push_chunk(tts.SpeechChunk(text="Bye.", said="Bye."))
    stream.end_input()
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()

    assert segments == [["Hello there.", "How are you?"], ["Bye."]]
