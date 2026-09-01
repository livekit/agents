"""Unit tests for the Google TTS plugin's streaming synthesis."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from google.api_core.exceptions import DeadlineExceeded
from google.cloud import texttospeech

from livekit.agents import APIConnectOptions, APITimeoutError
from livekit.plugins.google import tts as google_tts

pytestmark = pytest.mark.plugin("google")

# 100ms of 24kHz 16-bit mono PCM per synthesized sentence
_AUDIO_CHUNK = b"\x01\x00" * 2400

SENTENCES = [
    "This is the first sentence of a long reply.",
    "The language model keeps producing more text.",
    "And the very last sentence closes the reply.",
]


class _FakeStreamingCall:
    """Stand-in for the bidi StreamingSynthesize call.

    Mirrors grpc.aio: the request iterator is consumed by a background task, ``timeout`` is a
    deadline for the whole call, and cancelling a read cancels the RPC (and its request poller).
    Replies with one audio chunk per text input, unless ``respond`` is False (stalled server).
    """

    def __init__(
        self,
        requests: AsyncIterator[texttospeech.StreamingSynthesizeRequest],
        *,
        timeout: float | None,
        respond: bool,
    ) -> None:
        self.timeout = timeout
        self.received: list[texttospeech.StreamingSynthesizeRequest] = []
        self._respond = respond
        self._deadline = None if timeout is None else time.monotonic() + timeout
        self._responses = asyncio.Queue[texttospeech.StreamingSynthesizeResponse | None]()
        self._poller = asyncio.create_task(self._consume_requests(requests))

    async def _consume_requests(
        self, requests: AsyncIterator[texttospeech.StreamingSynthesizeRequest]
    ) -> None:
        async for req in requests:
            self.received.append(req)
            if self._respond and req.input.text:
                self._responses.put_nowait(
                    texttospeech.StreamingSynthesizeResponse(audio_content=_AUDIO_CHUNK)
                )
        # client half-closed the stream, the server finishes it
        self._responses.put_nowait(None)

    def cancel(self) -> None:
        self._poller.cancel()

    def __aiter__(self) -> _FakeStreamingCall:
        return self

    async def __anext__(self) -> texttospeech.StreamingSynthesizeResponse:
        remaining = None if self._deadline is None else self._deadline - time.monotonic()
        try:
            resp = await asyncio.wait_for(self._responses.get(), remaining)
        except asyncio.TimeoutError:
            raise DeadlineExceeded("Deadline Exceeded") from None
        except asyncio.CancelledError:
            self.cancel()
            raise

        if resp is None:
            raise StopAsyncIteration
        return resp


class _FakeTTSClient:
    def __init__(self, *, respond: bool = True) -> None:
        self.calls: list[_FakeStreamingCall] = []
        self._respond = respond

    async def streaming_synthesize(
        self,
        requests: AsyncIterator[texttospeech.StreamingSynthesizeRequest],
        *,
        timeout: float | None = None,
        **_kwargs: object,
    ) -> _FakeStreamingCall:
        call = _FakeStreamingCall(requests, timeout=timeout, respond=self._respond)
        self.calls.append(call)
        return call


def _make_tts(client: _FakeTTSClient) -> google_tts.TTS:
    tts = google_tts.TTS()
    tts._client = client  # type: ignore[assignment]
    return tts


async def _collect_audio(stream: google_tts.SynthesizeStream) -> int:
    num_bytes = 0
    async for ev in stream:
        num_bytes += ev.frame.data.nbytes
    return num_bytes


async def test_streaming_not_cut_off_by_connect_timeout() -> None:
    # the LLM keeps producing text for longer than the connect timeout: the RPC must stay
    # open and every sentence must be synthesized
    client = _FakeTTSClient()
    tts = _make_tts(client)
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.2))

    async def _push() -> None:
        for sentence in SENTENCES:
            stream.push_text(sentence + " ")
            await asyncio.sleep(0.15)
        stream.end_input()

    push_task = asyncio.create_task(_push())
    try:
        num_bytes = await asyncio.wait_for(_collect_audio(stream), timeout=5.0)
    finally:
        await asyncio.gather(push_task, return_exceptions=True)
        await stream.aclose()

    assert len(client.calls) == 1
    texts = [req.input.text for req in client.calls[0].received if req.input.text]
    assert " ".join(texts).split() == " ".join(SENTENCES).split()
    assert num_bytes == len(SENTENCES) * len(_AUDIO_CHUNK)


async def test_streaming_first_response_timeout() -> None:
    # a server that never answers must still surface an APITimeoutError within the connect
    # timeout, even while the LLM is still producing text
    client = _FakeTTSClient(respond=False)
    tts = _make_tts(client)
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.2))
    stream.push_text(SENTENCES[0] + " " + SENTENCES[1] + " ")

    started_at = time.monotonic()
    try:
        with pytest.raises(APITimeoutError):
            await asyncio.wait_for(_collect_audio(stream), timeout=5.0)
    finally:
        await stream.aclose()

    assert time.monotonic() - started_at < 2.0
    assert len(client.calls) == 1
    # the first sentence was sent and the input was still open when the timeout fired
    assert any(req.input.text for req in client.calls[0].received)
