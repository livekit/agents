"""Tests for ``_CloudTransport`` (cloud WS body, driven by the unified
``_BaseStreamingTurnDetectorStream`` stream).

Uses the in-process ``FakeTurnDetectorWS`` to drive the transport
deterministically. Covers:

- Retry counter resets after a successful connect (so transient drops across
  the session lifetime don't accumulate toward ``max_retry``).
- All outbound messages are FIFO-ordered on the wire, even when control
  hooks fire synchronously between two awaited audio frames.
"""

from __future__ import annotations

import pytest

from livekit import rtc
from livekit.agents import llm
from livekit.agents._exceptions import APIConnectionError
from livekit.protocol.agent_pb.agent_session import ChatRole

from .fake_turn_detector_ws import (
    drain_send_queue,
    make_stream,
    wait_until_connected,
)

pytestmark = pytest.mark.audio_eot


def _pcm_frame(samples: int = 320) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=samples,
    )


class TestCloudStreamRetry:
    """``_num_retries`` lifecycle across reconnects."""

    async def test_num_retries_resets_after_successful_connect(self) -> None:
        """Regression: after a successful WS connect, transient-failure
        counters must reset so transient drops across the session lifetime
        don't accumulate toward ``max_retry``."""
        stream, _fake_ws, transport = make_stream(
            connect_script=[APIConnectionError("transient", retryable=True), None],
            max_retry=3,
            retry_interval=0.0,
        )
        try:
            await wait_until_connected(transport)
            # Two attempts total: first raised (counter went 0 → 1), second
            # succeeded and must have reset to 0.
            assert transport._connect_calls == 2
            assert transport._num_retries == 0
        finally:
            await stream.aclose()


class TestCloudStreamSendOrdering:
    """FIFO delivery for the unified outbound channel."""

    async def test_inference_start_precedes_input_audio(self) -> None:
        """Regression: ``run_inference`` (sync hook) used to schedule its
        send via ``asyncio.create_task``, which could land on the wire after
        an awaited ``input_audio`` send. With the unified channel, the
        sender drains FIFO so ``inference_start`` always reaches the wire
        first."""
        stream, fake_ws, transport = make_stream(connect_script=[None])
        try:
            await wait_until_connected(transport)
            stream.predict()
            stream.push_audio(_pcm_frame())
            await drain_send_queue(transport)

            kinds = [m.WhichOneof("message") for m in fake_ws.sent]
            inference_start_idx = kinds.index("inference_start")
            input_audio_idx = kinds.index("input_audio")
            assert inference_start_idx < input_audio_idx
        finally:
            await stream.aclose()

    async def test_consecutive_inference_starts_serialized(self) -> None:
        """Regression: two sync ``run_inference`` hooks back-to-back (a
        predict superseding another) used to race at the ``ws.send_bytes``
        await because each ran in its own task. The unified channel serializes
        them in call order."""
        stream, fake_ws, transport = make_stream(connect_script=[None])
        try:
            await wait_until_connected(transport)
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="assistant", content="First question")
            stream.predict(chat_ctx)
            first_id = stream._request_id
            chat_ctx.add_message(role="assistant", content="Second question")
            stream.predict(chat_ctx)
            second_id = stream._request_id
            await drain_send_queue(transport)

            start_ids = [
                m.inference_start.request_id
                for m in fake_ws.sent
                if m.WhichOneof("message") == "inference_start"
            ]
            assert start_ids == [first_id, second_id]
            assert [m.WhichOneof("message") for m in fake_ws.sent] == [
                "eot_input_chat_context",
                "inference_start",
                "eot_input_chat_context",
                "inference_start",
            ]
            assert [
                m.eot_input_chat_context.messages[0].content[0].text
                for m in fake_ws.sent
                if m.WhichOneof("message") == "eot_input_chat_context"
            ] == ["First question", "Second question"]
        finally:
            await stream.aclose()


class TestCloudStreamChatContext:
    async def test_sends_only_last_assistant_text(self) -> None:
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="System instructions")
        chat_ctx.add_message(role="assistant", content="Earlier question")
        chat_ctx.add_message(role="user", content="Earlier answer")
        chat_ctx.add_message(
            role="assistant", content=["Which city?", "<expr value='curious'/>どちらですか？"]
        )
        chat_ctx.items.append(llm.FunctionCall(call_id="call", name="lookup", arguments="{}"))
        chat_ctx.items.append(
            llm.FunctionCallOutput(
                call_id="call", name="lookup", output="Tool result", is_error=False
            )
        )
        chat_ctx.add_message(role="user", content="Current answer")

        stream, fake_ws, transport = make_stream(connect_script=[None])
        try:
            await wait_until_connected(transport)
            stream.predict(chat_ctx)
            await drain_send_queue(transport)

            assert [m.WhichOneof("message") for m in fake_ws.sent] == [
                "eot_input_chat_context",
                "inference_start",
            ]
            messages = fake_ws.sent[0].eot_input_chat_context.messages
            assert len(messages) == 1
            assert messages[0].role == ChatRole.ASSISTANT
            assert [part.text for part in messages[0].content] == ["Which city?\nどちらですか？"]
            assert len(chat_ctx.items) == 7
        finally:
            await stream.aclose()

    @pytest.mark.parametrize(
        "chat_ctx",
        [
            None,
            llm.ChatContext(),
            llm.ChatContext([llm.ChatMessage(role="user", content=["Hello"])]),
            llm.ChatContext(
                [
                    llm.ChatMessage(role="assistant", content=["Earlier question"]),
                    llm.ChatMessage(role="assistant", content=[]),
                ]
            ),
        ],
        ids=["omitted", "empty", "no-assistant", "empty-last-assistant"],
    )
    async def test_clears_previous_assistant_text(self, chat_ctx: llm.ChatContext | None) -> None:
        stream, fake_ws, transport = make_stream(connect_script=[None])
        try:
            await wait_until_connected(transport)
            stream.predict(
                llm.ChatContext([llm.ChatMessage(role="assistant", content=["Old question"])])
            )
            stream.predict(chat_ctx)
            await drain_send_queue(transport)

            assert fake_ws.sent[-2].WhichOneof("message") == "eot_input_chat_context"
            assert list(fake_ws.sent[-2].eot_input_chat_context.messages) == []
            assert fake_ws.sent[-1].WhichOneof("message") == "inference_start"
        finally:
            await stream.aclose()
