"""
Regression tests for Gemini Live usage metrics emission.

The Gemini Live API sends usageMetadata messages periodically, each carrying the
*cumulative* token counts for the session, and a final one around turn
completion. The realtime session used to emit a `metrics_collected` event for
every one of these messages, so a single generation was reported (and summed by
the usage collectors) multiple times.

These tests pin the intended behavior: usage is reported exactly once per
generation, with the final cumulative value seen for that generation.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.genai import types

from livekit.agents import llm
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.plugins.google.realtime.realtime_api import RealtimeSession


def _model_turn_message(text: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(parts=[types.Part(text=text)]),
        )
    )


def _usage_message(
    prompt_tokens: int, output_tokens: int, total_tokens: int
) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        usage_metadata=types.UsageMetadata(
            prompt_token_count=prompt_tokens,
            response_token_count=output_tokens,
            total_token_count=total_tokens,
        )
    )


def _turn_complete_message(
    usage: types.UsageMetadata | None = None,
) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(turn_complete=True),
        usage_metadata=usage,
    )


class _FakeAsyncSession:
    """Yields a scripted sequence of LiveServerMessages, then stops the recv loop."""

    def __init__(self, messages: list[types.LiveServerMessage], on_drained) -> None:
        self._messages = list(messages)
        self._on_drained = on_drained

    async def receive(self):
        for message in self._messages:
            yield message
        if self._on_drained:
            self._on_drained()


def _make_session(
    messages: list[types.LiveServerMessage],
) -> tuple[RealtimeSession, list[RealtimeModelMetrics]]:
    model = MagicMock()
    model.capabilities.audio_output = False
    model.label = "google-realtime-test"
    model.model = "gemini-test"
    model.provider = "google"

    session = object.__new__(RealtimeSession)
    llm.RealtimeSession.__init__(session, model)  # EventEmitter + _realtime_model

    session._opts = SimpleNamespace(tool_choice="auto", output_audio_transcription=None)
    session._chat_ctx = llm.ChatContext.empty()
    session._current_generation = None
    session._pending_generation_fut = None
    session._rejected_tool_calls = 0
    session._num_retries = 0
    session._session_error = None
    session._session_resumption_handle = None
    session._session_should_close = asyncio.Event()
    session._session_lock = asyncio.Lock()

    collected: list[RealtimeModelMetrics] = []
    session.on("metrics_collected", collected.append)

    session._active_session = _FakeAsyncSession(
        messages, on_drained=session._session_should_close.set
    )
    return session, collected


async def _run_recv_task(session: RealtimeSession) -> None:
    await RealtimeSession._recv_task(session, session._active_session)


async def test_usage_emitted_once_per_generation():
    """One turn with an incremental and a final usageMetadata: a single emission."""
    messages = [
        _model_turn_message("hello"),
        _usage_message(prompt_tokens=100, output_tokens=10, total_tokens=110),
        _usage_message(prompt_tokens=100, output_tokens=40, total_tokens=140),
        _turn_complete_message(),
    ]

    session, collected = _make_session(messages)
    await _run_recv_task(session)

    assert len(collected) == 1
    assert collected[0].input_tokens == 100
    assert collected[0].output_tokens == 40
    assert collected[0].total_tokens == 140


async def test_final_usage_bundled_with_turn_complete_is_reported():
    """The final usageMetadata may arrive on the turnComplete message itself."""
    messages = [
        _model_turn_message("hello"),
        _usage_message(prompt_tokens=100, output_tokens=10, total_tokens=110),
        _turn_complete_message(
            usage=types.UsageMetadata(
                prompt_token_count=100,
                response_token_count=40,
                total_token_count=140,
            )
        ),
    ]

    session, collected = _make_session(messages)
    await _run_recv_task(session)

    assert len(collected) == 1
    assert collected[0].output_tokens == 40
    assert collected[0].total_tokens == 140


async def test_usage_reported_once_per_turn_across_multiple_turns():
    messages = [
        _model_turn_message("hello"),
        _usage_message(prompt_tokens=100, output_tokens=10, total_tokens=110),
        _usage_message(prompt_tokens=100, output_tokens=40, total_tokens=140),
        _turn_complete_message(),
        _model_turn_message("bye"),
        _usage_message(prompt_tokens=150, output_tokens=70, total_tokens=220),
        _turn_complete_message(
            usage=types.UsageMetadata(
                prompt_token_count=150,
                response_token_count=75,
                total_token_count=225,
            )
        ),
    ]

    session, collected = _make_session(messages)
    await _run_recv_task(session)

    assert [metrics.output_tokens for metrics in collected] == [40, 75]
