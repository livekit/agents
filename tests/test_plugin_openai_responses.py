from __future__ import annotations

import json

import aiohttp
import pytest
from openai.types import Reasoning
from openai.types.responses import (
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)

from livekit.agents import APIStatusError
from livekit.plugins.openai.responses.llm import (
    LLMStream,
    _looks_like_tool_scaffolding,
    _model_may_leak_scaffolding,
    _ResponsesWebsocket,
)

pytestmark = pytest.mark.plugin("openai")


class _FakeWSMsg:
    def __init__(self, data: str) -> None:
        self.type = aiohttp.WSMsgType.TEXT
        self.data = data
        self.extra = None


class _RecordingWS:
    """Minimal aiohttp-websocket stand-in: records what was sent, then replays
    a single terminal frame so `generate_response` returns."""

    def __init__(self, reply: dict) -> None:
        self.sent: str | None = None
        self._reply = reply

    async def send_str(self, data: str) -> None:
        self.sent = data

    async def receive(self) -> _FakeWSMsg:
        return _FakeWSMsg(json.dumps(self._reply))


class _FakePool:
    def __init__(self, ws: _RecordingWS) -> None:
        self._ws = ws

    def connection(self, timeout: float | None = None):  # noqa: ANN201
        ws = self._ws

        class _Ctx:
            async def __aenter__(self):  # noqa: ANN202
                return ws

            async def __aexit__(self, *exc):  # noqa: ANN002, ANN202
                return False

        return _Ctx()

    async def aclose(self) -> None:
        return None


async def _capture_sent_payload(msg: dict) -> dict:
    """Run the real `_ResponsesWebsocket.generate_response` against a fake pool
    and return the JSON that was actually put on the wire."""
    ws = _ResponsesWebsocket(api_key="test-key", timeout=1.0)
    rec = _RecordingWS({"type": "response.completed", "response": {"output": []}})
    ws._pool = _FakePool(rec)  # type: ignore[assignment]
    async for _ in ws.generate_response(msg):
        pass
    assert rec.sent is not None
    return json.loads(rec.sent)


async def test_reasoning_object_serialized_without_null_fields() -> None:
    """The WS transport serializes request models itself (not via the openai
    SDK). It must drop unset/None fields, otherwise Optional fields default to
    an explicit `null` on the wire and the Responses API 400s — e.g. after
    openai-python added `Reasoning.mode` (default None), `Reasoning(effort=...)`
    began emitting `"mode": null`, rejected with 'expected one of standard or
    pro, but got null instead.' Regression guard for that class of bug."""
    payload = {
        "type": "response.create",
        "model": "gpt-5.4",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "reasoning": Reasoning(effort="none"),
    }

    sent = await _capture_sent_payload(payload)

    assert sent["reasoning"] == {"effort": "none"}
    # No serialized request model may carry an explicit null-valued key.
    assert None not in sent["reasoning"].values()


def test_error_event_missing_sequence_number_parses_cleanly() -> None:
    """Top-level protocol error frames (e.g. a request-validation 400) don't
    carry `sequence_number`, which ResponseErrorEvent marks required. Parsing
    must not raise a pydantic ValidationError that masks the real API message —
    it should surface the message so it reaches the caller as an APIStatusError."""
    frame = {
        "type": "error",
        "message": "Invalid type for 'reasoning.mode': expected one of "
        "'standard' or 'pro', but got null instead.",
        "code": "invalid_type",
        "param": "reasoning.mode",
        "status": 400,
    }

    # `_parse_ws_event` does not read `self`; invoke it directly on the frame.
    parsed = LLMStream._parse_ws_event(object(), frame)  # type: ignore[arg-type]

    assert parsed is not None
    assert parsed.type == "error"
    assert parsed.message == frame["message"]
    assert parsed.param == "reasoning.mode"


# --- internal tool-call scaffolding leak guard --------------------------------
# The gpt-5.4 series intermittently emits its internal `multi_tool_use.parallel`
# tool-call format as assistant TEXT (interleaved with training-data spam) instead
# of as structured function calls. The guard detects that signature at the start of
# the assistant text -- before any is streamed -- and raises a retryable error so
# the stream is re-generated, rather than surfacing corrupt text to the caller.
_TOOL_USES = {
    "tool_uses": [
        {
            "recipient_name": "functions.do_thing",
            "parameters": {"id": "aaaa0000bbbb1111", "action": "skipped"},
        },
        {"recipient_name": "functions.read_thing", "parameters": {"id": "cccc2222dddd3333"}},
    ]
}
_LEAK_TEXT = "to=multi_tool_use.parallel ,最新高清无码专区" + json.dumps(_TOOL_USES) + "Two. Text."
_CLEAN_LONG = "This is an ordinary assistant reply that is comfortably longer than the head window."
_CLEAN_SHORT = "Yes."
_ITEM_ID = "msg_abc"


class _CollectingCh:
    def __init__(self) -> None:
        self.chunks: list = []

    def send_nowait(self, chunk) -> None:
        self.chunks.append(chunk)


def _make_stream(model: str = "gpt-5.4") -> LLMStream:
    s = LLMStream.__new__(LLMStream)
    s._model = model
    s._response_id = "resp_123"
    s._text_head = {}
    s._text_verdict = {}
    s._emitted_content = False
    s._scaffold_guard = _model_may_leak_scaffolding(str(model))
    s._pending_tool_calls = set()
    s._event_ch = _CollectingCh()
    return s


def _delta(piece: str, seq: int) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent.model_construct(
        item_id=_ITEM_ID,
        output_index=0,
        content_index=0,
        delta=piece,
        sequence_number=seq,
        type="response.output_text.delta",
        logprobs=[],
    )


def _item_done(full_text: str, phase: str | None = "final_answer") -> ResponseOutputItemDoneEvent:
    msg = ResponseOutputMessage.model_construct(
        id=_ITEM_ID,
        role="assistant",
        status="completed",
        type="message",
        phase=phase,
        content=[
            ResponseOutputText.model_construct(text=full_text, type="output_text", annotations=[])
        ],
    )
    return ResponseOutputItemDoneEvent.model_construct(
        item=msg, output_index=0, sequence_number=98, type="response.output_item.done"
    )


def _drive(stream: LLMStream, text: str) -> str:
    """Feed `text` as token-sized deltas + the item.done event through the real
    `_process_event` dispatch; return the content that reached the event channel.
    Propagates the guard's error if it fires."""
    for i in range(0, len(text), 6):
        stream._process_event(_delta(text[i : i + 6], i))
    stream._process_event(_item_done(text))
    return "".join(
        c.delta.content for c in stream._event_ch.chunks if c.delta is not None and c.delta.content
    )


def test_scaffolding_helpers() -> None:
    assert _looks_like_tool_scaffolding(_LEAK_TEXT)
    assert not _looks_like_tool_scaffolding(_CLEAN_LONG)
    # scope: only the gpt-5.4 family is guarded
    assert _model_may_leak_scaffolding("gpt-5.4")
    assert _model_may_leak_scaffolding("gpt-5.4-codex")
    assert not _model_may_leak_scaffolding("gpt-4.1")
    assert not _model_may_leak_scaffolding("gpt-5.0")
    assert not _model_may_leak_scaffolding("o3")


def test_scaffolding_leak_raises_retryable_before_emitting() -> None:
    stream = _make_stream()
    with pytest.raises(APIStatusError) as ei:
        _drive(stream, _LEAK_TEXT)
    assert ei.value.retryable is True  # nothing streamed yet -> safe to re-generate
    # none of the corrupt text reached the event channel
    for chunk in stream._event_ch.chunks:
        assert not (chunk.delta and chunk.delta.content)
    assert stream._emitted_content is False


def test_scaffolding_leak_after_content_raises_non_retryable() -> None:
    """If assistant text was already streamed this response, a later leak can't be
    retried without double-emitting, so it fails the turn instead."""
    stream = _make_stream()
    stream._emitted_content = True  # simulate a clean chunk already delivered
    with pytest.raises(APIStatusError) as ei:
        stream._handle_response_output_text_delta(_delta("to=multi_tool_use.parallel {", 0))
    assert ei.value.retryable is False


def test_clean_long_text_streams_unchanged() -> None:
    stream = _make_stream()
    assert _drive(stream, _CLEAN_LONG) == _CLEAN_LONG


def test_clean_short_text_flushed_at_item_done() -> None:
    stream = _make_stream()
    assert _drive(stream, _CLEAN_SHORT) == _CLEAN_SHORT


def test_unaffected_model_streams_leak_verbatim() -> None:
    """On a non-5.4 model the text path is byte-for-byte the original: no buffering,
    no detection, no raise -- even for the leak input."""
    stream = _make_stream(model="gpt-4.1")
    assert _drive(stream, _LEAK_TEXT) == _LEAK_TEXT
