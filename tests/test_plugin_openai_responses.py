from __future__ import annotations

import json

import aiohttp
import pytest

from openai.types import Reasoning
from livekit.plugins.openai.responses.llm import LLMStream, _ResponsesWebsocket

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
