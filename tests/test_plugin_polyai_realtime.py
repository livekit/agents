"""Tests for the PolyAI realtime plugin: defaults, session and tool shaping, reply options."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents import llm, utils
from livekit.plugins.openai.realtime import realtime_model as _openai
from livekit.plugins.polyai import KnowledgeTopic, RealtimeModel
from livekit.plugins.polyai.realtime.realtime_model import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RealtimeSession,
)

pytestmark = [pytest.mark.unit]

KEY = "poly-test-key"


# --------------------------------------------------------------- RealtimeModel


def test_model_requires_a_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="POLY_API_KEY"):
            RealtimeModel()


def test_model_defaults() -> None:
    with patch.dict("os.environ", {"POLY_API_KEY": KEY}, clear=True):
        model = RealtimeModel()
    assert model._opts.api_key == KEY
    assert model._opts.base_url == DEFAULT_BASE_URL
    assert model.model == DEFAULT_MODEL
    assert model.provider == "PolyAI"
    assert model._opts.modalities == ["audio"]
    assert model.capabilities.per_response_tool_choice is False


def test_model_reads_regional_url_from_env() -> None:
    url = "wss://api.us.poly.ai/v1/realtime"
    with patch.dict("os.environ", {"POLY_API_KEY": KEY, "DIALOG_REALTIME_URL": url}, clear=True):
        model = RealtimeModel()
    assert model._opts.base_url == url


def test_interrupt_response_false_is_rejected() -> None:
    from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

    with pytest.raises(ValueError, match="interrupt_response"):
        RealtimeModel(
            api_key=KEY, turn_detection=ServerVad(type="server_vad", interrupt_response=False)
        )


# ------------------------------------------------------------- RealtimeSession


def _make_session(model: RealtimeModel) -> RealtimeSession:
    """A session with the plugin's state but no socket."""
    session = RealtimeSession.__new__(RealtimeSession)
    utils.EventEmitter.__init__(session)
    session._realtime_model = model
    session._opts = model._opts
    session._instructions = "You are Ivy."
    session._response_created_futures = {}
    session._sent: list[Any] = []  # type: ignore[misc]
    return session


@pytest.fixture
def capture_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _openai.RealtimeSession, "send_event", lambda self, event: self._sent.append(event)
    )


def test_session_update_carries_poly_fields() -> None:
    model = RealtimeModel(
        api_key=KEY,
        knowledge=[
            KnowledgeTopic(name="Hours", content="9 to 5"),
            {"name": "Returns", "content": "30 days", "actions": "Offer a label."},
        ],
        response_language="es",
        first_message="Hello, this is Ivy.",
    )
    event = _make_session(model)._create_session_update_event()
    body = event["session"]
    assert event["type"] == "session.update"
    assert body["model"] == DEFAULT_MODEL
    assert body["instructions"] == "You are Ivy."
    assert body["poly_knowledge"] == [
        {"name": "Hours", "content": "9 to 5"},
        {"name": "Returns", "content": "30 days", "actions": "Offer a label."},
    ]
    assert body["poly_response_language"] == "es"
    assert body["poly_first_assistant_message"] == "Hello, this is Ivy."


def test_voice_is_left_unset_by_default() -> None:
    body = _make_session(RealtimeModel(api_key=KEY))._create_session_update_event()["session"]
    assert "voice" not in body["audio"]["output"]


def test_voice_is_sent_when_given() -> None:
    model = RealtimeModel(api_key=KEY, voice="cartesia-voice-id")
    body = _make_session(model)._create_session_update_event()["session"]
    assert body["audio"]["output"]["voice"] == "cartesia-voice-id"


def test_tools_carry_poly_speech_and_poly_spoken() -> None:
    @llm.function_tool
    async def look_up_order(order_number: str, hold_line: str) -> str:
        """Look up an order.

        Args:
            order_number: The order number.
            hold_line: What to say while looking.
        """
        return "shipped"

    @llm.function_tool
    async def log_call(note: str) -> None:
        """Log the call.

        Args:
            note: A note.
        """

    model = RealtimeModel(
        api_key=KEY,
        tool_speech={"log_call": "none"},
        spoken_arguments={"look_up_order": "hold_line"},
    )
    event = _make_session(model)._create_tools_update_event([look_up_order, log_call])
    tools = {t["name"]: t for t in event["session"]["tools"]}
    props = tools["look_up_order"]["parameters"]["properties"]
    assert props["hold_line"]["poly_spoken"] is True
    assert "poly_spoken" not in props["order_number"]
    assert tools["log_call"]["poly_speech"] == "none"
    assert "poly_speech" not in tools["look_up_order"]


@pytest.mark.usefixtures("capture_send")
async def test_generate_reply_drops_per_reply_tools() -> None:
    session = _make_session(RealtimeModel(api_key=KEY))
    fut = session.generate_reply(instructions="Be brief.", tool_choice="none")
    fut.cancel()
    (event,) = session._sent
    response = event.response.model_dump(exclude_none=True)
    assert response["instructions"] == "You are Ivy.\nBe brief."
    assert "tool_choice" not in response
    assert "tools" not in response


def test_response_done_reaches_the_callback() -> None:
    seen: list[dict[str, Any]] = []
    session = _make_session(RealtimeModel(api_key=KEY, on_response_done=seen.append))
    session._on_polyai_server_event({"type": "response.created", "response": {}})
    session._on_polyai_server_event(
        {
            "type": "response.done",
            "response": {"status": "completed", "poly_cited_topics": ["Hours"]},
        }
    )
    assert seen == [{"status": "completed", "poly_cited_topics": ["Hours"]}]
