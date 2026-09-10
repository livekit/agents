"""Tests for the ThunderPhone realtime plugin: URL/config shaping and call events."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from livekit.agents import utils
from livekit.plugins.openai.realtime import realtime_model as _openai, utils as _openai_utils
from livekit.plugins.thunderphone import RealtimeModel
from livekit.plugins.thunderphone.realtime.realtime_model import (
    DEFAULT_BASE_URL,
    SERVER_MODEL,
    RealtimeSession,
)

pytestmark = pytest.mark.unit

KEY = "sk_live_test"


# --------------------------------------------------------------- RealtimeModel


def test_model_requires_thunderphone_secret_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="sk_live_"):
            RealtimeModel(agent_id=1)
        with pytest.raises(ValueError, match="sk_live_"):
            RealtimeModel(agent_id=1, api_key="sk_test_nope")


def test_model_reads_key_from_env() -> None:
    with patch.dict("os.environ", {"THUNDERPHONE_API_KEY": KEY}):
        model = RealtimeModel(agent_id=1)
    assert model.agent_id == "1"
    assert model.provider == "ThunderPhone"


def test_agent_id_excludes_inline_options() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        RealtimeModel(api_key=KEY, agent_id=1, product="bolt")
    with pytest.raises(ValueError, match="agent_id"):
        RealtimeModel(api_key=KEY, agent_id=1, voice="olivia")


def test_saved_agent_url_pins_audio_and_call_events() -> None:
    model = RealtimeModel(api_key=KEY, agent_id=12, from_number="+15550001", to_number="+15550002")
    parts = urlsplit(model._opts.base_url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == DEFAULT_BASE_URL
    assert parse_qs(parts.query) == {
        "agent_id": ["12"],
        "from_number": ["+15550001"],
        "to_number": ["+15550002"],
        "input_audio_format": ["pcm16"],
        "input_rate": ["24000"],
        "output_audio_format": ["pcm16"],
        "output_rate": ["24000"],
        "call_events": ["1"],
    }


def test_inline_url_carries_product_and_language_and_keeps_custom_query() -> None:
    model = RealtimeModel(
        api_key=KEY,
        product="bolt",
        voice="olivia",
        language="es",
        base_url="ws://localhost:8002/v1/realtime?debug=1",
    )
    parts = urlsplit(model._opts.base_url)
    assert parts.netloc == "localhost:8002"
    query = parse_qs(parts.query)
    assert query["product"] == ["bolt"]
    assert query["language"] == ["es"]
    assert query["debug"] == ["1"]
    assert "agent_id" not in query
    assert model._opts.voice == "olivia"
    assert model.model == SERVER_MODEL


def test_model_freezes_instructions_and_tools_and_never_reconnects() -> None:
    model = RealtimeModel(api_key=KEY, agent_id=1)
    assert model.capabilities.mutable_instructions is False
    assert model.capabilities.mutable_tools is False
    assert model.capabilities.turn_detection is True
    assert model.capabilities.can_disable_turn_detection is False
    assert model._opts.conn_options.max_retry == 0
    # one socket is one call: never recycled, even if asked
    assert model._opts.max_session_duration is None
    recycled = RealtimeModel(api_key=KEY, agent_id=1, max_session_duration=60)
    assert recycled._opts.max_session_duration is None


async def test_sessions_are_registered_for_option_updates() -> None:
    model = RealtimeModel(api_key=KEY, agent_id=1)
    session = model.session()
    try:
        assert session in model._sessions
    finally:
        await session.aclose()


# ------------------------------------------------------------- RealtimeSession


def _make_session(*, agent_mode: bool, live_transcripts: bool = False) -> RealtimeSession:
    """A session with the plugin's state but no socket, like the upstream tests."""
    session = RealtimeSession.__new__(RealtimeSession)
    utils.EventEmitter.__init__(session)
    model = Mock()
    model.agent_id = "12" if agent_mode else None
    model._tp_live_transcripts = live_transcripts
    session._tp_model = model
    session._tp_agent_mode = agent_mode
    session._tp_holding = not agent_mode
    session._tp_pending_session = {}
    session._tp_call_ended = False
    session.call_id = None
    session._response_created_futures = {}
    session._tp_call_ids = {}
    session._msg_ch = utils.aio.Chan()
    session._sent = []
    return session


@pytest.fixture
def capture_upstream_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _openai.RealtimeSession,
        "send_event",
        lambda self, event: self._sent.append(event),
    )


def _framework_session_update() -> dict:
    """What livekit-plugins-openai sends on connect, before shaping."""
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["audio"],
            "instructions": "You are a receptionist.",
            "tools": [{"type": "function", "name": "book"}],
            "tool_choice": "auto",
            "max_output_tokens": "inf",
            "tracing": None,
            "truncation": "auto",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "marin",
                    "speed": 1.0,
                },
            },
        },
    }


async def test_agent_mode_update_keeps_only_audio_formats(capture_upstream_send: None) -> None:
    session = _make_session(agent_mode=True)

    session.send_event(_framework_session_update())

    assert len(session._sent) == 1
    shaped = session._sent[0]["session"]
    assert shaped == {
        "type": "realtime",
        "model": SERVER_MODEL,
        "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": 24000}},
            "output": {"format": {"type": "audio/pcm", "rate": 24000}},
        },
    }


async def test_inline_update_is_held_until_flush_then_opts_into_call_events(
    capture_upstream_send: None,
) -> None:
    session = _make_session(agent_mode=False, live_transcripts=True)

    session.send_event(_framework_session_update())
    session.send_event(
        {"type": "session.update", "session": {"tools": [{"type": "function", "name": "cancel"}]}}
    )
    assert session._sent == []  # held: the first update starts the call

    session._tp_flush()

    assert len(session._sent) == 1
    shaped = session._sent[0]["session"]
    assert shaped["model"] == SERVER_MODEL
    assert shaped["instructions"] == "You are a receptionist."
    assert shaped["tools"] == [{"type": "function", "name": "cancel"}]  # later update wins
    assert shaped["config"] == {"call_events": True}
    assert shaped["live_transcripts"] is True
    for key in ("tracing", "truncation", "max_output_tokens"):
        assert key not in shaped
    assert shaped["audio"]["output"]["voice"] == "marin"


async def test_inline_non_audio_event_flushes_configuration_first(
    capture_upstream_send: None,
) -> None:
    session = _make_session(agent_mode=False)
    session.send_event(_framework_session_update())

    session.send_event({"type": "response.create", "event_id": "r1"})

    assert [e["type"] for e in session._sent] == ["session.update", "response.create"]
    assert session._tp_holding is False


async def test_inline_audio_passes_through_while_holding(capture_upstream_send: None) -> None:
    session = _make_session(agent_mode=False)
    session.send_event(_framework_session_update())

    session.send_event({"type": "input_audio_buffer.append", "audio": "AAAA"})

    assert [e["type"] for e in session._sent] == ["input_audio_buffer.append"]
    assert session._tp_holding is True


async def test_unlabeled_response_is_credited_to_the_pending_reply() -> None:
    session = _make_session(agent_mode=False)
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    session._response_created_futures["response_create_x"] = fut
    event = {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}

    session._tp_on_server_event(event)

    assert event["response"]["metadata"] == {"client_event_id": "response_create_x"}


async def test_labeled_or_unrequested_responses_are_left_alone() -> None:
    session = _make_session(agent_mode=False)
    session._response_created_futures["response_create_x"] = (
        asyncio.get_running_loop().create_future()
    )
    labeled = {
        "type": "response.created",
        "response": {"id": "resp_1", "metadata": {"client_event_id": "response_create_y"}},
    }
    session._tp_on_server_event(labeled)
    assert labeled["response"]["metadata"] == {"client_event_id": "response_create_y"}

    session._response_created_futures.clear()
    unrequested = {"type": "response.created", "response": {"id": "resp_2"}}
    session._tp_on_server_event(unrequested)
    assert "metadata" not in unrequested["response"]


async def test_call_id_is_read_from_session_events() -> None:
    session = _make_session(agent_mode=True)
    session._tp_on_server_event({"type": "session.created", "session": {"call_id": "987"}})
    assert session.call_id == 987
    session._tp_on_server_event({"type": "session.updated", "session": {"call_id": 988}})
    assert session.call_id == 988


async def test_call_events_are_emitted_and_call_ended_closes_the_session() -> None:
    session = _make_session(agent_mode=True)
    seen: list[tuple[str, dict]] = []
    session.on("thunderphone_call_event", lambda ev: seen.append(("event", ev)))
    session.on("thunderphone_call_ended", lambda ev: seen.append(("ended", ev)))

    session._tp_on_server_event({"type": "call.transfer", "target": "+15550003"})
    assert seen == [("event", {"type": "call.transfer", "target": "+15550003"})]
    assert not session._msg_ch.closed

    ended = {"type": "call.ended", "reason": "agent_hangup"}
    session._tp_on_server_event(ended)
    session._tp_on_server_event(ended)  # a duplicate must not re-emit
    assert seen[1:] == [("event", ended), ("ended", ended), ("event", ended)]
    assert session._msg_ch.closed


async def test_long_function_call_ids_are_restored_on_the_output(
    capture_upstream_send: None,
) -> None:
    session = _make_session(agent_mode=True)
    long_id = "call_" + "a" * 32  # over OpenAI's 32-character cap
    session._tp_on_server_event(
        {
            "type": "response.output_item.done",
            "item": {
                "id": "item_1",
                "type": "function_call",
                "call_id": long_id,
                "name": "book",
                "arguments": "{}",
            },
        }
    )
    hashed = _openai_utils._shorten_call_id(long_id)
    assert hashed != long_id

    session.send_event(
        {
            "type": "conversation.item.create",
            "item": {
                "id": "item_2",
                "type": "function_call_output",
                "call_id": hashed,
                "output": "{}",
            },
        }
    )
    session.send_event(
        {
            "type": "conversation.item.create",
            "item": {
                "id": "item_3",
                "type": "function_call_output",
                "call_id": "call_short",
                "output": "{}",
            },
        }
    )

    assert session._sent[0]["item"]["call_id"] == long_id
    assert session._sent[1]["item"]["call_id"] == "call_short"
