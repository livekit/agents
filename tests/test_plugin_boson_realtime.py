import asyncio
import base64
import contextlib
import gc
import json
import logging
import time

import aiohttp
import pytest
from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemDeletedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from livekit.agents import APIError, llm, utils
from livekit.agents.types import APIConnectOptions
from livekit.plugins.boson import realtime

pytestmark = pytest.mark.unit

_URL = "ws://localhost:8000/v1/realtime/"
_DEFAULT_TURN_DETECTION = realtime.realtime_model._DEFAULT_TURN_DETECTION


async def _idle_run(_session):
    await asyncio.Future()


async def lookup_policy() -> str:
    return "ok"


class _FakeWebSocket:
    """Delivers any scripted `messages`, then waits for close_event and closes."""

    def __init__(self, close_message=None, messages=()) -> None:
        self.messages = list(messages)
        self.closed = False
        self.sent: list[str] = []
        self.close_event = asyncio.Event()
        self.close_message = close_message
        self.close_code = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.close_event.wait()
        raise StopAsyncIteration

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        await self.close_event.wait()
        if self.close_message is not None:
            self.close_code = self.close_message.data
            return self.close_message
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def exception(self):
        return None


class _SequencedHTTPClient:
    """Returns one scripted outcome (websocket or exception) per ws_connect call."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.connect_urls: list[str] = []

    async def ws_connect(self, url, **_kwargs):
        self.connect_urls.append(url)
        if not self.outcomes:
            raise aiohttp.ClientConnectionError("no more scripted connections")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ws_text(event: dict) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), None)


def _ws_close(code: int, reason: str) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, code, reason)


def _dropped_ws(messages=(), close_message=None) -> _FakeWebSocket:
    """A socket that delivers `messages` and then closes on its own."""
    ws = _FakeWebSocket(close_message, messages)
    ws.close_event.set()
    return ws


def _sent_response_create(ws: _FakeWebSocket) -> bool:
    return any('"type": "response.create"' in sent for sent in ws.sent)


async def _eventually(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _model(**kwargs) -> realtime.RealtimeModel:
    return realtime.RealtimeModel(**{"url": _URL, "api_key": "test-key", **kwargs})


async def _close_all(sessions: list) -> None:
    for session in reversed(sessions):
        await session.aclose()
        await session.realtime_model.aclose()


@pytest.fixture
async def open_session(monkeypatch):
    """Open a session with no socket behind it; events are injected with _server_event.

    By default the initial session.update is drained so the queue starts empty.
    """
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)
    opened: list = []

    async def _open(*, drain: bool = True, **kwargs):
        session = _model(**kwargs).session()
        opened.append(session)
        if drain:
            await session._msg_ch.recv()
        return session

    yield _open
    await _close_all(opened)


@pytest.fixture
async def socket_session():
    """Open a session that really runs its connection loop against a fake client."""
    opened: list = []

    def _open(client, *, max_retry: int = 3, **kwargs):
        conn_options = APIConnectOptions(max_retry=max_retry, retry_interval=0.01, timeout=1.0)
        session = _model(http_session=client, conn_options=conn_options, **kwargs).session()
        opened.append(session)
        return session

    yield _open
    await _close_all(opened)


def _collect(session, event: str) -> list:
    received: list = []
    session.on(event, received.append)
    return received


def _capture_warnings(monkeypatch) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(
        realtime.realtime_model.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg))
    )
    return warnings


def _reported_item_ids(caplog) -> list[str]:
    return [item_id for record in caplog.records for item_id in getattr(record, "item_ids", [])]


_ABSENT = object()
_TEXT_PART = {"type": "text", "text": ""}


def _at(payload: dict, path: str):
    """Look up a dotted path, or _ABSENT if any step is missing."""
    for key in path.split("."):
        if key not in payload:
            return _ABSENT
        payload = payload[key]
    return payload


def _ctx(*items) -> llm.ChatContext:
    """A ChatContext from (id, role, content) triples; a str content is one text part."""
    return llm.ChatContext(
        [
            llm.ChatMessage(id=item_id, role=role, content=[c] if isinstance(c, str) else c)
            for item_id, role, c in items
        ]
    )


# type -> (RealtimeSession handler, event model), mirroring the dispatch in the
# OpenAI base recv loop that the plugin inherits.
_SERVER_EVENT_HANDLERS = {
    "input_audio_buffer.speech_started": (
        "_handle_input_audio_buffer_speech_started",
        InputAudioBufferSpeechStartedEvent,
    ),
    "conversation.item.added": ("_handle_conversion_item_added", ConversationItemAdded),
    "conversation.item.deleted": ("_handle_conversion_item_deleted", ConversationItemDeletedEvent),
    "conversation.item.input_audio_transcription.completed": (
        "_handle_conversion_item_input_audio_transcription_completed",
        ConversationItemInputAudioTranscriptionCompletedEvent,
    ),
    "response.created": ("_handle_response_created", ResponseCreatedEvent),
    "response.output_item.added": (
        "_handle_response_output_item_added",
        ResponseOutputItemAddedEvent,
    ),
    "response.content_part.added": (
        "_handle_response_content_part_added",
        ResponseContentPartAddedEvent,
    ),
    "response.output_text.delta": ("_handle_response_text_delta", ResponseTextDeltaEvent),
    "response.output_text.done": ("_handle_response_text_done", ResponseTextDoneEvent),
    "response.output_audio.delta": ("_handle_response_audio_delta", ResponseAudioDeltaEvent),
    "response.output_item.done": ("_handle_response_output_item_done", ResponseOutputItemDoneEvent),
    "response.done": ("_handle_response_done", ResponseDoneEvent),
    "error": ("_handle_error", RealtimeErrorEvent),
}


def _server_event(session, *events: dict) -> None:
    """Inject server events as if they had been received on the socket."""
    for event in events:
        session.emit("openai_server_event_received", event)
        entry = _SERVER_EVENT_HANDLERS.get(event["type"])
        if entry is not None:
            method, event_cls = entry
            getattr(session, method)(event_cls.construct(**event))


def _item(item_id: str, item_type: str = "message", **fields) -> dict:
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": item_type,
        "status": "completed",
        **fields,
    }


def _item_added_echo(item: dict, previous_item_id: str | None = None) -> dict:
    """The server's conversation.item.added echo for a client-created item."""
    return {"type": "conversation.item.added", "previous_item_id": previous_item_id, "item": item}


def _user_item_added(item_id: str, content: list[dict]) -> dict:
    return _item_added_echo(_item(item_id, role="user", content=content))


def _user_text(item_id: str, text: str) -> dict:
    return _user_item_added(item_id, [{"type": "input_text", "text": text}])


def _deleted(item_id: str) -> dict:
    return {"type": "conversation.item.deleted", "item_id": item_id}


def _response_created(response_id: str, client_event_id: str | None = None) -> dict:
    response: dict = {"id": response_id, "status": "in_progress"}
    if client_event_id is not None:
        response["metadata"] = {"client_event_id": client_event_id}
    return {"type": "response.created", "response": response}


def _response_done(response_id: str, status: str = "completed") -> dict:
    return {
        "type": "response.done",
        "response": {"id": response_id, "status": status, "usage": None},
    }


def _output_item(kind: str, response_id: str, item: dict | str) -> dict:
    """response.output_item.{kind}; a str item is a message with that id."""
    if isinstance(item, str):
        item = {"id": item, "type": "message"}
    return {
        "type": f"response.output_item.{kind}",
        "response_id": response_id,
        "output_index": 0,
        "item": item,
    }


def _content(kind: str, response_id: str, item_id: str, **fields) -> dict:
    """A content-part-scoped response event, e.g. kind="output_text.delta"."""
    return {
        "type": f"response.{kind}",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        **fields,
    }


def _transcribed(item_id: str, transcript: str, **fields) -> dict:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
        "logprobs": None,
        **fields,
    }


def _speech_started(item_id: str, event_id: str) -> dict:
    return {
        "type": "input_audio_buffer.speech_started",
        "event_id": event_id,
        "item_id": item_id,
        "audio_start_ms": 0,
    }


def _error(error_type: str = "invalid_request_error", **fields) -> dict:
    return {"type": "error", "error": {"type": error_type, **fields}}


def _rejected(event_id: str) -> dict:
    return _error(code="item_rejected", message="conversation item rejected", event_id=event_id)


def _invalid_prev(missing: str, **fields) -> dict:
    message = f"previous_item_id '{missing}' not found in conversation"
    return _error("invalid_previous_item_id", message=message, **fields)


async def _start_generation(session, response_id: str = "resp_1", **kwargs):
    """generate_reply() and answer its response.create; returns (generation, response.create)."""
    generation_fut = session.generate_reply(**kwargs)
    response_create = await session._msg_ch.recv()
    assert response_create["type"] == "response.create"
    _server_event(session, _response_created(response_id, response_create["event_id"]))
    return await generation_fut, response_create


# --- model construction ------------------------------------------------------


@pytest.mark.parametrize("output_modalities", [["text", "audio"], [], ["video"]])
def test_boson_realtime_rejects_invalid_output_modalities(output_modalities):
    with pytest.raises(ValueError):
        _model(output_modalities=output_modalities)


@pytest.mark.parametrize(
    ("kwargs", "attr", "expected"),
    [
        pytest.param({}, "url", "wss://api.boson.ai/v1/realtime", id="hosted-endpoint-default"),
        pytest.param({"url": _URL}, "api_key", "env-key", id="key-from-environment"),
        pytest.param(
            {"url": _URL, "api_key": "passed-key"}, "api_key", "passed-key", id="explicit"
        ),
        # A local dev server with no auth. The environment must not be consulted:
        # a key left over from another run would otherwise be sent to it silently.
        pytest.param({"url": _URL, "api_key": None}, "api_key", None, id="explicit-none"),
    ],
)
def test_boson_realtime_endpoint_and_api_key_resolution(monkeypatch, kwargs, attr, expected):
    monkeypatch.setenv("BOSON_API_KEY", "env-key")
    model = realtime.RealtimeModel(**kwargs)
    assert getattr(model._boson_opts, attr) == expected


# An empty value counts as unset: otherwise the session connects with no
# credential and fails at the server, far from the blank line in the .env file.
@pytest.mark.parametrize("env_key", [None, ""])
def test_boson_realtime_requires_an_api_key(monkeypatch, env_key):
    if env_key is None:
        monkeypatch.delenv("BOSON_API_KEY", raising=False)
    else:
        monkeypatch.setenv("BOSON_API_KEY", env_key)
    with pytest.raises(ValueError, match="BOSON_API_KEY"):
        realtime.RealtimeModel(url=_URL)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # As in the OpenAI plugin, well inside the server's own limit. The replay
        # can put every user turn back, since each has a transcript.
        ({"input_audio_transcription_model": "higgs-stt-3.1"}, 20 * 60),
        ({"input_audio_transcription": {"model": "higgs-stt-3.1"}}, 20 * 60),
        # Without transcripts a replay drops every user turn, so a recycle would
        # make the model forget the user every 20 minutes.
        ({}, None),
        # An explicit value overrides the default either way.
        ({"input_audio_transcription_model": "higgs-stt-3.1", "max_session_duration": None}, None),
        ({"max_session_duration": 600}, 600),
    ],
)
def test_boson_realtime_max_session_duration_default(kwargs, expected):
    model = realtime.RealtimeModel(api_key="test-key", **kwargs)
    assert model._opts.max_session_duration == expected


@pytest.mark.parametrize("max_session_duration", [0, -1.0, float("nan")])
def test_boson_realtime_rejects_a_non_positive_max_session_duration(max_session_duration):
    # The base would recycle at once, every time: a reconnect loop.
    with pytest.raises(ValueError, match="max_session_duration"):
        realtime.RealtimeModel(api_key="test-key", max_session_duration=max_session_duration)


# --- session.update ----------------------------------------------------------

_BOSON_TURN_DETECTION = {
    "type": "server_vad",
    "silence_duration_ms": 800,
    "threshold": 0.6,
    "prefix_padding_ms": 200,
    "interrupt_response": False,
}


def _assert_session_update(session, event: dict, expected: dict) -> None:
    """Check dotted paths into the session.update; "caps.*" reads the model's capabilities."""
    assert event["type"] == "session.update"
    observed = {**event["session"], "caps": vars(session.realtime_model.capabilities)}
    assert {path: _at(observed, path) for path in expected} == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            {"model": "test-model", "voice": "coral"},
            {
                "model": "test-model",
                # The framework sends the Agent's instructions once the session
                # starts; until then the server keeps its own default.
                "instructions": _ABSENT,
                "audio.input.format": {"type": "audio/pcm", "rate": 24000},
                "audio.output.voice": "coral",
                "audio.input.transcription": _ABSENT,
                "audio.input.turn_detection.type": "server_vad",
                "audio.input.noise_reduction": _ABSENT,
                "output_modalities": ["audio"],
                "temperature": _ABSENT,
                "truncation": "auto",
                "caps.audio_output": True,
                # The server preserves client item ids, so the base chat-ctx diff works.
                "caps.mutable_chat_context": True,
                # item.create never triggers a reply; the framework sends response.create.
                "caps.auto_tool_reply_generation": False,
                # No transcription model set -> no user-transcript events.
                "caps.user_transcription": False,
                # The framework scopes tool_choice/tools at the session level instead.
                "caps.per_response_tool_choice": False,
            },
            id="defaults",
        ),
        pytest.param(
            {
                "model": "llm-model",
                "voice": "en_woman",
                "temperature": 0.2,
                "input_audio_transcription": {
                    "model": "asr-model",
                    "language": "english",
                    # `prompt` is not honored by the server; the wrapper drops it
                    # even from a raw transcription dict.
                    "prompt": "Transcribe names carefully.",
                    "temperature": 0.1,
                },
                "turn_detection": _BOSON_TURN_DETECTION,
                "input_audio_noise_reduction": "near_field",
                "tool_choice": {"type": "function", "function": {"name": "lookup_policy"}},
                "truncation": "disabled",
            },
            {
                "audio.input.transcription": {
                    "model": "asr-model",
                    "language": "english",
                    "temperature": 0.1,
                },
                "audio.input.turn_detection": _BOSON_TURN_DETECTION,
                "audio.output.voice": "en_woman",
                "temperature": 0.2,
                "audio.input.noise_reduction": {"type": "near_field"},
                "tool_choice": {"type": "function", "name": "lookup_policy"},
                "truncation": "disabled",
                "caps.user_transcription": True,
            },
            id="boson-options",
        ),
        pytest.param(
            {"input_audio_transcription": None},
            {"audio.input.transcription": _ABSENT, "caps.user_transcription": False},
            id="transcription-none",
        ),
        # A transcription block without a `model` still runs ASR internally on
        # the server but emits no client-facing transcription events.
        pytest.param(
            {"input_audio_transcription": {"language": "english"}},
            {
                "audio.input.transcription": {"language": "english"},
                "caps.user_transcription": False,
            },
            id="transcription-without-model",
        ),
        pytest.param(
            {"turn_detection": None},
            {"audio.input.turn_detection": None, "caps.turn_detection": False},
            id="turn-detection-none",
        ),
        # README documents False as equivalent to None; it must not crash.
        pytest.param(
            {"turn_detection": False},
            {"audio.input.turn_detection": None, "caps.turn_detection": False},
            id="turn-detection-false",
        ),
        pytest.param(
            {"output_modalities": ["text"]},
            {"output_modalities": ["text"], "caps.audio_output": False},
            id="text-only",
        ),
        pytest.param(
            {"input_audio_noise_reduction": {"type": "far_field"}},
            {"audio.input.noise_reduction": {"type": "far_field"}},
            id="noise-reduction-dict",
        ),
    ],
)
async def test_boson_realtime_initial_session_update(open_session, kwargs, expected):
    session = await open_session(drain=False, **kwargs)
    _assert_session_update(session, await session._msg_ch.recv(), expected)


@pytest.mark.parametrize(
    ("kwargs", "update", "expected"),
    [
        pytest.param(
            {},
            {
                # `prompt` is dropped here too.
                "input_audio_transcription": {"model": "asr-v2", "prompt": "ASR hints."},
                "input_audio_noise_reduction": "near_field",
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 900},
            },
            {
                "audio.input.transcription": {"model": "asr-v2"},
                "audio.input.turn_detection": {
                    **_DEFAULT_TURN_DETECTION,
                    "silence_duration_ms": 900,
                },
                "audio.input.noise_reduction": {"type": "near_field"},
                "caps.user_transcription": True,
            },
            id="boson-options",
        ),
        # An update with no audio fields carries no `audio` block at all.
        pytest.param(
            {},
            {"truncation": "disabled"},
            {"truncation": "disabled", "audio": _ABSENT},
            id="no-audio",
        ),
        # The server merges session.update, so leaving `transcription` out would
        # keep the old config; clearing it has to say so explicitly.
        pytest.param(
            {"input_audio_transcription": {"model": "asr-v1"}},
            {"input_audio_transcription": None},
            {"audio.input": {"transcription": None}, "caps.user_transcription": False},
            id="clear-transcription-with-null",
        ),
        pytest.param(
            {},
            {"turn_detection": False},
            {"audio.input.turn_detection": None, "caps.turn_detection": False},
            id="turn-detection-false",
        ),
        # `truncation` has no null on the wire (it ends the session), so None is a no-op.
        pytest.param(
            {"truncation": "disabled"},
            {"truncation": None, "voice": "coral"},
            {"truncation": _ABSENT, "audio.output.voice": "coral"},
            id="none-truncation-is-ignored",
        ),
    ],
)
async def test_boson_realtime_update_options_session_update(open_session, kwargs, update, expected):
    session = await open_session(**kwargs)
    session.realtime_model.update_options(**update)
    _assert_session_update(session, await session._msg_ch.recv(), expected)


async def test_boson_realtime_session_updates_send_only_what_changed(open_session):
    # The server merges session.update into the config it holds, so a change
    # carries only its own fields.
    session = await open_session(input_audio_noise_reduction="near_field")

    await session.update_tools([llm.function_tool(lookup_policy)])
    event = await session._msg_ch.recv()
    assert set(event["session"]) == {"type", "tools"}
    assert [tool["name"] for tool in event["session"]["tools"]] == ["lookup_policy"]

    await session.update_instructions("Be thorough.")
    event = await session._msg_ch.recv()
    assert event["session"] == {"type": "realtime", "instructions": "Be thorough."}

    session.update_options(voice="coral", temperature=0.5)
    event = await session._msg_ch.recv()
    assert event["session"] == {
        "type": "realtime",
        "temperature": 0.5,
        "audio": {"output": {"voice": "coral"}},
    }

    # Under merge, an omitted field keeps its old value; turning something off
    # has to send the empty value itself.
    await session.update_tools([])
    assert (await session._msg_ch.recv())["session"]["tools"] == []

    session.update_options(input_audio_noise_reduction=None)
    event = await session._msg_ch.recv()
    assert event["session"]["audio"] == {"input": {"noise_reduction": None}}


async def test_boson_realtime_reconnect_config_carries_every_partial_update(open_session):
    # A reconnect starts a session that holds nothing, so the config it sends first
    # must include everything changed piecemeal before.
    session = await open_session(input_audio_transcription={"model": "asr-v1", "language": "en"})
    await session.update_tools([llm.function_tool(lookup_policy)])
    await session.update_instructions("Be thorough.")
    session.update_options(voice="coral", truncation="disabled")
    # Dicts are merged into what the server holds, not swapped for it: the
    # server keeps the threshold and the language left out here.
    session.update_options(
        turn_detection={"type": "server_vad", "create_response": False},
        input_audio_transcription={"model": "asr-v2"},
    )

    full = session._create_session_update_event()["session"]
    assert full["instructions"] == "Be thorough."
    assert [tool["name"] for tool in full["tools"]] == ["lookup_policy"]
    assert full["audio"]["output"]["voice"] == "coral"
    assert full["truncation"] == "disabled"
    assert full["audio"]["input"]["turn_detection"] == {
        **_DEFAULT_TURN_DETECTION,
        "create_response": False,
    }
    assert full["audio"]["input"]["transcription"] == {"model": "asr-v2", "language": "en"}


async def test_boson_realtime_model_options_merge_dicts_like_sessions(open_session):
    # An open session merges a partial dict into what its server holds, so the
    # model has to merge it too: otherwise its capabilities, and every session
    # opened after the change, disagree with the sessions already open.
    first = await open_session(input_audio_transcription_model="higgs-stt-3.1")
    model = first.realtime_model
    model.update_options(
        input_audio_transcription={"language": "fr"},
        turn_detection={"threshold": 0.7},
    )
    await first._msg_ch.recv()  # the same change, merged by the server

    assert model.capabilities.user_transcription is True
    assert first.capabilities.user_transcription is True
    new_session = model.session()
    try:
        audio_input = (await new_session._msg_ch.recv())["session"]["audio"]["input"]
        assert audio_input["transcription"] == {"model": "higgs-stt-3.1", "language": "fr"}
        assert audio_input["turn_detection"]["type"] == "server_vad"
        assert audio_input["turn_detection"]["threshold"] == 0.7
        assert audio_input["turn_detection"]["silence_duration_ms"] == 500
    finally:
        await new_session.aclose()


async def test_boson_realtime_unsupported_tool_type_warns_once(open_session, monkeypatch):
    # A tool with no wire shape is left out of session.update, and silently dropping
    # it would leave no trace, so it warns.
    class _UnknownTool:
        pass

    session = await open_session()
    warnings = _capture_warnings(monkeypatch)

    await session.update_tools([_UnknownTool()])  # type: ignore[list-item]
    assert (await session._msg_ch.recv())["session"]["tools"] == []
    assert len(warnings) == 1

    # Warned once per session, not once per session.update.
    await session.update_tools([_UnknownTool()])  # type: ignore[list-item]
    await session._msg_ch.recv()
    assert len(warnings) == 1


async def test_boson_realtime_raw_function_tool_schema_uses_boson_shape(open_session):
    schema = {
        "name": "lookup_policy",
        "description": "Lookup a policy.",
        "parameters": {"type": "object", "properties": {}},
    }
    tool = llm.function_tool(lookup_policy, raw_schema={**schema, "meta": {"provider": "test"}})

    session = await open_session()
    await session.update_tools([tool])
    event = await session._msg_ch.recv()
    assert event["session"]["tools"] == [{"type": "function", **schema}]


async def test_boson_realtime_transcription_emits_stt_metrics_with_the_model(open_session):
    session = await open_session(input_audio_transcription_model="higgs-stt-3.1")
    metrics = _collect(session, "metrics_collected")

    def transcribed(item_id: str) -> None:
        usage = {"type": "duration", "seconds": 1.5}
        done = _transcribed(item_id, "hello", event_id=f"event_{item_id}", usage=usage)
        _server_event(session, _user_item_added(item_id, []), done)

    transcribed("user_1")
    session.update_options(input_audio_transcription={"model": "higgs-stt-4"})
    transcribed("user_2")

    assert [m.metadata.model_name for m in metrics] == ["higgs-stt-3.1", "higgs-stt-4"]
    assert [m.audio_duration for m in metrics] == [1.5, 1.5]
    assert metrics[0].request_id == "event_user_1"


# --- generations -------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "part", "modalities"),
    [
        pytest.param({"output_modalities": ["text"]}, _TEXT_PART, ["text"], id="text-only"),
        pytest.param({}, {"type": "audio", "transcript": ""}, ["audio", "text"], id="audio"),
    ],
)
async def test_boson_realtime_generation_streams_output(open_session, kwargs, part, modalities):
    session = await open_session(**kwargs)
    generation, _ = await _start_generation(session)
    assert generation.response_id == "resp_1"
    assert generation.user_initiated is True

    _server_event(session, _output_item("added", "resp_1", "item_1"))
    message = await generation.message_stream.recv()
    assert message.message_id == "item_1"

    _server_event(session, _content("content_part.added", "resp_1", "item_1", part=part))
    assert await message.modalities == modalities

    if "audio" in modalities:
        delta = base64.b64encode(b"\x00\x00" * 240).decode("utf-8")
        _server_event(session, _content("output_audio.delta", "resp_1", "item_1", delta=delta))
        frame = await message.audio_stream.recv()
        assert (frame.sample_rate, frame.num_channels, frame.samples_per_channel) == (24000, 1, 240)
    else:
        _server_event(session, _content("output_text.delta", "resp_1", "item_1", delta="Hi."))
        assert await message.text_stream.recv() == "Hi."
        _server_event(session, _content("output_text.done", "resp_1", "item_1", text="Hi."))

    _server_event(session, _output_item("done", "resp_1", "item_1"), _response_done("resp_1"))
    assert session._current_generation is None


async def test_boson_realtime_generate_reply_instructions_ride_in_response_create(open_session):
    # Per-response `instructions` ride in response.create, with no scoping
    # session.update around it.
    session = await open_session()
    await session.update_instructions("Base instructions.")
    await session._msg_ch.recv()

    _, response_create = await _start_generation(session, instructions="Greet the user.")
    # As with OpenAI, the server uses this in place of the session
    # instructions, so the event carries both.
    assert response_create["response"]["instructions"] == "Base instructions.\nGreet the user."

    # Nothing to restore: the session was never reconfigured.
    assert session._msg_ch.empty()
    assert session._instructions == "Base instructions."


async def test_boson_realtime_generate_reply_ignores_per_response_tools(open_session, monkeypatch):
    # The server accepts and ignores per-response tools/tool_choice, so they are not
    # forwarded; a caller reaching this session directly gets a warning.
    session = await open_session()
    warnings = _capture_warnings(monkeypatch)

    session.generate_reply(tool_choice="none", tools=[llm.function_tool(lookup_policy)])
    response_create = await session._msg_ch.recv()
    assert response_create["type"] == "response.create"
    assert "tool_choice" not in response_create["response"]
    assert "tools" not in response_create["response"]
    # No session.update either — the session config is untouched.
    assert session._msg_ch.empty()
    assert len(warnings) == 1

    # Warned once per session, not once per call.
    session.generate_reply(tool_choice="none")
    await session._msg_ch.recv()
    assert len(warnings) == 1


async def test_boson_realtime_discarded_response_keeps_active_generation_streams(open_session):
    # A response that timed out before the server created it must be discarded
    # without cutting off the streams of a newer in-progress generation.
    session = await open_session()
    session._discarded_event_ids.add("evt_stale")

    _server_event(session, _response_created("resp_active"))
    active_generation = session._current_generation
    assert active_generation is not None

    _server_event(session, _response_created("resp_stale", "evt_stale"))
    # the active generation's streams stay open and interrupt() targets it
    assert not active_generation.message_ch.closed
    assert not active_generation.function_ch.closed
    assert session._current_generation is active_generation
    assert session._current_response_id == "resp_active"

    # the base cancelled the stale response by id
    cancel_event = await session._msg_ch.recv()
    assert cancel_event["type"] == "response.cancel"
    assert cancel_event["response_id"] == "resp_stale"

    # the stale response's terminal event must not close the active one either
    _server_event(session, _response_done("resp_stale", "cancelled"))
    assert not active_generation.message_ch.closed
    assert session._current_generation is active_generation
    assert session._current_response_id == "resp_active"

    # the active response still completes normally afterwards
    _server_event(session, _response_done("resp_active"))
    assert session._current_generation is None
    assert session._current_response_id is None


async def test_boson_realtime_discarded_response_without_active_generation_is_skipped(
    open_session,
):
    # With no generation in progress, the base discard marker must stay in the
    # slot so the stale response's trailing events are eaten.
    session = await open_session()
    session._discarded_event_ids.add("evt_stale")

    _server_event(session, _response_created("resp_stale", "evt_stale"))
    assert session._current_generation is not None  # the discard marker
    assert session._current_response_id is None

    _server_event(session, _response_done("resp_stale", "cancelled"))
    assert session._current_generation is None


async def test_boson_realtime_superseded_response_still_records_call_arguments(open_session):
    # conversation.item.added carries a call before its arguments exist; only
    # output_item.done fills them in. If that arrives after the next response
    # took over, a reconnect must still replay the call with its arguments.
    session = await open_session()
    call_item = _item("call_item_a", "function_call", call_id="call_a", name="lookup_weather")
    call_item.update(status="in_progress", arguments="")
    _server_event(
        session,
        _response_created("resp_a"),
        _item_added_echo(call_item),
        _response_created("resp_b"),
    )
    generation_b = session._current_generation

    done_item = {**call_item, "status": "completed", "arguments": '{"city": "Paris"}'}
    _server_event(session, _output_item("done", "resp_a", done_item))

    remote_item = session._remote_chat_ctx.get("call_item_a")
    assert remote_item is not None
    assert remote_item.item.arguments == '{"city": "Paris"}'
    # Recorded only: the superseded call is not handed to B to run.
    assert session._current_generation is generation_b
    assert generation_b.function_ch.qsize() == 0


async def test_boson_realtime_stale_response_done_does_not_corrupt_new_generation(open_session):
    # A's response.done can arrive after B's response.created (it queues behind A's
    # audio on the server), so every response-scoped handler filters by response id.
    session = await open_session()
    _server_event(session, _response_created("resp_a"), _output_item("added", "resp_a", "item_a"))
    generation_a = session._current_generation

    # B supersedes A before A's response.done arrives, and closes A's streams.
    _server_event(session, _response_created("resp_b"))
    generation_b = session._current_generation
    assert generation_b is not None and generation_b is not generation_a
    assert generation_a.message_ch.closed and generation_a.function_ch.closed
    assert session._current_response_id == "resp_b"

    _server_event(session, _output_item("added", "resp_b", "item_b"))
    message_b = await generation_b.message_ch.recv()
    assert message_b.message_id == "item_b"

    _server_event(session, _content("content_part.added", "resp_b", "item_b", part=_TEXT_PART))
    assert await message_b.modalities == ["text"]

    # A's late delta names an item only A had: dropped, not misapplied to B.
    _server_event(session, _content("output_text.delta", "resp_a", "item_a", delta="stale from A"))

    # A's late response.done must not close B.
    _server_event(session, _response_done("resp_a"))
    assert session._current_generation is generation_b
    assert session._current_response_id == "resp_b"
    assert not generation_b.message_ch.closed
    assert not generation_b.function_ch.closed

    # B keeps streaming and completes normally afterwards.
    _server_event(session, _content("output_text.delta", "resp_b", "item_b", delta="hello from B"))
    assert await message_b.text_stream.recv() == "hello from B"

    _server_event(session, _response_done("resp_b"))
    assert session._current_generation is None
    assert session._current_response_id is None


async def test_boson_realtime_late_event_after_generation_finished_is_dropped(open_session):
    # A superseded response's trailing events can arrive after the slot emptied;
    # the base handlers assert on a generation, so they must be dropped, not passed.
    session = await open_session()
    _server_event(session, _response_created("resp_a"), _output_item("added", "resp_a", "item_a"))

    # B supersedes A, then runs to completion — clearing the slot.
    _server_event(session, _response_created("resp_b"), _response_done("resp_b"))
    assert session._current_generation is None
    assert session._current_response_id is None

    # Every response-scoped event for A must now be dropped, not forwarded.
    assert session._is_stale_response_scoped_event("resp_a") is True
    silence = base64.b64encode(b"\x00\x00").decode()
    _server_event(
        session,
        _content("output_text.delta", "resp_a", "item_a", delta="stale from A"),
        _content("output_audio.delta", "resp_a", "item_a", delta=silence),
        _output_item("added", "resp_a", "item_a"),
        _output_item("done", "resp_a", "item_a"),
    )
    assert session._current_generation is None


async def test_boson_realtime_discarded_response_trailing_event_after_active_done(open_session):
    # A response discarded on arrival while a real one streams is filtered by id, not
    # parked in the slot; its trailing events must still drop once the slot empties.
    session = await open_session()
    _server_event(session, _response_created("resp_a"))
    generation_a = session._current_generation

    session._discarded_event_ids.add("evt_discarded")
    _server_event(session, _response_created("resp_d", "evt_discarded"))
    # A keeps the slot; the discarded response never becomes current.
    assert session._current_generation is generation_a
    assert session._current_response_id == "resp_a"
    assert (await session._msg_ch.recv())["type"] == "response.cancel"

    _server_event(session, _response_done("resp_a"))
    assert session._current_generation is None

    _server_event(session, _output_item("added", "resp_d", "item_d"))
    assert session._current_generation is None


@pytest.mark.parametrize(
    ("kwargs", "server_vad"),
    [
        pytest.param({}, True, id="server-vad"),
        # Higgs server VAD always cancels the active response on speech start, whatever
        # `interrupt_response` says, so the client skips its own duplicate cancel
        # whenever server VAD is on.
        pytest.param(
            {"turn_detection": {"type": "server_vad", "interrupt_response": False}},
            True,
            id="interrupt-response-false",
        ),
        pytest.param({"turn_detection": None}, False, id="client-side"),
    ],
)
async def test_boson_realtime_interrupt_sends_response_cancel_unless_server_did(
    open_session, kwargs, server_vad
):
    session = await open_session(**kwargs)
    await _start_generation(session)

    if server_vad:
        # The server already cancelled on speech start: no duplicate cancel.
        session.on("input_speech_started", lambda _event: session.interrupt())
        _server_event(session, {"type": "input_audio_buffer.speech_started"})
        assert session._msg_ch.empty()

    # Only the one cancel the server already made is skipped.
    session.interrupt()
    cancel_event = await session._msg_ch.recv()
    assert cancel_event["type"] == "response.cancel"
    assert cancel_event["response_id"] == "resp_1"


# --- remote conversation mirror and transcripts --------------------------------


_AUDIO_ONLY_TURN = _user_item_added("user_1", [{"type": "input_audio", "audio": ""}])


@pytest.mark.parametrize(
    ("events", "content"),
    [
        # The server merges consecutive same-role turns into one item and
        # re-emits conversation.item.added with the same id and cumulative content.
        pytest.param(
            [_user_text("user_1", "what's"), _user_text("user_1", "what's the weather")],
            ["what's the weather"],
            id="merged-readd-updates-text",
        ),
        # Audio-input configs re-add the merged item with an empty input_audio part;
        # the transcript already applied must not be wiped.
        pytest.param(
            [_AUDIO_ONLY_TURN, _transcribed("user_1", "hello there"), _AUDIO_ONLY_TURN],
            ["hello there"],
            id="merged-readd-without-text-keeps-transcript",
        ),
        pytest.param(
            [_user_text("user_1", "hello"), _transcribed("user_1", "hello world")],
            ["hello world"],
            id="transcription-replaces-text",
        ),
    ],
)
async def test_boson_realtime_remote_item_text_follows_the_server(open_session, events, content):
    session = await open_session()
    completed = _collect(session, "input_audio_transcription_completed")
    _server_event(session, *events)

    assert len(session.chat_ctx.items) == 1
    remote_item = session._remote_chat_ctx.get("user_1")
    assert remote_item is not None
    assert remote_item.item.content == content
    # Each transcription also reaches the app as a final transcript.
    transcripts = [e["transcript"] for e in events if "transcript" in e]
    assert [(c.transcript, c.is_final) for c in completed] == [(t, True) for t in transcripts]

    # What the server already holds is not recreated by the next sync.
    await session.update_chat_ctx(_ctx(("user_1", "user", content[0])))
    assert session._msg_ch.empty()


async def test_boson_realtime_transcription_carries_the_turn_start(open_session):
    # The framework dates the user's message by turn_started_at so a late transcript
    # cannot land after the reply it prompted. The server merges consecutive speech
    # into one item and re-transcribes it, so every revision must report the turn's
    # own start, not the latest fragment's and not None after the first.
    session = await open_session()
    completed = _collect(session, "input_audio_transcription_completed")
    first = "so, these thought experiments"
    merged = f"{first} would go against quantum physics?"

    before = time.time()
    _server_event(session, _speech_started("user_1", "evt_speech_1"))
    after = time.time()
    turn_start = session._input_speech_started_at["user_1"]
    assert before <= turn_start <= after
    _server_event(
        session,
        _user_item_added("user_1", [{"type": "input_audio"}]),
        _transcribed("user_1", first),
    )
    # second fragment of the same turn, merged into the same item
    _server_event(
        session, _speech_started("user_1", "evt_speech_2"), _transcribed("user_1", merged)
    )

    assert [c.transcript for c in completed] == [first, merged]
    assert [c.turn_started_at for c in completed] == [turn_start, turn_start]


async def test_boson_realtime_tracks_server_assigned_session_id(open_session):
    session = await open_session(drain=False)
    assert session.session_id is None
    _server_event(
        session,
        {
            "type": "session.created",
            "event_id": "evt_1",
            "session": {"id": "sess_server123", "object": "realtime.session"},
        },
    )
    assert session.session_id == "sess_server123"


# --- chat context sync -------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected_item"),
    [
        pytest.param(
            ("asst_1", "assistant", "Assistant note."),
            {"role": "assistant", "content": [{"type": "text", "text": "Assistant note."}]},
            id="assistant-text",
        ),
        pytest.param(
            ("user_audio_1", "user", [llm.AudioContent(frame=[], transcript="hello from audio")]),
            {"role": "user", "content": [{"type": "input_text", "text": "hello from audio"}]},
            id="audio-uses-transcript",
        ),
    ],
)
async def test_boson_realtime_chat_ctx_item_uses_boson_shape(open_session, message, expected_item):
    session = await open_session()
    update_task = asyncio.create_task(session.update_chat_ctx(_ctx(message)))
    create_event = await session._msg_ch.recv()

    assert create_event["type"] == "conversation.item.create"
    assert create_event["previous_item_id"] is None
    assert create_event["item"]["id"] == message[0]
    assert {key: create_event["item"][key] for key in expected_item} == expected_item

    # update_chat_ctx resolves once the server echoes the created item
    # (the echo carries the client-supplied id).
    assert not update_task.done()
    _server_event(session, _item_added_echo(create_event["item"]))
    await asyncio.wait_for(update_task, timeout=1.0)


@pytest.mark.parametrize(
    "dropped",
    [
        # The server only stores assistant/user items; system/developer items would
        # never get their item.added echo.
        pytest.param(
            [("sys_1", "system", "System note."), ("dev_1", "developer", "Developer note.")],
            id="system-and-developer",
        ),
        # Nor can it store an item with no text content.
        pytest.param(
            [("img_1", "user", [llm.ImageContent(image="https://example.com/a.png")])],
            id="textless",
        ),
    ],
)
async def test_boson_realtime_chat_ctx_skips_unsendable_items_and_remaps_prev(
    open_session, dropped
):
    # Unsendable items are dropped client-side, and a previous_item_id pointing at
    # one is remapped to its predecessor.
    session = await open_session()
    update_task = asyncio.create_task(
        session.update_chat_ctx(
            _ctx(("user_1", "user", "first"), *dropped, ("user_2", "user", "second"))
        )
    )
    first_event = await session._msg_ch.recv()
    second_event = await session._msg_ch.recv()
    assert session._msg_ch.empty()

    assert first_event["item"]["id"] == "user_1"
    assert second_event["item"]["id"] == "user_2"
    assert second_event["previous_item_id"] == "user_1"

    _server_event(
        session,
        _item_added_echo(first_event["item"]),
        _item_added_echo(second_event["item"], "user_1"),
    )
    await asyncio.wait_for(update_task, timeout=1.0)


async def test_boson_realtime_update_chat_ctx_deletes_removed_items(open_session):
    # The server keeps client ids, so the base diff deletes removed items by id.
    session = await open_session()
    _server_event(session, _user_text("user_1", "hi"), _user_text("user_2", "again"))

    update_task = asyncio.create_task(session.update_chat_ctx(_ctx(("user_1", "user", "hi"))))
    delete_event = await session._msg_ch.recv()
    assert delete_event["type"] == "conversation.item.delete"
    assert delete_event["item_id"] == "user_2"
    assert session._msg_ch.empty()

    _server_event(session, _deleted("user_2"))
    await asyncio.wait_for(update_task, timeout=1.0)
    assert session._remote_chat_ctx.get("user_2") is None


async def test_boson_realtime_update_chat_ctx_rebuilds_on_head_insert(open_session):
    # The server has no insert-at-head (previous_item_id=None appends at the tail),
    # so prepending ahead of what it holds rebuilds the conversation instead.
    session = await open_session()
    asst = _item("asst_1", role="assistant", content=[{"type": "text", "text": "hello"}])
    _server_event(session, _user_text("user_1", "hi"), _item_added_echo(asst, "user_1"))
    assert len(session.chat_ctx.items) == 2

    update_task = asyncio.create_task(
        session.update_chat_ctx(
            _ctx(
                ("summary_1", "assistant", "Summary."),
                ("user_1", "user", "hi"),
                ("asst_1", "assistant", "hello"),
            )
        )
    )
    deletes = [await session._msg_ch.recv() for _ in range(2)]
    assert [e["type"] for e in deletes] == ["conversation.item.delete"] * 2
    assert {e["item_id"] for e in deletes} == {"user_1", "asst_1"}

    creates = [await session._msg_ch.recv() for _ in range(3)]
    assert session._msg_ch.empty()
    assert [(e["item"]["id"], e["previous_item_id"]) for e in creates] == [
        ("summary_1", None),
        ("user_1", "summary_1"),
        ("asst_1", "user_1"),
    ]

    _server_event(session, *(_deleted(e["item_id"]) for e in deletes))
    _server_event(session, *(_item_added_echo(e["item"], e["previous_item_id"]) for e in creates))
    await asyncio.wait_for(update_task, timeout=1.0)


async def test_boson_realtime_unsupported_role_at_head_does_not_rebuild(open_session):
    # A system item is never sent, so the diff would see it missing at the head on
    # every sync and rebuild forever. Unsupported roles are filtered before the diff.
    session = await open_session()
    _server_event(session, _user_text("user_1", "hi"))

    system = ("sys_1", "system", "You are helpful.")
    target = _ctx(system, ("user_1", "user", "hi"))
    # Already in sync apart from the undeliverable system item: nothing to do.
    assert session._create_update_chat_ctx_events(target) == []
    # And still nothing to do on the next sync of the same context.
    assert session._create_update_chat_ctx_events(target) == []

    # A genuine head insert of a *supported* item still rebuilds.
    events = session._create_update_chat_ctx_events(
        _ctx(system, ("summary_1", "assistant", "Summary."), ("user_1", "user", "hi"))
    )
    assert [ev.type for ev in events] == [
        "conversation.item.delete",
        "conversation.item.create",
        "conversation.item.create",
    ]
    assert events[0].item_id == "user_1"
    assert (events[1].item.id, events[1].previous_item_id) == ("summary_1", None)
    assert (events[2].item.id, events[2].previous_item_id) == ("user_1", "summary_1")


async def test_boson_realtime_head_text_change_rebuilds_rather_than_reorders(open_session):
    # Editing the head item's text reaches the rebuild too: the base sends a delete
    # and a create anchored at "root", which incrementally would append it after
    # the turns it preceded.
    session = await open_session()
    # Chained after user_1: _user_text reports no predecessor, which the mirror reads
    # as insert-at-head.
    second = _user_text("user_2", "again")
    second["previous_item_id"] = "user_1"
    _server_event(session, _user_text("user_1", "hi"), second)

    target = _ctx(("user_1", "user", "hello there"), ("user_2", "user", "again"))
    update_task = asyncio.create_task(session.update_chat_ctx(target))
    # Bounded, so a regression that sends two events fails instead of blocking.
    events = []
    with contextlib.suppress(asyncio.TimeoutError):
        while True:
            events.append(await asyncio.wait_for(session._msg_ch.recv(), timeout=0.5))

    # Everything torn down, then re-sent in target order.
    assert [e["type"] for e in events] == [
        "conversation.item.delete",
        "conversation.item.delete",
        "conversation.item.create",
        "conversation.item.create",
    ]
    assert [e["item_id"] for e in events[:2]] == ["user_1", "user_2"]
    assert [(e["item"]["id"], e["previous_item_id"]) for e in events[2:]] == [
        ("user_1", None),
        ("user_2", "user_1"),
    ]

    _server_event(session, _deleted("user_1"), _deleted("user_2"))
    _server_event(session, _user_text("user_1", "x"), _user_text("user_2", "y"))
    await asyncio.wait_for(update_task, timeout=1.0)


async def test_boson_realtime_rebuild_warns_about_turns_it_cannot_put_back(open_session, caplog):
    # A rebuild recreates only items with text, so a user turn still waiting on its
    # transcript is lost until a later sync. Inherent, but it must not be silent.
    session = await open_session()
    # An audio turn whose transcription has not arrived.
    _server_event(session, _user_item_added("user_1", [{"type": "input_audio"}]))
    assert session._remote_chat_ctx.get("user_1").item.content == []

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.boson"):
        target = _ctx(("summary", "user", "so far: ..."), ("user_1", "user", []))
        update_task = asyncio.create_task(session.update_chat_ctx(target))
        events = [await session._msg_ch.recv(), await session._msg_ch.recv()]
        assert session._msg_ch.empty()

    assert [e["type"] for e in events] == ["conversation.item.delete", "conversation.item.create"]
    assert events[0]["item_id"] == "user_1"
    # Deleted, and not among the recreates.
    assert events[1]["item"]["id"] == "summary"
    assert "user_1" in _reported_item_ids(caplog)

    _server_event(session, _deleted("user_1"), _user_text("summary", "s"))
    await asyncio.wait_for(update_task, timeout=1.0)


async def test_boson_realtime_emptied_turn_is_deleted_loudly(open_session, caplog):
    # A content change is a delete plus a create under one id. With no text there
    # is no create, so the turn is deleted rather than emptied, and that is reported
    # as the rebuild path reports what it cannot put back.
    session = await open_session()
    # Each added item goes in at the head, leaving [user_1, user_2]: user_2 is not
    # at the head, so this stays on the incremental path.
    _server_event(session, _user_text("user_2", "there"), _user_text("user_1", "hi"))
    assert [item.id for item in session._remote_chat_ctx.to_chat_ctx().items] == [
        "user_1",
        "user_2",
    ]

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.boson"):
        update_task = asyncio.create_task(
            session.update_chat_ctx(_ctx(("user_1", "user", "hi"), ("user_2", "user", [])))
        )
        events = [await session._msg_ch.recv()]
        assert session._msg_ch.empty()

    assert [e["type"] for e in events] == ["conversation.item.delete"]
    assert events[0]["item_id"] == "user_2"
    assert "user_2" in _reported_item_ids(caplog)

    _server_event(session, _deleted("user_2"))
    await asyncio.wait_for(update_task, timeout=1.0)
    # Gone from the mirror, so it returns as a create after user_1 once it has text.
    assert session._remote_chat_ctx.get("user_2") is None
    events = session._create_update_chat_ctx_events(
        _ctx(("user_1", "user", "hi"), ("user_2", "user", "there again"))
    )
    assert [ev.type for ev in events] == ["conversation.item.create"]
    assert events[0].item.id == "user_2"
    assert events[0].previous_item_id == "user_1"

    # The warning keys on a delete going out for the id, not on a create being
    # dropped: an item the server never held, here a system one, is no loss.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.boson"):
        events = session._create_update_chat_ctx_events(
            _ctx(("user_1", "user", "hi"), ("sys_1", "system", "Be brief."))
        )
    assert events == []
    assert not _reported_item_ids(caplog)


# --- server errors -----------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    ["insufficient_quota", "monthly_cap_reached", "contract_ended", "no_billing_account"],
)
async def test_boson_realtime_entitlement_refusal_is_not_recoverable(open_session, code):
    # The server sends this just before closing with 4429, so emitting it as
    # recoverable would promise a recovery the close rules out: it is raised and
    # _main_task reports it with recoverable=False. All four share the type
    # "insufficient_quota" but the base's fatal set matches only one code.
    session = await open_session()
    errors = _collect(session, "error")

    with pytest.raises(APIError) as exc_info:
        _server_event(
            session, _error("insufficient_quota", code=code, message="monthly cap reached")
        )
    assert exc_info.value.retryable is False
    # Raised, not emitted; _main_task emits it with recoverable=False.
    assert errors == []


_EMPTY_COMMIT = _error(code="input_audio_buffer_commit_empty", message="buffer is empty")


@pytest.mark.parametrize(
    ("kwargs", "error", "recoverable"),
    [
        # An error that is merely a bad request must not end the session.
        pytest.param(
            {}, _error(code="bad_request", message="unknown field"), [True], id="bad-request"
        ),
        # With server VAD on, commit_user_turn() routinely commits a buffer the server
        # already emptied, which is not a failure worth reporting.
        pytest.param({}, _EMPTY_COMMIT, [], id="empty-commit-under-server-vad"),
        # With turn detection off the client owns every commit, so it is a real bug.
        pytest.param(
            {"turn_detection": None}, _EMPTY_COMMIT, [True], id="empty-commit-without-server-vad"
        ),
        # A redundant response.cancel is not a user-facing error (as in Boson's pipecat).
        pytest.param(
            {},
            _error(
                code="response_not_active",
                message="No active response to cancel.",
                event_id="evt_cancel_1",
            ),
            [],
            id="response-not-active",
        ),
    ],
)
async def test_boson_realtime_error_reporting(open_session, kwargs, error, recoverable):
    session = await open_session(**kwargs)
    errors = _collect(session, "error")
    _server_event(session, error)
    assert [e.recoverable for e in errors] == recoverable


@pytest.mark.parametrize(
    ("error", "reported"),
    [
        # None: the error names this reply's response.create, so it fails that call.
        pytest.param(None, True, id="names-the-reply"),
        # An error whose event_id belongs to something else (e.g. an unrelated
        # conversation.item.create) still surfaces, so it isn't silently lost...
        pytest.param(
            _error(code="bad_request", message="item.create failed", event_id="chat_ctx_create_1"),
            True,
            id="unrelated-event-id",
        ),
        # ...while this one, raised without an event_id/code the client can
        # correlate, is swallowed.
        pytest.param(_invalid_prev("user_9"), False, id="invalid-previous-item-id"),
    ],
)
async def test_boson_realtime_error_fails_only_the_generate_reply_it_names(
    open_session, error, reported
):
    # An error that can't be correlated to a pending generate_reply() must not
    # blanket-fail every in-flight call.
    session = await open_session()
    errors = _collect(session, "error")
    generation_fut = session.generate_reply()
    event_id = (await session._msg_ch.recv())["event_id"]  # response.create

    _server_event(session, error or _error(code="bad_request", message="failed", event_id=event_id))
    assert bool(errors) is reported
    if error is None:
        with pytest.raises(llm.RealtimeError, match="failed"):
            await generation_fut
        assert session._response_created_futures == {}
    else:
        assert not generation_fut.done()
        assert generation_fut in session._response_created_futures.values()
        generation_fut.cancel()


@pytest.mark.parametrize("with_event_id", [False, True])
async def test_boson_realtime_invalid_previous_item_id_fails_update_chat_ctx_fast(
    open_session, with_event_id
):
    # An older server names neither its event id nor the rejected item, only the
    # missing previous_item_id, so attribution rests on item_id; without it
    # update_chat_ctx() would wait out the base's 5s timeout. The current server
    # also names the event id and settles the same future first, so escalation
    # must not depend on being the one to settle it.
    session = await open_session()
    _server_event(session, _user_text("user_1", "hi"))

    # user_2 chains onto user_1 as its previous_item_id.
    update_task = asyncio.create_task(
        session.update_chat_ctx(_ctx(("user_1", "user", "hi"), ("user_2", "user", "again")))
    )
    create_event = await session._msg_ch.recv()
    assert create_event["item"]["id"] == "user_2"
    assert create_event["previous_item_id"] == "user_1"
    assert create_event["event_id"]

    event_id = {"event_id": create_event["event_id"]} if with_event_id else {}
    _server_event(session, _invalid_prev("user_1", item_id="user_2", **event_id))

    with pytest.raises(llm.RealtimeError, match="previous_item_id 'user_1' not found"):
        await asyncio.wait_for(update_task, timeout=0.5)


@pytest.mark.parametrize("fatal", [False, True], ids=["item-rejected", "fatal-refusal"])
async def test_boson_realtime_rejected_item_settles_its_own_create(open_session, fatal):
    # Every other rejected item.create names its client event id and never gets
    # its item.added echo; matching them avoids the base's 5s timeout.
    session = await open_session()
    errors = _collect(session, "error")

    update_task = asyncio.create_task(session.update_chat_ctx(_ctx(("user_1", "user", "hi"))))
    create_event = await session._msg_ch.recv()
    assert create_event["item"]["id"] == "user_1"
    assert create_event["event_id"]

    if fatal:
        # Settling a chat-ctx future must not swallow a refusal that ends the session.
        with pytest.raises(APIError) as exc_info:
            _server_event(
                session,
                _error(
                    "insufficient_quota",
                    code="monthly_cap_reached",
                    message="monthly cap reached",
                    event_id=create_event["event_id"],
                ),
            )
        assert exc_info.value.retryable is False
    else:
        _server_event(session, _rejected(create_event["event_id"]))
    # Resolves on the error, not 5s later, and does not raise: one rejected item
    # is not worth failing the turn over.
    await asyncio.wait_for(update_task, timeout=0.5)
    # Nor is it also a session error: a false alarm during ordinary syncing.
    assert errors == []


async def test_boson_realtime_rejected_item_survives_a_later_echo(open_session):
    # A rejected create stays registered in _item_create_future until
    # update_chat_ctx() clears it, so an echo the server sends anyway finds a
    # settled future; resolving it would raise InvalidStateError.
    session = await open_session()
    _server_event(session, _user_text("user_1", "hi"))

    # Changing an item's content sends a delete and a create under its id.
    update_task = asyncio.create_task(session.update_chat_ctx(_ctx(("user_1", "user", "changed"))))
    events = [await session._msg_ch.recv(), await session._msg_ch.recv()]
    create_event = next(e for e in events if e["type"] == "conversation.item.create")

    _server_event(session, _rejected(create_event["event_id"]))
    # The echo the server should not have sent. Handling it must not raise.
    _server_event(session, _user_text("user_1", "hi"))

    _server_event(session, _deleted("user_1"))
    await asyncio.wait_for(update_task, timeout=1.0)


@pytest.mark.parametrize(
    "error",
    [
        # A server that names no rejected item gives nothing to attribute the failure
        # to: the turn falls back to update_chat_ctx()'s own timeout.
        pytest.param(
            _error("invalid_previous_item_id", message="the item you referenced does not exist"),
            id="unattributable",
        ),
        # A late error for a create that already timed out must not be pinned on a
        # different in-flight call.
        pytest.param(_invalid_prev("root_1", item_id="user_stale"), id="stale-item"),
    ],
)
async def test_boson_realtime_unmatched_invalid_previous_item_id_blames_nobody(open_session, error):
    session = await open_session()
    assert "user_stale" not in session._item_create_future

    # An append onto an empty remote context: previous_item_id is None.
    update_task = asyncio.create_task(session.update_chat_ctx(_ctx(("user_1", "user", "hi"))))
    create_event = await session._msg_ch.recv()
    assert create_event["item"]["id"] == "user_1"
    assert create_event["previous_item_id"] is None

    _server_event(session, error)
    # The append is untouched: no exception pinned on it, still pending.
    assert not session._item_create_future["user_1"].done()
    assert session._chat_ctx_sync_error is None
    assert not update_task.done()

    # It still resolves normally when its own echo arrives.
    _server_event(session, _item_added_echo(create_event["item"]))
    await asyncio.wait_for(update_task, timeout=1.0)


async def test_boson_realtime_concurrent_update_chat_ctx_keeps_sync_error(open_session):
    # The sync error travels through one session-level slot, its only channel. A
    # second update_chat_ctx() entering while the first is parked in the base must
    # not clear it, or the first reports success for a failed sync.
    session = await open_session()
    _server_event(session, _user_text("user_1", "hi"))

    first_task = asyncio.create_task(
        session.update_chat_ctx(_ctx(("user_1", "user", "hi"), ("user_2", "user", "again")))
    )
    create_event = await session._msg_ch.recv()
    assert create_event["item"]["id"] == "user_2"

    # Queued before the error lands, so it runs before the first call reads the slot.
    second_task = asyncio.create_task(session.update_chat_ctx(_ctx(("user_1", "user", "hi"))))
    try:
        _server_event(session, _invalid_prev("user_1", item_id="user_2"))
        with pytest.raises(llm.RealtimeError, match="previous_item_id 'user_1' not found"):
            await asyncio.wait_for(first_task, timeout=0.5)
    finally:
        await utils.aio.cancel_and_wait(second_task)


# --- connection lifecycle (real connection loop) -------------------------------


@pytest.mark.parametrize(
    ("close_message", "detail"),
    [
        pytest.param(None, "closed unexpectedly", id="no-close-code"),
        pytest.param(_ws_close(3000, "Invalid API key"), "close_code=3000", id="close-code"),
    ],
)
async def test_boson_realtime_websocket_close_fails_pending_generate_reply(
    socket_session, close_message, detail
):
    ws = _FakeWebSocket(close_message)
    session = socket_session(_SequencedHTTPClient([ws]), max_retry=0)
    errors = _collect(session, "error")

    generation_fut = session.generate_reply()
    assert await _eventually(lambda: _sent_response_create(ws))
    ws.close_event.set()

    # The point is that the future resolves rather than times out; the close detail
    # travels on the error event.
    with pytest.raises(llm.RealtimeError):
        await asyncio.wait_for(generation_fut, timeout=1.0)
    assert session._closed is True
    queued_before = session._msg_ch.qsize()
    session.send_event({"type": "input_audio_buffer.clear"})
    assert session._msg_ch.qsize() == queued_before
    assert errors
    assert errors[0].recoverable is False
    assert detail in str(errors[0].error)


_IDLE_TIMEOUT = {"type": "session.idle_timeout", "seconds_idle": 120}
_MAX_DURATION_REACHED = {"type": "session.max_duration_reached", "max_duration_sec": 7200}


@pytest.mark.parametrize(
    ("outcomes", "connects", "recoverable", "detail"),
    [
        # 4429 is a permanent billing refusal; reconnecting cannot fix it.
        pytest.param(
            lambda: [_dropped_ws(close_message=_ws_close(4429, "insufficient_quota"))],
            1,
            [False],
            "close_code=4429",
            id="billing-refusal-close-code",
        ),
        pytest.param(
            lambda: [_dropped_ws(close_message=_ws_close(3000, "Invalid API key"))],
            1,
            [False],
            "close_code=3000",
            id="invalid-api-key-close-code",
        ),
        # The server announces this and then closes: it reclaimed a session
        # nobody was using, so there is no reconnect.
        pytest.param(
            lambda: [_dropped_ws([_ws_text(_IDLE_TIMEOUT)])],
            1,
            [False],
            "idle timeout",
            id="idle-timeout",
        ),
        # The initial attempt plus max_retry=2, then give up.
        pytest.param(
            lambda: [aiohttp.ClientConnectionError("refused") for _ in range(3)],
            3,
            [True, True, False],
            "",
            id="retries-exhausted",
        ),
    ],
)
async def test_boson_realtime_terminal_connection_failure_closes_the_session(
    socket_session, outcomes, connects, recoverable, detail
):
    # A spare connection follows each case, so an unwanted reconnect would show up
    # in connect_urls.
    client = _SequencedHTTPClient([*outcomes(), _FakeWebSocket()])
    session = socket_session(client, max_retry=2)
    errors = _collect(session, "error")

    assert await _eventually(lambda: session._closed)
    assert len(client.connect_urls) == connects
    assert [e.recoverable for e in errors] == recoverable
    assert detail in str(errors[-1].error)
    assert session._msg_ch.closed


async def test_boson_realtime_reconnect_replays_chat_ctx_after_ws_drop(socket_session):
    # The first connection delivers a user turn and a tool call with its result,
    # then drops. The server keeps no state across connections, so all of it is
    # replayed; a replay without the call would let the model repeat it.
    call = _item("call_item_1", "function_call", call_id="call_1", name="lookup_weather")
    call["arguments"] = '{"location": "Paris"}'
    output = _item("output_item_1", "function_call_output", call_id="call_1", output="sunny")
    items = [
        (_item("user_1", role="user", content=[{"type": "input_text", "text": "hello"}]), None),
        (call, "user_1"),
        (output, "call_item_1"),
    ]
    ws2 = _FakeWebSocket()
    client = _SequencedHTTPClient(
        [_dropped_ws([_ws_text(_item_added_echo(item, prev)) for item, prev in items]), ws2]
    )
    session = socket_session(client)
    # A response active when the socket drops never gets its response.done, so its
    # id must not carry over as current.
    session._current_response_id = "resp_stale"
    errors = _collect(session, "error")
    reconnected = _collect(session, "session_reconnected")

    assert await _eventually(lambda: reconnected)
    assert len(client.connect_urls) == 2
    assert session._current_response_id is None

    # The session config is re-sent first on the new connection, then the
    # local chat-context mirror is replayed via conversation.item.create.
    sent = [json.loads(s) for s in ws2.sent]
    assert sent[0]["type"] == "session.update"
    replayed = [e["item"] for e in sent if e["type"] == "conversation.item.create"]
    assert [item["id"] for item in replayed] == ["user_1", "call_item_1", "output_item_1"]
    assert replayed[0]["content"] == [{"type": "input_text", "text": "hello"}]
    assert replayed[1]["type"] == "function_call"
    assert replayed[1]["name"] == "lookup_weather"
    assert replayed[1]["arguments"] == '{"location": "Paris"}'
    assert replayed[2]["type"] == "function_call_output"
    assert replayed[2]["output"] == "sunny"

    # The drop surfaced as a recoverable error and the session stayed usable.
    assert errors
    assert errors[0].recoverable is True
    assert session._closed is False
    assert not session._msg_ch.closed


async def test_boson_realtime_recycles_the_connection_after_max_session_duration(socket_session):
    ws1 = _FakeWebSocket()  # stays open: only the timer ends it
    ws2 = _FakeWebSocket()
    client = _SequencedHTTPClient([ws1, ws2])
    session = socket_session(client, max_session_duration=0.05)
    errors = _collect(session, "error")
    reconnected: list = []

    def on_reconnected(event):
        # One recycle is enough; stop the timer before the new connection runs.
        session._opts.max_session_duration = None
        reconnected.append(event)

    session.on("session_reconnected", on_reconnected)

    await session.update_instructions("Be thorough.")
    assert await _eventually(lambda: reconnected)
    assert ws1.closed
    assert len(client.connect_urls) == 2
    # A planned recycle, not a failure.
    assert errors == []
    # The new connection holds nothing, so it gets the whole config again.
    reconnect_update = json.loads(ws2.sent[0])
    assert reconnect_update["type"] == "session.update"
    assert reconnect_update["session"]["instructions"] == "Be thorough."


async def test_boson_realtime_server_max_duration_recycles_the_connection(socket_session):
    # The server's cap bounds one connection, not the conversation. The close that
    # follows its notice is recycled like max_session_duration: no error, and no
    # retry spent, so it reconnects even with max_retry=0.
    user = _item("user_1", role="user", content=[{"type": "input_text", "text": "hello"}])
    ws2 = _FakeWebSocket()
    client = _SequencedHTTPClient(
        [_dropped_ws([_ws_text(_item_added_echo(user)), _ws_text(_MAX_DURATION_REACHED)]), ws2]
    )
    session = socket_session(client, max_retry=0)
    errors = _collect(session, "error")
    reconnected = _collect(session, "session_reconnected")

    assert await _eventually(lambda: reconnected)
    assert len(client.connect_urls) == 2
    assert errors == []
    assert session._closed is False
    sent = [json.loads(s) for s in ws2.sent]
    assert sent[0]["type"] == "session.update"
    assert [e["item"]["id"] for e in sent if e["type"] == "conversation.item.create"] == ["user_1"]


async def test_boson_realtime_aclose_does_not_report_the_main_task_error_twice(monkeypatch):
    # aclose cancels the main task without reading its outcome, which looks like it
    # would make asyncio report "Task exception was never retrieved" for an error
    # already emitted. Task.cancel() clears that flag first, so it does not; this
    # asserts the missing report, not the flag.
    async def _failing_run(self):
        raise APIError("refused service", retryable=False)

    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _failing_run)

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    reports: list[str] = []
    loop.set_exception_handler(lambda _loop, ctx: reports.append(ctx.get("message", "")))
    try:
        model = _model()
        session = model.session()
        assert await _eventually(session._main_atask.done)

        await session.aclose()
        await model.aclose()

        del session, model
        gc.collect()
        assert [r for r in reports if "never retrieved" in r] == []
    finally:
        loop.set_exception_handler(previous_handler)
