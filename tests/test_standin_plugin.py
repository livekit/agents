"""Unit tests for the StandIn plugin: CallInfo, the TeamsCall attachment, the
chat schema, and the chat-channel handshake.

Hermetic. Nothing here dials StandIn, LiveKit, or the network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from livekit.plugins.standin import (
    SCHEMA_VERSION,
    TOPIC_CONTEXT,
    TOPIC_GOODBYE,
    CallInfo,
    ChatChannel,
    InboundMessage,
    StandInError,
    TeamsCall,
    build_reply,
    parse_inbound,
)
from livekit.plugins.standin._hmac import sign_handshake

pytestmark = pytest.mark.unit

SECRET = "test-secret"
NOW = 1_700_000_000_000


def _job(metadata: str) -> object:
    return SimpleNamespace(job=SimpleNamespace(metadata=metadata), room=None)


def _teams_job(**over: object) -> object:
    # The exact metadata shape StandIn dispatches with: source, caller_name,
    # tenant_id, call_direction, and user_id only when Teams reported an AAD id.
    data: dict[str, object] = {
        "source": "msteams",
        "caller_name": "Sara",
        "tenant_id": "t-1",
        "call_direction": "inbound",
        "user_id": "aad-1",
    }
    data.update(over)
    return _job(json.dumps(data))


# ── CallInfo: what the entrypoint reads ────────────────────────────────────────


def test_call_info_reads_the_dispatch_metadata() -> None:
    call = CallInfo.from_job(_teams_job(call_direction="outbound"))
    assert call.is_teams_call
    assert (call.caller_name, call.tenant_id, call.user_id) == ("Sara", "t-1", "aad-1")
    assert call.direction == "outbound"


def test_call_info_defaults_direction_to_inbound() -> None:
    call = CallInfo.from_job(_teams_job(call_direction=""))
    assert call.direction == "inbound"


def test_call_info_tolerates_a_missing_user_id() -> None:
    # StandIn includes user_id only when Teams reports an AAD id; guests and
    # anonymous callers arrive without one.
    job = _job(json.dumps({"source": "msteams", "caller_name": "caller", "tenant_id": "t"}))
    call = CallInfo.from_job(job)
    assert call.is_teams_call
    assert call.user_id == ""


@pytest.mark.parametrize(
    "metadata",
    [
        "",  # dispatched with no metadata at all
        "not json",  # someone else's dispatcher
        "[]",  # valid json, wrong shape
        '{"source": "sip", "caller_name": "x"}',  # another source entirely
    ],
)
def test_call_info_never_raises_on_a_foreign_job(metadata: str) -> None:
    # One worker can serve Teams, SIP and web rooms. A job from elsewhere must
    # read as "not a Teams call", never take the worker down.
    call = CallInfo.from_job(_job(metadata))
    assert not call.is_teams_call


def test_call_info_ignores_non_string_fields() -> None:
    call = CallInfo.from_job(_teams_job(caller_name=42))
    assert call.caller_name == ""


# ── TeamsCall: attaching to the dispatched job ─────────────────────────────────


class _FakeRoom:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def on(self, event: str, handler: object) -> None:
        self.handlers[event] = handler

    def emit_data(self, topic: str, payload: dict[str, object]) -> None:
        packet = SimpleNamespace(topic=topic, data=json.dumps(payload).encode())
        self.handlers["data_received"](packet)  # type: ignore[operator]


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def interrupt(self) -> None:
        self.calls.append(("interrupt", ""))

    def say(self, text: str) -> None:
        self.calls.append(("say", text))


async def test_teams_call_refuses_a_job_it_was_not_dispatched_for() -> None:
    with pytest.raises(StandInError, match="not dispatched by StandIn"):
        await TeamsCall().start(_FakeSession(), ctx=_job(""))  # type: ignore[arg-type]


async def test_teams_call_attaches_and_returns_the_caller() -> None:
    room = _FakeRoom()
    call = await TeamsCall().start(_FakeSession(), ctx=_teams_job(), room=room)  # type: ignore[arg-type]
    assert call.caller_name == "Sara"
    assert "data_received" in room.handlers  # the topics are wired


async def test_goodbye_interrupts_the_turn_and_speaks_the_line() -> None:
    # The default handler. Interrupting matters: the call is torn down shortly
    # after the goodbye, so one queued behind a long answer is never heard.
    room, session = _FakeRoom(), _FakeSession()
    await TeamsCall().start(session, ctx=_teams_job(), room=room)  # type: ignore[arg-type]
    room.emit_data(TOPIC_GOODBYE, {"text": "Wrapping up now."})
    assert session.calls == [("interrupt", ""), ("say", "Wrapping up now.")]


async def test_goodbye_handler_override_replaces_the_default() -> None:
    room, session = _FakeRoom(), _FakeSession()
    heard: list[str] = []
    await TeamsCall(on_goodbye=heard.append).start(session, ctx=_teams_job(), room=room)  # type: ignore[arg-type]
    room.emit_data(TOPIC_GOODBYE, {"text": "bye"})
    assert heard == ["bye"]
    assert session.calls == []  # the default did not also run


async def test_context_reaches_the_handler_as_text() -> None:
    # StandIn publishes context as {"text": ...} prose, not a structured object.
    room = _FakeRoom()
    seen: list[str] = []
    await TeamsCall(on_context=seen.append).start(_FakeSession(), ctx=_teams_job(), room=room)  # type: ignore[arg-type]
    room.emit_data(TOPIC_CONTEXT, {"text": "participants: 3"})
    assert seen == ["participants: 3"]


async def test_unrelated_topics_and_malformed_payloads_are_ignored() -> None:
    room, session = _FakeRoom(), _FakeSession()
    await TeamsCall().start(session, ctx=_teams_job(), room=room)  # type: ignore[arg-type]
    room.emit_data("lk.transcription", {"text": "not ours"})
    room.emit_data(TOPIC_GOODBYE, {"no": "text"})
    handler = room.handlers["data_received"]
    handler(SimpleNamespace(topic=TOPIC_GOODBYE, data=b"not json"))  # type: ignore[operator]
    assert session.calls == []


async def test_info_is_available_after_start_and_not_before() -> None:
    tc = TeamsCall()
    with pytest.raises(StandInError, match="has not run yet"):
        _ = tc.info
    await tc.start(_FakeSession(), ctx=_teams_job(), room=_FakeRoom())  # type: ignore[arg-type]
    assert tc.info.caller_name == "Sara"


# ── the embedded listener's teardown ───────────────────────────────────────────


async def test_end_soon_teardown_does_not_deadlock_on_itself() -> None:
    # _end_soon spawns aclose() into the same task set aclose cancels and
    # awaits. Without excluding the current task, that is a task awaiting its
    # own completion: the call hangs half-closed forever and the slot is never
    # released. This is exactly the room-deleted / agent-left teardown path.
    import asyncio

    from livekit.plugins.standin.bridge import _Call

    bridge = SimpleNamespace(_release=lambda _cid: None, audio_idle_timeout=0)
    ws = SimpleNamespace(closed=True)
    call = _Call(bridge, "c1", ws)  # type: ignore[arg-type]

    call._end_soon("room-disconnected")
    async with asyncio.timeout(2):
        while not call._closed or call._tasks:
            await asyncio.sleep(0.01)
    assert call._closed


# ── resource release and replay-window retention (upstream review) ─────────────


async def test_teardown_closes_the_rtc_primitives() -> None:
    # rtc.AudioStream and rtc.AudioSource each own an FFI subscription and an
    # internal task that only aclose() releases. Cancelling the pump task frees
    # neither, so without this every finished call leaks one of each for the
    # life of the worker.
    from livekit.plugins.standin.bridge import _Call

    closed: list[str] = []

    class _Closable:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            closed.append(self.name)

    bridge = SimpleNamespace(_release=lambda _cid: None, audio_idle_timeout=0)
    call = _Call(bridge, "c1", SimpleNamespace(closed=True))  # type: ignore[arg-type]
    call._audio_stream = _Closable("stream")  # type: ignore[assignment]
    call._source = _Closable("source")  # type: ignore[assignment]

    await call.aclose()
    assert sorted(closed) == ["source", "stream"]
    assert call._audio_stream is None and call._source is None


def test_replay_entries_outlive_a_future_dated_signature() -> None:
    # verify_handshake accepts a timestamp up to REPLAY_WINDOW_MS in the FUTURE,
    # so an entry aged from ARRIVAL would be pruned while its signature was
    # still valid, reopening the replay. Aged from the signing time, retention
    # matches validity exactly.
    from livekit.plugins.standin._hmac import REPLAY_WINDOW_MS, verify_handshake

    signed_at = NOW + REPLAY_WINDOW_MS  # the furthest-future timestamp accepted
    sig = sign_handshake(SECRET, signed_at, "c1")
    assert verify_handshake(SECRET, str(signed_at), "c1", sig, NOW)

    # Latest moment the signature still verifies, i.e. the entry must survive.
    last_valid = signed_at + REPLAY_WINDOW_MS
    assert verify_handshake(SECRET, str(signed_at), "c1", sig, last_valid)
    assert signed_at >= last_valid - REPLAY_WINDOW_MS, "prune cutoff would drop a live entry"

    # One millisecond later the signature is dead, so dropping it is correct.
    assert not verify_handshake(SECRET, str(signed_at), "c1", sig, last_valid + 1)


# ── chat channel ───────────────────────────────────────────────────────────────


def test_chat_channel_dials_out_and_needs_a_secret() -> None:
    async def respond(_: InboundMessage) -> str:
        return ""

    with pytest.raises(StandInError, match="secret"):
        ChatChannel(respond=respond, secret="")
    channel = ChatChannel(respond=respond, secret=SECRET)
    # Nothing listens: the channel is dialed out, like the call media.
    assert not hasattr(channel, "path")


def test_handshake_signature_is_over_timestamp_dot_id() -> None:
    # Pins the wire construction, so the StandIn side can verify against the
    # same vector: lowercase-hex HMAC-SHA256(secret, "{timestampMs}.{id}").
    import hashlib
    import hmac as _hmac

    expected = _hmac.new(SECRET.encode(), f"{NOW}.chat".encode(), hashlib.sha256).hexdigest()
    assert sign_handshake(SECRET, NOW, "chat") == expected


# ── chat wire schema ───────────────────────────────────────────────────────────


def _inbound(**over: object) -> str:
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "tenantId": "t-1",
        "conversationId": "c-1",
        "activityId": "a-1",
        "scope": "personal",
        "text": "hello",
        "sender": {"displayName": "Sara", "aadObjectId": "aad-1", "isLinkedOwner": True},
    }
    body.update(over)
    return json.dumps(body)


def test_parse_inbound_reads_the_message() -> None:
    msg = parse_inbound(_inbound())
    assert msg.tenant_id == "t-1"
    assert msg.text == "hello"
    assert msg.sender_name == "Sara"
    assert msg.sender_is_linked_owner is True
    assert msg.is_personal


@pytest.mark.parametrize("missing", ["tenantId", "conversationId", "activityId"])
def test_parse_inbound_requires_every_routing_key(missing: str) -> None:
    body = json.loads(_inbound())
    del body[missing]
    with pytest.raises(ValueError, match=missing):
        parse_inbound(json.dumps(body))


def test_parse_inbound_refuses_a_future_major_schema() -> None:
    # Additive evolution does NOT bump schemaVersion, so a higher integer means
    # semantics we would misread rather than fields we can ignore.
    with pytest.raises(ValueError, match="schemaVersion"):
        parse_inbound(_inbound(schemaVersion=SCHEMA_VERSION + 1))


def test_parse_inbound_tolerates_an_unknown_scope() -> None:
    # ChatScope is an OPEN enum: unknown values must relay, not fail.
    assert parse_inbound(_inbound(scope="sharedChannel")).scope == "sharedChannel"


def test_parse_inbound_survives_a_message_that_is_only_attachments() -> None:
    msg = parse_inbound(_inbound(text="", attachments=[{"kind": "image", "relayable": True}]))
    assert msg.text == ""
    assert len(msg.attachments) == 1


def test_reply_echoes_tenant_and_conversation_exactly() -> None:
    # The gateway rejects a mismatch; that check is the cross-tenant leak guard.
    msg = parse_inbound(_inbound())
    reply = build_reply(msg, "hi there")
    assert reply["tenantId"] == msg.tenant_id
    assert reply["conversationId"] == msg.conversation_id
    assert reply["replyToId"] == msg.activity_id
    assert reply["text"] == "hi there"
    assert reply["idempotencyKey"] == "a-1:message"


def test_typing_reply_carries_no_text() -> None:
    reply = build_reply(parse_inbound(_inbound()), "ignored", "typing")
    assert "text" not in reply
    assert reply["kind"] == "typing"


def test_reply_idempotency_key_separates_kinds() -> None:
    # A typing indicator and its answer share an activityId; if they shared an
    # idempotency key the gateway would drop the answer as a duplicate.
    msg = parse_inbound(_inbound())
    assert (
        build_reply(msg, "x")["idempotencyKey"] != build_reply(msg, "", "typing")["idempotencyKey"]
    )
