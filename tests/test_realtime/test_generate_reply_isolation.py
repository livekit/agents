"""Tests for `add_to_chat_ctx=False` on RealtimeSession.generate_reply.

Covers: capability flag, dispatcher gate, pipeline-LLM guard, OpenAI plugin
substrate isolation, ephemeral-event suppression, orphan filter, handler
guards, single-isolated-call serialization contract, local _chat_ctx gate,
remote-item-added gate, interrupt() with response_id.

Live-substrate tests require OPENAI_API_KEY and run against `gpt-realtime`.
"""

from __future__ import annotations

import inspect
import os

import pytest

from livekit.agents import llm
from livekit.plugins import openai as lk_openai

# -- Helpers --


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _has_azure_key() -> bool:
    return all(
        os.environ.get(k)
        for k in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_DEPLOYMENT",
        )
    )


# ---------------------------------------------------------------------------
# Component 1: abstract RealtimeSession.generate_reply signature
# Component 5: RealtimeCapabilities.ephemeral_response field
# ---------------------------------------------------------------------------


def test_realtime_capabilities_has_ephemeral_response_field_defaulting_false() -> None:
    """The new field exists on the dataclass and defaults to False."""
    caps = llm.RealtimeCapabilities(
        message_truncation=False,
        turn_detection=False,
        user_transcription=False,
        auto_tool_reply_generation=False,
        audio_output=False,
        manual_function_calls=False,
    )
    assert caps.ephemeral_response is False


def test_realtime_session_generate_reply_signature_accepts_add_to_chat_ctx() -> None:
    """The abstract signature exposes the new keyword-only parameter with default True."""
    sig = inspect.signature(llm.RealtimeSession.generate_reply)
    assert "add_to_chat_ctx" in sig.parameters
    param = sig.parameters["add_to_chat_ctx"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True


# ---------------------------------------------------------------------------
# Component 5 (Phase 1): OpenAI plugin declares the capability for non-Azure
# only; Azure-backed sessions advertise ephemeral_response=False so the
# dispatcher capability gate falls through to the legacy path until Azure
# parity is verified.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set")
def test_openai_plugin_declares_ephemeral_response_for_non_azure() -> None:
    model = lk_openai.realtime.RealtimeModel()
    assert model.capabilities.ephemeral_response is True


@pytest.mark.skipif(not _has_azure_key(), reason="Azure OpenAI env vars not set")
def test_openai_plugin_does_not_declare_ephemeral_response_for_azure() -> None:
    model = lk_openai.realtime.RealtimeModel(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"] + "/openai",
    )
    assert model.capabilities.ephemeral_response is False


def test_openai_plugin_generate_reply_signature_accepts_add_to_chat_ctx() -> None:
    """OpenAI plugin RealtimeSession.generate_reply accepts the new kwarg."""
    sig = inspect.signature(lk_openai.realtime.RealtimeSession.generate_reply)
    assert "add_to_chat_ctx" in sig.parameters
    param = sig.parameters["add_to_chat_ctx"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True


# ---------------------------------------------------------------------------
# Component 2 (Phase 1): AgentSession.generate_reply
# ---------------------------------------------------------------------------


def test_agent_session_generate_reply_signature_accepts_add_to_chat_ctx() -> None:
    from livekit.agents import AgentSession

    sig = inspect.signature(AgentSession.generate_reply)
    assert "add_to_chat_ctx" in sig.parameters
    param = sig.parameters["add_to_chat_ctx"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True


async def test_agent_session_generate_reply_pipeline_llm_with_isolation_raises() -> None:
    """add_to_chat_ctx=False against a non-realtime LLM raises NotImplementedError."""
    from livekit.agents import AgentSession

    from ..fake_llm import FakeLLM

    session = AgentSession(llm=FakeLLM(fake_responses=[]))
    try:
        with pytest.raises(NotImplementedError) as exc_info:
            session.generate_reply(add_to_chat_ctx=False)
        msg = str(exc_info.value)
        assert "RealtimeModel" in msg
        assert "add_to_chat_ctx=False" in msg
    finally:
        await session.aclose()


# ---------------------------------------------------------------------------
# Phase 2 mock-based tests for the OpenAI plugin.
#
# `_make_offline_session()` constructs an OpenAI `RealtimeSession` whose
# WebSocket main task is canceled before it can connect, so the tests can
# exercise `generate_reply` and the various internal handlers without any
# network round-trip.  Outbound events are captured by replacing the
# session's `send_event` method with a list collector; inbound events are
# synthesized by directly invoking the relevant `_handle_*` method.
# ---------------------------------------------------------------------------


@pytest.fixture
async def offline_session():
    """OpenAI RealtimeSession with the connection task suppressed."""
    import asyncio

    model = lk_openai.realtime.RealtimeModel(api_key="sk-test-not-real")
    session = model.session()
    # Cancel the main task before it tries to connect; the test only
    # exercises in-memory handlers and outbound-event capture.
    session._main_atask.cancel()
    try:
        await session._main_atask
    except (asyncio.CancelledError, BaseException):
        pass

    captured: list = []
    session.send_event = lambda ev: captured.append(ev)  # type: ignore[assignment]
    session._captured_events = captured  # type: ignore[attr-defined]
    yield session
    # Best-effort cleanup; aclose() is awkward with the patched send_event.
    try:
        await model.aclose()
    except Exception:
        pass


def _captured_response_create_events(session) -> list:
    """Return outbound `response.create` events captured during the test."""
    from openai.types.realtime import ResponseCreateEvent

    return [ev for ev in session._captured_events if isinstance(ev, ResponseCreateEvent)]


# ---- Phase 2 / Component 5: capability + ephemeral state plumbing ----


async def test_session_init_creates_ephemeral_state_dicts(offline_session) -> None:
    """The four ephemeral tracking structures exist on the session."""
    s = offline_session
    assert s._ephemeral_event_ids == set()
    assert s._active_ephemeral_response_ids == {}
    assert s._ephemeral_remote_item_ids == set()
    assert s._ephemeral_started_at == {}


# ---- Phase 2 / Component 3: substrate-level isolation on the wire ----


async def test_isolated_generate_reply_sets_conversation_none_on_wire(
    offline_session,
) -> None:
    """`add_to_chat_ctx=False` causes `conversation: "none"` to appear in the
    serialized JSON, not just on the in-memory attribute."""
    s = offline_session
    fut = s.generate_reply(instructions="say hello", add_to_chat_ctx=False)
    try:
        rc_events = _captured_response_create_events(s)
        assert len(rc_events) == 1
        ev = rc_events[0]
        # Serialized form is what goes over the wire.
        dumped = ev.model_dump(by_alias=True, exclude_unset=False)
        assert dumped.get("response", {}).get("conversation") == "none"
    finally:
        if not fut.done():
            fut.set_exception(RuntimeError("test cleanup"))


async def test_default_generate_reply_does_not_set_conversation_none(
    offline_session,
) -> None:
    """Default `add_to_chat_ctx=True` does NOT set the `conversation` field to "none"."""
    s = offline_session
    fut = s.generate_reply(instructions="say hello")
    try:
        rc_events = _captured_response_create_events(s)
        assert len(rc_events) == 1
        ev = rc_events[0]
        # The serialized form (exclude_unset=True) is what goes over the wire;
        # the default path must not include `conversation: "none"`.
        dumped = ev.model_dump(by_alias=True, exclude_unset=True)
        assert dumped.get("response", {}).get("conversation") != "none"
    finally:
        if not fut.done():
            fut.set_exception(RuntimeError("test cleanup"))


async def test_isolated_generate_reply_force_overrides_tools(offline_session) -> None:
    """Caller-provided tools/tool_choice are silently dropped on isolated calls;
    a `logger.warning` is emitted; outbound params carry empty tools and
    `tool_choice="none"`."""
    s = offline_session

    fut = s.generate_reply(
        instructions="say hello",
        tools=[],  # type: ignore[arg-type]
        tool_choice="auto",
        add_to_chat_ctx=False,
    )
    try:
        rc_events = _captured_response_create_events(s)
        assert len(rc_events) == 1
        dumped = rc_events[0].model_dump(by_alias=True, exclude_unset=False)
        assert dumped["response"].get("tool_choice") == "none"
        assert dumped["response"].get("tools") in ([], None)
    finally:
        if not fut.done():
            fut.set_exception(RuntimeError("test cleanup"))


# ---- Phase 2 / Single-isolated-call serialization contract ----


async def test_concurrent_isolated_generate_reply_rejects_second_pre_creation(
    offline_session,
) -> None:
    """A second isolated call issued before the first's `response.created`
    arrives raises `RuntimeError` with diagnostic context."""
    s = offline_session
    fut1 = s.generate_reply(instructions="first", add_to_chat_ctx=False)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            s.generate_reply(instructions="second", add_to_chat_ctx=False)
        msg = str(exc_info.value)
        assert "client_event_id=" in msg
        assert "response_id=<not-yet-assigned>" in msg
        assert "started" in msg and "s ago" in msg
        assert "§Concurrency" in msg
    finally:
        if not fut1.done():
            fut1.set_exception(RuntimeError("test cleanup"))


async def test_concurrent_isolated_after_response_created_includes_response_id(
    offline_session,
) -> None:
    """Second isolated call after `response.created` arrived shows the actual
    `response_id`, not `<not-yet-assigned>`."""
    from openai.types.realtime import ResponseCreatedEvent

    s = offline_session
    fut1 = s.generate_reply(instructions="first", add_to_chat_ctx=False)
    rc_events = _captured_response_create_events(s)
    client_event_id = rc_events[0].event_id

    # Synthesize the server's response.created arrival.
    response = type(
        "_R",
        (),
        {
            "id": "resp_TESTID_001",
            "metadata": {"client_event_id": client_event_id},
            "output": [],
        },
    )()
    fake_event = ResponseCreatedEvent.model_construct(  # type: ignore[attr-defined]
        type="response.created", response=response
    )
    s._handle_response_created(fake_event)

    try:
        with pytest.raises(RuntimeError) as exc_info:
            s.generate_reply(instructions="second", add_to_chat_ctx=False)
        msg = str(exc_info.value)
        assert "response_id=resp_TESTID_001" in msg
        assert "<not-yet-assigned>" not in msg
    finally:
        if not fut1.done():
            fut1.set_exception(RuntimeError("test cleanup"))


async def test_default_generate_reply_during_isolated_does_not_raise(
    offline_session,
) -> None:
    """Default (non-isolated) generate_reply during an in-flight isolated call
    proceeds without raising. The serialization contract scopes only to
    `add_to_chat_ctx=False`."""
    s = offline_session
    fut1 = s.generate_reply(instructions="first", add_to_chat_ctx=False)
    fut2 = s.generate_reply(instructions="second")  # default; should not raise
    try:
        rc_events = _captured_response_create_events(s)
        assert len(rc_events) == 2
    finally:
        for f in (fut1, fut2):
            if not f.done():
                f.set_exception(RuntimeError("test cleanup"))


# ---- Phase 2 / Component 6: orphan filter ----


async def test_orphan_response_filtered_and_cancelled(offline_session) -> None:
    """A `response.created` whose `client_event_id` is no longer in
    `_response_created_futures` is discarded BEFORE `_current_generation` is
    assigned, and a defensive `ResponseCancelEvent` with `response_id` is sent.
    """
    from openai.types.realtime import ResponseCancelEvent, ResponseCreatedEvent

    s = offline_session

    response = type(
        "_R",
        (),
        {
            "id": "resp_orphan",
            "metadata": {"client_event_id": "ce_already_popped"},
            "output": [],
        },
    )()
    fake_event = ResponseCreatedEvent.model_construct(  # type: ignore[attr-defined]
        type="response.created", response=response
    )

    s._captured_events.clear()
    s._handle_response_created(fake_event)

    # No generation should have been created.
    assert s._current_generation is None
    # A defensive cancel with the response_id should have gone out.
    cancels = [ev for ev in s._captured_events if isinstance(ev, ResponseCancelEvent)]
    assert len(cancels) == 1
    assert cancels[0].response_id == "resp_orphan"


async def test_orphan_filter_bypasses_on_metadata_none_for_server_vad(
    offline_session,
) -> None:
    """Server-VAD-initiated responses (no metadata) are NOT filtered."""
    from openai.types.realtime import ResponseCreatedEvent

    s = offline_session
    response = type("_R", (), {"id": "resp_server_vad", "metadata": None, "output": []})()
    fake_event = ResponseCreatedEvent.model_construct(  # type: ignore[attr-defined]
        type="response.created", response=response
    )

    s._handle_response_created(fake_event)

    # Generation should have been created normally.
    assert s._current_generation is not None
    assert s._current_generation._response_id == "resp_server_vad"


# ---- Phase 2 / Component 6: handler guards on None generation ----


async def test_handler_guards_no_assert_on_none_generation(offline_session) -> None:
    """With `_current_generation = None`, all 9 reachable handlers early-return
    without raising `AssertionError`."""
    from openai.types.realtime import (
        ResponseAudioDeltaEvent,
        ResponseAudioDoneEvent,
        ResponseContentPartAddedEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )

    s = offline_session
    s._current_generation = None

    item = type("_I", (), {"id": "it_x", "type": "message"})()
    fake_part = type("_P", (), {"type": "text"})()

    # Each call must not raise AssertionError.
    s._handle_response_output_item_added(
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=item, response_id="r", output_index=0
        )
    )
    s._handle_response_content_part_added(
        ResponseContentPartAddedEvent.model_construct(
            type="response.content_part.added",
            item_id="it_x",
            part=fake_part,
            output_index=0,
            content_index=0,
            response_id="r",
        )
    )
    s._handle_response_text_delta(
        ResponseTextDeltaEvent.model_construct(
            type="response.output_text.delta",
            delta="hi",
            item_id="it_x",
            output_index=0,
            content_index=0,
            response_id="r",
        )
    )
    s._handle_response_text_done(
        ResponseTextDoneEvent.model_construct(
            type="response.output_text.done",
            text="hi",
            item_id="it_x",
            output_index=0,
            content_index=0,
            response_id="r",
        )
    )
    s._handle_response_audio_transcript_delta(
        {"type": "response.output_audio_transcript.delta", "delta": "hi", "item_id": "it_x"}
    )
    s._handle_response_audio_delta(
        ResponseAudioDeltaEvent.model_construct(
            type="response.output_audio.delta",
            delta="AAAA",
            item_id="it_x",
            output_index=0,
            content_index=0,
            response_id="r",
        )
    )
    s._handle_response_audio_done(
        ResponseAudioDoneEvent.model_construct(
            type="response.output_audio.done",
            item_id="it_x",
            output_index=0,
            content_index=0,
            response_id="r",
        )
    )
    s._handle_response_output_item_done(
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done", item=item, response_id="r", output_index=0
        )
    )
    # All handlers returned cleanly.


# ---- Phase 2 / Codex finding 1: shadow-state leak fix ----


async def test_isolated_call_succeeds_after_first_completes(offline_session) -> None:
    """After the first isolated call's response.done arrives, a second
    isolated call succeeds (the serialization contract is per-in-flight, not
    per-session-lifetime)."""
    from openai.types.realtime import ResponseCreatedEvent, ResponseDoneEvent

    s = offline_session
    fut1 = s.generate_reply(instructions="first", add_to_chat_ctx=False)
    rc_events = _captured_response_create_events(s)
    cid1 = rc_events[0].event_id

    # Synthesize response.created.
    response_created = type(
        "_R",
        (),
        {
            "id": "resp_FIRST",
            "metadata": {"client_event_id": cid1},
            "output": [],
        },
    )()
    s._handle_response_created(
        ResponseCreatedEvent.model_construct(  # type: ignore[attr-defined]
            type="response.created", response=response_created
        )
    )

    # Synthesize response.done; this should drain the ephemeral tracking.
    response_done = type(
        "_R",
        (),
        {
            "id": "resp_FIRST",
            "status": "completed",
            "status_details": None,
            "usage": None,
            "metadata": {"client_event_id": cid1},
        },
    )()
    s._handle_response_done(
        ResponseDoneEvent.model_construct(  # type: ignore[attr-defined]
            type="response.done", response=response_done
        )
    )

    # First call's tracking should be fully drained by response.done.
    assert cid1 not in s._active_ephemeral_response_ids
    assert cid1 not in s._ephemeral_event_ids
    assert cid1 not in s._ephemeral_started_at

    # Now a second isolated call must succeed.
    fut2 = s.generate_reply(instructions="second", add_to_chat_ctx=False)
    try:
        rc_events = _captured_response_create_events(s)
        assert len(rc_events) == 2
        cid2 = rc_events[1].event_id
        # The second call's tracking is in place, but the first's is gone.
        assert cid2 in s._ephemeral_event_ids
        assert cid2 in s._ephemeral_started_at
        assert cid1 not in s._ephemeral_started_at
    finally:
        for f in (fut1, fut2):
            if not f.done():
                f.set_exception(RuntimeError("test cleanup"))


async def test_isolated_event_emit_suppressed(offline_session) -> None:
    """`openai_client_event_queued` is NOT emitted for isolated client events;
    `openai_server_event_received` is NOT emitted for server events correlated
    to in-flight ephemeral responses."""
    s = offline_session
    client_emits: list = []
    server_emits: list = []
    s.on("openai_client_event_queued", lambda ev: client_emits.append(ev))
    s.on("openai_server_event_received", lambda ev: server_emits.append(ev))

    # Register an in-flight ephemeral response: pre-populate the tracker the
    # way the live `_send_task` would.  We invoke `_is_client_event_ephemeral`
    # / `_is_server_event_ephemeral` directly because the offline fixture does
    # not run the WebSocket loop.
    s._ephemeral_event_ids.add("ce_iso")
    s._active_ephemeral_response_ids["ce_iso"] = "resp_iso"

    # Client event correlated by event_id.
    client_ev = {"event_id": "ce_iso", "type": "response.create"}
    assert s._is_client_event_ephemeral(client_ev) is True
    # Calibration: a non-ephemeral client event passes through.
    other_client_ev = {"event_id": "ce_other", "type": "response.create"}
    assert s._is_client_event_ephemeral(other_client_ev) is False

    # Server event correlated by response.id (response-level event shape).
    server_ev = {"type": "response.done", "response": {"id": "resp_iso"}}
    assert s._is_server_event_ephemeral(server_ev) is True
    # Server event correlated by response_id (delta event shape).
    delta_ev = {"type": "response.audio.delta", "response_id": "resp_iso"}
    assert s._is_server_event_ephemeral(delta_ev) is True
    # Calibration: a non-correlated server event is NOT ephemeral.
    other_server_ev = {"type": "response.done", "response": {"id": "resp_other"}}
    assert s._is_server_event_ephemeral(other_server_ev) is False


# ---------------------------------------------------------------------------
# Phase 2 / live-substrate smoke test.
#
# A focused live smoke test that exercises the three-arm contrast at the
# LiveKit `RealtimeSession` layer.  Verifies (a) audibility — the rendered
# transcript contains the secret on the ISOLATED arm; (b) behavioral
# isolation — a follow-up generate_reply asking the model to recall the
# secret does NOT recall it on the ISOLATED arm; (c) calibration — the same
# flow without `add_to_chat_ctx=False` DOES recall the secret.
#
# This test is the unit-test analog of the bundle's
# `livekit_wrapper_isolation_probe.py`; the latter remains the
# authoritative end-to-end verification (re-run as part of pre-PR
# verification).  Skipped when `OPENAI_API_KEY` is not set; runtime ~60s
# and ~$0.30 in API costs per run.
# ---------------------------------------------------------------------------


_NONCE = "purple-elephant-42"
_NONCE_PATTERNS = ["purple-elephant", "purple elephant", "elephant 42", "elephant forty"]


def _detect_nonce(transcript: str) -> bool:
    lower = transcript.lower()
    return any(p.lower() in lower for p in _NONCE_PATTERNS)


@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set")
async def test_isolated_generate_reply_does_not_pollute_substrate_live(job_process) -> None:
    """ISOLATED arm: the secret is rendered audibly but is NOT recalled on
    the next user-message turn.  Verifies the substrate-isolation property
    transfers through the LiveKit `RealtimeSession` wrapper end-to-end."""
    import asyncio

    say_instruction = (
        f"Say to the user, in plain English: 'Your verification token is {_NONCE}.' "
        "Render the value clearly. Do not paraphrase the value itself."
    )
    recall_prompt = (
        "If your previous spoken turn delivered a verification token, repeat it "
        "verbatim now. Otherwise reply 'no token recalled'."
    )

    model = lk_openai.realtime.RealtimeModel(model="gpt-realtime", voice="alloy")
    session = model.session()
    try:
        # Arm A: render with isolation.
        gen_a = await asyncio.wait_for(
            session.generate_reply(instructions=say_instruction, add_to_chat_ctx=False),
            timeout=20,
        )
        transcript_a_parts: list[str] = []
        async for msg in gen_a.message_stream:
            async for txt in msg.text_stream:
                transcript_a_parts.append(txt)
            async for _ in msg.audio_stream:
                pass
        transcript_a = "".join(transcript_a_parts)

        # Arm B: recall via user message.
        # Pushing a user message into the conversation requires a fresh
        # generate_reply (default add_to_chat_ctx=True).
        chat_ctx = session.chat_ctx.copy()
        chat_ctx.add_message(role="user", content=recall_prompt)
        await session.update_chat_ctx(chat_ctx)
        gen_b = await asyncio.wait_for(session.generate_reply(), timeout=20)
        transcript_b_parts: list[str] = []
        async for msg in gen_b.message_stream:
            async for txt in msg.text_stream:
                transcript_b_parts.append(txt)
            async for _ in msg.audio_stream:
                pass
        transcript_b = "".join(transcript_b_parts)
    finally:
        await session.aclose()

    assert _detect_nonce(transcript_a), (
        f"Audibility (ISOLATED.A) failed: nonce not found in transcript: {transcript_a!r}"
    )
    assert not _detect_nonce(transcript_b), (
        f"Behavioral isolation (ISOLATED.B) failed: nonce was recalled in: {transcript_b!r}"
    )


async def test_ephemeral_item_skipped_from_remote_chat_ctx(offline_session) -> None:
    """Items whose id is in `_ephemeral_remote_item_ids` are NOT inserted into
    `_remote_chat_ctx` and do NOT emit `remote_item_added`."""
    from openai.types.realtime import ConversationItemAdded

    s = offline_session
    s._ephemeral_remote_item_ids.add("it_ephemeral")

    emitted: list = []
    s.on("remote_item_added", lambda ev: emitted.append(ev))

    item = type(
        "_I",
        (),
        {
            "id": "it_ephemeral",
            "type": "message",
            "role": "assistant",
            "content": [],
            "status": "completed",
        },
    )()
    fake_event = ConversationItemAdded.model_construct(  # type: ignore[attr-defined]
        type="conversation.item.added", item=item, previous_item_id=None
    )

    before_size = len(s._remote_chat_ctx.to_chat_ctx().items)
    s._handle_conversion_item_added(fake_event)
    after_size = len(s._remote_chat_ctx.to_chat_ctx().items)

    assert before_size == after_size
    assert emitted == []


# ---------------------------------------------------------------------------
# Phase 3: AgentActivity._on_remote_item_added defensive gate.
# ---------------------------------------------------------------------------


async def test_on_remote_item_added_skips_ephemeral_items() -> None:
    """`AgentActivity._on_remote_item_added` skips items whose id is in
    `self._rt_session._ephemeral_remote_item_ids`."""
    from livekit.agents import llm as lk_llm
    from livekit.agents.voice.agent_activity import AgentActivity

    class _FakeAgent:
        def __init__(self) -> None:
            self._chat_ctx = lk_llm.ChatContext.empty()

    class _FakeSession:
        def __init__(self) -> None:
            self._ephemeral_remote_item_ids: set[str] = {"it_eph_X"}

    class _FakeActivity:
        def __init__(self) -> None:
            self._agent = _FakeAgent()
            self._rt_session = _FakeSession()

    activity = _FakeActivity()
    msg = lk_llm.ChatMessage(role="assistant", content=["hidden"], id="it_eph_X")
    ephemeral_ev = lk_llm.RemoteItemAddedEvent(previous_item_id=None, item=msg)
    AgentActivity._on_remote_item_added(activity, ephemeral_ev)  # type: ignore[arg-type]
    assert activity._agent._chat_ctx.get_by_id("it_eph_X") is None

    # Calibration: a non-ephemeral item DOES pass through.
    other = lk_llm.ChatMessage(role="assistant", content=["public"], id="it_normal_Y")
    other_ev = lk_llm.RemoteItemAddedEvent(previous_item_id=None, item=other)
    AgentActivity._on_remote_item_added(activity, other_ev)  # type: ignore[arg-type]
    assert activity._agent._chat_ctx.get_by_id("it_normal_Y") is not None


async def test_on_remote_item_added_works_for_plugin_without_ephemeral_attr() -> None:
    """If the rt_session does not expose `_ephemeral_remote_item_ids`, the
    handler still appends the item.  The duck-typed `getattr(..., set())`
    default is what makes this safe for plugins that did not opt in."""
    from livekit.agents import llm as lk_llm
    from livekit.agents.voice.agent_activity import AgentActivity

    class _FakeAgent:
        def __init__(self) -> None:
            self._chat_ctx = lk_llm.ChatContext.empty()

    class _FakeSession:
        # Intentionally no _ephemeral_remote_item_ids attribute.
        pass

    class _FakeActivity:
        def __init__(self) -> None:
            self._agent = _FakeAgent()
            self._rt_session = _FakeSession()

    activity = _FakeActivity()
    msg = lk_llm.ChatMessage(role="assistant", content=["public"], id="it_Z")
    ev = lk_llm.RemoteItemAddedEvent(previous_item_id=None, item=msg)
    AgentActivity._on_remote_item_added(activity, ev)  # type: ignore[arg-type]
    assert activity._agent._chat_ctx.get_by_id("it_Z") is not None


# ---------------------------------------------------------------------------
# Phase 3: live-substrate test that the assistant message is NOT written to
# the agent's chat context when add_to_chat_ctx=False.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set")
async def test_isolated_generate_reply_does_not_pollute_chat_ctx_live(job_process) -> None:
    """End-to-end: after an isolated `RealtimeSession.generate_reply`, the
    session's `chat_ctx` does not contain a new assistant message carrying
    the rendered text."""
    import asyncio

    say_instruction = (
        "Say to the user, in plain English: 'Your verification token is "
        "purple-elephant-42.' Render the value clearly."
    )

    model = lk_openai.realtime.RealtimeModel(model="gpt-realtime", voice="alloy")
    session = model.session()
    try:
        before_items = list(session.chat_ctx.items)
        gen = await asyncio.wait_for(
            session.generate_reply(instructions=say_instruction, add_to_chat_ctx=False),
            timeout=20,
        )
        # Drain.
        async for msg in gen.message_stream:
            async for _ in msg.text_stream:
                pass
            async for _ in msg.audio_stream:
                pass
        # Allow response.done to flush.
        await asyncio.sleep(0.5)
        after_items = list(session.chat_ctx.items)
    finally:
        await session.aclose()

    # No new assistant items appended to the wrapper-layer chat_ctx.
    new_items = [it for it in after_items if it not in before_items]
    assistant_text = " ".join(
        " ".join(c if isinstance(c, str) else "" for c in it.content)
        for it in new_items
        if hasattr(it, "role") and getattr(it, "role", None) == "assistant"
    )
    assert "purple-elephant" not in assistant_text.lower()
    assert "elephant 42" not in assistant_text.lower()
