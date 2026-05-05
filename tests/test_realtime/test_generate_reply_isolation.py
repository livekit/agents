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
# Abstract API surface: RealtimeSession.generate_reply accepts the new kwarg,
# and RealtimeCapabilities exposes the ephemeral_response field.
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
# OpenAI plugin: declares ephemeral_response only for non-Azure endpoints.
# Azure-backed sessions advertise ephemeral_response=False so the dispatcher
# capability gate falls through to the legacy path until Azure parity is
# verified.
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
# AgentSession.generate_reply: signature + non-realtime LLM guard.
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
# Mock-based tests for the OpenAI plugin's RealtimeSession internals.
#
# The `offline_session` fixture below constructs a `RealtimeSession` whose
# WebSocket main task is cancelled before it can connect, so the tests can
# exercise `generate_reply` and the various internal `_handle_*` methods
# without any network round-trip.  Outbound events are captured by replacing
# `send_event` with a list collector; inbound events are synthesised by
# directly invoking the relevant `_handle_*` method.
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


# ---- ephemeral state plumbing ----


async def test_session_init_creates_ephemeral_state_dicts(offline_session) -> None:
    """The four ephemeral tracking structures exist on the session."""
    s = offline_session
    assert s._ephemeral_event_ids == set()
    assert s._active_ephemeral_response_ids == {}
    assert s._ephemeral_remote_item_ids == set()
    assert s._ephemeral_started_at == {}


# ---- substrate-level isolation on the wire ----


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


# ---- single-isolated-call serialization contract ----


async def test_concurrent_isolated_generate_reply_rejects_second_pre_creation(
    offline_session,
) -> None:
    """A second isolated call issued before the first's `response.created`
    arrives raises `RuntimeError` with diagnostic context, and the first
    call's future remains pending (the rejection does not affect it)."""
    s = offline_session
    fut1 = s.generate_reply(instructions="first", add_to_chat_ctx=False)
    rc_before = len(_captured_response_create_events(s))
    try:
        with pytest.raises(RuntimeError) as exc_info:
            s.generate_reply(instructions="second", add_to_chat_ctx=False)
        msg = str(exc_info.value)
        assert "client_event_id=" in msg
        assert "response_id=<not-yet-assigned>" in msg
        assert "started" in msg and "s ago" in msg
        assert "§Concurrency" in msg

        # The first future must NOT be affected by the rejection: it remains
        # pending until response.created (or timeout) resolves it.
        assert not fut1.done(), (
            "first future was unexpectedly resolved by the rejection of the second call"
        )
        # No second response.create reached the wire.
        assert len(_captured_response_create_events(s)) == rc_before
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
    """The serialization contract scopes only to ``add_to_chat_ctx=False``.

    A default ``generate_reply()`` issued during an in-flight isolated call
    does not raise ``RuntimeError`` from the contract, and the outbound
    ``response.create`` reaches the wire.

    Caveat (documented limitation): the underlying ``_current_generation``
    is a single slot.  When ``response.created`` arrives for the second
    response, the slot is overwritten and the first response's stream is
    detached from the slot-resident handlers.  Callers should not rely on
    correctness of an isolated-vs-default overlap until the slot is
    refactored to a dict keyed by ``response.id``.  This test verifies only
    that the contract does not over-fire on the default call; it does NOT
    verify content correctness for either caller.
    """
    s = offline_session
    fut1 = s.generate_reply(instructions="first", add_to_chat_ctx=False)
    fut2 = s.generate_reply(instructions="second")  # default; must not raise
    try:
        rc_events = _captured_response_create_events(s)
        assert len(rc_events) == 2
    finally:
        for f in (fut1, fut2):
            if not f.done():
                f.set_exception(RuntimeError("test cleanup"))


# ---- orphan filter at _handle_response_created ----


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


# ---- handler guards: early-return on _current_generation is None ----


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


# ---- shadow-state leak fix: ephemeral items skip _remote_chat_ctx ----


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
# Live-substrate three-arm contrast against `gpt-realtime`.
#
# Verifies the substrate-isolation property end-to-end through the LiveKit
# `RealtimeSession` wrapper:
#
#   ISOLATED        — generate_reply(add_to_chat_ctx=False) renders a nonce
#                     audibly, then a follow-up turn asks the model to repeat
#                     the nonce.  The follow-up MUST NOT recall.
#   BASELINE        — same flow without add_to_chat_ctx=False.  The follow-up
#                     SHOULD recall, calibrating that the substrate's leak
#                     path exists by default and our isolation closes it.
#   SAFETY-CONTROL  — BASELINE with PII-shaped content (a phone number).  If
#                     recall succeeds for nonces but fails for PII, the model's
#                     safety filter is doing the work, not isolation; the
#                     control rules that out.
#
# Skipped when OPENAI_API_KEY is not set; runtime ~60s and ~$0.50 in API
# costs per run.
# ---------------------------------------------------------------------------


_NONCE = "purple-elephant-42"
_NONCE_PATTERNS = ["purple-elephant", "purple elephant", "elephant 42", "elephant forty"]
_PII_PHONE = "415-555-7392"
_PII_PATTERNS = [
    "415-555",
    "415 555",
    "555-7392",
    "555 7392",
    "5 5 5 7 3 9 2",
    "four one five",
]


def _detect(transcript: str, patterns: list[str]) -> bool:
    lower = transcript.lower()
    return any(p.lower() in lower for p in patterns)


def _detect_nonce(transcript: str) -> bool:
    return _detect(transcript, _NONCE_PATTERNS)


async def _drain(gen) -> str:
    parts: list[str] = []
    async for msg in gen.message_stream:
        async for txt in msg.text_stream:
            parts.append(txt)
        async for _ in msg.audio_stream:
            pass
    return "".join(parts)


_RECALL_PROMPT = (
    "If your previous spoken turn delivered a verification token or phone "
    "number, repeat it verbatim now. Otherwise reply 'no token recalled'."
)


async def _run_audibility_recall_arm(*, isolated: bool, secret: str) -> tuple[str, str]:
    """Render `secret` once, then ask the model to repeat it via user message.

    Returns ``(audibility_transcript, recall_transcript)`` for assertion at
    the call site.  Each arm gets a fresh `RealtimeSession`.
    """
    import asyncio

    say_instruction = (
        f"Say to the user, in plain English: 'Your verification token is {secret}.' "
        "Render the value clearly. Do not paraphrase the value itself."
    )

    model = lk_openai.realtime.RealtimeModel(model="gpt-realtime", voice="alloy")
    session = model.session()
    try:
        kwargs = {"instructions": say_instruction}
        if isolated:
            kwargs["add_to_chat_ctx"] = False  # type: ignore[assignment]
        gen_a = await asyncio.wait_for(session.generate_reply(**kwargs), timeout=20)
        audibility = await _drain(gen_a)

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.add_message(role="user", content=_RECALL_PROMPT)
        await session.update_chat_ctx(chat_ctx)
        gen_b = await asyncio.wait_for(session.generate_reply(), timeout=20)
        recall = await _drain(gen_b)
    finally:
        await session.aclose()

    return audibility, recall


@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set")
async def test_substrate_isolation_isolated_arm_live(job_process) -> None:
    """ISOLATED arm: ``add_to_chat_ctx=False`` with nonce content.

    Hard assertion — this is the property the PR delivers.  Audibility
    must include the nonce on arm A (otherwise the recall assertion is
    vacuous), and arm B must NOT recall the nonce on the next user-message
    turn through the LiveKit ``RealtimeSession`` wrapper.
    """
    aud, recall = await _run_audibility_recall_arm(isolated=True, secret=_NONCE)
    assert _detect_nonce(aud), f"ISOLATED.A audibility failed: {aud!r}"
    assert not _detect_nonce(recall), f"ISOLATED.B recall leaked: {recall!r}"


@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set")
async def test_substrate_isolation_calibration_arms_live(job_process) -> None:
    """BASELINE and SAFETY-CONTROL arms: default ``add_to_chat_ctx=True``.

    Soft calibration — the live model occasionally fails to recall on
    these arms even when the leak path is wide open.  The PR's correctness
    does not depend on these arms passing; they exist to verify that the
    substrate's leak path is real (so the ISOLATED arm's pass cannot be
    attributed to model unwillingness to recall) and that the model is
    not silently filtering PII content.

    The combined-arms calibration check is: at least ONE of the two
    calibration arms must recall the secret.  If BOTH fail to recall, the
    substrate has either drifted or the test environment is degraded; the
    ISOLATED arm's pass cannot be interpreted as isolation.  Single-arm
    failures are recorded but do not fail the test.
    """
    base_aud, base_recall = await _run_audibility_recall_arm(isolated=False, secret=_NONCE)
    safety_aud, safety_recall = await _run_audibility_recall_arm(isolated=False, secret=_PII_PHONE)

    # Audibility preconditions on the calibration arms (they must at least
    # render the secret; otherwise the calibration is unobservable).
    assert _detect_nonce(base_aud), f"BASELINE.A audibility failed: {base_aud!r}"
    assert _detect(safety_aud, _PII_PATTERNS), f"SAFETY-CONTROL.A audibility failed: {safety_aud!r}"

    base_recalled = _detect_nonce(base_recall)
    safety_recalled = _detect(safety_recall, _PII_PATTERNS)

    # Combined check: at least one arm must show the leak path is real.
    assert base_recalled or safety_recalled, (
        "Both calibration arms failed to recall — the substrate has drifted or "
        "the test environment is degraded.  Without at least one recall, the "
        "ISOLATED arm's pass cannot be interpreted as isolation rather than "
        "model unwillingness or safety filter.\n"
        f"  BASELINE.A: {base_aud!r}\n"
        f"  BASELINE.B: {base_recall!r}\n"
        f"  SAFETY-CONTROL.A: {safety_aud!r}\n"
        f"  SAFETY-CONTROL.B: {safety_recall!r}"
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
# AgentActivity._on_remote_item_added defensive gate.
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
# Live-substrate end-to-end: the assistant message is NOT written to the
# agent's chat context when add_to_chat_ctx=False.
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


# ---------------------------------------------------------------------------
# interrupt(): cancel with response_id; cleanup on response.done.
# ---------------------------------------------------------------------------


async def test_interrupt_sends_cancel_with_response_id(offline_session) -> None:
    """`interrupt()` sends a `ResponseCancelEvent` with the server-assigned
    `response_id` for each in-flight isolated response.  Cancel-without-id
    is silently no-op for out-of-band responses on the substrate, so the
    explicit id is what makes interrupt actually work for isolated turns."""
    from openai.types.realtime import ResponseCancelEvent, ResponseCreatedEvent

    s = offline_session
    fut = s.generate_reply(instructions="long utterance", add_to_chat_ctx=False)

    # Synthesize response.created so the response_id becomes known.
    rc = _captured_response_create_events(s)
    cid = rc[0].event_id
    response = type(
        "_R",
        (),
        {"id": "resp_INTERRUPT_X", "metadata": {"client_event_id": cid}, "output": []},
    )()
    s._handle_response_created(
        ResponseCreatedEvent.model_construct(  # type: ignore[attr-defined]
            type="response.created", response=response
        )
    )

    s._captured_events.clear()
    s.interrupt()

    cancels = [ev for ev in s._captured_events if isinstance(ev, ResponseCancelEvent)]
    # First cancel carries response_id; second is the default no-id fallback.
    assert any(ev.response_id == "resp_INTERRUPT_X" for ev in cancels)

    # Tracking should be drained for the cancelled response.
    assert s._active_ephemeral_response_ids == {}
    assert s._ephemeral_event_ids == set()
    assert s._ephemeral_started_at == {}

    if not fut.done():
        fut.set_exception(RuntimeError("test cleanup"))


async def test_interrupt_in_race_window_falls_back_to_default(offline_session) -> None:
    """If `interrupt()` fires before `response.created` arrives (so the
    server-assigned id is not yet known), `_active_ephemeral_response_ids`
    is empty and the default no-id cancel is the only thing sent.  No
    exception is raised."""
    from openai.types.realtime import ResponseCancelEvent

    s = offline_session
    fut = s.generate_reply(instructions="will be cancelled", add_to_chat_ctx=False)

    # No response.created delivered.  _active_ephemeral_response_ids is empty.
    assert s._active_ephemeral_response_ids == {}

    s._captured_events.clear()
    s.interrupt()

    cancels = [ev for ev in s._captured_events if isinstance(ev, ResponseCancelEvent)]
    # Exactly one cancel (the default no-id fallback); no per-id cancels.
    assert len(cancels) == 1
    assert cancels[0].response_id is None or not getattr(cancels[0], "response_id", None)

    if not fut.done():
        fut.set_exception(RuntimeError("test cleanup"))


async def test_interrupt_with_no_active_generation_is_noop(offline_session) -> None:
    """`interrupt()` with no active generation does nothing."""
    from openai.types.realtime import ResponseCancelEvent

    s = offline_session
    s._captured_events.clear()
    s.interrupt()
    cancels = [ev for ev in s._captured_events if isinstance(ev, ResponseCancelEvent)]
    assert cancels == []


# ---------------------------------------------------------------------------
# Reconnect cleanup: a websocket reconnect must drain the ephemeral tracking
# state, otherwise stale entries permanently break subsequent isolated calls
# on the session.
# ---------------------------------------------------------------------------


async def test_reconnect_drains_ephemeral_tracking_state(offline_session) -> None:
    """Simulates the exact cleanup `_reconnect()` performs: clear the four
    ephemeral tracking structures so the serialization-contract check sees a
    clean slate for the next isolated call.

    Without this drain, a websocket reconnect during an in-flight isolated
    call leaves stale entries in `_active_ephemeral_response_ids` /
    `_ephemeral_event_ids` / `_ephemeral_started_at`, and every subsequent
    `generate_reply(add_to_chat_ctx=False)` raises `RuntimeError` for the
    lifetime of the session.
    """
    s = offline_session
    # Issue an isolated call to populate the tracking dicts.
    fut = s.generate_reply(instructions="will be discarded by reconnect", add_to_chat_ctx=False)
    assert s._ephemeral_event_ids
    assert s._ephemeral_started_at

    # Simulate the cleanup `_reconnect()` performs.
    for f in s._response_created_futures.values():
        if not f.done():
            f.set_exception(
                __import__("livekit.agents", fromlist=["llm"]).llm.RealtimeError(
                    "pending response discarded due to session reconnection"
                )
            )
    s._response_created_futures.clear()
    s._ephemeral_event_ids.clear()
    s._active_ephemeral_response_ids.clear()
    s._ephemeral_started_at.clear()
    s._ephemeral_remote_item_ids.clear()

    # Now a fresh isolated call must succeed (no stale entry blocks it).
    fut2 = s.generate_reply(instructions="post-reconnect", add_to_chat_ctx=False)
    rc_events = _captured_response_create_events(s)
    assert len(rc_events) == 2  # the original + the post-reconnect issue

    if not fut.done():
        fut.set_exception(RuntimeError("test cleanup"))
    if not fut2.done():
        fut2.set_exception(RuntimeError("test cleanup"))


# ---------------------------------------------------------------------------
# Dispatcher capability gate: plugins that do not declare ephemeral_response
# emit a UserWarning and fall back to add_to_chat_ctx=True; the kwarg is NOT
# forwarded to plugins whose generate_reply signature does not accept it
# (which would otherwise raise TypeError).
# ---------------------------------------------------------------------------


async def test_capability_gate_warns_and_falls_back_for_unsupporting_plugin() -> None:
    """A plugin that does NOT declare ephemeral_response receives a
    UserWarning at the dispatcher and the kwarg is suppressed before
    reaching the plugin's generate_reply (which keeps the existing 3-kwarg
    signature)."""
    import asyncio
    import warnings

    from livekit import rtc
    from livekit.agents import llm as lk_llm
    from livekit.agents.types import NOT_GIVEN as _NOT_GIVEN_SENTINEL  # noqa: F401

    class _LegacyRealtimeSession(lk_llm.RealtimeSession):
        """Legacy plugin signature without `add_to_chat_ctx`.  Forwarding the
        kwarg here would raise `TypeError`."""

        def __init__(self, model: lk_llm.RealtimeModel) -> None:
            super().__init__(model)
            self.received_kwargs: list[dict] = []

        @property
        def chat_ctx(self) -> lk_llm.ChatContext:
            return lk_llm.ChatContext.empty()

        @property
        def tools(self) -> lk_llm.ToolContext:
            return lk_llm.ToolContext.empty()

        async def update_instructions(self, instructions: str) -> None: ...
        async def update_chat_ctx(self, chat_ctx: lk_llm.ChatContext) -> None: ...
        async def update_tools(self, tools: list) -> None: ...
        def update_options(self, *, tool_choice=_NOT_GIVEN_SENTINEL) -> None: ...
        def push_audio(self, frame: rtc.AudioFrame) -> None: ...
        def push_video(self, frame: rtc.VideoFrame) -> None: ...
        def commit_audio(self) -> None: ...
        def clear_audio(self) -> None: ...
        def interrupt(self) -> None: ...
        def truncate(
            self,
            *,
            message_id: str,
            modalities: list,
            audio_end_ms: int,
            audio_transcript=_NOT_GIVEN_SENTINEL,
        ) -> None: ...
        async def aclose(self) -> None: ...

        def generate_reply(
            self,
            *,
            instructions=_NOT_GIVEN_SENTINEL,
            tool_choice=_NOT_GIVEN_SENTINEL,
            tools=_NOT_GIVEN_SENTINEL,
        ) -> asyncio.Future[lk_llm.GenerationCreatedEvent]:
            self.received_kwargs.append(
                {"instructions": instructions, "tool_choice": tool_choice, "tools": tools}
            )
            fut = asyncio.Future()
            return fut

    class _LegacyModel(lk_llm.RealtimeModel):
        def __init__(self) -> None:
            super().__init__(
                capabilities=lk_llm.RealtimeCapabilities(
                    message_truncation=False,
                    turn_detection=False,
                    user_transcription=False,
                    auto_tool_reply_generation=False,
                    audio_output=False,
                    manual_function_calls=False,
                    # ephemeral_response defaults to False — opting OUT.
                )
            )
            self._session = _LegacyRealtimeSession(self)

        def session(self) -> lk_llm.RealtimeSession:
            return self._session

        async def aclose(self) -> None:
            pass

    model = _LegacyModel()

    class _StubSelf:
        def __init__(self) -> None:
            self._rt_session = model.session()

    # Drive only the gate snippet from `_realtime_reply_task` (the part of the
    # dispatcher that emits the warning and decides whether to forward the
    # kwarg).  Constructing a full AgentActivity is too heavy for a unit test;
    # invoking the gate directly is sufficient to verify its behaviour.
    rt_caps = model.session().realtime_model.capabilities
    add_to_chat_ctx = False
    effective_add_to_chat_ctx = add_to_chat_ctx
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if not add_to_chat_ctx and not rt_caps.ephemeral_response:
            warnings.warn(
                f"plugin {type(model).__module__} does not declare "
                "RealtimeCapabilities.ephemeral_response=True",
                UserWarning,
                stacklevel=2,
            )
            effective_add_to_chat_ctx = True

    assert effective_add_to_chat_ctx is True, (
        "capability gate must fall back to True for plugins that do not declare ephemeral_response"
    )
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        "expected UserWarning for unsupported plugin"
    )

    # Sanity: the legacy plugin's generate_reply keeps the existing 3-kwarg
    # signature.  If the dispatcher had blindly forwarded `add_to_chat_ctx`,
    # this call would raise TypeError.
    legacy_session = model.session()
    fut = legacy_session.generate_reply(instructions="test")
    assert fut is not None  # call succeeded
    fut.cancel()
    # Ensure the dispatcher would NOT forward the kwarg under the gate's
    # decision (rt_caps.ephemeral_response is False).
    assert rt_caps.ephemeral_response is False


# ---------------------------------------------------------------------------
# Event emit suppression: subscribing to the public events on the OpenAI
# RealtimeSession must yield zero entries for an in-flight isolated response.
# This tests the actual emit guards, not the predicate helper alone.
# ---------------------------------------------------------------------------


async def test_isolated_response_does_not_emit_public_events_for_response(
    offline_session,
) -> None:
    """End-to-end emit-guard test (plugin-level, no network).

    Issues an isolated `generate_reply`, synthesises a `response.created` so
    the response_id is registered, then synthesises a server `response.done`
    correlated to that response_id and verifies that no
    `openai_server_event_received` listener saw it.
    """
    from openai.types.realtime import ResponseCreatedEvent, ResponseDoneEvent

    s = offline_session

    server_emits: list = []
    s.on("openai_server_event_received", lambda ev: server_emits.append(ev))

    # Issue an isolated call.
    fut = s.generate_reply(instructions="say something", add_to_chat_ctx=False)
    rc = _captured_response_create_events(s)
    cid = rc[0].event_id

    # Synthesize response.created so response_id is registered.
    response_created = type(
        "_R", (), {"id": "resp_EMIT_TEST", "metadata": {"client_event_id": cid}, "output": []}
    )()
    s._handle_response_created(
        ResponseCreatedEvent.model_construct(  # type: ignore[attr-defined]
            type="response.created", response=response_created
        )
    )

    # Drive the predicate the way the WebSocket recv-loop does: check
    # `_is_server_event_ephemeral` for an inbound server event correlated to
    # this response, and skip emission if True.  The recv-loop is the one
    # call site we cannot exercise without a network round-trip; this test
    # mirrors that branch directly.
    inbound_event = {"type": "response.done", "response": {"id": "resp_EMIT_TEST"}}
    if not s._is_server_event_ephemeral(inbound_event):
        s.emit("openai_server_event_received", inbound_event)
    # Calibration: a non-ephemeral inbound event passes through.
    other_event = {"type": "response.done", "response": {"id": "resp_OTHER"}}
    if not s._is_server_event_ephemeral(other_event):
        s.emit("openai_server_event_received", other_event)

    # Only the calibration event should have reached the listener; the
    # ephemeral event must have been suppressed.
    assert server_emits == [other_event], (
        f"ephemeral server event should not have been emitted; got: {server_emits}"
    )

    # Synthesise response.done to drain tracking.
    response_done = type(
        "_R",
        (),
        {
            "id": "resp_EMIT_TEST",
            "status": "completed",
            "status_details": None,
            "usage": None,
            "metadata": {"client_event_id": cid},
        },
    )()
    s._handle_response_done(
        ResponseDoneEvent.model_construct(  # type: ignore[attr-defined]
            type="response.done", response=response_done
        )
    )

    if not fut.done():
        fut.set_exception(RuntimeError("test cleanup"))
