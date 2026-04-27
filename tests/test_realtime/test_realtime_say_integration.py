"""Integration and behavioral tests for ephemeral say() on the OpenAI Realtime plugin.

Verifies the isolation contract for ephemeral say():

- Audibility: text passed to say() reaches the audio channel.
- Server-side isolation: text passed with add_to_chat_ctx=False does not enter
  the substrate's conversation state (verified by a follow-up generate_reply
  that cannot retrieve the text).
- Wire-format metadata: outbound response.create carries client_event_id.
- Local chat_ctx gate (behavioral): the gate at _realtime_generation_task_impl
  prevents the secret from entering agent._chat_ctx when add_to_chat_ctx=False.
  Verified with positive control (add_to_chat_ctx=True must write to context).
- Orphan filter (behavioral): late-arriving response.created with a popped
  future is discarded, _current_generation is not set, response.cancel sent.

Tests requiring the real OpenAI API skip when OPENAI_API_KEY is not set.
The local-gate and orphan-filter tests run without a network connection.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import nullcontext
from typing import Any

import pytest
from dotenv import load_dotenv
from openai.types import realtime

from livekit.agents import llm
from livekit.plugins import openai


@pytest.fixture(scope="session", autouse=True)
def _load_env() -> None:
    load_dotenv(override=True)


_skip_no_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; integration tests require real OpenAI access",
)


def _openai_model() -> openai.realtime.RealtimeModel:
    return openai.realtime.RealtimeModel(
        voice="alloy",
        input_audio_transcription=realtime.AudioTranscription(model="gpt-4o-mini-transcribe"),
    )


@pytest.fixture
async def rt_session() -> llm.RealtimeSession:
    model = _openai_model()
    session = model.session()
    yield session
    await session.aclose()
    await asyncio.sleep(0.5)


async def _collect_text(gen_ev: llm.GenerationCreatedEvent) -> str:
    parts: list[str] = []
    async for msg_gen in gen_ev.message_stream:
        async for chunk in msg_gen.text_stream:
            parts.append(chunk)
    return "".join(parts)


async def _drain_generation(gen_ev: llm.GenerationCreatedEvent) -> int:
    """Drain the message + audio streams. Returns total audio frame count."""
    audio_frames = 0
    async for msg_gen in gen_ev.message_stream:
        async for _ in msg_gen.text_stream:
            pass
        async for _ in msg_gen.audio_stream:
            audio_frames += 1
    return audio_frames


@_skip_no_openai_key
async def test_openai_realtime_say_audio_renders(rt_session: llm.RealtimeSession) -> None:
    """say() produces audio frames containing the rendered text."""
    gen_ev = await asyncio.wait_for(
        rt_session.say("the verification token is alpha-bravo"),
        timeout=15,
    )
    audio_frames = await asyncio.wait_for(_drain_generation(gen_ev), timeout=20)
    assert audio_frames > 0, "expected at least one audio frame from say()"


@_skip_no_openai_key
async def test_openai_realtime_say_emits_metadata(rt_session: llm.RealtimeSession) -> None:
    """Outbound response.create carries metadata.client_event_id starting with the say prefix."""
    captured: list[object] = []
    real_send = rt_session.send_event

    def _capture(ev: object) -> None:
        captured.append(ev)
        return real_send(ev)

    rt_session.send_event = _capture  # type: ignore[method-assign]

    try:
        gen_ev = await asyncio.wait_for(rt_session.say("alpha-bravo"), timeout=15)
        await asyncio.wait_for(_drain_generation(gen_ev), timeout=20)
    finally:
        rt_session.send_event = real_send  # type: ignore[method-assign]

    create_events = [
        e
        for e in captured
        if getattr(e, "type", None) == "response.create"
        and isinstance(getattr(getattr(e, "response", None), "metadata", None), dict)
        and e.response.metadata.get("client_event_id", "").startswith("response_create_say_")
    ]
    assert len(create_events) >= 1, (
        f"expected response.create event with say-prefixed client_event_id; "
        f"captured {len(captured)} total events, none matched the prefix"
    )


@_skip_no_openai_key
async def test_openai_realtime_say_isolation_no_leak(rt_session: llm.RealtimeSession) -> None:
    """After say(secret, add_to_chat_ctx=False), generate_reply cannot recall the secret.

    Proves server-side isolation: the OpenAI Realtime substrate does not retain
    isolated text in conversation state. A follow-up generation that asks the
    model to repeat what it just said must not include the secret.
    """
    secret = "purple-elephant-42"

    say_gen = await asyncio.wait_for(
        rt_session.say(f"the verification token is {secret}", add_to_chat_ctx=False),
        timeout=15,
    )
    await asyncio.wait_for(_drain_generation(say_gen), timeout=30)

    follow_up = await asyncio.wait_for(
        rt_session.generate_reply(
            instructions=(
                "Repeat the last verification token you just said out loud, exactly. "
                "If you cannot recall it, say 'no token recalled'."
            )
        ),
        timeout=15,
    )
    follow_up_text = await asyncio.wait_for(_collect_text(follow_up), timeout=20)

    assert secret not in follow_up_text.lower(), (
        f"secret leaked into follow-up generation: {follow_up_text!r}"
    )


@_skip_no_openai_key
async def test_openai_realtime_say_isolation_no_remote_chat_ctx_leak(
    rt_session: llm.RealtimeSession,
) -> None:
    """After say(secret, add_to_chat_ctx=False), session.chat_ctx contains no items with the secret.

    Proves substrate-state isolation: the OpenAI Realtime API does not retain
    isolated text in the visible conversation history (server-side chat_ctx).
    """
    secret = "purple-elephant-42"

    say_gen = await asyncio.wait_for(
        rt_session.say(f"the verification token is {secret}", add_to_chat_ctx=False),
        timeout=15,
    )
    await asyncio.wait_for(_drain_generation(say_gen), timeout=30)

    # Allow substrate state to settle.
    await asyncio.sleep(0.5)

    chat_ctx = rt_session.chat_ctx
    flat = " ".join(
        repr(getattr(item, "content", "")) for item in getattr(chat_ctx, "items", [])
    ).lower()
    assert secret not in flat, f"secret leaked into substrate chat_ctx: {flat!r}"


# ---------------------------------------------------------------------------
# Behavioral tests (no real OpenAI API needed)
# ---------------------------------------------------------------------------


async def test_openai_realtime_say_isolation_no_local_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local chat_ctx gate prevents the secret from entering agent._chat_ctx.

    Behavioral proof with positive+negative control: first verifies that
    add_to_chat_ctx=True populates agent._chat_ctx (positive control — would
    catch a regression where local insertion silently stops working), then
    verifies add_to_chat_ctx=False does NOT.

    This test exercises the actual gate code at _realtime_generation_task_impl
    by running the function with minimal infrastructure patches. No OpenAI API
    call is made.
    """
    from livekit.agents.llm.chat_context import ChatContext  # noqa: PLC0415
    from livekit.agents.voice.agent import ModelSettings  # noqa: PLC0415
    from livekit.agents.voice.agent_activity import AgentActivity  # noqa: PLC0415
    from livekit.agents.voice.speech_handle import SpeechHandle  # noqa: PLC0415

    # Stub the tracer so start_as_current_span doesn't require an OTEL backend.
    class _NoopSpan:
        def set_attribute(self, *a: Any, **kw: Any) -> None:
            pass

        def set_attributes(self, *a: Any, **kw: Any) -> None:
            pass

    class _NoopTracer:
        def start_as_current_span(self, *a: Any, **kw: Any) -> Any:
            return nullcontext(_NoopSpan())

    monkeypatch.setattr("livekit.agents.voice.agent_activity.tracer", _NoopTracer())

    # Stub perform_tool_executions to return a pre-completed no-op task
    # with a _ToolOutput object that has the required first_tool_started_fut.
    from livekit.agents.voice.generation import _ToolOutput  # noqa: PLC0415

    async def _noop_tool_exec(*args: Any, **kwargs: Any) -> None:
        pass

    def _stub_perform_tool_executions(*args: Any, **kwargs: Any) -> Any:
        fut: asyncio.Future[None] = asyncio.Future()
        fut.set_result(None)  # pre-complete so callbacks fire immediately
        return (
            asyncio.create_task(_noop_tool_exec()),
            _ToolOutput(output=[], first_tool_started_fut=fut),
        )

    monkeypatch.setattr(
        "livekit.agents.voice.agent_activity.perform_tool_executions",
        _stub_perform_tool_executions,
    )

    async def _run_gate_test(add_to_chat_ctx: bool) -> list[Any]:
        """Run _realtime_generation_task_impl with a given add_to_chat_ctx flag.

        Returns the list of items captured by the _upsert_item spy.
        """
        secret = "purple-elephant-42"

        class _Caps:
            supports_say = True
            ephemeral_say = True
            turn_detection = False
            audio_output = False

        caps = _Caps()

        chat_ctx = ChatContext.empty()
        upsert_calls: list[Any] = []
        real_upsert = chat_ctx._upsert_item

        def _spy_upsert(item: Any) -> Any:
            upsert_calls.append(item)
            return real_upsert(item)

        chat_ctx._upsert_item = _spy_upsert  # type: ignore[method-assign]

        class _ModelStub(llm.RealtimeModel):
            def __init__(self) -> None:
                self._capabilities = caps

            @property
            def capabilities(self) -> Any:  # type: ignore[override]
                return self._capabilities

            @property
            def model(self) -> str:
                return "test-model"

            @property
            def provider(self) -> str:
                return "test"

            def session(self) -> Any:
                raise NotImplementedError

            async def aclose(self) -> None:
                pass

        rt_model = _ModelStub()

        activity = AgentActivity.__new__(AgentActivity)

        class _AgentShim:
            llm = rt_model
            tts = None
            stt = None
            allow_interruptions = True
            _chat_ctx = chat_ctx
            tools: list[Any] = []

            def transcription_node(self, text: Any, model_settings: Any) -> Any:
                return text

            def tts_node(self, *a: Any, **kw: Any) -> Any:
                return None

            @property
            def use_tts_aligned_transcript(self) -> bool:
                return False

        class _OutputShim:
            audio = None
            audio_enabled = False
            transcription = None
            transcription_enabled = False

        class _SessionShim:
            llm = rt_model
            tts = None
            output = _OutputShim()
            _room_io = None
            _root_span_context = None
            tools: list[Any] = []

            def _conversation_item_added(self, *a: Any) -> None:
                pass

            def _tool_items_added(self, *a: Any) -> None:
                pass

            def _update_agent_state(self, *a: Any, **kw: Any) -> None:
                pass

        activity._agent = _AgentShim()  # type: ignore[attr-defined]
        activity._session = _SessionShim()  # type: ignore[attr-defined]

        class _RtSessionStub:
            pass

        activity._rt_session = _RtSessionStub()  # type: ignore[attr-defined]
        activity._realtime_spans = None  # type: ignore[attr-defined]
        activity._audio_recognition = None  # type: ignore[attr-defined]
        activity._interruption_by_audio_activity_enabled = False  # type: ignore[attr-defined]
        activity._interruption_detection_enabled = False  # type: ignore[attr-defined]
        activity._mcp_tools: list[Any] = []  # type: ignore[attr-defined]
        activity._background_speeches: set[Any] = set()  # type: ignore[attr-defined]

        authorized = asyncio.Event()
        authorized.set()
        activity._authorization_allowed = authorized  # type: ignore[attr-defined]
        silence = asyncio.Event()
        silence.set()
        activity._user_silence_event = silence  # type: ignore[attr-defined]

        handle = SpeechHandle.create(allow_interruptions=True)
        # Pre-authorize: set the authorize event so _wait_for_authorization
        # resolves immediately, and mark scheduled so the handle is valid.
        handle._authorize_event.set()  # type: ignore[attr-defined]
        handle._scheduled_fut.set_result(None)  # type: ignore[attr-defined]
        # Also push one generation future so _mark_generation_done works.
        handle._generations.append(asyncio.Future())  # type: ignore[attr-defined]

        # Build a generation event with one message that yields our secret text.
        async def _text_stream() -> Any:
            yield secret

        async def _empty_audio() -> Any:
            return
            yield  # make it an async generator

        async def _empty_func_stream() -> Any:
            return
            yield  # make it an async generator

        modalities_fut: asyncio.Future[list[str]] = asyncio.Future()
        modalities_fut.set_result(["text"])

        msg_gen = llm.MessageGeneration(
            message_id="test-gate-msg",
            text_stream=_text_stream(),
            audio_stream=_empty_audio(),
            modalities=modalities_fut,
        )

        async def _msg_stream() -> Any:
            yield msg_gen

        gen_ev = llm.GenerationCreatedEvent(
            message_stream=_msg_stream(),
            function_stream=_empty_func_stream(),
            user_initiated=True,
        )

        await activity._realtime_generation_task_impl(
            speech_handle=handle,
            generation_ev=gen_ev,
            model_settings=ModelSettings(),
            add_to_chat_ctx=add_to_chat_ctx,
        )

        return upsert_calls

    # Positive control: add_to_chat_ctx=True MUST populate chat_ctx.
    positive_calls = await _run_gate_test(add_to_chat_ctx=True)
    assert len(positive_calls) > 0, (
        "positive control FAILED: add_to_chat_ctx=True did not populate chat_ctx. "
        "The local insertion path is broken — the negative test would be meaningless."
    )
    positive_texts = " ".join(repr(c) for c in positive_calls).lower()
    assert "purple-elephant-42" in positive_texts, (
        f"positive control: secret not found in upserted items: {positive_calls!r}"
    )

    # Negative: add_to_chat_ctx=False must NOT populate chat_ctx.
    negative_calls = await _run_gate_test(add_to_chat_ctx=False)
    assert len(negative_calls) == 0, (
        f"gate FAILED: add_to_chat_ctx=False still called _upsert_item with: {negative_calls!r}"
    )


async def test_orphaned_response_after_timeout_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Orphan filter discards a late response.created when the future has been popped.

    Behavioral proof: directly invokes _handle_response_created with a
    synthetic late event whose metadata.client_event_id is missing from
    _response_created_futures. Asserts:
    - _current_generation remains None (orphan filter early-returned).
    - A defensive ResponseCancelEvent with the correct response_id was sent.

    No OpenAI API call is made.
    """
    from openai.types.realtime import ResponseCancelEvent, ResponseCreatedEvent  # noqa: PLC0415
    from openai.types.realtime.realtime_response import RealtimeResponse  # noqa: PLC0415

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-not-real")
    from livekit.plugins.openai.realtime.realtime_model import (  # noqa: PLC0415
        RealtimeSession,
    )

    session = RealtimeSession.__new__(RealtimeSession)
    # Simulate the timeout-popped state: no in-flight futures.
    session._response_created_futures = {}  # type: ignore[attr-defined]
    session._current_generation = None  # type: ignore[attr-defined]
    captured: list[object] = []
    session.send_event = captured.append  # type: ignore[method-assign]

    orphan_event_id = "response_create_say_orphan-test-abc"
    response = RealtimeResponse.construct(
        id="resp_orphan_test_abc",
        metadata={"client_event_id": orphan_event_id},
    )
    event = ResponseCreatedEvent.construct(
        event_id="evt_orphan_test",
        response=response,
        type="response.created",
    )

    session._handle_response_created(event)

    # Orphan filter must have early-returned: _current_generation stays None.
    assert session._current_generation is None, (
        "_current_generation must remain None after orphan-filter early-return"
    )

    # Defensive cancel WITH response_id must have been sent.
    cancel_events = [
        e
        for e in captured
        if isinstance(e, ResponseCancelEvent)
        and getattr(e, "response_id", None) == "resp_orphan_test_abc"
    ]
    assert len(cancel_events) == 1, (
        f"expected exactly one ResponseCancelEvent(response_id='resp_orphan_test_abc'); "
        f"captured: {captured!r}"
    )


@pytest.mark.parametrize(
    "metadata_case",
    [
        pytest.param(None, id="metadata_None"),
        pytest.param({}, id="metadata_empty_dict"),
    ],
)
async def test_response_created_without_metadata_bypasses_orphan_filter(
    monkeypatch: pytest.MonkeyPatch,
    metadata_case: object,
) -> None:
    """Orphan filter MUST NOT fire when response.metadata has no client_event_id.

    This is the sister case to test_orphaned_response_after_timeout_filtered.
    The metadata-presence guard exists so that server-VAD-initiated responses
    (which carry no client_event_id, because we did not issue them) flow
    through the normal generation-creation path. If the guard is stripped or
    inverted, every server-VAD response would be silently discarded and the
    agent would stop responding to user speech — a critical regression.

    Parametrized across the two no-client_event_id substrate representations:
    metadata=None (no metadata field) and metadata={} (present but empty).

    For each case the test directly invokes _handle_response_created and
    asserts the OBSERVABLE post-guard contract:
    - The "generation_created" event IS emitted with the expected response_id
      (this is the user-visible success signal — without it, server-VAD
      turns stall even if _current_generation is set).
    - No ResponseCancelEvent was sent (the orphan filter did not fire).
    - _current_generation IS NOT None (post-guard _ResponseGeneration ran).

    No OpenAI API call is made.
    """
    from openai.types.realtime import ResponseCancelEvent, ResponseCreatedEvent  # noqa: PLC0415
    from openai.types.realtime.realtime_response import RealtimeResponse  # noqa: PLC0415

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-not-real")
    from livekit.plugins.openai.realtime.realtime_model import (  # noqa: PLC0415
        RealtimeSession,
    )

    session = RealtimeSession.__new__(RealtimeSession)
    session._response_created_futures = {}  # type: ignore[attr-defined]
    session._current_generation = None  # type: ignore[attr-defined]
    sent_events: list[object] = []
    session.send_event = sent_events.append  # type: ignore[method-assign]

    emitted_events: list[tuple[str, object]] = []

    def _capture_emit(name: str, payload: object, *args: object, **kwargs: object) -> None:
        emitted_events.append((name, payload))

    session.emit = _capture_emit  # type: ignore[method-assign]

    response_id = "resp_server_vad_test"
    response = RealtimeResponse.construct(
        id=response_id,
        metadata=metadata_case,
    )
    event = ResponseCreatedEvent.construct(
        event_id="evt_server_vad_test",
        response=response,
        type="response.created",
    )

    session._handle_response_created(event)

    # Observable contract: "generation_created" event IS emitted with the
    # response_id. This is what callers wait on; missing emit = stalled turn.
    generation_emits = [
        payload for (name, payload) in emitted_events if name == "generation_created"
    ]
    assert len(generation_emits) == 1, (
        f"metadata-presence guard FAILED: expected exactly one 'generation_created' "
        f"emission for a server-VAD response (metadata={metadata_case!r}); "
        f"all emits: {emitted_events!r}"
    )
    emitted_gen = generation_emits[0]
    assert getattr(emitted_gen, "response_id", None) == response_id, (
        f"emitted generation_created event has wrong response_id: {emitted_gen!r}"
    )

    # No defensive cancel must have been emitted (orphan filter did not fire).
    cancel_events = [e for e in sent_events if isinstance(e, ResponseCancelEvent)]
    assert cancel_events == [], (
        f"orphan filter incorrectly emitted ResponseCancelEvent for a server-VAD "
        f"response (metadata={metadata_case!r}): {cancel_events!r}"
    )

    # Internal state: _current_generation populated as a structural sanity check.
    assert session._current_generation is not None, (
        f"_current_generation should be populated post-guard (metadata={metadata_case!r})"
    )
