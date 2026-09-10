from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from google.genai import types

from livekit.agents import llm, utils
from livekit.plugins.google.realtime.api_proto import ClientEvents
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession
from livekit.plugins.google.utils import create_function_response

pytestmark = pytest.mark.unit


def _is_genai_client_teardown(task: asyncio.Task[Any]) -> bool:
    """Whether this task is a genai client's ``aclose()`` left behind by a finalizer.

    Keyed on the coroutine's defining module, not its name alone -- ``aclose``
    is a common method name and this must not touch unrelated tasks. Coroutine
    objects carry no ``__module__``, hence the walk through ``cr_frame``.
    """
    coro = task.get_coro()
    if not (getattr(coro, "__qualname__", "") or "").endswith(".aclose"):
        return False
    frame = getattr(coro, "cr_frame", None)
    module = frame.f_globals.get("__name__", "") if frame else ""
    return module.startswith("google.genai")


@pytest.fixture(autouse=True)
async def _settle_genai_finalizers() -> AsyncIterator[None]:
    """Finish the genai client teardown this test started, before the next one.

    ``AsyncClient.__del__`` schedules ``aclose()`` on whatever event loop is
    running when the collector reaches it, with no check for a client that was
    already closed explicitly -- so even the sessions this module closes
    properly leave a finalizer behind. Settled here, while this test still owns
    the loop, those tasks would otherwise surface as leaked tasks in an
    unrelated test in a later module.
    """
    yield
    gc.collect()
    if pending := [
        task for task in asyncio.all_tasks() if not task.done() and _is_genai_client_teardown(task)
    ]:
        await asyncio.gather(*pending, return_exceptions=True)


# 10ms of silence at the output sample rate (24kHz mono, 16-bit)
_PCM_FRAME = b"\x00\x01" * 240


@asynccontextmanager
async def _make_session(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[RealtimeSession]:
    """A session whose background connect loop is stopped before it hits the network.

    Closed on exit so the genai http clients are released here instead of by
    ``AsyncClient.__del__``, which schedules ``aclose()`` on whatever event loop
    is running when the collector happens to reach them.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    session = RealtimeModel().session()
    # cancel the connect loop before the event loop ever schedules it, so no
    # websocket connection is attempted
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    try:
        yield session
    finally:
        await session.aclose()


@asynccontextmanager
async def _make_configured_session(
    monkeypatch: pytest.MonkeyPatch, **options: object
) -> AsyncIterator[RealtimeSession]:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    session = RealtimeModel(**options).session()  # type: ignore[arg-type]
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    try:
        yield session
    finally:
        await session.aclose()


def _audio_content(**kwargs: object) -> types.LiveServerContent:
    return types.LiveServerContent(
        model_turn=types.Content(
            parts=[types.Part(inline_data=types.Blob(data=_PCM_FRAME, mime_type="audio/pcm"))]
        ),
        **kwargs,  # type: ignore[arg-type]
    )


async def _drain_generation(
    event: llm.GenerationCreatedEvent,
) -> tuple[str, int, list[str]]:
    text = ""
    audio_frames = 0
    async for message in event.message_stream:
        async for chunk in message.text_stream:
            text += chunk
        async for _frame in message.audio_stream:
            audio_frames += 1

    function_calls = [call.name async for call in event.function_stream]
    return text, audio_frames, function_calls


async def test_unspoken_model_text_is_omitted_in_audio_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(
                    parts=[types.Part(text="call:getWeather{location:Seattle")]
                ),
                output_transcription=types.Transcription(text="Let me check."),
            )
        )

        assert gen.output_text == "Let me check."
        assert gen.text_ch.recv_nowait() == "Let me check."
        assert gen.text_ch.empty()


async def test_model_text_is_forwarded_in_text_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, modalities=[types.Modality.TEXT]) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="Hello there.")])
            )
        )

        assert gen.output_text == "Hello there."
        assert gen.text_ch.recv_nowait() == "Hello there."


async def test_model_text_is_forwarded_without_output_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, output_audio_transcription=None) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="Hello there.")])
            )
        )

        assert gen.output_text == "Hello there."
        assert gen.text_ch.recv_nowait() == "Hello there."


async def test_transcript_contains_only_output_transcription_with_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        session._start_new_generation()

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="call:assetGenerator{context:")])
            )
        )
        session._handle_server_content(_audio_content())
        session._handle_server_content(
            types.LiveServerContent(output_transcription=types.Transcription(text="Tako je!"))
        )
        session._handle_server_content(types.LiveServerContent(generation_complete=True))
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert len(generations) == 1
        assert await _drain_generation(generations[0]) == ("Tako je!", 1, [])


async def test_tool_call_is_delivered_without_written_call_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        session._start_new_generation()

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="call:getWeather{location:")])
            )
        )
        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(id="fc-1", name="getWeather", args={"location": "Seattle"})
                ]
            )
        )

        assert len(generations) == 1
        assert await _drain_generation(generations[0]) == ("", 0, ["getWeather"])


async def test_transcript_keeps_model_text_in_text_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, modalities=[types.Modality.TEXT]) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        session._start_new_generation()

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="Hello there.")]),
                turn_complete=True,
            )
        )

        assert len(generations) == 1
        assert await _drain_generation(generations[0]) == ("Hello there.", 0, [])


async def test_transcript_keeps_model_text_without_output_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, output_audio_transcription=None) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        session._start_new_generation()

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="Hello there.")]),
                turn_complete=True,
            )
        )

        assert len(generations) == 1
        assert await _drain_generation(generations[0]) == ("Hello there.", 0, [])


async def test_output_streams_close_on_generation_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generation_complete ends the audio/text segment; finalization waits for turn_complete.

    Gemini delays turn_complete until it estimates client-side playback has finished, so
    keying the stream close off turn_complete makes AudioSegmentEnd (and the finalized
    transcript) arrive seconds late (issue #6421). Both streams must close on
    generation_complete, while the generation stays open until turn_complete for input
    transcription and metrics.
    """
    async with _make_session(monkeypatch) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(
            _audio_content(
                output_transcription=types.Transcription(text="hello"),
                generation_complete=True,
            )
        )

        # audio and text were consumed and both segments ended immediately
        assert gen._first_token_timestamp is not None
        assert gen.output_text == "hello"
        assert gen.audio_ch.closed
        assert gen.text_ch.closed
        # but the generation is still open for trailing input transcription until turn_complete
        assert not gen._done
        assert not gen.message_ch.closed

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert gen._done
        assert gen.message_ch.closed


async def test_late_content_after_generation_complete_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stray audio/text after generation_complete is dropped (not pushed to a closed stream)."""
    async with _make_session(monkeypatch) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(_audio_content(generation_complete=True))
        assert gen.audio_ch.closed and gen.text_ch.closed

        with caplog.at_level(logging.WARNING):
            # must not raise ChanClosed, must not append to the transcript, and must warn
            session._handle_server_content(
                _audio_content(output_transcription=types.Transcription(text="late"))
            )

        assert gen.audio_ch.closed and gen.text_ch.closed
        assert gen.output_text == ""
        assert not gen._done
        assert any("after generation completed" in r.message for r in caplog.records)


async def test_input_transcription_uses_generation_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interim and final transcripts stay on the timeline before the reply they prompted."""
    async with _make_session(monkeypatch) as session:
        transcripts: list[llm.InputTranscriptionCompleted] = []
        session.on("input_audio_transcription_completed", transcripts.append)
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None
        gen._created_timestamp = 1234.5

        session._handle_server_content(
            types.LiveServerContent(input_transcription=types.Transcription(text="hello"))
        )
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert [(event.is_final, event.turn_started_at) for event in transcripts] == [
            (False, 1234.5),
            (True, 1234.5),
        ]


async def test_session_close_releases_the_genai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aclose() must release the genai http clients.

    Otherwise they live until the collector runs ``AsyncClient.__del__``, which
    does ``asyncio.get_running_loop().create_task(self.aclose())`` - creating
    pending tasks on whatever event loop is running at that moment.
    """
    closed = False

    async with _make_session(monkeypatch) as session:
        real_aclose = session._client.aio.aclose

        async def _spy() -> None:
            nonlocal closed
            closed = True
            await real_aclose()

        monkeypatch.setattr(session._client.aio, "aclose", _spy)

    assert closed


def _tool_call(call_id: str = "fc_1", name: str = "lookup") -> types.LiveServerToolCall:
    return types.LiveServerToolCall(
        function_calls=[types.FunctionCall(id=call_id, name=name, args={})]
    )


def _tool_output(
    call_id: str = "fc_1", name: str = "lookup", *, reply_required: bool = True
) -> llm.FunctionCallOutput:
    return llm.FunctionCallOutput(
        call_id=call_id,
        name=name,
        output="42",
        is_error=False,
        reply_required=reply_required,
    )


@asynccontextmanager
async def _make_connected_session(
    monkeypatch: pytest.MonkeyPatch, *, non_blocking_tools: bool = False
) -> AsyncIterator[RealtimeSession]:
    """A session that believes it is connected, so update_chat_ctx actually emits.

    The placeholder is never called: the send task is not running, so client events just
    queue up in `_msg_ch` for the test to inspect. `_make_session` closes that channel to
    stop the connect loop, so it is replaced with an open one first.
    """
    async with _make_session(monkeypatch) as session:
        if non_blocking_tools:
            session._opts.tool_behavior = types.Behavior.NON_BLOCKING
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = object()  # type: ignore[assignment]
        try:
            yield session
        finally:
            # the placeholder has no close(), drop it before aclose() reaches for one
            session._active_session = None


async def _drain_sent(session: RealtimeSession) -> list[object]:
    sent: list[object] = []
    while not session._msg_ch.empty():
        sent.append(session._msg_ch.recv_nowait())
    return sent


@pytest.mark.parametrize(
    "reply_required, scheduling",
    [(False, types.FunctionResponseScheduling.SILENT), (True, None)],
)
async def test_tool_response_scheduling_follows_the_output(
    monkeypatch: pytest.MonkeyPatch,
    reply_required: bool,
    scheduling: types.FunctionResponseScheduling | None,
) -> None:
    """A result owed by an interrupted turn goes out SILENT; a normal one keeps the default.

    Gemini blocks the turn until every call is answered and offers no cancel, so dropping the
    result strands the session (issue #6569). SILENT records it without prompting speech.
    """
    async with _make_connected_session(monkeypatch, non_blocking_tools=True) as session:
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())
        await _drain_sent(session)

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output(reply_required=reply_required))
        await session.update_chat_ctx(chat_ctx)

        sent = await _drain_sent(session)
        responses = [m for m in sent if isinstance(m, types.LiveClientToolResponse)]
        assert len(responses) == 1, f"expected the tool response to be sent, got {sent}"
        assert responses[0].function_responses is not None
        assert responses[0].function_responses[0].id == "fc_1"
        assert responses[0].function_responses[0].scheduling == scheduling


async def test_blocking_tools_send_the_response_and_warn_it_cannot_be_silent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Gemini ignores scheduling on BLOCKING declarations, so the reply cannot be prevented.

    It is sent anyway, since unblocking the turn matters more, and every one is reported.
    """
    async with _make_connected_session(monkeypatch) as session:
        session._start_new_generation()
        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(id="fc_1", name="lookup", args={}),
                    types.FunctionCall(id="fc_2", name="search", args={}),
                ]
            )
        )
        await _drain_sent(session)

        with caplog.at_level(logging.WARNING):
            for call_id, name in (("fc_1", "lookup"), ("fc_2", "search")):
                chat_ctx = session.chat_ctx.copy()
                chat_ctx.items.append(_tool_output(call_id, name, reply_required=False))
                await session.update_chat_ctx(chat_ctx)

        responses = [
            m for m in await _drain_sent(session) if isinstance(m, types.LiveClientToolResponse)
        ]
        assert len(responses) == 2
        assert all(r.function_responses[0].scheduling is None for r in responses)  # type: ignore[index]

        warnings = [r for r in caplog.records if "wants no reply" in r.message]
        assert len(warnings) == 2, "every update reports what it could not keep quiet"
        assert [r.functions for r in warnings] == [["lookup"], ["search"]]  # type: ignore[attr-defined]


@pytest.mark.parametrize("vertexai", [False, True])
def test_function_response_scheduling_only_for_gemini_api(vertexai: bool) -> None:
    """Vertex AI rejects `scheduling` (and `id`), so neither is set for it."""
    res = create_function_response(
        _tool_output(),
        vertexai=vertexai,
        tool_response_scheduling=types.FunctionResponseScheduling.SILENT,
    )

    if vertexai:
        assert res.scheduling is None
        assert res.id is None
    else:
        assert res.scheduling == types.FunctionResponseScheduling.SILENT
        assert res.id == "fc_1"


def test_vertex_scheduling_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An explicitly set scheduling is dropped on Vertex AI, so say so instead of ignoring it."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    with caplog.at_level(logging.WARNING):
        RealtimeModel(
            vertexai=True,
            project="p",
            location="us-central1",
            tool_response_scheduling=types.FunctionResponseScheduling.SILENT,
        )

    assert any("tool_response_scheduling is not supported" in r.message for r in caplog.records)


def test_gemini_api_scheduling_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    with caplog.at_level(logging.WARNING):
        RealtimeModel(tool_response_scheduling=types.FunctionResponseScheduling.SILENT)

    assert not any("tool_response_scheduling is not supported" in r.message for r in caplog.records)


class _FakeLiveSession:
    """Stands in for the genai live session and records what the plugin sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self._closed = asyncio.Event()

    async def send_client_content(self, *, turns: object, turn_complete: bool) -> None:
        self.sent.append(("content", turns))

    async def send_tool_response(self, *, function_responses: object) -> None:
        self.sent.append(("tool_response", function_responses))

    async def send_realtime_input(self, **kwargs: object) -> None:
        pass

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        await self._closed.wait()
        return
        yield  # pragma: no cover - makes this an async generator

    async def close(self) -> None:
        self._closed.set()


@asynccontextmanager
async def _connected_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    handle: str | None,
    known: llm.ChatContext | None = None,
    sent_after_handle: llm.ChatContext | None = None,
    pending: llm.ChatContext | None = None,
    caller_handle: bool = False,
) -> AsyncIterator[tuple[RealtimeSession, _FakeLiveSession]]:
    """Connect once onto a fake socket.

    `known` is the state the handle stands for, `sent_after_handle` what the previous
    socket synced after the handle arrived, `pending` the update that arrives before the
    connect loop runs. `caller_handle` passes the handle through `RealtimeModel` instead,
    so its baseline is unknown.
    """
    from google.genai.live import AsyncLive

    fake = _FakeLiveSession()

    @asynccontextmanager
    async def _connect(self: AsyncLive, **kwargs: object) -> AsyncIterator[_FakeLiveSession]:
        yield fake

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", _connect)
    if caller_handle:
        session = RealtimeModel(
            session_resumption=types.SessionResumptionConfig(handle=handle)
        ).session()
    else:
        session = RealtimeModel().session()
        session._session_resumption_handle = handle
    if known is not None:
        session._resumption_chat_ctx = known
        session._chat_ctx = sent_after_handle if sent_after_handle is not None else known
    if pending is not None:
        await session.update_chat_ctx(pending)
    try:
        while session._active_session is None:
            await asyncio.sleep(0.01)
        # let the send task drain the queued events onto the fake socket
        await asyncio.sleep(0.05)
        yield session, fake
    finally:
        await session.aclose()


def _texts(sent: list[tuple[str, object]]) -> list[list[str]]:
    return [
        [p.text for c in turns for p in c.parts]  # type: ignore[attr-defined]
        for kind, turns in sent
        if kind == "content"
    ]


async def test_fresh_session_replays_chat_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="hello")
    ctx.add_message(role="assistant", content="hi")

    async with _connected_session(monkeypatch, handle=None, pending=ctx) as (session, fake):
        assert _texts(fake.sent) == [["hello", "hi"]]
        assert session._pending_chat_ctx is None


async def test_resumed_session_skips_chat_ctx_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server restored the conversation from the handle; replaying it would duplicate it."""
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="hello")

    async with _connected_session(monkeypatch, handle="resume-1", known=ctx, pending=ctx) as (
        session,
        fake,
    ):
        assert fake.sent == []
        assert session._pending_chat_ctx is None
        assert [m.text_content for m in session.chat_ctx.messages()] == ["hello"]


async def test_resumed_session_sends_only_the_disconnected_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Items appended during the restart are new to the resumed session; the rest is not."""
    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    updated = known.copy()
    updated.add_message(role="user", content="one more thing")

    async with _connected_session(monkeypatch, handle="resume-1", known=known, pending=updated) as (
        session,
        fake,
    ):
        assert _texts(fake.sent) == [["one more thing"]]
        assert [m.text_content for m in session.chat_ctx.messages()] == [
            "hello",
            "one more thing",
        ]


async def test_resumed_session_delivers_the_tool_result_from_the_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resumed session still holds the call open, so the result produced meanwhile answers it."""
    known = llm.ChatContext.empty()
    known.add_message(role="user", content="book it")
    known.items.append(llm.FunctionCall(call_id="call-1", name="book", arguments="{}"))
    updated = known.copy()
    updated.items.append(
        llm.FunctionCallOutput(call_id="call-1", name="book", output="done", is_error=False)
    )

    async with _connected_session(monkeypatch, handle="resume-1", known=known, pending=updated) as (
        _,
        fake,
    ):
        assert [kind for kind, _ in fake.sent] == ["tool_response"]
        responses = fake.sent[0][1]
        assert [r.id for r in responses] == ["call-1"]  # type: ignore[attr-defined]


async def test_resumed_session_resends_what_the_handle_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message synced after the last handle is not in the server snapshot; resend it."""
    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    later = known.copy()
    later.add_message(role="user", content="sent before the socket dropped")

    async with _connected_session(
        monkeypatch, handle="resume-1", known=known, sent_after_handle=later
    ) as (session, fake):
        assert _texts(fake.sent) == [["sent before the socket dropped"]]
        assert session._pending_chat_ctx is None


async def test_caller_provided_handle_adopts_the_history_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a handle from the constructor the baseline is unknown; the history is the server's."""
    history = llm.ChatContext.empty()
    history.add_message(role="user", content="from the previous process")
    history.add_message(role="assistant", content="noted")

    async with _connected_session(
        monkeypatch, handle="resume-1", pending=history, caller_handle=True
    ) as (session, fake):
        assert fake.sent == []
        assert [m.text_content for m in session.chat_ctx.messages()] == [
            "from the previous process",
            "noted",
        ]


@llm.function_tool
async def _restart_tool() -> str:
    """Any new tool makes update_tools restart the socket."""
    return ""


async def test_handle_does_not_claim_a_queued_but_unsent_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handle that lands while a diff is still queued must not cover it; a restart re-sends it."""
    from google.genai.live import AsyncLive

    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    updated = known.copy()
    updated.add_message(role="user", content="queued behind the handle")

    class _GatedSession(_FakeLiveSession):
        """First socket: the connect-time handle arrives while the send is still blocked."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = asyncio.Event()

        async def send_client_content(self, *, turns: object, turn_complete: bool) -> None:
            await self.gate.wait()
            await super().send_client_content(turns=turns, turn_complete=turn_complete)

        async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
            yield types.LiveServerMessage(
                session_resumption_update=types.LiveServerSessionResumptionUpdate(
                    new_handle="resume-2", resumable=True
                )
            )
            await self._closed.wait()

    sockets: list[_FakeLiveSession] = [_GatedSession(), _FakeLiveSession()]
    opened: list[_FakeLiveSession] = []

    @asynccontextmanager
    async def _connect(self: AsyncLive, **kwargs: object) -> AsyncIterator[_FakeLiveSession]:
        fake = sockets[len(opened)]
        opened.append(fake)
        yield fake

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", _connect)
    session = RealtimeModel().session()
    session._session_resumption_handle = "resume-1"
    session._resumption_chat_ctx = known
    session._chat_ctx = known
    await session.update_chat_ctx(updated)
    try:
        while session._session_resumption_handle != "resume-2":
            await asyncio.sleep(0.01)
        # the diff is queued but its send is blocked, so the new handle must not cover it
        assert [m.text_content for m in session._resumption_chat_ctx.messages()] == ["hello"]

        # restart before the send completes: the channel drain drops the queued diff
        await session.update_tools([_restart_tool])
        while len(opened) < 2:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert _texts(opened[1].sent) == [["queued behind the handle"]]
    finally:
        await session.aclose()


async def test_failed_send_with_a_queued_update_replays_each_item_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error restart drops the queue; the resume diff must be the only thing that re-sends."""
    from google.genai.live import AsyncLive

    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    first = known.copy()
    first.add_message(role="user", content="first update")

    class _FailingSession(_FakeLiveSession):
        """Blocks the first send until released, then fails it."""

        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()

        async def send_client_content(self, *, turns: object, turn_complete: bool) -> None:
            self.blocked.set()
            await self.release.wait()
            raise RuntimeError("socket gone")

    sockets: list[_FakeLiveSession] = [_FailingSession(), _FakeLiveSession()]
    opened: list[_FakeLiveSession] = []

    @asynccontextmanager
    async def _connect(self: AsyncLive, **kwargs: object) -> AsyncIterator[_FakeLiveSession]:
        fake = sockets[len(opened)]
        opened.append(fake)
        yield fake

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", _connect)
    session = RealtimeModel().session()
    session._session_resumption_handle = "resume-1"
    session._resumption_chat_ctx = known
    session._chat_ctx = known
    await session.update_chat_ctx(first)
    try:
        failing = sockets[0]
        assert isinstance(failing, _FailingSession)
        await asyncio.wait_for(failing.blocked.wait(), timeout=2)
        # a second update queues behind the blocked send
        second = first.copy()
        second.add_message(role="user", content="second update")
        await session.update_chat_ctx(second)
        failing.release.set()

        while len(opened) < 2:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert opened[0].sent == []
        assert _texts(opened[1].sent) == [["first update", "second update"]]
        assert session._unsent_item_ids == set()
    finally:
        await session.aclose()
