from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from google.genai import types

from livekit.agents import llm, utils
from livekit.plugins.google.realtime.api_proto import ClientEvents
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession
from livekit.plugins.google.utils import create_function_response

pytestmark = pytest.mark.unit

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
