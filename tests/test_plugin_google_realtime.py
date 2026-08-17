from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
from google.genai import types

from livekit import rtc
from livekit.agents import Agent, AgentSession, TurnHandlingOptions, llm, utils
from livekit.agents.llm.realtime import _UserMessageSyncStatus
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.plugins.google.realtime.api_proto import ClientEvents
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession
from livekit.plugins.google.utils import create_function_response

from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit

# 10ms of silence at the output sample rate (24kHz mono, 16-bit)
_PCM_FRAME = b"\x00\x01" * 240


@pytest.fixture(autouse=True)
async def _drain_google_client_finalizers() -> AsyncIterator[None]:
    """Finish google-genai's unconditional destructor cleanup in the owning test loop."""
    yield
    # AsyncClient.__del__ and BaseApiClient.__del__ schedule another aclose even after an
    # explicit close. Collect while the test loop is still active, then await only those
    # library finalizers so they cannot surface as leaked tasks in a later test.
    gc.collect()
    current = asyncio.current_task()
    finalizers = [
        task
        for task in asyncio.all_tasks()
        if task is not current
        and task.get_coro().__qualname__ in ("AsyncClient.aclose", "BaseApiClient.aclose")
    ]
    if finalizers:
        await asyncio.gather(*finalizers)


class _ActiveSessionStub:
    def __init__(self, *, block_close: bool = False) -> None:
        self.close_entered = asyncio.Event()
        self.close_release = asyncio.Event()
        if not block_close:
            self.close_release.set()

    async def close(self) -> None:
        self.close_entered.set()
        await self.close_release.wait()


class _RecordingInputSession(_ActiveSessionStub):
    def __init__(self) -> None:
        super().__init__()
        self.realtime_inputs: list[dict[str, object]] = []
        self.client_contents: list[dict[str, object]] = []

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.realtime_inputs.append(kwargs)

    async def send_client_content(self, **kwargs: object) -> None:
        self.client_contents.append(kwargs)


class _RecordingToolSession(_ActiveSessionStub):
    def __init__(self, *, fail_send: bool = False) -> None:
        super().__init__()
        self.fail_send = fail_send
        self.sent_order: list[str] = []
        self.realtime_inputs: list[dict[str, object]] = []
        self.tool_responses: list[list[types.FunctionResponse]] = []
        self.tool_response_sent = asyncio.Event()

    async def send_tool_response(self, *, function_responses: list[types.FunctionResponse]) -> None:
        if self.fail_send:
            raise RuntimeError("tool response send failed")
        self.sent_order.append("tool_response")
        self.tool_responses.append(function_responses)
        self.tool_response_sent.set()

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.sent_order.append("realtime_input")
        self.realtime_inputs.append(kwargs)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        await asyncio.Event().wait()
        if False:
            yield types.LiveServerMessage()


class _BlockingToolSession(_RecordingToolSession):
    def __init__(self) -> None:
        super().__init__()
        self.tool_response_send_started = asyncio.Event()
        self.release_tool_response = asyncio.Event()

    async def send_tool_response(self, *, function_responses: list[types.FunctionResponse]) -> None:
        self.tool_response_send_started.set()
        await self.release_tool_response.wait()
        await super().send_tool_response(function_responses=function_responses)


class _BlockingInputSession(_RecordingInputSession):
    def __init__(self) -> None:
        super().__init__()
        self.input_send_started = asyncio.Event()
        self.release_input_send = asyncio.Event()

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.input_send_started.set()
        await self.release_input_send.wait()
        await super().send_realtime_input(**kwargs)


async def _drain_queued_events(session: RealtimeSession, active: _RecordingInputSession) -> None:
    channel = session._msg_ch
    channel.close()
    await session._send_task(cast(Any, active), session._session_epoch, channel)


class _ControlledReceiveSession(_ActiveSessionStub):
    def __init__(
        self,
        response: types.LiveServerMessage,
        *,
        error_after_response: Exception | None = None,
    ) -> None:
        super().__init__()
        self.response = response
        self.error_after_response = error_after_response
        self.receive_started = asyncio.Event()
        self.release_response = asyncio.Event()

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        self.receive_started.set()
        await self.release_response.wait()
        yield self.response
        if self.error_after_response is not None:
            raise self.error_after_response


class _TimeoutHandle:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def _input_frame(duration_ms: int, *, fill: int = 0) -> rtc.AudioFrame:
    samples = 16_000 * duration_ms // 1000
    return rtc.AudioFrame(
        bytes([fill]) * samples * 2,
        sample_rate=16_000,
        num_channels=1,
        samples_per_channel=samples,
    )


@asynccontextmanager
async def _make_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manual_activity_detection: bool = False,
    activity_handling: types.ActivityHandling | None = None,
    session_resumption_handle: str | None = None,
) -> AsyncIterator[RealtimeSession]:
    """A session whose background connect loop is stopped before it hits the network.

    Closed on exit so the genai http clients are released here instead of by
    ``AsyncClient.__del__``, which schedules ``aclose()`` on whatever event loop
    is running when the collector happens to reach them.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    realtime_input_config = (
        types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
            activity_handling=activity_handling,
        )
        if manual_activity_detection
        else None
    )
    model_options: dict[str, object] = {}
    if realtime_input_config is not None:
        model_options["realtime_input_config"] = realtime_input_config
    if session_resumption_handle is not None:
        model_options["session_resumption"] = types.SessionResumptionConfig(
            handle=session_resumption_handle
        )
    model = RealtimeModel(**model_options)  # type: ignore[arg-type]
    session = model.session()
    # cancel the connect loop before the event loop ever schedules it, so no
    # websocket connection is attempted
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


async def test_manual_audio_activity_uses_one_generation_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An externally bounded audio turn must not be followed by a synthetic text turn."""
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        session.start_user_activity()
        session.push_audio(_input_frame(50))
        generation_fut = session.generate_reply()

        assert len(sent) == 3
        assert isinstance(sent[0], types.LiveClientRealtimeInput)
        assert sent[0].activity_start is not None
        assert isinstance(sent[1], types.LiveClientRealtimeInput)
        assert sent[1].audio is not None
        assert isinstance(sent[2], types.LiveClientRealtimeInput)
        assert sent[2].activity_end is not None
        assert not any(isinstance(event, types.LiveClientContent) for event in sent)

        session._start_new_generation()
        generation = await asyncio.wait_for(generation_fut, timeout=0.1)
        assert generation.user_initiated is True
        assert session._pending_generation_fut is None


async def test_manual_audio_flushes_each_turn_before_activity_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial chunks are delivered in their own turn and never mixed with later audio."""
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        session.start_user_activity()
        session.push_audio(_input_frame(70, fill=1))
        first_generation = session.generate_reply()
        session._start_new_generation()
        await first_generation
        session._mark_current_generation_done()

        session.start_user_activity()
        session.push_audio(_input_frame(30, fill=2))
        second_generation = session.generate_reply()

        audio_events = [
            event.audio
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert [len(blob.data or b"") for blob in audio_events] == [1600, 640, 960]
        assert audio_events[0].data == bytes([1]) * 1600
        assert audio_events[1].data == bytes([1]) * 640
        assert audio_events[2].data == bytes([2]) * 960

        ends = [
            index
            for index, event in enumerate(sent)
            if isinstance(event, types.LiveClientRealtimeInput) and event.activity_end is not None
        ]
        audio_indexes = [
            index
            for index, event in enumerate(sent)
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert len(ends) == 2
        assert audio_indexes[1] < ends[0] < audio_indexes[2] < ends[1]

        session._start_new_generation()
        await second_generation


async def test_default_server_turn_detection_needs_no_manual_generation_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        sent: list[object] = []
        created: list[llm.GenerationCreatedEvent] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        session.on("generation_created", created.append)

        session.push_audio(_input_frame(50))
        session._start_new_generation()

        assert len(sent) == 1
        assert isinstance(sent[0], types.LiveClientRealtimeInput)
        assert sent[0].audio is not None
        assert sent[0].activity_start is None and sent[0].activity_end is None
        assert len(created) == 1
        assert created[0].user_initiated is False
        assert session._pending_generation_fut is None
        session._mark_current_generation_done()


async def test_agent_activity_external_vad_drives_one_google_audio_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the framework-to-provider boundary without a network connection."""
    async with _make_session(monkeypatch, manual_activity_detection=True) as rt_session:
        agent_session = AgentSession(
            llm=rt_session.realtime_model,
            stt=FakeSTT(),
            vad=FakeVAD(fake_user_speeches=[]),
            turn_handling=TurnHandlingOptions(turn_detection="vad"),
        )
        activity = AgentActivity(Agent(instructions="test"), agent_session)
        activity._rt_session = rt_session
        activity._started = True
        activity._scheduling_paused = False
        sent: list[object] = []
        monkeypatch.setattr(rt_session, "_send_client_event", sent.append)

        generation_requested = asyncio.Event()
        real_generate_reply = rt_session.generate_reply

        def _observe_generate_reply(**kwargs: Any) -> asyncio.Future[llm.GenerationCreatedEvent]:
            future = real_generate_reply(**kwargs)
            generation_requested.set()
            return future

        monkeypatch.setattr(rt_session, "generate_reply", _observe_generate_reply)

        def _authorize_speech(speech: SpeechHandle, priority: int, force: bool = False) -> None:
            speech._mark_scheduled()
            speech._authorize_generation()

        monkeypatch.setattr(activity, "_schedule_speech", _authorize_speech)

        now = 10.0
        activity.on_start_of_speech(None, now)
        activity.push_audio(_input_frame(50))
        activity.on_end_of_speech(None)
        assert activity.on_end_of_turn(
            _EndOfTurnInfo(
                skip_reply=False,
                new_transcript="external transcript for local observability",
                transcript_confidence=0.9,
                metrics=_EndOfTurnMetrics(
                    started_speaking_at=now,
                    stopped_speaking_at=now + 0.05,
                    transcription_delay=0.01,
                    end_of_turn_delay=0.01,
                ),
                backchannel_over_agent=False,
            )
        )
        await asyncio.wait_for(generation_requested.wait(), timeout=0.1)

        starts = [
            event
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.activity_start is not None
        ]
        audio = [
            event
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        ends = [
            event
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.activity_end is not None
        ]
        assert len(starts) == len(audio) == len(ends) == 1
        assert not any(isinstance(event, types.LiveClientContent) for event in sent)
        pending_generation = rt_session._pending_generation_fut
        assert pending_generation is not None
        generation_epoch = rt_session._session_epoch

        # A new VAD turn can begin after the reply trigger was sent but before Gemini
        # acknowledges it. The framework must keep that boundary deferred; announcing it in
        # AUDIO_TRIGGER_SENT would restart the provider epoch and fail the preceding reply.
        activity.on_start_of_speech(None, now + 0.1)
        activity.push_audio(_input_frame(50, fill=2))
        deferred_input_ready = activity._deferred_realtime_audio_inputs[0].ready_fut

        assert rt_session._session_epoch == generation_epoch
        assert not pending_generation.done()
        assert (
            len(
                [
                    event
                    for event in sent
                    if isinstance(event, types.LiveClientRealtimeInput)
                    and event.activity_start is not None
                ]
            )
            == 1
        )

        rt_session._start_new_generation()
        await asyncio.wait_for(asyncio.shield(deferred_input_ready), timeout=0.1)

        assert rt_session._session_epoch == generation_epoch
        assert pending_generation.done() and pending_generation.exception() is None
        assert activity._rt_user_activity_started is True
        assert rt_session._activity_has_realtime_input is True
        assert (
            len(
                [
                    event
                    for event in sent
                    if isinstance(event, types.LiveClientRealtimeInput)
                    and event.activity_start is not None
                ]
            )
            == 2
        )

        activity.on_end_of_speech(None)
        rt_session._mark_current_generation_done()
        await asyncio.wait_for(asyncio.gather(*list(activity._speech_tasks)), timeout=0.5)
        assert rt_session._pending_generation_fut is None


async def test_application_generation_without_activity_keeps_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The placeholder remains for generation requests with no completed user input."""
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        generation_fut = session.generate_reply()

        assert len(sent) == 1
        assert isinstance(sent[0], types.LiveClientContent)
        assert sent[0].turn_complete is True
        assert sent[0].turns is not None
        assert [part.text for turn in sent[0].turns for part in (turn.parts or [])] == ["."]

        session._start_new_generation()
        assert (await generation_fut).user_initiated is True


async def test_application_generation_with_instructions_keeps_legacy_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        generation_fut = session.generate_reply(instructions="answer briefly")

        assert len(sent) == 1
        assert isinstance(sent[0], types.LiveClientContent)
        assert sent[0].turn_complete is True
        assert [
            (turn.role, part.text) for turn in sent[0].turns or [] for part in turn.parts or []
        ] == [("model", "answer briefly"), ("user", ".")]

        session._start_new_generation()
        await generation_fut


async def test_manual_activity_state_resets_between_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        for _ in range(2):
            session.start_user_activity()
            session.push_audio(_input_frame(50))
            generation_fut = session.generate_reply()
            session._start_new_generation()
            await generation_fut

        starts = [
            event
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.activity_start is not None
        ]
        ends = [
            event
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.activity_end is not None
        ]
        assert len(starts) == 2
        assert len(ends) == 2
        assert not any(isinstance(event, types.LiveClientContent) for event in sent)
        assert session._in_user_activity is False


async def test_empty_manual_activity_uses_fresh_legacy_generation_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        initial_epoch = session._session_epoch

        session.start_user_activity()
        generation_fut = session.generate_reply()

        assert session._session_epoch == initial_epoch + 1
        assert not any(
            isinstance(event, types.LiveClientRealtimeInput) and event.activity_end is not None
            for event in sent
        )
        placeholders = [
            event
            for event in sent
            if isinstance(event, types.LiveClientContent)
            and any(part.text == "." for turn in event.turns or [] for part in turn.parts or [])
        ]
        assert len(placeholders) == 1

        session._start_new_generation()
        assert (await generation_fut).user_initiated is True


async def test_restart_settles_pending_generation_without_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        monkeypatch.setattr(session, "_send_client_event", lambda event: None)
        session.start_user_activity()
        generation_fut = session.generate_reply()

        session._mark_restart_needed()

        with pytest.raises(llm.RealtimeError, match="restart"):
            await asyncio.wait_for(generation_fut, timeout=0.1)
        assert session._pending_generation_fut is None
        assert session._in_user_activity is False


async def test_stale_session_response_cannot_resolve_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_response = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(parts=[types.Part(text="stale")], role="model")
        )
    )
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        old_session = _ControlledReceiveSession(stale_response)
        session._active_session = cast(Any, old_session)
        old_epoch = session._session_epoch
        recv_task = asyncio.create_task(session._recv_task(cast(Any, old_session), old_epoch))
        await asyncio.wait_for(old_session.receive_started.wait(), timeout=0.1)

        session._mark_restart_needed()
        monkeypatch.setattr(session, "_send_client_event", lambda event: None)
        new_generation = session.generate_reply()
        old_session.release_response.set()
        await recv_task

        assert not new_generation.done()
        assert session._current_generation is None

        new_generation.cancel()
        await asyncio.sleep(0)
        session._active_session = None


async def test_second_generate_reply_supersedes_first_on_fresh_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        first = session.generate_reply()
        first_epoch = session._session_epoch
        second = session.generate_reply()

        assert first.cancelled()
        assert not second.done()
        assert session._session_epoch == first_epoch + 1
        assert session._pending_generation_fut is second

        session._start_new_generation()
        assert (await second).user_initiated is True


async def test_text_cancellation_does_not_rearm_completed_client_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _ActiveSessionStub()
        session._active_session = cast(Any, active)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="cancel this response")
        await session.update_chat_ctx(chat_ctx)
        cancelled = session.generate_reply()

        cancelled.cancel()
        await asyncio.sleep(0)
        assert session._client_content_user_turn_pending is False
        sent.clear()

        next_generation = session.generate_reply()
        assert len(sent) == 1
        assert isinstance(sent[0], types.LiveClientContent)
        assert sent[0].turn_complete is True
        assert [part.text for turn in sent[0].turns or [] for part in turn.parts or []] == ["."]

        session._start_new_generation()
        await next_generation
        session._active_session = None


async def test_completed_text_turn_does_not_rearm_after_fresh_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _ActiveSessionStub()
        session._active_session = cast(Any, active)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="already consumed")
        await session.update_chat_ctx(chat_ctx)
        completed = session.generate_reply()
        session._start_new_generation()
        await completed
        session._mark_current_generation_done()

        session._mark_restart_needed()
        sent.clear()
        next_generation = session.generate_reply()
        assert len(sent) == 1
        assert isinstance(sent[0], types.LiveClientContent)
        assert [part.text for turn in sent[0].turns or [] for part in turn.parts or []] == ["."]

        session._start_new_generation()
        await next_generation
        session._active_session = None


async def test_discard_restart_clears_resumption_handle_and_aborts_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._session_resumption_handle = "resume-me"
        session.start_user_activity()
        await _drain_queued_events(session, active)

        session.clear_audio()

        assert session.session_resumption_handle is None
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        with pytest.raises(llm.RealtimeError, match="discarded during a session restart"):
            await session.generate_reply()
        assert sent == []


async def test_resumable_restart_aborts_ambiguous_open_audio_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        session._session_resumption_handle = "resume-me"
        session.start_user_activity()
        session.push_audio(_input_frame(20))

        session._mark_restart_needed(resume_session=True)
        assert session.session_resumption_handle is None
        assert session._in_user_activity is False

        sent.clear()
        with pytest.raises(llm.RealtimeError, match="discarded during a session restart"):
            await session.generate_reply()
        assert sent == []


async def test_generation_timeout_discards_provider_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        monkeypatch.setattr(session, "_send_client_event", lambda event: None)
        loop = asyncio.get_running_loop()
        timeout_callbacks: list[object] = []

        def _capture_timeout(delay: float, callback: object) -> _TimeoutHandle:
            assert delay == 5.0
            timeout_callbacks.append(callback)
            return _TimeoutHandle()

        monkeypatch.setattr(loop, "call_later", _capture_timeout)
        initial_epoch = session._session_epoch
        generation_fut = session.generate_reply()
        cast(Any, timeout_callbacks[0])()

        with pytest.raises(llm.RealtimeError, match="timed out"):
            await generation_fut
        assert session._pending_generation_fut is None
        assert session._session_epoch == initial_epoch + 1


async def test_two_text_turns_each_send_one_message_and_one_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _ActiveSessionStub()
        session._active_session = cast(Any, active)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        for transcript in ("first external turn", "second external turn"):
            chat_ctx = session.chat_ctx.copy()
            chat_ctx.add_message(role="user", content=transcript)
            await session.update_chat_ctx(chat_ctx)
            generation_fut = session.generate_reply()
            session._start_new_generation()
            await generation_fut
            session._mark_current_generation_done()

        content_events = [event for event in sent if isinstance(event, types.LiveClientContent)]
        user_text = [
            part.text
            for event in content_events
            for turn in event.turns or []
            for part in turn.parts or []
        ]
        completions = [
            event for event in content_events if event.turn_complete is True and not event.turns
        ]
        assert user_text == ["first external turn", "second external turn"]
        assert len(completions) == 2
        assert session._client_content_user_turn_pending is False
        assert not any(text == "." for text in user_text)
        session._active_session = None


async def test_active_audio_ignores_unrepresentable_instructions_without_synthetic_turn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        session.start_user_activity()
        session.push_audio(_input_frame(50))
        with caplog.at_level(logging.WARNING):
            generation_fut = session.generate_reply(instructions="answer briefly")

        assert not any(isinstance(event, types.LiveClientContent) for event in sent)
        assert (
            sum(
                isinstance(event, types.LiveClientRealtimeInput) and event.activity_end is not None
                for event in sent
            )
            == 1
        )
        assert "ignoring instructions" in caplog.text

        session._start_new_generation()
        await generation_fut


async def test_cancelling_pending_generation_clears_manual_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        monkeypatch.setattr(session, "_send_client_event", lambda event: None)
        session.start_user_activity()
        generation_fut = session.generate_reply()

        generation_fut.cancel()
        await asyncio.sleep(0)

        assert generation_fut.cancelled()
        assert session._pending_generation_fut is None
        assert session._in_user_activity is False


async def test_pending_text_is_sent_then_completed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _ActiveSessionStub()
        session._active_session = cast(Any, active)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="edited external transcript")

        await session.update_chat_ctx(chat_ctx)
        generation_fut = session.generate_reply()

        content_events = [event for event in sent if isinstance(event, types.LiveClientContent)]
        assert len(content_events) == 2
        assert content_events[0].turn_complete is False
        assert [
            part.text for turn in content_events[0].turns or [] for part in turn.parts or []
        ] == ["edited external transcript"]
        assert content_events[1].turn_complete is True
        assert not content_events[1].turns

        session._start_new_generation()
        await generation_fut
        session._active_session = None


async def test_pending_text_with_instructions_does_not_inject_placeholder(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _ActiveSessionStub()
        session._active_session = cast(Any, active)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="external transcript")

        await session.update_chat_ctx(chat_ctx)
        with caplog.at_level(logging.WARNING):
            generation_fut = session.generate_reply(instructions="answer briefly")

        all_text = [
            part.text
            for event in sent
            if isinstance(event, types.LiveClientContent)
            for turn in event.turns or []
            for part in turn.parts or []
        ]
        assert all_text == ["external transcript"]
        assert "ignoring instructions" in caplog.text

        session._start_new_generation()
        await generation_fut
        session._active_session = None


async def test_interruption_only_activity_keeps_application_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        session.interrupt()
        generation_fut = session.generate_reply()

        assert any(
            isinstance(event, types.LiveClientRealtimeInput) and event.activity_start is not None
            for event in sent
        )
        assert not any(
            isinstance(event, types.LiveClientRealtimeInput) and event.activity_end is not None
            for event in sent
        )
        placeholder_events = [
            event
            for event in sent
            if isinstance(event, types.LiveClientContent)
            and any(part.text == "." for turn in event.turns or [] for part in turn.parts or [])
        ]
        assert len(placeholder_events) == 1

        session._start_new_generation()
        await generation_fut


async def test_pending_text_closes_interruption_activity_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _ActiveSessionStub()
        session._active_session = cast(Any, active)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        session.interrupt()
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="barge in text")

        await session.update_chat_ctx(chat_ctx)
        generation_fut = session.generate_reply()

        assert not any(
            isinstance(event, types.LiveClientRealtimeInput) and event.activity_end is not None
            for event in sent
        )
        assert not any(
            part.text == "."
            for event in sent
            if isinstance(event, types.LiveClientContent)
            for turn in event.turns or []
            for part in turn.parts or []
        )
        completions = [
            event
            for event in sent
            if isinstance(event, types.LiveClientContent)
            and event.turn_complete is True
            and not event.turns
        ]
        assert len(completions) == 1

        session._start_new_generation()
        await generation_fut
        session._active_session = None


async def test_restart_preserves_pending_text_completion_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="retry after reconnect")
        await session.update_chat_ctx(chat_ctx)
        assert session._client_content_user_turn_pending is True

        session._mark_restart_needed()
        assert session._client_content_user_turn_pending is True
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        generation_fut = session.generate_reply()

        assert len(sent) == 1
        assert isinstance(sent[0], types.LiveClientContent)
        assert sent[0].turn_complete is True
        assert not sent[0].turns

        session._start_new_generation()
        await generation_fut


async def test_update_chat_ctx_during_restart_is_replayed_not_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _ActiveSessionStub()
        session._active_session = cast(Any, active)
        session._session_should_close.set()
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        chat_ctx = llm.ChatContext.empty()
        message = chat_ctx.add_message(role="user", content="during restart")

        await session.update_chat_ctx(chat_ctx)

        assert sent == []
        assert session.chat_ctx.get_by_id(message.id) is not None
        assert session._client_content_user_turn_pending is True
        session._active_session = None


async def test_tool_response_added_during_resumption_is_sent_after_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = _RecordingToolSession()

    class _Connect:
        async def __aenter__(self) -> _RecordingToolSession:
            return replacement

        async def __aexit__(self, *_args: Any) -> None:
            return None

    def _connect(_live: Any, **_kwargs: Any) -> _Connect:
        return _Connect()

    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, _ActiveSessionStub())
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._session_should_close.set()
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())

        chat_ctx = session.chat_ctx.copy()
        output = _tool_output()
        chat_ctx.items.append(output)
        await session.update_chat_ctx(chat_ctx)

        assert session.chat_ctx.get_by_id(output.id) is not None
        assert not session._msg_ch.empty()

        session._active_session = None
        monkeypatch.setattr(type(session._client.aio.live), "connect", _connect)
        session._main_atask = asyncio.create_task(session._main_task())

        await asyncio.wait_for(replacement.tool_response_sent.wait(), timeout=1.0)

        assert len(replacement.tool_responses) == 1
        assert len(replacement.tool_responses[0]) == 1
        assert replacement.tool_responses[0][0].id == "fc_1"

        session._msg_ch.close()
        session._session_should_close.set()
        await asyncio.wait_for(asyncio.shield(session._main_atask), timeout=1.0)


async def test_resumable_restart_preserves_tool_response_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, _RecordingToolSession())
        session._session_should_close.clear()
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())

        chat_ctx = session.chat_ctx.copy()
        output = _tool_output()
        chat_ctx.items.append(output)
        await session.update_chat_ctx(chat_ctx)
        session._send_client_event(types.LiveClientRealtimeInput(text="later input"))

        assert len(session._tool_response_outbox) == 1

        session._mark_restart_needed(resume_session=True)
        session._mark_restart_needed(resume_session=True)
        assert len(session._tool_response_outbox) == 1

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        session._queue_pending_tool_responses()
        channel = session._msg_ch
        channel.close()
        await session._send_task(cast(Any, replacement), session._session_epoch, channel)

        assert len(replacement.tool_responses) == 1
        assert replacement.tool_responses[0][0].id == output.call_id
        assert replacement.sent_order == ["tool_response", "realtime_input"]
        assert replacement.realtime_inputs == [{"text": "later input"}]
        assert session._tool_response_outbox == {}
        session._active_session = None


async def test_tool_response_added_during_resumption_keeps_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, _RecordingToolSession())
        session._session_should_close.clear()
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())

        session._mark_restart_needed(resume_session=True)
        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output())
        await session.update_chat_ctx(chat_ctx)
        session._send_client_event(types.LiveClientRealtimeInput(text="later input"))

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        session._queue_pending_tool_responses()
        channel = session._msg_ch
        channel.close()
        await session._send_task(cast(Any, replacement), session._session_epoch, channel)

        assert len(replacement.tool_responses) == 1
        assert replacement.sent_order == ["tool_response", "realtime_input"]
        assert replacement.realtime_inputs == [{"text": "later input"}]
        assert session._tool_response_outbox == {}
        session._active_session = None


async def test_successful_tool_response_is_not_replayed_after_concurrent_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        active = _BlockingToolSession()
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, active)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output())
        await session.update_chat_ctx(chat_ctx)

        send_task = asyncio.create_task(
            session._send_task(cast(Any, active), session._session_epoch, session._msg_ch)
        )
        await asyncio.wait_for(active.tool_response_send_started.wait(), timeout=1.0)

        session._mark_restart_needed(resume_session=True)
        active.release_tool_response.set()
        await asyncio.wait_for(send_task, timeout=1.0)

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        session._queue_pending_tool_responses()
        replacement_channel = session._msg_ch
        replacement_channel.close()
        await session._send_task(
            cast(Any, replacement), session._session_epoch, replacement_channel
        )

        assert len(active.tool_responses) == 1
        assert replacement.tool_responses == []
        assert session._tool_response_outbox == {}
        session._active_session = None


async def test_late_tool_response_is_not_sent_to_fresh_provider_session(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, _RecordingToolSession())
        session._session_should_close.clear()
        session._provider_session_established = True
        session._start_new_generation()
        session._handle_tool_calls(_tool_call("old_epoch_call"))

        session._mark_restart_needed()
        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        session._provider_session_established = True

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output("old_epoch_call"))
        with caplog.at_level(logging.WARNING):
            await session.update_chat_ctx(chat_ctx)

        session._msg_ch.close()
        await session._send_task(cast(Any, replacement), session._session_epoch, session._msg_ch)

        assert replacement.tool_responses == []
        assert session._tool_response_outbox == {}
        assert any("old_epoch_call" in record.message for record in caplog.records)
        session._active_session = None


async def test_mixed_tool_outputs_send_only_calls_owned_by_current_provider(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _make_session(monkeypatch) as session:
        active = _RecordingToolSession()
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, active)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._start_new_generation()
        session._handle_tool_calls(_tool_call("current_call"))

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.extend([_tool_output("current_call"), _tool_output("abandoned_call")])
        with caplog.at_level(logging.WARNING):
            await session.update_chat_ctx(chat_ctx)

        channel = session._msg_ch
        channel.close()
        await session._send_task(cast(Any, active), session._session_epoch, channel)

        assert len(active.tool_responses) == 1
        assert [response.id for response in active.tool_responses[0]] == ["current_call"]
        assert session._tool_response_outbox == {}
        assert any("abandoned_call" in record.message for record in caplog.records)
        session._active_session = None


async def test_cancelled_tool_call_drops_queued_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        active = _RecordingToolSession()
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, active)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._start_new_generation()
        session._handle_tool_calls(_tool_call("cancelled_call"))

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output("cancelled_call"))
        await session.update_chat_ctx(chat_ctx)
        session._handle_tool_call_cancellation(
            types.LiveServerToolCallCancellation(ids=["cancelled_call"])
        )

        channel = session._msg_ch
        channel.close()
        await session._send_task(cast(Any, active), session._session_epoch, channel)

        assert active.tool_responses == []
        assert session._tool_response_outbox == {}
        assert session._delivered_tool_response_event_ids == set()
        session._active_session = None


async def test_cancelling_one_call_preserves_other_queued_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        active = _RecordingToolSession()
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, active)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._start_new_generation()
        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(id="cancelled_call", name="lookup", args={}),
                    types.FunctionCall(id="retained_call", name="lookup", args={}),
                ]
            )
        )

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.extend([_tool_output("cancelled_call"), _tool_output("retained_call")])
        await session.update_chat_ctx(chat_ctx)
        session._handle_tool_call_cancellation(
            types.LiveServerToolCallCancellation(ids=["cancelled_call"])
        )

        channel = session._msg_ch
        channel.close()
        await session._send_task(cast(Any, active), session._session_epoch, channel)

        assert len(active.tool_responses) == 1
        assert [response.id for response in active.tool_responses[0]] == ["retained_call"]
        assert session._tool_response_outbox == {}
        session._active_session = None


async def test_sender_cancellation_requeues_owned_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        active = _BlockingToolSession()
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, active)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output())
        await session.update_chat_ctx(chat_ctx)

        send_task = asyncio.create_task(
            session._send_task(cast(Any, active), session._session_epoch, session._msg_ch)
        )
        await asyncio.wait_for(active.tool_response_send_started.wait(), timeout=1.0)
        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task

        assert len(session._tool_response_outbox) == 1
        assert not session._msg_ch.empty()

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        channel = session._msg_ch
        channel.close()
        await session._send_task(cast(Any, replacement), session._session_epoch, channel)

        assert len(replacement.tool_responses) == 1
        assert session._tool_response_outbox == {}
        session._active_session = None


async def test_fresh_restart_prevents_late_send_from_recreating_outbox_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        active = _BlockingToolSession()
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, active)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output())
        await session.update_chat_ctx(chat_ctx)
        send_task = asyncio.create_task(
            session._send_task(cast(Any, active), session._session_epoch, session._msg_ch)
        )
        await asyncio.wait_for(active.tool_response_send_started.wait(), timeout=1.0)

        session._mark_restart_needed(resume_session=True)
        session._mark_restart_needed()
        active.release_tool_response.set()
        await asyncio.wait_for(send_task, timeout=1.0)

        assert session._tool_response_outbox == {}
        assert session._delivered_tool_response_event_ids == set()
        session._active_session = None


async def test_turn_complete_receive_end_starts_next_receive_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TurnCompleteReceiveSession(_ActiveSessionStub):
        def __init__(self) -> None:
            super().__init__()
            self.receive_calls = 0
            self.next_receive_started = asyncio.Event()

        async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
            self.receive_calls += 1
            if self.receive_calls == 1:
                yield types.LiveServerMessage(
                    server_content=types.LiveServerContent(turn_complete=True)
                )
                return
            self.next_receive_started.set()
            await asyncio.Event().wait()
            if False:
                yield types.LiveServerMessage()

    async with _make_session(monkeypatch) as session:
        active = _TurnCompleteReceiveSession()
        session._active_session = cast(Any, active)
        session._session_should_close.clear()
        initial_epoch = session._session_epoch
        session._start_new_generation(initial_epoch)

        recv_task = asyncio.create_task(session._recv_task(cast(Any, active), initial_epoch))
        await asyncio.wait_for(active.next_receive_started.wait(), timeout=1.0)

        assert active.receive_calls == 2
        assert session._session_epoch == initial_epoch
        assert not session._session_should_close.is_set()

        recv_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await recv_task
        session._active_session = None


async def test_resumption_revocation_drops_tool_lineage_before_fresh_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revocation = types.LiveServerMessage(
        session_resumption_update=types.LiveServerSessionResumptionUpdate(resumable=False)
    )

    async with _make_session(monkeypatch) as session:
        active = _ControlledReceiveSession(
            revocation,
            error_after_response=llm.RealtimeError("transport closed"),
        )
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, active)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._session_resumption_handle = "prior-handle"
        session._start_new_generation()
        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(id="pending_call", name="lookup", args={}),
                    types.FunctionCall(id="late_call", name="lookup", args={}),
                ]
            )
        )

        chat_ctx = session.chat_ctx.copy()
        pending_output = _tool_output("pending_call")
        chat_ctx.items.append(pending_output)
        await session.update_chat_ctx(chat_ctx)
        assert len(session._tool_response_outbox) == 1

        initial_epoch = session._session_epoch
        recv_task = asyncio.create_task(session._recv_task(cast(Any, active), initial_epoch))
        await asyncio.wait_for(active.receive_started.wait(), timeout=1.0)
        active.release_response.set()
        await asyncio.wait_for(recv_task, timeout=1.0)

        assert session.session_resumption_handle is None
        assert session._session_epoch == initial_epoch + 1
        assert session._tool_response_outbox == {}
        assert session._provider_tool_call_ids == set()

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output("late_call"))
        await session.update_chat_ctx(chat_ctx)
        assert session._tool_response_outbox == {}

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        channel = session._msg_ch
        channel.close()
        await session._send_task(cast(Any, replacement), session._session_epoch, channel)

        assert replacement.tool_responses == []
        session._active_session = None


async def test_send_failure_preserves_observed_tool_call_until_result_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRealtimeInputSession(_RecordingToolSession):
        async def send_realtime_input(self, **_kwargs: object) -> None:
            raise RuntimeError("realtime input send failed")

    async with _make_session(monkeypatch) as session:
        failing = _FailingRealtimeInputSession()
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, failing)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._start_new_generation()
        session._handle_tool_calls(_tool_call("pending_call"))
        session._send_client_event(types.LiveClientRealtimeInput(text="unrelated input"))

        failed_epoch = session._session_epoch
        await session._send_task(cast(Any, failing), failed_epoch, session._msg_ch)

        assert session._session_epoch == failed_epoch + 1
        assert session._session_should_close.is_set()
        assert session.session_resumption_handle == "resume-handle"
        assert session._provider_tool_call_ids == {"pending_call"}
        assert session._tool_response_outbox == {}

        chat_ctx = session.chat_ctx.copy()
        output = _tool_output("pending_call")
        chat_ctx.items.append(output)
        await session.update_chat_ctx(chat_ctx)

        assert len(session._tool_response_outbox) == 1

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        replacement_channel = session._msg_ch
        replacement_channel.close()
        await session._send_task(
            cast(Any, replacement), session._session_epoch, replacement_channel
        )

        assert len(replacement.tool_responses) == 1
        assert [response.id for response in replacement.tool_responses[0]] == ["pending_call"]
        assert session._tool_response_outbox == {}
        assert session._provider_tool_call_ids == set()
        session._active_session = None


async def test_tool_choice_none_rejection_send_failure_resumes_provider_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RejectingToolSession(_RecordingToolSession):
        def __init__(self, *, fail_send: bool) -> None:
            super().__init__(fail_send=fail_send)
            self.client_contents: list[dict[str, object]] = []

        async def send_client_content(self, **kwargs: object) -> None:
            self.client_contents.append(kwargs)

    async with _make_session(monkeypatch) as session:
        failing = _RejectingToolSession(fail_send=True)
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, failing)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._opts.tool_choice = "none"

        generation_fut = session.generate_reply()
        session._reject_tool_calls([types.FunctionCall(id="rejected_call", name="lookup", args={})])

        assert session._input_state.name == "IDLE"
        assert session._provider_turn_active
        assert not generation_fut.done()
        assert session._current_generation is None
        assert len(session._tool_response_outbox) == 1

        failed_epoch = session._session_epoch
        failed_channel = session._msg_ch
        failed_channel.close()
        await session._send_task(cast(Any, failing), failed_epoch, failed_channel)

        assert session._session_epoch == failed_epoch + 1
        assert session.session_resumption_handle == "resume-handle"
        assert session._provider_tool_call_ids == {"rejected_call"}
        assert len(session._tool_response_outbox) == 1
        with pytest.raises(llm.RealtimeError, match="restarted before generation started"):
            await generation_fut

        replacement = _RejectingToolSession(fail_send=False)
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        replacement_channel = session._msg_ch
        replacement_channel.close()
        await session._send_task(
            cast(Any, replacement), session._session_epoch, replacement_channel
        )

        assert len(replacement.tool_responses) == 1
        assert [response.id for response in replacement.tool_responses[0]] == ["rejected_call"]
        assert session._tool_response_outbox == {}
        assert session._provider_tool_call_ids == set()
        session._active_session = None


async def test_tool_choice_none_rejection_waits_for_direct_model_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._opts.tool_choice = "none"
        generation_fut = session.generate_reply()

        session._reject_tool_calls([types.FunctionCall(id="rejected_call", name="lookup", args={})])

        assert session._input_state.name == "IDLE"
        assert session._provider_turn_active
        assert session._current_generation is None
        assert not generation_fut.done()

        session._start_new_generation()
        generation_event = await generation_fut
        assert generation_event.user_initiated
        assert session._current_generation is not None

        session._handle_server_content(types.LiveServerContent(turn_complete=True))
        assert not session._provider_turn_active


async def test_tool_response_send_failure_retries_on_replacement_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        failing = _RecordingToolSession(fail_send=True)
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, failing)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())

        chat_ctx = session.chat_ctx.copy()
        output = _tool_output()
        chat_ctx.items.append(output)
        await session.update_chat_ctx(chat_ctx)

        failed_epoch = session._session_epoch
        failed_channel = session._msg_ch
        failed_channel.close()
        await session._send_task(cast(Any, failing), failed_epoch, failed_channel)

        assert session._session_epoch == failed_epoch + 1
        assert session._session_should_close.is_set()
        assert session.session_resumption_handle == "resume-handle"
        assert len(session._tool_response_outbox) == 1

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        session._queue_pending_tool_responses()
        replacement_channel = session._msg_ch
        replacement_channel.close()
        await session._send_task(
            cast(Any, replacement), session._session_epoch, replacement_channel
        )

        assert len(replacement.tool_responses) == 1
        assert replacement.tool_responses[0][0].id == output.call_id
        assert session._tool_response_outbox == {}
        session._active_session = None


async def test_tool_response_send_failure_is_not_replayed_on_fresh_session(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _make_session(monkeypatch) as session:
        failing = _RecordingToolSession(fail_send=True)
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, failing)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output())
        await session.update_chat_ctx(chat_ctx)

        failed_channel = session._msg_ch
        failed_channel.close()
        with caplog.at_level(logging.WARNING):
            await session._send_task(cast(Any, failing), session._session_epoch, failed_channel)

        assert session.session_resumption_handle is None
        assert session._tool_response_outbox == {}
        assert any("cannot be resumed" in record.message for record in caplog.records)

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        session._queue_pending_tool_responses()
        session._msg_ch.close()
        await session._send_task(cast(Any, replacement), session._session_epoch, session._msg_ch)

        assert replacement.tool_responses == []
        session._active_session = None


async def test_historical_tool_output_before_first_connection_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        chat_ctx = session.chat_ctx.copy()
        output = _tool_output()
        chat_ctx.items.append(output)

        await session.update_chat_ctx(chat_ctx)

        assert session.chat_ctx.get_by_id(output.id) is not None
        assert session._tool_response_outbox == {}
        assert session._msg_ch.empty()

        first_connection = _RecordingToolSession()
        session._active_session = cast(Any, first_connection)
        session._session_should_close.clear()
        session._provider_session_established = True
        session._queue_pending_tool_responses()
        session._msg_ch.close()
        await session._send_task(
            cast(Any, first_connection), session._session_epoch, session._msg_ch
        )

        assert first_connection.tool_responses == []
        session._active_session = None


async def test_initial_resumption_handle_does_not_replay_unobserved_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch, session_resumption_handle="external-resume-handle"
    ) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        chat_ctx = session.chat_ctx.copy()
        output = _tool_output("restored_call")
        chat_ctx.items.append(output)

        await session.update_chat_ctx(chat_ctx)

        assert session.chat_ctx.get_by_id(output.id) is not None
        assert session._tool_response_outbox == {}
        assert session._msg_ch.empty()

        replacement = _RecordingToolSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        channel = session._msg_ch
        channel.close()
        await session._send_task(cast(Any, replacement), session._session_epoch, channel)

        assert replacement.tool_responses == []
        assert session._tool_response_outbox == {}
        session._active_session = None


async def test_session_close_clears_unsent_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, _ActiveSessionStub())
        session._session_should_close.set()
        session._provider_session_established = True
        session._session_resumption_handle = "resume-handle"
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())
        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output())

        await session.update_chat_ctx(chat_ctx)
        assert len(session._tool_response_outbox) == 1

        await session.aclose()

        assert session._tool_response_outbox == {}
        assert session._msg_ch.closed


async def test_tool_response_during_fresh_reconnect_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = cast(Any, _ActiveSessionStub())
        session._session_should_close.set()
        session._provider_session_established = True
        chat_ctx = session.chat_ctx.copy()
        output = _tool_output()
        chat_ctx.items.append(output)

        with caplog.at_level(logging.WARNING):
            await session.update_chat_ctx(chat_ctx)

        assert session.chat_ctx.get_by_id(output.id) is not None
        assert session._tool_response_outbox == {}
        assert any("not owned" in record.message for record in caplog.records)
        session._active_session = None


async def test_new_text_during_resumption_forces_fresh_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._session_resumption_handle = "resume-me"
        chat_ctx = llm.ChatContext.empty()
        message = chat_ctx.add_message(role="user", content="arrived while reconnecting")

        await session.update_chat_ctx(chat_ctx)

        assert session.session_resumption_handle is None
        assert session.chat_ctx.get_by_id(message.id) is not None
        assert session._client_content_user_turn_pending is True


async def test_text_send_failure_uses_fresh_replayable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingSendSession(_ActiveSessionStub):
        async def send_client_content(self, **kwargs: object) -> None:
            raise RuntimeError("send failed")

    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _FailingSendSession()
        session._active_session = cast(Any, active)
        session._session_resumption_handle = "resume-me"
        session._msg_ch = utils.aio.Chan[Any]()
        chat_ctx = llm.ChatContext.empty()
        message = chat_ctx.add_message(role="user", content="replay after send failure")
        await session.update_chat_ctx(chat_ctx)

        await session._send_task(cast(Any, active), session._session_epoch)

        assert session.session_resumption_handle is None
        assert session.chat_ctx.get_by_id(message.id) is not None
        assert session._client_content_user_turn_pending is True
        session._active_session = None


async def test_audio_send_failure_aborts_unreplayable_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingSendSession(_ActiveSessionStub):
        async def send_realtime_input(self, **kwargs: object) -> None:
            raise RuntimeError("send failed")

    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _FailingSendSession()
        session._active_session = cast(Any, active)
        session._session_resumption_handle = "resume-me"
        session._msg_ch = utils.aio.Chan[Any]()
        session.start_user_activity()
        session.push_audio(_input_frame(50))

        await session._send_task(cast(Any, active), session._session_epoch)

        assert session.session_resumption_handle is None
        assert session._in_user_activity is False
        with pytest.raises(llm.RealtimeError, match="discarded during a session restart"):
            await session.generate_reply()
        session._active_session = None


async def test_repeated_restart_discards_new_epoch_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._mark_restart_needed()
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="abandoned generation")
        await session.update_chat_ctx(chat_ctx)
        generation_fut = session.generate_reply()
        assert not session._msg_ch.empty()

        session._mark_restart_needed()

        with pytest.raises(llm.RealtimeError, match="restart"):
            await generation_fut
        assert session._msg_ch.empty()
        assert session._pending_generation_fut is None


async def test_manual_clear_discards_partial_audio_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        session.push_audio(_input_frame(20))
        assert sent == []
        session.clear_audio()
        session.push_audio(_input_frame(30))

        assert not any(
            isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
            for event in sent
        )


async def test_close_settles_generation_before_blocked_transport_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        monkeypatch.setattr(session, "_send_client_event", lambda event: None)
        session.start_user_activity()
        generation_fut = session.generate_reply()
        active = _ActiveSessionStub(block_close=True)
        session._active_session = cast(Any, active)

        close_task = asyncio.create_task(session.aclose())
        await asyncio.wait_for(active.close_entered.wait(), timeout=0.1)

        assert generation_fut.cancelled()
        assert session._pending_generation_fut is None

        active.close_release.set()
        await close_task


async def test_completed_text_send_failure_leaves_next_turn_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingSendSession(_ActiveSessionStub):
        async def send_client_content(self, **kwargs: object) -> None:
            raise RuntimeError("send failed")

    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _FailingSendSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="retry this exact text")
        await session.update_chat_ctx(chat_ctx)
        failed_generation = session.generate_reply()

        await session._send_task(cast(Any, active), session._session_epoch)

        with pytest.raises(llm.RealtimeError, match="restart"):
            await failed_generation
        assert session._client_content_user_turn_pending is False

        next_ctx = session.chat_ctx.copy()
        next_ctx.add_message(role="user", content="new turn after failure")
        await session.update_chat_ctx(next_ctx)
        assert session._client_content_user_turn_pending is True

        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        retry_generation = session.generate_reply()

        assert len(sent) == 1
        assert isinstance(sent[0], types.LiveClientContent)
        assert sent[0].turn_complete is True
        assert not sent[0].turns
        assert [msg.raw_text_content for msg in session.chat_ctx.messages()] == [
            "retry this exact text",
            "new turn after failure",
        ]
        session._start_new_generation()
        await retry_generation
        session._active_session = None


async def test_completed_audio_send_failure_stays_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingSendSession(_ActiveSessionStub):
        async def send_realtime_input(self, **kwargs: object) -> None:
            raise RuntimeError("send failed")

    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _FailingSendSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session.start_user_activity()
        session.push_audio(_input_frame(50))
        failed_generation = session.generate_reply()

        await session._send_task(cast(Any, active), session._session_epoch)

        with pytest.raises(llm.RealtimeError, match="restart"):
            await failed_generation
        with pytest.raises(llm.RealtimeError, match="discarded during a session restart"):
            await session.generate_reply()
        session._active_session = None


async def test_idle_audio_does_not_make_next_manual_activity_nonempty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        session.push_audio(_input_frame(50))
        sent.clear()
        initial_epoch = session._session_epoch

        session.start_user_activity()
        generation_fut = session.generate_reply()

        assert session._session_epoch == initial_epoch + 1
        assert not any(
            isinstance(event, types.LiveClientRealtimeInput) and event.activity_end is not None
            for event in sent
        )
        placeholders = [
            part.text
            for event in sent
            if isinstance(event, types.LiveClientContent)
            for turn in event.turns or []
            for part in turn.parts or []
        ]
        assert placeholders == ["."]
        session._start_new_generation()
        await generation_fut


async def test_default_clear_audio_preserves_partial_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        session.push_audio(_input_frame(20, fill=1))
        session.clear_audio()
        session.push_audio(_input_frame(30, fill=2))

        audio = [
            event.audio
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert len(audio) == 1
        assert audio[0].data == bytes([1]) * 640 + bytes([2]) * 960


async def test_resumable_restart_migrates_queued_native_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_resumption_handle = "resume-me"
        session.push_audio(_input_frame(50))
        old_channel = session._msg_ch
        initial_epoch = session._session_epoch

        session._mark_restart_needed(resume_session=True)

        assert session._session_epoch == initial_epoch + 1
        assert session.session_resumption_handle == "resume-me"
        assert old_channel.closed
        assert session._msg_ch.qsize() == 1
        queued = session._msg_ch.recv_nowait()
        assert isinstance(queued, types.LiveClientRealtimeInput)
        assert queued.audio is not None


async def test_stale_send_task_does_not_consume_new_epoch_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        active = _ActiveSessionStub()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._send_client_event(types.LiveClientContent(turn_complete=True))
        channel = session._msg_ch
        session._session_should_close.set()

        await session._send_task(cast(Any, active), session._session_epoch, channel)

        assert channel.qsize() == 1
        session._active_session = None


async def test_cancel_then_close_cannot_reopen_transport_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._msg_ch = utils.aio.Chan[Any]()
        generation_fut = session.generate_reply()
        generation_fut.cancel()

        await session.aclose()

        assert generation_fut.cancelled()
        assert session._msg_ch.closed
        assert session._pending_generation_fut is None


async def test_stale_receiver_cannot_finalize_new_epoch_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_response = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(parts=[types.Part(text="stale")], role="model")
        )
    )
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        old_session = _ControlledReceiveSession(stale_response)
        session._active_session = cast(Any, old_session)
        old_epoch = session._session_epoch
        recv_task = asyncio.create_task(session._recv_task(cast(Any, old_session), old_epoch))
        await asyncio.wait_for(old_session.receive_started.wait(), timeout=0.1)

        session._mark_restart_needed()
        monkeypatch.setattr(session, "_send_client_event", lambda event: None)
        generation_fut = session.generate_reply()
        session._start_new_generation()
        await generation_fut
        current_generation = session._current_generation
        assert current_generation is not None and not current_generation._done

        old_session.release_response.set()
        await recv_task

        assert session._current_generation is current_generation
        assert not current_generation._done
        session._mark_current_generation_done()
        session._active_session = None


async def test_terminal_error_fails_current_and_future_generations_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        monkeypatch.setattr(session, "_send_client_event", lambda event: None)
        current_generation = session.generate_reply()
        client_closed = False
        real_client_aclose = session._client.aio.aclose

        async def _close_client() -> None:
            nonlocal client_closed
            client_closed = True
            await real_client_aclose()

        monkeypatch.setattr(session._client.aio, "aclose", _close_client)

        session._set_terminal_error(llm.RealtimeError("terminal transport failure"))

        assert current_generation.done()
        with pytest.raises(llm.RealtimeError, match="terminal transport failure"):
            await current_generation

        subsequent_generation = session.generate_reply()
        assert subsequent_generation.done()
        with pytest.raises(llm.RealtimeError, match="terminal transport failure"):
            await subsequent_generation

        assert session._pending_generation_fut is None
        assert session._msg_ch.closed
        assert not session._closed

        await session.aclose()
        assert session._closed
        assert client_closed


async def test_terminal_error_ignores_restart_and_input_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._msg_ch = utils.aio.Chan[Any]()
        session._set_terminal_error(llm.RealtimeError("terminal transport failure"))
        terminal_channel = session._msg_ch
        terminal_epoch = session._session_epoch
        terminal_input_state = session._input_state

        # Reply cleanup and user-input producers may still race with a terminal connection
        # failure. None of them may reopen the transport or leave input for a nonexistent
        # reconnect loop to consume.
        session.update_options(voice="Aoede")
        session.clear_audio()
        session.start_user_activity()
        session.push_audio(_input_frame(10))
        late_ctx = llm.ChatContext.empty()
        late_ctx.add_message(role="user", content="late user input")
        with pytest.raises(llm.RealtimeError, match="terminal transport failure"):
            await session.update_chat_ctx(late_ctx)

        assert session._msg_ch is terminal_channel
        assert terminal_channel.closed
        assert session._session_epoch == terminal_epoch
        assert session._input_state == terminal_input_state
        assert not session._activity_has_realtime_input
        assert not session._bstream._buf
        assert list(session.chat_ctx.messages()) == []


async def test_fatal_connect_error_completes_main_task_without_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingConnect:
        async def __aenter__(self) -> Any:
            raise RuntimeError("fatal connect failure")

        async def __aexit__(self, *_args: Any) -> None:
            return None

    def _connect(_live: Any, **_kwargs: Any) -> _FailingConnect:
        return _FailingConnect()

    async with _make_session(monkeypatch) as session:
        monkeypatch.setattr(type(session._client.aio.live), "connect", _connect)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._main_atask = asyncio.create_task(session._main_task())

        await asyncio.wait_for(asyncio.shield(session._main_atask), timeout=1.0)

        assert session._main_atask.done()
        assert not session._main_atask.cancelled()
        assert session._main_atask.exception() is None
        assert session._terminal_error is not None
        assert session._msg_ch.closed


async def test_quarantined_manual_audio_follows_activity_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)
        session.clear_audio()
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        session.push_audio(_input_frame(50, fill=2))
        session.start_user_activity()

        assert len(sent) == 2
        assert isinstance(sent[0], types.LiveClientRealtimeInput)
        assert sent[0].activity_start is not None
        assert isinstance(sent[1], types.LiveClientRealtimeInput)
        assert sent[1].audio is not None
        assert sent[1].audio.data == bytes([2]) * 1600


async def test_local_manual_audio_clear_preserves_active_output_and_resumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._start_new_generation()
        generation = session._current_generation
        assert generation is not None
        session._session_resumption_handle = "resume-current-output"
        initial_epoch = session._session_epoch

        # This tail has not reached the provider and needs no transport restart.
        session.push_audio(_input_frame(20, fill=1))
        session.clear_audio()

        assert not generation._done
        assert session._session_epoch == initial_epoch
        assert session.session_resumption_handle == "resume-current-output"


async def test_provider_visible_manual_clear_waits_for_active_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        generation = session._current_generation
        assert generation is not None
        session._session_resumption_handle = "resume-current-output"
        initial_epoch = session._session_epoch

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)
        session.clear_audio()

        assert not generation._done
        assert session._session_epoch == initial_epoch
        assert session.session_resumption_handle == "resume-current-output"

        session._handle_server_content(
            _audio_content(output_transcription=types.Transcription(text="still complete"))
        )
        assert generation.output_text == "still complete"

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert generation._done
        assert session._session_epoch == initial_epoch + 1
        assert session.session_resumption_handle is None


async def test_new_manual_activity_replays_only_post_clear_audio_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        generation = session._current_generation
        assert generation is not None
        initial_epoch = session._session_epoch

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)
        session.clear_audio()
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        # A repeated discard drops quarantined input but retains the one deferred restart.
        session.push_audio(_input_frame(50, fill=2))
        session.clear_audio()
        session.push_audio(_input_frame(50, fill=3))
        session.start_user_activity()

        assert generation._done
        assert session._session_epoch == initial_epoch + 1
        assert len(sent) == 2
        assert isinstance(sent[0], types.LiveClientRealtimeInput)
        assert sent[0].activity_start is not None
        assert isinstance(sent[1], types.LiveClientRealtimeInput)
        assert sent[1].audio is not None
        assert sent[1].audio.data == bytes([3]) * 1600


async def test_no_interruption_defers_next_manual_turn_until_output_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        generation = session._current_generation
        assert generation is not None
        initial_epoch = session._session_epoch

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)
        session.clear_audio()

        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)
        timeout_callbacks: list[object] = []
        loop = asyncio.get_running_loop()

        def _capture_timeout(delay: float, callback: object) -> _TimeoutHandle:
            assert delay == 5.0
            timeout_callbacks.append(callback)
            return _TimeoutHandle()

        monkeypatch.setattr(loop, "call_later", _capture_timeout)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        next_generation_fut = session.generate_reply()

        assert not generation._done
        assert session._session_epoch == initial_epoch
        assert sent == []
        assert timeout_callbacks == []
        assert not next_generation_fut.done()

        session._handle_server_content(
            _audio_content(output_transcription=types.Transcription(text="still complete"))
        )
        assert generation.output_text == "still complete"

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert generation._done
        assert session._session_epoch == initial_epoch + 1
        realtime_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(realtime_inputs) == 3
        assert realtime_inputs[0].activity_start is not None
        assert realtime_inputs[1].audio is not None
        assert realtime_inputs[1].audio.data == bytes([2]) * 1600
        assert realtime_inputs[2].activity_end is not None
        assert len(timeout_callbacks) == 1

        session._start_new_generation()
        generation_event = await next_generation_fut
        assert generation_event.user_initiated


async def _prepare_no_interruption_deferred_restart(
    session: RealtimeSession,
) -> tuple[Any, int]:
    active = _RecordingInputSession()
    session._active_session = cast(Any, active)
    session._msg_ch = utils.aio.Chan[Any]()
    session._session_should_close.clear()
    session._start_new_generation()
    generation = session._current_generation
    assert generation is not None
    initial_epoch = session._session_epoch

    session.start_user_activity()
    session.push_audio(_input_frame(50, fill=1))
    await _drain_queued_events(session, active)
    session.clear_audio()
    return generation, initial_epoch


async def test_no_interruption_deferred_video_replays_inside_owned_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, _ = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)
        monkeypatch.setattr(
            "livekit.plugins.google.realtime.realtime_api.images.encode",
            lambda frame, options: b"deferred-video",
        )

        session.start_user_activity()
        session.push_video(rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGB24, bytes(range(12))))
        generation_fut = session.generate_reply()

        assert sent == []
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        realtime_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(realtime_inputs) == 3
        assert realtime_inputs[0].activity_start is not None
        assert realtime_inputs[1].video is not None
        assert realtime_inputs[1].video.data == b"deferred-video"
        assert realtime_inputs[2].activity_end is not None
        assert not any(isinstance(event, types.LiveClientContent) for event in sent)

        session._start_new_generation()
        generation_event = await generation_fut
        assert generation_event.user_initiated


async def test_no_interruption_deferred_media_preserves_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, _ = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)
        monkeypatch.setattr(
            "livekit.plugins.google.realtime.realtime_api.images.encode",
            lambda frame, options: b"ordered-video",
        )

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        session.push_video(rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGB24, bytes(range(12))))
        session.push_audio(_input_frame(50, fill=3))
        generation_fut = session.generate_reply()

        assert sent == []
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        realtime_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(realtime_inputs) == 5
        assert realtime_inputs[0].activity_start is not None
        assert realtime_inputs[1].audio is not None
        assert realtime_inputs[1].audio.data == bytes([2]) * 1600
        assert realtime_inputs[2].video is not None
        assert realtime_inputs[2].video.data == b"ordered-video"
        assert realtime_inputs[3].audio is not None
        assert realtime_inputs[3].audio.data == bytes([3]) * 1600
        assert realtime_inputs[4].activity_end is not None

        session._start_new_generation()
        generation_event = await generation_fut
        assert generation_event.user_initiated


async def test_quarantined_video_waits_for_deferred_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, initial_epoch = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)
        monkeypatch.setattr(
            "livekit.plugins.google.realtime.realtime_api.images.encode",
            lambda frame, options: b"quarantined-video",
        )

        session.push_audio(_input_frame(50, fill=2))
        session.push_video(rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGB24, bytes(range(12))))

        # Neither medium may leak onto the abandoned provider epoch before the next
        # logical activity owns the quarantined residue.
        assert sent == []

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=3))
        generation_fut = session.generate_reply()
        assert sent == []

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        realtime_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(realtime_inputs) == 5
        assert realtime_inputs[0].activity_start is not None
        assert realtime_inputs[1].audio is not None
        assert realtime_inputs[1].audio.data == bytes([2]) * 1600
        assert realtime_inputs[2].video is not None
        assert realtime_inputs[2].video.data == b"quarantined-video"
        assert realtime_inputs[3].audio is not None
        assert realtime_inputs[3].audio.data == bytes([3]) * 1600
        assert realtime_inputs[4].activity_end is not None
        assert session._session_epoch == initial_epoch + 1

        session._start_new_generation()
        generation_event = await generation_fut
        assert generation_event.user_initiated


async def test_no_interruption_preserves_pre_activity_audio_in_deferred_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        generation, initial_epoch = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        # Manual mode accepts audio before ActivityStart. The deferred transaction must own
        # this partial buffered prefix just as an immediately started activity does.
        session.push_audio(_input_frame(20, fill=2))
        session.start_user_activity()
        next_generation_fut = session.generate_reply()

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert generation._done
        assert session._session_epoch == initial_epoch + 1
        realtime_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(realtime_inputs) == 3
        assert realtime_inputs[0].activity_start is not None
        assert realtime_inputs[1].audio is not None
        assert realtime_inputs[1].audio.data == bytes([2]) * 640
        assert realtime_inputs[2].activity_end is not None
        assert not any(isinstance(event, types.LiveClientContent) for event in sent)

        session._start_new_generation()
        generation_event = await next_generation_fut
        assert generation_event.user_initiated


async def test_no_interruption_deferred_turn_preserves_audio_beyond_quarantine_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, _ = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        session.push_audio(_input_frame(500, fill=2))
        session.push_audio(_input_frame(1_000, fill=3))
        generation_fut = session.generate_reply()

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        audio_payloads = [
            event.audio.data
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert audio_payloads == [bytes([2]) * 1600] * 10 + [bytes([3]) * 1600] * 20

        session._start_new_generation()
        generation_event = await generation_fut
        assert generation_event.user_initiated


async def test_no_interruption_deferred_turn_becomes_live_without_losing_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, initial_epoch = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        session.push_audio(_input_frame(500, fill=2))

        # The provider becomes available while this user turn is still active. Its buffered
        # prefix must become live input before subsequent audio, without a second ActivityStart.
        session._handle_server_content(types.LiveServerContent(turn_complete=True))
        assert session._session_epoch == initial_epoch + 1

        session.push_audio(_input_frame(500, fill=3))
        generation_fut = session.generate_reply()

        realtime_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert realtime_inputs[0].activity_start is not None
        assert realtime_inputs[-1].activity_end is not None
        audio_payloads = [event.audio.data for event in realtime_inputs if event.audio is not None]
        assert audio_payloads == [bytes([2]) * 1600] * 10 + [bytes([3]) * 1600] * 10
        assert not any(isinstance(event, types.LiveClientContent) for event in sent)

        session._start_new_generation()
        generation_event = await generation_fut
        assert generation_event.user_initiated


async def test_no_interruption_seals_deferred_audio_at_generate_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, _ = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        next_generation_fut = session.generate_reply()
        # Audio arriving after EOU cannot be appended to the already committed turn.
        session.push_audio(_input_frame(50, fill=3))

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        audio_payloads = [
            event.audio.data
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert audio_payloads == [bytes([2]) * 1600]

        session._start_new_generation()
        await next_generation_fut

        # The post-EOU frame is preserved as pre-roll, then enters the following activity
        # exactly once instead of leaking into or being dropped with the sealed turn.
        session.start_user_activity()
        following_generation_fut = session.generate_reply()
        audio_payloads = [
            event.audio.data
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert audio_payloads == [bytes([2]) * 1600]

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        audio_payloads = [
            event.audio.data
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert audio_payloads == [bytes([2]) * 1600, bytes([3]) * 1600]
        session._start_new_generation()
        await following_generation_fut


async def test_no_interruption_cancelled_deferred_turn_cannot_leak_into_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, _ = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        cancelled_generation = session.generate_reply()
        cancelled_generation.cancel()
        assert cancelled_generation.cancelled()

        # Do not yield to the event loop: public lifecycle boundaries must synchronously reap
        # cancelled ownership instead of depending on a scheduled Future callback.
        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=3))
        replacement_generation = session.generate_reply()
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        audio_payloads = [
            event.audio.data
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert audio_payloads == [bytes([3]) * 1600]

        session._start_new_generation()
        generation_event = await replacement_generation
        assert generation_event.user_initiated


async def test_no_interruption_queues_multiple_complete_turns_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, initial_epoch = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        first_generation_fut = session.generate_reply()

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=3))
        second_generation_fut = session.generate_reply()

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert session._session_epoch == initial_epoch + 1
        assert not first_generation_fut.cancelled()
        assert not second_generation_fut.cancelled()
        first_turn_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(first_turn_inputs) == 3
        assert first_turn_inputs[0].activity_start is not None
        assert first_turn_inputs[1].audio is not None
        assert first_turn_inputs[1].audio.data == bytes([2]) * 1600
        assert first_turn_inputs[2].activity_end is not None

        session._start_new_generation()
        first_event = await first_generation_fut
        assert first_event.user_initiated
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        all_turn_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(all_turn_inputs) == 6
        assert all_turn_inputs[3].activity_start is not None
        assert all_turn_inputs[4].audio is not None
        assert all_turn_inputs[4].audio.data == bytes([3]) * 1600
        assert all_turn_inputs[5].activity_end is not None
        assert session._session_epoch == initial_epoch + 1

        session._start_new_generation()
        second_event = await second_generation_fut
        assert second_event.user_initiated


async def test_no_interruption_cancelled_fifo_head_does_not_block_later_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, initial_epoch = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        cancelled_generation = session.generate_reply()

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=3))
        retained_generation = session.generate_reply()

        cancelled_generation.cancel()
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert cancelled_generation.cancelled()
        assert not retained_generation.cancelled()
        audio_payloads = [
            event.audio.data
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert audio_payloads == [bytes([3]) * 1600]
        assert session._session_epoch == initial_epoch + 1

        session._start_new_generation()
        generation_event = await retained_generation
        assert generation_event.user_initiated


async def test_no_interruption_clear_discards_fifo_tail_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        generation, initial_epoch = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        retained_generation = session.generate_reply()

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=3))
        discarded_generation = session.generate_reply()
        session.clear_audio()

        with pytest.raises(llm.RealtimeError, match="discarded before generation started"):
            await discarded_generation
        assert not retained_generation.done()
        assert not generation._done
        assert session._session_epoch == initial_epoch

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        audio_payloads = [
            event.audio.data
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert audio_payloads == [bytes([2]) * 1600]
        assert session._session_epoch == initial_epoch + 1

        session._start_new_generation()
        generation_event = await retained_generation
        assert generation_event.user_initiated


async def test_no_interruption_repeated_clear_discards_only_staged_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        generation, initial_epoch = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        discarded_generation = session.generate_reply()
        session.clear_audio()

        with pytest.raises(llm.RealtimeError, match="discarded before generation started"):
            await discarded_generation
        assert not generation._done
        assert session._session_epoch == initial_epoch
        assert sent == []

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=3))
        replacement_generation = session.generate_reply()
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        audio_payloads = [
            event.audio.data
            for event in sent
            if isinstance(event, types.LiveClientRealtimeInput) and event.audio is not None
        ]
        assert audio_payloads == [bytes([3]) * 1600]

        session._start_new_generation()
        await replacement_generation


async def test_no_interruption_empty_deferred_turn_keeps_legacy_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, _ = await _prepare_no_interruption_deferred_restart(session)
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)

        session.start_user_activity()
        next_generation_fut = session.generate_reply(instructions="answer briefly")
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert len(sent) == 1
        event = sent[0]
        assert isinstance(event, types.LiveClientContent)
        assert event.turn_complete is True
        assert event.turns is not None
        assert [part.text for turn in event.turns for part in turn.parts or []] == [
            "answer briefly",
            ".",
        ]

        session._start_new_generation()
        generation_event = await next_generation_fut
        assert generation_event.user_initiated


async def test_session_close_cancels_deferred_manual_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        _, _ = await _prepare_no_interruption_deferred_restart(session)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        first_generation_fut = session.generate_reply()

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=3))
        second_generation_fut = session.generate_reply()
        assert session._pending_generation_fut is None

        await session.aclose()

        assert first_generation_fut.cancelled()
        assert second_generation_fut.cancelled()
        assert not session._deferred_manual_inputs
        assert not session._deferred_manual_input_pipeline_active
        assert session._pending_generation_fut is None
        assert not session._quarantined_manual_inputs


async def test_terminal_error_fails_all_deferred_fifo_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ) as session:
        await _prepare_no_interruption_deferred_restart(session)

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        first_generation_fut = session.generate_reply()

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=3))
        second_generation_fut = session.generate_reply()

        session._set_terminal_error(llm.RealtimeError("terminal FIFO failure"))

        with pytest.raises(llm.RealtimeError, match="terminal FIFO failure"):
            await first_generation_fut
        with pytest.raises(llm.RealtimeError, match="terminal FIFO failure"):
            await second_generation_fut
        assert not session._deferred_manual_inputs
        assert not session._deferred_manual_input_pipeline_active
        assert session._pending_generation_fut is None
        assert not session._quarantined_manual_inputs


async def test_manual_clear_invalidates_queued_input_while_send_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BlockingSendSession(_ActiveSessionStub):
        def __init__(self) -> None:
            super().__init__()
            self.first_send_started = asyncio.Event()
            self.release_first_send = asyncio.Event()
            self.realtime_inputs: list[dict[str, object]] = []

        async def send_realtime_input(self, **kwargs: object) -> None:
            if not self.realtime_inputs:
                self.first_send_started.set()
                await self.release_first_send.wait()
            self.realtime_inputs.append(kwargs)

    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _BlockingSendSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        initial_epoch = session._session_epoch

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        send_task = asyncio.create_task(
            session._send_task(cast(Any, active), initial_epoch, session._msg_ch)
        )
        await asyncio.wait_for(active.first_send_started.wait(), timeout=0.1)

        session.clear_audio()
        assert session._session_epoch == initial_epoch

        active.release_first_send.set()
        session._msg_ch.close()
        await asyncio.wait_for(send_task, timeout=0.1)

        assert len(active.realtime_inputs) == 1
        assert active.realtime_inputs[0].get("activity_start") is not None

        session._handle_server_content(types.LiveServerContent(turn_complete=True))
        assert session._session_epoch == initial_epoch + 1
        session._active_session = None


async def test_manual_clear_removes_exact_pending_text_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        initial_epoch = session._session_epoch

        chat_ctx = llm.ChatContext.empty()
        kept_user = chat_ctx.add_message(role="user", content="keep user")
        kept_assistant = chat_ctx.add_message(role="assistant", content="keep assistant")
        discarded = chat_ctx.add_message(role="user", content="discard exact pending input")
        await session.update_chat_ctx(chat_ctx)
        await _drain_queued_events(session, active)
        assert len(active.client_contents) == 1

        session.clear_audio()

        assert session.chat_ctx.get_by_id(kept_user.id) is not None
        assert session.chat_ctx.get_by_id(kept_assistant.id) is not None
        assert session.chat_ctx.get_by_id(discarded.id) is None
        assert session._session_epoch == initial_epoch

        session._handle_server_content(types.LiveServerContent(turn_complete=True))
        assert session._session_epoch == initial_epoch + 1
        assert session.chat_ctx.get_by_id(discarded.id) is None


async def test_deferred_clear_waits_for_provider_turn_after_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        generation = session._current_generation
        assert generation is not None
        initial_epoch = session._session_epoch

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)
        session.clear_audio()
        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[types.FunctionCall(id="call-1", name="lookup", args={})]
            )
        )

        assert generation._done
        assert session._session_epoch == initial_epoch

        session._handle_server_content(types.LiveServerContent(turn_complete=True))
        assert session._session_epoch == initial_epoch + 1


async def test_recoverable_restart_preserves_quarantined_manual_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)
        session.clear_audio()
        session.push_audio(_input_frame(50, fill=2))
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)

        session._mark_restart_needed(on_error=True, resume_session=True)
        session.start_user_activity()

        realtime_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(realtime_inputs) == 2
        assert realtime_inputs[0].activity_start is not None
        assert realtime_inputs[1].audio is not None
        assert realtime_inputs[1].audio.data == bytes([2]) * 1600


async def test_manual_audio_quarantine_truncation_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)
        session.clear_audio()

        with caplog.at_level(logging.WARNING):
            for fill in range(2, 23):
                session.push_audio(_input_frame(50, fill=fill))

        assert "manual audio quarantine exceeded" in caplog.text
        assert 0 < session._quarantined_manual_audio_duration <= 1.0
        assert len(session._quarantined_manual_inputs) < 21
        first_audio = session._quarantined_manual_inputs[0].realtime_input.audio
        last_audio = session._quarantined_manual_inputs[-1].realtime_input.audio
        assert first_audio is not None
        assert last_audio is not None
        assert first_audio.data != bytes([2]) * 1600
        assert last_audio.data == bytes([22]) * 1600


async def test_manual_media_quarantine_bounds_audio_and_video_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._manual_audio_quarantine_active = True
        encoded = iter((b"video-1", b"video-2", b"video-3"))
        monkeypatch.setattr(
            "livekit.plugins.google.realtime.realtime_api.images.encode",
            lambda frame, options: next(encoded),
        )

        session.push_audio(_input_frame(500, fill=1))
        for _ in range(3):
            session.push_video(rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGB24, bytes(range(12))))
        session.push_audio(_input_frame(600, fill=2))

        retained = list(session._quarantined_manual_inputs)
        retained_video = [
            item.realtime_input.video for item in retained if item.realtime_input.video
        ]
        retained_audio = [item for item in retained if item.realtime_input.audio]
        assert session._quarantined_manual_audio_duration <= 1.0
        assert 0 < len(retained_audio) <= 20
        assert len(retained_video) == 1
        assert retained_video[0].data == b"video-3"
        video_index = next(i for i, item in enumerate(retained) if item.realtime_input.video)
        assert any(item.realtime_input.audio for item in retained[:video_index])
        assert any(item.realtime_input.audio for item in retained[video_index + 1 :])


@pytest.mark.parametrize("old_send_succeeds", [False, True])
async def test_resumable_restart_preserves_exact_dequeued_input(
    monkeypatch: pytest.MonkeyPatch,
    old_send_succeeds: bool,
) -> None:
    async with _make_session(monkeypatch, session_resumption_handle="resume-handle") as session:
        active = _BlockingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        event = types.LiveClientRealtimeInput(text="owned dequeued input")
        assert session._send_input_event(event)
        old_epoch = session._session_epoch
        send_task = asyncio.create_task(
            session._send_task(cast(Any, active), old_epoch, session._msg_ch)
        )
        await asyncio.wait_for(active.input_send_started.wait(), timeout=1.0)

        session._mark_restart_needed(resume_session=True)
        if old_send_succeeds:
            active.release_input_send.set()
            await asyncio.wait_for(send_task, timeout=1.0)
        else:
            send_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await send_task

        replacement = _RecordingInputSession()
        session._active_session = cast(Any, replacement)
        session._session_should_close.clear()
        replacement_channel = session._msg_ch
        replacement_channel.close()
        await session._send_task(
            cast(Any, replacement), session._session_epoch, replacement_channel
        )

        assert active.realtime_inputs == (
            [{"text": "owned dequeued input"}] if old_send_succeeds else []
        )
        assert replacement.realtime_inputs == (
            [] if old_send_succeeds else [{"text": "owned dequeued input"}]
        )
        session._active_session = None


async def test_manual_clear_invalidates_dequeued_input_before_second_send_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ObservedChannel:
        def __init__(self) -> None:
            self._inner = utils.aio.Chan[Any]()
            self.recv_started = asyncio.Event()
            self.dequeued = asyncio.Event()

        @property
        def closed(self) -> bool:
            return self._inner.closed

        def empty(self) -> bool:
            return self._inner.empty()

        def send_nowait(self, value: object) -> None:
            self._inner.send_nowait(value)

        def recv_nowait(self) -> object:
            return self._inner.recv_nowait()

        async def recv(self) -> object:
            self.recv_started.set()
            value = await self._inner.recv()
            self.dequeued.set()
            return value

        def close(self) -> None:
            self._inner.close()

    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        channel = _ObservedChannel()
        session._active_session = cast(Any, active)
        session._msg_ch = cast(Any, channel)
        session._session_should_close.clear()
        initial_epoch = session._session_epoch
        send_task = asyncio.create_task(
            session._send_task(cast(Any, active), initial_epoch, cast(Any, channel))
        )
        await asyncio.wait_for(channel.recv_started.wait(), timeout=0.1)

        await session._session_lock.acquire()
        try:
            session.start_user_activity()
            await asyncio.wait_for(channel.dequeued.wait(), timeout=0.1)
            session.clear_audio()
        finally:
            session._session_lock.release()

        channel.close()
        await asyncio.wait_for(send_task, timeout=0.1)

        assert active.realtime_inputs == []
        assert session._session_epoch == initial_epoch + 1
        session._active_session = None


async def test_tool_continuation_preserves_concurrent_manual_input_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        first_generation = session._current_generation
        assert first_generation is not None
        initial_epoch = session._session_epoch

        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[types.FunctionCall(id="call-1", name="lookup", args={})]
            )
        )
        assert first_generation._done

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)

        session._start_new_generation()
        continuation = session._current_generation
        assert continuation is not None
        assert continuation is not first_generation

        session.clear_audio()
        assert session._session_epoch == initial_epoch

        session._handle_server_content(types.LiveServerContent(turn_complete=True))
        assert continuation._done
        assert session._session_epoch == initial_epoch + 1


async def test_tool_cancellation_does_not_preserve_committed_input_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        first_generation = session._current_generation
        assert first_generation is not None
        initial_epoch = session._session_epoch

        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[types.FunctionCall(id="call-1", name="lookup", args={})]
            )
        )
        assert first_generation._done

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        reply_fut = session.generate_reply()
        await _drain_queued_events(session, active)
        session._msg_ch = utils.aio.Chan[Any]()
        assert active.realtime_inputs[-1].get("activity_end") is not None

        session._handle_tool_call_cancellation(types.LiveServerToolCallCancellation(ids=["call-1"]))
        session._start_new_generation()
        generation_event = await asyncio.wait_for(reply_fut, timeout=0.1)
        assert generation_event.user_initiated

        session.start_user_activity()

        assert session._session_epoch == initial_epoch
        session._active_session = None


async def test_tool_continuation_preserves_pre_activity_audio_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        session._start_new_generation()
        first_generation = session._current_generation
        assert first_generation is not None
        initial_epoch = session._session_epoch

        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[types.FunctionCall(id="call-1", name="lookup", args={})]
            )
        )
        assert first_generation._done

        # Manual mode accepts provider-bound audio before an explicit ActivityStart.
        session.push_audio(_input_frame(50, fill=1))
        await _drain_queued_events(session, active)
        assert active.realtime_inputs[-1].get("audio") is not None

        session._start_new_generation()
        continuation = session._current_generation
        assert continuation is not None
        session.clear_audio()
        assert session._session_epoch == initial_epoch

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert continuation._done
        assert session._session_epoch == initial_epoch + 1
        session._active_session = None


async def test_legacy_generation_preserves_pre_activity_audio_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        initial_epoch = session._session_epoch

        session.push_audio(_input_frame(50, fill=1))
        reply_fut = session.generate_reply()
        await _drain_queued_events(session, active)
        assert active.realtime_inputs[-1].get("audio") is not None
        assert active.client_contents[-1].get("turn_complete") is True

        session._start_new_generation()
        generation = session._current_generation
        assert generation is not None
        generation_event = await asyncio.wait_for(reply_fut, timeout=0.1)
        assert generation_event.user_initiated

        session.clear_audio()
        assert session._session_epoch == initial_epoch
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert generation._done
        assert session._session_epoch == initial_epoch + 1
        session._active_session = None


async def test_generation_preserves_next_sequence_pre_activity_audio_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        active = _RecordingInputSession()
        session._active_session = cast(Any, active)
        session._msg_ch = utils.aio.Chan[Any]()
        session._session_should_close.clear()
        initial_epoch = session._session_epoch

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        reply_fut = session.generate_reply()
        await _drain_queued_events(session, active)

        # Turn S is committed, but pre-activity audio for S+1 can reach the provider before
        # generation S starts.
        session._msg_ch = utils.aio.Chan[Any]()
        session.push_audio(_input_frame(50, fill=2))
        await _drain_queued_events(session, active)
        assert active.realtime_inputs[-1].get("audio") is not None
        assert active.realtime_inputs[-1]["audio"].data == bytes([2]) * 1600

        session._start_new_generation()
        generation = session._current_generation
        assert generation is not None
        generation_event = await asyncio.wait_for(reply_fut, timeout=0.1)
        assert generation_event.user_initiated

        session.clear_audio()
        assert session._session_epoch == initial_epoch
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert generation._done
        assert session._session_epoch == initial_epoch + 1
        session._active_session = None


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
        session._provider_session_established = True
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


async def test_sync_user_message_rejects_an_unqueued_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._active_session = cast(Any, _ActiveSessionStub())
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        monkeypatch.setattr(session, "_send_client_event", lambda event: False)
        chat_ctx = llm.ChatContext.empty()
        user_message = chat_ctx.add_message(role="user", content="must be queued")

        result = await session._sync_user_message(chat_ctx, user_message.id)

        assert result.status == _UserMessageSyncStatus.REJECTED
        assert isinstance(result.error, llm.RealtimeError)
        assert session.chat_ctx.get_by_id(user_message.id) is None
        assert session._pending_text_input_item_id is None
        session._active_session = None


async def test_sync_user_message_classifies_terminal_queue_race_as_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:

        async def fail_before_queue(
            chat_ctx: llm.ChatContext, *, target_message_id: str | None = None
        ) -> bool | None:
            del chat_ctx, target_message_id
            raise llm.RealtimeError("transport became terminal before queue insertion")

        monkeypatch.setattr(session, "_update_chat_ctx", fail_before_queue)
        chat_ctx = llm.ChatContext.empty()
        user_message = chat_ctx.add_message(role="user", content="race")

        result = await session._sync_user_message(chat_ctx, user_message.id)

        assert result.status == _UserMessageSyncStatus.REJECTED
        assert isinstance(result.error, llm.RealtimeError)


async def test_pending_user_text_survives_unrelated_assistant_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._active_session = cast(Any, _ActiveSessionStub())
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        chat_ctx = llm.ChatContext.empty()
        user_message = chat_ctx.add_message(role="user", content="owned external turn")
        await session.update_chat_ctx(chat_ctx)

        chat_ctx = session.chat_ctx
        chat_ctx.add_message(role="assistant", content="unrelated local append")
        await session.update_chat_ctx(chat_ctx)

        assert session._client_content_user_turn_pending is True
        assert session._pending_text_input_item_id == user_message.id

        generation_fut = session.generate_reply()
        all_text = [
            part.text
            for event in sent
            if isinstance(event, types.LiveClientContent)
            for turn in event.turns or []
            for part in turn.parts or []
        ]
        assert all_text == ["owned external turn", "unrelated local append"]
        assert "." not in all_text

        session._start_new_generation()
        await generation_fut
        session._active_session = None


async def test_batched_history_ending_with_assistant_does_not_open_text_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch, manual_activity_detection=True) as session:
        session._active_session = cast(Any, _ActiveSessionStub())
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="historical question")
        chat_ctx.add_message(role="assistant", content="historical answer")

        await session.update_chat_ctx(chat_ctx)

        assert session._client_content_user_turn_pending is False
        assert session._pending_text_input_item_id is None
        generation_fut = session.generate_reply()
        placeholders = [
            part.text
            for event in sent
            if isinstance(event, types.LiveClientContent) and event.turn_complete is True
            for turn in event.turns or []
            for part in turn.parts or []
        ]
        assert placeholders == ["."]
        generation_fut.cancel()
        session._active_session = None


async def test_go_away_waits_for_owned_provider_turn_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        session_resumption_handle="resume-at-boundary",
    ) as session:
        session._active_session = cast(Any, _ActiveSessionStub())
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        monkeypatch.setattr(session, "_send_client_event", lambda event: True)
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="finish this owned turn")
        await session.update_chat_ctx(chat_ctx)
        generation_fut = session.generate_reply()
        session._start_new_generation()
        await generation_fut
        initial_epoch = session._session_epoch

        session._handle_go_away(types.LiveServerGoAway(time_left="10s"))

        assert session._session_epoch == initial_epoch
        assert not session._session_should_close.is_set()
        assert session.session_resumption_handle == "resume-at-boundary"

        session._mark_current_generation_done()
        session._finish_provider_turn()

        assert session._session_epoch == initial_epoch + 1
        assert session._session_should_close.is_set()
        assert session.session_resumption_handle == "resume-at-boundary"
        session._active_session = None


async def test_go_away_deadline_replays_exact_pending_text_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        session_resumption_handle="expiring-handle",
    ) as session:
        session._active_session = cast(Any, _ActiveSessionStub())
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", sent.append)
        chat_ctx = llm.ChatContext.empty()
        user_message = chat_ctx.add_message(role="user", content="replay me exactly")
        await session.update_chat_ctx(chat_ctx)
        generation_fut = session.generate_reply()
        generation_fut.add_done_callback(lambda fut: None if fut.cancelled() else fut.exception())
        sent.clear()
        initial_epoch = session._session_epoch

        loop = asyncio.get_running_loop()
        original_call_later = loop.call_later
        deadline_callbacks: list[tuple[float, object]] = []

        def _capture_deadline(delay: float, callback: object, *args: object) -> _TimeoutHandle:
            assert not args
            deadline_callbacks.append((delay, callback))
            return _TimeoutHandle()

        monkeypatch.setattr(loop, "call_later", _capture_deadline)
        session._handle_go_away(types.LiveServerGoAway(time_left="10s"))

        assert session._session_epoch == initial_epoch
        assert len(deadline_callbacks) == 1
        assert 0.0 < deadline_callbacks[0][0] < 10.0

        cast(Any, deadline_callbacks[0][1])()
        monkeypatch.setattr(loop, "call_later", original_call_later)

        assert session._session_epoch == initial_epoch + 1
        assert not generation_fut.done()
        assert session._pending_text_input_item_id == user_message.id
        assert [
            item.raw_text_content
            for item in session.chat_ctx.messages()
            if item.id == user_message.id
        ] == ["replay me exactly"]
        completions = [
            event
            for event in sent
            if isinstance(event, types.LiveClientContent)
            and event.turn_complete is True
            and not event.turns
        ]
        assert len(completions) == 1

        session._start_new_generation()
        await generation_fut
        session._active_session = None


async def test_go_away_restarts_idle_session_with_current_resumption_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        session_resumption_handle="idle-handle",
    ) as session:
        session._active_session = cast(Any, _ActiveSessionStub())
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        initial_epoch = session._session_epoch

        session._handle_go_away(types.LiveServerGoAway(time_left="10s"))

        assert session._session_epoch == initial_epoch + 1
        assert session._session_should_close.is_set()
        assert session.session_resumption_handle == "idle-handle"
        assert session._go_away_deadline_handle is None
        session._active_session = None


@pytest.mark.parametrize("resumption_handle", [None, "resume-handle"])
async def test_go_away_activates_only_one_consecutive_deferred_turn(
    monkeypatch: pytest.MonkeyPatch,
    resumption_handle: str | None,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
        session_resumption_handle=resumption_handle,
    ) as session:
        session._active_session = cast(Any, _ActiveSessionStub())
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        sent: list[object] = []
        monkeypatch.setattr(session, "_send_client_event", lambda event: sent.append(event) or True)
        session._deferred_manual_input_pipeline_active = True

        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=1))
        first_generation = session.generate_reply()
        session.start_user_activity()
        session.push_audio(_input_frame(50, fill=2))
        second_generation = session.generate_reply()

        session._handle_go_away(types.LiveServerGoAway(time_left="10s"))

        realtime_inputs = [
            event for event in sent if isinstance(event, types.LiveClientRealtimeInput)
        ]
        assert len(realtime_inputs) == 3
        assert realtime_inputs[0].activity_start is not None
        assert realtime_inputs[1].audio is not None
        assert realtime_inputs[1].audio.data == bytes([1]) * 1600
        assert realtime_inputs[2].activity_end is not None
        assert len(session._deferred_manual_inputs) == 1
        assert not first_generation.done()
        assert not second_generation.done()
        first_generation.cancel()
        second_generation.cancel()
        session._active_session = None


async def test_go_away_deadline_fails_non_replayable_audio_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(
        monkeypatch,
        manual_activity_detection=True,
        session_resumption_handle="expiring-handle",
    ) as session:
        session._active_session = cast(Any, _ActiveSessionStub())
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._session_should_close.clear()
        monkeypatch.setattr(session, "_send_client_event", lambda event: True)
        session.start_user_activity()
        session.push_audio(_input_frame(50))
        generation_fut = session.generate_reply()
        generation_fut.add_done_callback(lambda fut: None if fut.cancelled() else fut.exception())
        initial_epoch = session._session_epoch

        loop = asyncio.get_running_loop()
        original_call_later = loop.call_later
        deadline_callbacks: list[tuple[float, object]] = []

        def _capture_deadline(delay: float, callback: object, *args: object) -> _TimeoutHandle:
            assert not args
            deadline_callbacks.append((delay, callback))
            return _TimeoutHandle()

        monkeypatch.setattr(loop, "call_later", _capture_deadline)
        session._handle_go_away(types.LiveServerGoAway(time_left="10s"))
        cast(Any, deadline_callbacks[0][1])()
        monkeypatch.setattr(loop, "call_later", original_call_later)

        with pytest.raises(llm.RealtimeError, match="raw audio input cannot be replayed"):
            await generation_fut
        assert session._session_epoch == initial_epoch + 1
        assert session.session_resumption_handle is None
        session._active_session = None


@pytest.mark.parametrize("time_left", ["0s", "0.001s", "0.01s"])
def test_go_away_restart_delay_never_exceeds_short_deadline(time_left: str) -> None:
    seconds = float(time_left.removesuffix("s"))

    delay = RealtimeSession._go_away_restart_delay(time_left)

    assert 0.0 <= delay <= seconds
