import asyncio
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, utils
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_stt import FakeSTT

pytestmark = pytest.mark.unit


_REALTIME_AUDIO_REDACTION_WARNING = (
    "RealtimeModel user turns lack complete speech timestamps, so audio redaction may be "
    "inaccurate; disable audio recording to prevent redaction leak."
)


def _fake_job_context(*, enable_redaction: bool) -> MagicMock:
    job_ctx = MagicMock()
    job_ctx.job.enable_recording = True
    job_ctx.job.enable_redaction = enable_redaction
    job_ctx.job.id = "test-job-id"
    job_ctx.job.room.sid = "test-room-sid"
    job_ctx.job.agent_name = "test-agent"
    job_ctx.room.name = "test-room"
    job_ctx._primary_agent_session = None
    job_ctx.session_directory = Path("/tmp/test-session")
    return job_ctx


async def _run_realtime_redaction_session(
    *,
    project_redaction: bool,
    session_redaction: bool,
    audio_recording: bool = True,
    stt: FakeSTT | None = None,
    server_turn_detection: bool = True,
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(turn_detection=server_turn_detection, audio_output=False)
    )
    job_ctx = _fake_job_context(enable_redaction=project_redaction)

    with patch("livekit.agents.voice.agent_session.get_job_context", return_value=job_ctx):
        async with AgentSession(
            llm=model,
            stt=stt,
            vad=None,
            turn_handling={"turn_detection": None},
        ) as session:
            await session.start(
                Agent(instructions="test"),
                record={"audio": audio_recording, "redaction": session_redaction},
            )


async def test_realtime_response_id_is_available_on_assistant_message() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    conversation_events: list[ConversationItemAddedEvent] = []

    async with AgentSession(llm=model) as session:
        session.on("conversation_item_added", conversation_events.append)
        await session.start(Agent(instructions="test"))

        speech_handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        function_ch.close()
        text_ch.send_nowait("Hello")
        text_ch.close()
        audio_ch.close()

        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="provider-response-id",
            )
        )
        await speech_handle

    assistant_messages = [
        event.item
        for event in conversation_events
        if event.item.type == "message" and event.item.role == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].metrics["provider_request_ids"] == ["provider-response-id"]


async def _transcribed_user_messages(
    *transcripts: llm.InputTranscriptionCompleted,
) -> list[llm.ChatMessage]:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    conversation_events: list[ConversationItemAddedEvent] = []

    async with AgentSession(llm=model) as session:
        session.on("conversation_item_added", conversation_events.append)
        await session.start(Agent(instructions="test"))

        for transcript in transcripts:
            model.active_session.emit("input_audio_transcription_completed", transcript)
        await asyncio.sleep(0)

    return [
        event.item
        for event in conversation_events
        if event.item.type == "message" and event.item.role == "user"
    ]


async def test_realtime_user_message_placed_where_the_turn_began() -> None:
    # the provider withholds the transcript until its reply is done generating
    turn_started_at = 1_000_000.0

    messages = await _transcribed_user_messages(
        llm.InputTranscriptionCompleted(
            item_id="item_1",
            transcript="what is my name?",
            is_final=True,
            turn_started_at=turn_started_at,
        )
    )

    assert len(messages) == 1
    assert messages[0].created_at == turn_started_at
    assert messages[0].metrics["started_speaking_at"] == turn_started_at


async def test_realtime_late_transcripts_keep_their_own_turn_start() -> None:
    messages = await _transcribed_user_messages(
        llm.InputTranscriptionCompleted(
            item_id="item_1", transcript="first", is_final=True, turn_started_at=1_000_000.0
        ),
        llm.InputTranscriptionCompleted(
            item_id="item_2", transcript="second", is_final=True, turn_started_at=1_005_000.0
        ),
    )

    assert [msg.created_at for msg in messages] == [1_000_000.0, 1_005_000.0]


async def test_realtime_user_message_falls_back_to_delivery_time() -> None:
    before = time.time()

    messages = await _transcribed_user_messages(
        llm.InputTranscriptionCompleted(
            item_id="item_1", transcript="text injected turn", is_final=True
        )
    )

    assert len(messages) == 1
    assert before <= messages[0].created_at <= time.time()
    assert "started_speaking_at" not in messages[0].metrics


async def test_realtime_user_turn_is_ordered_before_the_reply_it_prompted() -> None:
    """A duplex model answers over the caller, so the transcript arrives after its own reply."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    async with AgentSession(llm=model) as session:
        agent = Agent(instructions="test")
        await session.start(agent)

        reply = agent.chat_ctx.add_message(
            role="assistant", content="answering", id="reply", created_at=1_000_500.0
        )
        agent._chat_ctx.insert(reply)

        model.active_session.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id="user_turn",
                transcript="what is my name?",
                is_final=True,
                turn_started_at=1_000_000.0,
            ),
        )
        await asyncio.sleep(0)

        ids = [item.id for item in agent.chat_ctx.items if item.type == "message"]
        assert ids.index("user_turn") < ids.index("reply")


@pytest.mark.parametrize(
    ("project_redaction", "session_redaction"),
    [(True, False), (False, True)],
    ids=["project", "session"],
)
async def test_realtime_audio_redaction_warns_when_redaction_is_enabled(
    caplog: pytest.LogCaptureFixture,
    project_redaction: bool,
    session_redaction: bool,
) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await _run_realtime_redaction_session(
            project_redaction=project_redaction,
            session_redaction=session_redaction,
        )

    assert _REALTIME_AUDIO_REDACTION_WARNING in caplog.messages


@pytest.mark.parametrize(
    ("with_stt", "server_turn_detection"),
    [(True, True), (False, False)],
    ids=["with-stt", "without-server-turn-detection"],
)
async def test_realtime_audio_redaction_warning_covers_all_realtime_turn_modes(
    caplog: pytest.LogCaptureFixture,
    with_stt: bool,
    server_turn_detection: bool,
) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await _run_realtime_redaction_session(
            project_redaction=True,
            session_redaction=False,
            stt=FakeSTT() if with_stt else None,
            server_turn_detection=server_turn_detection,
        )

    assert _REALTIME_AUDIO_REDACTION_WARNING in caplog.messages


@pytest.mark.parametrize(
    ("project_redaction", "session_redaction", "audio_recording"),
    [(False, False, True), (True, False, False)],
    ids=["redaction-disabled", "audio-recording-disabled"],
)
async def test_realtime_audio_redaction_warning_requires_redacted_audio_recording(
    caplog: pytest.LogCaptureFixture,
    project_redaction: bool,
    session_redaction: bool,
    audio_recording: bool,
) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await _run_realtime_redaction_session(
            project_redaction=project_redaction,
            session_redaction=session_redaction,
            audio_recording=audio_recording,
        )

    assert _REALTIME_AUDIO_REDACTION_WARNING not in caplog.messages


async def test_realtime_audio_redaction_warning_is_emitted_once_per_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        async with AgentSession(
            llm=model, vad=None, turn_handling={"turn_detection": None}
        ) as session:
            await session.start(
                Agent(instructions="first"),
                record={"redaction": True},
            )
            session.update_agent(Agent(instructions="second"))
            assert session._update_activity_atask is not None
            await session._update_activity_atask

    assert caplog.messages.count(_REALTIME_AUDIO_REDACTION_WARNING) == 1
