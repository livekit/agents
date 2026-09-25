from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock

import pytest

from livekit import agents
from livekit.agents import AMDCategory, AMDCompletedEvent, AMDLifecycle, AMDReason, llm
from livekit.agents.voice.amd.detector import _DEFAULT_VOICEMAIL_INSTRUCTIONS

from .amd_test_utils import detector_clock, next_deadline  # noqa: F401
from .fake_llm import FakeLLMResponse
from .test_amd_detector import (
    commit,
    end_of_turn,
    eventually,
    running,
    speech_ended,
    speech_started,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "answered",
        "voicemail",
        "voicemail_menu",
        "voicemail_failed",
        "human_after_voicemail",
        "disconnected",
        "missing",
        "timeout",
        "cancelled",
        "console",
    ],
)
async def test_outbound_call_setup(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    monkeypatch.setenv("SIP_PHONE_NUMBER", "+15555550100")
    monkeypatch.setenv("SIP_PARTICIPANT_IDENTITY", "callee")
    monkeypatch.setenv("SIP_OUTBOUND_TRUNK_ID", "test-trunk")
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    for name in ("STT", "LLM", "TTS"):
        monkeypatch.setattr(agents.inference, name, Mock())
    session = Mock(start=AsyncMock())
    if outcome == "console":
        type(session).room_io = PropertyMock(
            side_effect=RuntimeError(
                "Cannot access room_io: the AgentSession was not started with a room."
            )
        )
    monkeypatch.setattr(agents, "AgentSession", Mock(return_value=session))
    category = {
        "voicemail": AMDCategory.MACHINE_VM,
        "voicemail_menu": AMDCategory.MACHINE_IVR,
        "voicemail_failed": AMDCategory.MACHINE_VM,
    }.get(outcome, AMDCategory.HUMAN)
    result = AMDCompletedEvent(
        category=category,
        reason=AMDReason.FINISHED if category == AMDCategory.HUMAN else AMDReason.IDLE_TIMEOUT,
        turn_id=1,
        transcript="Please leave a message." if outcome == "voicemail" else "Hello.",
        voicemail_message_played=outcome
        in {"voicemail", "voicemail_menu", "human_after_voicemail"},
    )
    detector = MagicMock(execute=AsyncMock(return_value=result))
    monkeypatch.setattr(agents, "AMD", Mock(return_value=detector))
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "telephony" / "amd.py")
    )

    room = SimpleNamespace(
        name="test-room",
        remote_participants={},
        isconnected=Mock(return_value=outcome != "console"),
    )

    async def create_participant(*args: object, **kwargs: object) -> None:
        detector.__aenter__.assert_awaited_once()
        if outcome == "timeout":
            raise asyncio.TimeoutError
        if outcome == "cancelled":
            raise asyncio.CancelledError
        if outcome != "missing":
            room.remote_participants["callee"] = SimpleNamespace(identity="callee")
        if outcome == "disconnected":
            room.remote_participants.pop("callee")

    create = AsyncMock(side_effect=create_participant)
    ctx = SimpleNamespace(
        room=room,
        api=SimpleNamespace(sip=SimpleNamespace(create_sip_participant=create)),
        shutdown=Mock(),
        add_shutdown_callback=Mock(),
    )
    if outcome == "console":
        with pytest.raises(RuntimeError, match="not started with a room"):
            await example["entrypoint"](ctx)
        detector.__aenter__.assert_not_awaited()
        create.assert_not_awaited()
        return
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await example["entrypoint"](ctx)
    else:
        await example["entrypoint"](ctx)

    create.assert_awaited_once()
    assert create.call_args.args[0].wait_until_answered
    detector.__aexit__.assert_awaited_once()
    if outcome in {
        "answered",
        "voicemail",
        "voicemail_menu",
        "voicemail_failed",
        "human_after_voicemail",
    }:
        detector.execute.assert_awaited_once()
        if outcome in {"voicemail", "voicemail_menu"}:
            ctx.shutdown.assert_called_once_with("voicemail completed")
        else:
            ctx.shutdown.assert_not_called()
    else:
        detector.execute.assert_not_awaited()
        if outcome == "cancelled":
            ctx.shutdown.assert_not_called()
        else:
            ctx.shutdown.assert_called_once_with(
                "call not answered" if outcome == "timeout" else "participant missing"
            )


@pytest.fixture
def example_agent(monkeypatch: pytest.MonkeyPatch) -> agents.Agent:
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "telephony" / "amd.py")
    )
    return example["MyAgent"]()


@pytest.mark.asyncio
@pytest.mark.virtual_time
@pytest.mark.parametrize("menu_arrives", [False, True])
async def test_voicemail_keeps_call_open_until_amd_finishes(
    monkeypatch: pytest.MonkeyPatch, example_agent: agents.Agent, menu_arrives: bool
) -> None:
    job = Mock()
    monkeypatch.setattr("livekit.agents.beta.tools.end_call.get_job_context", lambda: job)
    publish = AsyncMock()
    monkeypatch.setattr(
        "livekit.agents.beta.tools.send_dtmf.get_job_context",
        lambda: SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish))
        ),
    )
    async with running(agent=example_agent, voicemail_idle_timeout=10.0) as (
        detector,
        session,
        classifier,
        model,
    ):
        activity = session._activity
        model.fake_response_map[_DEFAULT_VOICEMAIL_INSTRUCTIONS].tool_calls = [
            llm.FunctionToolCall(name="end_call", arguments="{}", call_id="too-early")
        ]
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert "end_call" not in {tool.id for tool in call["tools"]}
        await eventually(lambda: detector._voicemail_message_played and activity._no_pending_speech)
        assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
            10, abs=0.1
        )
        await asyncio.sleep(5)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        job.shutdown.assert_not_called()

        if not menu_arrives:
            result = await asyncio.wait_for(detector.execute(), 6)
            assert result.reason == AMDReason.IDLE_TIMEOUT
            assert result.category == AMDCategory.MACHINE_VM
            assert result.voicemail_message_played
            job.shutdown.assert_not_called()
            return

        # The rejected tool call can produce an error follow-up before the next menu.
        while not model.calls.empty():
            call = model.calls.get_nowait()
            assert "end_call" not in {tool.id for tool in call["tools"]}

        instructions = detector._get_instructions(AMDCategory.MACHINE_IVR)
        model.fake_response_map[instructions] = FakeLLMResponse(
            input=instructions,
            content="",
            ttft=0,
            duration=0,
            tool_calls=[
                llm.FunctionToolCall(
                    name="send_dtmf_events", arguments='{"events":["1"]}', call_id="save-message"
                )
            ],
        )
        speech_started(detector)
        speech_ended(detector, 0)
        activity.on_end_of_turn(end_of_turn("To save your message, press 1."))
        await classifier.request()
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert {tool.id for tool in call["tools"]} == {"send_dtmf_events"}
        result = await asyncio.wait_for(detector.execute(), 12)
        publish.assert_awaited_once_with(code=1, digit="1")
        assert result.reason == AMDReason.IDLE_TIMEOUT
        assert result.category == AMDCategory.MACHINE_IVR
        assert result.voicemail_message_played
        job.shutdown.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.virtual_time
async def test_human_can_end_call_after_amd(
    monkeypatch: pytest.MonkeyPatch, example_agent: agents.Agent
) -> None:
    job = Mock()
    monkeypatch.setattr("livekit.agents.beta.tools.end_call.get_job_context", lambda: job)
    async with running(agent=example_agent) as (detector, session, classifier, model):
        model.fake_response_map["hello"].tool_calls = [
            llm.FunctionToolCall(name="end_call", arguments="{}", call_id="conversation-complete")
        ]
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.HUMAN)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert "end_call" in {tool.id for tool in call["tools"]}
        await eventually(lambda: job.shutdown.called)
        job.shutdown.assert_called_once_with(reason="user_initiated")
