from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from livekit.agents.beta.workflows import (
    TwilioConnectorWarmTransferTask,
    WarmTransferTask,
    warm_transfer,
)
from livekit.agents.voice.turn import TurnHandlingOptions

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def mock_background_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(warm_transfer, "BackgroundAudioPlayer", Mock())


def create_task(kind: str, **options: Any) -> WarmTransferTask:
    if kind == "sip":
        return WarmTransferTask("+15555550123", sip_trunk_id="test-trunk", **options)
    return TwilioConnectorWarmTransferTask(
        "+15555550123",
        twilio_from_number="+15555550124",
        twilio_account_sid="test-account",
        twilio_auth_token="test-token",
        **options,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sip", "twilio"])
async def test_turn_handling_is_accepted_by_warm_transfer_tasks(kind: str) -> None:
    options = TurnHandlingOptions(
        turn_detection="vad",
        endpointing={"min_delay": 0.2},
        interruption={"enabled": False},
    )

    task = create_task(kind, turn_handling=options, turn_detection="stt")

    assert task._turn_handling == options
    assert task.turn_detection == "vad"
    assert task.allow_interruptions is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sip", "twilio"])
async def test_legacy_turn_detection_still_works(kind: str) -> None:
    task = create_task(kind, turn_detection="vad", allow_interruptions=False)

    assert task._turn_handling == {"turn_detection": "vad", "interruption": {"enabled": False}}


@pytest.mark.asyncio
async def test_colleague_session_and_agent_receive_turn_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = TurnHandlingOptions(
        turn_detection="vad",
        interruption={"min_duration": 0.3},
        user_turn_limit={"max_words": 50},
    )
    task = create_task("sip", turn_handling=options)
    task.turn_detection = "stt"
    task._caller_room = Mock(name="caller-room")
    task._caller_room.name = "caller"
    task._caller_room.local_participant.identity = "caller-agent"
    monkeypatch.setattr(
        task,
        "_get_activity_or_raise",
        lambda: SimpleNamespace(
            session=SimpleNamespace(vad=None, llm=None, stt=None, tts=None, turn_detection="vad")
        ),
    )
    monkeypatch.setattr(
        warm_transfer,
        "get_job_context",
        lambda: SimpleNamespace(_info=SimpleNamespace(url="wss://example.test")),
    )
    room = Mock()
    room.connect = AsyncMock(side_effect=RuntimeError("stop before dialing"))
    room.disconnect = AsyncMock()
    monkeypatch.setattr(warm_transfer.rtc, "Room", lambda: room)
    token = Mock()
    token.with_identity.return_value = token
    token.with_grants.return_value = token
    token.with_kind.return_value = token
    token.to_jwt.return_value = "test-token"
    monkeypatch.setattr(warm_transfer.api, "AccessToken", lambda: token)
    session_factory = Mock()
    agent_factory = Mock()
    monkeypatch.setattr(warm_transfer, "AgentSession", session_factory)
    monkeypatch.setattr(warm_transfer, "Agent", agent_factory)

    with pytest.raises(RuntimeError, match="stop before dialing"):
        await task._dial_human_agent()

    expected = {**options, "turn_detection": "stt"}
    assert session_factory.call_args.kwargs["turn_handling"] == expected
    assert agent_factory.call_args.kwargs["turn_handling"] == expected
