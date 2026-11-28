import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import jwt
import pytest

from livekit.agents import JobContext, job
from livekit.agents.beta.workflows import (
    TwilioConnectorWarmTransferTask,
    WarmTransferTask,
    warm_transfer,
)
from livekit.agents.ipc.job_proc_lazy_main import _JobTokenClient
from livekit.agents.ipc.proto import CreateJobTokenRequest, CreateJobTokenResponse
from livekit.agents.voice.turn import TurnHandlingOptions
from livekit.protocol import agent, models

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
        lambda: SimpleNamespace(
            _info=SimpleNamespace(url="wss://example.test"),
            create_participant_token=AsyncMock(return_value="test-token"),
        ),
    )
    room = Mock()
    room.connect = AsyncMock(side_effect=RuntimeError("stop before dialing"))
    room.disconnect = AsyncMock()
    monkeypatch.setattr(warm_transfer.rtc, "Room", lambda: room)
    session_factory = Mock()
    agent_factory = Mock()
    monkeypatch.setattr(warm_transfer, "AgentSession", session_factory)
    monkeypatch.setattr(warm_transfer, "Agent", agent_factory)

    with pytest.raises(RuntimeError, match="stop before dialing"):
        await task._dial_human_agent()

    expected = {**options, "turn_detection": "stt"}
    assert session_factory.call_args.kwargs["turn_handling"] == expected
    assert agent_factory.call_args.kwargs["turn_handling"] == expected


# -- the token for the human agent's room: signed here, or by the server in token mode

_TOKEN_MODE_ENV = {"LIVEKIT_API_KEY": "", "LIVEKIT_API_SECRET": "", "LIVEKIT_TOKEN": "worker-token"}
_KEY_MODE_ENV = {
    "LIVEKIT_API_KEY": "devkey",
    "LIVEKIT_API_SECRET": "secret" * 6,
    "LIVEKIT_TOKEN": "",
}
_HUMAN_AGENT_ROOM_GRANTS = {
    "roomJoin": True,
    "room": "caller-human-agent",
    "canUpdateOwnMetadata": True,
    "canPublish": True,
    "canSubscribe": True,
    "canPublishData": True,
}


class _TokenWorker:
    """Answers the requests of a _JobTokenClient the way the worker relays the server's."""

    def __init__(self) -> None:
        self.claims: list[str] = []
        self.client = _JobTokenClient(self)  # type: ignore[arg-type]

    async def send(self, msg: CreateJobTokenRequest) -> None:
        self.claims.append(msg.claims)
        resp = CreateJobTokenResponse(
            request_id=msg.request_id, token="job-token", expires_at=int(time.time()) + 600
        )
        asyncio.get_running_loop().call_soon(self.client._on_response, resp)


async def _human_agent_room_token(monkeypatch: pytest.MonkeyPatch, worker: _TokenWorker) -> str:
    """Dials until the human agent's room is joined, and returns the token it is joined with."""
    task = create_task("sip")
    task._caller_room = Mock(name="caller-room")
    task._caller_room.name = "caller"
    task._caller_room.local_participant.identity = "caller-agent"
    monkeypatch.setattr(
        task,
        "_get_activity_or_raise",
        lambda: SimpleNamespace(
            session=SimpleNamespace(vad=None, llm=None, stt=None, tts=None, turn_detection=None)
        ),
    )
    info = job.RunningJobInfo(
        job=agent.Job(id="AJ_transfer", room=models.Room(name="caller")),
        url="wss://example.test",
        token="",
        accept_arguments=job.JobAcceptArguments(name="", identity="caller-agent", metadata=""),
        worker_id="AW_test",
        fake_job=True,
    )
    ctx = JobContext(
        proc=MagicMock(),
        info=info,
        room=MagicMock(),
        on_connect=lambda: None,
        on_shutdown=lambda _: None,
        inference_executor=MagicMock(),
        job_token_client=worker.client,
    )
    monkeypatch.setattr(warm_transfer, "get_job_context", lambda: ctx)
    room = Mock()
    room.connect = AsyncMock(side_effect=RuntimeError("stop before dialing"))
    room.disconnect = AsyncMock()
    monkeypatch.setattr(warm_transfer.rtc, "Room", lambda: room)
    monkeypatch.setattr(warm_transfer, "AgentSession", Mock())
    monkeypatch.setattr(warm_transfer, "Agent", Mock())

    with pytest.raises(RuntimeError, match="stop before dialing"):
        await task._dial_human_agent()

    ws_url, token = room.connect.call_args.args
    assert ws_url == "wss://example.test"
    return token  # type: ignore[no-any-return]


@pytest.mark.asyncio
@pytest.mark.no_concurrent  # patches os.environ
async def test_human_agent_room_token_comes_from_create_job_token_in_token_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _TokenWorker()
    # nothing may sign a token here: there is no API secret
    monkeypatch.setattr(job, "AccessToken", Mock(side_effect=AssertionError("signed locally")))
    with patch.dict(os.environ, _TOKEN_MODE_ENV):
        token = await _human_agent_room_token(monkeypatch, worker)

    assert token == "job-token"
    assert [json.loads(c) for c in worker.claims] == [
        {"identity": "caller-agent", "kind": "agent", "video": _HUMAN_AGENT_ROOM_GRANTS}
    ]


@pytest.mark.asyncio
@pytest.mark.no_concurrent  # patches os.environ
async def test_human_agent_room_token_is_signed_here_in_key_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _TokenWorker()
    with patch.dict(os.environ, _KEY_MODE_ENV):
        token = await _human_agent_room_token(monkeypatch, worker)

    payload = jwt.decode(token, "secret" * 6, algorithms=["HS256"])
    assert payload["iss"] == "devkey"
    assert payload["sub"] == "caller-agent"
    assert payload["kind"] == "agent"
    assert "name" not in payload
    assert payload["video"] == _HUMAN_AGENT_ROOM_GRANTS
    assert worker.claims == []
