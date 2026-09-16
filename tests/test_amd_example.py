from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock

import pytest

from livekit import agents

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["answered", "disconnected", "missing", "timeout", "cancelled", "console"]
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
    detector = MagicMock(execute=AsyncMock(return_value=Mock()))
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
    if outcome == "answered":
        detector.execute.assert_awaited_once()
        ctx.shutdown.assert_not_called()
    else:
        detector.execute.assert_not_awaited()
        if outcome == "cancelled":
            ctx.shutdown.assert_not_called()
        else:
            ctx.shutdown.assert_called_once_with(
                "call not answered" if outcome == "timeout" else "participant missing"
            )
