"""DataStreamAudioReceiver must survive RPC errors from the sender.

A sender on the default DataStreamAudioOutput(wait_playback_start=False) never
registers lk.playback_started, and any notification can fail after its retries.
Neither may close the receiver, which would stop all later audio.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import _datastream_io as ds

pytestmark = pytest.mark.unit

SENDER = "agent"


async def _start_receiver(perform_rpc: AsyncMock) -> ds.DataStreamAudioReceiver:
    room = MagicMock()
    room.local_participant._rpc_handlers = {}
    room.local_participant.perform_rpc = perform_rpc
    sender = SimpleNamespace(identity=SENDER)
    with patch.object(ds.utils, "wait_for_participant", AsyncMock(return_value=sender)):
        receiver = ds.DataStreamAudioReceiver(room, sender_identity=SENDER, rpc_max_retries=2)
        await receiver.start()
    return receiver


def _methods(perform_rpc: AsyncMock) -> list[str]:
    return [call.kwargs["method"] for call in perform_rpc.await_args_list]


async def _drain() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def test_unregistered_playback_started_is_only_sent_once(caplog):
    async def perform_rpc(*, method: str, **kwargs) -> str:
        if method == ds.RPC_PLAYBACK_STARTED:
            raise rtc.RpcError(rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD, "unsupported")
        return "ok"

    rpc = AsyncMock(side_effect=perform_rpc)
    receiver = await _start_receiver(rpc)
    try:
        with caplog.at_level(logging.INFO, logger="livekit.agents"):
            receiver.notify_playback_started()
            receiver.notify_playback_started()  # queued before the error arrived
            await _drain()
            receiver.notify_playback_finished(playback_position=1.0, interrupted=False)
            receiver.notify_playback_started()
            receiver.notify_playback_finished(playback_position=1.0, interrupted=False)
            await _drain()

        assert _methods(rpc) == [
            ds.RPC_PLAYBACK_STARTED,
            ds.RPC_PLAYBACK_FINISHED,
            ds.RPC_PLAYBACK_FINISHED,
        ]
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    finally:
        await receiver.aclose()


async def test_failed_notification_is_dropped_and_the_receiver_keeps_running():
    failures = 2  # every retry of the first notification fails

    async def perform_rpc(*, method: str, **kwargs) -> str:
        nonlocal failures
        if failures:
            failures -= 1
            raise rtc.RpcError(rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT, "timeout")
        return "ok"

    rpc = AsyncMock(side_effect=perform_rpc)
    receiver = await _start_receiver(rpc)
    try:
        receiver.notify_playback_finished(playback_position=1.0, interrupted=False)
        await asyncio.sleep(0.2)
        receiver.notify_playback_finished(playback_position=2.0, interrupted=True)
        await _drain()

        assert _methods(rpc) == [ds.RPC_PLAYBACK_FINISHED] * 3
        assert json.loads(rpc.await_args_list[-1].kwargs["payload"])["playback_position"] == 2.0
        assert receiver._main_atask is not None and not receiver._main_atask.done()
    finally:
        await receiver.aclose()
