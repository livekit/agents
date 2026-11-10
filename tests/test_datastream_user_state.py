"""The agent forwards the user's speaking state to a remote avatar worker over
the ``lk.user_state`` RPC: AvatarSession relays the session's
``user_state_changed`` event to the DataStreamAudioOutput, which sends it once
the avatar is ready, and DataStreamAudioReceiver re-emits it on the avatar side."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import _datastream_io as ds
from livekit.agents.voice.avatar._types import AvatarSession
from livekit.agents.voice.events import UserStateChangedEvent

from .test_audio_sink_proxy import _PassthroughWrapper

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

AVATAR = "avatar"
AGENT = "agent"


def _speaking(created_at: float = 1.0) -> UserStateChangedEvent:
    return UserStateChangedEvent(old_state="listening", new_state="speaking", created_at=created_at)


def _listening(created_at: float = 2.0) -> UserStateChangedEvent:
    return UserStateChangedEvent(old_state="speaking", new_state="listening", created_at=created_at)


def _rpc_calls(room: MagicMock, method: str) -> list[dict]:
    return [
        call.kwargs
        for call in room.local_participant.perform_rpc.call_args_list
        if call.kwargs.get("method") == method
    ]


def _fake_room() -> MagicMock:
    room = MagicMock()
    room.isconnected.return_value = True
    room.local_participant._rpc_handlers = {}
    room.local_participant.perform_rpc = AsyncMock(return_value="")
    return room


@pytest.fixture
async def output(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[ds.DataStreamAudioOutput, MagicMock]]:
    room = _fake_room()
    monkeypatch.setattr(ds.utils, "wait_for_participant", AsyncMock())
    output = ds.DataStreamAudioOutput(room, destination_identity=AVATAR)
    try:
        yield output, room
    finally:
        await asyncio.gather(*output._tasks)
        ds.DataStreamAudioOutput._playback_finished_handlers.pop(AVATAR, None)


async def test_user_state_is_sent_to_the_avatar_as_rpc(
    output: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    sink, room = output
    await sink._start_task()

    sink.notify_user_state(_speaking())
    sink.notify_user_state(_listening())
    await asyncio.gather(*sink._tasks)

    calls = _rpc_calls(room, ds.RPC_USER_STATE)
    assert [c["destination_identity"] for c in calls] == [AVATAR, AVATAR]
    payloads = [json.loads(c["payload"]) for c in calls]
    assert [p["new_state"] for p in payloads] == ["speaking", "listening"]
    assert payloads[0]["created_at"] == 1.0


async def test_only_the_latest_state_is_sent_once_the_avatar_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = _fake_room()
    avatar_joined = asyncio.Event()

    async def wait_for_participant(**_kwargs: object) -> None:
        await avatar_joined.wait()

    monkeypatch.setattr(ds.utils, "wait_for_participant", wait_for_participant)
    sink = ds.DataStreamAudioOutput(room, destination_identity=AVATAR)
    try:
        # the avatar has not joined yet: nothing is sent, the last state is kept
        sink.notify_user_state(_speaking())
        sink.notify_user_state(_listening())
        for _ in range(5):
            await asyncio.sleep(0)
        assert _rpc_calls(room, ds.RPC_USER_STATE) == []

        avatar_joined.set()
        await sink._start_task()
        await asyncio.gather(*sink._tasks)
        calls = _rpc_calls(room, ds.RPC_USER_STATE)
        assert len(calls) == 1
        assert json.loads(calls[0]["payload"])["new_state"] == "listening"
    finally:
        await asyncio.gather(*sink._tasks)
        ds.DataStreamAudioOutput._playback_finished_handlers.pop(AVATAR, None)


async def test_avatar_without_the_rpc_is_not_sent_further_states(
    output: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    sink, room = output
    await sink._start_task()
    room.local_participant.perform_rpc.side_effect = rtc.RpcError(
        rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD, "no such method"
    )

    sink.notify_user_state(_speaking())
    await asyncio.gather(*sink._tasks)
    sink.notify_user_state(_listening())
    await asyncio.gather(*sink._tasks)

    assert len(_rpc_calls(room, ds.RPC_USER_STATE)) == 1


async def test_rpc_failure_does_not_raise_or_stop_later_states(
    output: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    sink, room = output
    await sink._start_task()
    room.local_participant.perform_rpc.side_effect = rtc.RpcError(
        rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT, "avatar did not answer"
    )

    sink.notify_user_state(_speaking())
    await asyncio.gather(*sink._tasks)
    room.local_participant.perform_rpc.side_effect = None
    sink.notify_user_state(_listening())
    await asyncio.gather(*sink._tasks)

    assert len(_rpc_calls(room, ds.RPC_USER_STATE)) == 2


class _FakeAgentSession(rtc.EventEmitter[str]):
    def __init__(self, audio_output: object) -> None:
        super().__init__()
        self.output = SimpleNamespace(audio=audio_output)


class _TestAvatarSession(AvatarSession[str]):
    @property
    def avatar_identity(self) -> str:
        return AVATAR


async def test_avatar_session_forwards_the_session_event_through_the_output_chain(
    output: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    leaf, room = output
    await leaf._start_task()
    # a wrapper in front of the data stream output, as recorders and synchronizers add
    agent_session = _FakeAgentSession(_PassthroughWrapper(next_in_chain=leaf))
    avatar_room = MagicMock()
    avatar_room.isconnected.return_value = False

    session = _TestAvatarSession()
    await session.start(agent_session, avatar_room)  # type: ignore[arg-type]
    try:
        agent_session.emit("user_state_changed", _speaking())
        await asyncio.gather(*leaf._tasks)
        calls = _rpc_calls(room, ds.RPC_USER_STATE)
        assert len(calls) == 1
        assert json.loads(calls[0]["payload"])["new_state"] == "speaking"
    finally:
        await session.aclose()

    # after aclose the session no longer forwards
    agent_session.emit("user_state_changed", _listening())
    await asyncio.gather(*leaf._tasks)
    assert len(_rpc_calls(room, ds.RPC_USER_STATE)) == 1


@pytest.fixture
async def receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[ds.DataStreamAudioReceiver, MagicMock]]:
    room = _fake_room()
    rpc_methods: dict = {}
    room.local_participant.register_rpc_method.side_effect = lambda name, h: (
        rpc_methods.__setitem__(name, h),
        room.local_participant._rpc_handlers.__setitem__(name, h),
    )
    monkeypatch.setattr(
        ds.utils, "wait_for_participant", AsyncMock(return_value=SimpleNamespace(identity=AGENT))
    )
    receiver = ds.DataStreamAudioReceiver(room)
    await receiver.start()
    try:
        yield receiver, room
    finally:
        await receiver.aclose()
        ds.DataStreamAudioReceiver._clear_buffer_handlers.pop(AGENT, None)
        ds.DataStreamAudioReceiver._user_state_handlers.pop(AGENT, None)


def _invocation(caller: str, payload: str) -> MagicMock:
    data = MagicMock(spec=rtc.RpcInvocationData)
    data.caller_identity = caller
    data.payload = payload
    return data


async def test_receiver_emits_the_user_state_from_its_sender(
    receiver: tuple[ds.DataStreamAudioReceiver, MagicMock],
) -> None:
    recv, room = receiver
    handler = room.local_participant._rpc_handlers[ds.RPC_USER_STATE]
    events: list[UserStateChangedEvent] = []
    recv.on("user_state_changed", events.append)

    assert handler(_invocation(AGENT, _speaking().model_dump_json())) == "ok"
    assert handler(_invocation("someone-else", _listening().model_dump_json())) == "reject"
    assert handler(_invocation(AGENT, "not json")) == "reject"

    assert [ev.new_state for ev in events] == ["speaking"]
    assert events[0].created_at == 1.0
