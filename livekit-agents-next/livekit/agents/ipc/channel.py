from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, ClassVar, Protocol

from .. import _core
from . import proto


class Message(Protocol):
    MSG_ID: ClassVar[int]


MessagesDict = dict[int, type[Message]]


class _AsyncChannel(Protocol):
    def recv_bytes(self) -> Awaitable[bytes]: ...
    def send_bytes(self, data: bytes) -> Awaitable[None]: ...


class _SyncChannel(Protocol):
    def recv_bytes(self) -> bytes: ...
    def send_bytes(self, data: bytes) -> None: ...


def _read_message(data: bytes, messages: MessagesDict) -> Message:
    msg_id, fields = _core.ipc_decode(data)
    msg_type: Any = messages[msg_id]
    if msg_type is proto.StartJobRequest:
        return proto.StartJobRequest._from_wire(fields)
    return msg_type(**fields)  # type: ignore[no-any-return]


def _write_message(msg: Message) -> bytes:
    fields = msg._to_wire() if isinstance(msg, proto.StartJobRequest) else vars(msg)
    return _core.ipc_encode(msg.MSG_ID, fields)


async def arecv_message(dplx: _AsyncChannel, messages: MessagesDict) -> Message:
    return _read_message(await dplx.recv_bytes(), messages)


async def asend_message(dplx: _AsyncChannel, msg: Message) -> None:
    await dplx.send_bytes(_write_message(msg))


def recv_message(dplx: _SyncChannel, messages: MessagesDict) -> Message:
    return _read_message(dplx.recv_bytes(), messages)


def send_message(dplx: _SyncChannel, msg: Message) -> None:
    dplx.send_bytes(_write_message(msg))
