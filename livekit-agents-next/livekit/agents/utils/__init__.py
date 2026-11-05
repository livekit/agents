"""Shared helpers: ``is_given``, ``shortuuid``, ``log_exceptions``, participant and track waits,
and the ``aio``, ``media`` and ``http_context`` modules."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import platform
import uuid
from collections.abc import Callable, Coroutine, Sequence
from typing import Literal, Protocol, TypeVar, overload

from typing_extensions import ParamSpec, TypeIs

from livekit import rtc

from .. import _core
from ..types import NotGiven, NotGivenOr
from . import aio, http_context, media
from .aio import ConnectionPool

EventEmitter = rtc.EventEmitter


_T = TypeVar("_T")


def shortuuid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4().hex)[:12]


def is_given(obj: NotGivenOr[_T]) -> TypeIs[_T]:
    return not isinstance(obj, NotGiven)


def nodename() -> str:
    return platform.node()


def is_dev_mode() -> bool:
    """Return whether the agent is running in development mode.

    True when launched via ``console``, ``dev``.
    Reads the ``LIVEKIT_DEV_MODE`` environment variable.
    """
    return os.getenv("LIVEKIT_DEV_MODE") == "1"


def is_hosted() -> bool:
    """Return whether the agent is hosted on LiveKit Cloud."""
    return os.getenv("LIVEKIT_REMOTE_EOT_URL") is not None


F = TypeVar("F", bound=Callable[..., object])
_P = ParamSpec("_P")
_R = TypeVar("_R")


class _ExceptionDecorator(Protocol):
    def __call__(self, fn: Callable[_P, _R], /) -> Callable[_P, _R]: ...


def log_exceptions(
    msg: str = "",
    logger: logging.Logger = logging.getLogger(),  # noqa: B008
) -> _ExceptionDecorator:
    @overload
    def deco(
        fn: Callable[_P, Coroutine[object, object, _R]],
    ) -> Callable[_P, Coroutine[object, object, _R]]: ...

    @overload
    def deco(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...

    def deco(fn: Callable[..., object]) -> Callable[..., object]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_fn_logs(*args: object, **kwargs: object) -> object:
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    err = f"Error in {fn.__name__}"
                    if msg:
                        err += f" – {msg}"
                    logger.exception(err)
                    raise

            return async_fn_logs

        else:

            @functools.wraps(fn)
            def fn_logs(*args: object, **kwargs: object) -> object:
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    err = f"Error in {fn.__name__}"
                    if msg:
                        err += f" – {msg}"
                    logger.exception(err)
                    raise

            return fn_logs

    return deco


def _kinds(kind: Sequence[int] | int | None) -> list[int] | None:
    if kind is None:
        return None
    return [kind] if isinstance(kind, int) else list(kind)


def _room_session(room: rtc.Room) -> _core._core_RoomSession:
    from ..voice.room_io import Room

    return _core._core_create_room_session(Room.of(room))


async def wait_for_agent(
    room: rtc.Room,
    *,
    agent_name: str | None = None,
) -> rtc.RemoteParticipant:
    """
    Wait for an agent participant to join the room.

    Args:
        room: The room to wait for the agent in.
        agent_name: If provided, waits for an agent with matching lk.agent.name attribute.
                   If None, returns the first agent participant found.

    Returns:
        The agent participant.

    Raises:
        RuntimeError: If the room is not connected.
    """
    identity = await _core.wait_for_agent(_room_session(room), agent_name)
    return room.remote_participants[identity]


@overload
async def wait_for_participant(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType | None = None,
    include_local: Literal[False] = False,
) -> rtc.RemoteParticipant: ...


@overload
async def wait_for_participant(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType | None = None,
    include_local: Literal[True],
) -> rtc.Participant: ...


async def wait_for_participant(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType | None = None,
    include_local: bool = False,
) -> rtc.Participant:
    """
    Returns a participant that matches the given identity. If identity is None, the first
    participant that joins the room will be returned.
    If the participant has already joined, the function will return immediately.

    When `include_local` is True, the local participant is also considered.
    """
    found = await _core.wait_for_participant(
        _room_session(room), identity, _kinds(kind), include_local
    )
    return room.local_participant if found is None else room.remote_participants[found]


@overload
async def wait_for_track_publication(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.TrackKind.ValueType] | rtc.TrackKind.ValueType | None = None,
    include_local: Literal[False] = False,
    wait_for_subscription: bool = False,
) -> rtc.RemoteTrackPublication: ...


@overload
async def wait_for_track_publication(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.TrackKind.ValueType] | rtc.TrackKind.ValueType | None = None,
    include_local: Literal[True],
    wait_for_subscription: bool = False,
) -> rtc.TrackPublication: ...


async def wait_for_track_publication(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.TrackKind.ValueType] | rtc.TrackKind.ValueType | None = None,
    include_local: bool = False,
    wait_for_subscription: bool = False,
) -> rtc.TrackPublication:
    """Returns a track publication matching the given identity and kind.
    If identity is None, the first track matching the kind will be returned.
    If a matching track is already published (and subscribed when
    ``wait_for_subscription`` is set), the function returns immediately.

    When `include_local` is True, tracks published by the local participant are also considered;
    local publications resolve on publish and ignore ``wait_for_subscription``.

    When ``identity`` is set, raises :class:`RuntimeError` if that participant
    disconnects before a matching publication is found, so callers don't wait
    on a participant that is gone.
    """
    publisher, sid = await _core.wait_for_track_publication(
        _room_session(room), identity, _kinds(kind), include_local, wait_for_subscription
    )
    participant: rtc.Participant = (
        room.local_participant if publisher is None else room.remote_participants[publisher]
    )
    return participant.track_publications[sid]


__all__ = [
    "shortuuid",
    "http_context",
    "EventEmitter",
    "log_exceptions",
    "aio",
    "media",
    "is_given",
    "ConnectionPool",
]
