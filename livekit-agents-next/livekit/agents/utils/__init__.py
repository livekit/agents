from __future__ import annotations

import functools
import inspect
import logging
import os
from collections.abc import Callable, Coroutine, Sequence
from typing import Literal, Protocol, TypeVar, overload

from typing_extensions import ParamSpec, TypeIs

from livekit import rtc

from .. import _core
from .._core import shortuuid as shortuuid
from ..types import NotGiven, NotGivenOr
from . import aio, media
from .aio import ConnectionPool

EventEmitter = rtc.EventEmitter


_T = TypeVar("_T")


def is_given(obj: NotGivenOr[_T]) -> TypeIs[_T]:
    return not isinstance(obj, NotGiven)


def is_dev_mode() -> bool:
    """Whether the agent runs in development mode (``console`` or ``dev``)."""
    return os.getenv("LIVEKIT_DEV_MODE") == "1"


def is_hosted() -> bool:
    """Whether the agent is hosted on LiveKit Cloud."""
    return os.getenv("LIVEKIT_REMOTE_EOT_URL") is not None


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
        err = f"Error in {fn.__name__}" + (f" – {msg}" if msg else "")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_fn_logs(*args: object, **kwargs: object) -> object:
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    logger.exception(err)
                    raise

            return async_fn_logs

        @functools.wraps(fn)
        def fn_logs(*args: object, **kwargs: object) -> object:
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception(err)
                raise

        return fn_logs

    return deco


def _kinds(kind: Sequence[int] | int | None) -> list[int] | None:
    if kind is None:
        return None
    return [kind] if isinstance(kind, int) else list(kind)


async def wait_for_agent(
    room: rtc.Room,
    *,
    agent_name: str | None = None,
) -> rtc.RemoteParticipant:
    """Wait for an agent participant to join the room.

    Args:
        room: The room to wait in.
        agent_name: Wait for the agent whose ``lk.agent.name`` attribute matches; None returns
            the first agent participant.

    Raises:
        RuntimeError: If the room is not connected.
    """
    identity = await _core.wait_for_agent(_core._core_create_room_session(room), agent_name)
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
    """The participant with ``identity``, or the first to join when it is None. Returns at once
    if it already joined; ``include_local`` also considers the local participant."""
    found = await _core.wait_for_participant(
        _core._core_create_room_session(room), identity, _kinds(kind), include_local
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
    """A track publication matching ``identity`` (any participant when None) and ``kind``.

    Returns at once if one is already published (and subscribed, with
    ``wait_for_subscription``). ``include_local`` also considers local publications, which
    resolve on publish and ignore ``wait_for_subscription``.

    Raises:
        RuntimeError: ``identity`` disconnected before a matching publication was found.
    """
    found = await _core.wait_for_track_publication(
        _core._core_create_room_session(room),
        identity,
        _kinds(kind),
        include_local,
        wait_for_subscription,
    )
    participant: rtc.Participant = (
        room.local_participant
        if found.publisher is None
        else room.remote_participants[found.publisher]
    )
    return participant.track_publications[found.sid]


__all__ = [
    "shortuuid",
    "EventEmitter",
    "log_exceptions",
    "aio",
    "media",
    "is_given",
    "ConnectionPool",
]
