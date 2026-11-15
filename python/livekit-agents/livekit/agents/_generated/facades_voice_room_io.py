# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .. import _bindings, _core

if TYPE_CHECKING:
    from livekit.rtc import Room


class RoomIO:
    """Shared room I/O. Ordinary session startup obtains the same service automatically."""

    if TYPE_CHECKING:
        _native: _core._core_RoomAttachments

    def bind(self, key: str) -> RoomBinding:
        """Declare a binding without subscribing to tracks or publishing outputs."""
        return RoomBinding(
            self._native.bind(
                key,
                room=cast(Any, self)._room_native,
                sdk_room=cast(Any, self)._room,
            )
        )

    async def aclose(self) -> None:
        """Close this room's bindings without disconnecting the SDK room."""
        await self._native.aclose()


class RoomBinding:
    """One consumer's room input/output ports and lifetime. Obtain it with `RoomIO.bind(key)`."""

    def __init__(self, native: _core._core_RoomBinding) -> None:
        self._native = native

    @property
    def key(self) -> str:
        return self._native.key

    @property
    def room(self) -> Room:
        return cast("Room", self._native.room)

    async def aclose(self) -> None:
        await self._native.aclose()


RoomBinding.__module__ = "livekit.agents.voice.room_io"
