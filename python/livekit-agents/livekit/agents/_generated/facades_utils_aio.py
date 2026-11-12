# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core


class ConnectionPool:
    """`utils.ConnectionPool`'s reads of its native bookkeeping."""

    if TYPE_CHECKING:
        _native: _core.ConnectionPoolState

    @property
    def last_acquire_time(self) -> float:
        """Seconds the last `get()` spent connecting; 0 when it reused a connection."""
        return self._native.last_acquire_time

    @property
    def last_connection_reused(self) -> bool:
        return self._native.last_connection_reused
