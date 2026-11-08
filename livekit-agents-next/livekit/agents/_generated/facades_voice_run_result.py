# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core

if TYPE_CHECKING:
    from .usage import Usage


class RunStream:
    """The handle `async with agent.run_stream(...)` returns; applications do not build it."""

    def __init__(self) -> None:
        self._native = _core._core_RunStream()

    @property
    def usage(self) -> Usage:
        """Latest immutable usage; reading it never advances the run."""
        return self._native.usage

    def cancel(self) -> None:
        """Request cancellation; the native owner retains cleanup through settlement."""
        self._native.cancel()
