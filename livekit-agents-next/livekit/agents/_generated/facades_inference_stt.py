# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core


class STT:
    """`inference.STT`'s client, built by its constructor."""

    if TYPE_CHECKING:
        _native: _core.InferenceStt

        def _core_native(self) -> _core.InferenceStt: ...

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, STT, "stream")
        return self if replaced else self._core_native()
