# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast, overload

from .. import _bindings, _core, storage
from ..storage.session_state import SessionState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from .storage_session import BoundContext, Context_T


class ContextStore:
    """`storage.ContextStore`'s transactions, through its activation."""

    if TYPE_CHECKING:
        _activation: _core._core_ContextActivation

        def _session_history(self) -> Any: ...

    @overload
    def bind(self, context: Context_T) -> BoundContext[Context_T]:
        ...

    @overload
    def bind(self) -> BoundContext[None]:
        ...

    def bind(
        self,
        context: Context_T | None = None,
    ) -> BoundContext[Context_T] | BoundContext[None]:
        """Declare live updates without performing I/O. Execution activates the binding and
        tools receive the original context. Ordinary application fields still require saves.
        Omit context when the session has no application data.
        """
        return cast("BoundContext[Context_T] | BoundContext[None]", self._activation.declare_binding(
            context,
            self,
        ))

    async def load_session(self) -> SessionState:
        """Return a detached snapshot of committed conversation state."""
        return SessionState(await self._activation.load_session_snapshot())

    async def flush(self) -> None:
        """Flush accepted conversation state without committing staged application fields or SQL."""
        await self._activation.flush_bound_session(self._session_history())

    @contextlib.asynccontextmanager
    async def _transaction(self, *, independent: bool) -> AsyncIterator[storage.Transaction]:
        """A transaction, committed as the block ends and rolled back when it raises."""
        target = self._activation
        value = await target.transaction(independent)
        try:
            yield storage.Transaction(value)
        except BaseException:
            value.rollback()
            raise
        await value.commit()


class _SessionBinding:
    if TYPE_CHECKING:
        def _history(self) -> Any: ...

    def __init__(self, native: _core._core_SessionBinding) -> None:
        self._native = native

    def update(self, state: str) -> None:
        self._native.update(state)

    def update_voice(self, agent: str, history_ids: Sequence[str]) -> None:
        self._native.update_voice(agent, history_ids)

    def validate_voice(self, agent: str) -> None:
        self._native.validate_voice(agent)

    async def flush(self) -> None:
        await self._native.flush(self._history())

    @property
    def closed(self) -> bool:
        return self._native.closed

    def close(self) -> None:
        self._native.close()
