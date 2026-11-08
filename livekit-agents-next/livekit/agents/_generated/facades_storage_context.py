# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from .. import _bindings, _core, storage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class ContextStore:
    """`storage.ContextStore`'s transactions, through its activation."""

    if TYPE_CHECKING:
        _activation: _core._core_ContextActivation

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
