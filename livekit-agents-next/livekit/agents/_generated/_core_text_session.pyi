# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from .text_session import (
    TextSessionAgent,
    TextSessionDelivery,
    TextSessionRequest,
    TextSessionState,
    TextSessionUpdate,
)
from .usage import Usage

class TextSessionRunner:
    def __new__(
        cls,
        root_agent: str,
        agents: Sequence[TextSessionAgent],
        max_depth: int,
    ) -> Self: ...
    def run(self, request: TextSessionRequest) -> TextSessionUpdates: ...
    @property
    def usage(self) -> Usage:
        """The latest usage snapshot of the turn, across its local invocations; never advances it."""
    def release(self, delivery: TextSessionDelivery) -> None:
        """Acknowledge selected presentation before the child can release its parent."""
    def progress(self, invocation_id: str, operation: int, message: str) -> None: ...

class TextSessionUpdates:
    async def recv(self) -> TextSessionUpdate | None: ...
    async def aclose(self) -> None: ...

def _core_persist_text_session_state(record: TextSessionState) -> str: ...
def _core_restore_text_session_state(data: str) -> TextSessionState: ...
