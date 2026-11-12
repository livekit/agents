# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from livekit.agents.a2a import _Delegation

from .._core import _core_RunResources, _core_RunStream
from .chat_items import ChatItem
from .connect import APIConnectOptions
from .text_session import (
    LocalAgentTool,
    TextSessionAgent,
    TextSessionDelivery,
    TextSessionRequest,
    TextSessionState,
    TextSessionUpdate,
)
from .tool_report import ToolOutcome
from .usage import Usage

class TextSessionRunner:
    def __new__(
        cls,
        root_agent: str,
        agents: Sequence[TextSessionAgent],
        max_depth: int,
    ) -> Self: ...
    def run(self, request: TextSessionRequest) -> TextSessionUpdates: ...
    def run_stream(
        self,
        request: TextSessionRequest,
        stream: _core_RunStream,
        resources: _core_RunResources,
    ) -> None: ...
    @property
    def usage(self) -> Usage:
        """The latest usage snapshot of the turn, across its local invocations; never advances it."""
    def release(self, delivery: TextSessionDelivery) -> None:
        """Acknowledge selected presentation before the child can release its parent."""
    def progress(self, invocation_id: str, operation: int, message: str) -> None: ...

class TextSessionUpdates:
    async def recv(self) -> TextSessionUpdate | None: ...
    async def aclose(self) -> None: ...

class _core_LocalInteraction:
    """One call of an agent tool outside a TextSession: the child agent runs in process as a delegate
    of the call (tools.md "Agents as tools"). Its decisions are `lk_agents_core::agent_tool`'s.
    """
    def __new__(
        cls,
        child: TextSessionAgent,
        tool: LocalAgentTool,
        function_call: ChatItem,
        depth: int,
    ) -> Self:
        """`child` runs for `function_call`, the parent's call of `tool`; `depth` counts the agent
        tools it runs under, itself included.
        """
    async def run(
        self,
        host: _Delegation,
        arguments: object,
        history: Sequence[ChatItem],
        live: bool,
        conn_options: APIConnectOptions,
    ) -> ToolOutcome:
        """Run the child on `arguments` from the parent's `history`, under the caller's span: its
        tools' progress reaches `host.progress`; on a call (`live`) the request is announced
        through `host.delegated` and each question it asks is answered through `host.ask`, the
        answer resuming it. Returns the call's outcome; off a call, the child's question.
        """
    def cancelled(self) -> None:
        """The tool call was cancelled: the running child is cancelled; `settled()` waits for it."""
    async def settled(self) -> None:
        """After `cancelled()`: the child's settlement, bounded by the teardown limit."""
    @property
    def extra(self) -> dict[str, object]:
        """The call's retained state: the child's own history while it waits on the person."""
    @property
    def usage(self) -> Usage:
        """What the child consumed across its invocations, the running one included."""
    def progress(self, invocation_id: str, operation: int, message: str) -> None:
        """`ctx.update(message)` from a tool body of the child's invocation `invocation_id`."""

def _core_persist_text_session_state(record: TextSessionState) -> str: ...
def _core_restore_text_session_state(data: str) -> TextSessionState: ...
