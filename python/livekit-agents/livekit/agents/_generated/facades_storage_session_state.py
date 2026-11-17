# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .. import _bindings, _core
from ..llm.chat_context import _ReadOnlyChatContext

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..llm import ChatContext
    from ..storage.session_state import RemoteAgentState as HostRemoteAgentState
    from .session import AgentState
    from .session_snapshot import _SessionSnapshot
    from .storage_session import SessionKind


class SessionState:
    """A detached snapshot, or a view updated only by an explicit store binding."""

    def __init__(self, data: str | None = None) -> None:
        self._native = _core._core_PersistedSessionState(data)

    @property
    def kind(self) -> SessionKind:
        return self._native.kind

    @property
    def root_agent(self) -> AgentState | None:
        return self._native.root_agent

    @property
    def current_agent(self) -> AgentState | None:
        return self._native.current_agent

    @property
    def agents(self) -> dict[str, AgentState]:
        return self._native.agents

    @property
    def remote_agents(self) -> Mapping[str, HostRemoteAgentState]:
        return cast("Mapping[str, HostRemoteAgentState]", self._native.remote_agents)

    @property
    def chat_history(self) -> ChatContext:
        return _ReadOnlyChatContext(self._native.chat_history)

    @property
    def session_id(self) -> str | None:
        return self._native.session_id

    def snapshot(self) -> SessionState:
        return SessionState(self._native.snapshot())

    def _to_json(self) -> str:
        return self._native._to_json()

    def _snapshot_data(self) -> _SessionSnapshot | None:
        return self._native._snapshot_data()

    def _update(self, data: str) -> None:
        self._native._update(data)

    @staticmethod
    def _from_json(data: str) -> SessionState:
        return SessionState(_core._core_PersistedSessionState._from_json(data))


SessionState.__module__ = "livekit.agents.storage.session_state"
