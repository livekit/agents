# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .storage_session import SessionKind
    from .text_session import FrameId, SessionFrame, TextSessionState


class SessionState:
    """A detached snapshot, or a view updated only by an explicit store binding."""

    def __init__(self, data: str | None = None) -> None:
        self._native = _core._core_PersistedSessionState(data)

    @property
    def kind(self) -> SessionKind:
        return self._native.kind

    @property
    def active_agent(self) -> str | None:
        return self._native.active_agent

    @property
    def session_id(self) -> str | None:
        return self._native.session_id

    @property
    def execution_id(self) -> str | None:
        return self._native.execution_id

    @property
    def root_frame_id(self) -> FrameId:
        return self._native.root_frame_id

    @property
    def active_frame_id(self) -> FrameId:
        return self._native.active_frame_id

    @property
    def ancestors(self) -> list[FrameId]:
        return self._native.ancestors

    @property
    def frames(self) -> dict[FrameId, SessionFrame]:
        return self._native.frames

    def snapshot(self) -> SessionState:
        return SessionState(self._native.snapshot())

    def _to_json(self) -> str:
        return self._native._to_json()

    def _text_state(self) -> TextSessionState | None:
        return self._native._text_state()

    def _update(self, data: str) -> None:
        self._native._update(data)

    @staticmethod
    def _from_json(data: str) -> SessionState:
        return SessionState(_core._core_PersistedSessionState._from_json(data))

    @staticmethod
    def _from_text(state: TextSessionState) -> SessionState:
        return SessionState(_core._core_PersistedSessionState._from_text(state))

    @staticmethod
    def _from_voice(agent: str, history_ids: Sequence[str]) -> SessionState:
        return SessionState(_core._core_PersistedSessionState._from_voice(agent, history_ids))


SessionState.__module__ = "livekit.agents.storage.session_state"
