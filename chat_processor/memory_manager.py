# filler_agent/memory_manager.py
from __future__ import annotations

from typing import Optional, Any


class AgentStateTracker:
    """
    Small helper to remember whether the agent and user are speaking or not.

    We store the latest state names as strings and provide convenience checks
    like is_agent_speaking().
    """

    def __init__(self) -> None:
        self._agent_state: Optional[str] = None
        self._user_state: Optional[str] = None

    @staticmethod
    def _to_name(state: Any) -> str:
        # AgentState / UserState are enums; their string form usually
        # looks like "AgentState.speaking". We normalize to lower-case text.
        return str(state).lower()

    def update_agent_state(self, new_state: Any) -> None:
        self._agent_state = self._to_name(new_state)

    def update_user_state(self, new_state: Any) -> None:
        self._user_state = self._to_name(new_state)

    def is_agent_speaking(self) -> bool:
        return self._agent_state is not None and "speaking" in self._agent_state

    def is_user_speaking(self) -> bool:
        return self._user_state is not None and "speaking" in self._user_state
