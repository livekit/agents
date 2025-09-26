from __future__ import annotations

from ...voice.agent import AgentTask


class DtmfInputsTask(AgentTask[list[str]]):
    async def on_enter(self) -> None:
        return await super().on_enter()
