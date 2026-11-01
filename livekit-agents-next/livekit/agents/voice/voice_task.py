"""A typed temporary interaction on the current live voice session."""

from __future__ import annotations

from typing import Generic, cast

from .._generated.voice_session import UpdateAgentArgs
from .._generated.voice_task_errors import VoiceTaskFailed as VoiceTaskFailed
from ._session_callbacks import _live_binding
from .agent import Agent, Output_T
from .events import Context_T


class VoiceTask(Generic[Output_T, Context_T]):
    """One bounded live interaction run by ``agent`` on the active ``VoiceSession``.

    Like ``TextSession``, a ``VoiceTask`` is not an ``Agent``: it runs one. Await ``run()``
    inside a tool of the agent holding the call: ``agent`` takes the floor (its ``on_enter``,
    or a first reply without one) and ``run()`` returns once one of its tools returns
    ``Completed(value)``, validated against its ``output_type``, or raises ``VoiceTaskFailed``
    once one returns ``Failed(reason)``. The waiting agent then gets the floor back without its
    ``on_exit`` or ``on_enter``. Cancelling the await, or closing the session, ends the task.
    """

    def __init__(self, *, agent: Agent[Context_T, Output_T]) -> None:
        self._agent = agent

    @property
    def agent(self) -> Agent[Context_T, Output_T]:
        return self._agent

    async def run(self) -> Output_T:
        binding = _live_binding("VoiceTask.run()")
        session, agent = binding.session, self._agent
        reply = session._running.run_voice_task(
            UpdateAgentArgs(**session._frame(agent, task=True)),
            origin=binding.origin,
            opens_with_reply=not agent._has_on_enter(),
        )
        output = cast(Output_T, await reply)
        return output


VoiceTask.__module__ = VoiceTaskFailed.__module__ = "livekit.agents.voice.agent_session"
