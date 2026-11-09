from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any, Generic

from . import _core, llm, utils
from ._generated import public_text_session as _public, text_session as records
from ._generated.public_text_session import *  # noqa: F403
from ._generated.run_result import Output_T
from .types import DEFAULT_API_CONNECT_OPTIONS
from .voice import Agent, AgentRetries, _BindingContextVar, _LocalAgentTool
from .voice.agent_run import _TextRun, door_hooks, run_resources, settled_result
from .voice.events import Context_T
from .voice.run_result import RunResult, RunStream, _Started


@dataclass(frozen=True, kw_only=True)
class TextSessionTurn(Generic[Output_T]):
    result: RunResult[Output_T]
    state: TextSessionState


class TextSession(Generic[Context_T, Output_T]):
    def __init__(
        self,
        *,
        agent: Agent[Context_T, Output_T],
        max_depth: int = 8,
    ) -> None:
        self._agent = agent
        self._max_depth = max_depth

    async def run(
        self,
        message: str | llm.ChatMessage,
        *,
        state: TextSessionState | None = None,
        context: Context_T | None = None,
        retries: int | AgentRetries | None = None,
    ) -> TextSessionTurn[Output_T]:
        """One turn of the conversation. ``retries`` overrides every agent's retry budgets
        for this turn (``AgentRetries``); an int sets both."""
        hooks = door_hooks()
        async with self.run_stream(
            message, state=state, context=context, retries=retries
        ) as stream:
            async for text in stream.stream_text(delta=True):
                if hooks is not None and hooks.on_token is not None:
                    await hooks.on_token(text)
            return await stream.result()

    def run_stream(
        self,
        message: str | llm.ChatMessage,
        *,
        state: TextSessionState | None = None,
        context: Context_T | None = None,
        retries: int | AgentRetries | None = None,
    ) -> RunStream[TextSessionTurn[Output_T]]:
        async def start(stream: _core._core_RunStream) -> _Started[TextSessionTurn[Output_T]]:

            hooks = door_hooks()
            if isinstance(message, str):
                incoming = llm.ChatMessage(role="user", content=[message])
            else:
                incoming = message
            # every frame reads the request's context; runs resolve their toolsets (Toolset.for_run) from the request's context
            frames = [
                _TextRun(
                    agent,
                    history=[],
                    context=context,
                    hooks=hooks,
                    conn_options=DEFAULT_API_CONNECT_OPTIONS,
                )
                for agent in _agents(self._agent)
            ]
            runs: dict[str, _TextRun[Any]] = {}
            resources = run_resources(self._agent, hooks, frames)
            try:
                await resources.open()
                bindings = []
                shared_thread = None
                shared_store = None
                for run in frames:
                    agent = run.agent
                    run.bind_resources(
                        resources,
                        thread=shared_thread,
                        store=shared_store,
                    )
                    if not runs:
                        shared_thread = run.thread
                        shared_store = run.store
                    run.tools = llm.ToolContext(run.agent_tools)
                    captured = contextvars.copy_context()
                    captured.run(_BindingContextVar.set, run)
                    runs[agent.name] = run
                    bindings.append(
                        records.TextSessionAgent(
                            definition=agent._definition(run.tools, retries),
                            llm=run.llm,
                            host=agent._host,
                            output_type_explicit=agent._output_type_explicit,
                            agent_tools={
                                name: tool.routing
                                for name, tool in run.tools.function_tools.items()
                                if isinstance(tool, _LocalAgentTool)
                            },
                            host_context=(asyncio.get_running_loop(), captured),
                            tools_host=run.tools.flatten(),
                        )
                    )
                runner = _core.TextSessionRunner(self._agent.name, bindings, self._max_depth)
                for run in runs.values():
                    run.runner = runner
                runner.run_stream(
                    records.TextSessionRequest(
                        session_id=state.session_id
                        if state is not None
                        else utils.shortuuid("session_"),
                        execution_id=utils.shortuuid("execution_"),
                        task_id=hooks.task_id if hooks is not None else None,
                        message=incoming,
                        state=state,
                        delegated=hooks.delegated if hooks is not None else False,
                        conn_options=DEFAULT_API_CONNECT_OPTIONS,
                    ),
                    stream,
                    resources,
                )

                def settle(finished: records.TextSessionFinished) -> TextSessionTurn[Output_T]:
                    frame = finished.state.frames[finished.state.active_frame_id]
                    result = settled_result(finished.run, runs[frame.agent_name].hooks)
                    if hooks is not None and hooks.on_result is not None:
                        hooks.on_result(result)
                    return TextSessionTurn(result=result, state=finished.state)

                return _Started(settle=settle)

            except BaseException:
                await resources.rollback()
                raise

        return RunStream(start)


def _agents(root: Agent[Any, Any]) -> list[Agent[Any, Any]]:
    """The agents reachable from ``root``; the core refuses two that share a name."""
    agents: list[Agent[Any, Any]] = []
    todo = [root]
    while todo:
        agent = todo.pop()
        if any(agent is seen for seen in agents):
            continue
        agents.append(agent)
        todo.extend(
            tool.agent
            for tool in llm.ToolContext(agent.tools).function_tools.values()
            if isinstance(tool, _LocalAgentTool)
        )
    return agents


__all__ = [
    "TextSession",
    "TextSessionTurn",
]
__all__ += _public.__all__
