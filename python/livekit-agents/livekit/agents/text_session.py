from __future__ import annotations

import asyncio
import contextvars
import json
from dataclasses import dataclass, replace
from typing import Any, Generic

from . import _core, llm, utils
from ._generated import public_text_session as _public, text_session as records
from ._generated.public_text_session import *  # noqa: F403
from ._generated.run_result import Output_T
from .storage.context import _session_field
from .storage.session_state import BoundContext, SessionState
from .types import DEFAULT_API_CONNECT_OPTIONS
from .voice import Agent, AgentRetries, _BindingContextVar, _LocalAgentTool
from .voice.agent_run import RunHooks, _TextRun, door_hooks, run_resources, settled_result
from .voice.events import Context_T
from .voice.run_result import RunResult, RunStream, _Started


@dataclass(frozen=True, kw_only=True)
class TextSessionTurn(Generic[Output_T]):
    result: RunResult[Output_T]
    state: SessionState


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
        state: TextSessionState | SessionState | None = None,
        context: Context_T | BoundContext[Context_T] | None = None,
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
        state: TextSessionState | SessionState | None = None,
        context: Context_T | BoundContext[Context_T] | None = None,
        retries: int | AgentRetries | None = None,
    ) -> RunStream[TextSessionTurn[Output_T]]:
        async def start(stream: _core._core_RunStream) -> _Started[TextSessionTurn[Output_T]]:

            hooks = door_hooks()
            binding = None
            app = context.context if isinstance(context, BoundContext) else context
            selected = state
            if isinstance(context, BoundContext):
                if state is not None:
                    raise ValueError("explicit state cannot be combined with a bound context")
                if (
                    hooks is not None
                    and hooks.store is not None
                    and hooks.store is not context.store
                ):
                    raise ValueError("the context belongs to a different execution store")
                binding = context.store._bind_session(app)
                selected = binding.state
                hooks = (
                    replace(hooks, store=context.store)
                    if hooks is not None
                    else RunHooks(store=context.store)
                )
            elif app is not None:
                field = _session_field(app)
                if field is not None:
                    if state is not None:
                        raise ValueError("explicit state conflicts with the context session field")
                    selected = getattr(app, field)
            resources: _core._core_RunResources | None = None
            try:
                restored = (
                    selected._text_state() if isinstance(selected, SessionState) else selected
                )
                if isinstance(message, str):
                    incoming = llm.ChatMessage(role="user", content=[message])
                else:
                    incoming = message
                # every frame reads the request's context; runs resolve their toolsets (Toolset.for_run) from the request's context
                frames = [
                    _TextRun(
                        agent,
                        history=[],
                        context=app,
                        hooks=hooks,
                        conn_options=DEFAULT_API_CONNECT_OPTIONS,
                    )
                    for agent in _agents(self._agent)
                ]
                runs: dict[str, _TextRun[Any]] = {}
                resources = run_resources(self._agent, hooks, frames)
                if binding is not None:
                    resources.retain_binding(binding)
                await resources.open()
                if binding is not None:
                    current = binding.store._activation.session_state
                    binding.state._update(
                        (
                            SessionState()
                            if current is None
                            else SessionState._from_json(json.dumps(current))
                        )._to_json()
                    )
                    restored = binding.state._text_state()
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
                        session_sink=binding._native if binding is not None else None,
                        session_id=restored.session_id
                        if restored is not None
                        else utils.shortuuid("session_"),
                        execution_id=utils.shortuuid("execution_"),
                        task_id=hooks.task_id if hooks is not None else None,
                        message=incoming,
                        state=restored,
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
                    snapshot = SessionState._from_text(finished.state)
                    return TextSessionTurn(result=result, state=snapshot)

                return _Started(settle=settle)

            except BaseException:
                if binding is not None:
                    binding.close()
                if resources is not None:
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
