"""Request-scoped text conversations, executed by the Rust TextSession coordinator."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Generic

from . import _core, llm, utils
from ._generated import text_session as records
from ._generated.agent_run import RunSettled
from ._generated.text_session import TextSessionState
from .types import DEFAULT_API_CONNECT_OPTIONS
from .voice.agent import Agent, _BindingContextVar
from .voice.agent_run import _TextRun, door_hooks, run_error, settled_result
from .voice.agent_tool import _LocalAgentTool
from .voice.events import Context_T
from .voice.run_result import Output_T, RunResult, RunUsage
from .voice.run_stream import RunStream, _Started


@dataclass(frozen=True, kw_only=True)
class TextSessionTurn(Generic[Output_T]):
    result: RunResult[Output_T]
    state: TextSessionState


class TextSessionRoutingError(ValueError):
    """The supplied state or an agent-tool route cannot be used."""

    def __init__(self, error: records.RoutingError) -> None:
        self.error = error
        super().__init__(str(error))


class TextSession(Generic[Context_T, Output_T]):
    def __init__(self, *, agent: Agent[Context_T, Output_T], max_depth: int = 8) -> None:
        self._agent = agent
        self._max_depth = max_depth

    async def run(
        self,
        message: str | llm.ChatMessage,
        *,
        state: TextSessionState | None = None,
        context: Context_T | None = None,
    ) -> TextSessionTurn[Output_T]:
        hooks = door_hooks()
        async with self.run_stream(message, state=state, context=context) as stream:
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
    ) -> RunStream[TextSessionTurn[Output_T]]:
        async def start() -> _Started[TextSessionTurn[Output_T]]:
            from ._core_view import to_core_item

            hooks = door_hooks()
            if isinstance(message, str):
                incoming = llm.ChatMessage(role="user", content=[message])
            else:
                incoming = message
            root_scope = (
                hooks.store._scope_id if hooks is not None and hooks.store is not None else "root"
            )
            agents = _agents(self._agent)
            usage = RunUsage()
            runs: dict[str, _TextRun[Any]] = {}
            cleanup = AsyncExitStack()
            try:
                bindings = []
                for agent in agents:
                    run = _TextRun(
                        agent,
                        history=[],
                        context=context,
                        usage=usage,
                        hooks=hooks,
                        conn_options=DEFAULT_API_CONNECT_OPTIONS,
                    )
                    await run.acquire_toolsets()
                    cleanup.push_async_callback(run.release_toolsets)
                    run.tools = llm.ToolContext(agent.tools)
                    captured = contextvars.copy_context()
                    captured.run(_BindingContextVar.set, run)
                    runs[agent.name] = run
                    bindings.append(
                        records.TextSessionAgent(
                            definition=agent._definition(run.tools),
                            llm=run.llm,
                            host=agent._host,
                            output_type_explicit=agent._output_type_explicit,
                            agent_tools={
                                name: tool.routing
                                for name, tool in run.tools.function_tools.items()
                                if isinstance(tool, _LocalAgentTool)
                            },
                            host_context=(asyncio.get_running_loop(), captured),
                        )
                    )
                runner = _core.TextSessionRunner(self._agent.name, bindings, self._max_depth)
                for run in runs.values():
                    run.runner = runner
                updates = runner.run(
                    records.TextSessionRequest(
                        session_id=state.session_id
                        if state is not None
                        else utils.shortuuid("session_"),
                        execution_id=state.execution_id
                        if state is not None
                        else utils.shortuuid("execution_"),
                        root_scope=root_scope,
                        contexts={root_scope: context},
                        message=to_core_item(incoming),
                        state=state,
                        delegated=hooks.delegated if hooks is not None else False,
                        conn_options=DEFAULT_API_CONNECT_OPTIONS,
                    )
                )

                def settle(finished: records.TextSessionSettled) -> TextSessionTurn[Output_T]:
                    frame = finished.turn.state.frames[finished.turn.state.active_frame_id]
                    result = settled_result(
                        runs[frame.agent_name],
                        RunSettled(result=finished.turn.result, output=finished.output),
                        accumulate=False,
                    )
                    if hooks is not None and hooks.on_result is not None:
                        hooks.on_result(result)
                    return TextSessionTurn(result=result, state=finished.turn.state)

                def fail(error: records.TextSessionError) -> BaseException:
                    if isinstance(error, records.TextSessionErrorRun):
                        return run_error(error.error)
                    return TextSessionRoutingError(error.error)

                def parked(value: Any) -> None:
                    usage._add(
                        RunUsage(
                            requests=value.requests,
                            input_tokens=value.input_tokens,
                            output_tokens=value.output_tokens,
                            tool_calls=value.tool_calls,
                        )
                    )

                return _Started(
                    id="",
                    updates=updates,
                    settle=settle,
                    fail=fail,
                    cleanup=cleanup.aclose,
                    release=runner.release,
                    on_usage=parked,
                )

            except BaseException:
                await cleanup.aclose()
                raise

        return RunStream(start, agent_name=self._agent.name)


def _agents(root: Agent[Any, Any]) -> list[Agent[Any, Any]]:
    agents: dict[str, Agent[Any, Any]] = {}
    todo = [root]
    while todo:
        agent = todo.pop()
        previous = agents.get(agent.name)
        if previous is not None:
            if previous is not agent:
                raise ValueError(f"different agent definitions share the name {agent.name!r}")
            continue
        agents[agent.name] = agent
        todo.extend(
            tool.agent
            for tool in llm.ToolContext(agent.tools).function_tools.values()
            if isinstance(tool, _LocalAgentTool)
        )
    return list(agents.values())


__all__ = ["TextSession", "TextSessionRoutingError", "TextSessionState", "TextSessionTurn"]
