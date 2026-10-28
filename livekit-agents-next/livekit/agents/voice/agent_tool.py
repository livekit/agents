"""The host-authored parameter contract of a local agent tool. Rust owns its execution."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

from .._generated.text_session import LocalAgentTool
from ..llm.tool_context import FunctionTool, FunctionToolInfo, ReadBack, ToolFlag

if TYPE_CHECKING:
    from .agent import Agent


class _LocalAgentTool(FunctionTool[..., Awaitable[object]]):
    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        tool_name: str | None,
        tool_description: str | None,
        parameters: type[object] | None,
        needs_approval: bool | ReadBack,
        chat_history: Literal["fresh", "fork"],
        merge_chat_history: bool,
    ) -> None:
        if not agent._output_type_explicit:
            raise ValueError("an agent tool requires an explicitly declared output_type")
        if chat_history not in ("fresh", "fork"):
            raise ValueError('chat_history must be "fresh" or "fork"')
        self.agent = agent
        self.parameters = TypeAdapter(parameters) if parameters is not None else None
        self.schema = (
            self.parameters.json_schema()
            if self.parameters is not None
            else {"type": "object", "properties": {}, "additionalProperties": False}
        )
        if self.schema.get("type") != "object":
            raise ValueError("agent-tool parameters must describe an object")
        self.routing = LocalAgentTool(
            agent_name=agent.name,
            chat_history=chat_history,
            merge_chat_history=merge_chat_history,
            context_scope=None,
        )

        async def invoke(**arguments: object) -> object:
            raise RuntimeError("a local agent tool must run inside a TextSession")

        super().__init__(
            invoke,
            FunctionToolInfo(
                name=tool_name or agent.name,
                flags=ToolFlag.NONE,
                description=tool_description or f"Run {agent.name}.",
                needs_approval=bool(needs_approval),
                read_back=needs_approval if callable(needs_approval) else None,
            ),
        )

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.parameters is None:
            if arguments:
                raise ValueError("this agent tool accepts no parameters")
            return {}
        value = self.parameters.validate_python(arguments)
        encoded = to_jsonable_python(value)
        if not isinstance(encoded, dict):
            raise ValueError("validated agent-tool parameters must be an object")
        return encoded
