"""A tool's return as the call output the model reads, and the instructions message: shared by
the voice tool runner and ``agent.run()``."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .. import llm
from ..llm import ChatContext, utils as llm_utils
from ..llm.chat_context import Instructions
from ..llm.tool_context import (
    WAITING_PREFIX,
    Completed,
    Failed,
    InputRequired,
    Rejected,
    Transition,
    is_transition,
)
from ..log import logger

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class ToolExecutionOutput:
    fnc_call: llm.FunctionCall
    fnc_call_out: llm.FunctionCallOutput
    agent_task: Agent | None
    raw_output: Any
    raw_exception: BaseException | None
    transition: Transition | None = None
    """The transition the tool returned, applied by the activity once the call is recorded."""


def _transition_output(fnc_call: llm.FunctionCall, transition: Transition) -> ToolExecutionOutput:
    """The call output recorded for a transition.

    A model request cannot carry a call without an output, so the output stands in for the
    state: the value of ``Completed``, the error of ``Failed`` and ``Rejected``, the question
    an ``InputRequired`` waits on (replaced by the answer when it arrives) and the status of
    ``Working``. None of them asks the model for a reply; the activity acts on the transition.
    """
    is_error = False
    if isinstance(transition, Completed):
        value = transition.value
        output = value if isinstance(value, str) else _serialize_output(value)
    elif isinstance(transition, Failed):
        output, is_error = transition.error, True
    elif isinstance(transition, Rejected):
        output, is_error = transition.reason, True
    elif isinstance(transition, InputRequired):
        output = f"{WAITING_PREFIX}{transition.question}"
    else:
        output = transition.message or "working"

    return ToolExecutionOutput(
        fnc_call=fnc_call.model_copy(),
        fnc_call_out=llm.FunctionCallOutput(
            name=fnc_call.name,
            call_id=fnc_call.call_id,
            output=output,
            is_error=is_error,
            reply_required=False,
        ),
        agent_task=None,
        raw_output=transition,
        raw_exception=None,
        transition=transition,
    )


def _serialize_output(value: Any) -> str:
    if value is None:
        return "completed"
    try:
        return json.dumps(value, default=_json_default)
    except (TypeError, ValueError):
        return str(value)


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def make_tool_output(
    *, fnc_call: llm.FunctionCall, output: Any, exception: BaseException | None
) -> ToolExecutionOutput:
    from .agent import Agent

    if isinstance(output, BaseException):
        exception = output
        output = None

    if exception is None and is_transition(output):
        return _transition_output(fnc_call, output)

    if exception is not None:
        base_result = llm_utils.make_function_call_output(
            fnc_call=fnc_call, output=None, exception=exception
        )
        return ToolExecutionOutput(
            fnc_call=fnc_call.model_copy(),
            fnc_call_out=base_result.fnc_call_out,
            agent_task=None,
            raw_output=output,
            raw_exception=exception,
        )

    task: Agent | None = None
    fnc_out: Any = output
    if (
        isinstance(output, list)
        or isinstance(output, set)
        or isinstance(output, frozenset)
        or isinstance(output, tuple)
    ):
        agent_tasks = [item for item in output if isinstance(item, Agent)]
        other_outputs = [item for item in output if not isinstance(item, Agent)]
        if len(agent_tasks) > 1:
            logger.error(
                f"AI function `{fnc_call.name}` returned multiple AgentTask instances, ignoring the output",  # noqa: E501
                extra={"call_id": fnc_call.call_id, "lk.pii.output": output},
            )
            return ToolExecutionOutput(
                fnc_call=fnc_call.model_copy(),
                fnc_call_out=llm.FunctionCallOutput(
                    name=fnc_call.name,
                    call_id=fnc_call.call_id,
                    output="the tool returned more than one agent",
                    is_error=True,
                ),
                agent_task=None,
                raw_output=output,
                raw_exception=exception,
            )

        task = next(iter(agent_tasks), None)

        # fmt: off
        fnc_out = (
            other_outputs if task is None
            else None if not other_outputs
            else other_outputs[0] if len(other_outputs) == 1
            else other_outputs
        )
        # fmt: on

    elif isinstance(fnc_out, Agent):
        task = fnc_out
        fnc_out = None

    base_result = llm_utils.make_function_call_output(
        fnc_call=fnc_call, output=fnc_out, exception=None
    )
    if not isinstance(fnc_out, llm.ToolResult):
        # a tool with nothing to say, such as a bare handoff, expects no reply
        base_result.fnc_call_out.reply_required = fnc_out is not None

    return ToolExecutionOutput(
        fnc_call=fnc_call.model_copy(),
        fnc_call_out=base_result.fnc_call_out,
        agent_task=task,
        raw_output=output,
        raw_exception=exception,
    )


INSTRUCTIONS_MESSAGE_ID = "lk.agent_task.instructions"  #  value must not change
"""
The ID of the instructions message in the chat context. (only for stateless LLMs)
"""


def update_instructions(
    chat_ctx: ChatContext,
    *,
    instructions: str | Instructions,
    add_if_missing: bool,
    modality: Literal["audio", "text"] = "audio",
) -> None:
    """
    Update the instruction message in the chat context or insert a new one if missing.

    Instructions are resolved to a plain string using the given modality before storage.
    """
    text = (
        instructions.render(modality=modality)
        if isinstance(instructions, Instructions)
        else instructions
    )

    idx = chat_ctx.index_by_id(INSTRUCTIONS_MESSAGE_ID)
    if idx is not None:
        if chat_ctx.items[idx].type == "message":
            chat_ctx.items[idx] = llm.ChatMessage(
                id=INSTRUCTIONS_MESSAGE_ID,
                role="system",
                content=[text],
                created_at=chat_ctx.items[idx].created_at,
            )
        else:
            raise ValueError(
                "expected the instructions inside the chat_ctx to be of type 'message'"
            )
    elif add_if_missing:
        chat_ctx.items.insert(
            0,
            llm.ChatMessage(id=INSTRUCTIONS_MESSAGE_ID, role="system", content=[text]),
        )


def remove_instructions(chat_ctx: ChatContext) -> None:
    # loop in case there are items with the same id (shouldn't happen!)
    while True:
        if msg := chat_ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID):
            chat_ctx.items.remove(msg)
        else:
            break
