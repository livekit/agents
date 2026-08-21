from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import llm
from livekit.agents.log import logger

_DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE = "<instructions>\n{content}\n</instructions>"


def parse_tool_call_arguments(fnc_call: llm.FunctionCall) -> dict[str, Any]:
    """Parse a stored function call's arguments into a dict for JSON-object providers.

    ``FunctionCall.arguments`` is only canonicalized to valid JSON when the model's
    output parses successfully; when it can't be parsed the raw string is kept
    verbatim (an open-weight model's output that ``json_repair`` couldn't recover, or
    history restored via ``ChatContext.from_dict`` / a custom ``llm_node``). The
    Anthropic, Google, and AWS formatters send the arguments as a JSON object, so
    formatting such history would otherwise raise ``json.JSONDecodeError`` on an
    unrelated later turn. Fall back to an empty object rather than fabricating
    arguments the model never sent, since the historical call already produced its
    output.
    """
    arguments = fnc_call.arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    logger.warning(
        "could not parse stored tool call arguments as a JSON object, using empty arguments",
        extra={"call_id": fnc_call.call_id, "tool_name": fnc_call.name},
    )
    return {}


def convert_mid_conversation_instructions(
    chat_ctx: llm.ChatContext,
    *,
    role: llm.ChatRole = "user",
    template: str = _DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE,
) -> llm.ChatContext:
    """Convert mid-conversation system messages to the given role to preserve their position.

    The first system/developer message is kept as the base preamble.
    Every later system/developer message is rewritten with the given
    role, wrapped in ``template``. This covers both mid-conversation
    instructions and per-turn instructions appended by ``generate_reply``
    on the very first turn (when there's no user/assistant content yet
    to anchor "mid-conversation"). Without this, providers like Gemini,
    Anthropic, and AWS fall back to ``inject_dummy_user_message``
    (a literal ``"."`` user turn the model frequently responds to with
    "you didn't say anything").
    """
    first_system_seen = False
    items: list[llm.ChatItem] = []

    for item in chat_ctx.items:
        if (
            item.type == "message"
            and item.role in ("system", "developer")
            and first_system_seen
            and (text := item.raw_text_content)
        ):
            items.append(
                llm.ChatMessage(
                    id=item.id,
                    role=role,
                    content=[template.format(content=text)],
                    created_at=item.created_at,
                )
            )
        else:
            if item.type == "message" and item.role in ("system", "developer"):
                first_system_seen = True
            items.append(item)

    return llm.ChatContext(items)


def group_tool_calls(chat_ctx: llm.ChatContext) -> list[_ChatItemGroup]:
    """Group chat items (messages, function calls, and function outputs)
    into coherent groups based on their item IDs and call IDs.

    Each group will contain:
    - Zero or one assistant message
    - Zero or more function/tool calls
    - The corresponding function/tool outputs matched by call_id

    User and system messages are placed in their own individual groups.

    Args:
        chat_ctx: The chat context containing all conversation items

    Returns:
        A list of _ChatItemGroup objects representing the grouped conversation
    """
    item_groups: dict[str, _ChatItemGroup] = OrderedDict()  # item_id to group of items
    tool_outputs: list[llm.FunctionCallOutput] = []
    for item in chat_ctx.items:
        if (item.type == "message" and item.role == "assistant") or item.type == "function_call":
            # only assistant messages and function calls can be grouped
            # For function calls, use group_id if available (for parallel function calls),
            # otherwise fall back to id-based grouping for backwards compatibility
            if item.type == "function_call" and item.group_id:
                group_id = item.group_id
            else:
                group_id = item.id.split("/")[0]
            if group_id not in item_groups:
                item_groups[group_id] = _ChatItemGroup().add(item)
            else:
                item_groups[group_id].add(item)
        elif item.type == "function_call_output":
            tool_outputs.append(item)
        else:
            item_groups[item.id] = _ChatItemGroup().add(item)

    # add tool outputs to their corresponding groups
    call_id_to_group: dict[str, _ChatItemGroup] = {
        tool_call.call_id: group for group in item_groups.values() for tool_call in group.tool_calls
    }
    for tool_output in tool_outputs:
        if tool_output.call_id not in call_id_to_group:
            logger.warning(
                "function output missing the corresponding function call, ignoring",
                extra={"call_id": tool_output.call_id, "tool_name": tool_output.name},
            )
            continue

        call_id_to_group[tool_output.call_id].add(tool_output)

    # validate that each group and remove invalid tool calls and tool outputs
    for group in item_groups.values():
        group.remove_invalid_tool_calls()

    return list(item_groups.values())


@dataclass
class _ChatItemGroup:
    message: llm.ChatMessage | None = None
    tool_calls: list[llm.FunctionCall] = field(default_factory=list)
    tool_outputs: list[llm.FunctionCallOutput] = field(default_factory=list)

    def add(self, item: llm.ChatItem) -> _ChatItemGroup:
        if item.type == "message":
            assert self.message is None, "only one message is allowed in a group"
            self.message = item
        elif item.type == "function_call":
            self.tool_calls.append(item)
        elif item.type == "function_call_output":
            self.tool_outputs.append(item)
        return self

    def remove_invalid_tool_calls(self) -> None:
        if len(self.tool_calls) == len(self.tool_outputs):
            return

        valid_call_ids = {call.call_id for call in self.tool_calls} & {
            output.call_id for output in self.tool_outputs
        }

        valid_tool_calls = []
        valid_tool_outputs = []

        for tool_call in self.tool_calls:
            if tool_call.call_id not in valid_call_ids:
                logger.warning(
                    "function call missing the corresponding function output, ignoring",
                    extra={"call_id": tool_call.call_id, "tool_name": tool_call.name},
                )
                continue
            valid_tool_calls.append(tool_call)

        for tool_output in self.tool_outputs:
            if tool_output.call_id not in valid_call_ids:
                logger.warning(
                    "function output missing the corresponding function call, ignoring",
                    extra={"call_id": tool_output.call_id, "tool_name": tool_output.name},
                )
                continue
            valid_tool_outputs.append(tool_output)

        self.tool_calls = valid_tool_calls
        self.tool_outputs = valid_tool_outputs

    def flatten(self) -> list[llm.ChatItem]:
        items: list[llm.ChatItem] = []
        if self.message:
            items.append(self.message)
        items.extend(self.tool_calls)
        items.extend(self.tool_outputs)
        return items
