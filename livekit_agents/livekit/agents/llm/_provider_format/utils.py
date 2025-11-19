from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from livekit.agents import llm
from livekit.agents.log import logger


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
