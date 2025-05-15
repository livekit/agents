from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

from livekit.agents import llm
from livekit.agents.log import logger


def to_chat_ctx(
    chat_ctx: llm.ChatContext, generating_reply: bool, *, cache_key: Any
) -> tuple[list[dict], Literal[None]]:
    # OAI requires the tool calls to be followed by the corresponding tool outputs
    # we group them first and remove invalid tool calls and outputs before converting

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

    messages = []
    for group in item_groups.values():
        messages.extend(group.to_chat_items(cache_key))
    return messages, None


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

    def to_chat_items(self, cache_key: Any) -> list[dict[str, Any]]:
        tool_calls = {tool_call.call_id: tool_call for tool_call in self.tool_calls}
        tool_outputs = {tool_output.call_id: tool_output for tool_output in self.tool_outputs}

        valid_tools = set(tool_calls.keys()) & set(tool_outputs.keys())
        # remove invalid tool calls and tool outputs
        if len(tool_calls) != len(valid_tools) or len(tool_outputs) != len(valid_tools):
            for tool_call in self.tool_calls:
                if tool_call.call_id not in valid_tools:
                    logger.warning(
                        "function call missing the corresponding function output, ignoring",
                        extra={"call_id": tool_call.call_id, "tool_name": tool_call.name},
                    )
                    tool_calls.pop(tool_call.call_id)

            for tool_output in self.tool_outputs:
                if tool_output.call_id not in valid_tools:
                    logger.warning(
                        "function output missing the corresponding function call, ignoring",
                        extra={"call_id": tool_output.call_id, "tool_name": tool_output.name},
                    )
                    tool_outputs.pop(tool_output.call_id)

        if not self.message and not tool_calls and not tool_outputs:
            return []

        msg = (
            _to_chat_item(self.message, cache_key)
            if self.message
            else {"role": "assistant", "tool_calls": []}
        )
        if tool_calls:
            msg.setdefault("tool_calls", [])
        for tool_call in tool_calls.values():
            msg["tool_calls"].append(
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {"name": tool_call.name, "arguments": tool_call.arguments},
                }
            )
        items = [msg]
        for tool_output in tool_outputs.values():
            items.append(_to_chat_item(tool_output, cache_key))
        return items


def _to_chat_item(msg: llm.ChatItem, cache_key: Any) -> dict[str, Any]:
    if msg.type == "message":
        list_content: list[dict[str, Any]] = []
        text_content = ""
        for content in msg.content:
            if isinstance(content, str):
                if text_content:
                    text_content += "\n"
                text_content += content
            elif isinstance(content, llm.ImageContent):
                list_content.append(_to_image_content(content, cache_key))

        if not list_content:
            # certain providers require text-only content in a string vs a list.
            # for max-compatibility, we will combine all text content into a single string.
            return {
                "role": msg.role,  # type: ignore
                "content": text_content,
            }

        if text_content:
            list_content.append({"type": "text", "text": text_content})

        return {
            "role": msg.role,  # type: ignore
            "content": list_content,
        }

    elif msg.type == "function_call":
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": msg.call_id,
                    "type": "function",
                    "function": {
                        "name": msg.name,
                        "arguments": msg.arguments,
                    },
                }
            ],
        }

    elif msg.type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": msg.call_id,
            "content": msg.output,
        }


def _to_image_content(image: llm.ImageContent, cache_key: Any) -> dict[str, Any]:
    img = llm.utils.serialize_image(image)
    if img.external_url:
        return {
            "type": "image_url",
            "image_url": {
                "url": img.external_url,
                "detail": img.inference_detail,
            },
        }
    if cache_key not in image._cache:
        image._cache[cache_key] = img.data_bytes
    b64_data = base64.b64encode(image._cache[cache_key]).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{img.mime_type};base64,{b64_data}",
            "detail": img.inference_detail,
        },
    }
