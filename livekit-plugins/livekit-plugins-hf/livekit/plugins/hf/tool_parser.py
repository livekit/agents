# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from .log import logger

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


@dataclass
class ParsedToolCall:
    name: str
    arguments: str


class ToolCallStreamParser:
    """Separates tool calls from spoken text in streamed chat model output.

    Models write tool calls inline with their reply, either wrapped in
    ``<tool_call>...</tool_call>`` (Qwen, Hermes, Mistral-style templates) or as a bare
    JSON object for the whole reply (Llama 3.x, whose ``<|python_tag|>`` marker is a
    special token stripped while decoding). Text that may belong to a tool call is held
    back so it is never sent to TTS, and parsed once generation finishes.
    """

    def __init__(self, *, detect_tool_calls: bool) -> None:
        self._detect = detect_tool_calls
        self._mode: Literal["text", "json", "tagged"] = "text"
        self._pending = ""
        self._emitted_any = False

    def feed(self, text: str) -> str:
        """Add generated text and return the part that is safe to emit as content."""
        if not self._detect:
            return text

        self._pending += text
        if self._mode != "text":
            return ""

        if not self._emitted_any:
            stripped = self._pending.lstrip()
            if not stripped:
                return ""
            if stripped.startswith(("{", "[")):
                self._mode = "json"
                return ""

        idx = self._pending.find(_TOOL_CALL_OPEN)
        if idx != -1:
            self._mode = "tagged"
            out, self._pending = self._pending[:idx], self._pending[idx:]
            return self._emit(out)

        # hold back a suffix that could be the start of the opening tag
        hold = _partial_suffix_len(self._pending, _TOOL_CALL_OPEN)
        out = self._pending[: len(self._pending) - hold]
        self._pending = self._pending[len(out) :]
        return self._emit(out)

    def flush(self) -> tuple[str, list[ParsedToolCall]]:
        """Finish parsing and return the remaining content and the tool calls found."""
        pending, self._pending = self._pending, ""
        if self._mode == "text":
            return pending, []

        if self._mode == "json":
            tool_calls = _parse_json_tool_calls(pending.strip())
            if tool_calls is None:
                # a reply that merely starts with a brace is still a reply
                return pending, []
            return "", tool_calls

        tool_calls = []
        for raw in _TOOL_CALL_BLOCK_RE.findall(pending):
            tool_calls.extend(_parse_json_tool_calls(raw) or [])
        if not tool_calls:
            logger.warning("model emitted a tool call that could not be parsed")
        return "", tool_calls

    def _emit(self, text: str) -> str:
        if text:
            self._emitted_any = True
        return text


def _partial_suffix_len(text: str, tag: str) -> int:
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


def _parse_json_tool_calls(raw: str) -> list[ParsedToolCall] | None:
    """Parse one tool call object, a list of them, or ``;``-separated objects."""
    candidates: list[Any]
    try:
        data = json.loads(raw)
        candidates = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        try:
            candidates = [json.loads(part) for part in raw.split(";") if part.strip()]
        except json.JSONDecodeError:
            return None

    tool_calls: list[ParsedToolCall] = []
    for item in candidates:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        arguments = item.get("arguments", item.get("parameters", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        tool_calls.append(ParsedToolCall(name=item["name"], arguments=arguments))
    return tool_calls or None
