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
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from .log import logger

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass
class ParsedToolCall:
    name: str
    arguments: str


class ToolCallStreamParser:
    """Separates tool calls from spoken text in streamed chat model output.

    Models write tool calls inline with their reply, either wrapped in
    ``<tool_call>...</tool_call>`` (Qwen, Hermes, Mistral-style templates) or as a bare
    JSON object for the whole reply (Llama 3.x, whose ``<|python_tag|>`` marker is a
    special token stripped while decoding). Text inside a tool call block is never
    emitted as content, so it never reaches TTS; text around the blocks is.

    Only calls to the offered tools are accepted. A bare JSON reply also needs an
    ``arguments`` or ``parameters`` field, otherwise it is an ordinary answer.
    """

    def __init__(self, *, tool_names: Collection[str], hold_content: bool = False) -> None:
        self._tool_names = set(tool_names)
        # held content is only returned by flush(), e.g. when a tool call is required
        self._hold_content = hold_content
        self._buf = ""
        self._in_tag = False
        self._json_reply = False
        self._emitted_any = False
        self._calls: list[ParsedToolCall] = []

    def feed(self, text: str) -> str:
        """Add generated text and return the part that is safe to emit as content."""
        if not self._tool_names:
            return text

        self._buf += text
        if self._hold_content or self._json_reply:
            return ""

        out = ""
        while True:
            if self._in_tag:
                idx = self._buf.find(TOOL_CALL_CLOSE)
                if idx == -1:
                    return out
                self._add_tagged_call(self._buf[:idx])
                self._buf = self._buf[idx + len(TOOL_CALL_CLOSE) :]
                self._in_tag = False
                continue

            if not self._emitted_any and not self._calls and not out:
                stripped = self._buf.lstrip()
                if not stripped:
                    return out
                if stripped.startswith(("{", "[")):
                    self._json_reply = True
                    return out

            idx = self._buf.find(TOOL_CALL_OPEN)
            if idx != -1:
                out += self._emit(self._buf[:idx])
                self._buf = self._buf[idx + len(TOOL_CALL_OPEN) :]
                self._in_tag = True
                continue

            # hold back a suffix that could be the start of the opening tag
            hold = _partial_suffix_len(self._buf, TOOL_CALL_OPEN)
            emit_len = len(self._buf) - hold
            out += self._emit(self._buf[:emit_len])
            self._buf = self._buf[emit_len:]
            return out

    def flush(self) -> tuple[str, list[ParsedToolCall]]:
        """Finish parsing and return the remaining content and the tool calls found."""
        buf, self._buf = self._buf, ""
        if not self._tool_names:
            return buf, []

        if self._hold_content:
            parser = ToolCallStreamParser(tool_names=self._tool_names)
            content = parser.feed(buf)
            rest, calls = parser.flush()
            return content + rest, calls

        if self._json_reply:
            json_calls = _parse_json_tool_calls(
                buf.strip(), self._tool_names, require_arguments=True
            )
            if json_calls is None:
                # a reply that merely starts with a brace is still a reply
                return buf, []
            return "", json_calls

        if self._in_tag:
            # generation ended before the closing tag
            self._add_tagged_call(buf)
            buf = ""
        return buf, self._calls

    def _add_tagged_call(self, raw: str) -> None:
        calls = _parse_json_tool_calls(raw.strip(), self._tool_names, require_arguments=False)
        if calls is None:
            logger.warning("model emitted a tool call that could not be parsed or names no tool")
            return
        self._calls.extend(calls)

    def _emit(self, text: str) -> str:
        if text:
            self._emitted_any = True
        return text


def _partial_suffix_len(text: str, tag: str) -> int:
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


def _parse_json_tool_calls(
    raw: str, tool_names: Collection[str], *, require_arguments: bool
) -> list[ParsedToolCall] | None:
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
        if not isinstance(item, dict) or item.get("name") not in tool_names:
            return None
        if "arguments" in item:
            arguments = item["arguments"]
        elif "parameters" in item:
            arguments = item["parameters"]
        elif require_arguments:
            return None
        else:
            arguments = {}
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        tool_calls.append(ParsedToolCall(name=item["name"], arguments=arguments))
    return tool_calls or None
