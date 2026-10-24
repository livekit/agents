# SPDX-License-Identifier: Apache-2.0
"""Acknowledged Prism history changes shared by startup, updates, and restart."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from dataclasses import replace
from typing import cast

from rimelabs_sdk import realtime as r

from livekit.agents import llm, utils
from livekit.agents.voice.generation import remove_instructions


class HistoryUpdateError(llm.RealtimeError):
    """A rejected history change, or a partially applied change that requires recovery."""

    def __init__(self, message: str, *, partial: bool = False):
        super().__init__(message, code="history_update_failed")
        self.partial = partial


def _items(ctx: llm.ChatContext) -> list[llm.ChatItem]:
    projected = ctx.copy(exclude_handoff=True, exclude_config_update=True)
    remove_instructions(projected)
    return projected.items


def _wire(item: llm.ChatItem, *, validate_text: bool = True) -> r.HistoryItem:
    if isinstance(item, llm.ChatMessage):
        if item.role not in ("user", "assistant", "system") or any(
            not isinstance(c, str) for c in item.content
        ):
            raise HistoryUpdateError("Prism history requires user, assistant, or system text")
        text = item.text_content or ""
        if validate_text and len(text) > 4000:
            raise HistoryUpdateError("Prism history text must contain at most 4000 characters")
        result: r.HistoryItem = r.HistoryMessage(role=item.role, text=text)
    elif isinstance(item, llm.FunctionCall):
        if not item.call_id or not item.name:
            raise HistoryUpdateError("Past tool calls require a call ID and name")
        result = r.HistoryFunctionCall(
            call_id=item.call_id, name=item.name, arguments=item.arguments
        )
    elif isinstance(item, llm.FunctionCallOutput):
        if not item.call_id:
            raise HistoryUpdateError("Tool results require a call ID")
        output = item.output
        if item.is_error:
            output = json.dumps(
                {
                    "status": "failed",
                    "reason": "tool_error",
                    "retryable": False,
                    "message": output,
                    "result": None,
                }
            )
        result = r.HistoryFunctionResult(call_id=item.call_id, output=output)
    else:
        raise HistoryUpdateError("Unsupported Prism history item")
    # Validate encoding before the first operation of a batch reaches the SDK.
    for value in vars(result).values():
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise HistoryUpdateError("Prism history must be valid UTF-8") from error
    return result


class PrismHistory:
    """Keep local identity stable while applying changes through the actual SDK.

    Reads are snapshots, not replacements for the local conversation record.
    Deletion-only changes preserve existing turns. Changes to order or content
    rebuild completed history, avoiding guesses about whole-turn insertion anchors.
    This is a sequence of acknowledged operations, not a server transaction.
    """

    def __init__(
        self,
        sdk: r.RealtimeSession,
        chat: llm.ChatContext,
        calls: MutableMapping[str, tuple[r.ToolCallRef, llm.FunctionCall]],
    ):
        self._sdk, self._chat, self._calls = sdk, chat, calls
        self._ids: dict[str, str] = {}
        self._active: dict[str, set[str]] = {}
        self._response_revision = 0

    @staticmethod
    def validate(
        ctx: llm.ChatContext, *, existing: set[str] | None = None, unchanged: set[str] | None = None
    ) -> None:
        items = _items(ctx)
        ids: set[str] = set()
        calls: dict[str, str] = {}
        results: set[str] = set()
        existing = existing or set()
        for item in items:
            if item.id in ids:
                raise HistoryUpdateError("Prism history item IDs must be unique")
            ids.add(item.id)
            _wire(item, validate_text=item.id not in (unchanged or set()))
            if isinstance(item, llm.FunctionCall):
                if item.call_id in calls:
                    raise HistoryUpdateError("Prism history call IDs must be unique")
                calls[item.call_id] = item.id
            if isinstance(item, llm.FunctionCallOutput):
                if item.call_id not in calls or item.call_id in results:
                    raise HistoryUpdateError("A tool result must follow its call exactly once")
                results.add(item.call_id)
        if any(
            item_id not in existing and call_id not in results for call_id, item_id in calls.items()
        ):
            raise HistoryUpdateError("Imported tool calls require completed results")

    def bind_typed(self, local_id: str, server_ids: tuple[str, ...]) -> None:
        """Bind only the input of this accepted typed request, never a staged successor."""
        if len(server_ids) == 1:
            self._ids[local_id] = server_ids[0]

    def observe(self, payload: r.InputEvent | r.ResponseEvent | r.FaultEvent) -> None:
        # This method never awaits: event consumption must not wait on a history lock.
        if isinstance(payload, r.ResponseStarted):
            self._response_revision += 1
            ids = set(payload.input_item_ids)
            if payload.input_item_id:
                ids.add(payload.input_item_id)
            self._active[payload.response.response_id] = ids
        elif isinstance(payload, r.InputCommitted):
            self._ids.setdefault(payload.item_id, payload.item_id)
        elif isinstance(payload, r.TranscriptFinal):
            self._ids.setdefault(payload.item_id, payload.item_id)
        elif isinstance(payload, r.MessageStarted):
            self._ids[payload.output.item_id] = payload.output.item_id
            self._active.setdefault(payload.output.response.response_id, set()).add(
                payload.output.item_id
            )
        elif isinstance(payload, r.ToolCall):
            self._ids[payload.item_id] = payload.item_id
            self._active.setdefault(payload.call.response_id, set()).add(payload.item_id)
        elif isinstance(payload, r.ResponseEnded):
            self._active.pop(payload.response.response_id, None)

    async def sync(
        self, target: llm.ChatContext
    ) -> list[tuple[r.ToolCallRef, llm.FunctionCallOutput]]:
        response_revision = self._response_revision
        old = list(self._chat.items)
        desired = _items(target)
        old_ids = {i.id for i in old}
        old_by_id = {i.id: i for i in old}
        new_by_id = {i.id: i for i in desired}
        unchanged = {
            key
            for key in old_by_id.keys() & new_by_id.keys()
            if _wire(old_by_id[key], validate_text=False)
            == _wire(new_by_id[key], validate_text=False)
        }
        completed = {i.call_id for i in old if isinstance(i, llm.FunctionCallOutput)}
        pending = {key for key in self._calls if key not in completed}
        pending_ids = {self._calls[key][1].id for key in pending}
        self.validate(llm.ChatContext(desired), existing=pending_ids, unchanged=unchanged)
        removed = [i for i in old if i.id not in new_by_id]
        added = [i for i in desired if i.id not in old_by_id]
        retained = [i.id for i in old if i.id in new_by_id]
        desired_retained = [i.id for i in desired if i.id in old_by_id]
        append_only = (
            not removed
            and len(unchanged) == len(old)
            and [i.id for i in desired[: len(old)]] == [i.id for i in old]
        )
        delete_only = not added and len(unchanged) == len(desired) and retained == desired_retained
        rebuild = not append_only and not delete_only
        # A live call cannot be imported, replaced, or hidden by a history edit.
        if rebuild and (self._active or pending):
            raise HistoryUpdateError(
                "Finish active replies and tool calls before replacing history"
            )
        if any(isinstance(i, llm.FunctionCall) and i.call_id in pending for i in removed):
            raise HistoryUpdateError("An active tool call must receive its result before removal")
        protected = set().union(*self._active.values()) if self._active else set()
        if any(self._ids.get(i.id, i.id) in protected for i in removed):
            raise HistoryUpdateError("History edits cannot remove an active reply's input")
        # Importing a result has server-defined placement directly after its call.
        # Reject requested orders that would require splitting that pair.
        to_create = desired if rebuild else added
        for item in to_create:
            _wire(item)
            if isinstance(item, llm.FunctionCallOutput) and item.call_id not in self._calls:
                index = desired.index(item)
                previous = desired[index - 1] if index else None
                if not isinstance(previous, llm.FunctionCall) or previous.call_id != item.call_id:
                    raise HistoryUpdateError(
                        "Imported tool results must immediately follow their calls"
                    )
            if isinstance(item, llm.FunctionCallOutput) and item.call_id in self._calls:
                if self._calls[item.call_id][1].extra.get("__livekit_agents_tool_non_blocking"):
                    raise HistoryUpdateError("Prism does not support tool progress updates")
        deletions = old if rebuild else removed
        if deletions:
            # Resolve server IDs for typed turns and live results not returned by
            # the older submit_tool_result interface. Never interpret absent active
            # items as deleted; unknown identity prevents mutation.
            snapshot = await self._sdk.get_history()
            by_call = {i.call_id: i.id for i in snapshot if isinstance(i, r.HistoryFunctionResult)}
            for item in deletions:
                if item.id not in self._ids and isinstance(item, llm.FunctionCallOutput):
                    server_id = by_call.get(item.call_id)
                    if server_id:
                        self._ids[item.id] = server_id
                if item.id not in self._ids:
                    raise HistoryUpdateError("Prism has not published the history item's server ID")
        submitted: list[tuple[r.ToolCallRef, llm.FunctionCallOutput]] = []
        committed = False
        try:
            deleted_calls: set[str] = set()
            for item in deletions:
                if isinstance(item, (llm.FunctionCall, llm.FunctionCallOutput)):
                    if item.call_id in deleted_calls:
                        continue
                    deleted_calls.add(item.call_id)
                if rebuild and self._response_revision != response_revision:
                    raise HistoryUpdateError("A new reply started during history replacement")
                await self._sdk.delete_item(self._ids[item.id])
                committed = True
                if rebuild and self._response_revision != response_revision:
                    raise HistoryUpdateError("A new reply started during history replacement")
            for item in to_create:
                if (
                    not rebuild
                    and isinstance(item, llm.FunctionCallOutput)
                    and item.call_id in self._calls
                ):
                    ref = self._calls[item.call_id][0]
                    wire = cast(r.HistoryFunctionResult, _wire(item))
                    await self._sdk.submit_tool_result(ref, wire.output)
                    submitted.append((ref, item))
                else:
                    if rebuild and self._response_revision != response_revision:
                        raise HistoryUpdateError("A new reply started during history replacement")
                    # New IDs avoid reusing identities retained in Prism's record.
                    wire_item = replace(_wire(item), id=utils.shortuuid("lk_"))
                    ref_item = await self._sdk.create_item(wire_item)
                    self._ids[item.id] = ref_item.item_id
                committed = True
                if rebuild and self._response_revision != response_revision:
                    raise HistoryUpdateError("A new reply started during history replacement")
        except Exception as error:
            if committed:
                raise HistoryUpdateError(
                    "Prism history update was partially applied; restart required", partial=True
                ) from error
            raise
        # Events can append to the local record while acknowledgments are pending.
        # Keep those events rather than overwriting them with the caller's snapshot.
        arrived = [i for i in self._chat.items if i.id not in old_ids and i.id not in new_by_id]
        current = {i.id: i for i in self._chat.items}
        self._chat.items[:] = [
            current.get(i.id, i) if i.id in unchanged else i.model_copy(deep=True) for i in desired
        ] + arrived
        for item in deletions:
            if isinstance(item, llm.FunctionCall):
                self._calls.pop(item.call_id, None)
        for item in removed:
            self._ids.pop(item.id, None)
        return submitted
